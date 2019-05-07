import gym
import tensorflow as tf
import numpy as np

env = gym.make("CartPole-v0")


# Environment settings
state_size = 4
action_size = env.action_space.n

# Hyperparameter settings
learning_rate = 0.01
gamma = 0.95
max_episodes = 3000

# Miscellaneous settings
training = True
restart = True
render = False

def process_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    accumulator = 0.0
    for i in reversed(range(len(rewards))):
        accumulator = accumulator * gamma + rewards[i]
        discounted_rewards[i] = accumulator
        
    mean = np.mean(discounted_rewards)
    std = np.std(discounted_rewards)
    discounted_rewards = (discounted_rewards - mean) / std
    return discounted_rewards
    
class PolicyNet():
    def __init__(self, state_size, action_size, learning_rate, name='PolicyNet'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name
        
        with tf.variable_scope(name):
            self.inputs = tf.placeholder(tf.float32, [None, state_size], name="inputs")
            self.actions = tf.placeholder(tf.float32, [None, action_size], name="actions")
            self.discounted_rewards = tf.placeholder(tf.float32, [None,], name="reward")
            
            self.fc1 = tf.contrib.layers.fully_connected(inputs = self.inputs,
                                                        num_outputs = 10,  
                                                        activation_fn = tf.nn.elu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())
                                                        
            self.fc2 = tf.contrib.layers.fully_connected(inputs = self.fc1,
                                                        num_outputs = self.action_size,
                                                        activation_fn = tf.nn.elu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())
                
            self.fc3 = tf.contrib.layers.fully_connected(inputs = self.fc1,
                                                        num_outputs = self.action_size,
                                                        activation_fn = tf.nn.elu,
                                                        weights_initializer=tf.contrib.layers.xavier_initializer())
                                                        
                                                        
            self.softmax = tf.nn.softmax(self.fc3)
            
            # LOSS FUNCTION - ln (pi (A | S)), pi = softmax(neural net)
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fc3, labels=self.actions)
            
            # Mean of Expected value of discounted reward x neg log prob at each time step
            # i.e. - E[G * ln (pi(A|S))]
            self.loss = tf.reduce_mean(self.neg_log_prob * self.discounted_rewards)
            
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            
            
            
def main():
    print("Running")
    tf.reset_default_graph()
    network = PolicyNet(state_size, action_size, learning_rate)
    saver = tf.train.Saver()


    if training:
        reward_list = []
        all_rewards = []
        with tf.Session() as sess:
            if restart:
                sess.run(tf.global_variables_initializer())
            else:
                saver.restore(sess, "./models/model.ckpt")
            
            print("Network initialized")
            for episode in range(max_episodes):
                episode_states = []
                episode_rewards = []
                episode_actions = []
            
                state = env.reset()
                
                if render:
                    env.render()
                
                # run steps of episode
                while True:
                    action_prob_distribution = sess.run(network.softmax, feed_dict={network.inputs: state.reshape(1, state_size)})
                    action = np.random.choice(range(action_prob_distribution.shape[1]), p=action_prob_distribution.ravel())
                    
                    new_state, reward, done, _ = env.step(action)
                    episode_states.append(new_state)
                    
                    action_one_hot = np.zeros(action_size)
                    action_one_hot[action] = 1
                    episode_actions.append(action_one_hot)
                    
                    episode_rewards.append(reward)
                    
                    if done:
                        reward_sum = np.sum(episode_rewards)
                        all_rewards.append(reward_sum)
                        # Estimate discounted reward at each time step
                        discounted_rewards = process_rewards(episode_rewards)
                        loss, _ = sess.run([network.loss, network.optimize], feed_dict={network.inputs: episode_states,
                                                                                        network.actions: episode_actions,
                                                                                        network.discounted_rewards: discounted_rewards})
                        print("Episode: ", episode)
                        print("Reward: ", reward_sum)
                        print("Mean Reward: ", np.mean(all_rewards))
                        
                        break
                    
                    state = new_state
                
                if episode % 100 == 0:
                    saver.save(sess, "./models/model.ckpt")
                    print("Model Saved.")

main()