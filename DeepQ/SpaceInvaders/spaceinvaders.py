import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import retro                 # Retro Environment
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
import matplotlib.pyplot as plt # Display graphs
from collections import deque# Ordered collection with ends
import random
import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

env = retro.make(game='SpaceInvaders-Atari2600')
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
print(possible_actions)
    
# Model Hyperparameters
state_size = [110,84,4] #4 frames of 110x84
action_size = env.action_space.n
learning_rate = 0.00025

# Training Hyperparameters
total_episodes = 50
max_steps = 50000
batch_size = 64

# Epsilon values
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00001

# Discount
gamma = 0.9

# Memory Hyperparameters
pretrain_length = batch_size #initial memory size
memory_size = 1000000

# Preprocessing Hyperparameters
stack_size = 4

# Misc. Hyperparameters
training = True
episode_render = False

# Deep-Q Network
class DQNetwork():
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32,[None,*self.state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None,self.action_size], name="actions_")
            
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
            filters=32,
            kernel_size = [8,8],
            strides=[4,4],
            padding="VALID",
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name="conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
            filters=64,
            kernel_size=[4,4],
            strides=[2,2],
            padding="VALID",
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name="conv2")
            self.conv2_out = tf.nn.elu(self.conv2,name="conv2_out")
            
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
            filters=64,
            kernel_size=[3,3],
            strides=[2,2],
            padding="VALID",
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            name="conv3")
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            
            self.fc = tf.layers.dense(inputs = self.flatten,
            units=512,
            activation=tf.nn.elu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name="fc1")
            
            self.output = tf.layers.dense(inputs=self.fc,
            units=self.action_size,
            activation=None)
            
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_)) # list of q-values per possible action
            
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        indices = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        
        return [self.buffer[i] for i in indices]



# Preproccessing functions

#Grayscale and downscale
def preprocess_frame(frame):
    gray = rgb2gray(frame)
    
    cropped_frame = gray[8:-12,4:-12]
    
    normalized_frame = cropped_frame/255.0
    
    preprocessed_frame = transform.resize(cropped_frame, [110,84])
    return preprocessed_frame


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([], maxlen=stack_size)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        
        stacked_state = np.stack(stacked_frames,axis=2)
    else:
        #automatically removed oldest frame
        stacked_frames.append(frame)
        
        stacked_state = np.stack(stacked_frames,axis=2)
    
    return stacked_state,stacked_frames

def initialize_memory():
    stacked_frames = deque([np.zeros((110,84),dtype=np.int) for i in range(stack_size)], maxlen=stack_size)
    memory = Memory(max_size=memory_size)
    for i in range(pretrain_length):
        if i == 0:
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames,state,True)
        
        randChoice = random.randint(0, len(possible_actions)-1)
        action = possible_actions[randChoice]
        next_state,reward,done,_ = env.step(action)
        
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
        
        if done:
            next_state = np.zeros(state.shape)
            memory.add((state,action,reward,next_state,done))
            state = env.reset()
            state,stacked_frames = stack_frames(stacked_frames, state, True)
        else:
            memory.add((state,action,reward,next_state,done))
            state=next_state
    return memory, stacked_frames
        
def get_action(epsilon_initial, epsilon_final, decay_rate, decay_step, state, network, sess):
    # decay epsilon
    new_eps = epsilon_final + (epsilon_final - epsilon_initial) * np.exp(-decay_rate * decay_step)
    rand_value = random.uniform(0, 1)
    
    if rand_value < new_eps:
        action_choice = possible_actions[random.randint(0, len(possible_actions) - 1)]
    else:
        q_values = sess.run(network.output, feed_dict={network.inputs_: state.reshape((1, *state.shape))})
        action_index = np.argmax(q_values)
        action_choice = possible_actions[action_index]
    
    return action_choice, new_eps
    
    
def main():
    print("Running")
    tf.reset_default_graph()
    network = DQNetwork(state_size, action_size, learning_rate)
    
    writer = tf.summary.FileWriter("./tensorboard/dqn/1")
    tf.summary.scalar("Loss", network.loss)
    write_op = tf.summary.merge_all()
    
    
    memory, stacked_frames = initialize_memory()
    
    saver = tf.train.Saver()
    if training:
        rewards_list = []
        with tf.Session() as sess:
            #initialize weights
            sess.run(tf.global_variables_initializer())

            #initialize decay
            decay_step = 0
            
            for episode in range(total_episodes):
                #initialize environment
                step = 0
                state = env.reset()
                
                episode_rewards = []
                
                state,stacked_frames = stack_frames(stacked_frames, state, True)
                
                while (step < max_steps):
                    # play game
                    step += 1
                    decay_step += 1
                    action_choice, eps = get_action(explore_start, explore_stop, decay_rate, decay_step, state, network, sess)
                    new_state, reward, done, _ = env.step(action_choice)
                    
                    if episode_render:
                        env.render()
                        
                    episode_rewards.append(reward)
                    
                    if done:
                        new_state = np.zeros((110,84), dtype=int)
                        new_state, stacked_frames = stack_frames(stacked_frames, new_state, False)
                        
                        step = max_steps
                        
                        total_reward = np.sum(episode_rewards)
                        
                        print('Episode: {}'.format(episode),
                                  'Total reward: {}'.format(total_reward),
                                  'Explore P: {:.4f}'.format(explore_probability),
                                'Training Loss {:.4f}'.format(loss))

                        rewards_list.append((episode, total_reward))
                        
                        new_transition = (state, action_choice, reward, new_state, done)
                        memory.add(new_transition)
                        
                        
                    else:
                        new_state, stacked_frames = stack_frames(stacked_frames, new_state, False)
                        
                        new_transition = (state, action_choice, reward, new_state, done)
                        memory.add(new_transition)
                        
                        state = new_state
                    
                    # train network
                    
                    sample_transitions = memory.sample(batch_size) #shape: (64, 5)
                    sample_states = np.array([transition[0] for transition in sample_transitions])
                    sample_actions = np.array([transition[1] for transition in sample_transitions])
                    sample_rewards = np.array([transition[2] for transition in sample_transitions])
                    sample_dones = np.array([transition[4] for transition in sample_transitions])
                    sample_next_states = np.array([transition[3] for transition in sample_transitions])
                    
                    sample_target_qs = []
                    
                    sample_q_next_states = sess.run(network.output, feed_dict = {network.inputs_: sample_next_states})
                    
                    for i in range(len(sample_transitions)):
                        if sample_dones[i]:
                            sample_target_qs.append(sample_rewards[i])
                        else:
                            sample_target_qs.append(sample_rewards[i] + gamma*np.max(sample_q_next_states[i]))
                            
                    # optimize weights
                    
                    sess.run(network.optimizer, feed_dict={network.inputs_: sample_states, network.actions_: sample_actions, network.target_Q: sample_target_qs})
                    
                    summary = sess.run(write_op, feed_dict={network.inputs_: sample_states, network.actions_: sample_actions, network.target_Q: sample_target_qs})
                    
                    writer.add_summary(summary, episode)
                    writer.flush()
                
                if episode % 5 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model Saved")
        
main()