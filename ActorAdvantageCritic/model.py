import numpy as np
import tensorflow as tf
import os
import time
from baselines import logger
import cv2

import matplotlib.pyplot as plt

from baselines.a2c.utils import cat_entropy


from baselines.common import explained_variance
from baselines.common.runners import AbstractEnvRunner

def mse(pred, target):
    return tf.square(pred-target)/2

### A2C Model Class
# init: create two neural nets, the step and train net, and constructs loss graph
# train: takes as input a set of states, actions, returns, and values, performs training, and outputs losses
# save: saves model
# load: loads model

class Model():
    
    def __init__(self, policy, observation_space,
    action_space, number_envs, number_steps, ent_coef, vf_coef, max_grad_norm):
        sess = tf.get_default_session()
        
        actions = tf.placeholder(tf.int32, [None], name="actions")
        advantages = tf.placeholder(tf.float32, [None], name="advantages")
        rewards = tf.placeholder(tf.float32, [None], name="rewards")
        lr = tf.placeholder(tf.float32, name="learning_rate")
        
        step_model = policy(sess, observation_space, action_space, number_envs, 1, reuse=False)
        
        train_model = policy(sess, observation_space, action_space, number_envs*number_steps, number_steps, reuse=True)
        
        # Total Loss: policy gradient loss - (entropy * entropy_coefficient) + (value*value_coefficient)
        
        # -log(softmax(neural net))

        neglogp = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=actions)
        
        # policy gradient loss: 1/n * sum A(s_i,a_i) * -log(softmax(neural net(a_i | s_i)))
        pg_loss = tf.reduce_mean(advantages * neglogp)
        
        #value loss: 1/2 * sum (r - v(s))^2
        
        vf_loss = tf.reduce_mean (mse(tf.squeeze(train_model.vf), rewards))
        
        # improve exploration by limiting convergence
        entropy = tf.reduce_mean(train_model.pd.entropy())
        
        loss = pg_loss - (entropy * ent_coef) + (vf_loss * vf_coef)
        
        # get trainable parameters
        params = tf.trainable_variables("model")
        
        # get gradient
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads,params))
        
        trainer = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5)
        
        train_obj = trainer.apply_gradients(grads)
        
        # Calculate advantage from returns and values, use to compute
        def train(states_in, actions_, returns_, values_, lr_):
            
            # returns = (bootstrap) q-value estimation (reward + gamma*V(s'))
            advantages_ = returns_ - values_
            
            feed_dict = {train_model.inputs: states_in,
            actions: actions_,
            advantages: advantages_,
            rewards: returns_,
            lr: lr_}
            
            policy_loss, value_loss, entropy_, _ = sess.run([pg_loss, vf_loss, entropy, train_obj], feed_dict=feed_dict)
            
            return policy_loss, value_loss, entropy_
        
        def save(save_path):
            saver = tf.train.Saver()
            saver.save(sess, save_path)
            
        def load(save_path):
            saver = tf.train.Saver()
            saver.restore(sess, save_path)
            print("Model loaded.")
            
        
        self.train = train
        self.train_model = train_model
        self.save = save
        self.load = load
        self.step = step_model.step
        self.step_model = step_model
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        
        tf.global_variables_initializer().run(session=sess)

# Runs training
class Runner(AbstractEnvRunner):
    def __init__(self, env, model, number_steps, total_timesteps, gamma, lam):
        super().__init__(env=env, model=model, nsteps=number_steps)
        
        self.gamma = gamma
        
        self.lam = lam
        
        self.total_timesteps = total_timesteps
        
    # Collect a set of experiences
    def run(self):
        observations_list, actions_list, rewards_list, values_list, dones_list = [], [], [], [], []
        
        for n in range(self.nsteps):
            actions,values = self.model.step(self.obs)
            
            observations_list.append(np.copy(self.obs))
            actions_list.append(actions)
            values_list.append(values)
            dones_list.append(np.copy(self.dones))
            
            self.obs[:], rewards, self.dones, _ = self.env.step(actions)
            
            rewards_list.append(rewards)
        
        observations_list = np.asarray(observations_list, dtype=np.uint8)
        actions_list = np.asarray(actions_list, dtype=np.int32)
        rewards_list = np.asarray(rewards_list, dtype=np.float32)
        values_list = np.asarray(values_list, dtype=np.float32)
        dones_list = np.asarray(dones_list, dtype=np.bool)
        
        last_values = self.model.value(self.obs)
        
        returns_list = np.zeros_like(rewards_list)
        advantages_list = np.zeros_like(rewards_list)
        
        last_gae_lam = 0
        
        for t in reversed(range(self.nsteps)):
            
            # if we are in a final state, there is no value of the next state, so set modifier (nextnonterminal) to 0
            if t == self.nsteps - 1:
                next_non_terminal = 1.0 - self.dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones_list[t+1]
                next_values = values_list[t+1]
            
            # return function. r_t + gamma*V(s_t+1) - V(s_t)
            delta = rewards_list[t] + self.gamma * next_values * next_non_terminal - values_list[t]
            
            # advantage. delta + gamma * lambda * next_non_terminal * last_gae_lam
            advantages_list[t] = delta + self.gamma * self.lam * next_non_terminal * last_gae_lam
            last_gae_lam = advantages_list[t]
        
        returns_list = advantages_list + values_list
        
        return map(sf01, (observations_list, actions_list, returns_list, values_list, rewards_list))
        
def sf01(arr):
    s = arr.shape
    swapped = arr.swapaxes(0,1)
    reshaped = swapped.reshape(s[0] * s[1], *s[2:])
    return reshaped
    
## Learn Function
# Takes as input a policy, environment, number of steps, total number of time steps, gamma, lambda, value coefficient, entropy coefficient, learning rate, max gradient norm, log_interval and executes training
def learn(policy, env, nsteps, total_timesteps, gamma, lam, vf_coef, ent_coef, lr, max_grad_norm, log_interval, restart):
    number_epochs = 4
    number_mini_batches = 8
    
    nenvs = env.num_envs
    
    observation_space = env.observation_space
    action_space = env.action_space
    
    batch_size = nenvs * nsteps
    batch_train_size = batch_size // number_mini_batches
    
    assert batch_size % number_mini_batches == 0
    
    # Construct model
    model = Model(policy=policy,
                observation_space=observation_space,
                action_space=action_space,
                number_envs=nenvs,
                number_steps=nsteps,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm
                )

    if not restart:
        model.load("./models/model.ckpt")

    runner = Runner(env, model, nsteps, total_timesteps, gamma, lam)
    
    time_first_start = time.time()
    
    # For every batch of the game
    for update in range(1, total_timesteps // batch_size+1):
        time_start = time.time()
        
        # Generate observations
        observations, actions, returns, values, rewards = runner.run()
        
        reward_sum = np.sum(rewards)
        
        losses_list = []
        total_batches_train = 0
        
        indices = np.arange(batch_size)
        for epoch in range(number_epochs):
            np.random.shuffle(indices)
            
            # Feed minibatches to model for training
            for mini_start in range(0, batch_size, batch_train_size):
                mini_end = mini_start + batch_train_size
                mini_indices = indices[mini_start:mini_end]

                mini_states = observations[mini_indices]
                mini_actions = actions[mini_indices]
                mini_returns = returns[mini_indices]
                mini_values = values[mini_indices]

                losses_list.append(model.train(mini_states,mini_actions,mini_returns,mini_values,lr))
        
        loss_values = np.mean(losses_list, axis=0)
        time_now = time.time()
        fps = int(batch_size / (time_now - time_start))
        
        # Print out updates
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*batch_size)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_loss", float(loss_values[0]))
            logger.record_tabular("policy_entropy", float(loss_values[2]))
            logger.record_tabular("value_loss", float(loss_values[1]))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("time_elapsed", float(time_now - time_first_start))
            logger.record_tabular("total_reward", float(reward_sum))
            logger.dump_tabular()

            model.save("./models/model.ckpt")
            print("Model saved.")
    env.close()

def play(policy, env):
    observation_space = env.observation_space
    action_space = env.action_space
    
    model = Model(policy=policy,
                  observation_space=observation_space,
                  action_space=action_space,
                  number_envs=1,
                  number_steps=1,
                  ent_coef=0,
                  vf_coef=0,
                  max_grad_norm=0)
                  
    model.load("./models/model.ckpt")
    
    obs = env.reset()
    
    score=0
    done = False
    while done == False:
        actions,values = model.step(obs)
        obs, rewards,done,_ = env.step(actions)
        score += rewards
        
        env.render()
    
    print("Score: ", score)
    env.close()