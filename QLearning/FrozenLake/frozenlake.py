import numpy as np
import gym
import random


env = gym.make("FrozenLake-v0")
env.render()

action_size = env.action_space.n
state_size = env.observation_space.n
qtable = np.zeros((state_size,action_size))

total_episodes = 100000
total_test_episodes = 1000

max_steps = 999

learning_rate = 0.7
gamma = 0.95
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

average = 0
for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_reward = 0
    for step in range(max_steps):
        action = np.argmax(qtable[state,:])
        newState,reward,done,_ = env.step(action)
        total_reward += reward
        state = newState
        if done:
            break
    average += total_reward

print(average/total_test_episodes)

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    done = False
    for step in range(max_steps):
        randEps = random.uniform(0,1)
        if randEps > epsilon:
            action = np.argmax(qtable[state,:])
        else:
            action = env.action_space.sample()
        
        newState, reward, done, info = env.step(action)
        
        qtable[state,action] = qtable[state,action] + learning_rate * (reward + gamma*np.max(qtable[newState,:]) - qtable[state,action])
        
        state = newState
        
        if done:
            break
    
    epsilon = min_epsilon + (max_epsilon - min_epsilon * np.exp(-decay_rate*episode))

average = 0
for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_reward = 0
    for step in range(max_steps):
        action = np.argmax(qtable[state,:])
        newState,reward,done,_ = env.step(action)
        total_reward += reward
        state = newState
        if done:
            break
    average += total_reward

print(average/total_test_episodes)