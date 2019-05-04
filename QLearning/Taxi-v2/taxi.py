import numpy as np
import gym
import random

env = gym.make("Taxi-v2")
env.render()

#To create q-table, we need number of actions (columns) and number of states (rows)
action_size = env.action_space.n
print("Action size: ", action_size)

state_size = env.observation_space.n
print("State size: ", state_size)

qtable = np.zeros((state_size,action_size))

total_episodes = 50000
total_test_episodes = 100
max_steps = 99

learning_rate = 0.7
gamma = 0.618

#Exploration parameters
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.01

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_reward = 0
    for step in range(max_steps):
        action = np.argmax(qtable[state,:])
        new_state,reward,done,info = env.step(action)
        total_reward += reward        
        
        state = new_state
        if done:
            break
    print(total_reward)

for episode in range(total_episodes):
    print(episode)
    state = env.reset()
    step = 0
    done = False
    for step in range(max_steps):
        randEpsValue = random.uniform(0,1)
        if randEpsValue > epsilon:
            action = np.argmax(qtable[state,:]) #exploit
        else:
            action = env.action_space.sample() #explore
            
        new_state, reward, done, info = env.step(action)
        
        qtable[state,action] = qtable[state,action] + learning_rate * (reward + gamma*np.max(qtable[new_state,:]) - qtable[state,action])
        
        state = new_state
        
        if done:
            break
    
    epsilon = min_epsilon + (max_epsilon - min_epsilon * np.exp(-decay_rate*episode))

runningAverage = 0

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_reward = 0
    if episode % 1000 == 0:
        print(runningAverage)
    for step in range(max_steps):
        action = np.argmax(qtable[state,:])
        new_state,reward,done,info = env.step(action)
        total_reward += reward        
        
        state = new_state
        if done:
            break
    runningAverage = ((runningAverage * (episode)) + total_reward) / (episode+1)

print(runningAverage)