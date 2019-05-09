import numpy as np
import gym

from baselines.common.atari_wrappers import FrameStack

import cv2

# Custom observation wrapper to preprocess frames
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super(PreprocessFrame, self).__init__(env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)
    
    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:,:,None]
        
        return frame

# Scale rewards to more reasonable amount
class RewardScaler(gym.RewardWrapper):
    def reward(self, reward):
        return reward * 0.01
        
        
# Create environment
def make_env():
    env = gym.make("PongDeterministic-v0")
    env = PreprocessFrame(env)
    env = RewardScaler(env)
    env = FrameStack(env, 4)
    return env
    
