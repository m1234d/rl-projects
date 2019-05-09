import math
import os
import tensorflow as tf

import model
import architecture as policies
import env

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

def main():
    config = tf.ConfigProto()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    config.gpu_options.allow_growth = True
    environment_list = []
    for i in range(1):
        environment_list.append(env.make_env)
    
    env_vector = SubprocVecEnv(environment_list)
    
    with tf.Session(config=config):
        model.play(policy=policies.A2CNetwork,
                    env=env_vector)

if __name__ == "__main__":
    main()