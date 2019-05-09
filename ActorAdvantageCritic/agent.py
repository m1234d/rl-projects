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
    for i in range(10):
        environment_list.append(env.make_env)
    
    env_vector = SubprocVecEnv(environment_list)
    
    with tf.Session(config=config):
        model.learn(policy=policies.A2CNetwork,
                    env=env_vector,
                    nsteps=2048,
                    total_timesteps=10000000,
                    gamma=0.99,
                    lam=0.95,
                    vf_coef=0.5,
                    ent_coef=0.01,
                    lr=2e-4,
                    max_grad_norm=0.5,
                    log_interval=2,
                    restart=True)

if __name__ == "__main__":
    main()