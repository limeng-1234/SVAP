#%%  导入包库
import gym
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from stable_baselines3 import DQN
import highway_env

shap.initjs()
#%% 训练过程：可设置直接加载模型或是训练模式
# TRAIN = False
Learn = True



#%% Create the environment
env = gym.make("highway-v0")  # "highway-fast-v0"是一个环境ID，定义为env
env.config["real_time_rendering"] = True
env.config["lanes_count"] = 1
env.config["vehicles_count"] = 2
env.config["vehicles_density"] = 1   #这个改动了
env.config["collision_reward"] = -1
env.config["right_lane_reward"] = 0
env.config["reward_speed_range"] = [15, 30]
env.config["lane_change_reward"] = 0
env.config["high_speed_reward"] = 0.2
# env.config["manual_control"] = True
env.config["simulation_frequency"] = 15
env.config["duration"] = 40
env.config["show_trajectories"] = False
env.config.update({
    "observation": {
        "type": "Kinematics"
    },
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal":True,  # 只有横向动作
        "lateral":False
    }
})
obs = env.reset()  # 随机获取环境中的状态 # 随机获取环境中的状态
for i in range(16, 19, 1):
    #%% 训练模型
    file_path_model = 'dqn_attri/' + "model_%d" % (i)
    file_path_log = 'dqn_attri/' + "log_%d/" % (i)
    data_path = 'dqn_attri/' + "data_%d.pth" % (i)
    model = DQN('MlpPolicy', env,
                # policy_kwargs=dict(net_arch=[128, 128]),
                learning_rate=2e-4,
                buffer_size=15000,
                learning_starts=50,
                batch_size=64,
                gamma=0.5,
                train_freq=1,
                exploration_fraction=0.2,     #这个改动了
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                self_define=i,
                tensorboard_log=file_path_log)
    if Learn:
        model.learn(total_timesteps=15000)
        model.save(file_path_model)






