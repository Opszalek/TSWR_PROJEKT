import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG, DQN, PPO, A2C
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

environments =["Hopper-v4"]#] ["Hopper-v4", "Swimmer-v4", "CartPole-v1", "Acrobot-v1"]
model_names = ["A2C"]#["DDPG", "DQN", "PPO", "A2C"]
def return_model(model_name):
    if model_name == "PPO":
        return PPO
    elif model_name == "DDPG":
        return DDPG
    elif model_name == "A2C":
        return A2C
    elif model_name == "DQN":
        return DQN

def test_models(environment_name, model_name):
    algorithm = return_model(model_name)
    model = algorithm.load('models/' + model_name + '/' + model_name + '_' + environment_name + '_model')
    env = gym.make(environment_name, render_mode="human")
    obs, info = env.reset()
    for _ in range(700):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # if terminated or truncated:
        #     break
        env.render()
    env.close()

for environment in environments:
    for model in model_names:
        test_models(environment, model)