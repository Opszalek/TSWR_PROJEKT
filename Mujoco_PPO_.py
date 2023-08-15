import gymnasium as gym
import multiprocessing
import threading
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

import matplotlib.pyplot as plt
enviroment_name = "CartPole-v1"
# Tworzenie środowiska
vec_env = make_vec_env(enviroment_name, n_envs=15)
# env = gym.make(enviroment_name,render_mode="human")
time_steps = 1e3  # Liczba kroków uczenia

# Tworzenie modelu PPO
model_params = {
    "policy": 'MlpPolicy',
    "env": vec_env,
    "batch_size": 32,
    "clip_range": 0.2,
    "ent_coef": 0.00229519,
    "gae_lambda": 0.99,
    "gamma": 0.999,
    "learning_rate": 9.80828e-05,
    "max_grad_norm": 0.7,
    "n_steps": 512,
    "n_epochs": 5,
    "verbose": 1,
    "vf_coef": 0.835671,
}

# Tworzenie modelu PPO z dostarczonymi parametrami
log_dir = "tmp/"
model = PPO(**model_params)

desired_reward = 1900  # Pożądana nagroda

model.learn(total_timesteps=int(time_steps))  # Uczenie modelu

plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, enviroment_name)
plt.show()
model.save(enviroment_name + "_model")  # Zapisanie modelu

vec_env.close()
model = PPO.load(enviroment_name + "_model")  # Wczytanie modelu

#Testowanie modelu
env = gym.make(enviroment_name,render_mode="human")
obs, info = env.reset()

# Evaluate the agent
episode_reward = 0
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    env.render()
    # if terminated or truncated or info.get("is_success", False):
    #     print("Reward:", episode_reward, "Success?", info.get("is_success", False))
    #     episode_reward = 0.0
    #     obs, info = env.reset()
print("Reward:", episode_reward)
env.close()
def ppo_hopper_model():
    pass
def ppo_cartpole_model():
    pass
def ppo_swimmer_model():
    pass
def ppo_acrobot_model():
    pass

def ppo_models():
    ppo_hopper_model()
    ppo_cartpole_model()
    ppo_swimmer_model()
    ppo_acrobot_model()

def main():
    ppo_models()
    pass