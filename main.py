import gymnasium as gym
from stable_baselines3 import PPO, A2C, DDPG, DQN
from stable_baselines3.common import results_plotter
from stable_baselines3.common.env_util import make_vec_env
import csv
import os
import matplotlib.pyplot as plt
import contextlib
import io
import re

environments = ["Hopper-v4", "Swimmer-v4", "CartPole-v1", "Acrobot-v1"]
model_names = ["DDPG", "DQN", "PPO", "A2C"]


def load_hyperparameters(environment_name, algorithm_name):
    hyperparameters = {}
    if not os.path.exists('hyperparameters/' + algorithm_name + '_hyperparameters' + '.csv'):
        raise FileNotFoundError(f'File {algorithm_name}_hyperparameters.csv doesn\'t exist')
    with open('hyperparameters/' + algorithm_name + '_hyperparameters' + '.csv', mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            env_name, param_key, param_value = row
            if env_name == environment_name:
                try:
                    if '.' in param_value:
                        param_value = float(param_value)
                    else:
                        param_value = int(param_value)
                except ValueError:
                    pass
                hyperparameters[param_key] = param_value
        time_steps = int(hyperparameters.pop("total_timesteps"))
        n_envs = int(hyperparameters.pop("n_envs"))
    return hyperparameters, time_steps, n_envs


def results_plotter(reward_vec, timesteps_vec, environment_name, algorithm_name):
    plt.plot(timesteps_vec, reward_vec)
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title(algorithm_name + ' ' + environment_name)
    plt.savefig('models/'+algorithm_name + '/' + algorithm_name + '_' + environment_name + '_plot.png')
    plt.show()


def capture_output(capture, environment_name, algorithm_name):
    captured_text = capture.getvalue()
    rewards = []
    timesteps_list = []
    for line in captured_text.splitlines():
        match_reward = re.search(r"ep_rew_mean\s*\|\s*(-?[\d.]+)", line)
        match_timesteps = re.search(r"timesteps\s*\|\s*([\d.]+)", line)
        print(line)
        if match_reward:
            rewards.append(float(match_reward.group(1)))
        if match_timesteps:
            timesteps_list.append(float(match_timesteps.group(1)))
    if len(rewards) != len(timesteps_list):
        rewards.append(timesteps_list[-1] + 1)
    results_plotter(rewards, timesteps_list, environment_name, algorithm_name)


def learn_model(environment_name, parameters, rl_algorithm):
    hyperparameters, time_steps, n_envs = parameters
    vec_env = make_vec_env(environment_name, n_envs=n_envs)
    model = rl_algorithm(policy="MlpPolicy", env=vec_env, verbose=1, **hyperparameters)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        model.learn(total_timesteps=time_steps)  # Uczenie modelu
    capture_output(capture, environment_name, rl_algorithm.__name__)
    model.save('models/' + str(rl_algorithm.__name__) + '/' + str(
        rl_algorithm.__name__) + '_' + environment_name + "_model")  # Zapisanie modelu
    vec_env.close()


def create_model(environment_name, model_name):
    print(environment_name)
    if model_name == "PPO":
        learn_model(environment_name, load_hyperparameters(environment_name, "PPO"), PPO)
    elif model_name == "DDPG" and environment_name != "CartPole-v1" and environment_name != "Acrobot-v1":
        learn_model(environment_name, load_hyperparameters(environment_name, "DDPG"), DDPG)
    elif model_name == "A2C":
        learn_model(environment_name, load_hyperparameters(environment_name, "A2C"), A2C)
    elif model_name == "DQN" and (environment_name == "CartPole-v1" or environment_name == "Acrobot-v1"):
        learn_model(environment_name, load_hyperparameters(environment_name, "DQN"), DQN)


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
    model = algorithm.load('models/' + model_name + '/' + model_name + '_' + environment_name + '_model.zip')
    model_chart = plt.imread('models/' + model_name + '/' + model_name + '_' + environment_name + '_plot.png')
    plt.imshow(model_chart)
    plt.axis('off')
    plt.show()
    env = gym.make(environment_name, render_mode="human")
    obs, info = env.reset()
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
        env.render()
    env.close()


def main():
    for environment_name in environments:
        for model in model_names:
            if not os.path.exists('models/' + model + '/' + model + '_' + environment_name + '_model.zip'):
                create_model(environment_name, model)
            else:
                test_models(environment_name, model)
    pass


main()