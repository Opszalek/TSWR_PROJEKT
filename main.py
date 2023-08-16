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
import csv
import os
import matplotlib.pyplot as plt

environments = ["Hopper-v4", "Swimmer-v4", "Cartpole-v1", "Acrobot-v1"]
model_names = ["PPO", "DDPG", "A2C", "SAC"]

def load_hyperparameters(environment_name, algoritm_name):
    hyperparameters = {}

    with open('hyperparameters/' + algoritm_name + '_hyperparameters' + '.csv', mode='r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            env_name, param_key, param_value = row
            if env_name == environment_name:
                hyperparameters[param_key] = param_value

    return hyperparameters

def ppo_create_model(environment_name, parameters):
    time_steps = int(float(parameters.pop("total_timesteps")))
    n_envs = parameters.pop("n_envs")
    print(parameters)
    vec_env = make_vec_env(environment_name, n_envs=int(n_envs))
    model = PPO(env=vec_env, verbose=1, **parameters)
    model.learn(total_timesteps=time_steps)  # Uczenie modelu
    model.save('models/' + environment_name + "_model")  # Zapisanie modelu
    vec_env.close()

def ppo_algoritm(environments):



    for environment_name in environments:
        parameters = load_hyperparameters(environment_name, "PPO")
        time_steps = int(float(parameters.pop("total_timesteps")))
        n_envs = parameters.pop("n_envs")
        print(parameters)
        vec_env = make_vec_env(environment_name, n_envs=int(n_envs))
        model = PPO(env=vec_env,verbose=1,**parameters)
        model.learn(total_timesteps=time_steps)  # Uczenie modelu
        model.save('models/' + environment_name + "_model")  # Zapisanie modelu
        vec_env.close()

def create_model(environment_name,model_name):

    pass
def test_models():
    for environment_name in environments:
        for model in model_names:
            # print(model,environment_name)
            if not os.path.exists('models/'+model+'/'+environment_name+'_model.zip'):
                pass


def main():
    # ppo_models(environments)##
    # ddpg_models(environments)
    # a2c_models(environments) ##
    # sac_models(environments)
    test_models()
    params = load_hyperparameters("Swimmer-v4", "PPO")
    print(params)

    pass


main()

############Function to create a plot in real time

# env_id = "Swimmer-v4"
# env = gym.make(env_id)
# log_dir = "tmp/"
# model = PPO('MlpPolicy', env, verbose=1)
# timesteps = 1000
#
# model.learn(total_timesteps=int(timesteps))
#
# plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "CartPole-v1")
# plt.show()


#
# # Funkcja do tworzenia wykresu w czasie rzeczywistym
# class RealtimePlotCallback(EvalCallback):
#     def __init__(self, *args, update_interval=10, queue=None, **kwargs):
#         super(RealtimePlotCallback, self).__init__(*args, **kwargs)
#         self.update_interval = update_interval
#         self.queue = queue
#
#     def _on_step(self):
#         if self.queue is not None:
#             self.queue.put(self.best_mean_reward)
#

# # Środowisko i model
# env_id = "Swimmer-v4"
# env = gym.make(env_id)
# model = PPO('MlpPolicy', env, verbose=1)
#
# update_interval = 5
# queue = Queue()
#
# # Uruchomienie wątku do aktualizacji wykresu
# plotter = threading.Thread(target=plotter_thread, args=(queue, update_interval))
# plotter.daemon = True
# plotter.start()
#
# # Callback do generowania danych dla wykresu
# callback = RealtimePlotCallback(eval_env=env, n_eval_episodes=10, eval_freq=1000, update_interval=update_interval,
#                                 queue=queue)
#
# # Trening modelu z callbackiem
# model.learn(total_timesteps=100000, callback=callback)
#######################################
#
#
# param_sets = [
#     {"env_id": "Swimmer-v4", "model_name": "swimmer_model", "total_timesteps": 100000},
#     {"env_id": "Hopper-v4", "model_name": "hopper_model", "total_timesteps": 200000}
# ]
#
# processes = []
# for param_set in param_sets:
#     process = multiprocessing.Process(target=train_model, kwargs=param_set)
#     processes.append(process)
#     process.start()
#
# # Czekanie na zakończenie wszystkich procesów
# for process in processes:
#     process.join()
