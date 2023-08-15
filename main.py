import gymnasium as gym
import multiprocessing
import threading
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor

import matplotlib.pyplot as plt
import time
from queue import Queue













































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