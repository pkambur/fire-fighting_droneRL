import csv
import os

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class TrainingLogCallback(BaseCallback):
    def __init__(self, verbose=0, log_file="./logs/training_logs.csv"):
        super(TrainingLogCallback, self).__init__(verbose)
        self.log_file = log_file
        self.step_count = 0
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestep", "Mean Reward", "Episode Length", "Fires Left"])

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % 1000 == 0:
            env = self.training_env.envs[0]
            while hasattr(env, 'env'):
                env = env.env

            try:
                mean_reward, _ = evaluate_policy(self.model, self.training_env, n_eval_episodes=1)
                episode_length = env.iteration_count
                fires_left = len(env.fires)
            except AttributeError as e:
                print(f"Ошибка доступа к атрибутам среды: {e}")
                mean_reward = evaluate_policy(self.model, self.training_env, n_eval_episodes=1)[0]
                episode_length = self.step_count
                fires_left = -1

            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.num_timesteps, mean_reward, episode_length, fires_left])
            print(f"Логирование на шаге {self.num_timesteps}: Mean Reward = {mean_reward},"
                  f" Fires Left = {fires_left}")
        return True
