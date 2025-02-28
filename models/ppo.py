import os
import csv
import json
import logging
# import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from render.user_interface import show_input_window, show_summary_window
from envs.fire_env import FireEnv


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
                mean_reward, _ = evaluate_policy(self.model, self.training_env, n_eval_episodes=1, deterministic=True)
                episode_length = env.iteration_count
                fires_left = len(env.fires)
            except AttributeError as e:
                print(f"Ошибка доступа к атрибутам среды: {e}")
                mean_reward = evaluate_policy(self.model, self.training_env, n_eval_episodes=1, deterministic=True)[0]
                episode_length = self.step_count
                fires_left = -1

            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.num_timesteps, mean_reward, episode_length, fires_left])

            print(f"Логирование на шаге {self.num_timesteps}: Mean Reward = {mean_reward},"
                  f" Fires Left = {fires_left}")
        return True


def run():
    logger_file = "./logs/program.log"
    log_csv = "./logs/logs.csv"
    if os.path.exists(log_csv):
        os.remove(log_csv)

    logging.basicConfig(filename=logger_file, level=logging.INFO,
                        format='%(asctime)s,%(message)s', filemode='w')
    # logger = logging.getLogger()

    with open(log_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Итерация", "Заряд", "Средства", "Очагов осталось", "Вознаграждение", "Action"])

    fire_count, obstacle_count = show_input_window()

    if fire_count is None or obstacle_count is None:
        print("Программа завершена пользователем.")
        return

    logging.info("Начинаем обучение модели...")
    model = train_and_evaluate(fire_count, obstacle_count)
    logging.info("Обучение завершено!")

    test_env = FireEnv(fire_count=fire_count, obstacle_count=obstacle_count, render_mode="human")
    obs, _ = test_env.reset()
    total_reward = 0
    iteration_count = 0
    max_steps = 1000
    rewards = []

    logging.info("Начинаем test модели...")
    for step in range(max_steps):
        action, _states = model.predict(obs)#, deterministic=True)
        logging.info(f"Выбрано действие: {action}")
        obs, reward, done, _, info = test_env.step(action)
        total_reward += reward
        iteration_count += 1
        rewards.append(reward)

        log_message = [iteration_count, test_env.battery_level, test_env.extinguisher_count,
                       len(test_env.fires), reward, action]
        with open(log_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_message)

        logging.info(f"Шаг {step + 1}: Награда = {reward}, Общая награда = {total_reward}, "
              f"Очагов осталось: {len(test_env.fires)}, Батарея: {test_env.battery_level}, "
              f"Огнетушителей: {test_env.extinguisher_count}")
        test_env.render()

        if done:
            if len(test_env.fires) == 0:
                print(f"Тестирование завершено на шаге {step + 1}:"
                      f" Все очаги потушены! Общая награда: {total_reward}")
            elif test_env.battery_level <= 0:
                print(f"Тестирование завершено на шаге {step + 1}:"
                      f" Батарея разрядилась! Общая награда: {total_reward}")
            elif step + 1 == max_steps:
                print(f"Тестирование завершено на шаге {step + 1}:"
                      f" Достигнут лимит шагов. Общая награда: {total_reward}")
            else:
                print(f"Тестирование завершено на шаге {step + 1} по "
                      f"неизвестной причине. Общая награда: {total_reward}")
            break
    #
    # plt.plot(rewards)
    # plt.title("Награды за каждый шаг тестирования")
    # plt.xlabel("Шаг")
    # plt.ylabel("Награда")
    # plt.show()

    show_summary_window(fire_count, obstacle_count, iteration_count, total_reward)
    test_env.close()


def train_and_evaluate(fire_count, obstacle_count):
    def make_env():
        env = FireEnv(fire_count=fire_count, obstacle_count=obstacle_count, render_mode=None)
        return env

    vec_env = make_vec_env(make_env, n_envs=1)

    log_dir = "./logs/ppo_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=128,
        n_epochs=3,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        tensorboard_log=log_dir
    )

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path="./data/best_model/",
        log_path=log_dir,
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    # ОЛЯ: временно закомментированы колбеки
    # training_log_callback = TrainingLogCallback()

    model.learn(total_timesteps=5000)  # , callback=[eval_callback, training_log_callback], progress_bar=True)

    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10, deterministic=True)
    print(f"Средняя награда после обучения: {mean_reward} +/- {std_reward}")

    model.save("ppo_fire_model")
    return model
