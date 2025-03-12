import os
import csv
import json
from utils.logger import logging
# import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from render.user_interface import show_input_window, show_test_prompt_window, quit_pygame
from envs.fire_env import FireEnv

summary_shown = False


def run():
    global summary_shown
    log_csv = "./logs/logs.csv"
    if os.path.exists(log_csv):
        os.remove(log_csv)

    with open(log_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Шаг", "Заряд", "Средства", "Очагов осталось", "Вознаграждение", "Действие"])

    fire_count, obstacle_count = show_input_window()

    logging.info("Начинаем обучение модели...")
    model = train_and_evaluate(fire_count, obstacle_count)
    logging.info("Обучение завершено!")

    if show_test_prompt_window():
        test_env = FireEnv(fire_count=fire_count, obstacle_count=obstacle_count, render_mode="human")

        logging.info("Начинаем тест модели...")
        for episode in range(1, 5):
            obs, _ = test_env.reset()
            total_reward = 0
            rewards = []
            while True:
                action, _ = model.predict(obs)
                logging.info(f"Выбрано действие: {action}")
                obs, reward, terminated, truncated, info = test_env.step(action)
                total_reward += reward
                rewards.append(reward)

                with open(log_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([episode, test_env.iteration_count, test_env.battery_level,
                                     test_env.extinguisher_count,
                                     len(test_env.fires), reward, action])

                logging.info(f"Eposode = {episode},"
                             f" Шаг {test_env.iteration_count + 1}:"
                             f" Награда = {reward},"
                             f" Общая награда = {total_reward}, "
                             f"Очагов осталось:{len(test_env.fires)}, "
                             f"Батарея: {test_env.battery_level}, "
                             f"Огнетушителей: {test_env.extinguisher_count}")
                test_env.render()

                if terminated or truncated:
                    if len(test_env.fires) == 0:
                        print(f"Тестирование завершено на шаге {test_env.iteration_count}:"
                              f" Все очаги потушены! Общая награда: {total_reward}")
                    # elif test_env.battery_level <= 0:
                    #     print(f"Тестирование завершено на шаге {test_env.iteration_count}:"
                    #           f" Батарея разрядилась! Общая награда: {total_reward}")
                    elif test_env.iteration_count + 1 >= test_env.max_steps:
                        print(f"Тестирование завершено на шаге {test_env.iteration_count}:"
                              f" Достигнут лимит шагов. Общая награда: {total_reward}")
                    else:
                        print(f"Тестирование завершено на шаге {test_env.iteration_count} по "
                              f"неизвестной причине. Общая награда: {total_reward}")
                    if not summary_shown:
                        test_env.close()  # Вызываем close только один раз
                        summary_shown = True
                    break

        if not summary_shown:
            test_env.close()  # Закрываем среду, если не было вызова ранее
            summary_shown = True

    # Завершаем Pygame в конце программы
    quit_pygame()


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
        learning_rate=0.0001,
        n_steps=2048,
        batch_size=128,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        # ent_coef=0.05,
        clip_range_vf=0.2,
        # vf_coef=0.7,
        tensorboard_log=log_dir
    )

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path="./data/best_model/",
        log_path=log_dir,
        eval_freq=1000,
        render=False
    )

    # ОЛЯ: временно закомментированы колбеки
    # training_log_callback = TrainingLogCallback()

    model.learn(total_timesteps=500000,
                progress_bar=True)  # , callback=[eval_callback, training_log_callback], progress_bar=True)

    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    print(f"Средняя награда после обучения: {mean_reward} +/- {std_reward}")

    model.save("ppo_fire_model")
    return model