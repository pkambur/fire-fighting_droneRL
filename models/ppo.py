import os
import csv
import json
import random
import logging
import matplotlib.pyplot as plt

from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from render.user_interface import show_input_window, show_summary_window
from envs.fire_env import FireEnv


def run():
    log_file = "./logs/logs.csv"
    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig(filename = log_file, level = logging.INFO,
                        format = '%(asctime)s,%(message)s',
                        filemode = 'w')
    logger = logging.getLogger()

    with open(log_file, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(["Итерация", "Заряд", "Средства", "Очагов осталось", "Вознаграждение"])

    fire_count, obstacle_count = show_input_window()

    if fire_count is None or obstacle_count is None:
        print("Программа завершена пользователем.")
        return

    test_env = FireEnv(fire_count = fire_count, obstacle_count = obstacle_count, render_mode = "human")
    test_env.reset()
    total_reward = 0
    iteration_count = 0
    max_steps = 1000
    rewards = []

    for step in range(max_steps):
        # Если огнетушителей нет, возвращаемся на базу
        if test_env.extinguisher_count == 0:
            action = test_env.get_action_towards_base()
            print(f"Возвращаемся на базу за огнетушителем: Огнетушителей = {test_env.extinguisher_count}")
        # Если заряд меньше 30 и не на базе, возвращаемся
        elif test_env.battery_level < 30 and test_env.position != test_env.base:
            action = test_env.get_action_towards_base()
            print(f"Возвращаемся на базу: Батарея = {test_env.battery_level}")
        # Если на очаге и есть огнетушитель, тушим
        elif test_env.position in test_env.fires and test_env.extinguisher_count > 0:
            action = 4
        # Иначе случайное движение
        else:
            action = random.randint(0, 3)

        obs, reward, done, _, info = test_env.step(action)
        total_reward += reward
        iteration_count += 1
        rewards.append(reward)

        log_message = [iteration_count, test_env.battery_level, test_env.extinguisher_count,
                       len(test_env.fires), reward]
        with open(log_file, mode = 'a', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow(log_message)

        print(f"Шаг {step + 1}: Награда = {reward}, Общая награда = {total_reward}, "
              f"Очагов осталось: {len(test_env.fires)}, Батарея: {test_env.battery_level}, "
              f"Огнетушителей: {test_env.extinguisher_count}")
        test_env.render()

        if done:
            if len(test_env.fires) == 0:
                print(f"Тестирование завершено на шаге {step + 1}: Все очаги потушены! Общая награда:"
                      f"{total_reward}")
            elif test_env.battery_level <= 0:
                print(f"Тестирование завершено на шаге {step + 1}: Батарея разрядилась! Общая награда:"
                      f"{total_reward}")
            else:
                print(f"Тестирование завершено на шаге {step + 1} по неизвестной причине. Общая награда:"
                      f"{total_reward}")
            break

    plt.plot(rewards)
    plt.title("Награды за каждый шаг тестирования")
    plt.xlabel("Шаг")
    plt.ylabel("Награда")
    plt.show()

    show_summary_window(fire_count, obstacle_count, iteration_count, total_reward)
    test_env.close()


    # # обучение модели
    # print("Начинаем обучение модели...")
    # model = train_and_evaluate(fire_count, obstacle_count)
    # print("Обучение завершено!")
    #
    # # предложение протестировать модель
    # test_choice = show_test_prompt_window()
    #
    # # Создаем среду для тестирования с визуализацией
    # test_env = FireEnv(fire_count = fire_count, obstacle_count = obstacle_count, render_mode = "human")
    #
    # # Тестирование среды без модели (случайные действия)
    # obs, info = test_env.reset()
    # total_reward = 0
    # iteration_count = 0
    # max_steps = 100
    # rewards = []
    #
    # for step in range(max_steps):
    #     # Закомментируем предсказание модели
    #     # action, _states = model.predict(obs, deterministic = True)
    #     # Используем случайное действие вместо модели
    #     action = random.randint(0, 4)  # 0: up, 1: down, 2: left, 3: right, 4: extinguish
    #
    #     obs, reward, done, _, info = test_env.step(action)
    #     total_reward += reward
    #     iteration_count += 1
    #     rewards.append(reward)
    #
    #     # Логирование
    #     log_message = [iteration_count, test_env.battery_level, test_env.extinguisher_count, len(test_env.fires), reward]
    #     with open(log_file, mode = 'a', newline = '') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(log_message)
    #
    #     # Дополнительное логирование прогресса
    #     log_training_progress(step, reward, total_reward)
    #
    #     # Вывод информации
    #     print(f"Шаг {step + 1}: Награда = {reward}, Общая награда = {total_reward}")
    #
    #     # Отрисовка
    #     test_env.render()
    #
    #     if done:
    #         print(f"Тестирование завершено на шаге {step + 1}. Общая награда: {total_reward}")
    #         break
    #
    # # Визуализация вознаграждений после тестирования
    # plt.plot(rewards)
    # plt.title("Награды за каждый шаг тестирования")
    # plt.xlabel("Шаг")
    # plt.ylabel("Награда")
    # plt.show()
    #
    # # Показать итоговое окно
    # show_summary_window(fire_count, obstacle_count, iteration_count, total_reward)
    #
    # # Закрытие
    # test_env.close()


def train_and_evaluate(fire_count, obstacle_count):
    def make_env():
        env = FireEnv(fire_count = fire_count, obstacle_count = obstacle_count, render_mode = None)
        return env

    vec_env = make_vec_env(lambda: make_env(), n_envs = 1)

    log_dir = "./logs/ppo_tensorboard/"
    os.makedirs(log_dir, exist_ok = True)

    model = PPO("MlpPolicy", vec_env, verbose = 1, learning_rate = 0.0003,
                n_steps = 1024, batch_size = 64, n_epochs = 5, gamma = 0.99, tensorboard_log = log_dir)

    eval_callback = EvalCallback(vec_env, best_model_save_path = ".data/best_model/",
                                 log_path = log_dir, eval_freq = 500, deterministic = True, render = False)

    model.learn(total_timesteps = 2000, callback = eval_callback)

    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes = 10, deterministic = True)
    print(f"Средняя награда после обучения: {mean_reward} +/- {std_reward}")

    model.save("ppo_fire_model")

    return model

def log_training_progress(step, reward, total_reward, log_file = "./logs/training_progress.json"):
    data = {
        "step": step,
        "reward": reward,
        "total_reward": total_reward,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(log_file, 'a') as f:
        json.dump(data, f)
        f.write('\n')