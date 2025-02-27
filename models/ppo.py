import os
import csv
import json
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

    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s,%(message)s', filemode='w')
    logger = logging.getLogger()

    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Итерация", "Заряд", "Средства", "Очагов осталось", "Вознаграждение"])

    fire_count, obstacle_count = show_input_window()

    if fire_count is None or obstacle_count is None:
        print("Программа завершена пользователем.")
        return

    # Обучение модели
    print("Начинаем обучение модели...")
    model = train_and_evaluate(fire_count, obstacle_count)
    print("Обучение завершено!")

    # Тестирование с использованием обученной модели
    test_env = FireEnv(fire_count=fire_count, obstacle_count=obstacle_count, render_mode="human")
    obs, _ = test_env.reset()
    total_reward = 0
    iteration_count = 0
    max_steps = 1000
    rewards = []

    for step in range(max_steps):
        # Используем модель для предсказания действия
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, done, _, info = test_env.step(action)
        total_reward += reward
        iteration_count += 1
        rewards.append(reward)

        log_message = [iteration_count, test_env.battery_level, test_env.extinguisher_count,
                       len(test_env.fires), reward]
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_message)

        print(f"Шаг {step + 1}: Награда = {reward}, Общая награда = {total_reward}, "
              f"Очагов осталось: {len(test_env.fires)}, Батарея: {test_env.battery_level}, "
              f"Огнетушителей: {test_env.extinguisher_count}")
        test_env.render()

        if done:
            if len(test_env.fires) == 0:
                print(f"Тестирование завершено на шаге {step + 1}: Все очаги потушены! Общая награда: {total_reward}")
            elif test_env.battery_level <= 0:
                print(f"Тестирование завершено на шаге {step + 1}: Батарея разрядилась! Общая награда: {total_reward}")
            else:
                print(f"Тестирование завершено на шаге {step + 1} по неизвестной причине. Общая награда: {total_reward}")
            break

    plt.plot(rewards)
    plt.title("Награды за каждый шаг тестирования")
    plt.xlabel("Шаг")
    plt.ylabel("Награда")
    plt.show()

    show_summary_window(fire_count, obstacle_count, iteration_count, total_reward)
    test_env.close()

def train_and_evaluate(fire_count, obstacle_count):
    def make_env():
        env = FireEnv(fire_count=fire_count, obstacle_count=obstacle_count, render_mode=None)
        return env

    vec_env = make_vec_env(make_env, n_envs=1)

    log_dir = "./logs/ppo_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    # Настройка гиперпараметров PPO
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1,
        learning_rate=0.0001,  # Уменьшенный learning_rate для стабильности
        n_steps=2048,          # Увеличенные шаги для лучшего планирования
        batch_size=128,        # Увеличенный размер батча
        n_epochs=10,           # Больше эпох для лучшей оптимизации
        gamma=0.95,            # Уменьшенный gamma для краткосрочных целей
        gae_lambda=0.95,       # GAE для лучшей оценки преимущества
        clip_range=0.2,        # Стандартное значение для PPO
        ent_coef=0.01,         # Коэффициент энтропии для поощрения исследования
        tensorboard_log=log_dir
    )

    # Callback для сохранения лучшей модели
    eval_callback = EvalCallback(
        vec_env, 
        best_model_save_path="./data/best_model/",
        log_path=log_dir, 
        eval_freq=1000, 
        deterministic=True, 
        render=False
    )

    # Увеличенное количество шагов обучения
    model.learn(total_timesteps=100000, callback=eval_callback, progress_bar=True)

    # Оценка модели
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10, deterministic=True)
    print(f"Средняя награда после обучения: {mean_reward} +/- {std_reward}")

    model.save("ppo_fire_model")
    return model

def log_training_progress(step, reward, total_reward, log_file="./logs/training_progress.json"):
    data = {
        "step": step,
        "reward": reward,
        "total_reward": total_reward,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(log_file, 'a') as f:
        json.dump(data, f)
        f.write('\n')

if __name__ == "__main__":
    run()




# import pygame
# #import config.colors as colors

# import os
# import csv
# import json
# import random
# import logging
# import matplotlib.pyplot as plt

# from datetime import datetime
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.callbacks import EvalCallback
# from render.user_interface import show_input_window, show_summary_window, show_test_prompt_window
# from envs.fire_env import FireEnv


# def run():
#     log_file = "./logs/logs.csv"
#     if os.path.exists(log_file):
#         os.remove(log_file)

#     logging.basicConfig(filename = log_file, level = logging.INFO,
#                         format = '%(asctime)s,%(message)s',
#                         filemode = 'w')
#     logger = logging.getLogger()

#     with open(log_file, mode = 'w', newline = '') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Итерация", "Заряд", "Средства", "Очагов осталось", "Вознаграждение"])

#     fire_count, obstacle_count = show_input_window()

#     if fire_count is None or obstacle_count is None:
#         print("Программа завершена пользователем.")
#         return

#     test_env = FireEnv(fire_count = fire_count, obstacle_count = obstacle_count, render_mode = "human")
#     test_env.reset()
#     total_reward = 0
#     iteration_count = 0
#     max_steps = 1000
#     rewards = []

#     for step in range(max_steps):
#         # Если огнетушителей нет, возвращаемся на базу
#         if test_env.extinguisher_count == 0:
#             action = test_env.get_action_towards_base()
#             print(f"Возвращаемся на базу за огнетушителем: Огнетушителей = {test_env.extinguisher_count}")
#         # Если заряд меньше 30 и не на базе, возвращаемся
#         elif test_env.battery_level < 30 and test_env.position != test_env.base:
#             action = test_env.get_action_towards_base()
#             print(f"Возвращаемся на базу: Батарея = {test_env.battery_level}")
#         # Если на очаге и есть огнетушитель, тушим
#         elif test_env.position in test_env.fires and test_env.extinguisher_count > 0:
#             action = 4
#         # Иначе случайное движение
#         else:
#             action = random.randint(0, 3)

#         obs, reward, done, _, info = test_env.step(action)
#         total_reward += reward
#         iteration_count += 1
#         rewards.append(reward)

#         log_message = [iteration_count, test_env.battery_level, test_env.extinguisher_count,
#                        len(test_env.fires), reward]
#         with open(log_file, mode = 'a', newline = '') as file:
#             writer = csv.writer(file)
#             writer.writerow(log_message)

#         print(f"Шаг {step + 1}: Награда = {reward}, Общая награда = {total_reward}, "
#               f"Очагов осталось: {len(test_env.fires)}, Батарея: {test_env.battery_level}, "
#               f"Огнетушителей: {test_env.extinguisher_count}")
#         test_env.render()

#         if done:
#             if len(test_env.fires) == 0:
#                 print(f"Тестирование завершено на шаге {step + 1}: Все очаги потушены! Общая награда:"
#                       f"{total_reward}")
#             elif test_env.battery_level <= 0:
#                 print(f"Тестирование завершено на шаге {step + 1}: Батарея разрядилась! Общая награда:"
#                       f"{total_reward}")
#             else:
#                 print(f"Тестирование завершено на шаге {step + 1} по неизвестной причине. Общая награда:"
#                       f"{total_reward}")
#             break

#     plt.plot(rewards)
#     plt.title("Награды за каждый шаг тестирования")
#     plt.xlabel("Шаг")
#     plt.ylabel("Награда")
#     plt.show()

#     show_summary_window(fire_count, obstacle_count, iteration_count, total_reward)
#     test_env.close()


#     # # обучение модели
#     # print("Начинаем обучение модели...")
#     # model = train_and_evaluate(fire_count, obstacle_count)
#     # print("Обучение завершено!")
#     #
#     # # предложение протестировать модель
#     # test_choice = show_test_prompt_window()
#     #
#     # # Создаем среду для тестирования с визуализацией
#     # test_env = FireEnv(fire_count = fire_count, obstacle_count = obstacle_count, render_mode = "human")
#     #
#     # # Тестирование среды без модели (случайные действия)
#     # obs, info = test_env.reset()
#     # total_reward = 0
#     # iteration_count = 0
#     # max_steps = 100
#     # rewards = []
#     #
#     # for step in range(max_steps):
#     #     # Закомментируем предсказание модели
#     #     # action, _states = model.predict(obs, deterministic = True)
#     #     # Используем случайное действие вместо модели
#     #     action = random.randint(0, 4)  # 0: up, 1: down, 2: left, 3: right, 4: extinguish
#     #
#     #     obs, reward, done, _, info = test_env.step(action)
#     #     total_reward += reward
#     #     iteration_count += 1
#     #     rewards.append(reward)
#     #
#     #     # Логирование
#     #     log_message = [iteration_count, test_env.battery_level, test_env.extinguisher_count, len(test_env.fires), reward]
#     #     with open(log_file, mode = 'a', newline = '') as file:
#     #         writer = csv.writer(file)
#     #         writer.writerow(log_message)
#     #
#     #     # Дополнительное логирование прогресса
#     #     log_training_progress(step, reward, total_reward)
#     #
#     #     # Вывод информации
#     #     print(f"Шаг {step + 1}: Награда = {reward}, Общая награда = {total_reward}")
#     #
#     #     # Отрисовка
#     #     test_env.render()
#     #
#     #     if done:
#     #         print(f"Тестирование завершено на шаге {step + 1}. Общая награда: {total_reward}")
#     #         break
#     #
#     # # Визуализация вознаграждений после тестирования
#     # plt.plot(rewards)
#     # plt.title("Награды за каждый шаг тестирования")
#     # plt.xlabel("Шаг")
#     # plt.ylabel("Награда")
#     # plt.show()
#     #
#     # # Показать итоговое окно
#     # show_summary_window(fire_count, obstacle_count, iteration_count, total_reward)
#     #
#     # # Закрытие
#     # test_env.close()


# def train_and_evaluate(fire_count, obstacle_count):
#     def make_env():
#         env = FireEnv(fire_count = fire_count, obstacle_count = obstacle_count, render_mode = None)
#         return env

#     vec_env = make_vec_env(lambda: make_env(), n_envs = 1)

#     log_dir = "./logs/ppo_tensorboard/"
#     os.makedirs(log_dir, exist_ok = True)

#     model = PPO("MlpPolicy", vec_env, verbose = 1, learning_rate = 0.0003,
#                 n_steps = 1024, batch_size = 64, n_epochs = 5, gamma = 0.99, tensorboard_log = log_dir)

#     eval_callback = EvalCallback(vec_env, best_model_save_path = ".data/best_model/",
#                                  log_path = log_dir, eval_freq = 500, deterministic = True, render = False)

#     model.learn(total_timesteps = 2000, callback = eval_callback)

#     mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes = 10, deterministic = True)
#     print(f"Средняя награда после обучения: {mean_reward} +/- {std_reward}")

#     model.save("ppo_fire_model")

#     return model

# def log_training_progress(step, reward, total_reward, log_file = "./logs/training_progress.json"):
#     data = {
#         "step": step,
#         "reward": reward,
#         "total_reward": total_reward,
#         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     }
#     with open(log_file, 'a') as f:
#         json.dump(data, f)
#         f.write('\n')









# # import gymnasium as gym
# # from stable_baselines3 import PPO
# # from stable_baselines3.common.env_checker import check_env
# # from stable_baselines3.common.vec_env import DummyVecEnv
# # from envs.fire_env import FireEnv  # Импортируем вашу среду

# # # Параметры обучения
# # TOTAL_TIMESTEPS = 100000  # Общее количество шагов для обучения
# # LEARNING_RATE = 0.0003    # Скорость обучения
# # N_STEPS = 2048            # Количество шагов для обновления политики

# # def run():
# #     # Создаём среду
# #     env = FireEnv(fire_count=5, obstacle_count=5, render_mode=None)
    
# #     # Проверяем корректность среды
# #     check_env(env)
    
# #     # Оборачиваем среду в векторизованный формат (необходим для stable-baselines3)
# #     env = DummyVecEnv([lambda: env])
    
# #     # Создаём модель PPO
# #     model = PPO(
# #         "MlpPolicy",           # Используем многослойный перцептрон (MLP) для политики
# #         env,                   # Передаём среду
# #         learning_rate=LEARNING_RATE,
# #         n_steps=N_STEPS,
# #         batch_size=64,
# #         n_epochs=10,
# #         gamma=0.99,            # Коэффициент дисконтирования
# #         gae_lambda=0.95,       # Коэффициент GAE (Generalized Advantage Estimation)
# #         verbose=1              # Выводим информацию о процессе обучения
# #     )
    
# #     # Обучаем модель
# #     print("Начинаем обучение PPO...")
# #     model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
# #     # Сохраняем обученную модель
# #     model.save("ppo_fire_agent")
# #     print("Модель сохранена как 'ppo_fire_agent.zip'")
    
# #     return model

# # def test_ppo(model):
# #     # Создаём среду для тестирования с рендерингом
# #     env = FireEnv(fire_count=5, obstacle_count=5, render_mode="human")
# #     obs, _ = env.reset()
# #     done = False
# #     total_reward = 0
    
# #     print("Тестируем обученного агента...")
# #     while not done:
# #         # Предсказываем действие на основе текущего состояния
# #         action, _states = model.predict(obs, deterministic=True)
# #         obs, reward, done, _, _ = env.step(action)
# #         total_reward += reward
# #         env.render()
    
# #     env.close()
# #     print(f"Тест завершён. Общая награда: {total_reward}")

# # if __name__ == "__main__":
# #     # Обучаем модель
# #     trained_model = run()
    
# #     # Тестируем обученную модель
# #     test_ppo(trained_model)