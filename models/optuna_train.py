import os
import optuna

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from envs.FireEnv import FireEnv
from envs.FireEnv2 import FireEnv2
from models.model_config import test_episodes
from utils.logging_files import tensorboard_log_dir, best_model_path


def train_and_evaluate(scenario, fire_count,
                       obstacle_count, learning_rate, n_steps,
                       batch_size, n_epochs, gamma, gae_lambda,
                       clip_range, clip_range_vf, ent_coef):
    """
    Обучает модель с заданными гиперпараметрами и возвращает среднюю длину эпизода.
    """

    def make_env():
        if scenario == 1:
            env = FireEnv(fire_count=fire_count, obstacle_count=obstacle_count, render_mode=None)
        elif scenario == 2:
            env = FireEnv2(fire_count=fire_count, obstacle_count=obstacle_count, render_mode=None)
        else:
            raise ValueError(f"Неизвестный сценарий: {scenario}. Допустимые значения: 1 или 2.")
        return env

    vec_env = make_vec_env(make_env, n_envs=1)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=ent_coef,
        tensorboard_log=tensorboard_log_dir
    )

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=best_model_path,
        log_path=tensorboard_log_dir,
        eval_freq=1000,
        render=False
    )

    model.learn(total_timesteps=5000, progress_bar=True, callback=eval_callback)

    # Оценка длины эпизода
    episode_lengths = []
    obs = vec_env.reset()
    for _ in range(test_episodes):
        done = False
        episode_length = 0
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = vec_env.step(action)
            episode_length += 1
        episode_lengths.append(episode_length)
        obs = vec_env.reset()

    mean_episode_length = sum(episode_lengths) / len(episode_lengths)
    return mean_episode_length


def objective(trial, scenario, fire_count, obstacle_count):
    """
    Целевая функция для Optuna. Подбирает гиперпараметры и возвращает среднюю длину эпизода.
    """
    print(f"Trial {trial.number}: scenario={scenario}, fire_count={fire_count}, obstacle_count={obstacle_count}")

    # Подбор гиперпараметров
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3])
    n_steps = trial.suggest_categorical("n_steps", [2048, 4096, 8192])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    n_epochs = trial.suggest_int("n_epochs", 2, 5)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.9999)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    clip_range_vf = trial.suggest_float("clip_range_vf", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)

    # Обучаем модель с текущими гиперпараметрами
    mean_episode_length = train_and_evaluate(
        scenario, fire_count, obstacle_count,
        learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, ent_coef
    )

    # Возвращаем среднюю длину эпизода для минимизации
    return mean_episode_length


def optimize_hyperparameters(scenario, fire_count, obstacle_count):
    """
    Оптимизирует гиперпараметры с помощью Optuna для минимизации длины эпизода.
    """
    # Создаем исследование Optuna
    study = optuna.create_study(direction="minimize")  # Минимизируем длину эпизода

    # Оптимизируем гиперпараметры
    study.optimize(lambda trial: objective(trial, scenario, fire_count, obstacle_count),
                   n_trials=20, show_progress_bar=False)

    # Выводим лучшие гиперпараметры
    print("Лучшие гиперпараметры: ", study.best_params)
    print("Лучшая средняя длина эпизода: ", study.best_value)

    # Обучаем модель с лучшими гиперпараметрами
    best_params = study.best_params
    mean_episode_length = train_and_evaluate(
        scenario, fire_count, obstacle_count,
        best_params["learning_rate"], best_params["n_steps"], best_params["batch_size"],
        best_params["n_epochs"], best_params["gamma"], best_params["gae_lambda"],
        best_params["clip_range"], best_params["clip_range_vf"], best_params["ent_coef"]
    )

    print(f"Средняя длина эпизода с лучшими гиперпараметрами: {mean_episode_length}")
