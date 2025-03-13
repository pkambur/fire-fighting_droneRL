import os
import csv
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from models.test_model import test_model
from render.user_interface import show_input_window, show_test_prompt_window, show_start_window
from envs.fire_env import FireEnv
from utils.logging_files import log_csv, tensorboard_log_dir


def run():
    if show_start_window():

        if os.path.exists(log_csv):
            os.remove(log_csv)
        with open(log_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Step", "Battery1", "Battery2", "Battery3",
                             "Extinguishers1", "Extinguishers2", "Extinguishers3",
                             "Fires Left", "Reward", "Action1", "Action2", "Action3"])

        fire_count, obstacle_count = show_input_window()
        logging.info("Starting model training...")
        model = train_and_evaluate(fire_count, obstacle_count)
        logging.info("Training completed!")

        if show_test_prompt_window():
            test_model(model, fire_count, obstacle_count)


def train_and_evaluate(fire_count, obstacle_count):
    def make_env():
        return FireEnv(fire_count=fire_count, obstacle_count=obstacle_count, render_mode=None)

    vec_env = make_vec_env(make_env, n_envs=1)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0001,
        n_steps=4096,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,
        ent_coef=0.01,
        tensorboard_log=tensorboard_log_dir
    )
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path="./data/best_model/",
        log_path=tensorboard_log_dir,
        eval_freq=1000,
        render=False
    )
    model.learn(total_timesteps=5000, progress_bar=True)
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    print(f"Среднее вознаграждение после тренировки: {mean_reward} +/- {std_reward}")
    model.save("ppo_fire_model")
    return model
