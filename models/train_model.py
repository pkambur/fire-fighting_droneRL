import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from envs.FireEnv import FireEnv
from envs.FireEnv2 import FireEnv2
from models.model_config import PPO_DEFAULT_CONFIG, PPO_FINISH_EXTINGUISHING_CONFIG
from utils.logging_files import tensorboard_log_dir, model_name, best_model_path


def train_and_evaluate(scenario, fire_count, obstacle_count):
    cfg = PPO_DEFAULT_CONFIG
    # cfg = PPO_FINISH_EXTINGUISHING_CONFIG

    def make_env():
        if scenario == 1:
            env = FireEnv(fire_count=fire_count, obstacle_count=obstacle_count, render_mode=None)
        elif scenario == 2:
            env = FireEnv2(fire_count=fire_count, obstacle_count=obstacle_count, render_mode=None)
        else:
            raise ValueError(f"Неизвестный сценарий: {scenario}.\n"
                             f" Допустимые значения: 1 или 2.")
        return env

    vec_env = make_vec_env(make_env, n_envs=cfg["n_envs"])
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    model = PPO(
        policy=cfg["policy"],
        env=vec_env,
        verbose=cfg["verbose"],
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        clip_range_vf=cfg["clip_range_vf"],
        ent_coef=cfg["ent_coef"],
        tensorboard_log=tensorboard_log_dir
    )
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=best_model_path,
        log_path=tensorboard_log_dir,
        eval_freq=1000,
        render=False
    )
    model.learn(total_timesteps=cfg["total_timesteps"], progress_bar=True)
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    print(f"Среднее вознаграждение после тренировки: {mean_reward} +/- {std_reward}")
    model.save(model_name + str(scenario))
    return model
