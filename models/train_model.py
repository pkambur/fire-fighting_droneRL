import os
import models.model_config as config

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from envs.FireEnv import FireEnv
from envs.FireEnv2 import FireEnv2
from utils.logging_files import tensorboard_log_dir, model_name, best_model_path


def train_and_evaluate(scenario, fire_count, obstacle_count):
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    cfg = {}
    def make_env():
        nonlocal cfg
        if scenario == 1:
            env = FireEnv(fire_count = fire_count,
                          obstacle_count = obstacle_count,
                          render_mode = None)
            cfg = config.ppo_config["PPO_SCENARIO1_CONFIG"]
        elif scenario == 2:
            env = FireEnv2(fire_count = fire_count,
                           obstacle_count = obstacle_count,
                           render_mode = None)
            cfg = config.ppo_config["PPO_SCENARIO2_CONFIG"]
        else:
            raise ValueError(f"Неизвестный сценарий: {scenario}.\n"
                             f" Допустимые значения: 1 или 2.")
        return env

    train_env = make_env()
    vec_env = make_vec_env(lambda: train_env, n_envs = cfg["n_envs"])
    os.makedirs(tensorboard_log_dir, exist_ok = True)

    model = PPO(
        policy = cfg["policy"],
        env = vec_env,
        verbose = cfg["verbose"],
        learning_rate = cfg["learning_rate"],
        n_steps = cfg["n_steps"],
        batch_size = cfg["batch_size"],
        n_epochs = cfg["n_epochs"],
        gamma = cfg["gamma"],
        gae_lambda = cfg["gae_lambda"],
        clip_range = cfg["clip_range"],
        clip_range_vf = cfg["clip_range_vf"],
        ent_coef = cfg["ent_coef"],
        tensorboard_log = tensorboard_log_dir
    )

    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path = best_model_path,
        log_path = tensorboard_log_dir,
        eval_freq = 1000,
        render = False
    )

    model.learn(total_timesteps = cfg["total_timesteps"], progress_bar = True)
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes = 10)
    print(f"Среднее вознаграждение после тренировки: {mean_reward} +/- {std_reward}")
    model_save_path = os.path.join(logs_dir, model_name + str(scenario))
    model.save(model_save_path)
    print(f"Модель сохранена по пути: {model_save_path}")
    return model
