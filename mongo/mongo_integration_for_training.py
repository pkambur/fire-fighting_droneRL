import models.model_config as config

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList

from envs.FireEnv import FireEnv
from envs.FireEnv2 import FireEnv2
from mongo.mongo_integration import MongoDBLoggerCallback
from utils.logging_files import tensorboard_log_dir, model_name


def train_and_evaluate_with_mongo(scenario, fire_count, obstacle_count, experiment_name=None):
    """
    Обучает модель с логированием в MongoDB.
    """
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
            raise ValueError(f"Неизвестный сценарий: {scenario}. Допустимые значения: 1 или 2.")
        return env

    train_env = make_env()
    vec_env = make_vec_env(lambda: train_env, n_envs = cfg["n_envs"])

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

    mongo_logger = MongoDBLoggerCallback(
        mongodb_uri="mongodb://mongo:27017/",
        db_name="rl_logs",
        experiment_name=experiment_name or f"FireFighter_Scenario{scenario}",
        scenario=scenario,
        fire_count=fire_count,
        obstacle_count=obstacle_count,
        eval_freq=1000,
        n_eval_episodes=1,
        verbose=1
    )

    # Создаем список коллбэков
    callbacks = [mongo_logger]

    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=CallbackList(callbacks),
        progress_bar=True
    )

    model_path = model_name + str(scenario)
    model.save(model_path)

    # Записываем в MongoDB финальную информацию
    mongo_logger.on_training_end()
    return model, mongo_logger.experiment_id
