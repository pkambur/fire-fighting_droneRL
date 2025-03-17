import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback


from envs.FireEnv import FireEnv
from utils.logging_files import tensorboard_log_dir, model_name, best_model_path


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
        best_model_save_path=best_model_path,
        log_path=tensorboard_log_dir,
        eval_freq=1000,
        render=False
    )
    model.learn(total_timesteps=50000)
    mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
    print(f"Среднее вознаграждение после тренировки: {mean_reward} +/- {std_reward}")
    model.save(model_name)
    return model
