import csv
import os

from envs.FireEnv import FireEnv
from envs.FireEnv2 import FireEnv2
from models.model_config import test_episodes
from render.user_interface import quit_pygame
from utils.logging_files import log_csv
from utils.logger import setup_logger

summary_shown = False
logger = setup_logger()


def test_model(scenario, model, fire_count, obstacle_count, render=True):
    global summary_shown

    if os.path.exists(log_csv):
        os.remove(log_csv)
    with open(log_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Step",
                         "Fires Left", "Reward", "Action1", "Action2", "Action3"])

    metrics = {
        "successful_attempts": 0,
        "total_reward": 0,
        "total_steps": 0,
        "collision_count": 0,
        "achieved_goals": 0
    }
    if scenario == 1:
        test_env = FireEnv(fire_count=fire_count, obstacle_count=obstacle_count, render_mode="human")
    elif scenario == 2:
        test_env = FireEnv2(fire_count=fire_count, obstacle_count=obstacle_count, render_mode="human")

    logger.info("Starting model testing...")

    for episode in range(1, test_episodes + 1):
        obs, _ = test_env.reset()
        total_reward = 0

        while True:
            actions, _ = model.predict(obs)  # Единое действие для всех агентов
            obs, reward, terminated, truncated, info = test_env.step(actions)
            total_reward += reward

            with open(log_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, test_env.iteration_count] +
                                [len(test_env.fires), round(reward, 2)] + list(actions))

            if render:
                test_env.render()

            metrics["total_reward"] += reward
            metrics["total_steps"] += 1

            if info.get("The goal has been achieved", False):
                metrics["achieved_goals"] += 1

            if info.get("Collision", False):
                metrics["collision_count"] += 1

            if terminated or truncated:
                if len(test_env.fires) == 0:
                    print(f"Testing completed at step {test_env.iteration_count}: "
                          f"All fires extinguished! Total Reward: {round(total_reward, 2)}")
                    metrics["successful_attempts"] += 1
                elif test_env.iteration_count + 1 >= test_env.max_steps:
                    print(f"Testing completed at step {test_env.iteration_count}: "
                          f"Step limit reached. Total Reward: {round(total_reward, 2)}")
                else:
                    print(f"Testing completed at step {test_env.iteration_count} "
                          f"for unknown reason. Total Reward: {round(total_reward, 2)}")
                if not summary_shown:
                    test_env.close()
                    summary_shown = True
                break

    success_rate = (metrics["successful_attempts"] / test_episodes) * 100
    avg_reward = metrics["total_reward"] / test_episodes
    step_efficiency = (metrics["total_steps"] / metrics["achieved_goals"]) if metrics["achieved_goals"] else 0
    collision_rate = metrics["collision_count"] / test_episodes
    print(f"""
    Evaluation results:
    Success Rate: {success_rate:.1f} %
    Average Reward: {avg_reward:.2f}
    Step Efficiency: {step_efficiency:.1f} steps/goal
    Collision Rate: {collision_rate:.1f} in episode
    """)

    if not summary_shown:
        test_env.close()
        summary_shown = True
    quit_pygame()
