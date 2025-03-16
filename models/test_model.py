import csv
import os

from envs.FireEnv import FireEnv
from models import test_episodes
from render.user_interface import quit_pygame
from utils.logging_files import log_csv
from utils.logger import setup_logger

summary_shown = False
logger = setup_logger()


def test_model(model, fire_count, obstacle_count, render=True):
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

    test_env = FireEnv(fire_count=fire_count, obstacle_count=obstacle_count, render_mode="human")

    logger.info("Starting model testing...")

    for episode in range(1, test_episodes + 1):
        obs, _ = test_env.reset()
        total_reward = 0

        while True:
            actions, _ = model.predict(obs)  # Единое действие для всех агентов
            obs, reward, terminated, truncated, info = test_env.step(actions)
            total_reward += reward

            metrics["total_reward"] += reward
            metrics["total_steps"] += 1

            with open(log_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, test_env.iteration_count] +
                                [len(test_env.fires), reward] + list(actions))
            logger.info(f"Episode = {episode}, Step {test_env.iteration_count + 1}: "
                        f"Reward = {reward}, Total Reward = {total_reward}, "
                        f"Fires Left: {len(test_env.fires)}")

            if render:
                test_env.render()

            if info.get("The goal has been achieved", False):
                metrics["achieved_goals"] += 1

            if info.get("Collision with an obstacle", False):
                metrics["collision_count"] += 1

            if terminated or truncated:
                if len(test_env.fires) == 0:
                    print(f"Testing completed at step {test_env.iteration_count}: "
                          f"All fires extinguished! Total Reward: {total_reward}")
                    metrics["successful_attempts"] += 1
                elif test_env.iteration_count + 1 >= test_env.max_steps:
                    print(f"Testing completed at step {test_env.iteration_count}: "
                          f"Step limit reached. Total Reward: {total_reward}")
                else:
                    print(f"Testing completed at step {test_env.iteration_count} "
                          f"for unknown reason. Total Reward: {total_reward}")
                if not summary_shown:
                    test_env.close()
                    summary_shown = True
                break

    success_rate = (metrics["successful_attempts"] / test_episodes) * 100
    avg_reward = metrics["total_reward"] / test_episodes
    step_efficiency = (metrics["total_steps"] / metrics["achieved_goals"]) if metrics["achieved_goals"] else 0
    collision_rate = metrics["collision_count"] / test_episodes * 100

    print(f"""
    Evaluation results:
    Success Rate: {success_rate:.1f} %
    Average Reward: {avg_reward:.2f}
    Step Efficiency: {step_efficiency:.1f} steps/goal
    Collision Rate: {collision_rate:.1f} %
    """)

    if not summary_shown:
        test_env.close()
        summary_shown = True
    quit_pygame()
