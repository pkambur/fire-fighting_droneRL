import csv
import logging

from envs.fire_env import FireEnv
from render.user_interface import quit_pygame
from utils.logging_files import log_csv

summary_shown = False


def test_model(model, fire_count, obstacle_count):
    global summary_shown
    test_env = FireEnv(fire_count=fire_count, obstacle_count=obstacle_count, render_mode="human")

    logging.info("Starting model testing...")
    for episode in range(1, 5):
        obs, _ = test_env.reset()
        total_reward = 0
        while True:
            actions, _ = model.predict(obs)  # Единое действие для всех агентов
            obs, reward, terminated, truncated, info = test_env.step(actions)
            total_reward += reward
            with open(log_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, test_env.iteration_count] +
                                test_env.battery_levels + test_env.extinguisher_counts +
                                [len(test_env.fires), reward] + list(actions))
            logging.info(f"Episode = {episode}, Step {test_env.iteration_count + 1}: "
                         f"Reward = {reward}, Total Reward = {total_reward}, "
                         f"Fires Left: {len(test_env.fires)}")
            test_env.render()
            if terminated or truncated:
                if len(test_env.fires) == 0:
                    print(f"Testing completed at step {test_env.iteration_count}: "
                          f"All fires extinguished! Total Reward: {total_reward}")
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
    if not summary_shown:
        test_env.close()
        summary_shown = True
    quit_pygame()
