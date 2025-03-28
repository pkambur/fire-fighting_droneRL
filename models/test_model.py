import csv
import os
from datetime import datetime

from envs.FireEnv import FireEnv
from envs.FireEnv2 import FireEnv2
import envs as e
from models.model_config import test_episodes
from render.user_interface import quit_pygame
from utils.logging_files import log_csv
from utils.logger import setup_logger

summary_shown = False
logger = setup_logger()

def test_model(scenario, model, fire_count, obstacle_count, render=True):
    global summary_shown

    # Создаем уникальный файл для логов наград с временной меткой
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rewards_log_file = os.path.join(logs_dir, f"test_rewards_{timestamp}.csv")

    # Удаляем старый log_csv, если он существует (оставляем для совместимости)
    if os.path.exists(log_csv):
        os.remove(log_csv)

    # Инициализируем основной лог (log_csv)
    with open(log_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Step", "Fires Left", "Reward", "Action1", "Action2", "Action3"])

    # Инициализируем лог наград и штрафов
    with open(rewards_log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Step", "Agent", "Total Reward", "Fire Reward", "Step Penalty",
                         "Crash Penalty", "Out of Bounds Penalty", "Obstacle Penalty", "Wind Penalty",
                         "Wind Avoid Bonus", "Stagnation Penalty", "Close Penalty", "Final Reward"])

    metrics = {
        "successful_attempts": 0,
        "total_reward": 0,
        "total_steps": 0,
        "collision_count": 0,
        "achieved_goals": 0
    }

    # Выбор среды в зависимости от сценария
    if scenario == 1:
        test_env = FireEnv(fire_count=fire_count, obstacle_count=obstacle_count, render_mode="human" if render else None)
    elif scenario == 2:
        test_env = FireEnv2(fire_count=fire_count, obstacle_count=obstacle_count, render_mode="human" if render else None)

    logger.info("Starting model testing...")

    for episode in range(1, test_episodes + 1):
        obs, _ = test_env.reset()
        total_reward = 0
        if render and scenario == 2:
            test_env.render_airplane()

        while True:
            actions, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = test_env.step(actions)
            total_reward += reward

            # Запись в основной лог (log_csv)
            with open(log_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode, test_env.iteration_count] +
                                [len(test_env.fires), round(reward, 2)] + list(actions))

            # Детализированная запись наград и штрафов в rewards_log_file
            with open(rewards_log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                for agent_idx in range(test_env.num_agents):
                    # Собираем награды и штрафы из логики среды (примерно)
                    fire_reward = e.FIRE_REWARD if info.get("The goal has been achieved", False) else 0
                    step_penalty = e.STEP_PENALTY
                    crash_penalty = e.CRASH_PENALTY if info.get("Collision", False) and test_env.positions[agent_idx] in [test_env.positions[i] for i in range(test_env.num_agents) if i != agent_idx] else 0
                    out_of_bounds_penalty = e.OUT_OF_BOUNDS_PENALTY if not (0 <= test_env.positions[agent_idx][0] < test_env.grid_size and 0 <= test_env.positions[agent_idx][1] < test_env.grid_size) else 0
                    obstacle_penalty = e.OBSTACLE_PENALTY if test_env.positions[agent_idx] in test_env.obstacles else 0
                    wind_penalty = e.WIND_PENALTY if test_env.wind.active and test_env.positions[agent_idx] in test_env.wind.cells else 0
                    wind_avoid_bonus = e.WIND_AVOID_BONUS if test_env.wind.active and test_env.positions[agent_idx] not in test_env.wind.cells else 0
                    stagnation_penalty = e.STAGNATION_PENALTY if test_env.steps_without_progress[agent_idx] >= e.STAGNATION_THRESHOLD else 0
                    close_penalty = -0.2 if any(abs(test_env.positions[agent_idx][0] - test_env.positions[i][0]) + abs(test_env.positions[agent_idx][1] - test_env.positions[i][1]) <= 1 for i in range(test_env.num_agents) if i != agent_idx) else 0
                    final_reward = e.FINAL_REWARD if terminated or truncated else 0

                    writer.writerow([episode, test_env.iteration_count, agent_idx, round(reward, 2),
                                     fire_reward, step_penalty, crash_penalty, out_of_bounds_penalty,
                                     obstacle_penalty, wind_penalty, wind_avoid_bonus, stagnation_penalty,
                                     close_penalty, final_reward])

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

    # Вычисление метрик
    success_rate = (metrics["successful_attempts"] / test_episodes) * 100
    avg_reward = metrics["total_reward"] / test_episodes
    avg_steps = metrics["total_steps"] / test_episodes
    step_efficiency = (metrics["total_steps"] / metrics["achieved_goals"]) if metrics["achieved_goals"] else 0
    collision_rate = metrics["collision_count"] / test_episodes

    print(f"""
    Evaluation results:
    Success Rate: {success_rate:.1f} %
    Average Reward: {avg_reward:.2f}
    Average Steps: {avg_steps:.2f} in episode
    Step Efficiency: {step_efficiency:.1f} steps/goal
    Collision Rate: {collision_rate:.1f} in episode
    """)

    if not summary_shown:
        test_env.close()
        summary_shown = True
    quit_pygame()

    logger.info(f"Rewards and penalties logged to {rewards_log_file}")