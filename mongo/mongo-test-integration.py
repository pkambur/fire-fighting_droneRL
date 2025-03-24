import pymongo
from datetime import datetime
import uuid

from envs.FireEnv import FireEnv
from envs.FireEnv2 import FireEnv2
from models.model_config import test_episodes
from render.user_interface import quit_pygame
from utils.logger import setup_logger

logger = setup_logger()

def test_model_with_mongo(scenario, model, fire_count, obstacle_count, 
                         experiment_id=None, render=True, 
                         mongodb_uri="mongodb://mongo:27017/", 
                         db_name="rl_logs"):
    """
    Тестирует модель и записывает результаты в MongoDB.
    
    Args:
        scenario: Номер сценария (1 или 2)
        model: Модель для тестирования
        fire_count: Количество очагов пожара
        obstacle_count: Количество препятствий
        experiment_id: ID эксперимента для привязки результатов тестирования
        render: Включение визуализации
        mongodb_uri: URI для подключения к MongoDB
        db_name: Имя базы данных
        
    Returns:
        dict: Словарь с метриками тестирования
    """
    # Подключение к MongoDB
    client = pymongo.MongoClient(mongodb_uri)
    db = client[db_name]
    test_results = db["test_results"]
    test_episodes_coll = db["test_episodes"]
    
    # Создаем ID для тестирования
    test_id = str(uuid.uuid4())
    
    # Создаем запись о тестировании
    test_data = {
        "test_id": test_id,
        "experiment_id": experiment_id,
        "scenario": scenario,
        "fire_count": fire_count,
        "obstacle_count": obstacle_count,
        "start_time": datetime.now(),
        "num_episodes": test_episodes,
        "status": "running"
    }
    test_results.insert_one(test_data)
    
    # Создаем окружение для тестирования
    if scenario == 1:
        test_env = FireEnv(fire_count=fire_count, obstacle_count=obstacle_count, 
                          render_mode="human" if render else None)
    elif scenario == 2:
        test_env = FireEnv2(fire_count=fire_count, obstacle_count=obstacle_count, 
                           render_mode="human" if render else None)
    else:
        raise ValueError(f"Неизвестный сценарий: {scenario}. Допустимые значения: 1 или 2.")
    
    # Метрики для отслеживания
    metrics = {
        "successful_attempts": 0,
        "total_reward": 0,
        "total_steps": 0,
        "collision_count": 0,
        "achieved_goals": 0,
        "wind_impacts": 0,
        "total_fires_extinguished": 0
    }
    
    logger.info("Starting model testing with MongoDB logging...")
    
    # Проходим тестовые эпизоды
    for episode in range(1, test_episodes + 1):
        obs, _ = test_env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        episode_collisions = 0
        episode_wind_impacts = 0
        fires_extinguished = 0
        initial_fires = len(test_env.fires)
        
        # Эпизод
        while not (done or truncated):
            actions, _ = model.predict(obs)
            obs, reward, done, truncated, info = test_env.step(actions)
            
            total_reward += reward
            steps += 1
            
            # Отслеживаем события
            if info.get("The goal has been achieved", False):
                metrics["achieved_goals"] += 1
                fires_extinguished += 1
            
            if info.get("Collision", False):
                episode_collisions += 1
            
            # Логируем события в MongoDB по ходу эпизода (опционально)
            if steps % 10 == 0:  # Каждые 10 шагов
                step_data = {
                    "test_id": test_id,
                    "episode": episode,
                    "step": steps,
                    "reward": float(reward),
                    "fires_left": len(test_env.fires),
                    "timestamp": datetime.now(),
                    "actions": actions.tolist()
                }
                test_episodes_coll.insert_one(step_data)
            
            if render:
                test_env.render()
        
        # Обновляем метрики
        metrics["total_reward"] += total_reward
        metrics["total_steps"] += steps
        metrics["collision_count"] += episode_collisions
        metrics["wind_impacts"] += episode_wind_impacts
        fires_extinguished = initial_fires - len(test_env.fires)
        metrics["total_fires_extinguished"] += fires_extinguished
        
        if len(test_env.fires) == 0:
            metrics["successful_attempts"] += 1
        
        # Записываем информацию об эпизоде в MongoDB
        episode_data = {
            "test_id": test_id,
            "episode": episode,
            "total_reward": float(total_reward),
            "steps": steps,
            "success": len(test_env.fires) == 0,
            "fires_extinguished": fires_extinguished,
            "initial_fires": initial_fires,
            "collisions": episode_collisions,
            "wind_impacts": episode_wind_impacts,
            "timestamp": datetime.now()
        }
        test_episodes_coll.insert_one(episode_data)
        
        logger.info(f"Episode {episode} completed: reward={total_reward:.2f}, fires_extinguished={fires_extinguished}")
    
    # Рассчитываем итоговые метрики
    success_rate = (metrics["successful_attempts"] / test_episodes) * 100
    avg_reward = metrics["total_reward"] / test_episodes
    step_efficiency = (metrics["total_steps"] / metrics["achieved_goals"]) if metrics["achieved_goals"] else float('inf')
    collision_rate = metrics["collision_count"] / test_episodes
    
    # Обновляем запись о тестировании в MongoDB
    final_metrics = {
        "status": "completed",
        "end_time": datetime.now(),
        "success_rate": success_rate,
        "avg_reward": float(avg_reward),
        "step_efficiency": float(step_efficiency),
        "collision_rate": float(collision_rate),
        "total_steps": metrics["total_steps"],
        "total_fires_extinguished": metrics["total_fires_extinguished"]
    }
    
    test_results.update_one(
        {"test_id": test_id},
        {"$set": final_metrics}
    )
    
    # Выводим результаты
    print(f"""
    Test results (ID: {test_id}):
    Success Rate: {success_rate:.1f} %
    Average Reward: {avg_reward:.2f}
    Step Efficiency: {step_efficiency:.1f} steps/goal
    Collision Rate: {collision_rate:.1f} per episode
    """)
    
    # Закрываем окружение
    if hasattr(test_env, 'close'):
        test_env.close()
    quit_pygame()
    
    return {
        "test_id": test_id,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "step_efficiency": step_efficiency,
        "collision_rate": collision_rate
    }
