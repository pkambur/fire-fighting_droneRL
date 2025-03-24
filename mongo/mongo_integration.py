import pymongo
from datetime import datetime
import uuid
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class MongoDBLoggerCallback(BaseCallback):
    """
    Callback для логирования метрик обучения модели RL в MongoDB.
    Сохраняет информацию об эксперименте, эпизодах и метриках обучения.
    """
    
    def __init__(self, 
                 mongodb_uri="mongodb://mongo:27017/", 
                 db_name="rl_logs",
                 experiment_name=None,
                 scenario=1,
                 fire_count=5,
                 obstacle_count=10,
                 description=None,
                 eval_freq=1000,
                 n_eval_episodes=1,
                 verbose=1):
        super(MongoDBLoggerCallback, self).__init__(verbose)
        # MongoDB соединение
        self.client = pymongo.MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.experiments = self.db["experiments"]
        self.episodes = self.db["episodes"]
        self.steps = self.db["training_steps"]
        
        # Метаданные эксперимента
        self.experiment_id = str(uuid.uuid4())
        self.experiment_name = experiment_name or f"FireFighter_Exp_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.scenario = scenario
        self.fire_count = fire_count
        self.obstacle_count = obstacle_count
        self.description = description or f"Обучение модели PPO для сценария {scenario} с {fire_count} пожарами и {obstacle_count} препятствиями"
        
        # Настройки оценки
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        # Счетчики
        self.last_time = datetime.now()
        self.episode_count = 0
        self.total_fires_extinguished = 0
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _init_callback(self):
        """Вызывается в начале обучения."""
        # Получаем информацию о модели и окружении
        model_info = self._get_model_info()
        env_info = self._get_env_info()
        
        # Сохраняем информацию об эксперименте
        experiment_data = {
            "experiment_id": self.experiment_id,
            "name": self.experiment_name,
            "description": self.description,
            "start_time": datetime.now(),
            "status": "running",
            "scenario": self.scenario,
            "fire_count": self.fire_count,
            "obstacle_count": self.obstacle_count,
            "model": model_info,
            "environment": env_info
        }
        
        self.experiments.insert_one(experiment_data)
        self.last_time = datetime.now()
        print(f"MongoDB Logger: Начало эксперимента {self.experiment_id}")
    
    def _on_step(self):
        """Вызывается на каждом шаге обучения."""
        # Периодически сохраняем информацию о процессе обучения
        if self.num_timesteps % self.eval_freq == 0:
            # Оцениваем текущую модель
            env = self.training_env.envs[0]
            
            try:
                # Получаем состояние окружения
                fires_left = len(env.fires) if hasattr(env, 'fires') else None
                iteration_count = env.iteration_count if hasattr(env, 'iteration_count') else None
                
                # Получаем производительность модели
                mean_reward = 0
                episode_length = 0
                
                try:
                    mean_reward, _ = evaluate_policy(self.model, self.training_env, n_eval_episodes=self.n_eval_episodes)
                    episode_length = iteration_count
                except Exception as e:
                    print(f"Ошибка при оценке модели: {e}")
                    
                # Получаем дополнительные метрики
                elapsed_time = (datetime.now() - self.last_time).total_seconds()
                steps_per_second = self.eval_freq / elapsed_time if elapsed_time > 0 else 0
                
                # Записываем метрики в MongoDB
                step_data = {
                    "experiment_id": self.experiment_id,
                    "timestep": self.num_timesteps,
                    "datetime": datetime.now(),
                    "mean_reward": float(mean_reward),
                    "episode_length": episode_length,
                    "fires_left": fires_left,
                    "steps_per_second": steps_per_second
                }
                
                self.steps.insert_one(step_data)
                self.last_time = datetime.now()
                
                if self.verbose > 0:
                    print(f"MongoDB Logger: Шаг {self.num_timesteps}, награда: {mean_reward:.2f}, "
                          f"огней осталось: {fires_left}")
            
            except Exception as e:
                print(f"MongoDB Logger: Ошибка при логировании шага: {str(e)}")
        
        return True
    
    def on_episode_end(self, episode_info=None):
        """
        Записывает информацию по окончании эпизода.
        Должна вызываться явно в цикле обучения.
        """
        try:
            if episode_info:
                self.episode_count += 1
                reward = episode_info.get('r', 0)
                length = episode_info.get('l', 0)
                fires_extinguished = episode_info.get('fires_extinguished', 0)
                collisions = episode_info.get('collisions', 0)
                wind_impacts = episode_info.get('wind_impacts', 0)
                success = episode_info.get('success', False)
                
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                self.total_fires_extinguished += fires_extinguished
                
                # Записываем информацию об эпизоде в MongoDB
                episode_data = {
                    "experiment_id": self.experiment_id,
                    "episode": self.episode_count,
                    "timestep": self.num_timesteps,
                    "datetime": datetime.now(),
                    "reward": reward,
                    "length": length,
                    "fires_extinguished": fires_extinguished,
                    "collisions": collisions,
                    "wind_impacts": wind_impacts,
                    "success": success
                }
                
                self.episodes.insert_one(episode_data)
                
                if self.verbose > 0 and self.episode_count % 10 == 0:
                    print(f"MongoDB Logger: Эпизод {self.episode_count}, награда: {reward:.2f}, "
                          f"потушено огней: {fires_extinguished}")
        
        except Exception as e:
            print(f"MongoDB Logger: Ошибка при логировании эпизода: {str(e)}")
    
    def on_training_end(self):
        """Вызывается в конце обучения."""
        try:
            # Вычисляем агрегированные метрики
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
            
            # Обновляем информацию об эксперименте
            self.experiments.update_one(
                {"experiment_id": self.experiment_id},
                {"$set": {
                    "end_time": datetime.now(),
                    "status": "completed",
                    "total_timesteps": self.num_timesteps,
                    "total_episodes": self.episode_count,
                    "avg_reward": float(avg_reward),
                    "avg_episode_length": float(avg_length),
                    "total_fires_extinguished": self.total_fires_extinguished
                }}
            )
            
            print(f"MongoDB Logger: Завершение эксперимента {self.experiment_id}")
            print(f"Средняя награда: {avg_reward:.2f}, среднее количество шагов: {avg_length:.1f}")
        
        except Exception as e:
            print(f"MongoDB Logger: Ошибка при завершении логирования: {str(e)}")
    
    def _get_model_info(self):
        """Извлекает информацию о модели."""
        try:
            model_info = {
                "type": self.model.__class__.__name__,
                "policy": self.model.policy.__class__.__name__
            }
            
            # Добавляем гиперпараметры
            for param in ["learning_rate", "n_steps", "batch_size", "n_epochs", 
                          "gamma", "gae_lambda", "clip_range", "ent_coef"]:
                if hasattr(self.model, param):
                    value = getattr(self.model, param)
                    if callable(value):
                        value = str(value)
                    elif isinstance(value, (np.ndarray, np.number)):
                        value = value.item()
                    model_info[param] = value
            
            return model_info
        except Exception as e:
            print(f"MongoDB Logger: Не удалось получить информацию о модели: {str(e)}")
            return {"error": str(e)}
    
    def _get_env_info(self):
        """Извлекает информацию об окружении."""
        try:
            env = self.training_env.envs[0]
            env_info = {
                "type": env.__class__.__name__,
                "observation_space": str(env.observation_space),
                "action_space": str(env.action_space)
            }
            
            # Добавляем специфичные для FireEnv параметры
            for param in ["grid_size", "fire_count", "obstacle_count", "max_steps"]:
                if hasattr(env, param):
                    env_info[param] = getattr(env, param)
            
            return env_info
        except Exception as e:
            print(f"MongoDB Logger: Не удалось получить информацию об окружении: {str(e)}")
            return {"error": str(e)}
