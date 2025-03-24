## Использование MongoDB для логирования процесса обучения моделей FireFighterRL

MongoDB обеспечивает гибкое решение для сохранения, анализа и сравнения экспериментов по обучению RL моделей в проекте FireFighterRL.

### Зачем нам MongoDB?

1. **Отслеживание прогресса обучения**
   - Запись метрик на уровне эпизодов (награды, количество шагов, успешность тушения пожаров)
   - Мониторинг специфичных метрик (столкновения, воздействия ветра, эффективность тушения)
   - Сохранение параметров экспериментов для последующего анализа

2. **Сравнение экспериментов**
   - Сопоставление разных алгоритмов (PPO с разными гиперпараметрами)
   - Анализ влияния параметров окружения (количество пожаров, препятствий) на обучение
   - Выявление наиболее эффективных конфигураций

3. **Анализ динамики обучения**
   - График наград по ходу обучения
   - Изменение эффективности тушения пожаров
   - Динамика уменьшения столкновений с препятствиями

### Примеры использования

#### 1. Обучение модели с логированием в MongoDB

```python
from mongodb_logger import MongoDBLoggerCallback

# Обучение модели с логированием в MongoDB
experiment_id = train_and_evaluate_with_mongo(
    scenario=1, 
    fire_count=5, 
    obstacle_count=10,
    experiment_name="PPO_FireEnv_Test1"
)

print(f"Обучение завершено, ID эксперимента: {experiment_id}")
```

#### 2. Тестирование модели с записью результатов

```python
from stable_baselines3 import PPO

# Загрузка обученной модели
model = PPO.load("ppo_sceanrio_1")

# Тестирование с логированием в MongoDB
test_metrics = test_model_with_mongo(
    scenario=1,
    model=model,
    fire_count=5,
    obstacle_count=10,
    experiment_id="61fa3e2c-85b7-4f12-a3a2-dbd81ec3bb2a"  # ID эксперимента обучения
)

print(f"Тестирование завершено, результаты: {test_metrics}")
```

#### 3. Анализ результатов из Jupyter Notebook

```python
import pymongo
import pandas as pd
import matplotlib.pyplot as plt

# Подключение к MongoDB
client = pymongo.MongoClient("mongodb://mongo:27017/")
db = client["rl_logs"]

# Получение списка экспериментов
experiments = list(db.experiments.find({}, {"_id": 0, "experiment_id": 1, "name": 1, "scenario": 1, "fire_count": 1}))
for exp in experiments:
    print(f"ID: {exp['experiment_id']}, Имя: {exp['name']}, Сценарий: {exp['scenario']}, Пожары: {exp['fire_count']}")

# Выбор конкретного эксперимента для анализа
exp_id = "61fa3e2c-85b7-4f12-a3a2-dbd81ec3bb2a"  # Замените на нужный ID

# Получение шагов обучения
steps_data = list(db.training_steps.find({"experiment_id": exp_id}))
steps_df = pd.DataFrame(steps_data)

# Визуализация прогресса обучения
plt.figure(figsize=(12, 6))
plt.plot(steps_df["timestep"], steps_df["mean_reward"], marker='o')
plt.title("Прогресс обучения")
plt.xlabel("Шаги")
plt.ylabel("Средняя награда")
plt.grid(True)
plt.show()
```

### Структура данных в MongoDB

#### 1. Коллекция `experiments`
Хранит информацию о каждом запуске обучения:
- `experiment_id` - уникальный идентификатор
- `name` - название эксперимента
- `scenario` - используемый сценарий (1 или 2)
- `fire_count` - количество очагов пожара
- `obstacle_count` - количество препятствий
- `start_time` - время начала обучения
- `end_time` - время завершения обучения
- `status` - статус эксперимента ("running", "completed")
- `model` - информация о модели (тип, гиперпараметры)
- `environment` - информация об окружении
- `metrics` - итоговые метрики обучения

#### 2. Коллекция `training_steps`
Информация о ходе обучения:
- `experiment_id` - привязка к эксперименту
- `timestep` - текущий шаг обучения
- `datetime` - время записи
- `mean_reward` - средняя награда
- `episode_length` - длина эпизода
- `fires_left` - количество оставшихся пожаров

#### 3. Коллекция `episodes`
Детальная информация о каждом эпизоде:
- `experiment_id` - привязка к эксперименту
- `episode` - номер эпизода
- `reward` - полученная награда
- `length` - количество шагов
- `fires_extinguished` - потушено пожаров
- `collisions` - количество столкновений
- `success` - успешность (все пожары потушены)

#### 4. Коллекция `test_results`
Результаты тестирования обученных моделей:
- `test_id` - ID тестирования
- `experiment_id` - привязка к эксперименту обучения
- `success_rate` - процент успешных попыток
- `avg_reward` - средняя награда
- `step_efficiency` - эффективность шагов
- `collision_rate` - частота столкновений

### Полезные запросы

1. **Лучшие эксперименты по успешности тушения пожаров**
```
db.test_results.find({}, {experiment_id: 1, success_rate: 1, avg_reward: 1})
  .sort({success_rate: -1}).limit(5)
```

2. **Сравнение разных сценариев**
```
db.test_results.aggregate([
  {$lookup: {from: "experiments", localField: "experiment_id", foreignField: "experiment_id", as: "exp_info"}},
  {$unwind: "$exp_info"},
  {$group: {
    _id: "$exp_info.scenario", 
    avg_success: {$avg: "$success_rate"}, 
    avg_reward: {$avg: "$avg_reward"},
    