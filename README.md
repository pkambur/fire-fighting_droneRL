# FireFighter drone RL

**English**: A drone coordination system for firefighting using Reinforcement Learning (RL) methods.  
**Русский**: Система координации дронов для тушения пожаров с использованием методов обучения с подкреплением (Reinforcement Learning).

## 📋 Table of Contents / Содержание

- [Project Overview / Обзор проекта](#project-overview--обзор-проекта)
- [Key Features / Ключевые особенности](#key-features--ключевые-особенности)
- [Project Structure / Структура проекта](#project-structure--структура-проекта)
- [Simulation Environment / Среда симуляции](#simulation-environment--среда-симуляции)
- [Learning Algorithm / Алгоритм обучения](#learning-algorithm--алгоритм-обучения)
- [Visualization and Interface / Визуализация и интерфейс](#visualization-and-interface--визуализация-и-интерфейс)
- [Running the Project / Запуск проекта](#running-the-project--запуск-проекта)
- [Use Cases / Сценарии использования](#use-cases--сценарии-использования)
- [Performance Metrics / Метрики эффективности](#performance-metrics--метрики-эффективности)
- [MongoDB Integration / Интеграция с MongoDB](#mongodb-integration--интеграция-с-mongodb)
- [Result Interpretation / Интерпретация результатов](#result-interpretation--интерпретация-результатов)

## Project Overview / Обзор проекта

**English**: FireFighterRL is a simulation environment and trainable model for managing a team of drones in firefighting tasks. It leverages reinforcement learning to optimize firefighting strategies across various scenarios, accounting for obstacles, environmental conditions (e.g., wind), and resource constraints (e.g., battery life).  

**Русский**: FireFighterRL — это симуляционная среда и обучаемая модель для управления командой дронов при тушении пожаров. Проект использует алгоритмы обучения с подкреплением для оптимизации стратегий пожаротушения в различных сценариях, учитывая препятствия, природные условия (например, ветер) и ограничения ресурсов (например, заряд батареи).

## Key Features / Ключевые особенности

**English**:
- 🤖 Multi-agent control of a team of 3 drones
- 🔄 Two distinct simulation scenarios (FireEnv and FireEnv2)
- 🌪️ Modeling of external factors (wind, obstacles)
- 🎮 Visualization using Pygame
- 🧠 Model training with Proximal Policy Optimization (PPO)
- 🔧 Hyperparameter optimization using Optuna
- 📊 Testing and evaluation of trained models

**Русский**:
- 🤖 Мультиагентное управление командой из 3 дронов
- 🔄 Два разных сценария симуляции (FireEnv и FireEnv2)
- 🌪️ Моделирование внешних факторов (ветер, препятствия)
- 🎮 Визуализация с помощью Pygame
- 🧠 Обучение моделей с использованием алгоритма PPO
- 🔧 Оптимизация гиперпараметров с помощью Optuna
- 📊 Тестирование и оценка эффективности обученных моделей

## Project Structure / Структура проекта

```
FireFighterRL/
├── app.py                  # Main application / Основное приложение
├── constants/              # Constants / Константы
│   ├── colors.py           # Colors for visualization / Цвета для визуализации
│   ├── agent.py            # Agent state constants / Константы состояния агента
│   └── grid.py             # Grid and visualization constants / Константы сетки
├── data/                   # Application data / Данные приложения
│   ├── best_model/         # Saved best models / Сохраненные лучшие модели
│   └── images/             # Images for visualization / Изображения для визуализации
├── envs/                   # Gymnasium environments / Окружения Gymnasium
│   ├── __init__.py         # Environment constants / Константы среды
│   ├── FireEnv.py          # Main firefighting environment / Основная среда
│   ├── FireEnv2.py         # Alternative environment / Альтернативная среда
│   ├── reward_sys.py       # Reward function for scenario 1 / Функция наград для первого сценария
│   ├── reward_sys2.py      # Reward function for scenario 2 / Функция наград для второго сценария
│   ├── Fire.py             # Fire spread simulation / Моделирование распространения огня
│   └── Wind.py             # Wind simulation / Моделирование ветра
├── logs/                   # Logs and results / Логи и результаты
│   ├── logs.csv            # Testing logs / Логи тестирования
│   ├── test_logs.csv       # Detailed test logs / Подробные логи тестов
│   ├── test_rewards.csv    # Detailed reward logs / Подробные логи наград
│   └── ppo_tensorboard/    # TensorBoard logs / Логи для TensorBoard
├── main.py                 # Entry point / Точка входа
├── models/                 # RL models / Модели RL
│   ├── __init__.py
│   ├── model_config.py     # PPO model configuration / Конфигурация модели PPO
│   ├── optuna_train.py     # Hyperparameter optimization / Оптимизация гиперпараметров
│   ├── test_model.py       # Model testing / Тестирование модели
│   ├── TrainingCallBack.py # Training callbacks / Коллбэки для обучения
│   └── train_model.py      # Model training / Обучение модели
├── mongo/                  # Visualization / Визуализация
│   ├── mongo_integration_for_training.py  # MongoDB integration for training / Интеграция MongoDB для обучения
│   ├── mongo_integration.py               # MongoDB connection / Соединение с MongoDB
│   └── mongo_test_integration.py          # MongoDB integration for testing / Интеграция MongoDB для тестирования
├── render/                 # Visualization / Визуализация
│   ├── __init__.py
│   ├── load_images.py      # Image loading / Загрузка изображений
│   └── user_interface.py   # User interface / Пользовательский интерфейс
├── utils/                  # Utilities / Утилиты
│   ├── logger.py           # Logging / Логирование
│   ├── get_console_data.py # Console logging / Логирование в консоли
│   └── logging_files.py    # Log file paths / Пути к файлам логов
├── README.md
└── requirements.txt        # Dependencies / Зависимости
```

## Simulation Environment / Среда симуляции

**English**: The environment is built using the Gymnasium framework with the following characteristics:  
- 🏁 20×20 grid  
- 🚁 Team of 3 drones starting from a base  
- 🔥 Randomly distributed fire sources  
- 🚧 Obstacles to navigate  
- 💨 Random wind affecting drone movement  
- 🔋 Step limit based on drone battery  

**Русский**: Среда разработана с использованием фреймворка Gymnasium и имеет следующие характеристики:  
- 🏁 Сетка размером 20×20 клеток  
- 🚁 Команда из 3 дронов, начинающих с базы  
- 🔥 Случайно распределенные очаги пожаров  
- 🚧 Препятствия, которые нужно обходить  
- 💨 Случайный ветер, влияющий на движение дронов  
- 🔋 Ограничение на максимальное количество шагов (аккумулятор дрона)  

### Reward System / Система наград

| **Action / Действие**                     | **Reward / Вознаграждение**                                      |
|-------------------------------------------|------------------------------------------------------------------|
| Extinguishing a fire / Тушение пожара     | +1.0                                                             |
| Quickly extinguishing all fires / Быстрое тушение всех пожаров | +5.0 (increases with faster completion) / +5.0 (увеличивается при быстром завершении) |
| Approaching a fire / Приближение к огню   | +0.05                                                            |
| Colliding with an obstacle / Столкновение с препятствием | -0.2                                                             |
| Colliding with another drone / Столкновение с другим дроном | -0.3                                                             |
| Wind impact / Воздействие ветра           | -0.15                                                            |
| Idling (no action) / Застаивание         | -0.1                                                             |
| Step / Шаг                                | -0.02                                                            |

## Learning Algorithm / Алгоритм обучения

**English**: The project uses the Proximal Policy Optimization (PPO) algorithm from Stable-Baselines3. Key parameters:  

```python
PPO_DEFAULT_CONFIG = {
    "policy": "MlpPolicy",        # Multi-layer perceptron policy
    "verbose": 1,                 # Logging verbosity
    "learning_rate": 0.0001,      # Learning rate
    "n_steps": 4096,              # Steps before update
    "batch_size": 256,            # Batch size
    "n_epochs": 5,                # Training epochs
    "gamma": 0.99,                # Discount factor
    "gae_lambda": 0.95,           # GAE parameter
    "clip_range": 0.2,            # Clipping range
    "clip_range_vf": 0.2,         # Value function clipping
    "ent_coef": 0.01,             # Entropy coefficient
    "total_timesteps": 100000,    # Total training steps
}
```

Hyperparameter optimization is implemented using Optuna to find the best model configuration.  

**Русский**: Проект использует алгоритм PPO (Proximal Policy Optimization) из библиотеки Stable-Baselines3. Ключевые параметры:  

```python
PPO_DEFAULT_CONFIG = {
    "policy": "MlpPolicy",        # Политика на основе многослойного перцептрона
    "verbose": 1,                 # Уровень вывода информации
    "learning_rate": 0.0001,      # Скорость обучения
    "n_steps": 4096,              # Шагов перед обновлением
    "batch_size": 256,            # Размер батча
    "n_epochs": 5,                # Количество эпох обучения
    "gamma": 0.99,                # Фактор дисконтирования
    "gae_lambda": 0.95,           # Параметр GAE
    "clip_range": 0.2,            # Диапазон отсечения
    "clip_range_vf": 0.2,         # Диапазон отсечения для функции ценности
    "ent_coef": 0.01,             # Коэффициент энтропии
    "total_timesteps": 100000,    # Общее количество шагов обучения
}
```

Оптимизация гиперпараметров реализована с помощью библиотеки Optuna для поиска наилучшей конфигурации модели.

## Visualization and Interface / Визуализация и интерфейс

**English**: The environment and firefighting process are visualized using Pygame. The interface includes:  
- 🖼️ Graphical display of the game field  
- 📊 Dashboard with current state (steps, fires, reward)  
- 💬 Dialog windows for simulation parameter setup  
- 📝 Results window for test outcomes  

**Русский**: Визуализация среды и процесса тушения пожаров выполнена с использованием Pygame. Интерфейс включает:  
- 🖼️ Графическое отображение игрового поля  
- 📊 Информационную панель с текущим состоянием (шаги, очаги, награда)  
- 💬 Диалоговые окна для настройки параметров симуляции  
- 📝 Окно с результатами тестирования  

## Running the Project / Запуск проекта

### Installing Dependencies / Установка зависимостей

```bash
pip install -r requirements.txt
```

### Running the Application / Запуск приложения

```bash
python main.py
```

**English**: After launching, select the operation mode (training or testing) and configure environment parameters:  
- Scenario selection (1 or 2)  
- Number of fire sources  
- Number of obstacles  

**Русский**: После запуска выберите режим работы (обучение или тестирование) и настройте параметры среды:  
- Выбор сценария (1 или 2)  
- Количество очагов пожара  
- Количество препятствий  

## Use Cases / Сценарии использования

**English**:
1. **Training a New Model**: Select "Training" mode, configure environment parameters, and save the trained model.  
2. **Testing a Model**: Select "Test" mode, load an existing model, and review results.  
3. **Hyperparameter Optimization**: Select "Optuna" mode, run optimization, and obtain optimal parameters.  

**Русский**:
1. **Обучение новой модели**: Выберите режим "Обучение", настройте параметры среды и сохраните обученную модель.  
2. **Тестирование модели**: Выберите режим "Тест", загрузите существующую модель и просмотрите результаты.  
3. **Оптимизация гиперпараметров**: Выберите режим "Optuna", запустите оптимизацию и получите оптимальные параметры.  

## Performance Metrics / Метрики эффективности

**English**: The following metrics evaluate model performance:  
- ✅ **Success Rate**: Percentage of successfully extinguished fires  
- 📈 **Average Reward**: Total reward per episode  
- ⏱️ **Step Efficiency**: Steps required to extinguish fires  
- 💥 **Collision Rate**: Average collisions per episode  

**Русский**: Для оценки эффективности моделей используются следующие метрики:  
- ✅ **Успешность**: Процент успешно потушенных пожаров  
- 📈 **Среднее вознаграждение**: Итоговое вознаграждение за эпизод  
- ⏱️ **Эффективность шагов**: Количество шагов на тушение пожара  
- 💥 **Частота столкновений**: Среднее количество столкновений за эпизод  

## MongoDB Integration / Интеграция с MongoDB

**English**: The project integrates with MongoDB for logging, storing, and analyzing experiment results:  
- 📝 **Training Logging**: Real-time storage of training metrics  
- 📊 **Test Results**: Structured storage of model test outcomes  
- 📈 **Data Analysis Tools**: Jupyter notebooks for result analysis  
- 🧮 **Analytical Dashboards**: Visualization and experiment comparison  

Main MongoDB collections:  
- `experiments`: Training experiment details  
- `training_steps`: Metrics at training stages  
- `test_results`: Model test outcomes  
- `test_episodes`: Detailed test episode data  

**Русский**: Проект включает интеграцию с MongoDB для логирования, хранения и анализа результатов:  
- 📝 **Логирование обучения**: Сохранение метрик обучения в реальном времени  
- 📊 **Результаты тестирования**: Структурированное хранение результатов тестов  
- 📈 **Инструменты анализа**: Jupyter-ноутбуки для анализа данных  
- 🧮 **Аналитические дашборды**: Визуализация и сравнение экспериментов  

Основные коллекции MongoDB:  
- `experiments`: Информация об экспериментах  
- `training_steps`: Метрики на этапах обучения  
- `test_results`: Результаты тестирования моделей  
- `test_episodes`: Детальная информация по тестовым эпизодам  

### Data Analysis Capabilities / Возможности анализа данных

**English**: Jupyter notebooks provide advanced data analysis:  
1. **Data Structure Monitoring**: Track document counts and data integrity.  
2. **Training Visualization**: Plot reward trends and parameter impacts.  
3. **Scenario Comparison**: Compare model performance across scenarios.  
4. **Hyperparameter Analysis**: Evaluate different model configurations.  

**Русский**: Jupyter-ноутбуки обеспечивают расширенный анализ данных:  
1. **Мониторинг структуры данных**: Отслеживание количества документов и целостности.  
2. **Визуализация обучения**: Построение графиков наград и анализ параметров.  
3. **Сравнение сценариев**: Сравнение эффективности моделей в разных сценариях.  
4. **Анализ гиперпараметров**: Оценка различных конфигураций моделей.  

## Result Interpretation / Интерпретация результатов

### Training Log Analysis / Анализ логов обучения

**English**: Training logs are saved in CSV format for analysis. Key parameters:  

| **Parameter / Параметр** | **Description / Описание**                          |
|--------------------------|----------------------------------------------------|
| Timestep                 | Current training step / Текущий шаг обучения       |
| Mean Reward              | Average reward per period / Среднее вознаграждение |
| Episode Length           | Episode length (steps) / Длина эпизода (шаги)     |
| Fires Left               | Remaining fires / Оставшиеся очаги                |

Increasing mean reward and decreasing remaining fires indicate successful training.  

**Русский**: Логи обучения сохраняются в формате CSV. Основные параметры:  

| **Параметр**             | **Описание**                                      |
|--------------------------|--------------------------------------------------|
| Timestep                 | Текущий шаг обучения                            |
| Mean Reward              | Среднее вознаграждение за период                |
| Episode Length           | Длина эпизода (количество шагов)                |
| Fires Left               | Количество оставшихся очагов                    |

Увеличение среднего вознаграждения и уменьшение очагов указывают на успешное обучение.

### TensorBoard Visualization / Визуализация в TensorBoard

**English**: Use TensorBoard for in-depth analysis:  

```bash
tensorboard --logdir=./logs/ppo_tensorboard/
```

Tracks:  
- 📈 Reward dynamics  
- 📉 Loss function  
- 🔄 Policy entropy  
- 💰 Value function  

**Русский**: Используйте TensorBoard для глубокого анализа:  

```bash
tensorboard --logdir=./logs/ppo_tensorboard/
```

Отслеживает:  
- 📈 Динамику вознаграждения  
- 📉 Функцию потерь  
- 🔄 Энтропию политики  
- 💰 Функцию ценности  

### Test Results / Результаты тестирования

**English**: Example test output:  

```
Evaluation results:
Success Rate: 80.0%              # Successfully completed episodes
Average Reward: 25.45            # Average reward per episode
Step Efficiency: 120.5 steps/goal # Steps per goal
Collision Rate: 2.3 in episode    # Average collisions
```

**Interpretation**:  
- **Success Rate** > 70%: Good performance  
- **Average Reward** > 20: Effective strategy  
- **Step Efficiency** < 150: Efficient paths  
- **Collision Rate** < 3: Good obstacle avoidance  

**Русский**: Пример результатов тестирования:  

```
Результаты оценки:
Success Rate: 80.0%              # Процент успешно завершенных эпизодов
Average Reward: 25.45            # Среднее вознаграждение за эпизод
Step Efficiency: 120.5 шагов/цель # Шагов на достижение цели
Collision Rate: 2.3 за эпизод     # Среднее количество столкновений
```

**Интерпретация**:  
- **Успешность** > 70%: Хороший показатель  
- **Среднее вознаграждение** > 20: Эффективная стратегия  
- **Эффективность шагов** < 150: Короткие пути  
- **Частота столкновений** < 3: Хорошее избегание препятствий
