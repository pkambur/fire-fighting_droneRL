# Константы сетки и визуализации
GRID_SIZE = 20
CELL_SIZE = 35
MAX_ELEMENTS = GRID_SIZE * GRID_SIZE // 2
RENDER_FPS = 10
TREE_PERCENT = 0.5

# Константы состояния агента
MAX_BATTERY = 500
MIN_BATTERY = 10
BATTERY_THRESHOLD = 10
BASE_RECHARGE = 50
BASE_POSITION = (0, GRID_SIZE - 1)
AGENT_VIEW = 5

# Награды и штрафы
STEP_PENALTY = -0.001  # штраф за шаг
FIRE_REWARD = 2  # награда за тушение
OBSTACLE_PENALTY = -0.025  # штраф за препятствие
OUT_OF_BOUNDS_PENALTY = -0.3  # штраф за выход за границы
CRASH_PENALTY = -0.1
WIND_PENALTY = -0.025
FIRE_SPREAD_PENALTY = -1.0    # Штраф за распространение огня
WIND_AVOID_BONUS = 0.01       # Бонус за избегание ветра
NEAR_FIRE_BONUS = 0.02  # бонус за приближение
STAGNATION_THRESHOLD = 20  # порог застоя
STAGNATION_PENALTY = -0.01  # штраф за застой
FINAL_REWARD = 3  # награда за тушение всех пожаров

NEW_STEP_REWARD = 0.05
REPEAT_STEP_PENALTY = -0.05