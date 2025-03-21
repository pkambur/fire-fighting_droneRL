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
STEP_PENALTY = -0.02  # штраф за шаг
FIRE_REWARD = 1  # награда за тушение
OBSTACLE_PENALTY = -0.2  # штраф за препятствие
OUT_OF_BOUNDS_PENALTY = -0.1  # штраф за выход за границы
CRASH_PENALTY = -0.3
WIND_PENALTY = -0.15

NEAR_FIRE_BONUS = 0.05  # бонус за приближение
STAGNATION_THRESHOLD = 10  # порог застоя
STAGNATION_PENALTY = -0.1  # штраф за застой
FINAL_REWARD = 0.5  # награда за тушение всех пожаров
