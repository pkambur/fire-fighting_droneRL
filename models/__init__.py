# Константы сетки и визуализации
GRID_SIZE = 10
CELL_SIZE = 50
MAX_ELEMENTS = GRID_SIZE * GRID_SIZE // 2
RENDER_FPS = 10

# Константы состояния агента
MAX_BATTERY = 100
BATTERY_THRESHOLD = 30
BASE_RECHARGE = 50
BASE_POSITION = (0, 9)

# Награды и штрафы
STEP_PENALTY = -2  # штраф за шаг для минимизации итераций
FIRE_REWARD = 100  # Большая награда за тушение
NEAR_FIRE_BONUS = 5  # бонус за приближение к очагу
NO_EXTINGUISHER_PENALTY = -10
OBSTACLE_PENALTY = -3
OUT_OF_BOUNDS_PENALTY = -3
BASE_BONUS = 2
STAGNATION_THRESHOLD = 5
STAGNATION_PENALTY = -2