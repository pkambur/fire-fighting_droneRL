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
STEP_PENALTY = -1  #  штраф за шаг
FIRE_REWARD = 300  #  награда за тушение
NEAR_FIRE_BONUS = 10  #  бонус за приближение
NO_EXTINGUISHER_PENALTY = -50  # штраф за неудачное тушение
OBSTACLE_PENALTY = -5  #  штраф за препятствие
OUT_OF_BOUNDS_PENALTY = -5  #  штраф за выход за границы
BASE_BONUS = 10  #  бонус за базу
STAGNATION_THRESHOLD = 10  #  порог застоя
STAGNATION_PENALTY = -5  #  штраф за застой
FINAL_REWARD = 1000  #  награда за тушение всех пожаров
NO_EXTINGUISHER_PENALTY = -50  #  штраф за попытку тушения без огнетушителя
EXTINGUISHER_RECHARGE_BONUS = 50  # бонус за пополнение огнетушителя



# # Константы сетки и визуализации
# GRID_SIZE = 10
# CELL_SIZE = 50
# MAX_ELEMENTS = GRID_SIZE * GRID_SIZE // 2
# RENDER_FPS = 10

# # Константы состояния агента
# MAX_BATTERY = 100
# BATTERY_THRESHOLD = 30
# BASE_RECHARGE = 50
# BASE_POSITION = (0, 9)

# # Награды и штрафы
# STEP_PENALTY = -1  # штраф за шаг для минимизации итераций
# FIRE_REWARD = 300  # Большая награда за тушение
# NEAR_FIRE_BONUS = 10  # бонус за приближение к очагу
# NO_EXTINGUISHER_PENALTY = -50
# OBSTACLE_PENALTY = -5 # штраф за столкновение с препятствием
# OUT_OF_BOUNDS_PENALTY = -5 #штраф за выход за границы
# BASE_BONUS = 10 #награда за возвращение на базу
# STAGNATION_THRESHOLD = 10 #порог бездействие (отсутствие прогресса)
# STAGNATION_PENALTY = -5 # штраф за бездействие (отсутствие прогресса)
# FINAL_REWARD = 1000  # награда за тушение всех пожаров
# EXTINGUISHER_RECHARGE_BONUS = 50  # бонус за пополнение огнетушителя