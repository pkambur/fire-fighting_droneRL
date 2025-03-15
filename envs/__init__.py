# Константы сетки и визуализации
GRID_SIZE = 10
CELL_SIZE = 50
MAX_ELEMENTS = GRID_SIZE * GRID_SIZE // 2
RENDER_FPS = 10

# Константы состояния агента
MAX_BATTERY = 500
MIN_BATTERY = 10
BATTERY_THRESHOLD = 10
BASE_RECHARGE = 50
BASE_POSITION = (0, 9)
AGENT_VIEW = 5

# Награды и штрафы
STEP_PENALTY = -1  #  штраф за шаг
FIRE_REWARD = 500  #  награда за тушение
OBSTACLE_PENALTY = -100  #  штраф за препятствие
OUT_OF_BOUNDS_PENALTY = -5  #  штраф за выход за границы
BASE_PENALTY = 50
CRASH_PENALTY = -100
WIND_PENALTY = - 30

NEAR_FIRE_BONUS = 10  #  бонус за приближение
NO_EXTINGUISHER_PENALTY = -10  # штраф за неудачное тушение
BASE_BONUS = 100  #  бонус за базу
STAGNATION_THRESHOLD = 10  #  порог застоя
STAGNATION_PENALTY = -5  #  штраф за застой
FINAL_REWARD = 2000  #  награда за тушение всех пожаров
EXTINGUISHER_RECHARGE_BONUS = 50  # бонус за пополнение огнетушителя
BATTERY_PENALTY = -500