# Константы сетки и визуализации
GRID_SIZE = 20
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
STEP_PENALTY = -0.1  #  штраф за шаг
FIRE_REWARD = 500  #  награда за тушение
OBSTACLE_PENALTY = -30  #  штраф за препятствие
OUT_OF_BOUNDS_PENALTY = -5  #  штраф за выход за границы
BASE_PENALTY = 50
CRASH_PENALTY = -100
WIND_PENALTY = - 30
NEAR_FIRE_BONUS = 20  #  бонус за приближение
STAGNATION_THRESHOLD = 20  #  порог застоя
STAGNATION_PENALTY = -2  #  штраф за застой
FINAL_REWARD = 2000  #  награда за тушение всех пожаров


