import random

from utils.logger import setup_logger

logger = setup_logger()

wind_period_start = 5
wind_period_end = 15
strength_start = 1
strength_end = 3
steps_from_wind_start = 30
steps_from_wind_end = 50


class Wind:
    DIRECTION_NAMES = {
        (0, 1): "Север",
        (0, -1): "Юг",
        (1, 0): "Восток",
        (-1, 0): "Запад",
        (1, 1): "Северо-восток",
        (-1, 1): "Северо-запад",
        (1, -1): "Юго-восток",
        (-1, -1): "Юго-запад",
        (0, 0): "Штиль"
    }

    def __init__(self, env):
        self.env = env
        self.active = False
        self.cells = []
        self.strength = None
        self.direction = None
        self.duration = None
        self.steps_from_last_wind = None
        self.steps_with_wind = None

    def reset(self):
        self.active = False
        self.strength = 0
        self.direction = []
        self.cells = []
        self.duration = random.randint(wind_period_start, wind_period_end)
        self.steps_with_wind = 0
        self.steps_from_last_wind = 0

    def wind_activation(self):
        self.active = True

        wind_start_cell = random.choices(range(self.env.grid_size), k=2)
        while tuple(wind_start_cell) in self.env.positions:
            wind_start_cell = random.choices(range(self.env.grid_size), k=2)

        wind_start_cell = random.choices(range(self.env.grid_size), k=2)
        self.direction = random.choices([-1, 0, 1], k=2)
        while self.direction == [0, 0]:
            self.direction = random.choices([-1, 0, 1], k=2)

        self.strength = random.randint(strength_start, strength_end)
        self.cells = self._calculate_wind_cells(wind_start_cell)
        logger.info(
            f"Wind activated: start={wind_start_cell}, direction={self.direction},"
            f" strength={self.strength}, cells={self.cells}")

    def check(self):
        if self.active:
            self.steps_with_wind += 1
            self.steps_from_last_wind = 0
            if self.steps_with_wind == self.duration:
                self.reset()
        else:
            self.steps_with_wind = 0
            self.steps_from_last_wind += 1
            if self.steps_from_last_wind >= random.randint(steps_from_wind_start, steps_from_wind_end):
                self.wind_activation()

    def _calculate_wind_cells(self, start):
        x_start, y_start = start
        dx, dy = self.direction
        wind_cells = []

        for i in range(self.strength + 1):
            x = x_start + i * dx
            y = y_start + i * dy
            if not self.env.is_valid(x, y):
                break
            wind_cells.append((x, y))
        return wind_cells
