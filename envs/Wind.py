import random

from utils.logger import setup_logger

logger = setup_logger()

wind_period_start = 5
wind_period_end = 15
strength_start = 1
strength_end = 3


class Wind:
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
        self.duration = random.randint(wind_period_start, wind_period_end)
        self.steps_with_wind = 0
        self.steps_from_last_wind = 0

    def wind_activation(self):
        self.active = True
        if self.env.iteration_count == 1:
            wind_start_cell = self.env.base
            while wind_start_cell not in self.env.positions:
                wind_start_cell = random.choices(list(range(self.env.grid_size)), k=2)

        wind_start_cell = random.choices(list(range(self.env.grid_size)), k=2)
        self.direction = random.choices([-1, 0, 1], k=2)
        self.strength = random.randint(strength_start, strength_end)
        self.cells = self._calculate_wind_cells(wind_start_cell)
        logger.info(f"wind {self.cells}")

    def _calculate_wind_cells(self, start):
        x_start, y_start = start
        dx, dy = self.direction
        wind_cells = []

        for i in range(self.strength + 1):
            x = x_start + i * dx
            y = y_start + i * dy
            if 0 <= x < self.env.grid_size and 0 <= y < self.env.grid_size:
                wind_cells.append((x, y))
            else:
                break
        return wind_cells
