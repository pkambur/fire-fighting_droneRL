import random


class Fire:

    def __init__(self, env):
        self.env = env
        self.fire_cells = set()
        self.goals = set()
        self.radius = 0
        self.center_x = 0
        self.center_y = 0

    def reset(self):
        self.center_x, self.center_y = self._generate_fires_center_position()
        self.goals = self._generate_goals()
        self.radius = random.randint(1, 3)
        self.generate_fire()

    def generate_fire(self):
        for i in range(1, self.radius + 1):
            for dx in range(-i, i + 1):
                for dy in range(-i, i + 1):
                    self.fire_cells.add((self.center_x + dx, self.center_y + dy))

    def _generate_fires_center_position(self) -> tuple:
        """Generates coordinates of the position of the fires central cell, relative to which
           the fires will be located"""
        x = self.env.np_random.integers(1, (self.env.grid_size - 1))
        y = self.env.np_random.integers(1, (self.env.grid_size - 1))
        return x, y

    def _generate_goals(self) -> set[tuple]:
        """Generates coordinates of goals from four sides relative to the goals central cell"""
        goals = set()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x = self.center_x + dx
            new_y = self.center_y + dy
            goals.add((new_x, new_y))
        return goals
