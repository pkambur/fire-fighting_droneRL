import gymnasium as gym
import numpy as np
import pygame
import random

from collections import deque
from render.user_interface import show_input_window


class FireEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, fire_count = None, obstacle_count = None, render_mode = None):
        super().__init__()
        self.grid_size = 10
        self.cell_size = 50
        self.screen_size = self.grid_size * self.cell_size
        self.base = (0, 9)
        self.position = self.base
        self.battery_level = 100
        self.extinguisher_count = 1
        self.render_mode = render_mode
        self.steps_without_progress = 0

        if fire_count is None or obstacle_count is None:
            fire_count, obstacle_count = show_input_window()

        self.fire_count = fire_count
        self.obstacle_count = obstacle_count

        self.max_elements = self.grid_size * self.grid_size // 2
        if fire_count + obstacle_count > self.max_elements:
            raise ValueError(f"Сумма очагов и препятствий ({fire_count + obstacle_count}) не должна превышать {self.max_elements}")

        self.fires, self.obstacles = self.generate_positions(self.fire_count, self.obstacle_count)
        self.update_distances_to_fires()

        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Fire Environment")

    def generate_positions(self, fire_count, obstacle_count):
        all_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        all_positions.remove(self.base)

        queue = deque([self.base])
        reachable = set([self.base])

        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in all_positions and (nx, ny) not in reachable:
                    reachable.add((nx, ny))
                    queue.append((nx, ny))

        fires = set(random.sample(list(reachable - {self.base}), fire_count))
        obstacles = set(random.sample(list(reachable - fires - {self.base}), obstacle_count))

        return fires, obstacles

    def update_distances_to_fires(self):
        self.distances_to_fires = sorted(
            [abs(x - self.position[0]) + abs(y - self.position[1]) for x, y in self.fires]
        ) if self.fires else []

    def get_local_view(self):
        px, py = self.position
        view_size = 2
        local_view = np.zeros((5, 5), dtype=np.int32)

        for dx in range(-view_size, view_size + 1):
            for dy in range(-view_size, view_size + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if (nx, ny) in self.fires:
                        local_view[dx + 2, dy + 2] = 1
                    elif (nx, ny) in self.obstacles:
                        local_view[dx + 2, dy + 2] = 2
                    elif (nx, ny) == self.base:
                        local_view[dx + 2, dy + 2] = 3
                    else:
                        local_view[dx + 2, dy + 2] = 0
        return local_view.flatten()

    def reset(self, seed = None, options = None):
        if seed is not None:
            np.random.seed(seed)
        self.position = self.base
        self.battery_level = 100
        self.extinguisher_count = 1
        self.steps_without_progress = 0

        self.fires, self.obstacles = self.generate_positions(self.fire_count, self.obstacle_count)
        self.update_distances_to_fires()

        local_view = self.get_local_view()

        state = np.concatenate([
            np.array([
                self.battery_level,
                self.extinguisher_count,
                len(self.fires),
                self.position[0] - self.base[0],
                self.position[1] - self.base[1],
            ], dtype = np.int32),
            np.array(self.distances_to_fires, dtype = np.int32),
            local_view
        ])

        return state, {}

    def get_action_towards_base(self):
        """Определяет действие, приближающее агента к базе, с учётом препятствий."""
        px, py = self.position
        bx, by = self.base

        possible_actions = []
        for action, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            new_x, new_y = px + dx, py + dy
            if (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size and
                (new_x, new_y) not in self.obstacles):
                possible_actions.append((action, abs(new_x - bx) + abs(new_y - by)))

        if possible_actions:
            # Выбираем действие, которое минимизирует расстояние до базы
            action = min(possible_actions, key=lambda x: x[1])[0]
            return action
        return random.randint(0, 3)  # Если все пути заблокированы, случайное действие

    def step(self, action):
        reward = -1

        if action == 4:  # Тушение пожара
            if self.position in self.fires:
                if self.extinguisher_count > 0:
                    self.fires.remove(self.position)
                    self.extinguisher_count -= 1
                    reward = 10
                    self.update_distances_to_fires()
                    self.steps_without_progress = 0
                    print(f"Очаг потушен! Осталось очагов: {len(self.fires)}")
                else:
                    reward = -5
                    self.steps_without_progress += 1
            else:
                reward = -5
                self.steps_without_progress += 1

        else:  # Движение
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            new_pos = (self.position[0] + dx, self.position[1] + dy)

            if new_pos == self.position:
                reward = -2
                self.steps_without_progress += 1

            elif 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                if new_pos in self.obstacles:
                    reward = -3
                    self.steps_without_progress += 1
                else:
                    distance_old = self.distances_to_fires[0] if self.distances_to_fires else float('inf')
                    self.position = new_pos
                    self.update_distances_to_fires()
                    distance_new = self.distances_to_fires[0] if self.distances_to_fires else float('inf')

                    if distance_new < distance_old:
                        if distance_old - distance_new == 1:
                            reward = 1
                        elif distance_old - distance_new > 1:
                            reward = 2
                        self.steps_without_progress = 0
                    elif distance_new > distance_old:
                        if distance_new - distance_old == 1:
                            reward = -2
                        else:
                            reward = -3
                        self.steps_without_progress += 1

                    self.battery_level -= 5

                    if self.position == self.base:
                        self.battery_level = min(100, self.battery_level + 20)
                        if self.extinguisher_count == 0:
                            self.extinguisher_count = 1
                        reward += 5

                    if self.battery_level < 30:
                        base_distance = abs(self.position[0] - self.base[0]) + abs(self.position[1] - self.base[1])
                        if base_distance > 3:
                            reward -= 10

                    if self.position == self.base and self.battery_level < 30:
                        reward += 5

            else:
                reward = -3
                self.steps_without_progress += 1

        px, py = self.position
        for fx, fy in self.fires:
            if abs(px - fx) <= 2 and abs(py - fy) <= 2:
                reward += 2
                break

        if self.steps_without_progress > 5:
            reward -= 2

        done = len(self.fires) == 0 or self.battery_level <= 0

        local_view = self.get_local_view()
        state = np.concatenate([
            np.array([
                self.battery_level,
                self.extinguisher_count,
                len(self.fires),
                self.position[0] - self.base[0],
                self.position[1] - self.base[1]
            ], dtype=np.int32),
            np.array(self.distances_to_fires, dtype=np.int32),
            local_view
        ])

        return state, reward, done, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        if not hasattr(self, 'images'):
            self.images = {
                "base": pygame.image.load("data/images/base.jpg"),
                "agent": pygame.image.load("data/images/agent.jpg"),
                "fire": pygame.image.load("data/images/fire.jpg"),
                "obstacle": pygame.image.load("data/images/tree.jpg"),
            }
            for key in self.images:
                self.images[key] = pygame.transform.scale(self.images[key], (self.cell_size, self.cell_size))

        self.screen.fill((255, 255, 255))

        for fire in self.fires:
            self.screen.blit(self.images["fire"], (fire[0] * self.cell_size, fire[1] * self.cell_size))

        for obstacle in self.obstacles:
            self.screen.blit(self.images["obstacle"], (obstacle[0] * self.cell_size, obstacle[1] * self.cell_size))

        self.screen.blit(self.images["base"], (self.base[0] * self.cell_size, self.base[1] * self.cell_size))
        self.screen.blit(self.images["agent"], (self.position[0] * self.cell_size, self.position[1] * self.cell_size))

        pygame.display.flip()
        pygame.time.delay(100)

    def close(self):
        if self.render_mode == "human":
            pygame.quit()