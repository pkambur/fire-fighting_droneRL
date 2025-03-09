import logging
import gymnasium as gym
import numpy as np
import pygame
import random

from constants.colors import WHITE
import envs as e

from render.user_interface import show_input_window, load_images
from gymnasium.spaces import Box, Discrete


class FireEnv(gym.Env):
    """Среда для симуляции тушения пожаров агентом на сетке."""
    metadata = {"render_modes": ["human"], "render_fps": e.RENDER_FPS}

    def __init__(self, fire_count: int = None, obstacle_count: int = None, render_mode: str = None):
        super().__init__()
        self.distances_to_fires = None
        self.grid_size = e.GRID_SIZE
        self.cell_size = e.CELL_SIZE
        self.screen_size = self.grid_size * self.cell_size
        self.base = e.BASE_POSITION
        self.position = None
        self.battery_level = None
        self.extinguisher_count = None
        self.render_mode = render_mode
        self.steps_without_progress = None
        self.iteration_count = None
        self.total_reward = None
        self.max_steps = 300
        self.view = e.AGENT_VIEW

        if fire_count is None or obstacle_count is None:
            fire_count, obstacle_count = show_input_window()

        self.fire_count = fire_count
        self.obstacle_count = obstacle_count
        self.images = load_images(self.cell_size)
        self.fires, self.obstacles = None, None

        self.action_space = Discrete(4)
        local_view_size = self.view ** 2
        low = np.array(
            [0, 0, 0, 0,
             -self.grid_size, -self.grid_size, -self.grid_size, -self.grid_size] +
            [0] * fire_count + [0] * local_view_size, dtype=np.float32
        )
        high = np.array(
            [e.MAX_BATTERY, 1, fire_count, 1,
             self.grid_size, self.grid_size, self.grid_size, self.grid_size] +
            [2 * self.grid_size] * fire_count + [3] * local_view_size, dtype=np.float32
        )

        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        """Сбрасывает среду в начальное состояние."""
        logging.info('Reset')
        if seed is not None:
            np.random.seed(seed)
        self.position = self.base
        self.battery_level = e.MAX_BATTERY
        self.extinguisher_count = 1
        self.steps_without_progress = 0
        self.iteration_count = 0
        self.total_reward = 0

        self.fires, self.obstacles = self.generate_positions(self.fire_count, self.obstacle_count)
        self.update_distances_to_fires()
        return self._get_state(), {}

    def generate_positions(self, fire_count: int, obstacle_count: int) -> tuple[set, set]:
        """Генерирует случайные позиции для пожаров и препятствий, достижимых от базы."""
        all_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        all_positions.remove(self.base)
        fires = set(random.sample(list(all_positions), fire_count))
        all_positions -= fires
        obstacles = set(random.sample(list(all_positions), obstacle_count))
        return fires, obstacles

    def update_distances_to_fires(self) -> None:
        """Обновляет расстояния от текущей позиции до всех пожаров."""
        self.distances_to_fires = sorted(
            [abs(x - self.position[0]) + abs(y - self.position[1]) for x, y in self.fires]
        ) if self.fires else []

    def get_local_view(self) -> np.array:
        """Возвращает локальное представление агента (5x5)."""
        pos_x, pos_y = self.position
        view_size = self.view // 2
        local_view = np.zeros((self.view, self.view), dtype=np.int32)

        for dx in range(-view_size, view_size + 1):
            for dy in range(-view_size, view_size + 1):
                x, y = pos_x + dx, pos_y + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    if (x, y) in self.fires:
                        local_view[dx + view_size, dy + view_size] = 1
                    elif (x, y) in self.obstacles:
                        local_view[dx + view_size, dy + view_size] = 2
                    elif (x, y) == self.base:
                        local_view[dx + view_size, dy + view_size] = 3
        return local_view.flatten()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.iteration_count += 1
        done = False

        if self.iteration_count == 1:
            logging.info("Эпизод начался")

        reward, done = self._take_action(action)
        reward += self.check_progress()

        # reward += e.STEP_PENALTY
        # logging.info(f'STEP_PENALTY = {e.STEP_PENALTY}')

        if len(self.fires) == 0:
            reward += e.FINAL_REWARD
            logging.info(f'FINAL_REWARD = {e.FINAL_REWARD}')
            done = True
        elif self.battery_level <= 0:
            done = True
            reward += e.BATTERY_PENALTY
            logging.info(f'BATTERY_PENALTY = {e.BATTERY_PENALTY}')
        elif self.iteration_count >= self.max_steps:
            reward -= 1000
            logging.info(f'MAX_STEPS DONE = 1000')
            done = True

        self.total_reward += reward
        state = self._get_state()
        return state, reward, done, False, {}

    def _take_action(self, action: int) -> (float, bool):
        """Обрабатывает действие агента (движение или тушение)."""
        reward = 0
        done = False
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_pos = (self.position[0] + dx, self.position[1] + dy)
        self.battery_level -= 1
        distance_old = self.distances_to_fires[0]

        if new_pos in self.fires:# and self.extinguisher_count > 0:
            self.fires.remove(new_pos)
            self.extinguisher_count -= 1
            self.steps_without_progress = 0
            reward = e.FIRE_REWARD
            self.position = new_pos
            self.update_distances_to_fires()
            logging.info(f'FIRE_REWARD = {e.FIRE_REWARD}')
            logging.info(f"Очаг потушен на {new_pos}! Осталось очагов: {len(self.fires)}")

        elif not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            reward = e.OUT_OF_BOUNDS_PENALTY
            logging.info(f'OUT_OF_BOUNDS_PENALTY = {e.OUT_OF_BOUNDS_PENALTY} {self.position}')
            self.steps_without_progress += 1

        elif new_pos in self.obstacles:
            reward = e.OBSTACLE_PENALTY
            logging.info(f'OBSTACLE_PENALTY = {e.OBSTACLE_PENALTY}')
            self.steps_without_progress += 1
            # done = True
        else:
            self.position = new_pos
            self.update_distances_to_fires()
            distance_new = self.distances_to_fires[0]
            reward += self.check_dist_from_fires(distance_old, distance_new)

            if self.battery_level < e.BATTERY_THRESHOLD or self.extinguisher_count == 0:
                if self.position == self.base:
                    self.battery_level = min(e.MAX_BATTERY, self.battery_level + e.BASE_RECHARGE)
                    reward += e.BASE_BONUS
                    logging.info(f'BASE CHARGE = {e.BASE_BONUS}')
                    self.extinguisher_count = 1
                else:
                    reward -= e.STEP_PENALTY
                    logging.info(f'BASE - CHARGE = {- e.STEP_PENALTY}')

        return reward, done

    def check_dist_from_fires(self, distance_old, distance_new):
        reward = 0
        if distance_new < distance_old:
            reward += 5
            logging.info(f'distance_new < distance_old = 5')
            self.steps_without_progress = 0
        elif distance_new > distance_old:
            reward -= 5
            logging.info(f'distance_new > distance_old = -5')
            self.steps_without_progress += 1
        return reward

    def check_progress(self) -> float:
        """Применяет дополнительные награды и штрафы."""
        reward = 0
        if self.steps_without_progress > e.STAGNATION_THRESHOLD:
            reward += e.STAGNATION_PENALTY
            logging.info(f'STAGNATION_PENALTY  = {e.STAGNATION_PENALTY}')
        return reward

    def _get_state(self) -> np.ndarray:
        local_view = self.get_local_view()
        x, y = self.position

        base_distances = [x - self.base[0], y - self.base[1]]

        nearest_fire = min(
            self.fires, key=lambda f: abs(f[0] - x) + abs(f[1] - y))\
            if self.fires else (0, 0)
        fire_distances = [x - nearest_fire[0], y - nearest_fire[1]]

        distances = self.distances_to_fires + (self.fire_count - len(self.distances_to_fires)) * [0]

        # индикатор необходимости базы
        base_priority = 1 if (self.extinguisher_count == 0 or self.battery_level < e.BATTERY_THRESHOLD) else 0

        state = np.concatenate([
            np.array([
                         self.battery_level,
                         self.extinguisher_count,
                         len(self.fires),
                         base_priority,
                     ] + base_distances + fire_distances, dtype=np.float32),
            np.array(distances, dtype=np.float32),
            local_view.astype(np.float32)
        ])
        return state

    def render(self) -> None:
        """Отрисовывает текущее состояние среды."""
        if self.render_mode != "human":
            return

        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("Fire Environment")

        self.screen.fill(WHITE)
        for fire in self.fires:
            self.screen.blit(self.images["fire"], (fire[0] * self.cell_size, fire[1] * self.cell_size))
        for obstacle in self.obstacles:
            self.screen.blit(self.images["obstacle"],
                             (obstacle[0] * self.cell_size, obstacle[1] * self.cell_size))
        self.screen.blit(self.images["base"],
                         (self.base[0] * self.cell_size, self.base[1] * self.cell_size))
        self.screen.blit(self.images["agent"],
                         (self.position[0] * self.cell_size, self.position[1] * self.cell_size))

        pygame.display.flip()
        pygame.time.delay(100)

    def close(self) -> None:
        if self.render_mode == "human" and hasattr(self, 'screen'):
            from render.user_interface import show_summary_window
            show_summary_window(self.fire_count,
                                self.fire_count - len(self.fires),
                                self.obstacle_count,
                                self.iteration_count,
                                self.total_reward)
            del self.screen
