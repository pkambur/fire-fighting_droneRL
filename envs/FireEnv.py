import gymnasium as gym
import numpy as np
import pygame
import random

from collections import deque
from gymnasium.spaces import MultiDiscrete, Box
from constants.colors import WHITE, GREEN, BLACK
import envs as e
from envs.Wind import Wind
from render import BAR_WIDTH, FONT_SIZE
from render.user_interface import show_input_window, draw_text
from render.load_images import load_images
from utils.logger import setup_logger

logger = setup_logger()


class FireEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": e.RENDER_FPS}

    def __init__(self, fire_count: int = None, obstacle_count: int = None, render_mode: str = None):
        super().__init__()
        self.grid_size = e.GRID_SIZE
        self.cell_size = e.CELL_SIZE
        self.screen_size = self.grid_size * self.cell_size
        self.base = e.BASE_POSITION
        self.positions = [self.base, (1, 9), (2, 9)]  # Начальные позиции 3 агентов
        self.num_agents = 3
        self.render_mode = render_mode
        self.steps_without_progress = [0] * self.num_agents
        self.iteration_count = 0
        self.total_reward = 0
        self.reward = 0
        self.max_steps = e.MAX_BATTERY
        self.view = e.AGENT_VIEW
        self.distances_to_fires = None
        self.wind = Wind(self)

        if fire_count is None or obstacle_count is None:
            fire_count, obstacle_count = show_input_window()
        self.fire_count = fire_count
        self.obstacle_count = obstacle_count
        self.tree_count = int((self.grid_size ** 2 - self.fire_count - self.obstacle_count) * e.TREE_PERCENT)
        self.images = load_images(self.cell_size)
        self.fires, self.obstacles, self.trees = None, None, None
        self.info = {}

        # Пространство действий: единое для observer, 4 действия для каждого из 3 агентов
        self.action_space = MultiDiscrete([4, 4, 4])

        # Пространство наблюдений для observer
        local_view_size = self.view ** 2
        low = np.array(
            [0] +
            [0] * self.num_agents * 2 +
            [0] * self.fire_count +
            [0] * (local_view_size * self.num_agents) +
            [0, -1, -1, 0],
            dtype=np.float32
        )

        high = np.array(
            [self.fire_count] +  # кол-во очагов
            [self.grid_size] * self.num_agents * 2 +  # agent positions
            [2 * self.grid_size] * self.fire_count +  # расстояние до очагов
            [3] * (local_view_size * self.num_agents) +  # что видят агенты
            [1, 1, 1, 3],  # ветер: активность направление x/y, сила
            dtype=np.float32
        )
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        self.positions = [self.base, (1, 9), (2, 9)]
        self.steps_without_progress = [0] * self.num_agents
        self.iteration_count = 0
        self.fires, self.obstacles, self.trees = self.generate_positions()
        self.update_fire_distances()
        self.total_reward = 0
        self.wind.reset()
        logger.info("Episode started")
        return self._get_state(), self.info

    def generate_positions(self) -> tuple[set, set, set]:
        all_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        all_positions -= {self.base, (1, 9), (2, 9)}
        fires = set(random.sample(list(all_positions), self.fire_count))
        all_positions -= fires
        obstacles = set(random.sample(list(all_positions), self.obstacle_count))

        all_fires_accessible = True
        for fire in fires:
            if not self._check_availability(self.base, fire, obstacles):
                all_fires_accessible = False
                break

        if all_fires_accessible:
            all_positions -= obstacles
            trees = set(random.sample(list(all_positions), self.tree_count))
            return fires, obstacles, trees

    def _check_availability(self, start, end, obstacles):
        start_x, start_y = start
        end_x, end_y = end
        queue = deque([(start_x, start_y)])
        visited = [[0] * self.grid_size for _ in range(self.grid_size)]
        visited[start_x][start_y] = True
        while queue:
            x, y = queue.popleft()
            if x == end_x and y == end_y:
                return True
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if self.is_valid(nx, ny) and not visited[nx][ny] and (nx, ny) not in obstacles:
                    visited[nx][ny] = True
                    queue.append((nx, ny))
        return False

    def is_valid(self, x, y):
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def update_fire_distances(self):
        """
        Update minimum distances to closed fire from every agent
        """
        self.distances_to_fires = [
            min(abs(px - fx) + abs(py - fy) for fx, fy in self.fires)
            for px, py in self.positions] if self.fires else []

    def get_local_view(self, agent_idx: int) -> np.ndarray:
        """
        Возвращает локальное представление агента (5x5 клеток вокруг).
        Returns:
        np.array: сплющенный массив локального вида
        """
        pos_x, pos_y = self.positions[agent_idx]
        view_size = self.view // 2
        local_view = np.zeros((self.view, self.view), dtype=np.int32)

        for dx in range(-view_size, view_size + 1):
            for dy in range(-view_size, view_size + 1):
                x, y = pos_x + dx, pos_y + dy
                if not self.is_valid(x, y):
                    local_view[dx + view_size, dy + view_size] = 4  # вне поля
                else:
                    if (x, y) in self.fires:
                        local_view[dx + view_size, dy + view_size] = 1  # пожар
                    elif (x, y) in self.obstacles:
                        local_view[dx + view_size, dy + view_size] = 2  # препятствие
                    elif (x, y) == self.base:
                        local_view[dx + view_size, dy + view_size] = 3  # база
                    elif (x, y) in self.wind.cells:
                        local_view[dx + view_size, dy + view_size] = 5  # ветер
        return local_view.flatten()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.iteration_count += 1
        logger.info(f'Step {self.iteration_count}')
        self.reward = 0
        self.info = {}
        # сделать вылет по очереди
        for i, action in enumerate(actions):
            self._take_action(i, action)

        self.reward += e.STEP_PENALTY * self.num_agents
        logger.info(f'STEP_PENALTY = {e.STEP_PENALTY * self.num_agents}')

        self.update_fire_distances()
        state = self._get_state()

        # расчет появления ветра должен быть привязан на макс кол-во шагов
        if self.wind.steps_from_last_wind >= random.randint(30, 50):
            self.wind.wind_activation()
        elif self.wind.steps_with_wind == self.wind.duration:
            self.wind.active = False

        terminated, truncated = self._check_termination()
        self.total_reward += self.reward

        return state, self.reward, terminated, truncated, self.info

    def _take_action(self, agent_idx: int, action: int):
        if self.fires:
            old_distance = self.distances_to_fires[agent_idx]
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_pos = (self.positions[agent_idx][0] + dx, self.positions[agent_idx][1] + dy)

        if self.wind.active:
            self.wind.steps_with_wind += 1
            self.wind.steps_from_last_wind = 0
        else:
            self.wind.steps_with_wind = 0
            self.wind.steps_from_last_wind += 1
            self.wind.cells = []

        if new_pos in self.wind.cells:
            new_pos = self._wind_influence(new_pos)
            self.positions[agent_idx] = new_pos
            self.steps_without_progress[agent_idx] += 1
        else:
            if self._check_collisions(new_pos, agent_idx):
                self.steps_without_progress[agent_idx] += 1
                self.info["Collision"] = True
            elif new_pos in self.fires:
                self.fires.remove(new_pos)
                self.steps_without_progress[agent_idx] = 0
                self.positions[agent_idx] = new_pos
                self.reward += e.FIRE_REWARD
                if self.iteration_count < self.max_steps // 2:
                    self.reward += e.FIRE_REWARD * 0.5
                    logger.info(f'Agent {agent_idx} fast fire extinguish bonus: +{e.FIRE_REWARD * 0.5}')
                self.info["The goal has been achieved"] = True
                self.positions[agent_idx] = new_pos
                logger.info(f'Agent {agent_idx} extinguished fire at {new_pos}: {e.FIRE_REWARD}')
            else:
                self.steps_without_progress[agent_idx] += 1
                self.positions[agent_idx] = new_pos
                # нужно проверить работает ли
                self.update_fire_distances()
                if self.fires:
                    new_distance = self.distances_to_fires[agent_idx]
                    if new_distance < old_distance:
                        self.steps_without_progress[agent_idx] = 0
                        self.reward += e.NEAR_FIRE_BONUS
                        logger.info(f'Agent {agent_idx} moved closer to fire: +{e.NEAR_FIRE_BONUS}')

        if self.steps_without_progress[agent_idx] >= e.STAGNATION_THRESHOLD:
            self.reward += e.STAGNATION_PENALTY
            logger.info(f'Stagnation penalty for agent {agent_idx}: {e.STAGNATION_PENALTY}')

        logger.info(f'Agent {agent_idx} Position = {self.positions[agent_idx]}')

    def _check_termination(self):
        terminated, truncated = False, False
        if len(self.fires) == 0:
            terminated = True
            if self.iteration_count < self.max_steps // 2:
                self.reward += e.FINAL_REWARD * 2
                logger.info(f'FINAL_REWARD = {e.FINAL_REWARD * 2}')
            else:
                self.reward += e.FINAL_REWARD
                logger.info(f'FINAL_REWARD = {e.FINAL_REWARD}')
            # привязать к батарее
            step_saving_bonus = (self.max_steps - self.iteration_count) * 0.5
            self.reward += step_saving_bonus
            logger.info(f'Step saving bonus: +{step_saving_bonus}')
        elif self.iteration_count >= self.max_steps:
            self.reward -= e.FINAL_REWARD
            logger.info(f'MAX_STEPS DONE = {-e.FINAL_REWARD}')
            truncated = True
        return terminated, truncated

    def _wind_influence(self, pos: tuple[int, int]) -> tuple[int, int]:
        # разделены новые и старые, чтобы проверить в логах работу
        x, y = pos
        new_x, new_y = pos
        if (x, y) in self.wind.cells:
            new_x = x + (self.wind.strength + 1) * self.wind.direction[0]
            new_y = y + (self.wind.strength + 1) * self.wind.direction[1]
            new_x = np.clip(new_x, 0, self.grid_size - 1)
            new_y = np.clip(new_y, 0, self.grid_size - 1)
            self.reward += e.WIND_PENALTY
            logger.info(f'wind penalty {e.WIND_PENALTY} in {x, y} to {new_x, new_y}')
        return new_x, new_y

    def _check_collisions(self, new_pos: tuple, agent_idx: int) -> bool:
        collision = False
        if new_pos in [self.positions[i] for i in range(self.num_agents) if i != agent_idx]:
            self.reward += e.CRASH_PENALTY
            logger.info(f'Agent {agent_idx} collision with another agent: {e.CRASH_PENALTY}')
            self.positions[agent_idx] = new_pos
            collision = True
        elif not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            self.reward += e.OUT_OF_BOUNDS_PENALTY
            logger.info(f'Agent {agent_idx} out of bounds: {e.OUT_OF_BOUNDS_PENALTY}')
            collision = True
        elif new_pos in self.obstacles:
            self.reward += e.OBSTACLE_PENALTY
            logger.info(f'Agent {agent_idx} hit obstacle: {e.OBSTACLE_PENALTY}')
            collision = True
            self.positions[agent_idx] = new_pos
        return collision

    def _get_state(self) -> np.ndarray:
        state_parts = [
            np.array([len(self.fires)], dtype=np.float32),
            (np.concatenate(self.positions))
        ]
        distances_to_fires = (self.distances_to_fires +
                              [0] * (self.fire_count - len(self.distances_to_fires)))

        # Информация о ветре
        wind_info = [
            int(self.wind.active),
            self.wind.direction[0] if self.wind.direction else 0,  # x-направление (-1, 0, 1)
            self.wind.direction[1] if self.wind.direction else 0,  # y-направление (-1, 0, 1)
            self.wind.strength
        ]

        state = np.concatenate(
            state_parts +
            [distances_to_fires] +
            [self.get_local_view(i) for i in range(self.num_agents)] +
            [wind_info])

        return state

    def render(self) -> None:
        cell = self.cell_size
        houses_margin = int(self.grid_size * 0.1)
        if self.render_mode != "human":
            return
        if not hasattr(self, 'screen'):
            pygame.init()
            size = self.screen_size + (houses_margin * cell)
            self.screen = pygame.display.set_mode((self.screen_size + BAR_WIDTH, size))
            pygame.display.set_caption("Fire Fighter")
        self.screen.fill(GREEN)

        for fire in self.fires:
            self.screen.blit(self.images["fire"], (fire[0] * cell, fire[1] * cell))
        for obstacle in self.obstacles:
            self.screen.blit(self.images["obstacle"], (obstacle[0] * cell, obstacle[1] * cell))
        self.screen.blit(self.images["base"], (self.base[0] * cell, self.base[1] * cell))
        for i in range(self.num_agents):
            self.screen.blit(self.images["agent"], (self.positions[i][0] * cell, self.positions[i][1] * cell))
        for tree in self.trees:
            self.screen.blit(self.images["tree"], (tree[0] * cell, tree[1] * cell))
        if self.wind.cells is not None:
            for wind in self.wind.cells:
                self.screen.blit(self.images["wind"], (wind[0] * cell, wind[1] * cell))

        for i in range(0, self.grid_size + houses_margin, 2):
            for j in range(self.grid_size, self.grid_size + houses_margin, 2):
                self.screen.blit(self.images["houses"], (i * cell, j * cell))

        # Правая панель
        size = self.screen_size + (houses_margin * cell)
        font = pygame.font.Font(None, FONT_SIZE)
        status_info = pygame.Rect(self.screen_size, 0, BAR_WIDTH, size)
        pygame.draw.rect(self.screen, WHITE, status_info)

        x_offset = self.screen_size + 5
        y_offset = 20
        draw_text(self.screen, "Info", font, BLACK, x_offset, y_offset)
        y_offset += 40

        # Основные показатели
        draw_text(self.screen, f"Step: {self.iteration_count}", font, BLACK, x_offset, y_offset)
        y_offset += 30
        draw_text(self.screen, f"Fires: {len(self.fires)}", font, BLACK, x_offset, y_offset)
        y_offset += 40

        # Награды
        draw_text(self.screen, f"Total Reward: {self.total_reward:.2f}", font, BLACK, x_offset, y_offset)
        y_offset += 30

        # Состояние ветра
        wind_status = "Wind: " + ("Active" if self.wind.active else "Inactive")
        draw_text(self.screen, wind_status, font, BLACK, x_offset, y_offset)
        y_offset += 30
        wind_dir = tuple(self.wind.direction) if self.wind.direction else (0, 0)
        draw_text(self.screen, f"Dir: {wind_dir}", font, BLACK, x_offset, y_offset)
        y_offset += 30
        draw_text(self.screen, f"Strength: {self.wind.strength if self.wind.strength else 0}", font, BLACK,
                  x_offset, y_offset)
        y_offset += 30
        draw_text(self.screen, f"Wind On: {self.wind.steps_with_wind}", font, BLACK, x_offset, y_offset)
        y_offset += 30
        draw_text(self.screen, f"Next Wind: {self.wind.steps_from_last_wind}", font, BLACK, x_offset, y_offset)

        pygame.display.flip()
        pygame.time.delay(100)

    def close(self) -> None:
        if self.render_mode == "human" and hasattr(self, 'screen'):
            from render.user_interface import show_summary_window
            show_summary_window(self.fire_count, self.fire_count - len(self.fires),
                                self.obstacle_count, self.iteration_count, self.total_reward)
            del self.screen
