import gymnasium as gym
import numpy as np
import pygame
import random

from collections import deque
from gymnasium.spaces import MultiDiscrete, Box
from constants.colors import WHITE, GREEN, BLACK
import envs as e
from envs.fire_utils import calculate_wind_cells
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
        self.battery_levels = [e.MAX_BATTERY] * self.num_agents
        self.extinguisher_counts = [1] * self.num_agents
        self.render_mode = render_mode
        self.steps_without_progress = [0] * self.num_agents
        self.iteration_count = 0
        self.total_reward = 0
        self.max_steps = 1000
        self.view = e.AGENT_VIEW
        self.distances_to_fires = None
        self.distances_to_obstacles = None

        self.wind_active = False
        self.wind_cells = None
        self.wind_strength = None
        self.wind_direction = None
        self.wind_duration = None
        self.steps_from_last_wind = None
        self.steps_with_wind = None

        if fire_count is None or obstacle_count is None:
            fire_count, obstacle_count = show_input_window()
        self.fire_count = fire_count
        self.obstacle_count = obstacle_count
        self.images = load_images(self.cell_size)
        self.fires, self.obstacles = None, None

        # Пространство действий: единое для observer, 4 действия для каждого из 3 агентов
        self.action_space = MultiDiscrete([4, 4, 4])

        # Пространство наблюдений для observer
        local_view_size = self.view ** 2
        low = np.array(
            [0] * self.num_agents + [0] * self.num_agents + [0] +
            [-self.grid_size, -self.grid_size] * self.num_agents +
            [0] * self.fire_count + [0] * self.obstacle_count * self.num_agents +
            [0] * (local_view_size * self.num_agents),
            dtype=np.float32
        )

        high = np.array(
            [e.MAX_BATTERY] * self.num_agents +  # батарея
            [1] * self.num_agents +  # огнетушитель
            [self.fire_count] +  # кол-во очагов
            [self.grid_size, self.grid_size] * self.num_agents +  # расстояние до базы
            [2 * self.grid_size] * self.fire_count +  # расстояние до очагов
            [2 * self.grid_size] * self.obstacle_count * self.num_agents +  # расстояния до препятствий
            [3] * (local_view_size * self.num_agents),  # что видят агенты
            dtype=np.float32
        )
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        self.positions = [self.base, (1, 9), (2, 9)]
        self.battery_levels = [e.MAX_BATTERY] * self.num_agents
        self.extinguisher_counts = [1] * self.num_agents
        self.steps_without_progress = [0] * self.num_agents
        self.iteration_count = 0
        self.total_reward = 0
        self.fires, self.obstacles = self.generate_positions(self.fire_count, self.obstacle_count)
        self.update_distances()
        self.wind_active = False
        self.wind_strength = 0
        self.wind_direction = []
        self.wind_duration = random.randint(2, 6)
        self.steps_with_wind = 0
        self.steps_from_last_wind = 0
        return self._get_state(), {}

    def generate_positions(self, fire_count: int, obstacle_count: int) -> tuple[set, set]:
        check = 0
        fires, obstacles = {}, {}
        while check != fire_count:
            all_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
            all_positions -= {self.base, (1, 9), (2, 9)}
            fires = set(random.sample(list(all_positions), fire_count))
            all_positions -= fires
            obstacles = set(random.sample(list(all_positions), obstacle_count))
            for fire in fires:
                check += self._check_availability(self.base, fire, obstacles)
        return fires, obstacles

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

    def update_distances(self) -> None:
        self.distances_to_fires = sorted(
            [min(abs(x - pos[0]) + abs(y - pos[1]) for pos in self.positions) for x, y in self.fires]
        ) if self.fires else []
        self.distances_to_obstacles = [abs(x - pos[0]) + abs(y - pos[1]) for pos
                                       in self.positions for x, y in self.obstacles]

    def get_local_view(self, agent_idx: int) -> np.ndarray:
        pos_x, pos_y = self.positions[agent_idx]
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

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.iteration_count += 1
        total_reward = 0

        if self.iteration_count == 1:
            logger.info("Episode started")

        # Применяем действия ко всем агентам одновременно
        for i, action in enumerate(actions):
            reward = self._take_action(i, action)
            total_reward += reward

        total_reward += e.STEP_PENALTY * self.num_agents
        logger.info(f'STEP_PENALTY = {e.STEP_PENALTY * self.num_agents}')

        self.total_reward += total_reward
        state = self._get_state()

        if self.steps_from_last_wind >= random.randint(10, 30):
            self._wind_activation()
        if self.steps_with_wind == self.wind_duration:
            self.wind_active = False

        reward, terminated, truncated = self._check_termination()
        total_reward += reward
        return state, total_reward, terminated, truncated, {}

    def _take_action(self, agent_idx: int, action: int) -> float:
        reward = 0
        self.battery_levels[agent_idx] -= 1
        done = False

        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        x, y = self.positions[agent_idx]
        x += dx
        y += dy
        # new_pos = (self.positions[agent_idx][0] + dx, self.positions[agent_idx][1] + dy)

        if self.wind_active:
            self.steps_with_wind += 1
            self.steps_from_last_wind = 0
            if (x, y) in self.wind_cells:
                x += self.wind_strength * self.wind_direction[0]
                y += self.wind_strength * self.wind_direction[1]
                # чтобы не вылетал за границы от ветра
                x = np.clip(x, 0, self.grid_size - 1)
                y = np.clip(y, 0, self.grid_size - 1)
                reward += e.WIND_PENALTY
                logger.info(f'wind penalty {e.WIND_PENALTY} in {x, y}')
        else:
            self.steps_with_wind = 0
            self.steps_from_last_wind += 1
            self.wind_cells = []

        new_pos = (x, y)
        penalty, collision = self._check_collisions(new_pos, agent_idx)
        if collision:
            reward += penalty

        elif new_pos in self.fires:
            self.fires.remove(new_pos)
            self.extinguisher_counts[agent_idx] -= 1
            self.steps_without_progress[agent_idx] = 0
            reward = e.FIRE_REWARD
            self.positions[agent_idx] = new_pos
            self.update_distances()
            logger.info(f'Agent {agent_idx} extinguished fire at {new_pos}: {e.FIRE_REWARD}')

        else:
            self.positions[agent_idx] = new_pos
            self.update_distances()

        self._recharge(agent_idx)
        logger.info(f'Agent {agent_idx} Position = {self.positions[agent_idx]}')
        return reward

    def _check_termination(self):
        reward = 0
        terminated, truncated = False, False
        if len(self.fires) == 0:
            terminated = True
            if self.iteration_count < self.max_steps // 2:
                reward += e.FINAL_REWARD * 2
                logger.info(f'FINAL_REWARD = {e.FINAL_REWARD * 2}')
            else:
                reward += e.FINAL_REWARD
                logger.info(f'FINAL_REWARD = {e.FINAL_REWARD}')
        elif self.iteration_count >= self.max_steps:
            reward -= e.FINAL_REWARD
            logger.info(f'MAX_STEPS DONE = {-e.FINAL_REWARD}')
            truncated = True
        return reward, terminated, truncated

    def _recharge(self, agent_idx: int):
        if self.battery_levels[agent_idx] < e.MIN_BATTERY or self.extinguisher_counts[agent_idx] == 0:
            self.positions[agent_idx] = self.base
            self.battery_levels[agent_idx] = e.MAX_BATTERY
            self.extinguisher_counts[agent_idx] = 1
            logger.info(f'Agent {agent_idx} recharged at base')

    def _check_collisions(self, new_pos: tuple, agent_idx: int) -> tuple[float, bool]:
        reward = 0
        collision = False
        if new_pos in [self.positions[i] for i in range(3) if i != agent_idx]:
            reward = e.CRASH_PENALTY
            logger.info(f'Agent {agent_idx} collision with another agent: {e.CRASH_PENALTY}')
            collision = True
        elif not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            reward = e.OUT_OF_BOUNDS_PENALTY
            logger.info(f'Agent {agent_idx} out of bounds: {e.OUT_OF_BOUNDS_PENALTY}')
            collision = True
        elif new_pos in self.obstacles:
            reward = e.OBSTACLE_PENALTY
            logger.info(f'Agent {agent_idx} hit obstacle: {e.OBSTACLE_PENALTY}')
            collision = True
        if collision is True:
            self.steps_without_progress[agent_idx] += 1
        return reward, collision

    def _get_state(self) -> np.ndarray:
        base_distances = [[self.positions[i][0] - self.base[0], self.positions[i][1] - self.base[1]]
                          for i in range(self.num_agents)]

        state_parts = [
            np.array(self.battery_levels + self.extinguisher_counts, dtype=np.float32),
            np.array([len(self.fires)], dtype=np.float32),
        ]
        distances_to_fires = (self.distances_to_fires +
                              [0] * (self.fire_count - len(self.distances_to_fires)))

        state = np.concatenate(
            state_parts +
            base_distances +
            [distances_to_fires] +
            [self.distances_to_obstacles] +
            [self.get_local_view(i) for i in range(self.num_agents)])

        return state

    def _wind_activation(self):
        self.wind_active = True
        wind_start_cell = random.choices(list(range(self.grid_size)), k=2)
        self.wind_direction = random.choices([-1, 0, 1], k=2)
        self.wind_strength = random.randint(1, 3)
        self.wind_cells = calculate_wind_cells(wind_start_cell,
                                               self.wind_direction,
                                               self.wind_strength,
                                               self.grid_size)
        logger.info(f"wind {self.wind_cells}")

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size + BAR_WIDTH, self.screen_size))
            pygame.display.set_caption("Fire Environment")
        self.screen.fill(GREEN)

        cell = self.cell_size
        for fire in self.fires:
            self.screen.blit(self.images["fire"], (fire[0] * cell, fire[1] * cell))
        for obstacle in self.obstacles:
            self.screen.blit(self.images["obstacle"], (obstacle[0] * cell, obstacle[1] * cell))
        self.screen.blit(self.images["base"], (self.base[0] * cell, self.base[1] * cell))
        for i in range(self.num_agents):
            self.screen.blit(self.images["agent"], (self.positions[i][0] * cell, self.positions[i][1] * cell))

        if self.wind_cells is not None:
            for wind in self.wind_cells:
                self.screen.blit(self.images["wind"], (wind[0] * cell, wind[1] * cell))

        font = pygame.font.Font(None, FONT_SIZE)
        status_info = pygame.Rect(self.screen_size, 0, BAR_WIDTH,
                                  self.screen_size)
        pygame.draw.rect(self.screen, WHITE, status_info)
        draw_text(self.screen, "Game info", font, BLACK, self.screen_size, 20)
        draw_text(self.screen, f"Step {self.iteration_count}", font, BLACK, self.screen_size, 60)
        pygame.display.flip()
        pygame.time.delay(100)

    def close(self) -> None:
        if self.render_mode == "human" and hasattr(self, 'screen'):
            from render.user_interface import show_summary_window
            show_summary_window(self.fire_count, self.fire_count - len(self.fires),
                                self.obstacle_count, self.iteration_count, self.total_reward)
            del self.screen
