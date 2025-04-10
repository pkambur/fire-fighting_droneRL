import gymnasium as gym
import numpy as np
import pygame
import random

from collections import deque
from gymnasium.spaces import MultiDiscrete, Box
from constants.colors import WHITE, GREEN, BLACK, LIGHT_GRAY
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
        self.num_agents = 3
        self.base = [(i, self.grid_size - 1) for i in range(self.num_agents)]
        self.positions = self.base.copy()
        self.render_mode = render_mode
        self.steps_without_progress = [0] * self.num_agents
        self.iteration_count = 0
        self.total_reward = 0
        self.reward = 0
        self.max_steps = e.MAX_BATTERY
        self.view = e.AGENT_VIEW
        self.distances_to_fires = None
        self.wind = Wind(self)
        self.visited_cells = set()  # Множество для отслеживания посещённых клеток всеми агентами

        if fire_count is None or obstacle_count is None:
            fire_count, obstacle_count = show_input_window()
        self.fire_count = fire_count
        self.obstacle_count = obstacle_count
        self.images = load_images(self.cell_size)
        self.fires, self.obstacles, self.trees = [], [], []
        self.info = {}

        self.action_space = MultiDiscrete([4, 4, 4])
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
            [self.fire_count] +
            [self.grid_size] * self.num_agents * 2 +
            [2 * self.grid_size] * self.fire_count +
            [3] * (local_view_size * self.num_agents) +
            [1, 1, 1, 3],
            dtype=np.float32
        )
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        self.positions = self.base.copy()
        self.steps_without_progress = [0] * self.num_agents
        self.iteration_count = 0
        self.fires, self.obstacles, self.trees = self.generate_positions()
        self.update_fire_distances()
        self.total_reward = 0
        self.wind.reset()
        self.visited_cells = set(self.base)  # Изначально посещены только базы
        logger.info("Episode started")
        return self._get_state(), self.info

    def generate_positions(self) -> tuple[set, set, set]:
        tree_count = int((self.grid_size ** 2 - self.fire_count - self.obstacle_count) * e.TREE_PERCENT)
        all_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        all_positions -= set(self.base)
        fires = set(random.sample(list(all_positions), self.fire_count))
        all_positions -= fires
        obstacles = self._generate_obstacles(fires, all_positions)
        all_positions -= obstacles
        trees = set(random.sample(list(all_positions), tree_count))
        return fires, obstacles, trees

    def _generate_obstacles(self, fires, all_positions):
        attempt = 0
        while attempt < 100:
            obstacles = set(random.sample(list(all_positions), self.obstacle_count))
            all_fires_accessible = True
            for fire in fires:
                if not self._check_availability(fire, obstacles):
                    all_fires_accessible = False
                    break
            attempt += 1
            if all_fires_accessible:
                return obstacles
        raise RuntimeError("Не удалось сгенерировать препятствия, при которых все пожары доступны.")

    def _check_availability(self, end, obstacles):
        start_x, start_y = 0, self.grid_size - 1
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
        self.distances_to_fires = [
            min(abs(px - fx) + abs(py - fy) for fx, fy in self.fires)
            for px, py in self.positions] if self.fires else []

    def get_local_view(self, agent_idx: int) -> np.ndarray:
        pos_x, pos_y = self.positions[agent_idx]
        view_size = self.view // 2
        local_view = np.zeros((self.view, self.view), dtype=np.int32)
        for dx in range(-view_size, view_size + 1):
            for dy in range(-view_size, view_size + 1):
                x, y = pos_x + dx, pos_y + dy
                if not self.is_valid(x, y):
                    local_view[dx + view_size, dy + view_size] = 4
                else:
                    if (x, y) in self.fires:
                        local_view[dx + view_size, dy + view_size] = 1
                    elif (x, y) in self.obstacles:
                        local_view[dx + view_size, dy + view_size] = 2
                    elif (x, y) in self.base:
                        local_view[dx + view_size, dy + view_size] = 3
                    elif (x, y) in self.wind.cells:
                        local_view[dx + view_size, dy + view_size] = 5
        return local_view.flatten()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.iteration_count += 1
        logger.info(f'Step {self.iteration_count}')
        self.reward = 0
        self.info = {}
        self.wind.check()

        for agent_idx, action in enumerate(actions):
            self._take_action(agent_idx, action)
            self.reward += e.STEP_PENALTY
            logger.info(f'STEP_PENALTY = {e.STEP_PENALTY}')

        self.spread_fire()
        self.update_fire_distances()
        state = self._get_state()

        terminated, truncated = self._check_termination()
        self.total_reward += self.reward

        return state, self.reward, terminated, truncated, self.info

    def spread_fire(self):
        if self.wind.active and random.random() < (0.05 * self.wind.strength / 3) / 1:
            new_fires = set()
            for fx, fy in self.fires:
                nx, ny = fx + self.wind.direction[0], fy + self.wind.direction[1]
                if (self.is_valid(nx, ny) and (nx, ny) not in self.fires and
                    (nx, ny) not in self.obstacles and (nx, ny) not in self.positions):
                    new_fires.add((nx, ny))
            if new_fires:
                self.fires.update(new_fires)
                self.reward += e.FIRE_SPREAD_PENALTY
                logger.info(f"Fire spread by wind to {new_fires}, penalty: {e.FIRE_SPREAD_PENALTY}")

    
    def _take_action(self, agent_idx: int, action: int):
        old_pos = self.positions[agent_idx]
        old_distance = self.distances_to_fires[agent_idx] if self.fires else float('inf')
        
        # Проверка соседних клеток на наличие очагов
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # вверх, вниз, влево, вправо
        fire_directions = []
        unvisited_fire_directions = []
        
        for i, (dx, dy) in enumerate(directions):
            new_x, new_y = old_pos[0] + dx, old_pos[1] + dy
            new_pos_candidate = (new_x, new_y)
            if self.is_valid(new_x, new_y) and new_pos_candidate in self.fires:
                fire_directions.append((i, new_pos_candidate))
                if new_pos_candidate not in self.visited_cells:
                    unvisited_fire_directions.append((i, new_pos_candidate))

        # Если есть соседний очаг, с 95% вероятностью выбираем действие к нему
        if fire_directions and random.random() < 0.15:  # 95% шанс следовать к очагу
            if unvisited_fire_directions:
                action, new_pos = random.choice(unvisited_fire_directions)
            else:
                action, new_pos = random.choice(fire_directions)
            logger.info(f'Agent {agent_idx} forced to fire at {new_pos}')
        else:
            # В 5% случаев или если очагов рядом нет, используем действие от модели
            dx, dy = directions[action]
            new_pos = (old_pos[0] + dx, old_pos[1] + dy)

        # Отталкивание от других агентов
        other_positions = [self.positions[i] for i in range(self.num_agents) if i != agent_idx]
        for pos in other_positions:
            distance = abs(new_pos[0] - pos[0]) + abs(new_pos[1] - pos[1])
            if distance == 1 and random.random() < 0.5:
                dx, dy = new_pos[0] - pos[0], new_pos[1] - pos[1]
                new_pos = (old_pos[0] - dx, old_pos[1] - dy)
                if self.is_valid(new_pos[0], new_pos[1]):
                    logger.info(f'Agent {agent_idx} repelled from {pos} to {new_pos}')
                    break

        # Обработка действия
        if self.wind.active and new_pos in self.wind.cells:
            new_pos = self._wind_influence(new_pos)
            self.positions[agent_idx] = new_pos
            self.steps_without_progress[agent_idx] += 1
            self.visited_cells.add(new_pos)
        else:
            if self._check_collisions(new_pos, agent_idx, old_pos):
                self.steps_without_progress[agent_idx] += 1
                self.info["Collision"] = True
            elif new_pos in self.fires:
                self.fires.remove(new_pos)
                self.steps_without_progress[agent_idx] = 0
                self.positions[agent_idx] = new_pos
                self.visited_cells.add(new_pos)
                self.reward += e.FIRE_REWARD
                if new_pos not in self.visited_cells:  # Бонус за непосещённую клетку (опционально)
                    self.reward += 0.05
                    logger.info(f'Agent {agent_idx} bonus for unvisited fire: +0.05')
                if self.iteration_count < self.max_steps // 2:
                    self.reward += e.FIRE_REWARD * 0.5
                    logger.info(f'Agent {agent_idx} fast fire extinguish bonus: +{e.FIRE_REWARD * 0.5}')
                self.info["The goal has been achieved"] = True
                logger.info(f'Agent {agent_idx} extinguished fire at {new_pos}: {e.FIRE_REWARD}')
            else:
                self.steps_without_progress[agent_idx] += 1
                self.positions[agent_idx] = new_pos
                self.visited_cells.add(new_pos)
                self.update_fire_distances()
                if self.fires:
                    new_distance = self.distances_to_fires[agent_idx]
                    if new_distance < old_distance:
                        self.steps_without_progress[agent_idx] = 0
                        self.reward += e.NEAR_FIRE_BONUS  # Добавляем бонус за приближение
                        logger.info(f'Agent {agent_idx} near fire bonus: +{e.NEAR_FIRE_BONUS}')
                        self.info["Near Fire Bonus"] = True

            if self.wind.active and old_pos in self.wind.cells and new_pos not in self.wind.cells:
                self.reward += e.WIND_AVOID_BONUS
                logger.info(f'Agent {agent_idx} avoided wind: +{e.WIND_AVOID_BONUS}')

        if self.steps_without_progress[agent_idx] >= e.STAGNATION_THRESHOLD:
            self.reward += e.STAGNATION_PENALTY
            logger.info(f'Stagnation penalty for agent {agent_idx}: {e.STAGNATION_PENALTY}')

        logger.info(f'Agent {agent_idx} Position = {self.positions[agent_idx]}')

    def _wind_influence(self, pos: tuple[int, int]) -> tuple[int, int]:
        x, y = pos
        new_x = x + self.wind.strength * self.wind.direction[0]
        new_y = y + self.wind.strength * self.wind.direction[1]
        new_x = np.clip(new_x, 0, self.grid_size - 1)
        new_y = np.clip(new_y, 0, self.grid_size - 1)
        self.reward += e.WIND_PENALTY
        logger.info(f'Wind penalty {e.WIND_PENALTY} in {x, y} to {new_x, new_y}')
        return int(new_x), int(new_y)

    def _check_collisions(self, new_pos: tuple, agent_idx: int, old_pos: tuple) -> bool:
        collision = False
        other_positions = [self.positions[i] for i in range(self.num_agents) if i != agent_idx]
        
        if new_pos in other_positions:
            self.reward += e.CRASH_PENALTY
            logger.info(f'Agent {agent_idx} collision with another agent: {e.CRASH_PENALTY}')
            collision = True
        
        for pos in other_positions:
            distance = abs(new_pos[0] - pos[0]) + abs(new_pos[1] - pos[1])
            if distance <= 1 and new_pos != pos:
                self.reward += -0.2  # Увеличено с -0.1
                logger.info(f'Agent {agent_idx} too close to another agent: -0.2')
        
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            self.reward += e.OUT_OF_BOUNDS_PENALTY
            logger.info(f'Agent {agent_idx} out of bounds: {e.OUT_OF_BOUNDS_PENALTY}')
            collision = True
        elif new_pos in self.obstacles:
            self.reward += e.OBSTACLE_PENALTY
            logger.info(f'Agent {agent_idx} hit obstacle: {e.OBSTACLE_PENALTY}')
            collision = True
            self.positions[agent_idx] = new_pos
        
        return collision

    def _check_termination(self):
        terminated, truncated = False, False
        if len(self.fires) == 0:
            terminated = True
            self.reward += e.FINAL_REWARD
            logger.info(f'FINAL_REWARD = {e.FINAL_REWARD}')
            step_saving_bonus = (self.max_steps - self.iteration_count) * 0.02
            self.reward += step_saving_bonus
            logger.info(f'Step saving bonus: +{step_saving_bonus}')
        elif self.iteration_count >= self.max_steps:
            self.reward += e.FINAL_REWARD
            logger.info(f'MAX_STEPS DONE = {e.FINAL_REWARD}')
            truncated = True
        return terminated, truncated

    def _get_state(self) -> np.ndarray:
        state_parts = [
            np.array([len(self.fires)], dtype=np.float32),
            np.concatenate(self.positions)
        ]
        distances_to_fires = (self.distances_to_fires +
                              [0] * (self.fire_count - len(self.distances_to_fires)))
        wind_info = [
            int(self.wind.active),
            self.wind.direction[0] if self.wind.direction else 0,
            self.wind.direction[1] if self.wind.direction else 0,
            self.wind.strength
        ]
        state = np.concatenate(
            state_parts +
            [distances_to_fires] +
            [self.get_local_view(i) for i in range(self.num_agents)] +
            [wind_info]
        )
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
        # for tree in self.trees:
        #     self.screen.blit(self.images["tree"], (tree[0] * cell, tree[1] * cell))
        if self.wind.cells:
            for wind in self.wind.cells:
                self.screen.blit(self.images["wind"], (wind[0] * cell, wind[1] * cell))

        for i in range(1 + len(self.base), self.grid_size + houses_margin, 2):
            for j in range(self.grid_size, self.grid_size + houses_margin, 2):
                self.screen.blit(self.images["houses"], (i * cell, j * cell))
        for base in self.base:
            self.screen.blit(self.images["base"], (base[0] * cell, base[1] * cell))
        for i in range(self.num_agents):
            self.screen.blit(self.images["agent"], (self.positions[i][0] * cell, self.positions[i][1] * cell))

        size = self.screen_size + (houses_margin * cell)
        font = pygame.font.Font(None, FONT_SIZE)
        status_info = pygame.Rect(self.screen_size, 0, BAR_WIDTH, size)
        pygame.draw.rect(self.screen, LIGHT_GRAY, status_info)

        x_offset = self.screen_size + 5
        y_offset = 20
        draw_text(self.screen, "Информация", font, BLACK, x_offset, y_offset)
        y_offset += 40
        draw_text(self.screen, f"Шагов: {self.iteration_count}", font, BLACK, x_offset, y_offset)
        y_offset += 30
        draw_text(self.screen, f"Очагов: {len(self.fires)}", font, BLACK, x_offset, y_offset)
        y_offset += 40
        draw_text(self.screen, f"Награда: {self.total_reward:.2f}", font, BLACK, x_offset, y_offset)
        y_offset += 30

        wind_status = "Ветер " + ("дует" if self.wind.active else "не дует")
        draw_text(self.screen, wind_status, font, BLACK, x_offset, y_offset)
        y_offset += 30
        wind_dir = tuple(self.wind.direction) if self.wind.direction else (0, 0)
        wind_dir_name = self.wind.DIRECTION_NAMES.get(wind_dir, "не ясно")
        draw_text(self.screen, f"Направление: {wind_dir_name}", font, BLACK, x_offset, y_offset)
        y_offset += 30
        draw_text(self.screen, f"Сила: {self.wind.strength}", font, BLACK, x_offset, y_offset)
        y_offset += 30
        draw_text(self.screen, f"Ветер дует: {self.wind.steps_with_wind} шагов", font, BLACK, x_offset, y_offset)
        y_offset += 30
        draw_text(self.screen, f"След ветер: {self.wind.steps_from_last_wind}", font, BLACK, x_offset, y_offset)

        pygame.display.flip()
        pygame.time.delay(100)

    def close(self) -> None:
        if self.render_mode == "human" and hasattr(self, 'screen'):
            from render.user_interface import show_summary_window
            show_summary_window(self.fire_count, self.fire_count - len(self.fires),
                                self.obstacle_count, self.iteration_count, self.total_reward)
            del self.screen




# import gymnasium as gym
# import numpy as np
# import pygame
# import random

# from collections import deque
# from gymnasium.spaces import MultiDiscrete, Box
# from constants.colors import WHITE, GREEN, BLACK, LIGHT_GRAY
# import envs as e
# from envs.Wind import Wind
# from render import BAR_WIDTH, FONT_SIZE
# from render.user_interface import show_input_window, draw_text
# from render.load_images import load_images
# from utils.logger import setup_logger

# logger = setup_logger()

# class FireEnv(gym.Env):
#     metadata = {"render_modes": ["human"], "render_fps": e.RENDER_FPS}

#     def __init__(self, fire_count: int = None, obstacle_count: int = None, render_mode: str = None):
#         super().__init__()
#         self.grid_size = e.GRID_SIZE
#         self.cell_size = e.CELL_SIZE
#         self.screen_size = self.grid_size * self.cell_size
#         self.num_agents = 3
#         self.base = [(i, self.grid_size - 1) for i in range(self.num_agents)]  # Базы агентов
#         self.positions = self.base.copy()  # Начальные позиции агентов
#         self.render_mode = render_mode
#         self.steps_without_progress = [0] * self.num_agents
#         self.iteration_count = 0
#         self.total_reward = 0
#         self.reward = 0
#         self.max_steps = e.MAX_BATTERY
#         self.view = e.AGENT_VIEW
#         self.distances_to_fires = None
#         self.wind = Wind(self)

#         if fire_count is None or obstacle_count is None:
#             fire_count, obstacle_count = show_input_window()
#         self.fire_count = fire_count
#         self.obstacle_count = obstacle_count
#         self.images = load_images(self.cell_size)
#         self.fires, self.obstacles, self.trees = [], [], []
#         self.info = {}

#         # Пространство действий: 4 действия для каждого из 3 агентов
#         self.action_space = MultiDiscrete([4, 4, 4])

#         # Пространство наблюдений
#         local_view_size = self.view ** 2
#         low = np.array(
#             [0] +
#             [0] * self.num_agents * 2 +
#             [0] * self.fire_count +
#             [0] * (local_view_size * self.num_agents) +
#             [0, -1, -1, 0],
#             dtype=np.float32
#         )
#         high = np.array(
#             [self.fire_count] +
#             [self.grid_size] * self.num_agents * 2 +
#             [2 * self.grid_size] * self.fire_count +
#             [3] * (local_view_size * self.num_agents) +
#             [1, 1, 1, 3],
#             dtype=np.float32
#         )
#         self.observation_space = Box(low=low, high=high, dtype=np.float32)

#     def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
#         if seed is not None:
#             np.random.seed(seed)
#         self.positions = self.base.copy()
#         self.steps_without_progress = [0] * self.num_agents
#         self.iteration_count = 0
#         self.fires, self.obstacles, self.trees = self.generate_positions()
#         self.update_fire_distances()
#         self.total_reward = 0
#         self.wind.reset()
#         logger.info("Episode started")
#         return self._get_state(), self.info

#     def generate_positions(self) -> tuple[set, set, set]:
#         tree_count = int((self.grid_size ** 2 - self.fire_count - self.obstacle_count) * e.TREE_PERCENT)
#         all_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
#         all_positions -= set(self.base)
#         fires = set(random.sample(list(all_positions), self.fire_count))
#         all_positions -= fires
#         obstacles = self._generate_obstacles(fires, all_positions)
#         all_positions -= obstacles
#         trees = set(random.sample(list(all_positions), tree_count))
#         return fires, obstacles, trees

#     def _generate_obstacles(self, fires, all_positions):
#         attempt = 0
#         while attempt < 100:
#             obstacles = set(random.sample(list(all_positions), self.obstacle_count))
#             all_fires_accessible = True
#             for fire in fires:
#                 if not self._check_availability(fire, obstacles):
#                     all_fires_accessible = False
#                     break
#             attempt += 1
#             if all_fires_accessible:
#                 return obstacles
#         raise RuntimeError("Не удалось сгенерировать препятствия, при которых все пожары доступны.")

#     def _check_availability(self, end, obstacles):
#         start_x, start_y = 0, self.grid_size - 1
#         end_x, end_y = end
#         queue = deque([(start_x, start_y)])
#         visited = [[0] * self.grid_size for _ in range(self.grid_size)]
#         visited[start_x][start_y] = True
#         while queue:
#             x, y = queue.popleft()
#             if x == end_x and y == end_y:
#                 return True
#             for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                 nx, ny = x + dx, y + dy
#                 if self.is_valid(nx, ny) and not visited[nx][ny] and (nx, ny) not in obstacles:
#                     visited[nx][ny] = True
#                     queue.append((nx, ny))
#         return False

#     def is_valid(self, x, y):
#         return 0 <= x < self.grid_size and 0 <= y < self.grid_size

#     def update_fire_distances(self):
#         self.distances_to_fires = [
#             min(abs(px - fx) + abs(py - fy) for fx, fy in self.fires)
#             for px, py in self.positions] if self.fires else []

#     def get_local_view(self, agent_idx: int) -> np.ndarray:
#         pos_x, pos_y = self.positions[agent_idx]
#         view_size = self.view // 2
#         local_view = np.zeros((self.view, self.view), dtype=np.int32)
#         for dx in range(-view_size, view_size + 1):
#             for dy in range(-view_size, view_size + 1):
#                 x, y = pos_x + dx, pos_y + dy
#                 if not self.is_valid(x, y):
#                     local_view[dx + view_size, dy + view_size] = 4  # вне поля
#                 else:
#                     if (x, y) in self.fires:
#                         local_view[dx + view_size, dy + view_size] = 1  # пожар
#                     elif (x, y) in self.obstacles:
#                         local_view[dx + view_size, dy + view_size] = 2  # препятствие
#                     elif (x, y) in self.base:
#                         local_view[dx + view_size, dy + view_size] = 3  # база
#                     elif (x, y) in self.wind.cells:
#                         local_view[dx + view_size, dy + view_size] = 5  # ветер
#         return local_view.flatten()

#     def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
#         self.iteration_count += 1
#         logger.info(f'Step {self.iteration_count}')
#         self.reward = 0
#         self.info = {}
#         self.wind.check()  # Управление состоянием ветра

#         for agent_idx, action in enumerate(actions):
#             self._take_action(agent_idx, action)
#             self.reward += e.STEP_PENALTY
#             logger.info(f'STEP_PENALTY = {e.STEP_PENALTY}')

#         self.spread_fire()  # Распространение огня с учетом "раздувания" ветром
#         self.update_fire_distances()
#         state = self._get_state()

#         terminated, truncated = self._check_termination()
#         self.total_reward += self.reward

#         return state, self.reward, terminated, truncated, self.info

#     def spread_fire(self):
#         if self.wind.active and random.random() < (0.05 * self.wind.strength / 3) / 9:
#             new_fires = set()
#             for fx, fy in self.fires:
#                 nx, ny = fx + self.wind.direction[0], fy + self.wind.direction[1]
#                 if (self.is_valid(nx, ny) and (nx, ny) not in self.fires and
#                     (nx, ny) not in self.obstacles and (nx, ny) not in self.positions):
#                     new_fires.add((nx, ny))
#             if new_fires:
#                 self.fires.update(new_fires)
#                 self.reward += e.FIRE_SPREAD_PENALTY
#                 logger.info(f"Fire spread by wind to {new_fires}, penalty: {e.FIRE_SPREAD_PENALTY}")

#     def _take_action(self, agent_idx: int, action: int):
#         if self.fires:
#             old_distance = self.distances_to_fires[agent_idx]
#         old_pos = self.positions[agent_idx]
#         dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
#         new_pos = (old_pos[0] + dx, old_pos[1] + dy)

#         if self.wind.active and new_pos in self.wind.cells:
#             new_pos = self._wind_influence(new_pos)
#             self.positions[agent_idx] = new_pos
#             self.steps_without_progress[agent_idx] += 1
#         else:
#             if self._check_collisions(new_pos, agent_idx, old_pos):  # Передаем old_pos
#                 self.steps_without_progress[agent_idx] += 1
#                 self.info["Collision"] = True
#             elif new_pos in self.fires:
#                 self.fires.remove(new_pos)
#                 self.steps_without_progress[agent_idx] = 0
#                 self.positions[agent_idx] = new_pos
#                 self.reward += e.FIRE_REWARD
#                 if self.iteration_count < self.max_steps // 2:
#                     self.reward += e.FIRE_REWARD * 0.5
#                     logger.info(f'Agent {agent_idx} fast fire extinguish bonus: +{e.FIRE_REWARD * 0.5}')
#                 self.info["The goal has been achieved"] = True
#                 logger.info(f'Agent {agent_idx} extinguished fire at {new_pos}: {e.FIRE_REWARD}')
#             else:
#                 self.steps_without_progress[agent_idx] += 1
#                 self.positions[agent_idx] = new_pos
#                 self.update_fire_distances()
#                 if self.fires:
#                     new_distance = self.distances_to_fires[agent_idx]
#                     if new_distance < old_distance:
#                         self.steps_without_progress[agent_idx] = 0
#                         self.reward += e.NEAR_FIRE_BONUS 

#             if self.wind.active and old_pos in self.wind.cells and new_pos not in self.wind.cells:
#                 self.reward += e.WIND_AVOID_BONUS
#                 logger.info(f'Agent {agent_idx} avoided wind: +{e.WIND_AVOID_BONUS}')

#         if self.steps_without_progress[agent_idx] >= e.STAGNATION_THRESHOLD:
#             self.reward += e.STAGNATION_PENALTY
#             logger.info(f'Stagnation penalty for agent {agent_idx}: {e.STAGNATION_PENALTY}')

#         logger.info(f'Agent {agent_idx} Position = {self.positions[agent_idx]}')

#     def _check_termination(self):
#         terminated, truncated = False, False
#         if len(self.fires) == 0:
#             terminated = True
#             self.reward += e.FINAL_REWARD
#             logger.info(f'FINAL_REWARD = {e.FINAL_REWARD}')
#             step_saving_bonus = (self.max_steps - self.iteration_count) * 0.05  # Увеличено с 0.005 до 0.01
#             self.reward += step_saving_bonus
#             logger.info(f'Step saving bonus: +{step_saving_bonus}')
#         elif self.iteration_count >= self.max_steps:
#             self.reward += e.FINAL_REWARD  # Используем награду вместо штрафа
#             logger.info(f'MAX_STEPS DONE = {e.FINAL_REWARD}')
#             truncated = True
#         return terminated, truncated

#     def _wind_influence(self, pos: tuple[int, int]) -> tuple[int, int]:
#         x, y = pos
#         new_x = x + self.wind.strength * self.wind.direction[0]  # Полная сила ветра для агентов
#         new_y = y + self.wind.strength * self.wind.direction[1]
#         new_x = np.clip(new_x, 0, self.grid_size - 1)
#         new_y = np.clip(new_y, 0, self.grid_size - 1)
#         self.reward += e.WIND_PENALTY
#         logger.info(f'Wind penalty {e.WIND_PENALTY} in {x, y} to {new_x, new_y}')
#         return new_x, new_y

#     def _check_collisions(self, new_pos: tuple, agent_idx: int, old_pos: tuple) -> bool:
#         collision = False
#         other_positions = [self.positions[i] for i in range(self.num_agents) if i != agent_idx]
        
#         # Проверка столкновения с другими агентами
#         if new_pos in other_positions:
#             self.reward += e.CRASH_PENALTY
#             logger.info(f'Agent {agent_idx} collision with another agent: {e.CRASH_PENALTY}')
#             collision = True
        
#         # Штраф за близость к другим агентам (манхэттенское расстояние <= 1)
#         for pos in other_positions:
#             distance = abs(new_pos[0] - pos[0]) + abs(new_pos[1] - pos[1])
#             if distance <= 1 and new_pos != pos:  # Исключаем случай, когда new_pos уже проверен выше
#                 self.reward += -0.1  # Штраф за близость
#                 logger.info(f'Agent {agent_idx} too close to another agent: -0.1')
        
#         # Проверка выхода за границы
#         if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
#             self.reward += e.OUT_OF_BOUNDS_PENALTY
#             logger.info(f'Agent {agent_idx} out of bounds: {e.OUT_OF_BOUNDS_PENALTY}')
#             collision = True
#         # Проверка столкновения с препятствиями
#         elif new_pos in self.obstacles:
#             self.reward += e.OBSTACLE_PENALTY
#             logger.info(f'Agent {agent_idx} hit obstacle: {e.OBSTACLE_PENALTY}')
#             collision = True
#             self.positions[agent_idx] = new_pos  # Агент всё равно перемещается на препятствие
        
#         return collision

#     def _get_state(self) -> np.ndarray:
#         state_parts = [
#             np.array([len(self.fires)], dtype=np.float32),
#             np.concatenate(self.positions)
#         ]
#         distances_to_fires = (self.distances_to_fires +
#                               [0] * (self.fire_count - len(self.distances_to_fires)))
#         wind_info = [
#             int(self.wind.active),
#             self.wind.direction[0] if self.wind.direction else 0,
#             self.wind.direction[1] if self.wind.direction else 0,
#             self.wind.strength
#         ]
#         state = np.concatenate(
#             state_parts +
#             [distances_to_fires] +
#             [self.get_local_view(i) for i in range(self.num_agents)] +
#             [wind_info]
#         )
#         return state

#     def render(self) -> None:
#         cell = self.cell_size
#         houses_margin = int(self.grid_size * 0.1)
#         if self.render_mode != "human":
#             return
#         if not hasattr(self, 'screen'):
#             pygame.init()
#             size = self.screen_size + (houses_margin * cell)
#             self.screen = pygame.display.set_mode((self.screen_size + BAR_WIDTH, size))
#             pygame.display.set_caption("Fire Fighter")
#         self.screen.fill(GREEN)

#         for fire in self.fires:
#             self.screen.blit(self.images["fire"], (fire[0] * cell, fire[1] * cell))
#         for obstacle in self.obstacles:
#             self.screen.blit(self.images["obstacle"], (obstacle[0] * cell, obstacle[1] * cell))
#         for tree in self.trees:
#             self.screen.blit(self.images["tree"], (tree[0] * cell, tree[1] * cell))
#         if self.wind.cells:
#             for wind in self.wind.cells:
#                 self.screen.blit(self.images["wind"], (wind[0] * cell, wind[1] * cell))

#         for i in range(1 + len(self.base), self.grid_size + houses_margin, 2):
#             for j in range(self.grid_size, self.grid_size + houses_margin, 2):
#                 self.screen.blit(self.images["houses"], (i * cell, j * cell))
#         for base in self.base:
#             self.screen.blit(self.images["base"], (base[0] * cell, base[1] * cell))
#         for i in range(self.num_agents):
#             self.screen.blit(self.images["agent"], (self.positions[i][0] * cell, self.positions[i][1] * cell))

#         # Правая панель
#         size = self.screen_size + (houses_margin * cell)
#         font = pygame.font.Font(None, FONT_SIZE)
#         status_info = pygame.Rect(self.screen_size, 0, BAR_WIDTH, size)
#         pygame.draw.rect(self.screen, LIGHT_GRAY, status_info)

#         x_offset = self.screen_size + 5
#         y_offset = 20
#         draw_text(self.screen, "Информация", font, BLACK, x_offset, y_offset)
#         y_offset += 40
#         draw_text(self.screen, f"Шагов: {self.iteration_count}", font, BLACK, x_offset, y_offset)
#         y_offset += 30
#         draw_text(self.screen, f"Очагов: {len(self.fires)}", font, BLACK, x_offset, y_offset)
#         y_offset += 40
#         draw_text(self.screen, f"Награда: {self.total_reward:.2f}", font, BLACK, x_offset, y_offset)
#         y_offset += 30

#         wind_status = "Ветер " + ("дует" if self.wind.active else "не дует")
#         draw_text(self.screen, wind_status, font, BLACK, x_offset, y_offset)
#         y_offset += 30
#         wind_dir = tuple(self.wind.direction) if self.wind.direction else (0, 0)
#         wind_dir_name = self.wind.DIRECTION_NAMES.get(wind_dir, "не ясно")
#         draw_text(self.screen, f"Направление: {wind_dir_name}", font, BLACK, x_offset, y_offset)
#         y_offset += 30
#         draw_text(self.screen, f"Сила: {self.wind.strength}", font, BLACK, x_offset, y_offset)
#         y_offset += 30
#         draw_text(self.screen, f"Ветер дует: {self.wind.steps_with_wind} шагов", font, BLACK, x_offset, y_offset)
#         y_offset += 30
#         draw_text(self.screen, f"След ветер: {self.wind.steps_from_last_wind}", font, BLACK, x_offset, y_offset)

#         pygame.display.flip()
#         pygame.time.delay(100)

#     def close(self) -> None:
#         if self.render_mode == "human" and hasattr(self, 'screen'):
#             from render.user_interface import show_summary_window
#             show_summary_window(self.fire_count, self.fire_count - len(self.fires),
#                                 self.obstacle_count, self.iteration_count, self.total_reward)
#             del self.screen