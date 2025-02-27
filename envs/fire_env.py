import gymnasium as gym
import numpy as np
import pygame
import random
from collections import deque
from render.user_interface import show_input_window, show_summary_window
from gymnasium.spaces import Box, Discrete

class FireEnv(gym.Env):
    """Среда для симуляции тушения пожаров агентом на сетке."""

    # Константы сетки и визуализации
    GRID_SIZE = 10
    CELL_SIZE = 50
    MAX_ELEMENTS = GRID_SIZE * GRID_SIZE // 2
    RENDER_FPS = 10

    # Константы состояния агента
    MAX_BATTERY = 100
    BATTERY_THRESHOLD = 30
    BASE_RECHARGE = 50
    BASE_POSITION = (0, 9)

    # Награды и штрафы
    STEP_PENALTY = -1
    FIRE_REWARD = 10
    NO_EXTINGUISHER_PENALTY = -5
    OBSTACLE_PENALTY = -3
    OUT_OF_BOUNDS_PENALTY = -3
    NEAR_FIRE_BONUS = 3
    BASE_BONUS = 2
    STAGNATION_THRESHOLD = 5
    STAGNATION_PENALTY = -2

    metadata = {"render_modes": ["human"], "render_fps": RENDER_FPS}

    def __init__(self, fire_count: int = None, obstacle_count: int = None, render_mode: str = None):
        super().__init__()
        self.grid_size = self.GRID_SIZE
        self.cell_size = self.CELL_SIZE
        self.screen_size = self.grid_size * self.cell_size
        self.base = self.BASE_POSITION
        self.position = self.base
        self.battery_level = self.MAX_BATTERY
        self.extinguisher_count = 1
        self.render_mode = render_mode
        self.steps_without_progress = 0
        self.iteration_count = 0
        self.total_reward = 0
        self.max_steps = 1000  # Максимальное количество шагов в эпизоде

        if fire_count is None or obstacle_count is None:
            fire_count, obstacle_count = show_input_window(max_elements=self.MAX_ELEMENTS)

        self.fire_count = fire_count
        self.obstacle_count = obstacle_count
        self.fires, self.obstacles = self.generate_positions(self.fire_count, self.obstacle_count)
        self.update_distances_to_fires()

        # Определение пространства действий: 5 действий (0-3 движение, 4 - тушение)
        self.action_space = Discrete(5)

        # Определение пространства наблюдений
        # Состояние состоит из: [battery, extinguishers, fires_left, dx_to_base, dy_to_base] + distances_to_fires + local_view (5x5)
        max_fires = self.MAX_ELEMENTS - self.obstacle_count
        max_distances = max_fires  # Максимальное количество расстояний до пожаров
        local_view_size = 25  # 5x5 сетка = 25 элементов
        state_size = 5 + max_distances + local_view_size

        # Границы для пространства наблюдений
        low = np.array(
            [0, 0, 0, -self.grid_size, -self.grid_size] +  # battery, extinguishers, fires_left, dx, dy
            [0] * max_distances +  # минимальное расстояние до пожаров
            [0] * local_view_size, dtype=np.float32  # локальная сетка (0-3)
        )
        high = np.array(
            [self.MAX_BATTERY, 1, max_fires, self.grid_size, self.grid_size] +  # battery, extinguishers, fires_left, dx, dy
            [2 * self.grid_size] * max_distances +  # максимальное расстояние (диагональ сетки)
            [3] * local_view_size, dtype=np.float32  # максимальное значение в local_view (база = 3)
        )

        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def generate_positions(self, fire_count: int, obstacle_count: int) -> tuple[set, set]:
        """Генерирует случайные позиции для пожаров и препятствий, достижимых от базы."""
        all_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        all_positions.remove(self.base)

        reachable = self._get_reachable_positions()
        if fire_count + obstacle_count > len(reachable):
            raise ValueError(f"Недостаточно достижимых позиций: {len(reachable)}")

        fires = set(random.sample(list(reachable), fire_count))
        obstacles = set(random.sample(list(reachable - fires), obstacle_count))
        return fires, obstacles

    def _get_reachable_positions(self) -> set:
        """Возвращает множество позиций, достижимых от базы."""
        queue = deque([self.base])
        reachable = {self.base}
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)} - reachable:
                    reachable.add((nx, ny))
                    queue.append((nx, ny))
        return reachable - {self.base}

    def update_distances_to_fires(self) -> None:
        """Обновляет расстояния от текущей позиции до всех пожаров."""
        self.distances_to_fires = sorted(
            [abs(x - self.position[0]) + abs(y - self.position[1]) for x, y in self.fires]
        ) if self.fires else []

    def get_local_view(self) -> np.ndarray:
        """Возвращает локальное представление агента (5x5)."""
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
        return local_view.flatten()

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        """Сбрасывает среду в начальное состояние."""
        if seed is not None:
            np.random.seed(seed)
        self.position = self.base
        self.battery_level = self.MAX_BATTERY
        self.extinguisher_count = 1
        self.steps_without_progress = 0
        self.iteration_count = 0
        self.total_reward = 0

        self.fires, self.obstacles = self.generate_positions(self.fire_count, self.obstacle_count)
        self.update_distances_to_fires()
        return self._get_state(), {}

    def get_action_towards_base(self) -> int:
        """Возвращает действие, приближающее агента к базе."""
        px, py = self.position
        bx, by = self.base
        possible_actions = []
        for action, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
            new_x, new_y = px + dx, py + dy
            if (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size and
                    (new_x, new_y) not in self.obstacles):
                possible_actions.append((action, abs(new_x - bx) + abs(new_y - by)))
        return min(possible_actions, key=lambda x: x[1])[0] if possible_actions else random.randint(0, 3)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = self.STEP_PENALTY
        done = False
        self.iteration_count += 1

        if action == 4:
            reward, done = self._handle_extinguish(reward)
        else:
            reward = self._handle_movement(action, reward)

        reward = self._apply_additional_rewards(reward)
        done = done or len(self.fires) == 0 or self.battery_level <= 0 or self.iteration_count >= self.max_steps
        self.total_reward += reward

        state = self._get_state()
        return state, reward, done, False, {}

    def _handle_extinguish(self, reward: float) -> tuple[float, bool]:
        if self.position in self.fires and self.extinguisher_count > 0:
            print(f"Перед тушением: fires = {self.fires}")
            self.fires.remove(self.position)
            print(f"После тушения: fires = {self.fires}")
            self.extinguisher_count -= 1
            self.update_distances_to_fires()
            self.steps_without_progress = 0
            print(f"Очаг потушен! Осталось очагов: {len(self.fires)}")
            return self.FIRE_REWARD, False
        self.steps_without_progress += 1
        return self.NO_EXTINGUISHER_PENALTY, False

    def _handle_movement(self, action: int, reward: float) -> float:
        """Обрабатывает движение агента."""
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_pos = (self.position[0] + dx, self.position[1] + dy)

        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            self.steps_without_progress += 1
            return self.OUT_OF_BOUNDS_PENALTY

        if new_pos in self.obstacles:
            self.steps_without_progress += 1
            return self.OBSTACLE_PENALTY

        distance_old = self.distances_to_fires[0] if self.distances_to_fires else float('inf')
        self.position = new_pos
        self.update_distances_to_fires()
        distance_new = self.distances_to_fires[0] if self.distances_to_fires else float('inf')

        if distance_new < distance_old:
            reward = 1 if distance_old - distance_new == 1 else 2
            self.steps_without_progress = 0
        elif distance_new > distance_old:
            reward = -2 if distance_new - distance_old == 1 else -3
            self.steps_without_progress += 1

        self.battery_level -= 1  # Уменьшение заряда батареи на 1 
        if self.position == self.base:
            self.battery_level = min(self.MAX_BATTERY, self.battery_level + self.BASE_RECHARGE)
            if self.extinguisher_count == 0:
                self.extinguisher_count = 1
            reward += self.BASE_BONUS

        if self.battery_level < self.BATTERY_THRESHOLD:
            base_distance = abs(self.position[0] - self.base[0]) + abs(self.position[1] - self.base[1])
            if base_distance > 3:
                reward -= 10
            elif self.position == self.base:
                reward += 5

        return reward

    def _apply_additional_rewards(self, reward: float) -> float:
        """Применяет дополнительные награды и штрафы."""
        px, py = self.position
        for fx, fy in self.fires:
            if abs(px - fx) <= 2 and abs(py - fy) <= 2:
                reward += self.NEAR_FIRE_BONUS
                break
        if self.steps_without_progress > self.STAGNATION_THRESHOLD:
            reward += self.STAGNATION_PENALTY
        return reward

    def _get_state(self) -> np.ndarray:
        """Возвращает текущее состояние среды."""
        local_view = self.get_local_view()
        base_distances = [
            self.position[0] - self.base[0],
            self.position[1] - self.base[1]
        ]
        distances = self.distances_to_fires + [0] * (self.MAX_ELEMENTS - self.obstacle_count - len(self.distances_to_fires))
        state = np.concatenate([
            np.array([
                self.battery_level,
                self.extinguisher_count,
                len(self.fires),
            ] + base_distances, dtype=np.float32),
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

        if not hasattr(self, 'images'):
            self._load_images()

        self.screen.fill((255, 255, 255))
        for fire in self.fires:
            self.screen.blit(self.images["fire"], (fire[0] * self.cell_size, fire[1] * self.cell_size))
        for obstacle in self.obstacles:
            self.screen.blit(self.images["obstacle"], (obstacle[0] * self.cell_size, obstacle[1] * self.cell_size))
        self.screen.blit(self.images["base"], (self.base[0] * self.cell_size, self.base[1] * self.cell_size))
        self.screen.blit(self.images["agent"], (self.position[0] * self.cell_size, self.position[1] * self.cell_size))

        pygame.display.flip()
        pygame.time.delay(100)


    def _load_images(self) -> None:
        """Загружает и масштабирует изображения для рендеринга."""
        try:
            self.images = {
                "base": pygame.transform.scale(pygame.image.load("data/images/base.jpg"), (self.cell_size, self.cell_size)),
                "agent": pygame.transform.scale(pygame.image.load("data/images/agent.jpg"), (self.cell_size, self.cell_size)),
                "fire": pygame.transform.scale(pygame.image.load("data/images/fire.jpg"), (self.cell_size, self.cell_size)),
                "obstacle": pygame.transform.scale(pygame.image.load("data/images/tree.jpg"), (self.cell_size, self.cell_size)),
            }
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {e}")

    def close(self) -> None:
        """Закрывает среду и отображает итоги, если требуется."""
        if self.render_mode == "human":
            show_summary_window(self.fire_count, self.obstacle_count, self.iteration_count, self.total_reward)
            pygame.quit()




# import gymnasium as gym
# import numpy as np
# import pygame
# import random
# from collections import deque
# from heapq import heappush, heappop
# from render.user_interface import show_input_window, show_summary_window


# class FireEnv(gym.Env):
#     GRID_SIZE = 10
#     CELL_SIZE = 50
#     MAX_ELEMENTS = GRID_SIZE * GRID_SIZE // 2
#     RENDER_FPS = 10

#     MAX_BATTERY = 100
#     BATTERY_THRESHOLD = 30
#     BASE_RECHARGE = 50
#     BASE_POSITION = (0, 9)

#     STEP_PENALTY = -1
#     FIRE_REWARD = 10
#     NO_EXTINGUISHER_PENALTY = -5
#     OBSTACLE_PENALTY = -3
#     OUT_OF_BOUNDS_PENALTY = -3
#     NEAR_FIRE_BONUS = 3
#     BASE_BONUS = 2
#     STAGNATION_THRESHOLD = 5
#     STAGNATION_PENALTY = -2

#     metadata = {"render_modes": ["human"], "render_fps": RENDER_FPS}

#     def __init__(self, fire_count: int = None, obstacle_count: int = None, render_mode: str = None):
#         super().__init__()
#         self.grid_size = self.GRID_SIZE
#         self.cell_size = self.CELL_SIZE
#         self.screen_size = self.grid_size * self.cell_size
#         self.base = self.BASE_POSITION
#         self.position = self.base
#         self.battery_level = self.MAX_BATTERY
#         self.extinguisher_count = 1
#         self.render_mode = render_mode
#         self.steps_without_progress = 0
#         self.iteration_count = 0
#         self.total_reward = 0

#         if fire_count is None or obstacle_count is None:
#             fire_count, obstacle_count = show_input_window(max_elements=self.MAX_ELEMENTS)

#         self.fire_count = fire_count
#         self.obstacle_count = obstacle_count
#         self.fires, self.obstacles = self.generate_positions(self.fire_count, self.obstacle_count)
#         self.update_distances_to_fires()

#     def generate_positions(self, fire_count: int, obstacle_count: int) -> tuple[set, set]:
#         all_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
#         all_positions.remove(self.base)

#         reachable = self._get_reachable_positions()
#         if fire_count + obstacle_count > len(reachable):
#             raise ValueError(f"Недостаточно достижимых позиций: {len(reachable)}")

#         fires = set(random.sample(list(reachable), fire_count))
#         obstacles = set(random.sample(list(reachable - fires), obstacle_count))
#         return fires, obstacles

#     def _get_reachable_positions(self) -> set:
#         queue = deque([self.base])
#         reachable = {self.base}
#         while queue:
#             x, y = queue.popleft()
#             for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                 nx, ny = x + dx, y + dy
#                 if (nx, ny) in {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)} - reachable:
#                     reachable.add((nx, ny))
#                     queue.append((nx, ny))
#         return reachable - {self.base}

#     def update_distances_to_fires(self) -> None:
#         self.distances_to_fires = sorted(
#             [abs(x - self.position[0]) + abs(y - self.position[1]) for x, y in self.fires]
#         ) if self.fires else []

#     def get_local_view(self) -> np.ndarray:
#         px, py = self.position
#         view_size = 2
#         local_view = np.zeros((5, 5), dtype=np.int32)

#         for dx in range(-view_size, view_size + 1):
#             for dy in range(-view_size, view_size + 1):
#                 nx, ny = px + dx, py + dy
#                 if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
#                     if (nx, ny) in self.fires:
#                         local_view[dx + 2, dy + 2] = 1
#                     elif (nx, ny) in self.obstacles:
#                         local_view[dx + 2, dy + 2] = 2
#                     elif (nx, ny) == self.base:
#                         local_view[dx + 2, dy + 2] = 3
#         return local_view.flatten()

#     def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
#         if seed is not None:
#             np.random.seed(seed)
#         self.position = self.base
#         self.battery_level = self.MAX_BATTERY
#         self.extinguisher_count = 1
#         self.steps_without_progress = 0
#         self.iteration_count = 0
#         self.total_reward = 0

#         self.fires, self.obstacles = self.generate_positions(self.fire_count, self.obstacle_count)
#         self.update_distances_to_fires()
#         return self._get_state(), {}

#     def find_path(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
#         """Находит кратчайший путь от start до goal с помощью A*."""
#         def heuristic(pos: tuple[int, int]) -> int:
#             return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

#         frontier = [(0, start)]
#         came_from = {}
#         cost_so_far = {start: 0}

#         while frontier:
#             _, current = heappop(frontier)

#             if current == goal:
#                 break

#             for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
#                 next_pos = (current[0] + dx, current[1] + dy)
#                 if (not (0 <= next_pos[0] < self.grid_size and 0 <= next_pos[1] < self.grid_size) or
#                         next_pos in self.obstacles):
#                     continue

#                 new_cost = cost_so_far[current] + 1
#                 if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
#                     cost_so_far[next_pos] = new_cost
#                     priority = new_cost + heuristic(next_pos)
#                     heappush(frontier, (priority, next_pos))
#                     came_from[next_pos] = current

#         # Восстановление пути
#         path = []
#         current = goal
#         if goal not in came_from:  # Если цель недостижима
#             print(f"Цель {goal} недостижима от {start}")
#             return []
#         while current in came_from:
#             path.append(current)
#             current = came_from[current]
#         path.reverse()
#         return path

#     def get_best_action(self) -> int:
#         """Выбирает лучшее действие на основе текущей цели."""
#         # Если находимся на месте пожара и есть огнетушитель, тушим
#         if self.position in self.fires and self.extinguisher_count > 0:
#             print(f"Агент на пожаре {self.position}, тушим!")
#             return 4

#         # Выбор цели
#         if self.battery_level < 10 or (self.extinguisher_count == 0 and self.position != self.base):
#             goal = self.base
#             print(f"Низкий заряд ({self.battery_level}) или нет огнетушителей, цель: база {goal}")
#         else:
#             if not self.fires:
#                 goal = self.base  # Если пожаров нет, идём на базу
#                 print("Пожаров нет, идём на базу")
#             else:
#                 goal = min(self.fires, key=lambda f: abs(f[0] - self.position[0]) + abs(f[1] - self.position[1]))
#                 print(f"Цель: ближайший пожар {goal}")

#         # Поиск пути с помощью A*
#         path = self.find_path(self.position, goal)
#         if not path:
#             print(f"Путь к {goal} не найден, стоим на месте")
#             return 0  # Если путь не найден, стоим (действие "вверх", но не двигаемся)

#         next_pos = path[0]
#         dx, dy = next_pos[0] - self.position[0], next_pos[1] - self.position[1]
#         action_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
#         action = action_map.get((dx, dy), 0)  # По умолчанию стоим, если что-то пошло не так
#         print(f"Текущая позиция: {self.position}, следующая: {next_pos}, действие: {action}")
#         return action

#     def step(self, action: int = None) -> tuple[np.ndarray, float, bool, bool, dict]:
#         reward = self.STEP_PENALTY
#         done = False
#         self.iteration_count += 1

#         if action is None:
#             action = self.get_best_action()

#         if action == 4:
#             reward, done = self._handle_extinguish(reward)
#         else:
#             reward = self._handle_movement(action, reward)

#         reward = self._apply_additional_rewards(reward)
#         done = done or len(self.fires) == 0 or self.battery_level <= 0
#         self.total_reward += reward

#         state = self._get_state()
#         return state, reward, done, False, {}

#     def _handle_extinguish(self, reward: float) -> tuple[float, bool]:
#         if self.position in self.fires and self.extinguisher_count > 0:
#             self.fires.remove(self.position)
#             self.extinguisher_count -= 1
#             self.update_distances_to_fires()
#             self.steps_without_progress = 0
#             print(f"Очаг потушен! Осталось очагов: {len(self.fires)}")
#             return self.FIRE_REWARD, False
#         self.steps_without_progress += 1
#         return self.NO_EXTINGUISHER_PENALTY, False

#     def _handle_movement(self, action: int, reward: float) -> float:
#         dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
#         new_pos = (self.position[0] + dx, self.position[1] + dy)

#         if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
#             self.steps_without_progress += 1
#             return self.OUT_OF_BOUNDS_PENALTY

#         if new_pos in self.obstacles:
#             self.steps_without_progress += 1
#             return self.OBSTACLE_PENALTY

#         distance_old = self.distances_to_fires[0] if self.distances_to_fires else float('inf')
#         self.position = new_pos
#         self.update_distances_to_fires()
#         distance_new = self.distances_to_fires[0] if self.distances_to_fires else float('inf')

#         if distance_new < distance_old:
#             reward += 5
#             self.steps_without_progress = 0
#         elif distance_new == distance_old:
#             reward += 0
#             self.steps_without_progress += 1
#         else:
#             reward -= 5
#             self.steps_without_progress += 1

#         self.battery_level -= 1
#         if self.position == self.base:
#             self.battery_level = min(self.MAX_BATTERY, self.battery_level + self.BASE_RECHARGE)
#             if self.extinguisher_count == 0:
#                 self.extinguisher_count = 1
#             reward += self.BASE_BONUS

#         if self.battery_level < self.BATTERY_THRESHOLD:
#             base_distance = abs(self.position[0] - self.base[0]) + abs(self.position[1] - self.base[1])
#             if base_distance <= 3:
#                 reward += 5
#             elif self.position == self.base:
#                 reward += 10
#             else:
#                 reward -= 10

#         return reward

#     def _apply_additional_rewards(self, reward: float) -> float:
#         px, py = self.position
#         for fx, fy in self.fires:
#             if abs(px - fx) <= 2 and abs(py - fy) <= 2:
#                 reward += self.NEAR_FIRE_BONUS
#                 break
#         if self.steps_without_progress > self.STAGNATION_THRESHOLD:
#             reward += self.STAGNATION_PENALTY
#         return reward

#     def _get_state(self) -> np.ndarray:
#         local_view = self.get_local_view()
#         base_distance = abs(self.position[0] - self.base[0]) + abs(self.position[1] - self.base[1])
#         nearest_fire_distance = self.distances_to_fires[0] if self.distances_to_fires else -1
#         return np.concatenate([
#             np.array([
#                 self.battery_level,
#                 self.extinguisher_count,
#                 len(self.fires),
#                 self.position[0] - self.base[0],
#                 self.position[1] - self.base[1],
#                 base_distance,
#                 nearest_fire_distance
#             ], dtype=np.int32),
#             np.array(self.distances_to_fires, dtype=np.int32),
#             local_view
#         ])

#     def render(self) -> None:
#         if self.render_mode != "human":
#             return

#         if not hasattr(self, 'screen'):
#             pygame.init()
#             self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
#             pygame.display.set_caption("Fire Environment")

#         if not hasattr(self, 'images'):
#             self._load_images()

#         self.screen.fill((255, 255, 255))
#         for fire in self.fires:
#             self.screen.blit(self.images["fire"], (fire[0] * self.cell_size, fire[1] * self.cell_size))
#         for obstacle in self.obstacles:
#             self.screen.blit(self.images["obstacle"], (obstacle[0] * self.cell_size, obstacle[1] * self.cell_size))
#         self.screen.blit(self.images["base"], (self.base[0] * self.cell_size, self.base[1] * self.cell_size))
#         self.screen.blit(self.images["agent"], (self.position[0] * self.cell_size, self.position[1] * self.cell_size))

#         pygame.display.flip()
#         pygame.time.delay(100)

#     def _load_images(self) -> None:
#         try:
#             self.images = {
#                 "base": pygame.transform.scale(pygame.image.load("data/images/base.jpg"), (self.cell_size, self.cell_size)),
#                 "agent": pygame.transform.scale(pygame.image.load("data/images/agent.jpg"), (self.cell_size, self.cell_size)),
#                 "fire": pygame.transform.scale(pygame.image.load("data/images/fire.jpg"), (self.cell_size, self.cell_size)),
#                 "obstacle": pygame.transform.scale(pygame.image.load("data/images/tree.jpg"), (self.cell_size, self.cell_size)),
#             }
#         except FileNotFoundError as e:
#             raise FileNotFoundError(f"Не удалось загрузить изображение: {e}")

#     def close(self) -> None:
#         if self.render_mode == "human":
#             show_summary_window(self.fire_count, self.obstacle_count, self.iteration_count, self.total_reward)
#             pygame.quit()


# if __name__ == "__main__":
#     env = FireEnv(fire_count=5, obstacle_count=5, render_mode="human")
#     state, _ = env.reset()
#     done = False
#     while not done:
#         state, reward, done, _, _ = env.step()
#         env.render()
#     env.close()