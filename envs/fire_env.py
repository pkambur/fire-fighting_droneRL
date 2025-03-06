import gymnasium as gym
import numpy as np
import pygame
import random
from collections import deque

from constants.colors import WHITE
from models import (RENDER_FPS, GRID_SIZE, CELL_SIZE, BASE_POSITION, MAX_BATTERY, MAX_ELEMENTS, FIRE_REWARD,
                   NEAR_FIRE_BONUS, OBSTACLE_PENALTY, OUT_OF_BOUNDS_PENALTY, NO_EXTINGUISHER_PENALTY, STAGNATION_PENALTY,
                   STEP_PENALTY, STAGNATION_THRESHOLD, BASE_BONUS, BASE_RECHARGE, BATTERY_THRESHOLD, FINAL_REWARD,
                   EXTINGUISHER_RECHARGE_BONUS)
from render.user_interface import show_input_window, show_summary_window
from gymnasium.spaces import Box, Discrete


class FireEnv(gym.Env):
    """Среда для симуляции тушения пожаров агентом на сетке."""
    metadata = {"render_modes": ["human"], "render_fps": RENDER_FPS}

    def __init__(self, fire_count: int = None, obstacle_count: int = None, render_mode: str = None):
        super().__init__()
        self.distances_to_fires = None
        self.grid_size = GRID_SIZE
        self.cell_size = CELL_SIZE
        self.screen_size = self.grid_size * self.cell_size
        self.base = BASE_POSITION
        self.position = self.base
        self.battery_level = MAX_BATTERY
        self.extinguisher_count = 1
        self.render_mode = render_mode
        self.steps_without_progress = 0
        self.iteration_count = 0
        self.total_reward = 0
        self.max_steps = 5000

        if fire_count is None or obstacle_count is None:
            fire_count, obstacle_count = show_input_window()

        self.fire_count = fire_count
        self.obstacle_count = obstacle_count
        self.fires, self.obstacles = self.generate_positions(self.fire_count, self.obstacle_count)
        self.update_distances_to_fires()

        self.action_space = Discrete(5)
        max_fires = MAX_ELEMENTS - self.obstacle_count
        max_distances = max_fires
        local_view_size = 25
        low = np.array(
            [0, 0, 0, 0, -self.grid_size, -self.grid_size, -self.grid_size, -self.grid_size] +
            [0] * max_distances + [0] * local_view_size, dtype=np.float32
        )
        high = np.array(
            [MAX_BATTERY, 1, max_fires, 1, self.grid_size, self.grid_size, self.grid_size, self.grid_size] +
            [2 * self.grid_size] * max_distances + [3] * local_view_size, dtype=np.float32
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
        self.battery_level = MAX_BATTERY
        self.extinguisher_count = 1
        self.steps_without_progress = 0
        self.iteration_count = 0
        self.total_reward = 0

        self.fires, self.obstacles = self.generate_positions(self.fire_count, self.obstacle_count)
        self.update_distances_to_fires()
        return self._get_state(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = STEP_PENALTY
        done = False
        self.iteration_count += 1

        if self.iteration_count == 1:
            print("Эпизод начался")

        reward, done = self._take_action(action)
        reward = self._apply_additional_rewards(reward)
        
        if len(self.fires) == 0:
            reward += FINAL_REWARD
            done = True
        
        # Завершаем эпизод при battery_level <= 0
        if self.battery_level <= 0:
            done = True
            reward -= 50  # Дополнительный штраф за полную разрядку батареи
        
        done = done or self.iteration_count >= self.max_steps
        self.total_reward += reward

        state = self._get_state()
        return state, reward, done, False, {}

    def _take_action(self, action: int) -> tuple[float, bool]:
        """Обрабатывает действие агента (движение или тушение)."""
        reward = STEP_PENALTY
        done = False

        if action == 4:  # Тушение
            if self.position in self.fires and self.extinguisher_count > 0 and self.battery_level > 0:
                self.fires.remove(self.position)
                self.extinguisher_count -= 1
                self.update_distances_to_fires()
                self.steps_without_progress = 0
                reward = FIRE_REWARD
                print(f"Очаг потушен на {self.position}! Осталось очагов: {len(self.fires)}")
            else:
                reward = NO_EXTINGUISHER_PENALTY
                self.steps_without_progress += 1
        else:  # Движение
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            new_pos = (self.position[0] + dx, self.position[1] + dy)

            if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
                reward = OUT_OF_BOUNDS_PENALTY
                self.steps_without_progress += 1
            elif new_pos in self.obstacles:
                reward = OBSTACLE_PENALTY
                self.steps_without_progress += 1
            else:
                base_dist_old = abs(self.position[0] - self.base[0]) + abs(self.position[1] - self.base[1])
                distance_old = self.distances_to_fires[0] if self.distances_to_fires else float('inf')
                
                self.position = new_pos
                self.update_distances_to_fires()
                
                distance_new = self.distances_to_fires[0] if self.distances_to_fires else float('inf')
                base_dist_new = abs(self.position[0] - self.base[0]) + abs(self.position[1] - self.base[1])

                if distance_new < distance_old:
                    if self.extinguisher_count == 0:
                        reward += 10
                    else:
                        reward += 30
                    self.steps_without_progress = 0
                elif distance_new > distance_old:
                    reward -= 5
                    self.steps_without_progress += 1

                if self.extinguisher_count == 0 and base_dist_new < base_dist_old:
                    reward += 20

                # Уменьшаем батарею только если она больше 0
                if self.battery_level > 0:
                    self.battery_level -= 1
                # Если батарея стала <= 0, это обработается в step

                if self.position == self.base:
                    self.battery_level = min(MAX_BATTERY, self.battery_level + BASE_RECHARGE)
                    if self.extinguisher_count == 0:
                        self.extinguisher_count = 1
                        reward += EXTINGUISHER_RECHARGE_BONUS
                    reward += BASE_BONUS

                if self.battery_level < BATTERY_THRESHOLD:
                    base_distance = abs(self.position[0] - self.base[0]) + abs(self.position[1] - self.base[1])
                    if base_distance > 3:
                        reward -= 30
                    elif self.position == self.base:
                        reward += 20

        return reward, done

    def _apply_additional_rewards(self, reward: float) -> float:
        """Применяет дополнительные награды и штрафы."""
        px, py = self.position
        for fx, fy in self.fires:
            if abs(px - fx) <= 2 and abs(py - fy) <= 2:
                reward += NEAR_FIRE_BONUS
                break
        if self.steps_without_progress > STAGNATION_THRESHOLD:
            reward += STAGNATION_PENALTY
        return reward

    def _get_state(self) -> np.ndarray:
        local_view = self.get_local_view()
        base_distances = [
            self.position[0] - self.base[0],
            self.position[1] - self.base[1]
        ]
        nearest_fire = min(
            self.fires, key=lambda f: abs(f[0] - self.position[0]) + abs(f[1] - self.position[1])) \
            if self.fires else (0, 0)
        fire_distances = [
            self.position[0] - nearest_fire[0],
            self.position[1] - nearest_fire[1]
        ]
        distances = (self.distances_to_fires + [0] *
                    (MAX_ELEMENTS - self.obstacle_count - len(self.distances_to_fires)))
        # Добавляем индикатор необходимости базы
        base_priority = 1.0 if (self.extinguisher_count == 0 or self.battery_level < BATTERY_THRESHOLD) else 0.0
        state = np.concatenate([
            np.array([
                self.battery_level,
                self.extinguisher_count,
                len(self.fires),
                base_priority,  # Новый элемент состояния
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

        if not hasattr(self, 'images'):
            self._load_images()

        self.screen.fill(WHITE)
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
                "base": pygame.transform.scale(pygame.image.load("data/images/base.jpg"),
                                               (self.cell_size, self.cell_size)),
                "agent": pygame.transform.scale(pygame.image.load("data/images/agent.jpg"),
                                                (self.cell_size, self.cell_size)),
                "fire": pygame.transform.scale(pygame.image.load("data/images/fire.jpg"),
                                               (self.cell_size, self.cell_size)),
                "obstacle": pygame.transform.scale(pygame.image.load("data/images/tree.jpg"),
                                                   (self.cell_size, self.cell_size)),
            }
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {e}")

    def close(self) -> None:
        if self.render_mode == "human" and hasattr(self, 'screen'):
            from render.user_interface import show_summary_window
            show_summary_window(self.fire_count, self.obstacle_count, self.iteration_count, self.total_reward)
            del self.screen
    
    # def close(self) -> None:
    #     """Закрывает среду и отображает итоги, если требуется."""
    #     if self.render_mode == "human":
    #         show_summary_window(self.fire_count, self.obstacle_count, self.iteration_count, self.total_reward)
    #         pygame.quit()

