import logging
import gymnasium as gym
import numpy as np
import pygame
import random
from gymnasium.spaces import MultiDiscrete, Box
from constants.colors import WHITE
import envs as e
from render.user_interface import show_input_window, load_images

class FireEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": e.RENDER_FPS}

    def __init__(self, fire_count: int = None, obstacle_count: int = None, render_mode: str = None):
        super().__init__()
        self.grid_size = e.GRID_SIZE
        self.cell_size = e.CELL_SIZE
        self.screen_size = self.grid_size * self.cell_size
        self.base = e.BASE_POSITION
        self.positions = [self.base, (1, 9), (2, 9)]  # Начальные позиции 3 агентов
        self.battery_levels = [e.MAX_BATTERY] * 3
        self.extinguisher_counts = [1] * 3
        self.render_mode = render_mode
        self.steps_without_progress = [0] * 3
        self.iteration_count = 0
        self.total_reward = 0
        self.max_steps = 1000
        self.view = e.AGENT_VIEW
        self.distances_to_fires = None

        if fire_count is None or obstacle_count is None:
            fire_count, obstacle_count = show_input_window()
        self.fire_count = fire_count
        self.obstacle_count = obstacle_count
        self.images = load_images(self.cell_size)
        self.fires, self.obstacles = None, None

        self.action_space = MultiDiscrete([4, 4, 4])

        local_view_size = self.view ** 2
        low = np.array(
            [0] * 3 + [0] * 3 + [0] + [-self.grid_size, -self.grid_size] * 3 +
            [-self.grid_size, -self.grid_size] + [0] * fire_count + [0] * local_view_size * 3,
            dtype=np.float32
        )
        high = np.array(
            [e.MAX_BATTERY] * 3 + [1] * 3 + [fire_count] + [self.grid_size, self.grid_size] * 3 +
            [self.grid_size, self.grid_size] + [2 * self.grid_size] * fire_count + [3] * local_view_size * 3,
            dtype=np.float32
        )
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def reset(self, seed: int = None, options: dict = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            np.random.seed(seed)
        self.positions = [self.base, (1, 9), (2, 9)]
        self.battery_levels = [e.MAX_BATTERY] * 3
        self.extinguisher_counts = [1] * 3
        self.steps_without_progress = [0] * 3
        self.iteration_count = 0
        self.total_reward = 0
        self.fires, self.obstacles = self.generate_positions(self.fire_count, self.obstacle_count)
        self.update_distances_to_fires()
        return self._get_state(), {}

    def generate_positions(self, fire_count: int, obstacle_count: int) -> tuple[set, set]:
        all_positions = {(x, y) for x in range(self.grid_size) for y in range(self.grid_size)}
        all_positions -= {self.base, (1, 9), (2, 9)}
        fires = set(random.sample(list(all_positions), fire_count))
        all_positions -= fires
        obstacles = set(random.sample(list(all_positions), obstacle_count))
        return fires, obstacles

    def update_distances_to_fires(self) -> None:
        self.distances_to_fires = sorted(
            [min(abs(x - pos[0]) + abs(y - pos[1]) for pos in self.positions) for x, y in self.fires]
        ) if self.fires else []

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
        terminated, truncated = False, False

        if self.iteration_count == 1:
            logging.info("Episode started")

        for i, action in enumerate(actions):
            reward, _ = self._take_action(i, action)
            total_reward += reward

        total_reward += e.STEP_PENALTY * 3
        logging.info(f'STEP_PENALTY = {e.STEP_PENALTY * 3}')

        if len(self.fires) == 0:
            terminated = True
            if self.iteration_count < self.max_steps // 2:
                total_reward += e.FINAL_REWARD * 2
                logging.info(f'FINAL_REWARD = {e.FINAL_REWARD * 2}')
            else:
                total_reward += e.FINAL_REWARD
                logging.info(f'FINAL_REWARD = {e.FINAL_REWARD}')
        elif self.iteration_count >= self.max_steps:
            total_reward -= e.FINAL_REWARD
            logging.info(f'MAX_STEPS DONE = {-e.FINAL_REWARD}')
            truncated = True

        self.total_reward += total_reward
        state = self._get_state()
        return state, total_reward, terminated, truncated, {}

    def _take_action(self, agent_idx: int, action: int) -> tuple[float, bool]:
        reward = 0
        self.battery_levels[agent_idx] -= 1
        done = False

        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_pos = (self.positions[agent_idx][0] + dx, self.positions[agent_idx][1] + dy)

        if self.battery_levels[agent_idx] < 10 or self.extinguisher_counts[agent_idx] == 0:
            self.positions[agent_idx] = self.base
            self.battery_levels[agent_idx] = e.MAX_BATTERY
            self.extinguisher_counts[agent_idx] = 1
            logging.info(f'Agent {agent_idx} recharged at base')
            return reward, done

        if new_pos in [self.positions[i] for i in range(3) if i != agent_idx]:
            reward = e.OBSTACLE_PENALTY
            logging.info(f'Agent {agent_idx} collision with another agent: {e.OBSTACLE_PENALTY}')
            self.steps_without_progress[agent_idx] += 1
            return reward, done

        if new_pos in self.fires:
            self.fires.remove(new_pos)
            self.extinguisher_counts[agent_idx] -= 1
            self.steps_without_progress[agent_idx] = 0
            reward = e.FIRE_REWARD
            self.positions[agent_idx] = new_pos
            self.update_distances_to_fires()
            logging.info(f'Agent {agent_idx} extinguished fire at {new_pos}: {e.FIRE_REWARD}')
        elif not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            reward = e.OUT_OF_BOUNDS_PENALTY
            logging.info(f'Agent {agent_idx} out of bounds: {e.OUT_OF_BOUNDS_PENALTY}')
            self.steps_without_progress[agent_idx] += 1
        elif new_pos in self.obstacles:
            reward = e.OBSTACLE_PENALTY
            logging.info(f'Agent {agent_idx} hit obstacle: {e.OBSTACLE_PENALTY}')
            self.steps_without_progress[agent_idx] += 1
        else:
            self.positions[agent_idx] = new_pos
            self.update_distances_to_fires()

        logging.info(f'Agent {agent_idx} Position = {self.positions[agent_idx]}')
        return reward, done

    def _get_state(self) -> np.ndarray:
        local_views = [self.get_local_view(i) for i in range(3)]
        state_parts = []
        for i in range(3):
            x, y = self.positions[i]
            base_distances = [x - self.base[0], y - self.base[1]]
            nearest_fire = min(self.fires, key=lambda f: abs(f[0] - x) + abs(f[1] - y)) if self.fires else (0, 0)
            fire_distances = [x - nearest_fire[0], y - nearest_fire[1]]
            state_parts.append(np.array([
                self.battery_levels[i],
                self.extinguisher_counts[i],
            ] + base_distances, dtype=np.float32))
        
        fires_count = np.array([len(self.fires)], dtype=np.float32)
        fire_distances_array = np.array(fire_distances, dtype=np.float32)
        distances_to_fires_array = np.array(
            self.distances_to_fires + [0] * (self.fire_count - len(self.distances_to_fires)),
            dtype=np.float32
        )

        state = np.concatenate(
            state_parts +
            [fires_count, fire_distances_array, distances_to_fires_array] +
            local_views
        )
        return state

    def render(self) -> None:
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
            self.screen.blit(self.images["obstacle"], (obstacle[0] * self.cell_size, obstacle[1] * self.cell_size))
        self.screen.blit(self.images["base"], (self.base[0] * self.cell_size, self.base[1] * self.cell_size))
        for i in range(3):
            self.screen.blit(self.images["agent"], (self.positions[i][0] * self.cell_size, self.positions[i][1] * self.cell_size))
        pygame.display.flip()
        pygame.time.delay(100)

    def close(self) -> None:
        if self.render_mode == "human" and hasattr(self, 'screen'):
            from render.user_interface import show_summary_window
            show_summary_window(self.fire_count, self.fire_count - len(self.fires),
                                self.obstacle_count, self.iteration_count, self.total_reward)
            del self.screen




