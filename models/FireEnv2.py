import random
import numpy as np
import pygame

from gymnasium import Env, spaces
from constants.colors import GREEN
from constants.grid import CELL_SIZE, GRID_SIZE, TREE_PERCENT
from constants.agent import AGENT_VIEW
from envs.Fire import Fire
from envs.reward_sys_sc2 import rewards, penalties
from render.load_images import load_images


CENTER_DIRECTION_COEFFICIENT = 0.4
FIRE_INTIMACY_COEFFICIENT = 0.05
RETURN_BASE_COEFFICIENT = 0.07


class FireEnv2(Env):

    def __init__(self, fire_count: int = None, obstacle_count: int = None,
                 render_mode: str = None) -> None:
        super().__init__()

        self.cell_size = CELL_SIZE
        self.grid_size = GRID_SIZE
        self.screen_size = self.grid_size * self.cell_size
        self.screen = None
        self.render_mode = render_mode
        self.images = load_images(self.cell_size)

        self.num_agents = 3
        self.agent_positions = [(i, (self.grid_size - 1)) for i in range(self.num_agents)]
        self.base_agent_positions = self.agent_positions.copy()
        self.num_goals = fire_count
        self.num_obstacles = obstacle_count
        self.max_steps = self._calculate_max_steps()
        self.local_view_size = AGENT_VIEW

        self.fire = Fire(self)
        self.grid = None
        self.fires = None
        self.obstacles = None
        self.trees = None
        self.burned = set()

        self.iteration_count = 0
        self.agent_steps_count = [0] * self.num_agents
        self.active_goals = 0
        self.visited_cells = set()

        self.observation_space = spaces.Box(low = -1,
                                            high = 1,
                                            shape = (self.num_agents,
                                                     self.local_view_size,
                                                     self.local_view_size, 6),
                                            dtype = np.float32)

        self.action_space = spaces.MultiDiscrete([4, 4, 4])


    def reset(self, seed = None, options = None) -> tuple:
        super().reset(seed = seed)
        self.fire.reset()
        self.fires = self.fire.goals
        self.burned.clear()
        self._generate_grid()
        self.iteration_count = 0
        self.active_goals = len(self.fires)
        self.agent_positions = self.base_agent_positions.copy()
        self.agent_steps_count = [0] * self.num_agents
        self.visited_cells.clear()
        obs = np.array([self._get_local_obs(pos) for pos in self.agent_positions],
                       dtype = np.float32)
        return obs, {}


    def step(self, actions: np.ndarray) -> tuple:
        reward = 0
        terminated = False
        truncated = False
        info = {}

        self.iteration_count = max(self.agent_steps_count)

        agents_rewards_list = [[] for _ in range(self.num_agents)]

        for agent_id, action in enumerate(actions):
            x, y = self.agent_positions[agent_id]
            direction_of_x = self.grid[x, y, 4]
            direction_of_y = self.grid[x, y, 5]

            direction_bonus = 0

            prev_new_x, prev_new_y = x, y
            movement_on_x = 0
            movement_on_y = 0

            if action == 0:
                prev_new_y -= 1
                movement_on_y = -1
            elif action == 1:
                prev_new_y += 1
                movement_on_y = 1
            elif action == 2:
                prev_new_x -= 1
                movement_on_x = -1
            elif action == 3:
                prev_new_x += 1
                movement_on_x = 1

            current_agent_rewards = []

            direction_bonus += ((movement_on_x * direction_of_x)
                                + (movement_on_y * direction_of_y))
            reward += direction_bonus * CENTER_DIRECTION_COEFFICIENT
            current_agent_rewards.append("CENTER_DIRECTION_BONUS")

            new_x, new_y = np.clip((prev_new_x, prev_new_y),
                                   0, (self.grid_size - 1))

            if (new_x, new_y) in self.obstacles:
                reward += penalties["obstacle"]
                current_agent_rewards.append("OBSTACLE_PENALTY")
                info["Collision"] = True
                new_x, new_y = x, y

            if (new_x, new_y) in [self.agent_positions[i] for i in range(self.num_agents)
                                   if i != agent_id]:
                reward += penalties["crash"]
                current_agent_rewards.append("CRASH_PENALTY")
                info["Collision"] = True
                new_x, new_y = x, y

            self.grid[x, y, 0] = 0
            self.agent_positions[agent_id] = (new_x, new_y)
            self.grid[new_x, new_y, 0] = 1
            self.agent_steps_count[agent_id] += 1

            if (new_x, new_y) not in self.visited_cells:
                reward += self._exploration_bonus()
                self.visited_cells.add((new_x, new_y))
                current_agent_rewards.append("EXPLORATION_BONUS")
            else:
                reward += self._revisit_penalty()
                current_agent_rewards.append("REVISIT_PENALTY")

            if (self.grid[new_x, new_y, 3] < (self._average_distance_to_fire_center() * 0.5)
                    and new_x != x and new_y != y):
                reward += (1 - self.grid[new_x, new_y, 3]) * FIRE_INTIMACY_COEFFICIENT
                current_agent_rewards.append("FIRE_INTIMACY_BONUS")

            goal_reward = self._check_agent_has_achieved_goal(agent_id)
            if goal_reward != 0.0:
                reward += goal_reward
                current_agent_rewards.append("FIRE_REWARD")
                info["The goal has been achieved"] = True

            agents_rewards_list[agent_id].append(current_agent_rewards or [""])

        info["agents_rewards_list"] = agents_rewards_list

        if self.active_goals == 0:
            if self.iteration_count <= (self.max_steps * 0.4):
                reward += rewards["final"]
            else:
                reward += rewards["final"] * 0.5
            info["All goals have been achieved"] = True
            terminated = True

        if self.iteration_count >= self.max_steps:
            reward += penalties["not_done"]
            info["Exceeded the maximum possible number of steps"] = True
            truncated = True

        obs = np.array([self._get_local_obs(pos) for pos in self.agent_positions],
                       dtype = np.float32)

        return obs, reward, terminated, truncated, info


    def render(self) -> None:
        houses_margin = int(self.grid_size * 0.1)
        if self.render_mode != "human":
            return
        if self.screen is None:
            pygame.init()
            size = self.screen_size + (houses_margin * self.cell_size)
            self.screen = pygame.display.set_mode((self.screen_size, size))
            pygame.display.set_caption("Fire Fighter")

        self.screen.fill(GREEN)

        for tree in self.trees:
            self.screen.blit(self.images["tree"], (tree[0] * self.cell_size,
                                                   tree[1] * self.cell_size))

        for base in self.base_agent_positions:
            self.screen.blit(self.images["base"], (base[0] * self.cell_size,
                                                   base[1] * self.cell_size))

        for obs in self.obstacles:
            self.screen.blit(self.images["obstacle"], (obs[0] * self.cell_size,
                                                       obs[1] * self.cell_size))

        self.screen.blit(self.images["burned"], (self.fire.center_x * self.cell_size,
                                                 self.fire.center_y * self.cell_size))

        for b in self.burned:
            self.screen.blit(self.images["burned"], (b[0] * self.cell_size,
                                                     b[1] * self.cell_size))

        for fire in self.fires:
            self.screen.blit(self.images["fire"], (fire[0] * self.cell_size,
                                                   fire[1] * self.cell_size))

        for agent in self.agent_positions:
            self.screen.blit(self.images["agent"], (agent[0] * self.cell_size,
                                                    agent[1] * self.cell_size))

        for i in range(0, self.grid_size + houses_margin, 2):
            for j in range(self.grid_size, self.grid_size + houses_margin, 2):
                self.screen.blit(self.images["houses"], (i * self.cell_size, j * self.cell_size))

        pygame.display.flip()
        pygame.time.delay(100)


    def render_airplane(self):
        houses_margin = int(self.grid_size * 0.1)
        if self.render_mode != "human":
            return
        if self.screen is None:
            pygame.init()
            size = self.screen_size + (houses_margin * self.cell_size)
            self.screen = pygame.display.set_mode((self.screen_size, size))
            pygame.display.set_caption("Fire Fighter")

        while len(self.fire.fire_cells) != len(self.fires):
            self.screen.fill(GREEN)

            for tree in self.trees:
                self.screen.blit(self.images["tree"], (tree[0] * self.cell_size,
                                                       tree[1] * self.cell_size))

            for base in self.base_agent_positions:
                self.screen.blit(self.images["base"], (base[0] * self.cell_size,
                                                       base[1] * self.cell_size))

            for obs in self.obstacles:
                self.screen.blit(self.images["obstacle"], (obs[0] * self.cell_size,
                                                           obs[1] * self.cell_size))

            for i in range(0, self.grid_size + houses_margin, 2):
                for j in range(self.grid_size, self.grid_size + houses_margin, 2):
                    self.screen.blit(self.images["houses"], (i * self.cell_size, j * self.cell_size))

            for fire in self.fire.fire_cells:
                self.screen.blit(self.images["fire"], (fire[0] * self.cell_size,
                                                       fire[1] * self.cell_size))

            aircraft_position = random.choice(list(self.fire.fire_cells))

            if aircraft_position not in self.fires:
                self.fire.fire_cells -= {aircraft_position}
                self.burned.add(aircraft_position)

            for b in self.burned:
                self.screen.blit(self.images["burned"], (b[0] * self.cell_size,
                                                           b[1] * self.cell_size))

            self.screen.blit(self.images["aircraft"], (aircraft_position[0] * self.cell_size,
                                                       aircraft_position[1] * self.cell_size))

            pygame.display.flip()
            pygame.time.delay(300)


    def stop(self) -> None:
        if self.screen is not None:
            pygame.quit()


    def _calculate_max_steps(self) -> int:
        """The maximum number of steps of the agent is approximately taken in accordance
           with the battery consumption of the drone. In 1 step, the battery charge is
           reduced by 0.6 %. Drone should have 20 % of its battery charge remaining when
           returning to base."""
        return int((self.grid_size ** 2) / 0.6 * 0.8)


    def _exploration_bonus(self) -> float:
        """Reward for visiting new cells. Increases in proportion to the discovery of new goals"""
        base_bonus = rewards["new_step"]
        if len(self.fires) == 0:
            return base_bonus
        goal_ratio = self.active_goals / len(self.fires)
        return base_bonus * (1 + goal_ratio)


    def _revisit_penalty(self) -> float:
        """Penalty for repeated visits to the cells. Decreases in proportion to the discovery
           of new goals"""
        base_penalty = penalties["repeat_step"]
        if len(self.fires) == 0:
            return base_penalty
        goal_ratio = self.active_goals / len(self.fires)
        return base_penalty * (1 - goal_ratio)


    def _average_distance_to_fire_center(self) -> np.array:
        """Calculates the average normalized Manhattan distance from each grid cell
           to the center of the fire"""
        return np.mean(self.grid[:, :, 3])


    def _get_local_obs(self, agent_pos: tuple) -> np.array:
        """Returns the agent's local view scope"""
        x, y = agent_pos
        obs = np.zeros((self.local_view_size, self.local_view_size, 6),
                       dtype = np.float32)

        half_view = self.local_view_size // 2
        x_min = max(x - half_view, 0)
        x_max = min(x + half_view + 1, self.grid_size)
        y_min = max(y - half_view, 0)
        y_max = min(y + half_view + 1, self.grid_size)

        x_offset = half_view - (x - x_min)
        y_offset = half_view - (y - y_min)

        obs[x_offset:x_offset + (x_max - x_min),
            y_offset:y_offset + (y_max - y_min)] = self.grid[x_min:x_max, y_min:y_max]

        return obs


    def _generate_grid(self) -> None:
        self.grid = np.zeros((self.grid_size, self.grid_size, 6), dtype = np.float32)

        for _, (agent_x, agent_y) in enumerate(self.base_agent_positions):
            self.grid[agent_x, agent_y, 0] = 1

        for (goal_x, goal_y) in self.fires:
            self.grid[goal_x, goal_y, 1] = 1

        available_cells = [
            (i, j) for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) not in self.fires and
               (i, j) != (self.fire.center_x, self.fire.center_y) and
               (i, j) not in set(self.base_agent_positions)
        ]

        selected_obstacles = self.np_random.choice(available_cells,
                                                   size = self.num_obstacles,
                                                   replace = False)
        self.obstacles = set(map(tuple, selected_obstacles))
        for (obs_x, obs_y) in self.obstacles:
            self.grid[obs_x, obs_y, 2] = 1

        tree_count = int((self.grid_size ** 2 - self.num_goals - self.num_obstacles
                          - len(self.base_agent_positions)) * TREE_PERCENT)
        available_cells_trees = [cell for cell in available_cells if cell
                                 not in self.obstacles]
        selected_trees = self.np_random.choice(available_cells_trees,
                                               size = tree_count, replace = False)
        self.trees = set(map(tuple, selected_trees))

        x_array, y_array = np.indices((self.grid_size, self.grid_size))

        manhattan_distance_array = (np.abs(x_array - self.fire.center_x)
                                    + np.abs(y_array - self.fire.center_y))
        normalized_distance_array = manhattan_distance_array / (2 * (self.grid_size - 1))
        self.grid[:, :, 3] = normalized_distance_array

        direction_of_x = self.fire.center_x - x_array
        direction_of_y = self.fire.center_y - y_array
        normalized_direction_of_x = direction_of_x / (self.grid_size - 1)
        normalized_direction_of_y = direction_of_y / (self.grid_size - 1)
        self.grid[:, :, 4] = normalized_direction_of_x
        self.grid[:, :, 5] = normalized_direction_of_y


    def _check_agent_has_achieved_goal(self, agent_id: int) -> float:
        """The method checks whether the agent has reached the goal. If the agent has reached
           the goal, he returns to the base to replenish the fire extinguishing agent.
           The return-to-base function is embedded in the drone's software. As an assumption,
           we assume that the drone's position becomes equal to the base's position. At the same
           time, the number of steps taken increases. The coefficient of 0.07 was obtained under
           the following conditions:
           - 10x10 grid of cells,
           - base position (2, 2),
           - 8 arbitrary grid cells were taken as goals, the number of steps from each goal
           to the base by the shortest path without obstacles was calculated, and the average
           value of the steps was taken."""
        if self.agent_positions[agent_id] in self.fires:
            self.grid[self.agent_positions[agent_id][0], self.agent_positions[agent_id][1], 0] = 0
            self.fires.remove(self.agent_positions[agent_id])
            self.active_goals -= 1

            self.agent_positions[agent_id] = self.base_agent_positions[agent_id]
            self.agent_steps_count[agent_id] += int((self.grid_size ** 2)
                                                    * RETURN_BASE_COEFFICIENT)

            return rewards["fire"]
        else:
            return 0.0
