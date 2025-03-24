import random

import numpy as np
import pygame
import envs
import constants.colors as colors
from envs.Fire import Fire

from render.load_images import load_images
from gymnasium import Env, spaces


class FireEnv2(Env):

    def __init__(self, fire_count: int = None, obstacle_count: int = None, render_mode: str = None) -> None:
        super().__init__()

        self.cell_size = envs.CELL_SIZE
        self.grid_size = envs.GRID_SIZE
        self.screen_size = self.grid_size * self.cell_size
        self.screen = None
        self.render_mode = render_mode
        self.images = load_images(self.cell_size)

        self.num_agents = 3
        self.agent_positions = [(0, 18), (1, 19), (0, 19)]
        self.base_agent_positions = self.agent_positions.copy()
        self.num_goals = fire_count
        self.num_obstacles = obstacle_count
        self.max_steps = self._calculate_max_steps()

        self.fire = Fire(self)
        self.grid = None
        self.fires = None
        self.obstacles = None
        self.trees = None

        self.iteration_count = 0
        self.agent_steps_count = [0] * self.num_agents
        self.active_goals = 0
        self.visited_cells = set()

        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(self.grid_size, self.grid_size, 4),
                                            dtype=np.uint8)

        self.action_space = spaces.MultiDiscrete([4, 4, 4])

    def reset(self, seed=None, options=None) -> tuple:
        super().reset(seed=seed)
        self.fire.reset()
        self.fires = self.fire.goals
        self._generate_grid()
        self.iteration_count = 0
        self.active_goals = len(self.fires)
        self.agent_positions = self.base_agent_positions.copy()
        self.agent_steps_count = [0] * self.num_agents
        self.visited_cells.clear()
        return self.grid.copy(), {}

    def step(self, actions: np.ndarray) -> tuple:
        reward = 0
        terminated = False
        truncated = False
        info = {}

        self.iteration_count = max(self.agent_steps_count)

        for agent_id, action in enumerate(actions):
            x, y = self.agent_positions[agent_id]
            prev_new_x, prev_new_y = x, y

            if action == 0:
                prev_new_y -= 1
            elif action == 1:
                prev_new_y += 1
            elif action == 2:
                prev_new_x -= 1
            elif action == 3:
                prev_new_x += 1

            new_x, new_y = np.clip((prev_new_x, prev_new_y),
                                   0, (self.grid_size - 1))

            self.grid[x, y, 2] = 0
            self.agent_positions[agent_id] = (new_x, new_y)
            self.grid[new_x, new_y, 2] = 1
            self.agent_steps_count[agent_id] += 1

            if (new_x, new_y) not in self.visited_cells:
                reward += self._exploration_bonus()
                self.visited_cells.add((new_x, new_y))
            else:
                reward += self._revisit_penalty()

            if (new_x, new_y) in self.obstacles:
                reward += envs.OBSTACLE_PENALTY
                info["Collision"] = True

            if (new_x, new_y) in self.agent_positions:
                reward += envs.CRASH_PENALTY
                info["Collision"] = True

            goal_reward = self._check_agent_has_achieved_goal(agent_id)
            if goal_reward != 0.0:
                reward += goal_reward
                info["The goal has been achieved"] = True

        if self.active_goals == 0:
            reward += envs.FINAL_REWARD
            info["All goals have been achieved"] = True
            terminated = True

        if self.iteration_count >= self.max_steps:
            info["Exceeded the maximum possible number of steps"] = True
            truncated = True

        return self.grid.copy(), reward, terminated, truncated, info

    def render(self) -> None:
        houses_margin = int(self.grid_size * 0.1)
        if self.render_mode != "human":
            return
        if self.screen is None:
            pygame.init()
            size = self.screen_size + (houses_margin * self.cell_size)
            self.screen = pygame.display.set_mode((self.screen_size, size))
            pygame.display.set_caption("Fire Fighter")

        self.screen.fill(colors.GREEN)

        for tree in self.trees:
            self.screen.blit(self.images["tree"], (tree[0] * self.cell_size,
                                                   tree[1] * self.cell_size))

        for base in self.base_agent_positions:
            self.screen.blit(self.images["base"], (base[0] * self.cell_size,
                                                   base[1] * self.cell_size))

        for agent in self.agent_positions:
            self.screen.blit(self.images["agent"], (agent[0] * self.cell_size,
                                                    agent[1] * self.cell_size))

        for obs in self.obstacles:
            self.screen.blit(self.images["obstacle"], (obs[0] * self.cell_size,
                                                       obs[1] * self.cell_size))

        self.screen.blit(self.images["burned"], (self.fire.center_x * self.cell_size,
                                                 self.fire.center_y * self.cell_size))

        for fire in self.fires:
            self.screen.blit(self.images["fire"], (fire[0] * self.cell_size,
                                                   fire[1] * self.cell_size))

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
            self.screen.fill(colors.GREEN)
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
            self.screen.blit(self.images["aircraft"], (aircraft_position[0] * self.cell_size,
                                                       aircraft_position[1] * self.cell_size))
            if aircraft_position not in self.fires:
                self.fire.fire_cells -= {aircraft_position}
            pygame.display.flip()
            pygame.time.delay(300)

    def stop(self) -> None:
        if self.screen is not None:
            pygame.quit()

    def _calculate_max_steps(self) -> int:
        """The maximum number of steps of the agent is approximately taken in accordance
           with the battery consumption of the drone. The coefficient 1.6 is obtained under
           the following assumptions:
           - a 10x10 grid of cells,
           - in 1 step, the battery charge is reduced by 0.5 %,
           - drone should have 20 % of its battery charge remaining when returning to base."""
        return int((self.grid_size ** 2) * 1.6)

    def _exploration_bonus(self) -> float:
        """Reward for visiting new cells. Increases in proportion to the discovery of new goals"""
        base_bonus = envs.NEW_STEP_REWARD
        if len(self.fires) == 0:
            return base_bonus
        goal_ratio = self.active_goals / len(self.fires)
        return base_bonus * (1 + goal_ratio)

    def _revisit_penalty(self) -> float:
        """Penalty for repeated visits to the cells. Decreases in proportion to the discovery
           of new goals"""
        base_penalty = envs.REPEAT_STEP_PENALTY
        if len(self.fires) == 0:
            return base_penalty
        goal_ratio = self.active_goals / len(self.fires)
        return base_penalty * (1 - goal_ratio)

    def _generate_grid(self) -> None:
        """Generates an environment with an agent and a random location of the central cell of
           goals, goals and obstacles"""
        self.grid = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.uint8)

        for _, (agent_x, agent_y) in enumerate(self.base_agent_positions):
            self.grid[agent_x, agent_y, 2] = 1

        self.grid[self.fire.center_x, self.fire.center_y, 3] = 1
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
                                                   size=self.num_obstacles, replace=False)
        self.obstacles = set(map(tuple, selected_obstacles))
        for (obs_x, obs_y) in self.obstacles:
            self.grid[obs_x, obs_y, 0] = 1

        tree_count = int((self.grid_size ** 2 - self.num_goals - self.num_obstacles
                          - len(self.base_agent_positions)) * envs.TREE_PERCENT)
        available_cells_trees = [cell for cell in available_cells if cell
                                 not in self.obstacles]
        selected_trees = self.np_random.choice(available_cells_trees,
                                               size=tree_count, replace=False)

        self.trees = set(map(tuple, selected_trees))
        for (tree_x, tree_y) in self.trees:
            self.grid[tree_x, tree_y, 3] = 1

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
            self.grid[self.agent_positions[agent_id][0], self.agent_positions[agent_id][1], 1] = 0
            self.fires.remove(self.agent_positions[agent_id])
            self.active_goals -= 1

            self.agent_positions[agent_id] = self.base_agent_positions[agent_id]
            self.agent_steps_count[agent_id] += int((self.grid_size ** 2) * 0.07)

            return envs.FIRE_REWARD
        else:
            return 0.0
