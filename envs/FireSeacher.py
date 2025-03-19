from abc import ABC, abstractmethod

from envs.BaseScenario import BaseScenario


class FireSeacher(BaseScenario, ABC):
    def __init__(self):
        pass

    def reset(self, *, seed=None, options=None):
        pass
        # return obs, {}

    @abstractmethod
    def _reset_scenario(self):
        pass

    def get_observation(self):
        obs = self._get_scenario_obs()
        return obs

    @abstractmethod
    def _get_scenario_obs(self):
        pass

    def step(self, action):
        pass
        # return obs, self.step_reward, terminated, truncated, info

    def render(self):
        pass

    # def __repr__(self):
    #     return f'scenario_{self.name}'
