from gymnasium.wrappers.normalize import RunningMeanStd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import Optional


class ContinuousEnv:

    def __init__(self,
                 num_envs: int,
                 obs_dim: int,
                 action_dim: int,
                 max_episode_length: float,
                 reward_lb: float,
                 reward_ub: float,
                 action_lb: float = -1,
                 action_ub: float = 1,
                 reward_shape_fn=None,
                 normalize: bool = False,
                 extra_config: Optional[dict] = None,
                 ):

        self.num_envs = num_envs
        self.obs_dim = obs_dim                                  # Shape of the observation
        self.action_dim = action_dim                            # Shape of the action
        self.max_episode_length = max_episode_length            # When to truncate trajectories
        self.reward_lb = reward_lb                              # Reward clipping LB
        self.reward_ub = reward_ub                              # Reward clipping UB
        self.action_lb = action_lb
        self.action_ub = action_ub
        self.reward_shape_fn = reward_shape_fn                  # Function () to compute shaped reward
        self.normalize = normalize                              # Whether to normalize observations
        self.obs_normalizer = RunningMeanStd(shape=self.obs_dim) if normalize else None
        self.discrete_actions = False
        # Save environment's config
        self.env_config = {'name': 'ContEnv', 'obs_dim': obs_dim, 'action_dim': action_dim, 'num_envs': num_envs,
                           'max_episode_length': max_episode_length, 'reward_lb': reward_lb, 'reward_ub': reward_ub,
                           'action_lb': action_lb, 'action_ub': action_ub, 'normalize': normalize
                           }

        if extra_config is not None:
            self.env_config.update(extra_config)
        # Create empty obs and logging dictionaries
        self.obs = np.empty(self.obs_dim)
        self.current_episode_stats = [defaultdict(list) for _ in range(self.num_envs)]
        self.image_metrics = {}
        self.all_obs = self.get_observations_grid()

    def get_observations_grid(self) -> np.ndarray:
        """ Returns observations grid which we use to estimate certain learning metrics, e.g., value maps. """
        raise NotImplementedError

    def get_observation(self) -> np.ndarray:
        """ Returns current observation """
        return self.obs.copy().reshape((self.num_envs, self.obs_dim))

    def normalize_obs(self, obs: np.ndarray, update=False) -> np.ndarray:
        """ Normalizes observation if normalization is on, otherwise keeps it unchanged. """
        if self.normalize:
            if update:
                self.obs_normalizer.update(obs)
            mean, var = self.obs_normalizer.mean, self.obs_normalizer.var
            obs = (obs - mean) / np.sqrt(var + 1e-8)
        return obs

    def _make_value_figure(self, evaluate_value_fn, **kwargs) -> plt.Figure:
        """ Make a matplotlib Figure that will be logged in wandb.
            Example use: value map of the maze """
        raise NotImplementedError

    def compute_shaped_reward(self, reward_true: np.ndarray, **kwargs) -> np.ndarray:
        """ Compute the shaped reward using self.reward_shape_fn """
        if self.reward_shape_fn is not None:
            reward_shaped = self.reward_shape_fn(reward_true=reward_true.copy(), **kwargs, env=self)
            reward_shaped = reward_shaped.clip(self.reward_lb, self.reward_ub)
            return reward_shaped
        else:
            return reward_true.clip(self.reward_lb, self.reward_ub)

    def reset(self) -> None:
        """ Resets the environment to start new episode. Does not return anything. """
        raise NotImplementedError

    def update_image_metrics(self, obs: np.ndarray, next_obs: np.ndarray, trunc: np.ndarray, done: np.ndarray):
        """ Update information in the image metrics.
            Example: visitation counts in the cells of the maze. """
        raise NotImplementedError

    def step(self, action: np.ndarray, evaluate_v_fn=None, disable_logging=False, **kwargs) -> dict:
        """ Applies action, perform transition and return a dictionary with the results """
        raise NotImplementedError

    def close(self):
        """ Used for environment based on Gym. Closes the underlying gym environment. """
        pass

    def render(self) -> None:
        """ Renders the environment """
        raise NotImplementedError


