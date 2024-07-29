from src.environments.continuous_env import ContinuousEnv

from typing import Optional
from collections import defaultdict
import gymnasium as gym

import numpy as np


class GymContinuousEnv(ContinuousEnv):

    def __init__(self,
                 num_envs: int,
                 env_name: str,
                 normalize: bool = False,
                 reward_lb=-10,
                 reward_ub=10,
                 reward_shape_fn=None,
                 extra_config: Optional[dict] = None,
                 record_video: bool = False,
                 **env_kwargs

                 ):
        # Create asynchronous parallel environment
        self.env_name = env_name
        # gym_env = gym.make(env_name, render_mode='rgb_array', **env_kwargs)
        # self._env = RecordVideo(gym_env, video_folder="./saved_videos",
        #                        episode_trigger=lambda x: True, disable_logger=True)
        # max_episode_steps = gym_env._max_episode_steps
        if record_video:
            raise NotImplementedError
        else:
            make_env_fn = lambda: gym.make(env_name, **env_kwargs)
            self._single_env = make_env_fn()
            self._env = gym.vector.SyncVectorEnv([make_env_fn] * num_envs)
            max_episode_steps = self._single_env._max_episode_steps

        # Create reward shaper
        self.reward_shaper = None
        # Initialize the parent class
        obs_dim = self._single_env.observation_space.shape[-1]
        action_dim = self._single_env.action_space.shape[-1]
        env_specific_config = {'name': env_name}
        if extra_config is not None:
            env_specific_config.update(extra_config)
        env_specific_config.update(env_kwargs)

        action_lb = self._env.action_space.low
        action_ub = self._env.action_space.high
        super().__init__(num_envs, obs_dim, action_dim, max_episode_steps,
                         reward_lb, reward_ub, action_lb, action_ub,
                         reward_shape_fn, normalize, env_specific_config)
        # Create parameters needed for interaction with the environment

    def get_observations_grid(self):
        return None

    def update_image_metrics(self, state, next_state, trunc, done):
        pass

    def reset(self):
        state, _ = self._env.reset()
        self.obs = state
        self.current_episode_stats = [defaultdict(list) for _ in range(self.num_envs)]

    def _get_true_next_obs(self, next_obs_np, info_gym):
        """
            When the episode is finished in a single env, the SyncVectorEnv class resets it automatically.
            We need the true next observation for learning.
        """
        next_obs_true = np.copy(next_obs_np)
        if 'final_observation' in info_gym:
            final_obs = info_gym['final_observation']
            for env_ind in range(self.num_envs):
                if final_obs[env_ind] is not None:
                    next_obs_true[env_ind] = final_obs[env_ind]
        return next_obs_true

    def step(self, action, evaluate_value_fn=None, **evaluate_value_kwargs):
        next_obs, reward_true, done, trunc, info_gym = self._env.step(action)
        next_obs_true = self._get_true_next_obs(next_obs, info_gym)
        # Save done, truncated, and reward
        terminated_episode_indices = np.where(done | trunc)[0]
        reward_shaped = self.compute_shaped_reward(reward_true=reward_true, obs=self.obs, action=action,
                                                   next_obs=next_obs, trunc=trunc, done=done)
        # Create info dict. All entries of the form episode/category/* will be logged by the agent.
        info = dict()
        info['reward'] = reward_shaped.reshape((self.num_envs, 1))
        info['reward_true'] = reward_true.reshape((self.num_envs, 1))
        info['done'] = done.reshape((self.num_envs, 1))
        info['trunc'] = trunc.reshape((self.num_envs, 1))
        info['next_obs_true'] = next_obs_true.reshape((self.num_envs, self.obs_dim))
        info['terminated_episode_indices'] = terminated_episode_indices

        # Then, write stats of the finished episodes into the statistics and save those of the unfinished
        episode_info = defaultdict(list)
        self.update_image_metrics(self.obs, next_obs, trunc, done)
        for i in range(self.num_envs):
            self.current_episode_stats[i]['reward'].append(info['reward'][i])
            self.current_episode_stats[i]['reward_true'].append(info['reward_true'][i])
            if done[i] or trunc[i]:
                episode_info['performance/reward_sum'].append(np.sum(self.current_episode_stats[i]['reward']))
                episode_info['performance/reward_max'].append(np.max(self.current_episode_stats[i]['reward']))
                episode_info['performance/reward_min'].append(np.min(self.current_episode_stats[i]['reward']))
                episode_info['performance/reward_true_sum'].append(np.sum(self.current_episode_stats[i]['reward_true']))
                episode_info['performance/reward_true_max'].append(np.max(self.current_episode_stats[i]['reward_true']))
                episode_info['performance/reward_true_min'].append(np.min(self.current_episode_stats[i]['reward_true']))
                episode_info['performance/episode_length'].append(len(self.current_episode_stats[i]['reward']))
                self.current_episode_stats[i] = defaultdict(list)
                info['episode'] = episode_info

        self.obs = next_obs
        return info

    def close(self):
        self._env.close()

    def render(self):
        return self._env.render()
