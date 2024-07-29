from src.environments.continuous_env import ContinuousEnv

from gymnasium.wrappers.record_video import RecordVideo
from typing import Optional
from collections import defaultdict
import gymnasium as gym
import numpy as np


class GymRoboticsContinuousEnv(ContinuousEnv):

    def __init__(self,
                 num_envs: int,
                 env_name: str,
                 normalize: bool = False,
                 reward_lb=-10,
                 reward_ub=10,
                 reward_shape_fn=None,
                 success_key='success',
                 extra_config: Optional[dict] = None,
                 record_video: bool = False,
                 **env_kwargs,
                 ):
        if record_video:
            num_envs = 1
            make_env_fn = lambda: gym.make(env_name, render_mode='rgb_array', **env_kwargs)
            self._single_env = make_env_fn()
            self.gym_env = RecordVideo(self._single_env, video_folder="./saved_videos",
                                       episode_trigger=lambda x: True, disable_logger=True)
        else:
            make_env_fn = lambda: gym.make(env_name, **env_kwargs)
            self._single_env = make_env_fn()
            self.gym_env = gym.vector.SyncVectorEnv([make_env_fn] * num_envs)
        max_episode_steps = self._single_env._max_episode_steps

        # Initialize the parent class
        obs_dim = (self._single_env.observation_space['observation'].shape[-1] +
                   self._single_env.observation_space['desired_goal'].shape[-1])
        action_dim = self._single_env.action_space.shape[-1]
        env_specific_config = {'name': env_name, 'env_name': env_name}
        if extra_config is not None:
            env_specific_config.update(extra_config)
        env_specific_config.update(env_kwargs)
        action_lb = self._single_env.action_space.low
        action_ub = self._single_env.action_space.high
        super().__init__(num_envs, obs_dim, action_dim, max_episode_steps, reward_lb, reward_ub,
                         action_lb, action_ub, reward_shape_fn, normalize, env_specific_config)
        self.success_key = success_key
        self.track_movement = ('Push' in env_name or 'Slide' in env_name) and not record_video

    def get_observations_grid(self):
        return None

    def update_image_metrics(self, state, next_state, trunc, done):
        pass

    def _get_true_next_obs(self, next_obs_dict, info_gym):
        if 'final_observation' in info_gym:
            next_obs_dict_true = {key: [] for key in next_obs_dict}
            final_obs = np.copy(info_gym['final_observation'])
            for env_ind in range(self.num_envs):
                if final_obs[env_ind] is None:
                    obs_to_use = {key: val[env_ind] for key, val in next_obs_dict.items()}
                else:
                    obs_to_use = final_obs[env_ind]
                for key in next_obs_dict:
                    next_obs_dict_true[key].append(obs_to_use[key][None])
            for key in next_obs_dict_true:
                next_obs_dict_true[key] = np.concatenate(next_obs_dict_true[key], axis=0)
        else:
            next_obs_dict_true = next_obs_dict
        next_obs_true = np.concatenate([next_obs_dict_true['observation'],
                                        next_obs_dict_true['desired_goal']], axis=-1)
        return next_obs_true

    def _get_success(self, info_gym):
        if 'final_info' in info_gym:
            return np.array([info_gym['final_info'][ind][self.success_key] if info_gym['final_info'][ind] is not None
                             else info_gym[self.success_key][ind] for ind in range(self.num_envs)]).reshape((self.num_envs, 1))

        else:
            return np.array(info_gym[self.success_key].reshape((self.num_envs, 1)))

    def reset(self, initial_state=None):
        state_dict, _ = self.gym_env.reset()
        self.obs = np.concatenate([state_dict['observation'], state_dict['desired_goal']], axis=-1)
        self.current_episode_stats = [defaultdict(list) for _ in range(self.num_envs)]
        if self.track_movement:
            for i in range(self.num_envs):
                self.current_episode_stats[i]['object_pos'].append(self.obs[i, 3:6])

    def step(self, action, evaluate_value_fn=None, disable_logging=False, **evaluate_value_kwargs):
        next_obs_dict, reward_true, done, trunc, info_gym = self.gym_env.step(action)
        # print('done:', done, 'trunc:', trunc, 'reward:', reward_true, 'info gym', info_gym)
        next_obs = np.concatenate([next_obs_dict['observation'], next_obs_dict['desired_goal']], axis=-1)
        next_obs_true = self._get_true_next_obs(next_obs_dict, info_gym)
        if self.num_envs == 1:
            reward_true = np.reshape(reward_true, (1, 1))
            done = np.reshape(done, (1, 1))
            trunc = np.reshape(trunc, (1, 1))
            next_obs = np.reshape(next_obs, (1, -1))
            next_obs_true = np.reshape(next_obs_true, (1, -1))

        # Save done, truncated, and reward
        terminated_episode_indices = np.where(done | trunc)[0]
        success = self._get_success(info_gym)
        reward_shaped = self.compute_shaped_reward(reward_true=reward_true, obs=self.obs, action=action,
                                                   next_obs=next_obs_true, trunc=trunc, done=done, success=success)
        # Create info dict. All entries of the form episode/category/* will be logged by the agent.
        info = dict()
        info['reward'] = np.reshape(reward_shaped, (self.num_envs, 1))
        info['reward_true'] = np.reshape(reward_true, (self.num_envs, 1))
        info['done'] = np.reshape(done, (self.num_envs, 1))
        info['trunc'] = np.reshape(trunc, (self.num_envs, 1))
        info['success'] = np.reshape(success, (self.num_envs, 1))
        info['next_obs_true'] = np.reshape(next_obs_true, (self.num_envs, self.obs_dim))
        info['terminated_episode_indices'] = terminated_episode_indices
        if disable_logging:
            self.obs = next_obs
            return info
        # Then, write stats of the finished episodes into the statistics and save those of the unfinished
        episode_info = defaultdict(list)
        self.update_image_metrics(self.obs, next_obs, trunc, done)
        for i in range(self.num_envs):
            self.current_episode_stats[i]['reward'].append(info['reward'][i])
            self.current_episode_stats[i]['reward_true'].append(info['reward_true'][i])
            self.current_episode_stats[i]['success'].append(info['success'][i])
            if self.track_movement:
                self.current_episode_stats[i]['object_pos'].append(next_obs_true[i, 3:6])
            if done[i] or trunc[i]:
                episode_info['performance/reward_sum'].append(np.sum(self.current_episode_stats[i]['reward']))
                episode_info['performance/reward_max'].append(np.max(self.current_episode_stats[i]['reward']))
                episode_info['performance/reward_min'].append(np.min(self.current_episode_stats[i]['reward']))
                episode_info['performance/reward_true_sum'].append(np.sum(self.current_episode_stats[i]['reward_true']))
                episode_info['performance/reward_true_max'].append(np.max(self.current_episode_stats[i]['reward_true']))
                episode_info['performance/reward_true_min'].append(np.min(self.current_episode_stats[i]['reward_true']))
                episode_info['performance/episode_length'].append(len(self.current_episode_stats[i]['reward']))
                episode_info['performance/success'].append(
                    np.max(self.current_episode_stats[i]['success']).astype(float))
                episode_info['performance/success_end'].append(
                    self.current_episode_stats[i]['success'][-1][0].astype(float))
                if self.track_movement:
                    obj_pos_delta = np.linalg.norm(self.current_episode_stats[i]['object_pos'][-1] -
                                                   self.current_episode_stats[i]['object_pos'][0])
                    episode_info['performance/object_pos_delta'].append(obj_pos_delta)
                    end_dist_to_goal = np.linalg.norm(self.current_episode_stats[i]['object_pos'][-1] -
                                                      next_obs_true[i, -3:])
                    episode_info['performance/end_dist_to_goal'].append(end_dist_to_goal)
                self.current_episode_stats[i] = defaultdict(list)
                info['episode'] = episode_info

        self.obs = next_obs
        return info

    def close(self):
        self.gym_env.close()

    def render(self):
        return self.gym_env.render()
