from src.environments.continuous_env import ContinuousEnv
from gymnasium.wrappers.record_video import RecordVideo
from src.environments.maze.maze_maps import *
from typing import Optional
from matplotlib.patches import Rectangle, Circle
from collections import defaultdict
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch


SUPPORTED_MAPS = {'sp_maze_single_goal': sp_maze_single_goal,
                  'sp_maze_two_goals': sp_maze_two_goals,
                  'simple_exploration_maze': simple_exploration_maze,
                  'simplest_exploration_maze': simplest_exploration_maze,
                  'hard_exploration_maze': hard_exploration_maze,
                  'pure_exploration_maze': pure_exploration_maze,
                  }

SUPPORTED_ROBOTS = ['point', 'ant']


class GymVectorMazeEnv(ContinuousEnv):

    def __init__(self,
                 num_envs: int,
                 robot: str,
                 maze_map_name: str,
                 reward_lb: float,
                 reward_ub: float,
                 reward_shape_fn=None,
                 continuing_task=True,
                 reset_target=False,
                 normalize: bool = False,
                 extra_config: Optional[dict] = None,
                 record_video: bool = False,
                 save_images_interval: int = 200,
                 slip_prob: float = 0,
                 **env_kwargs
                 ):
        # Check the input arguments
        assert maze_map_name in SUPPORTED_MAPS, 'Choose map from %s' % SUPPORTED_MAPS.keys()
        self.maze_map = SUPPORTED_MAPS[maze_map_name]
        assert robot in SUPPORTED_ROBOTS, 'Choose robot from %s' % SUPPORTED_ROBOTS
        # Create maze environment from gym-robotics
        env_name = '%sMaze_Open-v%d' % (robot[0].upper() + robot[1:], 4 if 'ant' in robot else 3)
        # Create synchronous parallel environment
        self.env_name = env_name
        if record_video:
            raise NotImplementedError
        else:
            make_env_fn = lambda: gym.make(env_name, maze_map=self.maze_map, reset_target=reset_target,
                                           continuing_task=continuing_task, **env_kwargs)
            self._single_env = make_env_fn()
            self.gym_env = gym.vector.SyncVectorEnv([make_env_fn] * num_envs)
            self.maze = self._single_env.maze
            max_episode_steps = self._single_env._max_episode_steps
        # Initialize the parent class
        obs_dim = (self._single_env.observation_space['observation'].shape[-1] +
                   self._single_env.observation_space['desired_goal'].shape[-1])
        action_dim = self._single_env.action_space.shape[-1]
        env_specific_config = {'name': env_name, 'maze_map_name': maze_map_name, 'slippery_chance': slip_prob}
        if extra_config is not None:
            env_specific_config.update(extra_config)
        env_specific_config.update(env_kwargs)
        self.slip_prob = slip_prob
        action_lb = self._single_env.action_space.low
        action_ub = self._single_env.action_space.high
        super().__init__(num_envs, obs_dim, action_dim, max_episode_steps, reward_lb, reward_ub,
                         action_lb, action_ub, reward_shape_fn, normalize, env_specific_config)
        # Create parameters needed for interaction with the environment
        self.save_images_interval = save_images_interval
        self.heat_map_base = np.ones((len(self.maze_map), len(self.maze_map[0]), 3))
        self.wall_inds = [(i, j) for i in range(len(self.maze_map)) for j in range(len(self.maze_map[0]))
                          if self.maze_map[i][j] == 1]
        self.image_metrics = {'visit_count_aggregated':  np.zeros((len(self.maze_map), len(self.maze_map[0]))),
                              'visit_counts_recent':  np.zeros((len(self.maze_map), len(self.maze_map[0]))),
                              'episodes_since_reset': 0,
                              }

    def get_observations_grid(self):
        top_left = self.maze.cell_rowcol_to_xy((0, 0))
        bot_right = self.maze.cell_rowcol_to_xy((self.maze.map_length - 1, self.maze.map_width - 1))
        x_range = np.linspace(top_left[0], bot_right[0], 100)
        y_range = np.linspace(top_left[1], bot_right[1], 100)
        goal_x, goal_y = self.maze.unique_goal_locations[0]
        all_states = []
        for x in x_range:
            for y in y_range:
                i, j = self.maze.cell_xy_to_rowcol((x, y))
                if not self.maze_map[i][j] == 1:
                    all_states.append(np.array([x, y, 0, 0, goal_x, goal_y]))
        all_states = np.stack(all_states)
        all_states[:, -2:] = all_states[:, -2:] + np.random.normal(size=(len(all_states), 2)) / 10
        return all_states

    def _make_value_figure(self, evaluate_value_fn_list, **kwargs):
        all_obs_norm = torch.from_numpy(self.normalize_obs(self.all_obs, update=False).astype(np.float32))
        list_of_figures = []
        for evaluate_value_fn in evaluate_value_fn_list:
            with torch.no_grad():
                value = evaluate_value_fn(all_obs_norm)
            # Dictionary (i,j) coordinate to list of corresponding values
            all_values_dict = defaultdict(list)
            for obs, v_i in zip(self.all_obs, value):
                i, j = self.maze.cell_xy_to_rowcol(obs[:2])
                all_values_dict[(i, j)].append(v_i.cpu().detach().numpy())
            # Average and normalize the values
            for key in all_values_dict.keys():
                all_values_dict[key] = np.mean(all_values_dict[key])
            # Make figure
            fig, ax = plt.subplots(1, 1, figsize=(len(self.maze_map), len(self.maze_map[0])))
            counts = np.zeros((*np.shape(self.maze_map), 3))
            for (i, j), v_ij in all_values_dict.items():
                counts[i, j, :] = v_ij
                ax.text(j - .3, i, '%.2f' % v_ij, c='red', fontsize=12)
            counts = counts - counts.min() + .1
            counts /= (counts.max() + 0)
            for i, j in self.wall_inds:
                counts[i, j] = 0
            ax.imshow(counts)
            list_of_figures.append(fig)
            plt.close(fig)
        return list_of_figures

    def _make_images(self):
        hm_recent = np.ones((len(self.maze_map), len(self.maze_map[0]), 3))
        hm_recent[:, :, 1] = (1 - self.image_metrics['visit_counts_recent'] /
                              self.image_metrics['visit_counts_recent'].sum())
        hm_recent[:, :, 2] = (1 - self.image_metrics['visit_counts_recent'] /
                              self.image_metrics['visit_counts_recent'].sum())
        hm_recent = hm_recent ** 2
        for i, j in self.wall_inds:
            hm_recent[i, j, :] = 0
        """hm_aggregated = np.ones((len(self.maze_map), len(self.maze_map[0]), 3))
        hm_aggregated[:, :, 1] = (1 - self.image_metrics['visit_count_aggregated']
                                  / self.image_metrics['visit_count_aggregated'].sum())
        hm_aggregated[:, :, 2] = (1 - self.image_metrics['visit_count_aggregated']
                                  / self.image_metrics['visit_count_aggregated'].sum())
        hm_aggregated = hm_aggregated ** 2"""
        hm_aggregated = np.ones((len(self.maze_map), len(self.maze_map[0]), 3))
        hm_aggregated[self.image_metrics['visit_count_aggregated'] > 0] = 0.5
        for i, j in self.wall_inds:
            hm_aggregated[i, j, :] = 0
        return {'visit_count_aggregated': hm_aggregated, 'visit_counts_recent': hm_recent}

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
        next_obs_true = np.concatenate([next_obs_dict_true['observation'], next_obs_dict_true['desired_goal']], axis=-1)
        return next_obs_true

    def _get_success(self, info_gym):
        if 'final_info' in info_gym:
            return np.array([info_gym['final_info'][ind]['success'] if info_gym['final_info'][ind] is not None
                             else info_gym['success'][ind] for ind in range(self.num_envs)]).reshape((self.num_envs, 1))

        else:
            return np.array(info_gym['success'].reshape((self.num_envs, 1)))

    def update_image_metrics(self, state, next_state, trunc, done):
        for next_state_i in next_state:
            i, j = self.maze.cell_xy_to_rowcol(next_state_i[:2])
            self.image_metrics['visit_counts_recent'][i, j] += 1
            self.image_metrics['visit_count_aggregated'][i, j] += 1

    def reset(self, initial_state=None):
        state_dict, _ = self.gym_env.reset()
        self.obs = np.concatenate([state_dict['observation'], state_dict['desired_goal']], axis=-1)
        self.current_episode_stats = [defaultdict(list) for _ in range(self.num_envs)]

    def step(self, action, evaluate_value_fn_list=None, **evaluate_value_kwargs):
        if self.slip_prob > 0:
            is_slip = np.random.uniform(0, 1, self.num_envs) <= self.slip_prob
            slip_term = np.random.uniform(self.action_lb, self.action_ub, size=(self.num_envs, self.action_dim))
            action[is_slip] = slip_term[is_slip]

        next_obs_dict, reward_true, done, trunc, info_gym = self.gym_env.step(action)
        # print('done:', done, 'trunc:', trunc, 'reward:', reward_true, 'info gym', info_gym)
        next_obs = np.concatenate([next_obs_dict['observation'], next_obs_dict['desired_goal']], axis=-1)
        next_obs_true = self._get_true_next_obs(next_obs_dict, info_gym)

        # Save done, truncated, and reward
        terminated_episode_indices = np.where(done | trunc)[0]
        success = self._get_success(info_gym)
        reward_shaped = self.compute_shaped_reward(reward_true=reward_true, obs=self.obs, action=action,
                                                   next_obs=next_obs, trunc=trunc, done=done, success=success)
        # Create info dict. All entries of the form episode/category/* will be logged by the agent.
        info = dict()
        info['reward'] = reward_shaped.reshape((self.num_envs, 1))
        info['reward_true'] = reward_true.reshape((self.num_envs, 1))
        info['done'] = done.reshape((self.num_envs, 1))
        info['trunc'] = trunc.reshape((self.num_envs, 1))
        info['success'] = success.reshape((self.num_envs, 1))
        info['next_obs_true'] = next_obs_true.reshape((self.num_envs, self.obs_dim))
        info['terminated_episode_indices'] = terminated_episode_indices

        # Then, write stats of the finished episodes into the statistics and save those of the unfinished
        episode_info = defaultdict(list)
        self.update_image_metrics(self.obs, next_obs, trunc, done)
        for i in range(self.num_envs):
            self.current_episode_stats[i]['reward'].append(info['reward'][i])
            self.current_episode_stats[i]['reward_true'].append(info['reward_true'][i])
            self.current_episode_stats[i]['success'].append(info['success'][i])
            if done[i] or trunc[i]:
                episode_info['performance/reward_sum'].append(np.sum(self.current_episode_stats[i]['reward']))
                episode_info['performance/reward_max'].append(np.max(self.current_episode_stats[i]['reward']))
                episode_info['performance/reward_min'].append(np.min(self.current_episode_stats[i]['reward']))
                episode_info['performance/reward_true_sum'].append(np.sum(self.current_episode_stats[i]['reward_true']))
                episode_info['performance/reward_true_max'].append(np.max(self.current_episode_stats[i]['reward_true']))
                episode_info['performance/reward_true_min'].append(np.min(self.current_episode_stats[i]['reward_true']))
                episode_info['performance/episode_length'].append(len(self.current_episode_stats[i]['reward']))
                episode_info['performance/success'].append(np.max(self.current_episode_stats[i]['success']).astype(float))
                episode_info['performance/success_end'].append(self.current_episode_stats[i]['success'][-1][0].astype(float))
                episode_info['performance/cells_visited'].append(np.sum(self.image_metrics['visit_count_aggregated'] > 0))
                self.current_episode_stats[i] = defaultdict(list)
                self.image_metrics['episodes_since_reset'] += 1
                if self.image_metrics['episodes_since_reset'] % self.save_images_interval == 0:
                    images_dict = self._make_images()
                    for key, img in images_dict.items():
                        episode_info['images/%s' % key].append(img)
                    self.image_metrics['episodes_since_reset'] = 0
                    self.image_metrics['visit_counts_recent'] *= 0
                    # If evaluate_v function is provided, make an image of the value map and save it in info
                    if evaluate_value_fn_list is not None:
                        list_of_figures = self._make_value_figure(evaluate_value_fn_list, **evaluate_value_kwargs)
                        for fig_ind, fig in enumerate(list_of_figures):
                            episode_info['images/value_map_%s' % fig_ind].append(fig)
                info['episode'] = episode_info

        self.obs = next_obs
        return info

    def close(self):
        self.gym_env.close()

    def render(self):
        return self.gym_env.render()


    def plot_maze_map(self, waypoints_dict=None, figsize=(7, 7)):
        maze_map = self.maze.maze_map
        X, Y = len(maze_map), len(maze_map[0])
        x_min, y_max = self.maze.cell_rowcol_to_xy((0, 0))
        x_max, y_min = self.maze.cell_rowcol_to_xy((X - 1, Y - 1))
        diff_x = x_max - x_min
        diff_y = y_max - y_min

        x_lims = (x_min - diff_x / 5, x_max + diff_x / 5)
        y_lims = (y_min - diff_y / 5, y_max + diff_y / 5)

        fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.add_patch(Rectangle((x_min, y_min), diff_x, diff_y,
                               facecolor='white', alpha=1, edgecolor='black', linewidth=0))
        ax.set_facecolor('#CAA5C4')
        cell_size = self.maze.maze_size_scaling
        for i in range(X):
            for j in range(Y):
                x, y = self.maze.cell_rowcol_to_xy((i, j))
                if waypoints_dict is not None and (i, j) in waypoints_dict and waypoints_dict[(i, j)] is not None:

                    ax.text(x - cell_size * 0.8, y - cell_size * 0.8, '%.4f' % waypoints_dict[(i, j)], zorder=2,
                            fontsize=8)

                if maze_map[i][j] == 1:
                    ax.add_patch(Rectangle((x - cell_size, y - cell_size), 1 * cell_size, 1 * cell_size,
                                           facecolor='#CAA5C4', alpha=1, edgecolor='black', linewidth=1))
                elif maze_map[i][j] == 0:
                    ax.add_patch(Rectangle((x - cell_size, y - cell_size), 1 * cell_size, 1 * cell_size,
                                           facecolor='white', alpha=1, edgecolor='black', linewidth=1))
                elif maze_map[i][j] == 'g':
                    ax.add_patch(Circle((x - cell_size/2, y-cell_size/2), 0.2 * cell_size,
                                        facecolor='red', alpha=1, edgecolor='black', linewidth=0.2))

                elif maze_map[i][j] == 'r':
                    ax.add_patch(Circle((x - cell_size/2, y-cell_size/2), 0.2 * cell_size,
                                        facecolor='green', alpha=1, edgecolor='black', linewidth=0.2))
        return fig, ax

