from src.environments.continuous_env import ContinuousEnv
from src.memory.replay_buffer import ReplayBuffer
from src.noise.pink_noise import PinkNoiseProcess
from src.noise.white_noise import WhiteNoiseProcess
from src.rl_agents.base_agent import Agent

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Callable
import numpy as np
import torch
import wandb
from copy import deepcopy
import time


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContinuousOffPolicyAgent(Agent):

    def __init__(self,
                 make_env_fn: Callable[[bool, ], ContinuousEnv],
                 hidden_dims_list: list[int],
                 wandb_project_name: str,
                 wandb_run_name: str,
                 env_steps: int = int(1e6),
                 test_each: int = 20000,
                 n_test_episodes: int = 20,
                 n_evaluation_episodes: int = 100,
                 discount: float = 0.99,
                 start_training_after: int = int(25e3),
                 train_batch_size: int = 200,
                 memory_size: int = int(1e6),
                 train_freq: int = 1,
                 action_noise_type: str = 'white',
                 action_noise_scale: float = 0.1,
                 target_action_noise_clip: float = 0.5,
                 wandb_tags: Optional[list[str]] = None,
                 use_wandb: bool = True,
                 extra_config: Optional[dict] = None,
                 **kwargs):
        # Save parameters
        self.env_test = None
        self.test_each = test_each
        self.n_test_episodes = n_test_episodes
        self.n_evaluation_episodes = n_evaluation_episodes
        self.start_training_after = start_training_after
        self.train_batch_size = train_batch_size
        self.memory_size = memory_size
        self.train_freq = train_freq
        self.action_noise_type = action_noise_type
        self.action_noise_scale = action_noise_scale
        self.target_action_noise_clip = target_action_noise_clip
        # Make configs and prepare logging
        agent_specific_config = {'name': 'OffPolicyAgent',
                                 'train_batch_size': train_batch_size, 'memory_size': memory_size,
                                 'start_training_after': start_training_after, 'train_freq': train_freq,
                                 'action_noise_type': action_noise_type, 'action_noise_scale': action_noise_scale,
                                 'action_noise_clip': target_action_noise_clip,
                                 }
        agent_specific_config.update(extra_config)
        super().__init__(make_env_fn, hidden_dims_list, wandb_project_name, wandb_run_name, env_steps, discount,
                         wandb_tags, use_wandb, agent_specific_config, **kwargs)
        self._initialize_exploration_noise()


    def _initialize_progress_variables(self):
        super()._initialize_progress_variables()
        self.next_train_at = self.start_training_after
        self.since_last_train = 0
        self.next_test_at = int(self.test_each)
        self.learning_results = defaultdict(list)

    @property
    def _is_initial_exploration(self):
        return self.global_t <= self.start_training_after

    def _create_memory(self):
        """ Create replay buffer (memory for off policy algorithms) """
        obs_dim, a_dim = (self.env.obs_dim,), (self.env.action_dim,)
        self.memory = ReplayBuffer(batch_size=self.num_envs,
                                   scheme={'obs': obs_dim, 'a': a_dim, 'r': (1,),
                                           'done': (1,), 'trunc': (1,), 'obs_next': obs_dim, },
                                   max_size=self.memory_size, min_size_to_sample=self.start_training_after)

    @abstractmethod
    def _initialize_exploration_noise(self):
        if self.action_noise_type == 'white':
            self.noise_process = WhiteNoiseProcess(self.num_envs, self.action_dim, self.action_noise_scale,
                                                   self.env.max_episode_length, noise_clip=None)
        elif self.action_noise_type == 'pink':
            self.noise_process = PinkNoiseProcess(self.num_envs, self.action_dim, self.action_noise_scale,
                                                  self.env.max_episode_length, noise_clip=None)
        else:
            raise ValueError(self.action_noise_type)

    @abstractmethod
    def _create_networks(self, ):
        """ Create actor and critic networks and the Adam optimizer """
        raise NotImplementedError

    def _save_transition(self, obs: np.ndarray, action: np.ndarray,  obs_next: np.ndarray, info: dict[str, np.ndarray]):
        """ Saves transition data into the rollout buffer """
        transition_dict = {'obs': obs, 'a': action, 'r': info['reward'],
                           'done': info['done'], 'trunc': info['trunc'], 'obs_next': obs_next, }
        transition_dict = {key: torch.from_numpy(val.astype(np.float32)) for key, val in transition_dict.items()}
        self.memory.add_transition(transition_dict)

    @abstractmethod
    def choose_action(self, observation: torch.Tensor, noisy: bool, **kwargs) -> torch.Tensor:
        """ Sample an action from the stochastic policy """
        raise NotImplementedError

    def evaluate_v(self, observation, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _learn(self) -> dict:
        """ Sample data from the rollout buffer and perform learning step(s) on it """
        raise NotImplementedError

    def _parse_info(self, info, print_each: int = 20):
        super()._parse_info(info, print_each)
        if len(info['terminated_episode_indices']):
            self.noise_process.reset(self.action_noise_scale, indices=info['terminated_episode_indices'])

    def _record_video(self, use_target=True):
        self.env_record = self.make_env_fn(record_video=True)
        self.env_record.gym_env.name_prefix = 't=%d_' % self.global_t
        self.env_record.reset()
        episode_finished = False
        while not episode_finished:
            obs_raw = self.env_record.get_observation()
            obs_norm = self.env_record.normalize_obs(obs_raw, update=False).reshape(1, -1)
            obs_norm_th = torch.from_numpy(obs_norm.astype(np.float32))
            # Choose action and evaluate value of the current observation
            with torch.no_grad():
                action_th = self.choose_action(obs_norm_th, use_target=use_target, noisy=False)
                action = action_th[0].cpu().numpy()
            # Make step in the environment and read the next observation
            info = self.env_record.step(action)
            episode_finished = info['done'][0, 0] or info['trunc'][0, 0]

    def _run_test_episodes(self, n_episodes, use_target=False):
        self.env_test = self.make_env_fn()
        self.env_test.reset()
        self.env_test.obs_normalizer = self.env.obs_normalizer
        test_episodes_finished = 0
        test_episodes_stats = defaultdict(list)
        while test_episodes_finished < n_episodes:
            obs_raw = self.env_test.get_observation()
            obs_norm = self.env_test.normalize_obs(obs_raw, update=False)
            obs_norm_th = torch.from_numpy(obs_norm.astype(np.float32))
            # Choose action and evaluate value of the current observation
            with torch.no_grad():
                action = self.choose_action(obs_norm_th, use_target=use_target, noisy=False)
                action = action.cpu().numpy()
            # Make step in the environment and read the next observation
            info = self.env_test.step(action)
            if 'episode' in info:
                test_episodes_finished += len(info['terminated_episode_indices'])
                for key, list_of_vals in info['episode'].items():
                    test_episodes_stats[key].extend(list_of_vals)
        test_results = {key: np.mean(vals, 0) for key, vals in test_episodes_stats.items()}
        print('-' * 40)
        print('Test results at timestep %s' % self.global_t)
        print('Avg. sum-reward: %.3f. Avg. max-reward: %.3f. Avg. success ratio: %.2f' %
              (np.mean(test_results['performance/reward_sum']),
               np.mean(test_results['performance/reward_max']),
               np.mean(test_results['performance/success'])))
        print('-' * 40)
        return test_results

    def train(self, print_each: int = 20, record_each: int = None,):
        self.print_each = int(np.ceil(print_each / self.num_envs) * self.num_envs)
        next_record_at = 999999999999 if record_each is None else int(np.ceil(record_each /
                                                                              self.num_envs) * self.num_envs)
        # Reset environment, memory, and run stats.
        self.env.reset()
        self.memory.reset()
        self._initialize_progress_variables()
        value_evaluations = [self.evaluate_v]
        # Run
        time_start = time.time()
        while self.global_t < self.env_steps:
            # If time to test -- test
            if self.global_t >= self.next_test_at:
                test_results = self._run_test_episodes(self.n_test_episodes, )
                for key, val in test_results.items():
                    if self.use_wandb:
                        val = wandb.Image(val) if 'image' in key else val
                        self.logger.log({'test_' + key: val}, step=self.global_t)
                self.next_test_at += self.test_each

            # Update the progress variable
            self._each_step_routine()

            # Read the current state of the environment
            obs_raw = self.env.get_observation()
            obs_norm = self.env.normalize_obs(obs_raw, update=True)
            obs_norm_th = torch.from_numpy(obs_norm.astype(np.float32))

            # Choose action and apply it to the environment
            with torch.no_grad():
                action_th = self.choose_action(obs_norm_th, use_target=False, noisy=True, )
                action = action_th.cpu().numpy()

            # Make step in the environment and read the next observation
            info = self.env.step(action, value_evaluations)  # TODO change value evaluations
            obs_next_raw = info['next_obs_true']

            # Save transition to the replay buffer
            self._save_transition(obs_raw, action, obs_next_raw, info)

            # As long as we can sample from the replay buffer, we train on a minibatch each timestep
            if self.memory.can_sample and not self._is_initial_exploration:
                self.since_last_train += self.num_envs
                if self.global_t >= self.next_train_at:
                    times_learn = self.since_last_train
                    for _ in range(times_learn):
                        self._learn()
                    self.next_train_at = int(self.global_t + self.train_freq)
                    self.since_last_train *= 0

            # If some of the trajectories have just finished, log the corresponding metrics
            self._parse_info(info, print_each)
            if self.global_t % (self.num_envs * 1000) == 0:
                print('Last %s steps took %.2f seconds' % (self.num_envs * 1000, time.time() - time_start))
                time_start = time.time()

            if self.episodes_finished >= next_record_at:
                print('Recording!')
                self._record_video()
                next_record_at = next_record_at + int(record_each)

            # Run final evaluation and close the logger
        self._final_evaluation()
        if self.use_wandb:
            self.logger.finish()

    def _final_evaluation(self):
        evaluation_results = self._run_test_episodes(self.n_evaluation_episodes, )
        print('Evaluation results:')
        for key, val in evaluation_results.items():
            if self.use_wandb:
                val = wandb.Image(val) if 'image' in key else val
                self.logger.log({'final_' + key: val}, step=self.global_t)
            if 'image' not in key:
                print(key, val)

    @abstractmethod
    def save(self, path):
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        raise NotImplementedError
