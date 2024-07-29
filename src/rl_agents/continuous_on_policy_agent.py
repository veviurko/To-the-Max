from src.environments.continuous_env import ContinuousEnv
from src.memory.rollout_buffer import RolloutBuffer
from src.rl_agents.base_agent import Agent

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Callable
import numpy as np
import torch
import wandb
import time


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContinuousOnPolicyAgent(Agent):

    def __init__(self,
                 make_env_fn: Callable[[bool, ], ContinuousEnv],
                 hidden_dims_list: list[int],
                 wandb_project_name: str,
                 wandb_run_name: str,
                 env_steps: int = int(1e6),
                 n_evaluation_episodes: int = 200,
                 minibatch_size: int = 32,
                 discount: float = 0.99,
                 rollout_length: int = 2048,
                 wandb_tags: Optional[list[str]] = None,
                 use_wandb: bool = True,
                 extra_config: Optional[dict] = None,
                 **kwargs):

        # Save parameters
        self.n_evaluation_episodes = n_evaluation_episodes
        self.minibatch_size = minibatch_size
        self.rollout_length = rollout_length

        # Make configs and prepare logging
        agent_specific_config = {'name': 'OnPolicyAgent',
                                 'minibatch_size': minibatch_size, 'rollout_length': rollout_length,
                                 }
        agent_specific_config.update(extra_config)
        super().__init__(make_env_fn, hidden_dims_list, wandb_project_name, wandb_run_name, env_steps, discount,
                         wandb_tags, use_wandb, agent_specific_config, **kwargs)
        self._initialize_action_distribution()

    def _create_memory(self):
        """ Create rollout buffer (memory for on policy algorithms) """
        obs_dim, a_dim = (self.env.obs_dim,), (self.env.action_dim,)
        self.memory = RolloutBuffer(self.num_envs, self.rollout_length,
                                    scheme={'obs': obs_dim, 'value': (1,),
                                            'a': a_dim, 'a_log_prob': (1, ), 'r': (1,),
                                            'done': (1,), 'trunc': (1,), 'obs_next': obs_dim})

    @abstractmethod
    def _initialize_action_distribution(self):
        """ Create action distribution object. This method can depend on the particular algorithm. """
        raise NotImplementedError

    @abstractmethod
    def _create_networks(self, ):
        """ Create actor and critic networks and the Adam optimizer """
        raise NotImplementedError

    @abstractmethod
    def choose_action(self,  observation: torch.Tensor, deterministic: bool = False, **kwargs) -> torch.Tensor:
        """ Sample an action from the policy """
        raise NotImplementedError

    @abstractmethod
    def evaluate_v(self, observation: torch.Tensor, **kwargs) -> torch.Tensor:
        """ Evaluate the value function network on the observation """
        raise NotImplementedError

    @abstractmethod
    def evaluate_v_int(self, observation: torch.Tensor, y_int=None) -> torch.Tensor:
        """ Evaluate the intrinsic value head on the observation. If there is no intrinsic reward, returns zeros. """
        raise NotImplementedError

    @abstractmethod
    def evaluate_policy(self, observation: torch.Tensor, action: torch.Tensor,
                        **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """ Compute logprobs and entropy of a given action at given observation using policy network """
        raise NotImplementedError

    def _save_transition(self, obs: np.ndarray, value: np.ndarray,
                         action: np.ndarray, log_probs: np.ndarray,
                         obs_next: np.ndarray, info: dict[str, np.ndarray]):
        """ Saves transition data into the rollout buffer """
        transition_dict = {'obs': obs, 'value': value,
                           'a': action, 'a_log_prob': log_probs, 'r': info['reward'],
                           'done': info['done'], 'trunc': info['trunc'], 'obs_next': obs_next}
        transition_dict = {key: torch.from_numpy(val.astype(np.float32)) for key, val in transition_dict.items()}
        self.memory.add_transition(transition_dict)

    def _parse_info(self, info, print_each: int = 20):
        super()._parse_info(info, print_each)

    @abstractmethod
    def _learn(self) -> dict:
        """ Sample data from the rollout buffer and perform learning step(s) on it """
        raise NotImplementedError

    def _run_test_episodes(self, n_episodes, deterministic=True):
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
                action, log_probs = self.choose_action(obs_norm_th, deterministic=deterministic)
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
        # When to print and when to record videos
        self.print_each = int(np.ceil(print_each/self.num_envs) * self.num_envs)
        next_record_at = 999999999999 if record_each is None else int(record_each)

        # Reset environment, memory, and run statistics.
        self.env.reset()
        self.memory.reset()
        self._initialize_progress_variables()
        value_evaluations = [self.evaluate_v]
        # Run
        time_start = time.time()
        while self.global_t < self.env_steps:

            # Update the progress variable
            self._each_step_routine()

            # Read the current state of the environment
            obs_raw = self.env.get_observation()
            obs_norm = self.env.normalize_obs(obs_raw, update=True)
            obs_norm_th = torch.from_numpy(obs_norm.astype(np.float32))

            # Choose action and evaluate value of the current observation
            with torch.no_grad():
                action_th, log_probs = self.choose_action(obs_norm_th)
                action = action_th.cpu().numpy()
                log_probs = log_probs.cpu().numpy()
                value = self.evaluate_v(obs_norm_th).cpu().numpy()

            # Make step in the environment and read the next observation
            info = self.env.step(action, value_evaluations)  # TODO change value evaluations
            obs_next_raw = info['next_obs_true']
            obs_next_norm = self.env.normalize_obs(obs_next_raw, update=True)

            # Save transition to the rollout buffer. Remark: we store normalized observations, unlike in off-policy,
            self._save_transition(obs_norm, value, action, log_probs, obs_next_norm, info)

            # As soon as the rollout buffer is full, run training, log training stats, and reset the buffer
            if self.memory.is_full:
                self._learn()
                self.memory.reset()

            # If some of the trajectories have just finished, log the corresponding metrics
            self._parse_info(info, print_each)
            if self.global_t % (self.num_envs * 1000) == 0:
                print('Last %s steps took %.2f seconds' % (self.num_envs * 1000, time.time() - time_start))
                time_start = time.time()

            # If it is time, record a video.
            if self.episodes_finished >= next_record_at:
                print('Recording!')
                self._record_video()
                next_record_at = next_record_at + int(record_each)

        # Run final evaluation and close the logger
        self._final_evaluation()
        if self.use_wandb:
            self.logger.finish()

    def _final_evaluation(self):
        evaluation_results = self._run_test_episodes(self.n_evaluation_episodes, deterministic=True)
        print('Deterministic evaluation results:')
        for key, val in evaluation_results.items():
            if self.use_wandb:
                val = wandb.Image(val) if 'image' in key else val
                self.logger.log({'final_deterministic_' + key: val}, step=self.global_t)
            if 'image' not in key:
                print(key, val)

        evaluation_results = self._run_test_episodes(self.n_evaluation_episodes, deterministic=False)
        print('Stochastic evaluation results:')
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


