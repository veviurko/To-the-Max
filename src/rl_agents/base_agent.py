from src.environments.continuous_env import ContinuousEnv
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Callable
import numpy as np
import torch
import wandb


class Agent(ABC):

    """ Base class for the agent """

    env: ContinuousEnv

    def __init__(self,
                 make_env_fn: Callable,
                 hidden_dims_list: list[int],
                 wandb_project_name: str,
                 wandb_run_name: str,
                 env_steps: int = int(1e6),
                 discount: float = 0.99,
                 wandb_tags: Optional[list[str]] = None,
                 use_wandb: bool = True,
                 extra_config: Optional[dict] = None,
                 **kwargs):

        # Save parameters
        self.make_env_fn = make_env_fn
        self.env = make_env_fn()
        self.num_envs = self.env.num_envs
        self.observation_dim = self.env.obs_dim
        self.action_dim = self.env.action_dim
        self.wandb_project_name = wandb_project_name
        self.hidden_dims_list = hidden_dims_list
        self.run_name = wandb_run_name
        self.env_steps = env_steps
        self.discount = discount
        self.wandb_tags = wandb_tags
        self.use_wandb = use_wandb
        self.print_each = int(self.num_envs)  # Every episode

        # Make configs
        self.agent_config = {'name': 'ContinuousAgent', 'run_name': wandb_run_name, 'discount': discount,}
        if extra_config is not None:
            self.agent_config.update(extra_config)
        self.env_config = self.env.env_config

        # Initialize all components of the agent
        self._initialize_progress_variables()
        self._initialize_logging()
        self._create_memory()
        self._create_networks()

    def _initialize_logging(self):
        """ Create wandb logger """
        if self.use_wandb:
            wandb_config = {}
            for key, val in self.agent_config.items():
                wandb_config['agent/' + key] = val
            for key, val in self.env_config.items():
                wandb_config['env/' + key] = val
            config_hash = ''.join([str(key) + str(val) for key, val in self.agent_config.items()] +
                                  [str(key) + str(val) for key, val in self.env_config.items()])

            self.logger = wandb.init(project=self.wandb_project_name,
                                     tags=self.wandb_tags,
                                     config=wandb_config,
                                     group=str(hash(config_hash)),
                                     monitor_gym=True)

    def _initialize_progress_variables(self):
        """ Initialize python objects that track progress """
        self.global_t = 0
        self.episodes_finished = 0
        self.next_print_at = int(self.print_each)
        self.full_run_stats = defaultdict(list)
        self.current_episode_data = [defaultdict(list) for _ in range(self.num_envs)]
        self.buffer_to_log = defaultdict(list)  # We log only each num_envs episodes!

    def _create_memory(self):
        """ Create memory buffer that stores experiences.
            Different types of memory should be used for on- and off- policy algorithms.
         """
        raise NotImplementedError

    def _create_networks(self):
        """ Create neural networks and their optimizers for actor, critic etc. """
        raise NotImplementedError

    def _learn(self):
        """ Sample data from the memory buffer and use it to update the models """
        raise NotImplementedError

    def _parse_info(self, info, print_each: int = 20):

        """ Parse info dict returned by environment at each step.
            Save and log progress, update progress variables, print recent results. """

        # If some episodes terminated - log their results.
        if 'episode' in info:
            self.episodes_finished += len(info['terminated_episode_indices'])
            for key, list_of_vals in info['episode'].items():
                self.buffer_to_log[key].extend(list_of_vals)
                self.full_run_stats[key].extend(list_of_vals)
            if self.use_wandb:
                for key, list_of_vals in info['episode'].items():
                    if 'image' in key:
                        self.logger.log({key: wandb.Image(list_of_vals[0])}, step=self.global_t)

                for key, list_of_vals in self.buffer_to_log.items():
                    if len(self.buffer_to_log[key]) < self.num_envs:
                        break
                    val = np.mean(list_of_vals)
                    self.logger.log({key: val}, step=self.global_t)
                    self.buffer_to_log[key] = []

            if self.episodes_finished >= self.next_print_at:
                print('Episodes finished: %d. Timestep: %d.' % (self.episodes_finished, self.global_t),
                      'Avg. sum-reward: %.3f. Avg. max-reward: %.3f. Avg. success ratio: %.2f' %
                      (np.mean(self.full_run_stats['performance/reward_sum'][-self.print_each:]),
                       np.mean(self.full_run_stats['performance/reward_max'][-self.print_each:]),
                       np.mean(self.full_run_stats['performance/success'][-self.print_each:]))
                      )
                self.next_print_at += self.print_each

        # Reset data for the terminated episodes
        for i in info['terminated_episode_indices']:
            self.current_episode_data[i] = defaultdict(list)

    def _each_step_routine(self, **kwargs):
        """ This function is called at each step. Can be used to anneal lr etc. """
        self.global_t += self.num_envs

    def train(self, **kwargs):
        """ The main function to be called.
            Run interaction with the environments, log the results, and performs learning. """
        raise NotImplementedError

    def _final_evaluation(self):
        """ Evaluates the final performance of the agent. Should be called in the end of the function train """
        raise NotImplementedError

