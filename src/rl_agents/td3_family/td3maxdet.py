from src.rl_agents.continuous_off_policy_agent import ContinuousOffPolicyAgent, DEVICE
from src.environments.continuous_env import ContinuousEnv
from src.noise.white_noise import WhiteNoiseProcess
from collections import defaultdict
from typing import Optional, Callable

from src.neural_networks.policies.deterministic_policy import DeterministicPolicy
from src.neural_networks.value_functions.qmaxdet_network import QMaxDetNetwork
from src.neural_networks.value_functions.q_network import QNetwork

import torch
from copy import deepcopy
import torch.nn.functional as F
import numpy as np


class TD3MaxDetAgent(ContinuousOffPolicyAgent):

    def __init__(self,
                 make_env_fn: Callable,
                 hidden_dims_list: list[int],
                 wandb_project_name: str,
                 wandb_run_name: str,
                 env_steps: int = int(1e6),
                 test_each: int = 20000,
                 n_test_episodes: int = 20,
                 n_evaluation_episodes: int = 100,
                 discount: float = 0.999,
                 start_training_after: int = 10000,
                 train_batch_size: int = 256,
                 memory_size: int = int(1e6),
                 train_freq: int = 1,
                 critic_lr=3e-4,
                 actor_lr=3e-4,
                 action_l2_regularization_weight=0,
                 action_noise_type: str = 'white',
                 action_noise_scale: float = 0.1,
                 target_action_noise_clip: float = 0.5,
                 target_action_noise_scale: float = 0.2,
                 tau=0.005,
                 policy_update_freq=2,
                 wandb_tags: Optional[list[str]] = None,
                 use_wandb: bool = True,
                 extra_config: Optional[dict] = None,
                 **kwargs
                 ):
        # Save TD3 specific parameters and add them to the config
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.target_action_noise_scale = target_action_noise_scale
        self.tau = tau
        self.policy_update_freq = policy_update_freq
        self.action_l2_regularization_weight = action_l2_regularization_weight
        td3_specific_config = {'name': 'TD3MaxDetAgent', 'critic_lr': critic_lr, 'actor_lr': actor_lr,
                               'target_action_noise_scale': target_action_noise_scale,
                               'tau': tau, 'policy_update_freq': policy_update_freq,
                               'action_l2_regularization_weight': action_l2_regularization_weight}
        if extra_config is not None:
            td3_specific_config.update(extra_config)
        super().__init__(make_env_fn, hidden_dims_list, wandb_project_name, wandb_run_name, env_steps, test_each,
                         n_test_episodes, n_evaluation_episodes, discount, start_training_after, train_batch_size,
                         memory_size, train_freq, action_noise_type, action_noise_scale, target_action_noise_clip,
                         wandb_tags, use_wandb, td3_specific_config, **kwargs)


    def _initialize_progress_variables(self):
        super()._initialize_progress_variables()
        self.critic_updates = 0


    def _initialize_exploration_noise(self):
        super()._initialize_exploration_noise()
        self.target_noise_process = WhiteNoiseProcess(self.train_batch_size, self.action_dim,
                                                      self.target_action_noise_scale,
                                                      self.env.max_episode_length,
                                                      noise_clip=self.target_action_noise_clip)

    def _create_networks(self):
        self.value_bounds = 0, self.env.reward_ub - self.env.reward_lb  # IMPORTANT: lower bound must be non-negative !
        # Create actor
        self.action_lb_th = torch.from_numpy(self.env.action_lb).to(torch.float32).to(DEVICE)
        self.action_ub_th = torch.from_numpy(self.env.action_ub).to(torch.float32).to(DEVICE)
        self.actor = DeterministicPolicy(self.observation_dim, self.action_dim, self.hidden_dims_list,
                                         self.action_lb_th, self.action_ub_th).to(DEVICE)
        self.actor_target = deepcopy(self.actor).to(DEVICE)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters_list, lr=self.actor_lr)

        # Create critic
        self.critic1 = QMaxDetNetwork(self.observation_dim, self.action_dim, self.hidden_dims_list,
                                      self.value_bounds).to(DEVICE)
        #self.critic1 = QNetwork(self.observation_dim, self.action_dim, self.hidden_dims_list).to(DEVICE)
        self.critic2 = QMaxDetNetwork(self.observation_dim, self.action_dim, self.hidden_dims_list,
                                      self.value_bounds).to(DEVICE)
        # self.critic2 = QNetwork(self.observation_dim, self.action_dim, self.hidden_dims_list).to(DEVICE)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        self.critics_optimizer = torch.optim.Adam(self.critic1.parameters_list +
                                                  self.critic2.parameters_list, lr=self.critic_lr)

    def choose_action(self, observation: torch.Tensor, noisy: bool, use_target: bool = False, **kwargs) -> torch.Tensor:
        # Check/fix input arguments
        assert len(observation.shape) == 2, 'Agents work with observations of shape (batch, shape)'

        # In the beginning, we do random exploration, and we use uniform sampling for actions
        if noisy and self._is_initial_exploration:
            noise = self.noise_process.sample().to(DEVICE)
            return noise.clamp(self.action_lb_th, self.action_ub_th)

        # Otherwise we choose action using the actor
        actor_to_use = self.actor_target if use_target else self.actor
        action = actor_to_use(observation.to(DEVICE))

        # Then, we add noise to the action. It is either zero (noisy=False) or sampled from one of the noise processes
        noise_process_to_use = self.target_noise_process if use_target else self.noise_process
        noise = noise_process_to_use.sample().to(DEVICE) if noisy else 0

        return (action + noise).clamp(self.action_lb_th, self.action_ub_th)

    def evaluate_q(self, observation: torch.Tensor, action: torch.Tensor,
                   use_target: bool = False, use_critic1: bool = True, **kwargs) -> torch.Tensor:

        # Check/fix input arguments
        assert len(observation.shape) == 2, 'Agents work with observations of shape (batch, shape)'

        # Choose which of the two critics to use and whether to use target or not
        if use_critic1:
            critic_to_use = self.critic1_target if use_target else self.critic1
        else:
            critic_to_use = self.critic2_target if use_target else self.critic2

        # Compute Q value
        q = critic_to_use(observation.to(DEVICE), action.to(DEVICE))

        return q

    def evaluate_v(self, observation, **kwargs):
        best_action = self.choose_action(observation, noisy=False, use_target=False)
        return self.evaluate_q(observation, best_action, use_target=False, use_critic1=True)

    def _learn(self) -> dict:
        # Sample from memory
        sample_dict = self.memory.get_data(batch_size=self.train_batch_size)
        obs = (self.env.normalize_obs(sample_dict['obs'], update=False)).to(torch.float32).to(DEVICE)
        obs_next = (self.env.normalize_obs(sample_dict['obs_next'], update=False)).to(torch.float32).to(DEVICE)
        action, reward, done, trunc = [(sample_dict[key]).to(DEVICE) for key in ['a', 'r', 'done', 'trunc']]
        reward = reward - self.env.reward_lb
        done = done.to(torch.bool)
        trunc = trunc.to(torch.bool)

        # Prepare a dictionary to store results of the learning epoch
        epoch_results = defaultdict(lambda: [])

        # Reset the noise process for target action smoothing
        self.target_noise_process.reset()

        # Compute target for the value function update
        with (torch.no_grad()):

            # Target actions are computed using the target policy network; we use smoothing with noise.
            target_actions = self.choose_action(obs_next, use_target=True, noisy=True)

            # Compute the target Q value
            q1_next = self.evaluate_q(obs_next, target_actions, use_target=True, use_critic1=True)
            q2_next = self.evaluate_q(obs_next, target_actions, use_target=True, use_critic1=False)
            q_target = torch.where(done, reward,
                                   torch.maximum(reward, self.discount * torch.min(q1_next, q2_next)))

        # Compute current Q estimates
        q1 = self.evaluate_q(obs, action, use_target=False, use_critic1=True)
        q2 = self.evaluate_q(obs, action, use_target=False, use_critic1=False)

        # Critic loss and update step
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critics_optimizer.zero_grad()
        critic_loss.backward()
        self.critics_optimizer.step()
        self.critic_updates += 1
        epoch_results['critic_loss'] = critic_loss.cpu().detach().numpy()

        # Once in `self.policy_update_freq` we perform the policy update step
        if self.critic_updates % self.policy_update_freq == 0:

            # Choose greedy actions
            best_actions = self.choose_action(obs, use_target=False, noisy=False)

            # Measure achieved value
            policy_loss = -self.evaluate_q(obs, best_actions, use_target=False, use_critic1=True).mean()

            # Measure L2 regularization loss
            if self.action_l2_regularization_weight > 0:
                # \in [-1, 1]
                best_actions_normalized = 2 * (best_actions - self.action_lb_th) / self.actor.bounds_diff - 1
                actions_l2_loss = self.action_l2_regularization_weight * torch.norm(best_actions_normalized,
                                                                                    dim=-1).mean()
                policy_loss += actions_l2_loss

            # Optimize actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            epoch_results['policy_loss'] = policy_loss.cpu().detach().numpy()

            # Update the frozen target models
            for param, target_param in zip(self.critic1.parameters_list, self.critic1_target.parameters_list):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters_list, self.critic2_target.parameters_list):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters_list, self.actor_target.parameters_list):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Save results of this learning epoch
        for key, val in epoch_results.items():
            self.learning_results[key].append(val)

        # If we did enough of learning steps, it is time to log
        if len(self.learning_results['critic_loss']) == self.env.max_episode_length:
            mean_critic_loss = np.mean(self.learning_results['critic_loss'])
            mean_policy_loss = np.mean(self.learning_results['policy_loss'])
            if self.use_wandb:
                self.logger.log({'learn/critic_loss': mean_critic_loss}, step=self.global_t)
                self.logger.log({'learn/policy_loss': mean_policy_loss}, step=self.global_t)
            self.learning_results['critic_loss'] = []
            self.learning_results['policy_loss'] = []

        return epoch_results

    def save(self, path):
        pass

    def load(self, path):
        pass

