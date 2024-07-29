from src.environments.continuous_env import ContinuousEnv

from src.neural_networks.policies.stochastic_policy import StochasticPolicy
from src.neural_networks.value_functions.vmax_network import VMaxNetwork
from src.neural_networks.value_functions.v_network import VNetwork
from src.memory.rollout_buffer import RolloutBuffer

from src.rl_agents.continuous_on_policy_agent import ContinuousOnPolicyAgent

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.distributions import DiagGaussianDistribution

from src.rl_agents.ppo_family._estimate_advantages import _estimate_max_returns_gae
from src.rl_agents.ppo_family._estimate_advantages import _estimate_max_returns_montecarlo

import wandb

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional, Callable
import torch
import numpy as np
import time


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContinuousPPOConditionalMaxAgent(ContinuousOnPolicyAgent):

    def __init__(self,
                 make_env_fn: Callable,
                 hidden_dims_list: list[int],
                 wandb_project_name: str,
                 wandb_run_name: str,
                 env_steps: int = int(1e6),
                 n_evaluation_episodes: int = 100,
                 minibatch_size: int = 32,
                 discount: float = 0.999,
                 rollout_length: int = 2048,
                 use_gae=False,
                 lr=3e-4,
                 anneal_lr=True,
                 entropy_weight=0.0,
                 value_weight=0.5,
                 policy_weight=1,
                 training_epochs=10,
                 clip_coef=0.2,
                 max_grad_norm=0.5,
                 y_init_ub=0,
                 wandb_tags: Optional[list[str]] = None,
                 use_wandb: bool = True,
                 extra_config: Optional[dict] = None,
                 **kwargs):
        # Save PPO specific parameters and add them to the config
        self.lr = lr
        self.use_gae = use_gae
        self.anneal_lr = anneal_lr
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.policy_weight = policy_weight
        self.training_epochs = training_epochs
        self.clip_coef = clip_coef
        self.max_grad_norm = max_grad_norm
        self.y_init_ub = y_init_ub
        ppomax_specific_config = ({'name': 'ContinuousPPOConditionalMax', 'lr': lr, 'anneal_lr': anneal_lr,
                                   'use_gae': use_gae, 'entropy_weight': entropy_weight,
                                   'value_weight': value_weight, 'policy_weight': policy_weight,
                                   'training_epochs': training_epochs, 'clip_coef': clip_coef,
                                   'max_grad_norm': max_grad_norm, 'y_init_ub': y_init_ub, })
        if extra_config is not None:
            ppomax_specific_config.update(extra_config)
        # Initialize the agent
        super().__init__(make_env_fn, hidden_dims_list, wandb_project_name, wandb_run_name, env_steps,
                         n_evaluation_episodes, minibatch_size, discount, rollout_length, wandb_tags,
                         use_wandb, ppomax_specific_config, **kwargs)

    def _initialize_action_distribution(self):
        self.action_distribution = SquashedDiagGaussianDistribution(self.action_dim)

    def _create_networks(self, ):
        self.value_bounds = 0, self.env.reward_ub - self.env.reward_lb  # IMPORTANT: lower bound must be non-negative !
        # Create actor
        self.action_lb_th = torch.from_numpy(self.env.action_lb).to(torch.float32).to(DEVICE)
        self.action_ub_th = torch.from_numpy(self.env.action_ub).to(torch.float32).to(DEVICE)
        self.actor = StochasticPolicy(self.observation_dim + 1, self.action_dim, self.hidden_dims_list,
                                      self.action_lb_th, self.action_ub_th).to(DEVICE)
        self.critic = VMaxNetwork(self.observation_dim, self.hidden_dims_list, self.value_bounds).to(DEVICE)
        self.parameters_list = self.actor.parameters_list + self.critic.parameters_list

        self.optimizer = torch.optim.Adam(self.parameters_list, lr=self.lr, eps=1e-5)

    def _create_memory(self):
        """ Create rollout buffer (memory for on policy algorithms) """
        obs_dim, a_dim = (self.env.obs_dim,), (self.env.action_dim,)
        self.memory = RolloutBuffer(self.num_envs, self.rollout_length,
                                    scheme={'obs': obs_dim, 'y_true': (1,),
                                            'value': (1,), 'a': a_dim, 'a_log_prob': (1,), 'r': (1,),
                                            'done': (1,), 'trunc': (1,), 'obs_next': obs_dim})

    def _save_transition(self, obs: np.ndarray, value: np.ndarray,
                         action: np.ndarray, log_probs: np.ndarray,
                         obs_next: np.ndarray, info: dict[str, np.ndarray]):
        """ Saves transition data into the rollout buffer """
        transition_dict = {'obs': obs, 'value': value,
                           'a': action, 'a_log_prob': log_probs, 'r': info['reward'],
                           'y_true': info['y'],
                           'done': info['done'], 'trunc': info['trunc'], 'obs_next': obs_next}
        transition_dict = {key: torch.from_numpy(val.astype(np.float32)) for key, val in transition_dict.items()}
        self.memory.add_transition(transition_dict)

    def choose_action(self, observation, y=None, deterministic=False, **kwargs):
        """ Sample an action from the stochastic policy """
        assert len(observation.shape) == 2, 'Agents work with observations of shape (batch, shape)'
        if y is None:
            y = torch.tensor(len(observation) * [self.value_bounds[0]], dtype=torch.float32).reshape(-1, 1)

        # Compute mean and std of the policy
        observation = torch.concatenate([observation, y], dim=-1).to(DEVICE)
        action_mu, action_log_std = self.actor(observation)

        # Sample action in [-1,1] range from the Squashed Gaussian distribution
        self.action_distribution.proba_distribution(action_mu, action_log_std)
        action = action_mu if deterministic else self.action_distribution.sample()
        log_probs = self.action_distribution.log_prob(action)[:, None]

        # Transform action to the correct range [action_lb, action_ub]
        action_scaled = (action * (self.action_ub_th - self.action_lb_th) + self.action_ub_th + self.action_lb_th) / 2
        log_probs = log_probs - torch.log(torch.prod(2 / (self.action_ub_th - self.action_lb_th), dim=-1, keepdim=True))

        return action_scaled, log_probs

    def evaluate_v(self, observation, y=None, **kwargs):
        assert len(observation.shape) == 2, 'Agents work with observations of shape (batch, shape)'
        if y is None:
            # When y is not provided, we use the minimum possible value.
            # This is a reasonable default choice, as we optimize the policy using V(obs, y=None)
            y = torch.tensor(len(observation) * [self.value_bounds[0]], dtype=torch.float32).reshape(-1, 1)
        V = self.critic(observation.to(DEVICE), y.to(DEVICE))
        return V

    def evaluate_v_int(self, observation, y_int=None):
        return torch.zeros(observation.shape[0], 1, dtype=torch.float32)

    def evaluate_policy(self, observation, action,  y=None, **kwargs):
        assert len(observation.shape) == 2, 'Agents work with observations of shape (batch, shape)'
        if y is None:
            y = torch.tensor(len(observation) * [self.value_bounds[0]], dtype=torch.float32).reshape(-1, 1)
        observation = torch.concatenate([observation, y], dim=-1).to(DEVICE)
        # Get distribution parameters
        action_mu, action_log_std = self.actor(observation)
        self.action_distribution.proba_distribution(action_mu, action_log_std)

        # Rescale the action and back to [-1, 1] range (support of Squashed Gaussian)
        action_unscaled = (2 * action - self.action_lb_th - self.action_ub_th) / (self.action_ub_th - self.action_lb_th)

        # Estimate log probs and entropy
        log_probs = self.action_distribution.log_prob(action_unscaled)[:, None]
        log_probs = log_probs - torch.log(torch.prod(2 / (self.action_ub_th - self.action_lb_th), dim=-1, keepdim=True))
        entropy = self.action_distribution.entropy()
        if entropy is None:
            entropy = -log_probs.mean(dim=-1, keepdims=True)
        else:
            entropy = entropy[:, None]

        return log_probs, entropy

    def _run_test_episodes(self, n_episodes, deterministic=True):
        self.env_test = self.make_env_fn()
        self.env_test.reset()
        self.env_test.obs_normalizer = self.env.obs_normalizer
        test_episodes_finished = 0
        test_episodes_stats = defaultdict(list)
        running_y = self.value_bounds[0] * np.ones((self.num_envs, 1), dtype=np.float32)
        while test_episodes_finished < n_episodes:
            obs_raw = self.env_test.get_observation()
            obs_norm = self.env_test.normalize_obs(obs_raw, update=False)
            obs_norm_th = torch.from_numpy(obs_norm.astype(np.float32))
            # Choose action and evaluate value of the current observation
            with torch.no_grad():
                running_y_th = torch.from_numpy(running_y).to(torch.float32)
                action_th, log_probs = self.choose_action(obs_norm_th, y=running_y_th, deterministic=deterministic)
                action = action_th.cpu().numpy()
                reward_int = np.zeros((self.num_envs, 1))
            # Make step in the environment and read the next observation
            info = self.env_test.step(action)
            running_y = np.maximum(info['reward'] + reward_int - self.env.reward_lb, running_y)
            if 'episode' in info:
                test_episodes_finished += len(info['terminated_episode_indices'])
                running_y[info['terminated_episode_indices']] = 0
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

    def _learn(self) -> dict:
        # Sample, check data, reshape/recast
        sample_dict = self.memory.get_data(reset=False)
        (obs, y_ext_true, value, action, action_log_prob, reward, done,
         trunc, obs_next) = [sample_dict[key].to(DEVICE) for key in ['obs', 'y_true', 'value', 'a',
                                                                     'a_log_prob', 'r',  'done',
                                                                     'trunc', 'obs_next']]
        y_ext_true = y_ext_true  # y is saved after subtracting reward lb

        done = done.to(torch.bool)
        trunc = trunc.to(torch.bool)

        reward = reward - self.env.reward_lb  # For max-reward RL, we need non-negative rewards !

        epoch_results = defaultdict(lambda: [])

        if not self.use_gae:
            time_start = time.time()
            advantages_ext = _estimate_max_returns_montecarlo(y_ext_true, reward, done, trunc, obs_next, DEVICE,
                                                              self.evaluate_v, self.discount) - value
        else:
            time_start_gae = time.time()
            advantages_ext = _estimate_max_returns_gae(y_ext_true, reward, done, trunc, obs_next, DEVICE,
                                                       self.evaluate_v, self.discount, gae_lambda=0.95) - value

        advantages = advantages_ext

        # Compute returns using true y. We can not sample random y since policy depends on it
        with (torch.no_grad()):
            returns_ext = _estimate_max_returns_montecarlo(y_ext_true, reward, done, trunc, obs_next, DEVICE,
                                                           self.evaluate_v, self.discount)

        # Reshape data to (n_samples, shape)
        n_samples = self.rollout_length * self.num_envs
        obs = obs.reshape(-1, *obs.shape[2:])
        action = action.reshape(-1, *action.shape[2:])
        action_log_prob = action_log_prob.reshape(-1, *action_log_prob.shape[2:])
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        returns_ext = returns_ext.reshape(-1, *returns_ext.shape[2:])
        y_ext_true = y_ext_true.reshape(-1, *y_ext_true.shape[2:])

        # Run training epochs
        for epoch in range(self.training_epochs):
            indices = np.arange(0, n_samples, dtype=int)
            np.random.shuffle(indices)
            minibatch_indices = [indices[i: i + self.minibatch_size] for i in range(0, n_samples, self.minibatch_size)]
            for inds in minibatch_indices:
                # Compute policy loss
                new_log_prob, entropy = self.evaluate_policy(obs[inds], action[inds], y=y_ext_true[inds])
                log_ratio = new_log_prob - action_log_prob[inds]
                ratio = log_ratio.exp()
                advantages_normalized = (advantages[inds] - advantages[inds].mean()) / (advantages[inds].std() + 1e-8)
                policy_loss1 = -advantages_normalized * ratio
                policy_loss2 = -advantages_normalized * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Compute critic loss
                new_value = self.evaluate_v(obs[inds], y=y_ext_true[inds])
                critic_loss_ext = 0.5 * ((new_value - returns_ext[inds]) ** 2).mean()
                critic_loss_int = torch.tensor(0.).to(DEVICE)

                critic_loss = critic_loss_ext + critic_loss_int

                # Compute entropy loss
                entropy_loss = -entropy.mean()

                # Make gradient update step
                loss = (self.policy_weight * policy_loss + self.entropy_weight * entropy_loss +
                        self.value_weight * critic_loss)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters_list, self.max_grad_norm)
                self.optimizer.step()

                # Compute approx. KL for logging  http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                # Save losses
                epoch_results['policy_loss'].append(policy_loss.cpu().detach().numpy())
                epoch_results['critic_loss'].append(critic_loss_ext.cpu().detach().numpy())
                epoch_results['critic_loss_int'].append(critic_loss_int.cpu().detach().numpy())
                epoch_results['entropy_loss'].append(entropy_loss.cpu().detach().numpy())
                epoch_results['approx_kl'].append(approx_kl.cpu().detach().numpy())
                epoch_results['policy_log_std'].append(self.actor.log_std.mean().cpu().detach().numpy())
        for key in epoch_results:
            epoch_results[key] = np.mean(epoch_results[key])
        if self.use_wandb:
            self.logger.log({'learn/%s' % key: val for key, val in epoch_results.items()}, step=self.global_t)
        return epoch_results

    def train(self, print_each: int = 20, record_each: int = None,):
        self.print_each = int(np.ceil(print_each/self.num_envs) * self.num_envs)
        next_record_at = 999999999999 if record_each is None else int(record_each)
        # Reset environment, memory, and run stats.
        self.env.reset()
        self.memory.reset()
        self._initialize_progress_variables()
        value_evaluations = [self.evaluate_v]
        # Run
        time_start = time.time()
        running_y = self.value_bounds[0] * np.ones((self.num_envs, 1), dtype=np.float32)
        while self.global_t < self.env_steps:

            # Update the progress variable
            self._each_step_routine()

            # Read the current state of the environment
            obs_raw = self.env.get_observation()
            obs_norm = self.env.normalize_obs(obs_raw, update=True)
            obs_norm_th = torch.from_numpy(obs_norm.astype(np.float32))

            # Choose action and evaluate value of the current observation
            with torch.no_grad():
                running_y_th = torch.from_numpy(running_y).to(torch.float32)
                action_th, log_probs = self.choose_action(obs_norm_th, y=running_y_th)
                action = action_th.cpu().numpy()
                log_probs = log_probs.cpu().numpy()
                value = self.evaluate_v(obs_norm_th, y=running_y_th).cpu().numpy()
                value_int = self.evaluate_v_int(obs_norm_th, y_int=None).cpu().numpy()
                reward_int = np.zeros((self.num_envs, 1))

            # Make step in the environment and read the next observation
            info = self.env.step(action, value_evaluations)  # TODO change value evaluations

            obs_next_raw = info['next_obs_true']
            obs_next_norm = self.env.normalize_obs(obs_next_raw, update=True)
            info['reward_int'] = reward_int
            info['y'] = np.copy(running_y)

            # Save transition to the rollout buffer. Remark: we store normalized observations, unlike in off-policy,
            self._save_transition(obs_norm, value, action, log_probs, obs_next_norm, info)
            running_y = np.maximum(info['reward'] + reward_int - self.env.reward_lb, running_y)
            if 'episode' in info:
                running_y[info['terminated_episode_indices']] = self.value_bounds[0]
            # As soon as the rollout buffer is full, run training, log training stats, and reset the buffer
            if self.memory.is_full:
                self._learn()
                self.memory.reset()

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

    def _each_step_routine(self, **kwargs):
        super()._each_step_routine(**kwargs)
        if self.anneal_lr:
            frac = 1.0 - (self.global_t - 1.0) / self.env_steps
            self.optimizer.param_groups[0]["lr"] = frac * self.lr

    def save(self, path):
        pass

    def load(self, path):
        pass

