from src.rl_agents.continuous_on_policy_agent import ContinuousOnPolicyAgent, DEVICE
from src.neural_networks.policies.stochastic_policy import StochasticPolicy
from src.neural_networks.value_functions.vmax_network import VMaxNetwork
from src.neural_networks.value_functions.v_network import VNetwork
from src.rl_agents.ppo_family._estimate_advantages import _estimate_advantages_montecarlo, _estimate_advantages_gae
from src.rl_agents.ppo_family._estimate_advantages import _estimate_max_returns_montecarlo

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution


from collections import defaultdict
from typing import Optional, Callable
import torch
import numpy as np


class ContinuousPPOAgent(ContinuousOnPolicyAgent):

    def __init__(self,
                 make_env_fn: Callable,
                 hidden_dims_list: list[int],
                 wandb_project_name: str,
                 wandb_run_name: str,
                 env_steps: int = int(1e6),
                 n_evaluation_episodes: int = 100,
                 minibatch_size: int = 32,
                 discount: float = 0.99,
                 rollout_length: int = 2048,
                 use_gae=True,
                 gae_lambda=0.95,
                 lr=3e-4,
                 anneal_lr=True,
                 entropy_weight=0.0,
                 value_weight=0.5,
                 training_epochs=10,
                 clip_coef=0.2,
                 max_grad_norm=0.5,
                 subtract_reward_ub: bool = False,
                 wandb_tags: Optional[list[str]] = None,
                 use_wandb: bool = True,
                 extra_config: Optional[dict] = None,
                 **kwargs):
        # Save PPO specific parameters and add them to the config
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.anneal_lr = anneal_lr
        self.entropy_weight = entropy_weight
        self.value_weight = value_weight
        self.training_epochs = training_epochs
        self.clip_coef = clip_coef
        self.max_grad_norm = max_grad_norm
        self.subtract_reward_ub = subtract_reward_ub
        ppo_specific_config = {'name': 'ContinuousPPO', 'lr': lr, 'entropy_weight': entropy_weight,
                               'value_weight': value_weight, 'training_epochs': training_epochs,
                               'use_gae': use_gae, 'gae_lambda': gae_lambda, 'anneal_lr': anneal_lr,
                               'clip_coef': clip_coef, 'max_grad_norm': max_grad_norm,
                               'subtract_reward_ub': subtract_reward_ub,
                               }
        if extra_config is not None:
            ppo_specific_config.update(extra_config)
        # Initialize the agent
        super().__init__(make_env_fn, hidden_dims_list, wandb_project_name, wandb_run_name, env_steps,
                         n_evaluation_episodes, minibatch_size, discount, rollout_length, wandb_tags,
                         use_wandb, ppo_specific_config, **kwargs)

    def _initialize_action_distribution(self):
        self.action_distribution = SquashedDiagGaussianDistribution(self.action_dim)

    def _create_networks(self, ):
        # Create actor and critic networks
        self.action_lb_th = torch.from_numpy(self.env.action_lb).to(torch.float32).to(DEVICE)
        self.action_ub_th = torch.from_numpy(self.env.action_ub).to(torch.float32).to(DEVICE)
        self.actor = StochasticPolicy(self.observation_dim, self.action_dim, self.hidden_dims_list,
                                      self.action_lb_th, self.action_ub_th).to(DEVICE)
        self.critic = VNetwork(self.observation_dim, self.hidden_dims_list).to(DEVICE)
        self.parameters_list = self.actor.parameters_list + self.critic.parameters_list

        self.optimizer = torch.optim.Adam(self.parameters_list, lr=self.lr, eps=1e-5)

    def choose_action(self, observation, deterministic=False, **kwargs):
        """ Sample an action from the stochastic policy """
        assert len(observation.shape) == 2, 'Agents work with observations of shape (batch, shape)'

        # Compute mean and std of the policy
        observation = observation.to(DEVICE)
        action_mu, action_log_std = self.actor(observation)

        # Sample action in [-1,1] range from the Squashed Gaussian distribution
        self.action_distribution.proba_distribution(action_mu, action_log_std)
        action = action_mu if deterministic else self.action_distribution.sample()
        log_probs = self.action_distribution.log_prob(action)[:, None]

        # Transform action to the correct range [action_lb, action_ub]
        action_scaled = (action * (self.action_ub_th - self.action_lb_th) + self.action_ub_th + self.action_lb_th) / 2
        log_probs = log_probs - torch.log(torch.prod(2 / (self.action_ub_th - self.action_lb_th), dim=-1, keepdim=True))

        return action_scaled, log_probs

    def evaluate_v(self, observation, **kwargs):
        assert len(observation.shape) == 2, 'Agents work with observations of shape (batch, shape)'
        V = self.critic(observation.to(DEVICE))
        return V

    def evaluate_v_int(self, observation, y_int=None):
        return torch.zeros(observation.shape[0], 1, dtype=torch.float32)

    def evaluate_policy(self, observation, action, **kwargs):
        assert len(observation.shape) == 2, 'Agents work with observations of shape (batch, shape)'

        # Get distribution parameters
        action_mu, action_log_std = self.actor(observation.to(DEVICE))
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

    def _learn(self):
        # Sample, check data, reshape/recast
        sample_dict = self.memory.get_data(reset=False)
        (obs, value, action, action_log_prob, reward, done,
         trunc, obs_next) = [sample_dict[key].to(DEVICE) for key in ['obs', 'value', 'a', 'a_log_prob',
                                                                     'r', 'done', 'trunc', 'obs_next']]
        done = done.to(torch.bool)
        trunc = trunc.to(torch.bool)

        # In goal-reaching problems, it may be helpful to have negative rewards to encourage exploration
        if self.subtract_reward_ub:
            reward = reward - self.env.reward_ub

        # Create dictionary that will store the learning results
        epoch_results = defaultdict(lambda: [])

        # Compute advantages for extrinsic reward
        if self.use_gae:
            advantages_ext = _estimate_advantages_gae(value, reward, done, trunc, obs_next,
                                                      self.gae_lambda, DEVICE, self.evaluate_v, self.discount)
        else:
            advantages_ext = _estimate_advantages_montecarlo(value, reward, done, trunc, obs_next,
                                                             DEVICE, self.evaluate_v, self.discount)
        advantages = advantages_ext
        returns_ext = advantages_ext + value

        # Reshape data to (n_samples, shape)
        n_samples = self.rollout_length * self.num_envs
        obs = obs.reshape(-1, *obs.shape[2:])
        action = action.reshape(-1, *action.shape[2:])
        action_log_prob = action_log_prob.reshape(-1, *action_log_prob.shape[2:])
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        returns_ext = returns_ext.reshape(-1, *returns_ext.shape[2:])

        # Run training epochs
        for epoch in range(self.training_epochs):
            indices = np.arange(0, n_samples, dtype=int)
            np.random.shuffle(indices)
            minibatch_indices = [indices[i: i + self.minibatch_size] for i in range(0, n_samples, self.minibatch_size)]
            for inds in minibatch_indices:
                # Compute clipped policy loss
                new_log_prob, entropy = self.evaluate_policy(obs[inds], action[inds])
                log_ratio = new_log_prob - action_log_prob[inds]
                ratio = log_ratio.exp()
                advantages_normalized = (advantages[inds] - advantages[inds].mean()) / (advantages[inds].std() + 1e-8)
                policy_loss1 = -advantages_normalized * ratio
                policy_loss2 = -advantages_normalized * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                # Compute critic loss
                new_value = self.evaluate_v(obs[inds])
                critic_loss = 0.5 * ((new_value - returns_ext[inds]) ** 2).mean()

                # Compute entropy loss
                entropy_loss = -entropy.mean()

                # Make gradient update step
                loss = policy_loss + self.entropy_weight * entropy_loss + critic_loss * self.value_weight
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters_list, self.max_grad_norm)
                self.optimizer.step()

                # Compute approx. KL for logging  http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                # Save losses
                epoch_results['policy_loss'].append(policy_loss.cpu().detach().numpy())
                epoch_results['critic_loss'].append(critic_loss.cpu().detach().numpy())
                epoch_results['entropy_loss'].append(entropy_loss.cpu().detach().numpy())
                epoch_results['approx_kl'].append(approx_kl.cpu().detach().numpy())
                epoch_results['policy_log_std'].append(self.actor.log_std.mean().cpu().detach().numpy())

        # Finalize and log the learning results
        for key in epoch_results:
            epoch_results[key] = np.mean(epoch_results[key])
        if self.use_wandb:
            self.logger.log({'learn/%s' % key: val for key, val in epoch_results.items()}, step=self.global_t)

        return epoch_results

    def _each_step_routine(self, **kwargs):
        super()._each_step_routine(**kwargs)
        if self.anneal_lr:
            frac = 1.0 - (self.global_t - 1.0) / self.env_steps
            self.optimizer.param_groups[0]["lr"] = frac * self.lr

    def save(self, path):
        pass

    def load(self, path):
        pass
