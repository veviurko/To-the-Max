import time

from src.rl_agents.continuous_off_policy_agent import ContinuousOffPolicyAgent, DEVICE
from src.memory.replay_buffer import ReplayBuffer
from src.environments.continuous_env import ContinuousEnv
from src.noise.white_noise import WhiteNoiseProcess
from collections import defaultdict
from typing import Optional, Callable

from src.neural_networks.policies.deterministic_policy import DeterministicPolicy
from src.neural_networks.value_functions.qmax_network import QMaxNetwork

import torch
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import wandb


class TD3ConditionalMaxAgent(ContinuousOffPolicyAgent):

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
                 y_init_ub=0,
                 use_intrinsic_reward: bool = False,
                 separate_intrinsic_reward_head: bool = False,
                 use_max_intrinsic_reward: bool = False,
                 intrinsic_reward_kwargs: Optional[dict[str, any]] = None,
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
        self.y_init_ub = y_init_ub
        self.use_intrinsic_reward = False
        td3_specific_config = {'name': 'TD3ConditionalMaxAgent', 'critic_lr': critic_lr, 'actor_lr': actor_lr,
                               'target_action_noise_scale': target_action_noise_scale,
                               'tau': tau, 'policy_update_freq': policy_update_freq,
                               'action_l2_regularization_weight': action_l2_regularization_weight,
                               'y_init_ub': y_init_ub}
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
        self.actor = DeterministicPolicy(self.observation_dim + 1, self.action_dim, self.hidden_dims_list,
                                         self.action_lb_th, self.action_ub_th).to(DEVICE)
        self.actor_target = deepcopy(self.actor).to(DEVICE)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters_list, lr=self.actor_lr)
        # Create critic
        self.critic1 = QMaxNetwork(self.observation_dim, self.action_dim,
                                   self.hidden_dims_list, self.value_bounds).to(DEVICE)
        self.critic2 = QMaxNetwork(self.observation_dim, self.action_dim,
                                   self.hidden_dims_list, self.value_bounds).to(DEVICE)
        self.critic1_target = deepcopy(self.critic1)
        self.critic2_target = deepcopy(self.critic2)
        self.critics_optimizer = torch.optim.Adam(self.critic1.parameters_list +
                                                  self.critic2.parameters_list, lr=self.critic_lr)

    def _create_memory(self):
        """ Create replay buffer (memory for off policy algorithms) """
        obs_dim, a_dim = (self.env.obs_dim,), (self.env.action_dim,)
        self.memory = ReplayBuffer(batch_size=self.num_envs,
                                   scheme={'obs': obs_dim,  'y_true': (1,), 'a': a_dim, 'r': (1,),
                                           'done': (1,), 'trunc': (1,), 'obs_next': obs_dim, },
                                   max_size=self.memory_size, min_size_to_sample=self.start_training_after)

    def _save_transition(self, obs: np.ndarray, action: np.ndarray,  obs_next: np.ndarray, info: dict[str, np.ndarray]):
        """ Saves transition data into the rollout buffer """
        transition_dict = {'obs': obs, 'y_true': info['y'], 'a': action, 'r': info['reward'],
                           'done': info['done'], 'trunc': info['trunc'], 'obs_next': obs_next, }
        transition_dict = {key: torch.from_numpy(val.astype(np.float32)) for key, val in transition_dict.items()}
        self.memory.add_transition(transition_dict)

    def choose_action(self, observation: torch.Tensor, noisy: bool, y=None,
                      use_target: bool = False, **kwargs) -> torch.Tensor:
        # Check/fix input arguments
        assert len(observation.shape) == 2, 'Agents work with observations of shape (batch, shape)'

        if y is None:
            y = torch.tensor(len(observation) * [self.value_bounds[0]], dtype=torch.float32).reshape(-1, 1)

        # Compute mean and std of the policy
        observation = torch.concatenate([observation, y], dim=-1)

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

    def evaluate_q(self, observation: torch.Tensor, action: torch.Tensor, y=None,
                   use_target: bool = False, use_critic1: bool = True, **kwargs) -> torch.Tensor:
        assert len(observation.shape) == 2, 'Predictor works with observations of shape (batch, shape)'
        if y is None:
            y = torch.tensor(len(observation) * [self.value_bounds[0]], dtype=torch.float32).reshape(-1, 1)
        y = y.clamp(self.value_bounds[0], self.value_bounds[1])
        if use_critic1:
            critic_to_use = self.critic1_target if use_target else self.critic1
        else:
            critic_to_use = self.critic2_target if use_target else self.critic2
        q = critic_to_use(observation.to(DEVICE), action.to(DEVICE), y.to(DEVICE))
        return q

    def evaluate_v(self, observation, y=None, **kwargs):
        best_action = self.choose_action(observation, noisy=False, use_target=False)
        return self.evaluate_q(observation, best_action, y=y, use_target=False, use_critic1=True)

    def _record_video(self, use_target=True):
        self.env_record = self.make_env_fn(record_video=True)
        self.env_record.gym_env.name_prefix = 't=%d_' % self.global_t
        self.env_record.reset()
        episode_finished = False
        running_y = self.value_bounds[0] * np.ones((1, 1), dtype=np.float32)
        while not episode_finished:
            obs_raw = self.env_record.get_observation()
            obs_norm = self.env_record.normalize_obs(obs_raw, update=False).reshape(1, -1)
            obs_norm_th = torch.from_numpy(obs_norm.astype(np.float32))
            # Choose action and evaluate value of the current observation
            with torch.no_grad():
                running_y_th = torch.from_numpy(running_y).to(torch.float32)
                action_th = self.choose_action(obs_norm_th, y=running_y_th, use_target=use_target, noisy=False)
                action = action_th[0].cpu().numpy()
                if self.use_intrinsic_reward:
                    obs_raw_th = torch.from_numpy(obs_raw.astype(np.float32))
                    reward_int = self.int_reward_module.compute_reward(obs_raw_th, action_th, update=True).cpu().numpy()
                else:
                    reward_int = np.zeros((1, 1))

            # Make step in the environment and read the next observation
            info = self.env_record.step(action)
            running_y = np.maximum(info['reward'] + reward_int - self.env.reward_lb, running_y)
            episode_finished = info['done'][0, 0] or info['trunc'][0, 0]

    def _run_test_episodes(self, n_episodes, use_target=False):
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
                action_th = self.choose_action(obs_norm_th, y=running_y_th, use_target=use_target, noisy=False)
                action = action_th.cpu().numpy()
                if self.use_intrinsic_reward:
                    obs_raw_th = torch.from_numpy(obs_raw.astype(np.float32))
                    reward_int = self.int_reward_module.compute_reward(obs_raw_th, action_th, update=True).cpu().numpy()
                else:
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
        # Sample from memory
        sample_dict = self.memory.get_data(batch_size=self.train_batch_size)
        obs = (self.env.normalize_obs(sample_dict['obs'], update=False)).to(torch.float32).to(DEVICE)
        obs_next = (self.env.normalize_obs(sample_dict['obs_next'], update=False)).to(torch.float32).to(DEVICE)
        y_ext_true, action, reward, done, trunc = [(sample_dict[key]).to(DEVICE) for key in ['y_true', 'a',
                                                                                             'r', 'done', 'trunc']]
        reward = reward - self.env.reward_lb
        done = done.to(torch.bool)
        trunc = trunc.to(torch.bool)
        if self.use_intrinsic_reward:
            reward_int = self.int_reward_module.compute_reward(obs, action, update=False).to(DEVICE)

        if self.use_intrinsic_reward and not self.separate_intrinsic_reward_head:
            reward = reward + reward_int

        # Prepare a dictionary to store results of the learning epoch
        epoch_results = defaultdict(list)

        # Reset the noise process for target action smoothing
        self.target_noise_process.reset()

        # Train Q function using y_true times
        y_next = torch.max(y_ext_true, reward) / self.discount

        # Compute target for the value function update
        with torch.no_grad():
            # Target actions are computed using the target policy network; we use smoothing with noise.
            target_actions = self.choose_action(obs_next, y=y_next, use_target=True, noisy=True)
            # Compute the target Q value
            q1_next = self.evaluate_q(obs_next, target_actions, y=y_next,
                                      use_target=True, use_critic1=True)
            q2_next = self.evaluate_q(obs_next, target_actions, y=y_next,
                                      use_target=True, use_critic1=False)
            q_next = torch.max(torch.min(q1_next, q2_next), y_next)
            q_target = done * y_next + ~done * self.discount * q_next
            if self.use_intrinsic_reward and self.separate_intrinsic_reward_head:
                raise NotImplementedError('Separate intrinsic head is not implemented for TD3 yet!')

        # Compute current Q estimates
        q1 = self.evaluate_q(obs, action, y=y_ext_true, use_target=False, use_critic1=True)
        q2 = self.evaluate_q(obs, action, y=y_ext_true, use_target=False, use_critic1=False)

        # Critic loss and update step
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critics_optimizer.zero_grad()
        critic_loss.backward()
        self.critics_optimizer.step()
        epoch_results['critic_loss'].append(critic_loss.cpu().detach().numpy())


        # We count it as one update  anyway
        self.critic_updates += 1

        # Once in `self.policy_update_freq` we perform the policy update step
        if self.critic_updates % self.policy_update_freq == 0:

            # Choose greedy actions
            best_actions = self.choose_action(obs, y=y_ext_true, use_target=False, noisy=False)

            # Measure achieved value
            policy_loss = -self.evaluate_q(obs, best_actions, y=y_ext_true, use_target=False, use_critic1=True).mean()

            # Measure L2 regularization loss
            best_actions_normalized = 2 * (best_actions - self.action_lb_th) / self.actor.bounds_diff - 1
            best_actions_normalized_norm = torch.norm(best_actions_normalized, dim=-1).mean()
            actions_l2_loss = self.action_l2_regularization_weight * best_actions_normalized_norm


            # Optimize actor
            self.actor_optimizer.zero_grad()
            (policy_loss + actions_l2_loss).backward()
            self.actor_optimizer.step()
            epoch_results['policy_loss'].append(policy_loss.cpu().detach().numpy())
            epoch_results['actions_l2'].append(torch.norm(best_actions_normalized_norm,
                                                          dim=-1).cpu().detach().numpy().mean())

            # Update the frozen target models
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for key in epoch_results:
            epoch_results[key] = np.mean(epoch_results[key])

        # Save results of this learning epoch
        for key, val in epoch_results.items():
            self.learning_results[key].append(val)

        # If we did enough of learning steps, it is time to log
        if len(self.learning_results['critic_loss']) == self.env.max_episode_length:
            mean_critic_loss = np.mean(self.learning_results['critic_loss'])
            mean_policy_loss = np.mean(self.learning_results['policy_loss'])
            mean_actions_l2 = np.mean(self.learning_results['actions_l2'])
            if self.use_wandb:
                self.logger.log({'learn/critic_loss': mean_critic_loss}, step=self.global_t)
                self.logger.log({'learn/policy_loss': mean_policy_loss}, step=self.global_t)
                self.logger.log({'learn/actions_l2': mean_actions_l2}, step=self.global_t)
            self.learning_results['critic_loss'] = []
            self.learning_results['policy_loss'] = []
            self.learning_results['actions_l2'] = []

        return epoch_results

    def train(self, print_each: int = 20, record_each: int = None,):
        self.print_each = int(np.ceil(print_each / self.num_envs) * self.num_envs)
        next_record_at = 999999999999 if record_each is None else int(record_each)
        # Reset environment, memory, and run stats.
        self.env.reset()
        self.memory.reset()
        self._initialize_progress_variables()
        value_evaluations = [self.evaluate_v]
        # Run
        running_y = self.value_bounds[0] * np.ones((self.num_envs, 1), dtype=np.float32)
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
                running_y_th = torch.from_numpy(running_y).to(torch.float32)
                action_th = self.choose_action(obs_norm_th, y=running_y_th, use_target=False, noisy=True, )
                action = action_th.cpu().numpy()
                if self.use_intrinsic_reward:
                    obs_raw_th = torch.from_numpy(obs_raw.astype(np.float32))
                    reward_int = self.int_reward_module.compute_reward(obs_raw_th, action_th, update=True).cpu().numpy()
                else:
                    reward_int = np.zeros((self.num_envs, 1))

            # Make step in the environment and read the next observation
            info = self.env.step(action, value_evaluations)  # TODO change value evaluations
            obs_next_raw = info['next_obs_true']
            info['reward_int'] = reward_int
            info['y'] = np.copy(running_y)

            # Save transition to the replay buffer
            self._save_transition(obs_raw, action, obs_next_raw, info)

            running_y = np.maximum(info['reward'] + reward_int - self.env.reward_lb, running_y)
            if 'episode' in info:
                running_y[info['terminated_episode_indices']] = self.value_bounds[0]

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

    def reset_before_episode(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
