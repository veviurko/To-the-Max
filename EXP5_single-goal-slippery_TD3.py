from src.environments.maze.shortest_path_utils import compute_rewards_dict_per_goal, build_reward_shape_fn
from src.environments.maze.vector_maze_env import GymVectorMazeEnv
from src.rl_agents.td3_family.td3max import TD3ConditionalMaxAgent
from src.rl_agents.td3_family.td3 import TD3Agent
from src.rl_agents.td3_family.td3maxdet import TD3MaxDetAgent
from src.rl_agents.ppo_family.continuous_ppo import ContinuousPPOAgent
from src.rl_agents.ppo_family.continuous_ppomax import ContinuousPPOConditionalMaxAgent
import time

from torch.multiprocessing import Pool, set_start_method

maze_map_name = 'sp_maze_single_goal'
wandb_project_name = 'Cells-visitted_SP-reward_%s' % maze_map_name


# Define some general parameters
NUM_ENVS = 16      # We use fixed value of 16 as it provides the most speed up
N_SEEDS = 1        # How many seeds to run for each algorithm/parameters configuration.
use_wandb = True   # Whether to log the results to wandb. Always keep it True unless debugging.
beta = 0.9


# Define agent's config
td3_sum_config = {'hidden_dims_list': [256, 256], 'env_steps': int(1e6), 'actor_lr': 3e-4, 'critic_lr': 3e-4,
                  'discount': 0.99, 'start_training_after': int(25e3), 'train_batch_size': 256, 'train_freq': 1,
                  'policy_update_freq': 2, 'action_noise_scale': 0.7, 'action_noise_type': 'pink',
                  'subtract_reward_ub': True}

td3_max_config = {'hidden_dims_list': [256, 256], 'env_steps': int(1e6), 'actor_lr': 3e-4, 'critic_lr': 3e-4,
                  'discount': 0.995, 'start_training_after': int(25e3), 'train_batch_size': 256, 'train_freq': 1,
                  'policy_update_freq': 2, 'action_noise_scale': 0.7, 'action_noise_type': 'pink',
                  'subtract_reward_ub': False}

td3_conditionalmax_config = {'hidden_dims_list': [256, 256], 'env_steps': int(1e6), 'actor_lr': 3e-4, 'critic_lr': 3e-4,
                             'discount': 0.995, 'start_training_after': int(25e3), 'train_batch_size': 256,
                             'train_freq': 1, 'policy_update_freq': 2, 'action_noise_scale': 0.7,
                             'action_l2_regularization_weight': 0,
                             'action_noise_type': 'pink', 'subtract_reward_ub': False}


ppo_sum_config = {'hidden_dims_list': [64, 64], 'env_steps': int(1e6), 'entropy_weight': 0.05, 'lr': 3e-4,
                  'anneal_lr': False, 'discount': 0.99, 'rollout_length': 1024, 'value_weight': 0.5,
                  'policy_weight': 1, 'training_epochs': 10, 'subtract_reward_ub': True}

ppo_conditionalmax_config = {'hidden_dims_list': [64, 64], 'env_steps': int(1e6), 'entropy_weight': 0.05, 'lr': 3e-4,
                             'anneal_lr': False, 'discount': 0.999, 'rollout_length': 2048, 'value_weight': 0.5,
                             'policy_weight': 1, 'training_epochs': 10, 'subtract_reward_ub': False}


# Define the reward frequencies we are to check
rew_freqs_to_check = [3]

def _run(ind):
    time.sleep(ind * 30)
    for slip_prob in [0]:
        for rew_freq in rew_freqs_to_check:

            # # # # # # # # # # # # # # # # DEFINE THE MAKE_ENV_FN # # # # # # # # # # # # # # # #
            # Create reward function using for the maze map
            wandb_tags = ['beta=%s' % beta, 'reward_freq=%s' % rew_freq, 'infinite', 'slip_prob=%s' % slip_prob]
            env = GymVectorMazeEnv(num_envs=1, robot='point', maze_map_name=maze_map_name,
                                   reward_lb=0, reward_ub=1, reward_type='sparse')
            rewards_dict_per_goal = compute_rewards_dict_per_goal(maze_map_name, beta=beta, wp_freq=rew_freq)
            reward_shape_fn = build_reward_shape_fn(env.maze, rewards_dict_per_goal, divide_by=1)
            env.close()
            del env

            def make_env_fn(**env_kwargs):
                # Define the function for environment creation. Important: normalize=True for PPO!
                return GymVectorMazeEnv(num_envs=NUM_ENVS, robot='point', reward_lb=0, reward_ub=1, slip_prob=slip_prob,
                                        reset_target=False, continuing_task=True, max_episode_steps=1000,
                                        normalize=False, reward_shape_fn=reward_shape_fn, save_images_interval=150,
                                        maze_map_name=maze_map_name, **env_kwargs)

            """
            # TD3ConditionalMax
            run_name = 'TD3ConditionalMax_freq=%s' % rew_freq
            run_name = '%s_slip=%s' % (run_name, slip_prob)
            agent = TD3ConditionalMaxAgent(make_env_fn, wandb_project_name=wandb_project_name, wandb_run_name=run_name,
                                           wandb_tags=wandb_tags, use_wandb=use_wandb, **td3_conditionalmax_config)
            agent.train(print_each=5)

            # TD3MaxDet
            run_name = 'TD3MaxDet_freq=%s' % rew_freq
            run_name = '%s_slip=%s' % (run_name, slip_prob)
            agent = TD3MaxDetAgent(make_env_fn, wandb_project_name=wandb_project_name, wandb_run_name=run_name,
                                   wandb_tags=wandb_tags, use_wandb=use_wandb, **td3_max_config)
            agent.train(print_each=5)

            # TD3Sum
            run_name = 'TD3SumNegative' if td3_sum_config['subtract_reward_ub'] else 'TD3Sum'
            run_name = run_name + '_freq=%s' % rew_freq
            run_name = '%s_slip=%s' % (run_name, slip_prob)
            agent = TD3Agent(make_env_fn, wandb_project_name=wandb_project_name, wandb_run_name=run_name,
                             wandb_tags=wandb_tags, use_wandb=use_wandb, **td3_sum_config)
            agent.train(print_each=5)

             """

            # PPOConditionalMax
            run_name = 'PPOConditionalMax_freq=%s' % rew_freq
            run_name = '%s_slip=%s' % (run_name, slip_prob)
            agent = ContinuousPPOConditionalMaxAgent(make_env_fn, wandb_project_name=wandb_project_name,
                                                     wandb_run_name=run_name, wandb_tags=wandb_tags,
                                                     use_wandb=use_wandb,
                                                     **ppo_conditionalmax_config)
            agent.train(print_each=5)

            # PPOSum
            run_name = 'PPOSumNegative' if ppo_sum_config['subtract_reward_ub'] else 'PPOSum'
            run_name = '%s_freq=%s' % (run_name, rew_freq)
            run_name = '%s_slip=%s' % (run_name, slip_prob)
            agent = ContinuousPPOAgent(make_env_fn, wandb_project_name=wandb_project_name, wandb_run_name=run_name,
                                       wandb_tags=wandb_tags, use_wandb=use_wandb, **ppo_sum_config)
            agent.train(print_each=5)


N_PROCESSES = 3
if __name__ == '__main__':
    set_start_method('spawn')
    pool = Pool(processes=N_PROCESSES)
    process_indices = list(range(N_PROCESSES))
    for _ in range(2):
        _ = pool.map(_run, process_indices)
