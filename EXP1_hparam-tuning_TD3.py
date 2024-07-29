from src.environments.maze.shortest_path_utils import compute_rewards_dict_per_goal, build_reward_shape_fn
from src.rl_agents.td3_family.td3 import TD3Agent
from src.environments.maze.vector_maze_env import GymVectorMazeEnv
from itertools import product


maze_map_name = 'sp_maze_single_goal'
wandb_project_name = 'Hyperparameters-tuning_TD3_%s' % maze_map_name

# Define some general parameters
NUM_ENVS = 16          # We use fixed value of 16 as it provides the most speed up
N_SEEDS = 1            # How many seeds to run for each algorithm/parameters configuration.
use_wandb = True       # Whether to log the results to wandb. Always keep it True unless debugging.
rew_freq = 1           # For tuning, we use fixed reward frequency of 1
beta = 0.9             # Optimal value of beta was determined in the Experiment 0.
subtract_reward_ub = True  # Obtained from Experiment 0

# # # # # # # # # # # # # # # # DEFINE THE MAKE_ENV_FN # # # # # # # # # # # # # # # #
# Create reward function using for the maze map
wandb_tags = ['beta=%s' % beta, 'reward_freq=%s' % rew_freq, 'infinite']
env = GymVectorMazeEnv(num_envs=1, robot='point', maze_map_name=maze_map_name,
                       reward_lb=0, reward_ub=1, reward_type='sparse')
rewards_dict_per_goal = compute_rewards_dict_per_goal(maze_map_name, beta=beta, wp_freq=rew_freq)
reward_shape_fn = build_reward_shape_fn(env.maze, rewards_dict_per_goal, divide_by=1)
env.close()
del env


def make_env_fn(**env_kwargs):
    # Define the function for environment creation. Important: normalize=True for PPO!
    return GymVectorMazeEnv(num_envs=NUM_ENVS, robot='point', reward_lb=0, reward_ub=1,
                            reset_target=False, continuing_task=True, max_episode_steps=1000,
                            normalize=False, reward_shape_fn=reward_shape_fn, save_images_interval=150,
                            maze_map_name=maze_map_name, **env_kwargs)


# Define base values for the  agent's config
TD3_sum_base_config = {'hidden_dims_list': [256, 256], 'env_steps': int(1e6), 'actor_lr': 3e-4, 'critic_lr': 3e-4,
                       'discount': 0.99, 'start_training_after': int(25e3), 'train_batch_size': 256, 'train_freq': 1,
                       'policy_update_freq': 2, 'action_noise_scale': 0.3, 'action_noise_type': 'pink',
                       'subtract_reward_ub': subtract_reward_ub,
                       }


# Define list of values for different parameters.
# ALL COMBINATIONS of these values will be checked

SEARCH_SPACE_JOINT = {'discount': [0.99, 0.995],
                      'action_noise_scale': [0.7, 0.9],
                      'start_training_after': [int(10e3), int(50e3)],
                      }


SHORTCUTS = {'discount': 'dc', 'action_noise_scale': 'noise', 'action_l2_regularization_weight': 'l2',
             'start_training_after': 'start'
             }

keys = list(SEARCH_SPACE_JOINT.keys())
keys_short = [SHORTCUTS[k] if k in SHORTCUTS else k for k in keys]

all_value_combinations = product(*[SEARCH_SPACE_JOINT[k] for k in keys])


# ~90 minutes/5e5 steps for PPOSum
# 12 combinations: 18 hours per seed
# 6 seeds: 3 jobs, 36 hours each

# Iterate over all possible combinations
for combination in all_value_combinations:
    # Create config for the combination
    td3_sum_config = dict(TD3_sum_base_config)
    td3_sum_config.update({keys[i]: combination[i] for i in range(len(combination))})
    # Run
    for _ in range(N_SEEDS):
        # TD3Sum
        run_name = 'TD3SumNegative' if td3_sum_config['subtract_reward_ub'] else 'TD3Sum'
        for i, val in enumerate(combination):
            run_name = run_name + '_%s=%s' % (keys_short[i], val)
        agent = TD3Agent(make_env_fn, wandb_project_name=wandb_project_name, wandb_run_name=run_name,
                         wandb_tags=wandb_tags, use_wandb=use_wandb, **td3_sum_config)
        agent.train(print_each=5)
