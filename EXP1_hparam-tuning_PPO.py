from src.environments.maze.shortest_path_utils import compute_rewards_dict_per_goal, build_reward_shape_fn
from src.rl_agents.ppo_family.continuous_ppo import ContinuousPPOAgent
from src.environments.maze.vector_maze_env import GymVectorMazeEnv
from itertools import product



maze_map_name = 'sp_maze_single_goal'
wandb_project_name = 'Hyperparameters-tuning_PPO_%s' % maze_map_name

# Define some general parameters
NUM_ENVS = 16        # We use fixed value of 16 as it provides the most speed up
N_SEEDS = 1          # How many seeds to run for each algorithm/parameters configuration.
use_wandb = True     # Whether to log the results to wandb. Always keep it True unless debugging.
rew_freq = 1         # For tuning, we use fixed reward frequency of 1
beta = 0.95           # Optimal value of beta was determined in the Experiment 0.
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
                            normalize=True, reward_shape_fn=reward_shape_fn, save_images_interval=150,
                            maze_map_name=maze_map_name, **env_kwargs)


# Define base values for the  agent's config
ppo_sum_base_config = {'hidden_dims_list': [64, 64], 'env_steps': int(1e6), 'entropy_weight': 0.05, 'lr': 3e-4,
                       'anneal_lr': False, 'discount': 0.99, 'rollout_length': 1000, 'value_weight': 0.5,
                       'policy_weight': 1, 'training_epochs': 10, 'subtract_reward_ub': subtract_reward_ub}

# Define list of values for different parameters.
# ALL COMBINATIONS of these values will be checked

SEARCH_SPACE_JOINT = {'rollout_length': [512, 1024, 2048],
                      'discount': [0.99, 0.995],
                      'lr': [1e-4, 3e-4, 1e-3]

                      }

SHORTCUTS = {'discount': 'dc', 'rollout_length': 'rl', 'lr': 'lr', 'entropy_weight': 'ent'}

keys = sorted(list(SEARCH_SPACE_JOINT.keys()))
keys_short = [SHORTCUTS[k] if k in SHORTCUTS else k for k in keys]

all_value_combinations = product(*[SEARCH_SPACE_JOINT[k] for k in keys])


# ~30 minutes/1e6 steps for PPOSum
# 8 combinations: 4 hours per seed
# 4 seeds: 2 batches, 12 hours each


# Iterate over all possible combinations
for combination in all_value_combinations:
    # Create config for the combination
    ppo_sum_config = dict(ppo_sum_base_config)
    ppo_sum_config.update({keys[i]: combination[i] for i in range(len(combination))})

    # Run
    for _ in range(N_SEEDS):

        # PPOSum
        run_name = 'PPOSumNegative' if ppo_sum_config['subtract_reward_ub'] else 'PPOSum'
        for i, val in enumerate(combination):
            run_name = run_name + '_%s=%s' % (keys_short[i], val)
        agent = ContinuousPPOAgent(make_env_fn, wandb_project_name=wandb_project_name, wandb_run_name=run_name,
                                   wandb_tags=wandb_tags, use_wandb=use_wandb, **ppo_sum_config)
        agent.train(print_each=5)

