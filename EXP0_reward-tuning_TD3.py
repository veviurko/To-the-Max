from src.environments.maze.shortest_path_utils import compute_rewards_dict_per_goal, build_reward_shape_fn
from src.rl_agents.td3_family.td3 import TD3Agent
from src.environments.maze.vector_maze_env import GymVectorMazeEnv

maze_map_name = 'sp_maze_single_goal'
wandb_project_name = 'Reward-tuning_TD3_%s' % maze_map_name

# Define some general parameters
NUM_ENVS = 16      # We use fixed value of 16 as it provides the most speed up
N_SEEDS = 6        # How many seeds to run for each algorithm/parameters configuration.
use_wandb = True   # Whether to log the results to wandb. Always keep it True unless debugging.
rew_freq = 1       # For tuning, we use fixed reward frequency of 1

# Define agent's config
td3_sum_config = {'hidden_dims_list': [256, 256], 'env_steps': int(1e6), 'actor_lr': 3e-4, 'critic_lr': 3e-4,
                  'discount': 0.99, 'start_training_after': int(25e3), 'train_batch_size': 256, 'train_freq': 1,
                  'policy_update_freq': 2, 'action_noise_scale': 0.3, 'action_noise_type': 'pink',
                  'subtract_reward_ub': False}

td3_sum_negative_config = dict(td3_sum_config)
td3_sum_negative_config['subtract_reward_ub'] = True

# Define the search space for betas
betas_to_check = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# ~90 minutes/1e6 steps for TD3

for beta in betas_to_check:

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


    # # # # # # # # # # # # # # # #  Run N_SEEDS # # # # # # # # # # # # # # # #
    for _ in range(N_SEEDS):

        # TD3Sum
        run_name = 'TD3Sum_beta=%s' % beta
        agent = TD3Agent(make_env_fn, wandb_project_name=wandb_project_name, wandb_run_name=run_name,
                         wandb_tags=wandb_tags, use_wandb=use_wandb, **td3_sum_config)
        agent.train(print_each=1)

        # TD3Sum with negative reward
        run_name = 'TD3SumNegative_beta=%s' % beta
        agent = TD3Agent(make_env_fn, wandb_project_name=wandb_project_name, wandb_run_name=run_name,
                         wandb_tags=wandb_tags, use_wandb=use_wandb, **td3_sum_negative_config)
        agent.train(print_each=1)


