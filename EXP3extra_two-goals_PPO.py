from src.environments.maze.shortest_path_utils import reward_shape_sparse_fn, reward_shape_dense_fn
from src.environments.maze.vector_maze_env import GymVectorMazeEnv
from src.rl_agents.td3_family.td3max import TD3ConditionalMaxAgent
from src.rl_agents.td3_family.td3 import TD3Agent
from src.rl_agents.td3_family.td3maxdet import TD3MaxDetAgent
from src.rl_agents.ppo_family.continuous_ppo import ContinuousPPOAgent
from src.rl_agents.ppo_family.continuous_ppomax import ContinuousPPOConditionalMaxAgent


maze_map_name = 'sp_maze_two_goals'
wandb_project_name = 'SP-reward_%s' % maze_map_name
# Define some general parameters
NUM_ENVS = 16  # We use fixed value of 16 as it provides the most speed up
N_SEEDS = 1  # How many seeds to run for each algorithm/parameters configuration.
use_wandb = True  # Whether to log the results to wandb. Always keep it True unless debugging.


# Define agent's config
ppo_sum_config = {'hidden_dims_list': [64, 64], 'env_steps': int(1e6), 'entropy_weight': 0.05, 'lr': 3e-4,
                  'anneal_lr': False, 'discount': 0.99, 'rollout_length': 1024, 'value_weight': 0.5,
                  'policy_weight': 1, 'training_epochs': 10, 'subtract_reward_ub': True}

ppo_max_config = {'hidden_dims_list': [64, 64], 'env_steps': int(1e6), 'entropy_weight': 0.05, 'lr': 3e-4,
                  'anneal_lr': False, 'discount': 0.999, 'rollout_length': 2048, 'value_weight': 0.5,
                  'policy_weight': 1, 'training_epochs': 10, 'subtract_reward_ub': False}

ppo_conditionalmax_config = {'hidden_dims_list': [64, 64], 'env_steps': int(1e6), 'entropy_weight': 0.05, 'lr': 3e-4,
                             'anneal_lr': False, 'discount': 0.999, 'rollout_length': 2048, 'value_weight': 0.5,
                             'policy_weight': 1, 'training_epochs': 10, 'subtract_reward_ub': False}


# # # # # # # # # # # # # # # # DEFINE THE MAKE_ENV_FN # # # # # # # # # # # # # # # #
# Create reward function using for the maze map
def make_env_fn_sparse(**env_kwargs):
    # Define the function for environment creation. Important: normalize=True for PPO!
    return GymVectorMazeEnv(num_envs=NUM_ENVS, robot='point', reward_lb=0, reward_ub=1,
                            reset_target=False, continuing_task=True, max_episode_steps=1000,
                            normalize=True, reward_shape_fn=reward_shape_sparse_fn, save_images_interval=150,
                            maze_map_name=maze_map_name, **env_kwargs)


def make_env_fn_dense(**env_kwargs):
    # Define the function for environment creation. Important: normalize=True for PPO!
    return GymVectorMazeEnv(num_envs=NUM_ENVS, robot='point', reward_lb=0, reward_ub=1,
                            reset_target=False, continuing_task=True, max_episode_steps=1000,
                            normalize=True, reward_shape_fn=reward_shape_dense_fn, save_images_interval=150,
                            maze_map_name=maze_map_name, **env_kwargs)


# Four runs per seed = 2 hours.
# 5 seeds = 10 hours.

# # # # # # # # # # # # # # # #  Run N_SEEDS # # # # # # # # # # # # # # # #
for _ in range(N_SEEDS):

    # # # # # # # # # # # # # SPARSE # # # # # # # # # # # # #
    wandb_tags = ['sparse', 'infinite']

    # PPOConditionalMax
    run_name = 'PPOConditionalMax_sparse'
    agent = ContinuousPPOConditionalMaxAgent(make_env_fn_sparse, wandb_project_name=wandb_project_name,
                                             wandb_run_name=run_name, wandb_tags=wandb_tags,
                                             use_wandb=use_wandb, **ppo_conditionalmax_config)
    agent.train(print_each=5)

    # PPOSum
    run_name = 'PPOSumNegative_sparse'
    agent = ContinuousPPOAgent(make_env_fn_sparse, wandb_project_name=wandb_project_name, wandb_run_name=run_name,
                               wandb_tags=wandb_tags, use_wandb=use_wandb, **ppo_sum_config)
    agent.train(print_each=5)



    # # # # # # # # # # # # # DENSE # # # # # # # # # # # # #
    wandb_tags = ['dense-exp', 'infinite']

    # PPOConditionalMax
    run_name = 'PPOConditionalMax_dense-exp'
    agent = ContinuousPPOConditionalMaxAgent(make_env_fn_dense, wandb_project_name=wandb_project_name,
                                             wandb_run_name=run_name, wandb_tags=wandb_tags,
                                             use_wandb=use_wandb, **ppo_conditionalmax_config)
    agent.train(print_each=5)

    # PPOSum
    run_name = 'PPOSumNegative_dense-exp'
    agent = ContinuousPPOAgent(make_env_fn_dense, wandb_project_name=wandb_project_name, wandb_run_name=run_name,
                               wandb_tags=wandb_tags, use_wandb=use_wandb, **ppo_sum_config)
    agent.train(print_each=5)


