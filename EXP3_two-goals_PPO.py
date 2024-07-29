from src.environments.maze.shortest_path_utils import compute_rewards_dict_per_goal, build_reward_shape_fn
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
beta = .95

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

# Define the reward frequencies we are to check
rew_freqs_to_check = [1]

# ~30 minutes/1e6 steps for PPO
# 4 frequencies, 3 algorithms: 6 hours per seed
# 10 seeds = 40 hours = 2 jobs 20 hours each

for rew_freq in rew_freqs_to_check:

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


    # # # # # # # # # # # # # # # #  Run N_SEEDS # # # # # # # # # # # # # # # #
    for _ in range(N_SEEDS):

        # PPOSum
        run_name = 'PPOSumNegative' if ppo_sum_config['subtract_reward_ub'] else 'PPOSum'
        run_name = '%s_freq=%s' % (run_name, rew_freq)
        agent = ContinuousPPOAgent(make_env_fn, wandb_project_name=wandb_project_name, wandb_run_name=run_name,
                                   wandb_tags=wandb_tags, use_wandb=use_wandb, **ppo_sum_config)
        agent.train(print_each=5)
        1/0

        # PPOConditionalMax
        run_name = 'PPOConditionalMax_freq=%s' % rew_freq
        agent = ContinuousPPOConditionalMaxAgent(make_env_fn, wandb_project_name=wandb_project_name,
                                                 wandb_run_name=run_name, wandb_tags=wandb_tags, use_wandb=use_wandb,
                                                 **ppo_conditionalmax_config)
        agent.train(print_each=5)



