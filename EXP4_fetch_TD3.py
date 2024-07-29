import time

from src.environments.maze.shortest_path_utils import compute_rewards_dict_per_goal, build_reward_shape_fn
from src.environments.gymrobotics_continuous_env import GymRoboticsContinuousEnv
from src.rl_agents.td3_family.td3max import TD3ConditionalMaxAgent
from src.rl_agents.td3_family.td3 import TD3Agent
from src.rl_agents.td3_family.td3maxdet import TD3MaxDetAgent
from src.rl_agents.ppo_family.continuous_ppo import ContinuousPPOAgent
from src.rl_agents.ppo_family.continuous_ppomax import ContinuousPPOConditionalMaxAgent
from src.rl_agents.td3_family.td3 import TD3Agent

# Define some general parameters
wandb_project_name = 'Fetch_final'
env_name = 'FetchPushDense-v2'
use_wandb = True
T = 100
wandb_tags = [env_name, 'r_dense', 'T=%s' % T]
NUM_ENVS = 16
N_SEEDS = 1



def make_env_fn(**env_kwargs):
    env = GymRoboticsContinuousEnv(num_envs=NUM_ENVS, env_name=env_name, normalize=False, reward_lb=-1,
                                   reward_ub=0,  reward_shape_fn=None,
                                   success_key='is_success', max_episode_steps=T, **env_kwargs,
                                   )
    return env


td3_sum_config = {'hidden_dims_list': [256, 256], 'env_steps': int(2e6), 'actor_lr': 3e-4, 'critic_lr': 3e-4,
                  'discount': 0.99, 'start_training_after': int(25e3), 'train_batch_size': 256, 'train_freq': 1,
                  'policy_update_freq': 2, 'action_noise_scale': 0.1, 'action_noise_type': 'pink',
                  'action_l2_regularization_weight': 0.0,
                  'memory_size': int(2e6), 'subtract_reward_ub': False}

td3_conditionalmax_config = {'hidden_dims_list': [256, 256], 'env_steps': int(2e6), 'actor_lr': 3e-4, 'critic_lr': 3e-4,
                             'discount': 0.995, 'start_training_after': int(25e3), 'train_batch_size': 256,
                             'train_freq': 1, 'policy_update_freq': 2, 'action_noise_scale': 0.1,
                             'action_l2_regularization_weight': 0.000,
                             'memory_size': int(2e6), 'action_noise_type': 'pink', 'subtract_reward_ub': False}

# Per run ~2.5 hours/1e6 steps
for _ in range(N_SEEDS):

    # TD3ConditionalMax
    run_name = 'TD3ConditionalMax'
    if td3_conditionalmax_config['action_l2_regularization_weight'] > 0:
        run_name = run_name + '_L2=%s' % td3_conditionalmax_config['action_l2_regularization_weight']
    agent = TD3ConditionalMaxAgent(make_env_fn, wandb_project_name=wandb_project_name,
                                   wandb_run_name=run_name, wandb_tags=wandb_tags, use_wandb=use_wandb,
                                   **td3_conditionalmax_config)
    agent.train(print_each=5, record_each=200000000000)

    time.sleep(60 * 10)
    # TD3Sum
    run_name = 'TD3Sum'
    agent = TD3Agent(make_env_fn, wandb_project_name=wandb_project_name, wandb_run_name=run_name,
                     wandb_tags=wandb_tags, use_wandb=use_wandb, **td3_sum_config)
    agent.train(print_each=5, record_each=2000000000000000)