from src.environments.maze.vector_maze_env import SUPPORTED_MAPS
import numpy as np


def find_paths_from_goal(i_goal, j_goal, maze_map):
    """ Computes the shortest path from each cell in the maze map to the cell [i_goal, j_goal] """
    def _get_neighbours(i_cell, j_cell):
        all_neighbours = [(i_cell - 1, j_cell), (i_cell + 1, j_cell), (i_cell, j_cell - 1), (i_cell, j_cell + 1)]
        return [(i, j) for (i, j) in all_neighbours if maze_map[i][j] != 1]

    queue = [(i_goal, j_goal, [])]
    paths_dict = {(i_goal, j_goal): []}

    while len(queue):
        i_cell, j_cell, path_cell = queue.pop(0)
        if (i_cell, j_cell) not in paths_dict or len(paths_dict[(i_cell, j_cell)]) > len(path_cell):
            paths_dict[(i_cell, j_cell)] = path_cell
        neighbours = _get_neighbours(i_cell, j_cell)
        neighbours = [(i, j) for (i, j) in neighbours if (i, j) not in paths_dict]
        queue.extend([(i, j, [(i, j)] + path_cell) for (i, j) in neighbours])

    return paths_dict


def precompute_paths_dict_per_goal(maze_map):
    """ Computes shortest paths dictionaries for each possible goal location in the maze """
    I_dim = len(maze_map)
    J_dim = len(maze_map[0])
    possible_goals = [(i, j) for i in range(I_dim) for j in range(J_dim)
                      if maze_map[i][j] != 1 and any([maze_map[i + a][j + b] == 'g'
                                                      for a in [-1, 0, 1] for b in [-1, 0, 1]])]
    print('Possible goals:', possible_goals)
    distances_dict_per_goal = {}
    for i_goal, j_goal in possible_goals:
        paths_dict = find_paths_from_goal(i_goal, j_goal, maze_map)
        distances_dict_per_goal[(i_goal, j_goal)] = paths_dict
    return distances_dict_per_goal


def compute_rewards_dict_per_goal(maze_map_name, beta, wp_freq):
    """ Create reward dictionary for each possible goal location """
    maze_map = SUPPORTED_MAPS[maze_map_name]
    paths_dict_per_goal = precompute_paths_dict_per_goal(maze_map)
    rewards_dict_per_goal = {}
    for (i_goal, j_goal), paths_dict in paths_dict_per_goal.items():
        rewards_dict = {key: beta ** (len(val) + 1) if (len(val) % wp_freq == 0) else 0
                        for key, val in paths_dict.items()}
        rewards_dict_per_goal[(i_goal, j_goal)] = rewards_dict
    return rewards_dict_per_goal


def build_reward_shape_fn(maze, rewards_dict_per_goal, divide_by=1):
    """ Create rewarding shaping function based on the dictionaries """
    def reward_shape_fn_flat(obs, next_obs, reward_true, success, **kwargs):
        """ For non-batched observations """
        xy_goal = obs[-2:]
        xy_pre = obs[:2]
        xy_post = next_obs[:2]

        ij_goal = tuple(maze.cell_xy_to_rowcol(xy_goal))
        ij_post = tuple(maze.cell_xy_to_rowcol(xy_post))
        ij_pre = tuple(maze.cell_xy_to_rowcol(xy_pre))

        r_dict = rewards_dict_per_goal[ij_goal]
        reward_shaped = r_dict[ij_post] / divide_by
        reward_shaped = np.array([reward_shaped])

        return reward_shaped * (~success) + reward_true * success

    def reward_shape_fn(obs, next_obs, reward_true, success, **kwargs):
        """ For batched observations """
        return np.array([reward_shape_fn_flat(obs[i], next_obs[i], reward_true[i], success[i], **kwargs)
                        for i in range(len(obs))])


    return reward_shape_fn


def reward_shape_sparse_fn(obs, next_obs, reward_true, success, **kwargs):
    reward_shaped = np.zeros_like(reward_true)
    return reward_shaped * (~success.reshape(-1)) + reward_true * success.reshape(-1)


def reward_shape_dense_fn(obs, next_obs, reward_true, success, **kwargs):
    xy_goal = obs[:, -2:]
    xy_pre = obs[:, :2]
    xy_post = next_obs[:, :2]
    dist_post = -np.linalg.norm(xy_post - xy_goal, axis=-1).reshape(reward_true.shape)

    return np.exp(dist_post) * (~success.reshape(-1)) + reward_true * success.reshape(-1)

