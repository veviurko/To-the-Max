from src.neural_networks.util import layer_init
import torch.functional as F
import torch.nn as nn
import torch


EPSILON = 1e-4


class QMaxNetwork(nn.Module):

    def __init__(self,
                 observation_dim: int,
                 action_dim: int,
                 hidden_dims_list: list[int],
                 value_bounds: tuple[float, float],
                 ):
        # Init parent nn.Module class and save the parameters
        super().__init__()
        self.observation_dim = observation_dim      # Size of the observation vector
        self.value_bounds = value_bounds            # Lower- and upper- bound of the value function
        self.action_dim = action_dim                # Size of the action vector
        self.bound_output = True
        # Create the neural network
        self.processing_layers = nn.ModuleList()
        input_shape = observation_dim + action_dim + 1  # An extra '1' is because of the extended MDP
        for hidden_dim in hidden_dims_list:
            self.processing_layers.append(layer_init(nn.Linear(input_shape, hidden_dim)))
            self.processing_layers.append(nn.ReLU())
            input_shape = hidden_dim
        self.Q_head = layer_init(nn.Linear(input_shape, 1), std=1.0)
        self.Q_activation = nn.Tanh()
        self.parameters_list = list(self.parameters())

    def forward(self, observation, action, y):
        """ Input must be tensors """
        x = torch.cat([observation, action, y], -1)
        for layer in self.processing_layers:
            x = layer(x)
        Q_val = self.Q_activation(self.Q_head(x) - 1)
        lb = self.value_bounds[0] - EPSILON
        ub = self.value_bounds[1] + EPSILON
        Q_val = (Q_val * (ub - lb) + (ub + lb)) / 2
        return nn.functional.leaky_relu(Q_val - y - EPSILON, negative_slope=0.01) + y + 0.01 * EPSILON
