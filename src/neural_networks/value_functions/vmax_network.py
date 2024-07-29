from src.neural_networks.util import layer_init
import torch.nn as nn
import torch


EPSILON = 1e-4


class VMaxNetwork(nn.Module):

    def __init__(self,
                 observation_dim: int,
                 hidden_dims_list: list[int],
                 value_bounds: tuple[float, float],
                 ):
        # Init parent nn.Module class and save the parameters
        super().__init__()
        self.observation_dim = observation_dim  # Size of the observation vector
        self.value_bounds = value_bounds
        # Create the neural network
        self.processing_layers = nn.ModuleList()
        input_shape = observation_dim + 1
        for hidden_dim in hidden_dims_list:
            self.processing_layers.append(layer_init(nn.Linear(input_shape, hidden_dim)))
            self.processing_layers.append(nn.Tanh())
            input_shape = hidden_dim
        self.V_head = layer_init(nn.Linear(input_shape, 1), std=1.0)
        self.V_activation = nn.Tanh()
        self.parameters_list = list(self.parameters())


    def forward(self, observation, y):
        """ Input must be tensors """
        x = torch.concat([observation, y], dim=-1)
        for layer in self.processing_layers:
            x = layer(x)
        V_val = self.V_activation(self.V_head(x))
        lb = self.value_bounds[0] - EPSILON
        ub = self.value_bounds[1] + EPSILON
        V_val = (V_val * (ub - lb) + (ub + lb)) / 2
        return nn.functional.leaky_relu(V_val - y - EPSILON, negative_slope=0.01) + y + 0.01 * EPSILON
