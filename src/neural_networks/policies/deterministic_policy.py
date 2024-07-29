from src.neural_networks.util import layer_init
import torch.nn as nn
import torch


class DeterministicPolicy(nn.Module):

    def __init__(self,
                 observation_dim: int,
                 action_dim: int,
                 hidden_dims_list: list[int],
                 action_lb: torch.Tensor,
                 action_ub: torch.Tensor):
        # Init parent nn.Module class and save the parameters
        super().__init__()
        self.observation_dim = observation_dim    # Size of the observation vector
        self.action_dim = action_dim              # Size of the action vector
        self.action_lb = action_lb
        self.action_ub = action_ub
        self.bounds_diff = action_ub - action_lb
        self.bounds_sum = action_ub + action_lb
        # Create the neural network
        self.processing_layers = nn.ModuleList()
        input_shape = observation_dim
        for hidden_dim in hidden_dims_list:
            self.processing_layers.append(layer_init(nn.Linear(input_shape, hidden_dim)))
            self.processing_layers.append(nn.ReLU())
            input_shape = hidden_dim
        self.pi_head = layer_init(nn.Linear(input_shape, action_dim), std=0.01)
        self.pi_activation = torch.tanh
        self.parameters_list = list(self.parameters())

    def forward(self, observation):
        """ Input must be tensors """
        x = observation
        for layer in self.processing_layers:
            x = layer(x)
        pi_value = self.pi_activation(self.pi_head(x))
        pi_value = (pi_value * self.bounds_diff + self.bounds_sum) / 2
        return pi_value
