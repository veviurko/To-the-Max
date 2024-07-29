from src.neural_networks.util import layer_init
import torch.nn as nn
import torch


class QNetwork(nn.Module):

    def __init__(self,
                 observation_dim: int,
                 action_dim: int,
                 hidden_dims_list: list[int],
                 ):
        # Init parent nn.Module class and save the parameters
        super().__init__()
        self.observation_dim = observation_dim  # Size of the observation vector
        self.action_dim = action_dim            # Size of the action vector
        # Create the neural network
        self.processing_layers = nn.ModuleList()
        input_shape = observation_dim + action_dim
        for hidden_dim in hidden_dims_list:
            self.processing_layers.append(nn.Linear(input_shape, hidden_dim))
            self.processing_layers.append(nn.ReLU())
            input_shape = hidden_dim
        self.Q_head = nn.Linear(input_shape, 1)
        self.Q_activation = lambda x: x
        self.parameters_list = list(self.parameters())

    def forward(self, observation, action):
        """ Input must be tensors """
        x = torch.cat([observation, action], -1)
        for layer in self.processing_layers:
            x = layer(x)
        q_val = self.Q_activation(self.Q_head(x))  # - 1
        return q_val
