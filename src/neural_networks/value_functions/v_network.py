from src.neural_networks.util import layer_init
import torch.nn as nn
import torch


class VNetwork(nn.Module):

    def __init__(self,
                 observation_dim: int,
                 hidden_dims_list: list[int],
                 ):
        # Init parent nn.Module class and save the parameters
        super().__init__()
        self.observation_dim = observation_dim  # Size of the observation vector
        # Create the neural network
        self.processing_layers = nn.ModuleList()
        input_shape = observation_dim
        for hidden_dim in hidden_dims_list:
            self.processing_layers.append(layer_init(nn.Linear(input_shape, hidden_dim)))
            self.processing_layers.append(nn.Tanh())
            input_shape = hidden_dim
        self.V_head = layer_init(nn.Linear(input_shape, 1), std=1.0)
        self.V_activation = lambda x: x
        self.parameters_list = list(self.parameters())


    def forward(self, observation):
        """ Input must be tensors """
        x = observation
        for layer in self.processing_layers:
            x = layer(x)
        V_val = self.V_activation(self.V_head(x))
        return V_val
