import numpy as np
import torch


class ReplayBuffer:
    """ Replay buffer for off-policy algorithms """
    def __init__(self, batch_size, scheme, max_size=1000, min_size_to_sample=100):

        self.batch_size = batch_size
        self.scheme = scheme
        self.max_size = max_size
        self.min_size_to_sample = min_size_to_sample
        self.data = {key: torch.nan * torch.ones(self.max_size, *shape, dtype=torch.float32)
                     for key, shape in self.scheme.items()}
        self.current_index = 0
        self.size = 0

    @property
    def can_sample(self):
        return self.size >= self.min_size_to_sample

    @property
    def is_full(self):
        return self.size >= self.max_size

    def add_transition(self, transition_dict):
        for key, val in transition_dict.items():
            self.data[key][self.current_index: self.current_index + self.batch_size] = val
        if not self.is_full:
            self.size = min(self.max_size, self.size + self.batch_size)
        self.current_index += self.batch_size
        self.current_index %= self.max_size

    def get_data(self, batch_size: int, keys: list[str] = None):
        indexes = np.random.randint(0, self.size, size=batch_size)
        keys = self.scheme.keys() if keys is None else keys
        sample_dict = {key: self.data[key][indexes] for key in keys}
        return sample_dict

    def reset(self):
        self.data = {key: torch.nan * torch.ones(self.max_size, *shape, dtype=torch.float32)
                     for key, shape in self.scheme.items()}
        self.current_index = 0
        self.size = 0
