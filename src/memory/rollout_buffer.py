import torch


class RolloutBuffer:
    """ Rollout buffer for on-policy algorithms """
    def __init__(self,
                 batch_size: int,
                 rollout_length: int,
                 scheme: dict[str, tuple[int]]):
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.scheme = scheme
        self.data = {key: torch.nan * torch.ones(self.batch_size, self.rollout_length, *shape, dtype=torch.float32)
                     for key, shape in self.scheme.items()}
        self.current_index = 0

    def reset(self):
        self.data = {key: torch.nan * torch.ones(self.batch_size, self.rollout_length, *shape, dtype=torch.float32)
                     for key, shape in self.scheme.items()}
        self.current_index = 0

    @property
    def is_full(self):
        return self.free_space == 0

    @property
    def free_space(self):
        return self.rollout_length - self.current_index

    def add_transition(self, transition_dict):
        assert self.free_space > 0, 'Trying to add transition to a full rollout buffer!'
        for key, val in transition_dict.items():
            self.data[key][:, self.current_index] = val
        self.current_index += 1

    def get_data(self, keys: list[str] = None, reset: bool = False):
        keys = list(self.scheme.keys()) if keys is None else keys
        sample_dict = {key: self.data[key] for key in keys if key in self.scheme}
        if reset:
            self.reset()
        return sample_dict
