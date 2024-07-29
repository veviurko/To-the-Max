from src.noise.util import powerlaw_psd_gaussian
import numpy as np
import torch


class PinkNoiseProcess:

    def __init__(self, batch_size, action_dim, scale, max_episode_length, noise_clip=None, **kwargs):
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.max_episode_length = max_episode_length
        self.scale = scale
        self.t_ind = torch.zeros(batch_size, dtype=torch.int)
        self.data = torch.tensor(powerlaw_psd_gaussian(1, (batch_size, self.action_dim, self.max_episode_length)),
                                 dtype=torch.float32) * self.scale
        self.noise_clip = noise_clip

    def reset(self, scale=None, indices=None):
        self.scale = scale if scale is not None else self.scale
        indices = np.arange(0, self.batch_size) if indices is None else indices
        self.data[indices] = torch.tensor(powerlaw_psd_gaussian(1,
                                                                size=(len(indices), self.action_dim,
                                                                      self.max_episode_length)),
                                          dtype=torch.float32) * self.scale
        self.t_ind[indices] = torch.zeros(len(indices), dtype=torch.int)

    def sample(self):
        assert self.t_ind.max() < self.max_episode_length
        self.t_ind += 1
        result = torch.stack([self.data[b_i, :, t_ind_i - 1] for b_i, t_ind_i in enumerate(self.t_ind)])
        result = result if self.noise_clip is None else result.clamp(-self.noise_clip, self.noise_clip)
        return result
