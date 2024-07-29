import torch


class WhiteNoiseProcess:

    def __init__(self, batch_size, action_dim, scale, max_episode_length, noise_clip=None, **kwargs):
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.max_episode_length = max_episode_length
        self.scale = scale
        self.data = None
        self.noise_clip = noise_clip

    def reset(self, scale=None, **kwargs):
        self.scale = scale if scale is not None else self.scale

    def sample(self):
        result = torch.randn((self.batch_size, self.action_dim)) * self.scale
        result = result if self.noise_clip is None else result.clamp(-self.noise_clip, self.noise_clip)
        return result

