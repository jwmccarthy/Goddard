"""Episode-persistent categorical policy inputs, never discriminator features."""

import numpy as np
import torch
from gymnasium.spaces import Box
from gymnasium.vector.utils import batch_space
from torch.nn import functional as F

from training.tensors import plain_tensor


class LatentObservationEnv:

    def __init__(self, env, code_count):
        if code_count < 2:
            raise ValueError("info requires at least two latent codes")
        self.env = env
        self.code_count = code_count
        self.codes = None
        space = env.single_observation_space
        self.single_observation_space = Box(
            low=np.concatenate((space.low, np.zeros(code_count, dtype=space.dtype))),
            high=np.concatenate((space.high, np.ones(code_count, dtype=space.dtype))),
            dtype=space.dtype,
        )
        self.observation_space = batch_space(self.single_observation_space, env.n_envs)

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def reset_state_provider(self):
        return self.env.reset_state_provider

    @reset_state_provider.setter
    def reset_state_provider(self, value):
        self.env.reset_state_provider = value

    def condition(self, observations, codes):
        observations = plain_tensor(observations)
        return torch.cat((observations, F.one_hot(codes, self.code_count).to(observations.dtype)), dim=-1)

    def reset(self):
        observation = plain_tensor(self.env.reset())
        self.codes = torch.randint(self.code_count, (len(observation),), device=observation.device)
        return self.condition(observation, self.codes)

    def step(self, actions):
        observation, reward, terminated, truncated, info = self.env.step(actions)
        # Timeout bootstraps must retain the OLD episode's code, not the reset's.
        if "final_obs" in info:
            info = dict(info)
            info["final_obs"] = self.condition(info["final_obs"], self.codes)
        done = terminated | truncated
        replacement = torch.randint(self.code_count, self.codes.shape, device=self.codes.device)
        self.codes = torch.where(done, replacement, self.codes)
        return self.condition(observation, self.codes), reward, terminated, truncated, info
