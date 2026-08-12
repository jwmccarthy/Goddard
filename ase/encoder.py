from typing import Self

import torch as th

from jarl.envs.space import observation_space
from jarl.modules.encoder.base import Encoder

from config import CARL_LAYOUT, REPLAY_LAYOUT


DISCRIMINATOR_FEATURES = 38


def _index_tensor(indices: tuple[int, ...]) -> th.Tensor:
    return th.tensor(indices, dtype=th.long)


def _validate(observation: th.Tensor, expected: int, source: str) -> None:
    if observation.shape[-1] != expected:
        raise ValueError(
            f"Expected {source} observation with {expected} features, "
            f"got {observation.shape[-1]}"
        )


class LatentEncoder(Encoder):

    def __init__(self, latents) -> None:
        super().__init__()
        self.latents = latents

    def build(self, env) -> Self:
        super().build(env)

        space = observation_space(env)

        self.obs_start_dim = -len(space.shape)
        self.feats = space.flat_dim + self.latents.latent_dim

        return self

    def forward(
        self,
        inputs: th.Tensor | tuple[th.Tensor, th.Tensor]
    ) -> th.Tensor:
        if isinstance(inputs, tuple):
            obs, latent = inputs
        else:
            obs = inputs
            latent = self.latents.latent

        if latent is None:
            raise RuntimeError("latent capture must be reset before policy evaluation")

        obs = th.flatten(obs, start_dim=self.obs_start_dim)

        if obs.shape[:-1] != latent.shape[:-1]:
            raise ValueError("observation and latent batch shapes differ")
        
        return th.cat((obs, latent), dim=-1)


class TransitionEncoder(Encoder):

    def __init__(self, obs_encoder: Encoder):
        super().__init__()
        self.obs_encoder = obs_encoder

    def build(self, env):
        super().build(env)

        if not self.obs_encoder.built:
            self.obs_encoder.build(env)

        self.feats = 2 * self.obs_encoder.feats

        return self

    def forward(self, transition: tuple[th.Tensor, th.Tensor]) -> th.Tensor:
        x_t0, x_t1 = transition

        return th.cat(
            (self.obs_encoder(x_t0), self.obs_encoder(x_t1)),
            dim=-1
        )


class CARLDiscriminatorEncoder(Encoder):
    
    def __init__(self) -> None:
        super().__init__()

        self.register_buffer(
            "indices",
            _index_tensor(
                CARL_LAYOUT.base
                + CARL_LAYOUT.ego_ball_relative
                + CARL_LAYOUT.ego_other_relative
            ),
            persistent=False
        )
        self.feats = DISCRIMINATOR_FEATURES

    def build(self, env) -> Self:
        super().build(env)

        size = observation_space(env).flat_dim

        if size != CARL_LAYOUT.size:
            raise ValueError(
                f"Expected 1v1 CARL observation with {CARL_LAYOUT.size} features, "
                f"got {size}"
            )
        
        return self

    def forward(self, observation: th.Tensor) -> th.Tensor:
        _validate(observation, CARL_LAYOUT.size, "CARL")
        return observation.index_select(-1, self.indices)


class ExpertDiscriminatorEncoder(Encoder):

    def __init__(self) -> None:
        super().__init__()

        self.register_buffer(
            "indices",
            _index_tensor(
                REPLAY_LAYOUT.base
                + REPLAY_LAYOUT.ego_ball_relative
                + REPLAY_LAYOUT.ego_other_relative
            ),
            persistent=False
        )
        self.feats = DISCRIMINATOR_FEATURES

    def build(self, env) -> Self:
        super().build(env)
        return self

    def forward(self, observation: th.Tensor) -> th.Tensor:
        _validate(observation, REPLAY_LAYOUT.size, "expert replay")
        return observation.index_select(-1, self.indices)
