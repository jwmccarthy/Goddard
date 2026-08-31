from typing import Self

import torch as th
import torch.nn as nn

from jarl.envs.gym import SyncGymEnv
from jarl.envs.space import observation_space
from jarl.modules.encoder.base import Encoder


class OtherCarAttentionEncoder(Encoder):
    """Adds masked replay-car context to the existing tracker core features."""

    def __init__(
        self,
        core_size: int,
        max_cars: int,
        token_size: int,
        out_dim: int = 512,
        attention_dim: int = 128,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.core_size = core_size
        self.max_cars = max_cars
        self.token_size = token_size
        self.out_dim = out_dim
        self.attention_dim = attention_dim
        self.heads = heads

    def build(self, env: SyncGymEnv) -> Self:
        super().build(env)
        expected = self.core_size + self.max_cars * self.token_size + self.max_cars
        space = observation_space(env)
        if space.flat_dim != expected:
            raise ValueError(
                f"attention encoder expected {expected} observations, got {space.flat_dim}"
            )

        self.core = nn.Sequential(nn.Linear(self.core_size, self.out_dim), nn.ReLU())
        self.query = nn.Linear(self.out_dim, self.attention_dim)
        self.token = nn.Sequential(
            nn.Linear(self.token_size, self.attention_dim),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(
            self.attention_dim,
            self.heads,
            batch_first=True,
        )
        self.context = nn.Linear(self.attention_dim, self.out_dim)

        nn.init.orthogonal_(self.core[0].weight)
        nn.init.zeros_(self.core[0].bias)
        nn.init.zeros_(self.context.weight)
        nn.init.zeros_(self.context.bias)

        self.feats = self.out_dim
        return self

    def forward(self, observation: th.Tensor) -> th.Tensor:
        core = self.core(observation[..., :self.core_size])
        token_end = self.core_size + self.max_cars * self.token_size
        tokens = observation[..., self.core_size:token_end].view(
            *observation.shape[:-1],
            self.max_cars,
            self.token_size,
        )
        valid = observation[..., token_end:].bool()

        # MultiheadAttention cannot consume a row where every key is masked.
        if (~valid.any(dim=-1)).any():
            valid = valid.clone()
            valid[~valid.any(dim=-1), 0] = True

        token_features = self.token(tokens)
        attended, _ = self.attention(
            self.query(core).unsqueeze(-2),
            token_features,
            token_features,
            key_padding_mask=~valid,
            need_weights=False,
        )
        return core + self.context(attended.squeeze(-2))
