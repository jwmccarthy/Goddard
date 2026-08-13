import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Self

from jarl.modules.base import CompositeNet
from jarl.modules.policy import (
    PolicyOutput,
    DiagonalGaussianPolicy,
    MultiCategoricalPolicy
)

from encoder import TransitionEncoder


class ASEDiscriminator(CompositeNet):

    def __init__(
        self,
        foot: nn.Module,
        expert_foot: nn.Module,
        body: nn.Module,
        head: nn.Module = None
    ) -> None:
        super().__init__(
            foot=TransitionEncoder(foot),
            body=body,
            head=head
        )
        self.expert_foot = TransitionEncoder(expert_foot)

    def build(self, env) -> Self:
        super().build(env, out_dim=1)
        if not self.expert_foot.built:
            self.expert_foot.build(env)
        if self.expert_foot.feats != self.foot.feats:
            raise ValueError("agent and expert discriminator features differ")
        return self

    def forward(self, transition) -> th.Tensor:
        features, _ = self.body(self.foot(transition))
        logits = self.head(features).squeeze(-1)
        return logits

    def forward_expert(self, transition) -> th.Tensor:
        features = self.expert_foot(transition)
        features, _ = self.body(features)
        logits = self.head(features).squeeze(-1)
        return logits


class LatentMultiCategoricalPolicy(MultiCategoricalPolicy):

    def _grouped_distributions(self, logits, observation):
        if isinstance(observation, tuple):
            observation = observation[0]
        return super()._grouped_distributions(logits, observation)


class SkillEncoder(CompositeNet):
    def __init__(
        self,
        foot: nn.Module,
        body: nn.Module,
        head: nn.Module = None,
        latent_dim: int = 64,
    ):
        super().__init__(
            foot=TransitionEncoder(foot),
            body=body,
            head=head,
        )

        self.latent_dim = latent_dim

    def build(self, env):
        return super().build(
            env,
            out_dim=self.latent_dim,
        )

    def forward(self, transition):
        features = self.foot(transition)
        output = self.body(features)
        if isinstance(output, tuple):
            output = output[0]
        output = self.head(output)
        return F.normalize(output, p=2, dim=-1)


class LatentGaussianPolicy(DiagonalGaussianPolicy):

    def __init__(
        self,
        foot,
        body,
        head=None,
        latent_dim:   int = 64,
        init_log_std: float = -2.3  # ~0.1 std
    ) -> None:
        super().__init__(foot, body, head)

        self.latent_dim = latent_dim
        self.init_log_std = init_log_std

    def build(self, env) -> Self:
        CompositeNet.build(self, env, out_dim=self.latent_dim)

        self.log_std = th.nn.Parameter(
            th.full((self.latent_dim,), self.init_log_std)
        )

        return self

    def act(
        self,
        observation:   th.Tensor,
        state:         th.Tensor | None = None,
        *,
        deterministic: bool = False
    ) -> PolicyOutput:
        output = super().act(
            observation,
            state,
            deterministic=deterministic
        )

        output["extras"] = F.normalize(
            output.action, p=2, dim=-1
        )

        return output
