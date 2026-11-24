from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from training.models import mlp

if TYPE_CHECKING:
    from carl.gymnasium.state import StepContext

    from rewards import GameplayDiagnostics
    from training.models import Agent


class ResidualActor(nn.Module):
    """Frozen policy logits plus a zero-initialized, trainable MLP correction."""

    def __init__(
        self,
        base:             nn.Module,
        observation_size: int,
        output_size:      int,
        hidden_size:      int,
        layers:           int,
        scale:            float
    ) -> None:
        super().__init__()
        self.base = base.requires_grad_(False).eval()

        self.correction = mlp(
            observation_size, output_size, hidden_size, layers, gain=0.0
        )

        self.scale = scale

    def train(self, mode: bool = True) -> ResidualActor:
        super().train(mode)
        self.base.eval()

        return self

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.base(observations)

        correction = self.correction(observations)

        return logits.float() + self.scale * correction.float()


def attach_residual_actor(
    agent:            Agent,
    observation_size: int,
    config:           dict[str, Any]
) -> None:
    agent.actor = ResidualActor(
        agent.actor,
        observation_size,
        sum(agent.action_sizes),
        config["residual_hidden_size"],
        config["residual_layers"],
        config["residual_scale"]
    )


class ResidualReward:
    """Team score reward, with optional Nexto shaping and gameplay diagnostics."""

    def __init__(self, gameplay: GameplayDiagnostics, args: Namespace) -> None:
        self.gameplay = gameplay
        self.goal_weight = args.goal_reward_weight
        self.gameplay_weight = args.gameplay_reward_weight

    def __call__(self, context: StepContext) -> torch.Tensor:
        shaped = self.gameplay(context)
        goals = context.events.score_delta[:, None] * context.current.team_sign

        return self.goal_weight * goals + self.gameplay_weight * shaped


def residual_opponent_logits(
    observations: torch.Tensor,
    slots:        torch.Tensor,
    weights:      tuple[torch.Tensor, ...],
    biases:       tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Evaluate an ensemble without per-opponent indexing or host syncs."""
    hidden = observations.unsqueeze(0)

    for index, (weight, bias) in enumerate(zip(weights, biases)):
        hidden = hidden @ weight.transpose(-1, -2) + bias[:, None]

        if index + 1 < len(weights):
            hidden = hidden.relu()

    rows = torch.arange(len(observations), device=observations.device)

    return hidden[slots, rows].float()


class ResidualOpponents:
    """One frozen-base pass and batched corrections for all resident policies."""

    def __init__(self, actor: ResidualActor, args: Namespace) -> None:
        self.base = actor.base
        self.scale = actor.scale
        self.logits = residual_opponent_logits

        if args.compile:
            # Pool membership changes independently of rollout graph lifetimes.
            self.logits = torch.compile(
                self.logits, fullgraph=True, dynamic=True
            )

        self.identities: torch.Tensor | None = None
        self.weights: tuple[torch.Tensor, ...] = ()
        self.biases: tuple[torch.Tensor, ...] = ()

    def refresh(self, actors: dict[int, ResidualActor]) -> None:
        identities = sorted(actors)
        models = [actors[identity].correction for identity in identities]

        layers = [index for index, layer in enumerate(models[0])
                  if isinstance(layer, nn.Linear)]

        self.weights = tuple(
            torch.stack([model[index].weight for model in models])
            for index in layers
        )

        self.biases = tuple(
            torch.stack([model[index].bias for model in models])
            for index in layers
        )

        self.identities = torch.tensor(
            identities, device=self.weights[0].device
        )

    def __call__(
        self,
        observations: torch.Tensor,
        identities:   torch.Tensor
    ) -> torch.Tensor:
        slots = torch.searchsorted(self.identities, identities)

        correction = self.logits(
            observations, slots, self.weights, self.biases
        )

        logits = self.base(observations)

        return logits.float() + self.scale * correction
