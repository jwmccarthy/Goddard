from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F


class ActionCodec(Protocol):

    def mask(self, observations: torch.Tensor) -> torch.Tensor: ...


def mlp(
    input_size:  int,
    output_size: int,
    hidden_size: int,
    layers:      int,
    gain:        float
) -> nn.Sequential:
    modules = []

    for _ in range(layers):
        linear = nn.Linear(input_size, hidden_size)

        nn.init.orthogonal_(linear.weight, math.sqrt(2))
        nn.init.zeros_(linear.bias)

        modules.extend((linear, nn.ReLU()))

        input_size = hidden_size

    linear = nn.Linear(input_size, output_size)

    nn.init.orthogonal_(linear.weight, gain)
    nn.init.zeros_(linear.bias)

    return nn.Sequential(*modules, linear)


class Agent(nn.Module):

    def __init__(
        self,
        observation_size: int,
        action_sizes:     tuple[int, ...],
        action_codec:     ActionCodec,
        hidden_size:      int = 512,
        layers:           int = 4,
        *,
        policy_hidden: int | None = None,
        policy_layers: int | None = None,
        critic_hidden: int | None = None,
        critic_layers: int | None = None
    ) -> None:
        super().__init__()
        self.action_sizes = tuple(int(size) for size in action_sizes)
        self.action_codec = action_codec

        policy_hidden = hidden_size if policy_hidden is None else policy_hidden
        policy_layers = layers if policy_layers is None else policy_layers
        critic_hidden = hidden_size if critic_hidden is None else critic_hidden
        critic_layers = layers if critic_layers is None else critic_layers

        self.actor = mlp(
            observation_size,
            sum(self.action_sizes),
            policy_hidden,
            policy_layers,
            0.01
        )

        self.critic = mlp(
            observation_size, 1, critic_hidden, critic_layers, 1.0
        )

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations).squeeze(-1).float()

    def get_action_and_log_prob(
        self,
        observations:     torch.Tensor,
        actions:          torch.Tensor | None = None,
        sample:           bool = True,
        entropy_per_head: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(observations).float()

        return self.sample_action_logits(
            logits,
            observations,
            actions,
            sample,
            entropy_per_head
        )

    def sample_action_logits(
        self,
        logits:           torch.Tensor,
        observations:     torch.Tensor,
        actions:          torch.Tensor | None = None,
        sample:           bool = True,
        entropy_per_head: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return sample_actions(
            logits,
            self.action_sizes,
            self.action_codec.mask(observations),
            actions,
            sample,
            entropy_per_head
        )

    def get_action(
        self,
        observations: torch.Tensor,
        sample:       bool = True
    ) -> torch.Tensor:
        return self.get_action_and_log_prob(observations, sample=sample)[0]

    def get_action_and_value(
        self,
        observations:     torch.Tensor,
        actions:          torch.Tensor | None = None,
        entropy_per_head: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, log_prob, entropy = self.get_action_and_log_prob(
            observations,
            actions,
            entropy_per_head=entropy_per_head
        )

        return (actions, log_prob, entropy, self.get_value(observations))


def sample_actions(
    logits:           torch.Tensor,
    action_sizes:     tuple[int, ...],
    mask:             torch.Tensor,
    actions:          torch.Tensor | None = None,
    sample:           bool = True,
    entropy_per_head: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = logits.masked_fill(~mask, -1e9)
    width = max(action_sizes)

    logits = torch.stack(
        [
            F.pad(head, (0, width - size), value=-1e9)
            for head, size in zip(
                logits.split(action_sizes, dim=-1),
                action_sizes
            )
        ],
        dim=-2
    )

    log_probabilities = logits.log_softmax(-1)

    if actions is None:
        if sample:
            noise = torch.empty_like(logits).exponential_().log()
            actions = (logits - noise).argmax(-1)
        else:
            actions = logits.argmax(-1)

    log_prob = log_probabilities.gather(-1, actions.long().unsqueeze(-1))
    entropy = -(log_probabilities.exp() * log_probabilities).sum(-1)

    if not entropy_per_head:
        entropy = entropy.sum(-1)

    return (actions, log_prob.squeeze(-1).sum(-1), entropy)
