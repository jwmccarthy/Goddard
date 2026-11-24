from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F

from training.recurrent import GRUSequence


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


class GRUModel(nn.Module):
    """GRU with batched projections, fused gates, and batched gradients."""

    def __init__(
        self,
        observation_size: int,
        output_size:      int,
        hidden_size:      int,
        layers:           int,
        gain:             float = 0.01
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = mlp(
            observation_size, 3 * hidden_size, hidden_size, layers, 1.0
        )
        self.recurrent = nn.Linear(hidden_size, 3 * hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        for weight in self.recurrent.weight.chunk(3):
            nn.init.orthogonal_(weight)

        nn.init.zeros_(self.recurrent.bias)
        nn.init.orthogonal_(self.output.weight, gain)
        nn.init.zeros_(self.output.bias)

    def cell(
        self,
        projected: torch.Tensor,
        state:     torch.Tensor
    ) -> torch.Tensor:
        # GEMMs use autocast; gates and persistent memory accumulate in FP32.
        inputs = projected.float().chunk(3, dim=-1)
        hidden = self.recurrent(state).float().chunk(3, dim=-1)
        reset = (inputs[0] + hidden[0]).sigmoid()
        update = (inputs[1] + hidden[1]).sigmoid()
        candidate = (inputs[2] + reset * hidden[2]).tanh()

        return candidate + update * (state - candidate)

    def forward(
        self,
        observations: torch.Tensor,
        state:        torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.cell(self.encoder(observations), state)

        return self.output(state), state

    def sequence(
        self,
        observations: torch.Tensor,
        state:        torch.Tensor,
        resets:       torch.Tensor
    ) -> torch.Tensor:
        states = self.sequence_states(observations, state, resets)

        return self.output(states.flatten(0, 1))

    def sequence_states(
        self,
        observations: torch.Tensor,
        state:        torch.Tensor,
        resets:       torch.Tensor
    ) -> torch.Tensor:
        # One large encoder/projection batch instead of a GEMM per timestep.
        projected = self.encoder(observations.flatten(0, 1))
        projected = projected.reshape(*observations.shape[:2], -1)

        return GRUSequence.apply(
            projected,
            state,
            resets,
            self.recurrent.weight.to(projected.dtype),
            self.recurrent.bias.to(projected.dtype)
        )


class HistoryMLPCritic(nn.Module):

    def __init__(
        self,
        observation_size: int,
        length:           int,
        n_cars:           int,
        relative:         bool,
        hidden_size:      int,
        layers:           int
    ) -> None:
        super().__init__()
        self.observation_size = observation_size
        self.length = length
        self.relative = relative

        physics = torch.zeros(observation_size, dtype=torch.bool)
        physics[:9] = True

        for car in range(n_cars):
            start = 9 + 21 * car
            physics[start : start + 15] = True

        self.register_buffer("physics", physics, persistent=False)

        self.network = mlp(
            length * (observation_size + 1),
            1,
            hidden_size,
            layers,
            1.0
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        frames = history.reshape(-1, self.length, self.observation_size + 1)

        observations, valid = frames[..., :-1], frames[..., -1:]

        if self.relative:
            # Current frame stays absolute; only earlier raw scene physics
            # becomes a temporal delta. Fuel, flags, derived inputs and
            # controller state stay absolute.
            relative = observations - observations[:, -1:]
            earlier = (
                torch.arange(self.length, device=history.device)
                < self.length - 1
            )

            observations = torch.where(
                earlier[None, :, None] & self.physics,
                relative,
                observations
            )

        encoded = torch.cat((observations * valid, valid), -1).flatten(1)

        return self.network(encoded)


class Agent(nn.Module):

    def __init__(
        self,
        observation_size: int,
        action_sizes:     tuple[int, ...],
        action_codec:     ActionCodec,
        hidden_size:      int = 512,
        layers:           int = 4,
        *,
        policy_hidden:        int | None = None,
        policy_layers:        int | None = None,
        critic_hidden:        int | None = None,
        critic_layers:        int | None = None,
        mlp_critic:           bool = False,
        critic_window_length: int = 4,
        critic_relative:      bool = False,
        n_cars:               int | None = None,
        joint_actions:        bool = False,
        gru_policy:           bool = False,
        gru_critic:           bool = False
    ) -> None:
        super().__init__()
        self.action_sizes = tuple(int(size) for size in action_sizes)
        self.action_codec = action_codec
        self.joint_actions = joint_actions
        self.gru_policy = gru_policy
        self.gru_critic = gru_critic
        policy_hidden = hidden_size if policy_hidden is None else policy_hidden
        policy_layers = layers if policy_layers is None else policy_layers
        critic_hidden = hidden_size if critic_hidden is None else critic_hidden
        critic_layers = layers if critic_layers is None else critic_layers

        if joint_actions:
            axes = (torch.arange(size) for size in self.action_sizes)
            table = torch.cartesian_prod(*axes)
            offsets = torch.tensor((0, *self.action_sizes[:-1])).cumsum(0)

            strides = torch.tensor(
                [
                    math.prod(self.action_sizes[index + 1 :])
                    for index in range(len(self.action_sizes))
                ]
            )

            self.register_buffer("action_table", table, persistent=False)
            self.register_buffer(
                "action_mask_indices",
                table + offsets,
                persistent=False
            )
            self.register_buffer("action_strides", strides, persistent=False)

        output_size = (
            math.prod(self.action_sizes)
            if joint_actions else sum(self.action_sizes)
        )

        if gru_policy:
            self.actor = GRUModel(
                observation_size, output_size, policy_hidden, policy_layers
            )
        else:
            self.actor = mlp(
                observation_size,
                output_size,
                policy_hidden,
                policy_layers,
                0.01
            )

        self.mlp_critic = mlp_critic

        if gru_critic:
            self.critic = GRUModel(
                observation_size, 1, critic_hidden, critic_layers, gain=1.0
            )
        elif mlp_critic:
            self.critic = HistoryMLPCritic(
                observation_size,
                critic_window_length,
                n_cars,
                critic_relative,
                critic_hidden,
                critic_layers
            )
        else:
            self.critic = mlp(
                observation_size, 1, critic_hidden, critic_layers, 1.0
            )

    def get_value(
        self,
        observations:        torch.Tensor,
        critic_observations: torch.Tensor | None = None,
        critic_state:        torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.gru_critic:
            return self.get_recurrent_value(observations, critic_state)[0]

        inputs = critic_observations if self.mlp_critic else observations

        return self.critic(inputs).squeeze(-1).float()

    def get_recurrent_value(
        self,
        observations: torch.Tensor,
        state:        torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values, state = self.critic(observations, state)

        return values.squeeze(-1).float(), state

    def get_recurrent_critic_action_and_value(
        self,
        observations: torch.Tensor,
        critic_state: torch.Tensor,
        policy_state: torch.Tensor | None = None
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor | None, torch.Tensor
    ]:
        if self.gru_policy:
            logits, policy_state = self.actor(observations, policy_state)
        else:
            logits = self.actor(observations)

        actions, log_prob, entropy = self.sample_action_logits(
            logits.float(), observations
        )
        values, critic_state = self.get_recurrent_value(
            observations, critic_state
        )

        return actions, log_prob, entropy, values, policy_state, critic_state

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
        mask = self.action_codec.mask(observations)

        if self.joint_actions:
            # A combination is legal exactly when every constituent control is legal.
            legal = mask[..., self.action_mask_indices[:, 0]]

            for head in range(1, len(self.action_sizes)):
                legal = legal & mask[..., self.action_mask_indices[:, head]]

            logits = logits.masked_fill(~legal, -1e9)
            log_probabilities = logits.log_softmax(-1)

            if actions is None:
                if sample:
                    noise = torch.empty_like(logits).exponential_().log()
                    indices = (logits - noise).argmax(-1)
                else:
                    indices = logits.argmax(-1)

                actions = self.action_table[indices]
            else:
                indices = (actions.long() * self.action_strides).sum(-1)

            log_prob = log_probabilities.gather(
                -1,
                indices.unsqueeze(-1)
            ).squeeze(-1)
            entropy = -(log_probabilities.exp() * log_probabilities).sum(-1)

            if entropy_per_head:
                entropy = entropy.unsqueeze(-1)

            return (actions, log_prob, entropy)

        return sample_actions(
            logits,
            self.action_sizes,
            mask,
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
        observations:        torch.Tensor,
        actions:             torch.Tensor | None = None,
        entropy_per_head:    bool = False,
        critic_observations: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, log_prob, entropy = self.get_action_and_log_prob(
            observations,
            actions,
            entropy_per_head=entropy_per_head
        )

        return (
            actions,
            log_prob,
            entropy,
            self.get_value(observations, critic_observations)
        )

    def get_recurrent_action(
        self,
        observations: torch.Tensor,
        state:        torch.Tensor,
        sample:       bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, state = self.actor(observations, state)
        actions = self.sample_action_logits(
            logits.float(), observations, sample=sample
        )[0]

        return actions, state

    def get_recurrent_action_and_value(
        self,
        observations:        torch.Tensor,
        state:               torch.Tensor,
        critic_observations: torch.Tensor | None = None
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        logits, state = self.actor(observations, state)
        actions, log_prob, entropy = self.sample_action_logits(
            logits.float(), observations
        )

        return (
            actions, log_prob, entropy,
            self.get_value(observations, critic_observations), state
        )


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
