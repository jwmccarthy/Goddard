"""Causal, reset-aware decision-observation history for a critic only."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from training.rollout import Rollout


class ObservationHistory:

    def __init__(
        self,
        observation: torch.Tensor,
        length:      int,
        stride:      int
    ) -> None:
        self.length, self.stride = length, stride

        self.lag = (length - 1) * stride
        self._frames = observation.new_zeros(
            (len(observation), self.lag, observation.shape[-1])
        ).transpose(0, 1)

        self.age = torch.zeros(
            len(observation),
            dtype=torch.long,
            device=observation.device
        )

        self.cursor = 0
        self.positions = torch.arange(self.lag, device=observation.device)
        self.sample_positions = self.positions[::stride]
        self.current_valid = torch.ones_like(self.age, dtype=torch.bool)[None]

    @property
    def frames(self) -> torch.Tensor:
        if not self.lag:
            return self._frames

        return self._frames[(self.positions + self.cursor) % self.lag]

    @property
    def valid(self) -> torch.Tensor:
        return self.positions[:, None] >= self.lag - self.age[None]

    def window(self, observation: torch.Tensor) -> torch.Tensor:
        # Gather only the sampled frames, not a full lag-sized copy.
        if self.lag:
            slots = (self.sample_positions + self.cursor) % self.lag
            past = self._frames[slots]
        else:
            past = self._frames

        frames = torch.cat((past, observation[None]), 0)

        past_valid = self.sample_positions[:, None] >= self.lag - self.age[None]
        valid = torch.cat((past_valid, self.current_valid), 0)

        return pack_history(frames.transpose(0, 1), valid.transpose(0, 1))

    def advance(self, observation: torch.Tensor) -> None:
        if self.lag:
            self._frames[self.cursor].copy_(observation)
            self.cursor = (self.cursor + 1) % self.lag
            self.age.add_(1).clamp_max_(self.lag)

    def clear(self, done: torch.Tensor) -> None:
        self.age.masked_fill_(done, 0)


def pack_history(frames: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    frames = frames.masked_fill(~valid[..., None], 0)

    return torch.cat(
        (frames, valid[..., None].to(frames.dtype)),
        -1
    ).flatten(1)


def critic_frame_cache(rollout: Rollout) -> None:
    if rollout.critic_frames is not None:
        return

    rollout.critic_frames = torch.cat(
        (rollout.critic_prefix, rollout.observations)
    )
    rollout.critic_boundaries = torch.cat(
        (torch.zeros_like(rollout.critic_prefix_valid), rollout.dones)
    )


def history_frames(
    rollout: Rollout,
    indices: torch.Tensor,
    shift:   int,
    count:   int,
    scan:    int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Frames and validity from step + shift; resets scanned over `scan`."""

    history = rollout.critic_history
    critic_frame_cache(rollout)

    agents = rollout.observations.shape[1]
    step, actor = indices // agents, indices % agents
    offsets = torch.arange(count, device=indices.device) * history.stride

    times = step[:, None] + shift + offsets
    frames = rollout.critic_frames[times, actor[:, None]]
    valid = torch.ones_like(times, dtype=torch.bool)

    if history.lag:
        prefix_times = times.clamp_max(history.lag - 1)
        prefix_valid = rollout.critic_prefix_valid[prefix_times, actor[:, None]]
        valid &= (times >= history.lag) | prefix_valid

    if scan > 0:
        ahead = torch.arange(scan, device=indices.device)
        steps = step[:, None] + shift + ahead
        boundaries = rollout.critic_boundaries[steps, actor[:, None]]
        last_reset = torch.where(boundaries, steps + 1, 0).amax(1)
        valid &= times >= last_reset[:, None]

    return frames, valid


def rollout_critic_inputs(
    rollout: Rollout,
    indices: torch.Tensor
) -> torch.Tensor:
    history = rollout.critic_history
    frames, valid = history_frames(
        rollout, indices, 0, history.length, history.lag
    )

    return pack_history(frames, valid)


def timeout_critic_inputs(
    rollout:           Rollout,
    indices:           torch.Tensor,
    final_observation: torch.Tensor
) -> torch.Tensor | None:
    """History at t+1 before clearing the terminal episode at t."""
    history = rollout.critic_history

    if history is None:
        return None

    frames, valid = history_frames(
        rollout, indices, 1, history.length - 1, history.lag - 1
    )
    frames = torch.cat((frames, final_observation[:, None]), 1)
    valid = torch.cat(
        (valid, torch.ones_like(indices[:, None], dtype=torch.bool)),
        1
    )

    return pack_history(frames, valid)
