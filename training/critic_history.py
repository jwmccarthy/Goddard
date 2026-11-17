"""Causal, reset-aware decision-observation history for a critic only."""
import torch


class ObservationHistory:
    def __init__(self, observation, length, stride):
        self.length, self.stride = length, stride
        self.lag = (length - 1) * stride
        self._frames = observation.new_zeros((len(observation), self.lag, observation.shape[-1])).transpose(0, 1)
        self.age = torch.zeros(len(observation), dtype=torch.long, device=observation.device)
        self.cursor = 0
        self.positions = torch.arange(self.lag, device=observation.device)
        self.sample_positions = self.positions[::stride]
        self.current_valid = torch.ones_like(self.age, dtype=torch.bool)[None]

    @property
    def frames(self):
        if not self.lag:
            return self._frames
        return self._frames[(self.positions + self.cursor) % self.lag]

    @property
    def valid(self):
        return self.positions[:, None] >= self.lag - self.age[None]

    def window(self, observation):
        # Gather only the sampled frames, not a full lag-sized copy per value call.
        if self.lag:
            past = self._frames[(self.sample_positions + self.cursor) % self.lag]
        else:
            past = self._frames
        frames = torch.cat((past, observation[None]), 0)
        past_valid = self.sample_positions[:, None] >= self.lag - self.age[None]
        valid = torch.cat((past_valid, self.current_valid), 0)
        return pack_history(frames.transpose(0, 1), valid.transpose(0, 1))

    def advance(self, observation):
        if self.lag:
            self._frames[self.cursor].copy_(observation)
            self.cursor = (self.cursor + 1) % self.lag
            self.age.add_(1).clamp_max_(self.lag)

    def clear(self, done):
        self.age.masked_fill_(done, 0)


def pack_history(frames, valid):
    frames = frames.masked_fill(~valid[..., None], 0)
    return torch.cat((frames, valid[..., None].to(frames.dtype)), -1).flatten(1)


def rollout_critic_inputs(rollout, indices):
    history = rollout.critic_history
    if rollout.critic_frames is None:
        rollout.critic_frames = torch.cat((rollout.critic_prefix, rollout.observations))
        rollout.critic_boundaries = torch.cat((torch.zeros_like(rollout.critic_prefix_valid), rollout.dones))
    agents = rollout.observations.shape[1]
    step, actor = indices // agents, indices % agents
    offsets = torch.arange(history.length, device=indices.device) * history.stride
    times = step[:, None] + offsets
    frames = rollout.critic_frames[times, actor[:, None]]
    valid = torch.ones_like(times, dtype=torch.bool)
    if history.lag:
        prefix_times = times.clamp_max(history.lag - 1)
        valid &= (times >= history.lag) | rollout.critic_prefix_valid[prefix_times, actor[:, None]]
        scan = step[:, None] + torch.arange(history.lag, device=indices.device)
        boundaries = rollout.critic_boundaries[scan, actor[:, None]]
        last_reset = torch.where(boundaries, scan + 1, 0).amax(1)
        valid &= times >= last_reset[:, None]
    return pack_history(frames, valid)


def timeout_critic_inputs(rollout, indices, final_observation):
    """History at t+1 before clearing the terminal episode at t."""
    history = rollout.critic_history
    if history is None:
        return None

    if rollout.critic_frames is None:
        rollout.critic_frames = torch.cat((rollout.critic_prefix, rollout.observations))
        rollout.critic_boundaries = torch.cat((
            torch.zeros_like(rollout.critic_prefix_valid), rollout.dones,
        ))

    agents = rollout.observations.shape[1]
    step, actor = indices // agents, indices % agents
    offsets = torch.arange(history.length - 1, device=indices.device) * history.stride
    times = step[:, None] + 1 + offsets
    frames = rollout.critic_frames[times, actor[:, None]]
    valid = torch.ones_like(times, dtype=torch.bool)

    if history.lag:
        prefix_times = times.clamp_max(history.lag - 1)
        valid &= (times >= history.lag) | rollout.critic_prefix_valid[prefix_times, actor[:, None]]

    if history.lag > 1:
        scan = step[:, None] + 1 + torch.arange(history.lag - 1, device=indices.device)
        boundaries = rollout.critic_boundaries[scan, actor[:, None]]
        last_reset = torch.where(boundaries, scan + 1, 0).amax(1)
        valid &= times >= last_reset[:, None]

    frames = torch.cat((frames, final_observation[:, None]), 1)
    valid = torch.cat((valid, torch.ones_like(indices[:, None], dtype=torch.bool)), 1)
    return pack_history(frames, valid)
