import math

import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_size: int, output_size: int, hidden_size: int, layers: int, gain: float):
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


class HistoryMLPCritic(nn.Module):
    def __init__(self, observation_size, length, n_cars, relative, hidden_size, layers):
        super().__init__()
        self.observation_size, self.length, self.relative = observation_size, length, relative
        physics = torch.zeros(observation_size, dtype=torch.bool)
        physics[:9] = True
        for car in range(n_cars):
            start = 9 + 21 * car
            physics[start:start + 15] = True
        self.register_buffer("physics", physics, persistent=False)
        self.network = mlp(length * (observation_size + 1), 1, hidden_size, layers, 1.)

    def forward(self, history):
        frames = history.reshape(-1, self.length, self.observation_size + 1)
        observations, valid = frames[..., :-1], frames[..., -1:]
        if self.relative:
            # Current frame stays absolute; only earlier raw scene physics becomes
            # a temporal delta. Fuel, flags, derived inputs and controller state stay absolute.
            relative = observations - observations[:, -1:]
            earlier = torch.arange(self.length, device=history.device) < self.length - 1
            observations = torch.where(earlier[None, :, None] & self.physics, relative, observations)
        encoded = torch.cat((observations * valid, valid), -1).flatten(1)
        return self.network(encoded)


class Agent(nn.Module):

    def __init__(self, observation_size, action_sizes, action_codec, hidden_size=384, layers=3,
                 *, mlp_critic=False, critic_window_length=4, critic_relative=False, n_cars=None,
                 joint_actions=False):
        super().__init__()
        self.action_sizes = tuple(int(size) for size in action_sizes)
        self.action_codec = action_codec
        self.joint_actions = joint_actions
        if joint_actions:
            table = torch.cartesian_prod(*(torch.arange(size) for size in self.action_sizes))
            offsets = torch.tensor((0, *self.action_sizes[:-1])).cumsum(0)
            strides = torch.tensor([math.prod(self.action_sizes[index + 1:])
                                    for index in range(len(self.action_sizes))])
            self.register_buffer("action_table", table, persistent=False)
            self.register_buffer("action_mask_indices", table + offsets, persistent=False)
            self.register_buffer("action_strides", strides, persistent=False)
        output_size = math.prod(self.action_sizes) if joint_actions else sum(self.action_sizes)
        self.actor = mlp(observation_size, output_size, hidden_size, layers, .01)
        self.mlp_critic = mlp_critic
        if mlp_critic:
            if n_cars is None or critic_window_length < 1:
                raise ValueError("history critic requires car count and positive window length")
            self.critic = HistoryMLPCritic(observation_size, critic_window_length, n_cars,
                                          critic_relative, hidden_size, layers)
        else:
            self.critic = mlp(observation_size, 1, hidden_size, layers, 1.)

    def get_value(self, observations, critic_observations=None):
        if self.mlp_critic and critic_observations is None:
            raise ValueError("history critic requires causal critic observations")
        inputs = critic_observations if self.mlp_critic else observations
        return self.critic(inputs).squeeze(-1).float()

    def get_action_and_log_prob(
        self,
        observations,
        actions=None,
        sample=True,
        entropy_per_head=False,
    ):
        logits = self.actor(observations).float()
        return self.sample_action_logits(logits, observations, actions, sample, entropy_per_head)

    def sample_action_logits(self, logits, observations, actions=None, sample=True, entropy_per_head=False):
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
            log_prob = log_probabilities.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
            entropy = -(log_probabilities.exp() * log_probabilities).sum(-1)
            if entropy_per_head:
                entropy = entropy.unsqueeze(-1)
            return actions, log_prob, entropy
        return sample_actions(
            logits,
            self.action_sizes,
            mask,
            actions,
            sample,
            entropy_per_head,
        )

    def get_action(self, observations, sample=True):
        return self.get_action_and_log_prob(observations, sample=sample)[0]

    def get_action_and_value(
        self,
        observations,
        actions=None,
        entropy_per_head=False,
        critic_observations=None,
    ):
        actions, log_prob, entropy = self.get_action_and_log_prob(
            observations,
            actions,
            entropy_per_head=entropy_per_head,
        )
        return actions, log_prob, entropy, self.get_value(observations, critic_observations)


def sample_actions(
    logits,
    action_sizes,
    mask,
    actions=None,
    sample=True,
    entropy_per_head=False,
):
    logits = logits.masked_fill(~mask, -1e9)
    width = max(action_sizes)
    logits = torch.stack([
        F.pad(head, (0, width - size), value=-1e9)
        for head, size in zip(logits.split(action_sizes, dim=-1), action_sizes)
    ], dim=-2)
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
    return actions, log_prob.squeeze(-1).sum(-1), entropy
