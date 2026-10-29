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


class Agent(nn.Module):

    def __init__(self, observation_size, action_sizes, action_codec, hidden_size=256, layers=2):
        super().__init__()
        self.action_sizes = tuple(int(size) for size in action_sizes)
        self.action_codec = action_codec
        self.actor = mlp(observation_size, sum(self.action_sizes), hidden_size, layers, 0.01)
        self.critic = mlp(observation_size, 1, hidden_size, layers, 1.0)

    def get_value(self, observations):
        return self.critic(observations).squeeze(-1).float()

    def get_action_and_value(self, observations, actions=None):
        logits = self.actor(observations).float()
        mask = self.action_codec.mask(observations)
        logits = logits.masked_fill(~mask, -1e9)
        width = max(self.action_sizes)
        logits = torch.stack(
            [
                F.pad(head, (0, width - size), value=-1e9)
                for head, size in zip(logits.split(self.action_sizes, dim=-1), self.action_sizes)
            ],
            dim=-2
        )
        log_probabilities = logits.log_softmax(-1)

        if actions is None:
            noise = torch.empty_like(logits).exponential_().log()
            actions = (logits - noise).argmax(-1)

        log_prob = log_probabilities.gather(-1, actions.long().unsqueeze(-1))
        entropy = -(log_probabilities.exp() * log_probabilities).sum((-1, -2))

        return actions, log_prob.squeeze(-1).sum(-1), entropy, self.get_value(observations)
