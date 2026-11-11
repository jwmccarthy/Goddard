import math

import torch
from torch import nn
from torch.nn import functional as F

from imitation.models import WindowHead


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


def policy_window_length(config):
    """Read independent history, preserving the old coupling for legacy checkpoints."""
    values = config if isinstance(config, dict) else vars(config)
    return values.get("policy_window_length", values.get("discriminator_window_length", 1))


def critic_hidden_size(config):
    """Old GRU critics had a fixed width of 64; new configs store an explicit width."""
    values = config if isinstance(config, dict) else vars(config)
    if "critic_hidden_size" in values:
        width = values["critic_hidden_size"]
        return values["hidden_size"] if width is None else width
    return 64 if values.get("gru_critic", values.get("gru", False)) else values["hidden_size"]


def critic_layers(config):
    """MLP critic depth; legacy checkpoints used the recorded shared layer count."""
    values = config if isinstance(config, dict) else vars(config)
    depth = values.get("critic_layers")
    return values["layers"] if depth is None else depth


class ObservationWindowMLP(nn.Module):
    """Ordered raw observation window, with explicit valid-frame indicators."""

    def __init__(self, observation_size, output_size, hidden_size, layers, window_length, gain):
        super().__init__()
        self.window_length = window_length
        self.network = mlp((observation_size + 1) * window_length, output_size, hidden_size, layers, gain)

    def forward(self, observations):
        if observations.ndim != 3 or not 0 < observations.shape[1] <= self.window_length:
            raise ValueError("windowed MLP requires a batch of observation histories within its configured window")
        padding = torch.isnan(observations[..., 0])
        frames = torch.nan_to_num(observations).masked_fill(padding[..., None], 0)
        features = torch.cat((frames, (~padding)[..., None].to(frames.dtype)), dim=-1)
        features = F.pad(features, (0, 0, self.window_length - observations.shape[1], 0))
        return self.network(features.flatten(1))


class Agent(nn.Module):

    def __init__(self, observation_size, action_sizes, action_codec, hidden_size=384, layers=3,
                 critic_window_length=1, gru_critic=False, gru_policy=False, info_codes=0, *,
                 policy_window_length=None, critic_hidden_size=None, mlp_policy=False, mlp_critic=False,
                 critic_layers=None):
        super().__init__()
        if (gru_policy and mlp_policy) or (gru_critic and mlp_critic):
            raise ValueError("GRU and windowed MLP are mutually exclusive for each actor/critic head")
        length = critic_window_length if policy_window_length is None else policy_window_length
        if length < 1:
            raise ValueError("policy window length must be at least 1")
        self.policy_window_length = length
        self.critic_window_length = length  # Compatibility alias; both windowed heads share this history.
        self.critic_hidden_size = hidden_size if critic_hidden_size is None else critic_hidden_size
        if self.critic_hidden_size < 1:
            raise ValueError("critic hidden size must be positive")
        self.critic_layers = layers if critic_layers is None else critic_layers
        if self.critic_layers < 0:
            raise ValueError("critic layers must be nonnegative")
        self.action_sizes = tuple(int(size) for size in action_sizes)
        self.action_codec = action_codec
        self.info_codes = info_codes
        self.gru_policy = gru_policy
        self.mlp_policy = mlp_policy
        self.mlp_critic = mlp_critic
        self.actor_uses_history = gru_policy or mlp_policy
        self.critic_uses_history = gru_critic or mlp_critic
        if gru_policy:
            self.actor = WindowHead(observation_size, hidden_size, hidden_size, length,
                                    gru=True, output_size=sum(self.action_sizes))
            nn.init.orthogonal_(self.actor.output.weight, 0.01)
            nn.init.zeros_(self.actor.output.bias)
        elif mlp_policy:
            self.actor = ObservationWindowMLP(observation_size, sum(self.action_sizes), hidden_size, layers, length, .01)
        else:
            self.actor = mlp(observation_size, sum(self.action_sizes), hidden_size, layers, 0.01)
        self.gru_critic = gru_critic
        if gru_critic:
            self.critic = WindowHead(observation_size, self.critic_hidden_size, self.critic_hidden_size, length, gru=True)
        elif mlp_critic:
            self.critic = ObservationWindowMLP(observation_size, 1, self.critic_hidden_size,
                                               self.critic_layers, length, 1.)
        else:
            self.critic = mlp(observation_size, 1, self.critic_hidden_size, self.critic_layers, 1.)

    def get_value(self, observations):
        if self.gru_critic:
            padding = torch.isnan(observations[..., 0])
            return self.critic(torch.nan_to_num(observations), padding).float()
        if self.mlp_critic:
            return self.critic(observations).squeeze(-1).float()
        if observations.ndim == 3:
            observations = observations[:, -1]
        return self.critic(observations).squeeze(-1).float()

    def get_value_encoded(self, encoded, padding):
        return self.critic.forward_encoded(encoded, padding).float()

    def get_action_and_log_prob(self, observations, actions=None, sample=True):
        if self.gru_policy:
            padding = torch.isnan(observations[..., 0])
            logits = self.actor(torch.nan_to_num(observations), padding).float()
            mask_observations = observations[:, -1]
        elif self.mlp_policy:
            logits = self.actor(observations).float()
            mask_observations = observations[:, -1]
        else:
            logits = self.actor(observations).float()
            mask_observations = observations
        if self.info_codes:
            mask_observations = mask_observations[..., :-self.info_codes]
        mask = self.action_codec.mask(mask_observations)
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
            if sample:
                noise = torch.empty_like(logits).exponential_().log()
                actions = (logits - noise).argmax(-1)
            else:
                actions = logits.argmax(-1)

        log_prob = log_probabilities.gather(-1, actions.long().unsqueeze(-1))
        entropy = -(log_probabilities.exp() * log_probabilities).sum((-1, -2))

        return actions, log_prob.squeeze(-1).sum(-1), entropy

    def get_action(self, observations, sample=True):
        return self.get_action_and_log_prob(observations, sample=sample)[0]

    def get_action_and_value(self, observations, actions=None):
        actor_observations = observations[:, -1] if observations.ndim == 3 and not self.actor_uses_history else observations
        actions, log_prob, entropy = self.get_action_and_log_prob(actor_observations, actions)
        return actions, log_prob, entropy, self.get_value(observations)
