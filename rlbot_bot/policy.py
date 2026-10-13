"""CPU inference for GAIFO checkpoints, without the CUDA CARL training runtime."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


ACTION_SIZES = (3, 3, 3, 2, 2, 3, 2)
ARCHITECTURES = {
    "scene-marl-gaifo-1v1-v3": (1, False),
    "scene-marl-gaifo-1v1-v3-gru": (1, True),
    "scene-marl-gaifo-teams-v1": (None, False),
    "scene-marl-gaifo-teams-v1-gru": (None, True),
}


def observation_size(team_size: int) -> int:
    n_cars = 2 * team_size
    return 9 + 21 * n_cars + 68 + 6 + 6 * (n_cars - 1) + 6 + 2


@dataclass(frozen=True)
class PolicyConfig:
    team_size: int
    frameskip: int
    hidden: int
    layers: int
    gru: bool

    @property
    def observation_size(self) -> int:
        return observation_size(self.team_size)


def hidden_layers(count: int, width: int) -> list[nn.Module]:
    modules = []
    for _ in range(count):
        modules.extend((nn.Linear(width, width), nn.ReLU()))
    return modules


class GAIFOPolicy(nn.Module):
    """Reproduce gaifo.build_policy's state-dict layout and forward pass."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.foot = nn.Module()
        self.foot.model = nn.Sequential(
            nn.Linear(config.observation_size, config.hidden), nn.ReLU(),
        )
        self.body = nn.Module()
        self.head = nn.Module()
        if config.gru:
            self.body.rnn = nn.GRU(config.hidden, config.hidden)
            self.head.model = nn.Sequential(
                *hidden_layers(config.layers - 1, config.hidden),
                nn.Linear(config.hidden, sum(ACTION_SIZES)),
            )
        else:
            self.body.model = nn.Sequential(*hidden_layers(config.layers, config.hidden))
            self.head.model = nn.Sequential(nn.Linear(config.hidden, sum(ACTION_SIZES)))

    def initial_state(self, batch_size: int = 1) -> torch.Tensor | None:
        return torch.zeros(1, batch_size, self.config.hidden) if self.config.gru else None

    def forward(
        self, observation: torch.Tensor, state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if observation.ndim != 2 or observation.shape[-1] != self.config.observation_size:
            raise ValueError(f"GAIFO policy expects {self.config.observation_size} observations")
        features = self.foot.model(observation)
        if self.config.gru:
            features, state = self.body.rnn(features.unsqueeze(0), state)
            features = features.squeeze(0)
        else:
            if state is not None:
                raise ValueError("MLP policy does not use recurrent state")
            features = self.body.model(features)
        return self.head.model(features), state

    @torch.no_grad()
    def act(
        self, observation: torch.Tensor, state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits, next_state = self(observation, state)
        # GAIFO uses GroundAerialActionCodec: grounded pitch/roll remain selectable.
        mask = torch.ones_like(logits, dtype=torch.bool)
        on_ground = observation[..., 25] > 0.5
        mask[..., 10] = on_ground                          # powerslide
        mask[..., 12] = observation[..., 24] > 0           # boost
        mask[..., 17] = on_ground | (                      # jump / dodge
            (observation[..., -2] > 0.5) & (observation[..., -1] > 1 / 120)
        )
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return torch.stack(
            [part.argmax(dim=-1) for part in logits.split(ACTION_SIZES, dim=-1)],
            dim=-1,
        ), next_state


def load_policy_checkpoint(path: Path) -> GAIFOPolicy:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"GAIFO checkpoint has no configuration: {path}")
    saved = payload["config"]
    architecture = saved.get("architecture")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unsupported GAIFO architecture {architecture!r}: {path}")
    expected_team, gru = ARCHITECTURES[architecture]
    config = PolicyConfig(
        team_size=int(saved.get("team_size", 1)),
        frameskip=int(saved["frameskip"]),
        hidden=int(saved["policy_hidden"]),
        layers=int(saved.get("policy_layers", 1)),
        gru=gru,
    )
    if (config.team_size not in (1, 2, 3)
            or (expected_team is None and config.team_size == 1)
            or (expected_team is not None and config.team_size != expected_team)
            or saved.get("gru", False) != gru
            or min(config.hidden, config.layers, config.frameskip) <= 0):
        raise ValueError(f"invalid or inconsistent GAIFO policy configuration: {path}")
    state = payload.get("policy")
    if not isinstance(state, dict) or "foot.model.0.weight" not in state:
        raise ValueError(f"GAIFO checkpoint has no policy weights: {path}")
    if state["foot.model.0.weight"].shape[1] != config.observation_size:
        raise ValueError(
            f"checkpoint policy needs {state['foot.model.0.weight'].shape[1]} features; "
            f"RLBot provides {config.observation_size}: {path}"
        )
    policy = GAIFOPolicy(config)
    policy.load_state_dict(state, strict=True)
    return policy.eval().requires_grad_(False)


def decode_action(action: list[int]) -> dict[str, float | bool]:
    if len(action) != len(ACTION_SIZES):
        raise ValueError("GAIFO action has the wrong number of controls")
    axis = lambda value: -1.0 if value == 1 else 1.0 if value == 2 else 0.0
    horizontal, vertical, throttle, powerslide, boost, air_roll, jump = action
    return {
        "throttle": axis(throttle),
        "steer": axis(horizontal),
        "yaw": axis(horizontal),
        "pitch": axis(vertical),
        "roll": axis(air_roll),
        "jump": jump == 1,
        "boost": boost == 1,
        "handbrake": powerslide == 1,
    }
