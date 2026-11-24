"""Helpers shared by the CARL checkpoint tools."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from carl.events import Event
from carl.gymnasium import (
    CARLBall,
    CARLCars,
    CARLResetState,
    CARLTorchVectorEnv,
)
from carl.gymnasium.state import CarlState

from training.checkpoints import load_payload
from training.models import Agent
from training.tensors import plain_tensor
from vis.playback import build_agent

TOUCH = int(Event.BALL_TOUCH)
JUMP = int(Event.JUMP)
DOUBLE_JUMP = int(Event.DOUBLE_JUMP)
DODGE = int(Event.DODGE)
CONTACT = int(Event.CONTACT_SURFACE)
RESET = int(Event.FLIP_RESET)

CAR_FIELDS = (
    "position", "velocity", "angular_velocity", "forward", "up", "boost",
    "on_ground", "demoed", "has_flipped", "has_double_jumped", "is_boosting"
)


def build(config: dict, n_sim: int, seed: int) -> CARLTorchVectorEnv:
    args = SimpleNamespace(**config)

    return CARLTorchVectorEnv(
        n_sim=n_sim,
        n_blue=args.team_size,
        n_orange=args.team_size,
        seed=seed,
        frameskip=args.frameskip,
        max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout,
        normalize=True,
        discrete_actions=True,
        previous_controls=args.previous_controls,
        masked_resets=True,
        defer_episode_reporting=True,
        capture_events=True
    )


def frame(env: CARLTorchVectorEnv) -> torch.Tensor:
    """Raw ball (pos, vel, angvel) and per car the 21 parsed-layout fields."""
    state = CarlState.from_raw(
        torch.from_dlpack(env._env.get_state()),
        env.n_cars,
        env._boost_pad_positions,
        env._team_sign
    )
    cars = torch.cat(
        [
            getattr(state, f"car_{name}").float().reshape(
                *state.car_position.shape[:2], -1
            )
            for name in CAR_FIELDS
        ],
        -1
    )

    return torch.cat(
        (
            state.ball_position,
            state.ball_velocity,
            state.ball_angular_velocity,
            cars.flatten(1)
        ),
        -1
    )


def reset_from(env: CARLTorchVectorEnv, start: torch.Tensor) -> torch.Tensor:
    """Reset every arena to the raw states in start (default hidden state)."""
    n = len(start)

    def provider(mask: torch.Tensor) -> CARLResetState:
        return CARLResetState(
            simulation_indices=None,
            reset_mask=mask,
            ball=CARLBall.from_tensor(start[:, :9]),
            cars=CARLCars.from_tensor(start[:, 9:].view(n, 2, 21), 2),
            normalized=False
        )

    env.reset_state_provider = provider
    observation = plain_tensor(env.reset())
    env.reset_state_provider = None

    return observation


def load_sampler(
    checkpoint: Path,
    n_sim:      int,
    seed:       int
) -> tuple[dict[str, Any], CARLTorchVectorEnv, Agent]:
    """Load a 1v1 feed-forward checkpoint for sampled self-play."""
    payload = load_payload(checkpoint)
    config = payload["config"]

    if (
        config["gru_policy"]
        or config.get("residual_policy")
        or config["team_size"] != 1
    ):
        raise ValueError("supports 1v1 feed-forward actors only")

    torch.manual_seed(seed)
    env = build(config, n_sim, seed)

    return payload, env, build_agent(env, config, payload)


def sample_actions(
    agent:       Agent,
    observation: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    logits = agent.actor(observation).float() / temperature

    return agent.sample_action_logits(logits, observation, sample=True)[0]


def actor_layers(payload: dict) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Float (weight, bias) of each actor linear layer, in order."""
    agent = payload["agent"]
    indices = sorted({
        int(key.split(".")[1]) for key in agent if key.startswith("actor.")
    })

    return [
        (
            agent[f"actor.{i}.weight"].float(),
            agent[f"actor.{i}.bias"].float()
        )
        for i in indices
    ]


def read_clip(
    ring:         torch.Tensor,
    ring_actions: torch.Tensor,
    arena:        int,
    slot:         int,
    filled:       torch.Tensor,
    length:       int
) -> tuple[np.ndarray, np.ndarray]:
    """Frames and actions of one arena, oldest first, from the ring buffers."""
    span = min(int(filled[arena]), length)
    order = [(slot - span + 1 + i) % length for i in range(span)]

    return (
        ring[order, arena].cpu().numpy(),
        ring_actions[order, arena].cpu().numpy()
    )


def write_summary(path: Path, summary: dict, skip: str) -> None:
    """Write summary.json and echo it without the long per-goal list."""
    path.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != skip}, indent=1))
