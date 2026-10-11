#!/usr/bin/env python3
"""Watch BASIC, GAIFO, or deep contrastive checkpoints play in the browser."""

import argparse
import json
import math
import mimetypes
import threading
import time
import webbrowser

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import carl
import numpy as np
import torch as th
import torch.nn as nn
from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import ACTION_NVECS
from jarl.data.records import PolicyOutput
from jarl.envs import DatasetResetSampler
from jarl.modules import GoalActor

from action_codec import enable_grounded_aerial_controls
from action_delay import QueuedActionEnv
from basic import BASIC_POLICY_ARCHITECTURE, build_policy_and_critic, policy_checkpoint
from deep import ARCHITECTURE as DEEP_ARCHITECTURE, GOAL_SLICES, goal_size
from gaifo import (
    GAIFO_ARCHITECTURE,
    GAIFO_GRU_ARCHITECTURE,
    GAIFO_TEAM_ARCHITECTURE,
    GAIFO_TEAM_GRU_ARCHITECTURE,
    SKILL_CATEGORIES,
    CuratedReplayResetTransform,
    ExpertSceneDataset,
    build_policy as build_gaifo_policy,
)
from replay_resets import (
    ReplayResetProvider, reset_index_dataset,
)
from replay_layout import TEAM_SIZES


ROOT = Path(__file__).resolve().parent
CAR_OFFSET = (13.8757, 0.0, 20.755)
CHECKPOINT_PATTERNS = (
    "gaifo_*.pt",
    "deep_*.pt",
    "training_latest.pt",
    "actor_critic_final.pt",
    "policy_*.pt",
    "basic_*.pt",
)


def checkpoint_kind(path: Path) -> str:
    if path.match("gaifo_*.pt"):
        return "gaifo"
    if path.match("deep_*.pt"):
        return "deep"
    return "basic"


def is_checkpoint(path: Path) -> bool:
    return any(path.match(pattern) for pattern in CHECKPOINT_PATTERNS)


@dataclass(frozen=True)
class CheckpointMetadata:
    path: Path
    relative_path: str
    step: int
    modified: int
    kind: str

    def as_dict(self) -> dict:
        return {
            "path": self.relative_path,
            "label": self.relative_path,
            "step": self.step,
            "modified": self.modified,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class EpisodeLimits:
    max_ticks: int
    no_touch_timeout_seconds: float | None


def checkpoint_episode_limits(path: Path, payload: dict) -> EpisodeLimits:
    """Recover training limits, with defaults for older checkpoint formats."""
    kind = checkpoint_kind(path)
    if kind == "gaifo":
        config, default_ticks = payload.get("config", {}), 1_000_000
    elif kind == "deep":
        config, default_ticks = payload.get("arguments", {}), 36_000
    else:
        config = payload.get("config", {})
        default_ticks = 36_000 if config else 4096
    if not isinstance(config, dict):
        raise ValueError(f"invalid checkpoint episode settings in {path}")
    max_ticks = config.get("max_ticks", default_ticks)
    seconds = config.get("no_touch_timeout", 30.0)
    if type(max_ticks) is not int or max_ticks < 1:
        raise ValueError(f"invalid checkpoint max ticks in {path}")
    if seconds is not None and (
        not isinstance(seconds, (int, float))
        or not math.isfinite(seconds) or seconds <= 0
    ):
        raise ValueError(f"invalid checkpoint no-touch timeout in {path}")
    return EpisodeLimits(max_ticks, seconds)


class CheckpointRegistry:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()

    def list(self) -> list[CheckpointMetadata]:
        checkpoints = []
        for pattern in CHECKPOINT_PATTERNS:
            for path in self.directory.rglob(pattern):
                try:
                    resolved = path.resolve(strict=True)
                    try:
                        step = int(resolved.stem.rsplit("_", 1)[-1])
                    except ValueError:
                        step = 0
                    checkpoints.append(CheckpointMetadata(
                        resolved,
                        resolved.relative_to(self.directory).as_posix(),
                        step,
                        resolved.stat().st_mtime_ns,
                        checkpoint_kind(resolved),
                    ))
                except (OSError, ValueError):
                    continue
        return sorted(
            checkpoints,
            key=lambda item: (item.modified, item.step, item.relative_path),
            reverse=True,
        )

    def newest_pair(self) -> tuple[Path, Path]:
        checkpoints = self.list()
        if not checkpoints:
            raise FileNotFoundError(
                f"no BASIC, GAIFO, or deep checkpoints found in {self.directory}"
            )
        newest = checkpoints[0]
        orange = next(
            (
                candidate
                for candidate in checkpoints[1:]
                if candidate.path.parent == newest.path.parent
                and candidate.kind == newest.kind
            ),
            newest,
        )
        return newest.path, orange.path

    def resolve(self, value: str) -> Path:
        path = (self.directory / value).resolve()
        if (
            self.directory not in path.parents
            or not path.is_file()
            or not is_checkpoint(path)
        ):
            raise ValueError("invalid checkpoint path")
        return path


class SpectatorState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.reset = threading.Event()
        self.kickoff = threading.Event()
        self.sequence = 0
        self.frame = None
        self.pending_match: tuple[Path, Path] | None = None
        self.reset_types: tuple[str, ...] = ()
        self.reset_type = "mixed"

    def publish(self, frame: dict) -> None:
        with self.condition:
            self.sequence += 1
            self.frame = frame
            self.condition.notify_all()

    def select_match(self, blue: Path, orange: Path) -> None:
        with self.condition:
            self.pending_match = (blue, orange)

    def take_match(self) -> tuple[Path, Path] | None:
        with self.condition:
            match = self.pending_match
            self.pending_match = None
            return match


    def configure_reset_types(self, types: tuple[str, ...]) -> None:
        with self.condition:
            self.reset_types = ("mixed", *types)
            if self.reset_type not in self.reset_types:
                self.reset_type = "mixed"

    def reset_options(self) -> dict:
        with self.condition:
            return {
                "types": [
                    {"id": name, "label": (
                        "Training mix" if name == "mixed"
                        else name.replace("_", " ").title()
                    )}
                    for name in self.reset_types
                ],
                "selected": self.reset_type,
            }

    def request_reset(self, reset_type: str | None = None) -> None:
        with self.condition:
            if not self.reset_types:
                raise ValueError("replay reset states are still loading")
            if reset_type is not None:
                if reset_type not in self.reset_types:
                    raise ValueError(f"unavailable reset type: {reset_type}")
                self.reset_type = reset_type
            self.reset.set()

    def request_kickoff(self) -> None:
        with self.condition:
            self.kickoff.set()

    def take_reset_request(self) -> tuple[bool, str] | None:
        with self.condition:
            if not (self.reset.is_set() or self.kickoff.is_set()):
                return None
            kickoff = self.kickoff.is_set()
            self.reset.clear()
            self.kickoff.clear()
            return kickoff, self.reset_type


class CuratedViewerResetProvider:
    """One-match replay resets with GAIFO's weighted or selected skill pools."""

    def __init__(
        self,
        frames: th.Tensor,
        internal_states: th.Tensor,
        pools: dict[str, th.Tensor],
        weights: th.Tensor,
        seed: int,
    ) -> None:
        self.providers = {
            name: ReplayResetProvider(
                DatasetResetSampler(
                    reset_index_dataset(indices), probability=1.0, seed=seed,
                ),
                frames, internal_states,
            )
            for name, indices in pools.items()
        }
        self.weights = weights.cpu()
        self.generator = th.Generator(device="cpu").manual_seed(seed)
        self.reset_type = "mixed"

    def select(self, reset_type: str) -> None:
        if reset_type != "mixed" and reset_type not in self.providers:
            raise ValueError(f"unavailable reset type: {reset_type}")
        self.reset_type = reset_type

    def __call__(self, reset_mask: th.Tensor):
        if not bool(reset_mask.any()):
            return None
        category = self.reset_type
        if category == "mixed":
            index = th.multinomial(
                self.weights, 1, replacement=True, generator=self.generator,
            ).item()
            category = SKILL_CATEGORIES[index]
        return self.providers[category](reset_mask)


def load_curated_reset_provider(
    replay_dir: Path,
    device: str | th.device,
    frame_skip: int,
    state_limit: int,
    corpus_limit: int | None,
    seed: int,
    team_size: int = 1,
) -> CuratedViewerResetProvider:
    """Classify complete replay periods, then retain a bounded GPU reset cache."""
    expert = ExpertSceneDataset(
        replay_dir, trajectory_length=8, limit=corpus_limit, seed=seed,
        frame_skip=frame_skip, device="cpu", reject_discontinuities=True,
        skill_sampling=True, team_size=team_size,
    )
    transform = CuratedReplayResetTransform(expert)
    weights = transform.weights.cpu().numpy()
    random = np.random.default_rng(seed)
    available = np.flatnonzero(weights)
    reserved = np.zeros(len(SKILL_CATEGORIES), dtype=np.int64)
    if state_limit >= len(available):
        reserved[available] = 1
    counts = random.multinomial(state_limit - int(reserved.sum()), weights) + reserved

    frames = []
    internals = []
    pools = {}
    offset = 0
    for category, pool, count in zip(SKILL_CATEGORIES, transform.pools, counts):
        if not count:
            continue
        choice = random.choice(len(pool), size=min(int(count), len(pool)), replace=False)
        indices = pool[th.from_numpy(choice)]
        frames.append(expert.frames[indices])
        internals.append(expert.internal_states[indices])
        pools[category] = th.arange(offset, offset + len(indices), device=device)
        offset += len(indices)

    if not frames:
        raise ValueError("no safe curated replay reset frames")
    active_weights = th.tensor([
        float(weights[index]) if category in pools else 0.0
        for index, category in enumerate(SKILL_CATEGORIES)
    ])
    active_weights /= active_weights.sum()
    provider = CuratedViewerResetProvider(
        th.cat(frames).to(device), th.cat(internals).to(device), pools, active_weights, seed,
    )
    return provider


def deep_watch_goal(observation: th.Tensor, kind: str) -> th.Tensor:
    """Chase the ball and aim its trajectory at the opponent goal in ego space."""
    if kind not in GOAL_SLICES:
        raise ValueError(f"unknown deep goal kind: {kind}")
    if observation.shape[-1] < 12:
        raise ValueError("deep goals require ball and ego-car positions")
    ball = observation[..., :3]
    net = th.zeros_like(ball)
    # CARL normalizes position by (4108, 6000, 2076) and rotates orange
    # into the acting car's frame. Both teams therefore attack positive Y.
    net[..., 1] = 5120.0 / 6000.0
    net[..., 2] = 321.3875 / 2076.0
    if kind == "car":
        return ball
    if kind == "ball":
        return net
    return th.cat((net, ball), dim=-1)


class WatchedDeepPolicy(nn.Module):
    """Adapt a stateless, goal-conditioned actor to the spectator policy API."""

    def __init__(self, actor: GoalActor, kind: str) -> None:
        super().__init__()
        self.actor = actor
        self.goal_kind = kind
        self.observation_size = actor.observation_size

    def initial_state(self, batch_size: int) -> None:
        return None

    @th.no_grad()
    def act(
        self, observation: th.Tensor, state=None, *, deterministic: bool = False,
    ) -> PolicyOutput:
        goal = deep_watch_goal(observation, self.goal_kind)
        return PolicyOutput(self.actor.act(observation, goal, deterministic=deterministic))


def load_deep_policy(path: Path, payload: dict, env: CARLTorchVectorEnv):
    """Restore either a standalone deep training checkpoint or its final copy."""
    if payload.get("architecture") != DEEP_ARCHITECTURE:
        raise ValueError(f"unsupported deep checkpoint architecture in {path}")
    config = payload.get("config", {})
    state = payload.get("actor")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError(f"deep checkpoint has no actor or configuration: {path}")
    kind = config.get("goal_kind")
    if kind not in GOAL_SLICES:
        raise ValueError(f"unsupported deep goal kind in {path}")
    if getattr(env, "n_cars", 2) != 2:
        raise ValueError(f"deep checkpoints require the 1v1 viewer: {path}")
    if tuple(config.get("action_nvec", ())) != tuple(env.single_action_space.nvec):
        raise ValueError(f"deep checkpoint action space differs from the viewer: {path}")
    observation_size = int(config["observation_size"])
    raw_size = env.single_observation_space.shape[0]
    if observation_size != raw_size:
        raise ValueError(
            f"checkpoint policy needs {observation_size} native observation "
            f"features; viewer provides {raw_size}: {path}"
        )
    if state.get("network.stem.0.weight") is None or (
        state["network.stem.0.weight"].shape[1] != observation_size + goal_size(kind)
    ):
        raise ValueError(f"deep checkpoint goal and observation sizes do not match weights: {path}")
    actor = GoalActor(
        observation_size, goal_size(kind), int(config["actor_width"]),
        int(config["actor_depth"]), ACTION_NVECS, env.action_codec,
    ).to(env.device)
    actor.load_state_dict(state)
    return WatchedDeepPolicy(actor, kind).eval().requires_grad_(False), (
        "deep", observation_size, kind, int(config["actor_width"]),
        int(config["actor_depth"]),
    )


def load_policy_checkpoint(
    path: Path,
    env: CARLTorchVectorEnv,
    frameskip: int,
    hidden_size: int | None,
):
    payload = th.load(path, map_location="cpu", weights_only=True)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    kind = checkpoint_kind(path)
    saved_frameskip = config.get("frameskip")
    if kind == "deep" and saved_frameskip is None:
        saved_frameskip = payload.get("arguments", {}).get("frameskip")
    elif kind == "basic" and saved_frameskip is None and config:
        # Structured BASIC checkpoints predating cadence metadata trained at 8.
        # Bare policy weights have no reliable way to identify their cadence.
        saved_frameskip = 8
    if saved_frameskip is not None and int(saved_frameskip) != frameskip:
        raise ValueError(
            f"checkpoint was trained at frameskip {saved_frameskip}, "
            f"watching at {frameskip}; pass --frameskip {saved_frameskip}"
        )

    if kind == "deep":
        policy, signature = load_deep_policy(path, payload, env)
        policy.episode_limits = checkpoint_episode_limits(path, payload)
        return policy, signature
    if kind == "gaifo":
        architecture = config.get("architecture")
        if architecture not in (
            GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE,
            GAIFO_TEAM_ARCHITECTURE, GAIFO_TEAM_GRU_ARCHITECTURE,
        ):
            raise ValueError(f"unsupported GAIFO architecture in {path}")
        team_size = config.get("team_size", 1)
        if 2 * team_size != getattr(env, "n_cars", 2):
            raise ValueError(f"checkpoint team size does not match the viewer: {path}")
        gru = architecture in (GAIFO_GRU_ARCHITECTURE, GAIFO_TEAM_GRU_ARCHITECTURE)
        if config.get("gru", False) != gru:
            raise ValueError(f"checkpoint GRU setting does not match architecture in {path}")
        hidden = int(config["policy_hidden"])
        layers = int(config.get("policy_layers", 1))
        policy_state = payload["policy"]
        policy = build_gaifo_policy(
            checkpoint_policy_environment(env, policy_state, path),
            argparse.Namespace(policy_hidden=hidden, policy_layers=layers, gru=gru),
        )
    else:
        checkpoint = policy_checkpoint(payload, path)
        policy_state = checkpoint.state
        hidden = checkpoint.hidden_size
        layers = checkpoint.policy_layers
        architecture = (
            None if checkpoint.architecture == BASIC_POLICY_ARCHITECTURE
            else checkpoint.architecture
        )
        delay_steps = basic_action_delay_steps(payload, checkpoint.observation_size, env)
        policy_env = checkpoint_policy_environment(
            QueuedActionEnv(env, delay_steps) if delay_steps else env, policy_state, path,
        )
        if architecture is None:
            policy, _ = build_policy_and_critic(
                policy_env, argparse.Namespace(
                    hidden_size=hidden, policy_layers=layers,
                    policy_gru_layers=checkpoint.policy_gru_layers,
                ),
            )
        else:
            policy = build_gaifo_policy(
                policy_env,
                argparse.Namespace(
                    policy_hidden=hidden,
                    policy_layers=layers,
                    gru=architecture == GAIFO_GRU_ARCHITECTURE,
                ),
            )
        policy.action_delay_steps = delay_steps

    policy.load_state_dict(policy_state)
    policy.episode_limits = checkpoint_episode_limits(path, payload)
    signature = (kind, hidden, architecture, layers)
    if architecture is None:
        signature = (*signature, checkpoint.policy_gru_layers)
    if kind == "gaifo" and team_size > 1:
        signature = (*signature, team_size)
    return policy.eval().requires_grad_(False), signature


def basic_action_delay_steps(payload: dict, input_size: int, env) -> int:
    """Use saved timing, or infer queue width for bare BASIC policy snapshots."""
    config = payload.get("config", {})
    native_size = env.single_observation_space.shape[0]
    width = len(ACTION_NVECS)
    if "action_delay_steps" in config:
        steps = config["action_delay_steps"]
        if not isinstance(steps, int) or steps < 0 or input_size != native_size + steps * width:
            raise ValueError("checkpoint action delay does not match its policy input width")
        return steps
    extra = input_size - native_size
    return extra // width if not config and extra > 0 and extra % width == 0 else 0


def checkpoint_policy_environment(
    env: CARLTorchVectorEnv, state: dict, path: Path,
):
    """Require the network's saved input width to match the viewer observation."""
    foot = state.get("foot.model.0.weight")
    if not isinstance(foot, th.Tensor) or foot.ndim != 2:
        raise ValueError(f"checkpoint has no supported policy encoder: {path}")
    raw_size = env.single_observation_space.shape[0]
    input_size = foot.shape[1]
    if input_size != raw_size:
        raise ValueError(
            f"checkpoint policy needs {input_size} observation features; "
            f"viewer provides {raw_size}: {path}"
        )
    return env


def load_match(
    blue_path: Path,
    orange_path: Path,
    base: CARLTorchVectorEnv,
    frameskip: int,
    hidden_size: int | None,
):
    blue_kind = checkpoint_kind(blue_path)
    orange_kind = checkpoint_kind(orange_path)
    if blue_kind != orange_kind and "deep" not in (blue_kind, orange_kind):
        raise ValueError("selected policies use different trainer architectures")
    blue, blue_signature = load_policy_checkpoint(blue_path, base, frameskip, hidden_size)
    orange, orange_signature = load_policy_checkpoint(orange_path, base, frameskip, hidden_size)
    if blue_signature != orange_signature and "deep" not in (blue_kind, orange_kind):
        raise ValueError("selected policies use different trainer architectures")
    blue_delay = getattr(blue, "action_delay_steps", 0)
    orange_delay = getattr(orange, "action_delay_steps", 0)
    if blue_delay != orange_delay:
        raise ValueError("selected policies use different reaction times")
    return QueuedActionEnv(base, blue_delay) if blue_delay else base, blue, orange


def configure_match_timing(base: CARLTorchVectorEnv, blue: nn.Module, args: argparse.Namespace) -> None:
    """Use the blue checkpoint's episode limits, unless the viewer overrides them."""
    limits = blue.episode_limits
    seconds = (limits.no_touch_timeout_seconds if args.no_touch_timeout is None
               else args.no_touch_timeout)
    base._env.max_ticks = limits.max_ticks if args.max_ticks is None else args.max_ticks
    base._env.no_touch_timeout_ticks = (
        math.ceil(seconds * carl.PHYS_TICKS_PER_SECOND) if seconds else 0
    )


def raw_state(environment: CARLTorchVectorEnv) -> th.Tensor:
    th.cuda.synchronize(environment.device)
    return th.from_dlpack(environment._env.get_state()).clone()


def vector(values: th.Tensor) -> list[float]:
    return [float(value) for value in values]


def render_frame(
    raw: th.Tensor,
    root: Path,
    blue_path: Path,
    orange_path: Path,
    blue_score: int,
    orange_score: int,
    round_number: int,
    tick: int,
    team_size: int = 1,
) -> dict:
    raw = raw[0].cpu()
    cars = raw[9:9 + 44 * team_size].view(2 * team_size, 22)
    rendered = []
    for index, car in enumerate(cars):
        forward = car[9:12]
        up = car[12:15]
        right = th.linalg.cross(up, forward, dim=-1)
        center = (
            car[:3]
            + forward * CAR_OFFSET[0]
            + right * CAR_OFFSET[1]
            + up * CAR_OFFSET[2]
        )
        rendered.append({
            "team": index // team_size,
            "player": index % team_size + 1,
            "pos": vector(center),
            "fwd": vector(forward),
            "rgt": vector(right),
            "up": vector(up),
            "boost": float(car[15]),
            "boosting": bool(car[20]),
            "demoed": bool(car[17]),
        })
    return {
        "tick": tick,
        "round": round_number,
        "blue": {
            "checkpoint": blue_path.stem,
            "path": blue_path.relative_to(root).as_posix(),
            "score": blue_score,
        },
        "orange": {
            "checkpoint": orange_path.stem,
            "path": orange_path.relative_to(root).as_posix(),
            "score": orange_score,
        },
        "cars": rendered,
        "ball": {"pos": vector(raw[:3])},
    }


def reset_observation(env: CARLTorchVectorEnv | QueuedActionEnv, kickoff: bool):
    """Reset via demonstration states or a plain random kickoff."""
    if not kickoff:
        return env.reset()
    provider = env.reset_state_provider
    env.reset_state_provider = None
    try:
        return env.reset()
    finally:
        env.reset_state_provider = provider


def simulate(
    state: SpectatorState,
    registry: CheckpointRegistry,
    blue_path: Path,
    orange_path: Path,
    args: argparse.Namespace,
) -> None:
    base = None
    try:
        reset_provider = load_curated_reset_provider(
            args.replay_dir, "cuda:0", args.frameskip, args.reset_state_limit,
            args.reset_corpus_limit or None, args.seed,
            team_size=args.team_size,
        )
        state.configure_reset_types(tuple(reset_provider.providers))
        base = enable_grounded_aerial_controls(CARLTorchVectorEnv(
            n_sim=1,
            n_blue=args.team_size,
            n_orange=args.team_size,
            seed=args.seed,
            frameskip=args.frameskip,
            max_ticks=args.max_ticks or 4096,
            normalize=True,
            synchronize=True,
            reset_state_provider=reset_provider,
            discrete_actions=True,
        ))
        env, blue, orange = load_match(
            blue_path, orange_path, base, args.frameskip, args.hidden_size,
        )
        configure_match_timing(base, blue, args)
        observation = env.reset()
        blue_state = blue.initial_state(args.team_size)
        orange_state = orange.initial_state(args.team_size)
        blue_score = orange_score = 0
        round_number = 1
        tick = 0
        next_step = time.perf_counter()

        while not state.stop.is_set():
            pending = state.take_match()
            if pending is not None:
                try:
                    next_env, next_blue, next_orange = load_match(
                        pending[0], pending[1], base, args.frameskip,
                        args.hidden_size,
                    )
                    configure_match_timing(base, next_blue, args)
                except Exception as error:
                    state.publish({"error": f"{type(error).__name__}: {error}"})
                else:
                    blue_path, orange_path = pending
                    env = next_env
                    blue, orange = next_blue, next_orange
                    blue_state = blue.initial_state(args.team_size)
                    orange_state = orange.initial_state(args.team_size)
                    state.request_reset()

            request = state.take_reset_request()
            if request is not None:
                kickoff, reset_type = request
                reset_provider.select(reset_type)
                observation = reset_observation(env, kickoff)
                blue_state = blue.initial_state(args.team_size)
                orange_state = orange.initial_state(args.team_size)
                blue_score = orange_score = 0
                round_number = 1
                tick = 0

            with th.inference_mode():
                blue_output = blue.act(
                    observation[:args.team_size], blue_state,
                    deterministic=not args.sample,
                )
                orange_output = orange.act(
                    observation[args.team_size:], orange_state,
                    deterministic=not args.sample,
                )
                blue_state = blue_output.next_state
                orange_state = orange_output.next_state
                action = th.cat((blue_output.action, orange_output.action))
            observation, reward, terminated, truncated, _ = env.step(action)
            tick += args.frameskip

            goal = int(reward[0].item())
            blue_score += max(goal, 0)
            orange_score += max(-goal, 0)
            if (terminated | truncated).any():
                blue_state = blue.initial_state(args.team_size)
                orange_state = orange.initial_state(args.team_size)
                round_number += 1
                tick = 0

            frame = render_frame(
                raw_state(base),
                registry.directory,
                blue_path,
                orange_path,
                blue_score,
                orange_score,
                round_number,
                tick,
                args.team_size,
            )
            for team, policy in (("blue", blue), ("orange", orange)):
                if isinstance(policy, WatchedDeepPolicy):
                    frame[team]["objective"] = (
                        "chase ball" if policy.goal_kind == "car" else
                        "shoot at opponent goal" if policy.goal_kind == "ball" else
                        "chase ball & shoot at goal"
                    )
            state.publish(frame)
            next_step += args.frameskip / 120.0
            delay = next_step - time.perf_counter()
            if delay > 0:
                state.stop.wait(delay)
            else:
                next_step = time.perf_counter()
    except Exception as error:
        state.publish({"error": f"{type(error).__name__}: {error}"})
    finally:
        if base is not None:
            base.close()


def make_handler(
    state: SpectatorState,
    frontend: Path,
    arena: Path,
    registry: CheckpointRegistry,
):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path == "/api/reset":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 0 or length > 1_024:
                        raise ValueError("reset request is too large")
                    payload = json.loads(self.rfile.read(length)) if length else {}
                    if not isinstance(payload, dict):
                        raise ValueError("reset request must be an object")
                    state.request_reset(payload.get("reset_type"))
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    self.send_error(HTTPStatus.BAD_REQUEST, str(error))
                    return
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            if self.path == "/api/kickoff":
                state.request_kickoff()
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            if self.path != "/api/match":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                state.select_match(
                    registry.resolve(payload["blue"]),
                    registry.resolve(payload["orange"]),
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self.send_error(HTTPStatus.BAD_REQUEST, str(error))
                return
            self.send_response(HTTPStatus.ACCEPTED)
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/api/checkpoints":
                self._json([item.as_dict() for item in registry.list()])
                return
            if self.path == "/api/reset-types":
                self._json(state.reset_options())
                return
            if self.path == "/api/stream":
                self._stream()
                return
            path = {
                "/": frontend / "index.html",
                "/app.js": frontend / "app.js",
                "/arena.obj": arena,
            }.get(self.path)
            if path is None or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream"
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, value) -> None:
            payload = json.dumps(value).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _stream(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            sequence = 0
            try:
                while True:
                    with state.condition:
                        state.condition.wait_for(
                            lambda: state.sequence > sequence, timeout=10
                        )
                        if state.sequence == sequence:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue
                        sequence = state.sequence
                        payload = json.dumps(state.frame, separators=(",", ":"))
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--team-size", type=int, choices=TEAM_SIZES, default=1)
    parser.add_argument(
        "--replay-dir", type=Path,
        help="parsed POV directory (default: pro_<team-size>v<team-size>_fs4)",
    )
    parser.add_argument("--blue")
    parser.add_argument("--orange")
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument(
        "--policy-hidden", "--hidden-size", dest="hidden_size",
        type=int, metavar="POLICY_HIDDEN",
    )
    parser.add_argument(
        "--max-ticks", type=int,
        help="match duration in physics ticks (default: blue checkpoint's training limit)",
    )
    parser.add_argument(
        "--no-touch-timeout", type=float, metavar="SECONDS",
        help="seconds since any ball touch before reset (default: blue checkpoint; 0 disables)",
    )
    parser.add_argument("--reset-state-limit", type=int, default=4096)
    parser.add_argument(
        "--reset-corpus-limit", type=int, default=200_000,
        help="replay frames to classify for curated resets (0 scans the full corpus)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.replay_dir is None:
        args.replay_dir = (ROOT / "parsed_replays" /
                           f"pro_{args.team_size}v{args.team_size}_fs4")
    if (args.blue is None) != (args.orange is None):
        parser.error("--blue and --orange must be provided together")
    if (args.frameskip < 1 or (args.max_ticks is not None and args.max_ticks < 1)
            or args.reset_state_limit < 1):
        parser.error("frame, episode, and replay limits must be positive")
    if args.no_touch_timeout is not None and (
        not math.isfinite(args.no_touch_timeout) or args.no_touch_timeout < 0
    ):
        parser.error("no-touch timeout must be non-negative and finite")
    if args.reset_corpus_limit < 0 or 0 < args.reset_corpus_limit < 8:
        parser.error("reset corpus limit must be zero or at least eight frames")
    if args.hidden_size is not None and args.hidden_size < 1:
        parser.error("hidden size must be positive")
    return args


def main() -> None:
    args = parse_args()
    registry = CheckpointRegistry(args.checkpoint_dir)
    if args.blue is None:
        blue_path, orange_path = registry.newest_pair()
    else:
        blue_path = registry.resolve(args.blue)
        orange_path = registry.resolve(args.orange)

    frontend = ROOT / "web" / "self_play"
    arena = Path(carl.__file__).resolve().parent / "assets" / "arena.obj"
    if not (frontend / "index.html").is_file() or not (frontend / "app.js").is_file():
        raise FileNotFoundError(f"checkpoint viewer assets not found in {frontend}")
    if not arena.is_file():
        raise FileNotFoundError(f"CARL arena asset not found: {arena}")

    state = SpectatorState()
    server = ThreadingHTTPServer(
        (args.host, args.port), make_handler(state, frontend, arena, registry)
    )
    thread = threading.Thread(
        target=simulate,
        args=(state, registry, blue_path, orange_path, args),
        daemon=True,
    )
    thread.start()
    url = f"http://{args.host}:{args.port}"
    print(f"Blue:   {blue_path}")
    print(f"Orange: {orange_path}")
    print(f"Viewer: {url}")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop.set()
        thread.join(timeout=5)
        server.server_close()


if __name__ == "__main__":
    main()
