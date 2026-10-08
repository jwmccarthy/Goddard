#!/usr/bin/env python3
"""Watch Basic or GAIFO checkpoints play in the browser."""

import argparse
import json
import mimetypes
import threading
import time
import webbrowser

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import carl
import numpy as np
import torch as th
from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import CARLActionCodec
from gymnasium.spaces import Box
from jarl.envs import DatasetResetSampler

from basic import BASIC_POLICY_ARCHITECTURE, build_policy_and_critic, policy_checkpoint
from dodge_window import DodgeAwareCARLTorchVectorEnv
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
    "training_latest.pt",
    "actor_critic_final.pt",
    "policy_*.pt",
    "basic_*.pt",
)


def checkpoint_kind(path: Path) -> str:
    if path.match("gaifo_*.pt"):
        return "gaifo"
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
                f"no Basic or GAIFO checkpoints found in {self.directory}"
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


def load_policy_checkpoint(
    path: Path,
    env: CARLTorchVectorEnv,
    frameskip: int,
    hidden_size: int | None,
):
    payload = th.load(path, map_location="cpu", weights_only=True)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    if "frameskip" in config and int(config["frameskip"]) != frameskip:
        raise ValueError(
            f"checkpoint was trained at frameskip {config['frameskip']}, "
            f"watching at {frameskip}; pass --frameskip {config['frameskip']}"
        )

    kind = checkpoint_kind(path)
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
            checkpoint_policy_environment(env, policy_state, path, config),
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
        policy_env = checkpoint_policy_environment(env, policy_state, path, config)
        if architecture is None:
            policy, _ = build_policy_and_critic(
                policy_env, argparse.Namespace(hidden_size=hidden)
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

    policy.load_state_dict(policy_state)
    signature = (
        (kind, hidden, architecture, layers) if architecture is not None else
        (kind, hidden, architecture)
    )
    if kind == "gaifo" and team_size > 1:
        signature = (*signature, team_size)
    return policy.eval().requires_grad_(False), signature


def policy_environment(env: CARLTorchVectorEnv, dodge_window: bool):
    """Build legacy networks against CARL's original observation and codec."""
    if dodge_window:
        if not getattr(env, "dodge_window_features", False):
            raise ValueError("checkpoint requires dodge-window observations")
        return env
    if not getattr(env, "dodge_window_features", False):
        return env
    return SimpleNamespace(
        device=env.device,
        action_codec=CARLActionCodec().to(env.device),
        single_observation_space=Box(
            -np.inf, np.inf, (env.raw_observation_size,), np.float32,
        ),
        single_action_space=env.single_action_space,
    )


def checkpoint_policy_environment(
    env: CARLTorchVectorEnv, state: dict, path: Path, config: dict,
):
    """Match the network's saved input width to CARL or its jump-age extension."""
    foot = state.get("foot.model.0.weight")
    if not isinstance(foot, th.Tensor) or foot.ndim != 2:
        raise ValueError(f"checkpoint has no supported policy encoder: {path}")
    raw_size = getattr(env, "raw_observation_size", env.single_observation_space.shape[0])
    input_size = foot.shape[1]
    if input_size not in (raw_size, raw_size + 1):
        raise ValueError(
            f"checkpoint policy needs {input_size} observation features; "
            f"viewer supports {raw_size} or {raw_size + 1}: {path}"
        )
    dodge_window = input_size == raw_size + 1
    if ("expired_dodge_mask" in config
            and bool(config["expired_dodge_mask"]) != dodge_window):
        raise ValueError(f"checkpoint dodge-window setting does not match weights: {path}")
    return policy_environment(env, dodge_window)


def policy_observation(policy, observation: th.Tensor) -> th.Tensor:
    """New policies see the jump age; older policies retain their saved width."""
    return observation[..., :policy.foot.model[0].in_features]


def load_match(
    blue_path: Path,
    orange_path: Path,
    base: CARLTorchVectorEnv,
    frameskip: int,
    hidden_size: int | None,
):
    kind = checkpoint_kind(blue_path)
    if kind != checkpoint_kind(orange_path):
        raise ValueError("selected policies use different trainer architectures")
    blue, blue_signature = load_policy_checkpoint(blue_path, base, frameskip, hidden_size)
    orange, orange_signature = load_policy_checkpoint(orange_path, base, frameskip, hidden_size)
    if blue_signature != orange_signature:
        raise ValueError("selected policies use different trainer architectures")
    return base, blue, orange


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


def reset_observation(env: CARLTorchVectorEnv, kickoff: bool):
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
        base = DodgeAwareCARLTorchVectorEnv(
            n_sim=1,
            n_blue=args.team_size,
            n_orange=args.team_size,
            seed=args.seed,
            frameskip=args.frameskip,
            max_ticks=args.max_ticks,
            normalize=True,
            synchronize=True,
            reset_state_provider=reset_provider,
            discrete_actions=True,
        )
        env, blue, orange = load_match(
            blue_path, orange_path, base, args.frameskip, args.hidden_size,
        )
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
                    policy_observation(blue, observation[:args.team_size]), blue_state,
                    deterministic=not args.sample,
                )
                orange_output = orange.act(
                    policy_observation(orange, observation[args.team_size:]), orange_state,
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

            state.publish(render_frame(
                raw_state(base),
                registry.directory,
                blue_path,
                orange_path,
                blue_score,
                orange_score,
                round_number,
                tick,
                args.team_size,
            ))
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
    parser.add_argument("--max-ticks", type=int, default=4096)
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
    if args.frameskip < 1 or args.max_ticks < 1 or args.reset_state_limit < 1:
        parser.error("frame, episode, and replay limits must be positive")
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
