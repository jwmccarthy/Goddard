#!/usr/bin/env python3
"""Watch BASIC or GAIFO checkpoints play a 1v1 match in the browser."""

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

import carl
import torch as th

from carl.gymnasium import CARLTorchVectorEnv
from jarl.envs import DatasetResetSampler

from basic import BASIC_POLICY_ARCHITECTURE, build_policy_and_critic, policy_checkpoint
from gaifo import (
    GAIFO_ARCHITECTURE,
    GAIFO_GRU_ARCHITECTURE,
    build_policy as build_gaifo_policy,
)
from replay_resets import load_demonstration_reset_dataset


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
                f"no BASIC or GAIFO checkpoints found in {self.directory}"
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
        if architecture not in (GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE):
            raise ValueError(f"unsupported GAIFO architecture in {path}")
        gru = architecture == GAIFO_GRU_ARCHITECTURE
        if config.get("gru", False) != gru:
            raise ValueError(f"checkpoint GRU setting does not match architecture in {path}")
        hidden = int(config["policy_hidden"])
        policy = build_gaifo_policy(
            env, argparse.Namespace(policy_hidden=hidden, gru=gru)
        )
        policy_state = payload["policy"]
    else:
        checkpoint = policy_checkpoint(payload, path)
        policy_state = checkpoint.state
        hidden = checkpoint.hidden_size
        architecture = (
            None if checkpoint.architecture == BASIC_POLICY_ARCHITECTURE
            else checkpoint.architecture
        )
        if architecture is None:
            policy, _ = build_policy_and_critic(
                env, argparse.Namespace(hidden_size=hidden)
            )
        else:
            policy = build_gaifo_policy(
                env,
                argparse.Namespace(
                    policy_hidden=hidden,
                    gru=architecture == GAIFO_GRU_ARCHITECTURE,
                ),
            )

    policy.load_state_dict(policy_state)
    return policy.eval().requires_grad_(False), (
        kind, hidden, architecture
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
) -> dict:
    raw = raw[0].cpu()
    cars = raw[9:53].view(2, 22)
    rendered = []
    for team, car in enumerate(cars):
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
            "team": team,
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
    env = None
    try:
        reset_dataset = load_demonstration_reset_dataset(
            args.replay_dir,
            "cuda:0",
            args.frameskip,
            args.reset_state_limit,
            args.seed,
            require_frame_skip_match=False,
        )
        reset_sampler = DatasetResetSampler(
            reset_dataset, probability=1.0, seed=args.seed
        )
        env = CARLTorchVectorEnv(
            n_sim=1,
            n_blue=1,
            n_orange=1,
            seed=args.seed,
            frameskip=args.frameskip,
            max_ticks=args.max_ticks,
            normalize=True,
            synchronize=True,
            reset_state_provider=reset_sampler,
            discrete_actions=True,
        )
        blue, blue_signature = load_policy_checkpoint(
            blue_path, env, args.frameskip, args.hidden_size
        )
        orange, orange_signature = load_policy_checkpoint(
            orange_path, env, args.frameskip, args.hidden_size
        )
        if blue_signature != orange_signature:
            raise ValueError("selected policies use different trainer architectures")
        observation = env.reset()
        blue_state = blue.initial_state(1)
        orange_state = orange.initial_state(1)
        blue_score = orange_score = 0
        round_number = 1
        tick = 0
        next_step = time.perf_counter()

        while not state.stop.is_set():
            pending = state.take_match()
            if pending is not None:
                try:
                    next_blue, next_blue_signature = load_policy_checkpoint(
                        pending[0], env, args.frameskip, args.hidden_size
                    )
                    next_orange, next_orange_signature = load_policy_checkpoint(
                        pending[1], env, args.frameskip, args.hidden_size
                    )
                    if next_blue_signature != next_orange_signature:
                        raise ValueError(
                            "selected policies use different trainer architectures"
                        )
                except Exception as error:
                    state.publish({"error": f"{type(error).__name__}: {error}"})
                else:
                    blue_path, orange_path = pending
                    blue, orange = next_blue, next_orange
                    blue_state = blue.initial_state(1)
                    orange_state = orange.initial_state(1)
                    state.reset.set()

            if state.reset.is_set() or state.kickoff.is_set():
                kickoff = state.kickoff.is_set()
                state.reset.clear()
                state.kickoff.clear()
                observation = reset_observation(env, kickoff)
                blue_state = blue.initial_state(1)
                orange_state = orange.initial_state(1)
                blue_score = orange_score = 0
                round_number = 1
                tick = 0

            with th.inference_mode():
                blue_output = blue.act(
                    observation[:1], blue_state, deterministic=not args.sample
                )
                orange_output = orange.act(
                    observation[1:], orange_state, deterministic=not args.sample
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
                blue_state = blue.initial_state(1)
                orange_state = orange.initial_state(1)
                round_number += 1
                tick = 0

            state.publish(render_frame(
                raw_state(env),
                registry.directory,
                blue_path,
                orange_path,
                blue_score,
                orange_score,
                round_number,
                tick,
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
        if env is not None:
            env.close()


def make_handler(
    state: SpectatorState,
    frontend: Path,
    arena: Path,
    registry: CheckpointRegistry,
):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path == "/api/reset":
                state.reset.set()
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            if self.path == "/api/kickoff":
                state.kickoff.set()
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
    parser.add_argument(
        "--replay-dir", type=Path, default=ROOT / "parsed_replays/pro_1v1_fs4"
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if (args.blue is None) != (args.orange is None):
        parser.error("--blue and --orange must be provided together")
    if args.frameskip < 1 or args.max_ticks < 1 or args.reset_state_limit < 1:
        parser.error("frame, episode, and replay limits must be positive")
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
