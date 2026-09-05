#!/usr/bin/env python3
"""Watch deterministic PULSE self-play from demonstration starting states."""

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

from self_play import (
    FrozenPulseController,
    PulseLatentEnv,
    build_policy,
    file_sha256,
    load_demonstration_reset_dataset,
)


ROOT = Path(__file__).parent
CAR_OFFSET = (13.8757, 0.0, 20.755)


@dataclass(frozen=True)
class CheckpointMetadata:
    path: Path
    relative_path: str
    step: int
    modified: int

    def as_dict(self) -> dict:
        return {
            "path": self.relative_path,
            "label": self.relative_path,
            "step": self.step,
            "modified": self.modified,
        }


class CheckpointRegistry:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()

    def list(self) -> list[CheckpointMetadata]:
        checkpoints = []
        for path in self.directory.rglob("self_play_*.pt"):
            try:
                resolved = path.resolve(strict=True)
                checkpoints.append(CheckpointMetadata(
                    resolved,
                    resolved.relative_to(self.directory).as_posix(),
                    int(resolved.stem.removeprefix("self_play_")),
                    resolved.stat().st_mtime_ns,
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
                f"no self-play checkpoints found in {self.directory}"
            )
        newest = checkpoints[0]
        orange = next(
            (
                candidate
                for candidate in checkpoints[1:]
                if candidate.path.parent == newest.path.parent
            ),
            newest,
        )
        return newest.path, orange.path

    def resolve(self, value: str) -> Path:
        path = (self.directory / value).resolve()
        if (
            self.directory not in path.parents
            or not path.is_file()
            or not path.match("self_play_*.pt")
        ):
            raise ValueError("invalid checkpoint path")
        return path


class SpectatorState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.reset = threading.Event()
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


def load_checkpoint(path: Path, env: PulseLatentEnv):
    payload = th.load(path, map_location="cpu", weights_only=True)
    config = payload["config"]
    policy = build_policy(
        env,
        float(config["exploration_std"]),
        config.get("gru_hidden_size"),
        config.get("gru_input_size"),
    )
    policy.load_state_dict(payload["policy"])
    metadata = {
        "distill_sha256": payload["distill_sha256"],
        "pulse_artifact": payload["pulse_artifact"],
        "pulse_sha256": payload["pulse_sha256"],
        "bf16": bool(config.get("bf16", False)),
    }
    return policy.eval().requires_grad_(False), metadata


def resolve_pulse_artifact(
    explicit: Path | None,
    blue_path: Path,
    blue_payload: dict,
    orange_payload: dict,
) -> Path:
    blue = str(blue_payload["distill_sha256"])
    orange = str(orange_payload["distill_sha256"])
    if blue != orange:
        raise ValueError("selected policies use different distillation artifacts")
    if explicit is not None:
        if file_sha256(explicit) != blue:
            raise ValueError("explicit distillation artifact does not match checkpoint")
        return explicit
    if blue_payload["pulse_sha256"] != orange_payload["pulse_sha256"]:
        raise ValueError("selected policies embed different frozen PULSE artifacts")
    artifact = blue_path.parent / str(blue_payload["pulse_artifact"])
    if file_sha256(artifact) != blue_payload["pulse_sha256"]:
        raise ValueError("embedded frozen PULSE artifact failed verification")
    return artifact


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


def simulate(
    state: SpectatorState,
    registry: CheckpointRegistry,
    blue_path: Path,
    orange_path: Path,
    args: argparse.Namespace,
) -> None:
    base = None
    try:
        reset_dataset = load_demonstration_reset_dataset(
            args.replay_dir,
            "cuda:0",
            args.frameskip,
            args.reset_state_limit,
            args.seed,
        )
        reset_sampler = DatasetResetSampler(
            reset_dataset, probability=1.0, seed=args.seed
        )
        base = CARLTorchVectorEnv(
            n_sim=1,
            n_blue=1,
            n_orange=1,
            seed=args.seed,
            frameskip=args.frameskip,
            max_ticks=args.max_ticks,
            normalize=True,
            synchronize=True,
            reset_state_provider=reset_sampler,
        )
        blue_payload = th.load(blue_path, map_location="cpu", weights_only=True)
        orange_payload = th.load(orange_path, map_location="cpu", weights_only=True)
        blue_bf16 = bool(blue_payload["config"].get("bf16", False))
        orange_bf16 = bool(orange_payload["config"].get("bf16", False))
        if blue_bf16 != orange_bf16:
            raise ValueError("selected policies use different decoder precision")
        artifact_path = resolve_pulse_artifact(
            args.distill_checkpoint,
            blue_path,
            blue_payload,
            orange_payload,
        )
        artifact_id = str(blue_payload["distill_sha256"])
        controller = FrozenPulseController.load(
            artifact_path,
            base.action_codec,
            base.device,
            frame_skip=args.frameskip,
            bf16=blue_bf16,
        )
        env = PulseLatentEnv(base, controller)
        del blue_payload, orange_payload
        blue, _ = load_checkpoint(blue_path, env)
        orange, _ = load_checkpoint(orange_path, env)
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
                    next_blue, next_blue_payload = load_checkpoint(pending[0], env)
                    next_orange, next_orange_payload = load_checkpoint(pending[1], env)
                    if next_blue_payload["bf16"] != next_orange_payload["bf16"]:
                        raise ValueError(
                            "selected policies use different decoder precision"
                        )
                    next_artifact = str(next_blue_payload["distill_sha256"])
                    if next_artifact != artifact_id:
                        raise ValueError(
                            "selected policies use a different distillation artifact"
                        )
                    resolve_pulse_artifact(
                        None, pending[0], next_blue_payload, next_orange_payload
                    )
                except Exception as error:
                    state.publish({"error": f"{type(error).__name__}: {error}"})
                else:
                    blue_path, orange_path = pending
                    blue, orange = next_blue, next_orange
                    controller.bf16 = next_blue_payload["bf16"]
                    state.reset.set()

            if state.reset.is_set():
                state.reset.clear()
                observation = env.reset()
                blue_state = blue.initial_state(1)
                orange_state = orange.initial_state(1)
                blue_score = orange_score = 0
                round_number = 1
                tick = 0

            with th.inference_mode():
                blue_output = blue.act(
                    observation[:1], blue_state, deterministic=True
                )
                orange_output = orange.act(
                    observation[1:], orange_state, deterministic=True
                )
                residual = th.cat((blue_output.action, orange_output.action))
                blue_state = blue_output.next_state
                orange_state = orange_output.next_state
            observation, reward, terminated, truncated, _ = env.step(residual)
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
                raw_state(base),
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
                state.reset.set()
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
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/self_play")
    )
    parser.add_argument("--distill-checkpoint", type=Path)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--blue")
    parser.add_argument("--orange")
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=4096)
    parser.add_argument("--reset-state-limit", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if (args.blue is None) != (args.orange is None):
        parser.error("--blue and --orange must be provided together")
    if args.frameskip < 1 or args.max_ticks < 1 or args.reset_state_limit < 1:
        parser.error("frame, episode, and replay limits must be positive")
    return args


def main() -> None:
    args = parse_args()
    registry = CheckpointRegistry(args.checkpoint_dir)
    if args.blue is None:
        blue_path, orange_path = registry.newest_pair()
    else:
        blue_path = registry.resolve(args.blue)
        orange_path = registry.resolve(args.orange)

    state = SpectatorState()
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

    arena = Path(carl.__file__).resolve().parent / "assets" / "arena.obj"
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(state, ROOT / "web" / "self_play", arena, registry),
    )
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
