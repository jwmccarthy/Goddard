#!/usr/bin/env python3
"""Watch current Goddard checkpoints play through a localhost 3D viewer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
import time
import webbrowser

import torch
import torch.nn as nn

import carl
from carl.gymnasium import CARLTorchVectorEnv
from jarl.envs import DatasetResetSampler
from jarl.modules import GRU, MLP
from jarl.modules.encoder import LinearEncoder
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.modules.utils import init_layer

from replay_states import load_replay_dataset


CAR_OFFSET = (13.8757, 0.0, 20.755)
ACTION_LOGITS = 18
CHECKPOINT_PATTERNS = ("policy_*.pt", "actor_critic_final.pt")


@dataclass(frozen=True)
class CheckpointMetadata:
    path:             Path
    relative_path:    str
    observation_size: int
    hidden_size:      int
    modified:         int

    def as_dict(self) -> dict:
        return {
            "path":             self.relative_path,
            "label":            self.relative_path,
            "observation_size": self.observation_size,
            "hidden_size":      self.hidden_size,
            "modified":         self.modified,
        }


class CheckpointRegistry:
    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self._cache: dict[Path, CheckpointMetadata] = {}
        self._lock = threading.Lock()

    def list(self) -> list[CheckpointMetadata]:
        paths = self._paths()

        with self._lock:
            live_paths = set(paths)
            for stale_path in self._cache.keys() - live_paths:
                del self._cache[stale_path]

            checkpoints = []
            for path in paths:
                try:
                    modified = path.stat().st_mtime_ns
                    cached = self._cache.get(path)
                    if cached is None or cached.modified != modified:
                        cached = inspect_checkpoint(self.directory, path)
                        self._cache[path] = cached
                except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                    continue
                checkpoints.append(cached)

        return checkpoints

    def newest_pair(self) -> tuple[Path, Path]:
        checkpoints = self.list()
        if len(checkpoints) < 2:
            raise FileNotFoundError(
                f"Need two current-format checkpoints under {self.directory}"
            )

        newest = checkpoints[0]
        for candidate in checkpoints[1:]:
            if candidate.observation_size == newest.observation_size:
                return newest.path, candidate.path
        raise ValueError("No second checkpoint is compatible with the newest")

    def resolve(self, value: str) -> Path:
        path = (self.directory / value).resolve()
        valid_name = any(path.match(pattern) for pattern in CHECKPOINT_PATTERNS)
        if (
            self.directory not in path.parents
            or not path.is_file()
            or not valid_name
        ):
            raise ValueError("Invalid checkpoint path")
        inspect_checkpoint(self.directory, path)
        return path

    def _paths(self) -> list[Path]:
        paths = []

        for pattern in CHECKPOINT_PATTERNS:
            for candidate in self.directory.rglob(pattern):
                try:
                    path = candidate.resolve(strict=True)
                    if self.directory not in path.parents or not path.is_file():
                        continue
                    paths.append((path.stat().st_mtime_ns, path))
                except OSError:
                    continue

        paths.sort(key=lambda item: (item[0], item[1].as_posix()), reverse=True)
        return list(dict.fromkeys(path for _, path in paths))


class SpectatorState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.reset = threading.Event()
        self.sequence = 0
        self.frame: dict | None = None
        self.pending_match: tuple[Path, Path, bool, bool] | None = None

    def publish(self, frame: dict) -> None:
        with self.condition:
            self.sequence += 1
            self.frame = frame
            self.condition.notify_all()

    def select_match(
        self,
        blue_path:      Path,
        orange_path:    Path,
        sample_actions: bool,
        replay_resets:  bool,
    ) -> None:
        with self.condition:
            self.pending_match = (
                blue_path,
                orange_path,
                sample_actions,
                replay_resets,
            )

    def take_match(self) -> tuple[Path, Path, bool, bool] | None:
        with self.condition:
            match = self.pending_match
            self.pending_match = None
            return match


class ReplayResetController:
    def __init__(
        self,
        dataset_path: Path,
        seed:         int,
    ) -> None:
        self.dataset_path = dataset_path
        self.seed = seed
        self.sampler: DatasetResetSampler | None = None

    def configure(
        self,
        environment: CARLTorchVectorEnv,
        enabled:     bool,
    ) -> None:
        if enabled and self.sampler is None:
            dataset = load_replay_dataset(self.dataset_path, environment.device)
            self.sampler = DatasetResetSampler(dataset, seed=self.seed)
        environment.reset_state_provider = self.sampler if enabled else None


def actor_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    if "head.model.0.weight" in state:
        return state

    actor = {
        key.removeprefix("actor."): value
        for key, value in state.items()
        if key.startswith("actor.")
    }
    if "head.model.0.weight" not in actor:
        raise ValueError("Checkpoint is not a current Goddard actor")
    return actor


def inspect_checkpoint(root: Path, path: Path) -> CheckpointMetadata:
    state = actor_state_dict(path)
    observation_size, hidden_size = actor_dimensions(state)

    return CheckpointMetadata(
        path=path,
        relative_path=path.relative_to(root).as_posix(),
        observation_size=observation_size,
        hidden_size=hidden_size,
        modified=path.stat().st_mtime_ns,
    )


def actor_dimensions(state: dict[str, torch.Tensor]) -> tuple[int, int]:
    head = state["head.model.0.weight"]
    if head.ndim != 2:
        raise ValueError("Checkpoint has an incompatible actor architecture")

    hidden_size, observation_size = head.shape
    half_hidden = hidden_size // 2
    expected_shapes = {
        "head.model.0.weight":      (hidden_size, observation_size),
        "head.model.0.bias":        (hidden_size,),
        "body.rnn.weight_ih_l0":    (3 * hidden_size, hidden_size),
        "body.rnn.weight_hh_l0":    (3 * hidden_size, hidden_size),
        "body.rnn.bias_ih_l0":      (3 * hidden_size,),
        "body.rnn.bias_hh_l0":      (3 * hidden_size,),
        "foot.model.0.weight":      (hidden_size, hidden_size),
        "foot.model.0.bias":        (hidden_size,),
        "foot.model.2.weight":      (half_hidden, hidden_size),
        "foot.model.2.bias":        (half_hidden,),
        "foot.model.4.weight":      (ACTION_LOGITS, half_hidden),
        "foot.model.4.bias":        (ACTION_LOGITS,),
    }
    if hidden_size < 2 or any(
        key not in state or tuple(state[key].shape) != shape
        for key, shape in expected_shapes.items()
    ):
        raise ValueError("Checkpoint has an incompatible actor architecture")
    return observation_size, hidden_size


def build_actor(
    environment: CARLTorchVectorEnv,
    hidden_size: int,
) -> MultiCategoricalPolicy:
    head = LinearEncoder(hidden_size, func=nn.ReLU).build(environment)
    body = GRU(hidden_size=hidden_size).build(head.feats)
    actor = MultiCategoricalPolicy(
        head=head,
        body=body,
        foot=MLP(
            dims=[hidden_size, hidden_size // 2],
            func=nn.LeakyReLU,
            out_init_func=partial(init_layer, std=0.01),
        ),
        action_codec=environment.action_codec,
    )
    return actor.build_composed(environment, body.feats).to(environment.device)


def load_actor(
    path:        Path,
    environment: CARLTorchVectorEnv,
) -> MultiCategoricalPolicy:
    state = actor_state_dict(path)
    observation_size, hidden_size = actor_dimensions(state)
    expected_size = environment.single_observation_space.shape[0]
    if observation_size != expected_size:
        raise ValueError(
            f"Checkpoint expects {observation_size} observations, "
            f"but CARL provides {expected_size}"
        )

    actor = build_actor(environment, hidden_size)
    actor.load_state_dict(state)
    actor.eval().requires_grad_(False)
    return actor


def load_pair(
    blue_path:   Path,
    orange_path: Path,
    environment: CARLTorchVectorEnv,
) -> tuple[MultiCategoricalPolicy, MultiCategoricalPolicy]:
    blue = load_actor(blue_path, environment)
    orange = load_actor(orange_path, environment)
    return blue, orange


def vector(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values]


def checkpoint_label(path: Path) -> str:
    return path.stem.removeprefix("policy_")


def raw_state(environment: CARLTorchVectorEnv) -> torch.Tensor:
    torch.cuda.synchronize(environment.device)
    return torch.from_dlpack(environment._env.get_state()).clone()


def render_frame(
    raw:             torch.Tensor,
    checkpoint_root: Path,
    blue_path:       Path,
    orange_path:     Path,
    blue_score:      int,
    orange_score:    int,
    round_number:    int,
    tick:            int,
    sample_actions:  bool,
    replay_resets:   bool,
) -> dict:
    raw = raw[0].detach().cpu()
    ball = raw[:9]
    car_values = raw[9 : 9 + 2 * 22].view(2, 22)
    cars = []

    for team, car in enumerate(car_values):
        forward = car[9:12]
        up = car[12:15]
        right = torch.linalg.cross(up, forward, dim=-1)
        center = (
            car[:3]
            + forward * CAR_OFFSET[0]
            + right * CAR_OFFSET[1]
            + up * CAR_OFFSET[2]
        )
        cars.append(
            {
                "team":     team,
                "pos":      vector(center),
                "fwd":      vector(forward),
                "rgt":      vector(right),
                "up":       vector(up),
                "boost":    float(car[15]),
                "boosting": bool(car[20]),
                "demoed":   bool(car[17]),
            }
        )

    return {
        "tick":           tick,
        "round":          round_number,
        "sample_actions": sample_actions,
        "replay_resets":  replay_resets,
        "blue": {
            "checkpoint": checkpoint_label(blue_path),
            "path":       blue_path.relative_to(checkpoint_root).as_posix(),
            "score":      blue_score,
        },
        "orange": {
            "checkpoint": checkpoint_label(orange_path),
            "path":       orange_path.relative_to(checkpoint_root).as_posix(),
            "score":      orange_score,
        },
        "cars": cars,
        "ball": {"pos": vector(ball[:3])},
    }


def simulate(
    state:           SpectatorState,
    checkpoint_root: Path,
    blue_path:       Path,
    orange_path:     Path,
    tick_skip:       int,
    max_ticks:       int,
    sample_actions:  bool,
    replay_resets:   bool,
    replay_dataset:  Path,
    seed:            int,
) -> None:
    environment = CARLTorchVectorEnv(
        n_sim=1,
        n_blue=1,
        n_orange=1,
        seed=seed,
        frameskip=tick_skip,
        max_ticks=max_ticks,
        synchronize=True,
    )
    reset_controller = ReplayResetController(replay_dataset, seed)

    try:
        blue, orange = load_pair(blue_path, orange_path, environment)
        reset_controller.configure(environment, replay_resets)
        observations = environment.reset()
        hidden = [blue.initial_state(1), orange.initial_state(1)]
        blue_score = orange_score = 0
        round_number = 1
        tick = 0
        next_step = time.perf_counter()

        while not state.stop.is_set():
            pending_match = state.take_match()
            if pending_match is not None:
                (
                    next_blue_path,
                    next_orange_path,
                    next_sample,
                    next_replay_resets,
                ) = pending_match
                try:
                    next_blue, next_orange = load_pair(
                        next_blue_path, next_orange_path, environment
                    )
                    reset_controller.configure(environment, next_replay_resets)
                    next_observations = environment.reset()
                except Exception as error:
                    state.publish({"error": f"{type(error).__name__}: {error}"})
                else:
                    blue_path = next_blue_path
                    orange_path = next_orange_path
                    blue = next_blue
                    orange = next_orange
                    sample_actions = next_sample
                    replay_resets = next_replay_resets
                    observations = next_observations
                    hidden = [blue.initial_state(1), orange.initial_state(1)]
                    blue_score = orange_score = 0
                    round_number = 1
                    tick = 0
                    next_step = time.perf_counter()

            if state.reset.is_set():
                state.reset.clear()
                observations = environment.reset()
                hidden = [blue.initial_state(1), orange.initial_state(1)]
                blue_score = orange_score = 0
                round_number = 1
                tick = 0

            with torch.inference_mode():
                blue_output = blue.act(
                    observations[0:1],
                    hidden[0],
                    deterministic=not sample_actions,
                )
                orange_output = orange.act(
                    observations[1:2],
                    hidden[1],
                    deterministic=not sample_actions,
                )
            hidden = [blue_output.next_state, orange_output.next_state]
            actions = torch.cat((blue_output.action, orange_output.action), dim=0)
            observations, reward, terminated, truncated, _ = environment.step(actions)
            tick += tick_skip

            goal = int(reward[0].item())
            if goal > 0:
                blue_score += goal
            elif goal < 0:
                orange_score -= goal

            if (terminated | truncated).any():
                hidden = [blue.initial_state(1), orange.initial_state(1)]
                round_number += 1
                tick = 0

            state.publish(
                render_frame(
                    raw_state(environment),
                    checkpoint_root,
                    blue_path,
                    orange_path,
                    blue_score,
                    orange_score,
                    round_number,
                    tick,
                    sample_actions,
                    replay_resets,
                )
            )

            next_step += tick_skip / 120.0
            delay = next_step - time.perf_counter()
            if delay > 0:
                state.stop.wait(delay)
            else:
                next_step = time.perf_counter()
    except Exception as error:
        state.publish({"error": f"{type(error).__name__}: {error}"})
    finally:
        environment.close()


def make_handler(
    state:    SpectatorState,
    frontend: Path,
    arena:    Path,
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
                if not 0 < length <= 8192:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                blue_path = registry.resolve(payload["blue"])
                orange_path = registry.resolve(payload["orange"])
                sample_actions = payload.get("sample_actions", False)
                if not isinstance(sample_actions, bool):
                    raise TypeError("sample_actions must be a Boolean")
                replay_resets = payload.get("replay_resets", False)
                if not isinstance(replay_resets, bool):
                    raise TypeError("replay_resets must be a Boolean")
                blue = inspect_checkpoint(registry.directory, blue_path)
                orange = inspect_checkpoint(registry.directory, orange_path)
                if blue.observation_size != orange.observation_size:
                    raise ValueError("Selected checkpoints are incompatible")
            except (
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                self.send_error(HTTPStatus.BAD_REQUEST, str(error))
                return

            state.select_match(
                blue_path,
                orange_path,
                sample_actions,
                replay_resets,
            )
            self.send_response(HTTPStatus.ACCEPTED)
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/api/checkpoints":
                self._send_json([item.as_dict() for item in registry.list()])
                return
            if self.path == "/api/stream":
                self._stream_frames()
                return

            paths = {
                "/":          frontend / "index.html",
                "/index.html": frontend / "index.html",
                "/app.js":     frontend / "app.js",
                "/arena.obj":  arena,
            }
            path = paths.get(self.path)
            if path is None or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            payload = path.read_bytes()
            content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, value) -> None:
            payload = json.dumps(value).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _stream_frames(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            sequence = 0

            try:
                while True:
                    with state.condition:
                        state.condition.wait_for(
                            lambda: state.sequence > sequence,
                            timeout=10.0,
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


def parse_arguments() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=root / "checkpoints")
    parser.add_argument("--blue")
    parser.add_argument("--orange")
    parser.add_argument("--tick-skip", type=int, default=8)
    parser.add_argument("--max-ticks", type=int, default=4096)
    parser.add_argument("--sample-actions", action="store_true")
    parser.add_argument("--replay-resets", action="store_true")
    parser.add_argument(
        "--replay-dataset",
        type=Path,
        default=root / "data" / "ballchasing-ssl-1v1" / "dataset",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--open", action="store_true")
    arguments = parser.parse_args()

    if (arguments.blue is None) != (arguments.orange is None):
        parser.error("--blue and --orange must be provided together")
    if arguments.tick_skip < 1 or arguments.max_ticks < 1:
        parser.error("tick and episode lengths must be positive")
    if not 1 <= arguments.port <= 65535:
        parser.error("port must be between 1 and 65535")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CARL checkpoint playback requires CUDA")

    registry = CheckpointRegistry(arguments.checkpoint_dir)
    if arguments.blue is None:
        blue_path, orange_path = registry.newest_pair()
    else:
        blue_path = registry.resolve(arguments.blue)
        orange_path = registry.resolve(arguments.orange)

    root = Path(__file__).resolve().parent
    frontend = root / "web" / "checkpoint"
    arena = Path(carl.__file__).resolve().parent / "assets" / "arena.obj"
    if not frontend.is_dir() or not arena.is_file():
        raise FileNotFoundError("Checkpoint spectator assets are missing")

    state = SpectatorState()
    thread = threading.Thread(
        target=simulate,
        args=(
            state,
            registry.directory,
            blue_path,
            orange_path,
            arguments.tick_skip,
            arguments.max_ticks,
            arguments.sample_actions,
            arguments.replay_resets,
            arguments.replay_dataset,
            arguments.seed,
        ),
        daemon=True,
    )
    thread.start()

    url = f"http://{arguments.host}:{arguments.port}"
    print(f"Blue:   {blue_path}")
    print(f"Orange: {orange_path}")
    print(f"Viewer: {url}")
    if arguments.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    server = ThreadingHTTPServer(
        (arguments.host, arguments.port),
        make_handler(state, frontend, arena, registry),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop.set()
        thread.join(timeout=5.0)
        server.server_close()


if __name__ == "__main__":
    main()
