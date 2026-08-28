#!/usr/bin/env python3
"""Watch the newest tracker checkpoint in a browser."""

import argparse
import json
import threading
import time
import webbrowser

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import carl
import torch as th
import torch.nn as nn

from carl.gymnasium import CARLObservation, CARLTorchVectorEnv
from jarl.modules import MLP
from jarl.modules.encoder import LinearEncoder
from jarl.modules.policy import MultiCategoricalPolicy

from tracker import ExpertGoalStates, ExpertLookaheadEnv, POSITION_SCALE


ROOT = Path(__file__).parent
CAR_OFFSET = (13.8757, 0.0, 20.755)


class ViewerState:

    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.reset = threading.Event()
        self.sequence = 0
        self.frame = None

    def publish(self, frame: dict) -> None:
        with self.condition:
            self.sequence += 1
            self.frame = frame
            self.condition.notify_all()


def newest_checkpoint(directory: Path) -> Path:
    paths = list(directory.glob("tracker_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no tracker checkpoints found in {directory}")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def load_policy(path: Path, env: ExpertLookaheadEnv):
    payload = th.load(path, map_location=env.device, weights_only=True)
    policy = MultiCategoricalPolicy(
        foot=LinearEncoder(512, func=nn.ReLU),
        body=MLP(dims=[512, 512], func=nn.ReLU),
        head=MLP(dims=[]),
        action_codec=env.action_codec,
    ).build(env).to(env.device)
    policy.load_state_dict(payload["policy"])
    return policy.eval().requires_grad_(False)


def frame_from_expert(expert: CARLObservation) -> dict:
    ball = expert.ball
    cars = expert.cars
    scale = th.tensor(POSITION_SCALE, device=expert.device)
    rendered = []

    for index in range(expert.n_cars):
        forward = cars.forward[index]
        up = cars.up[index]
        right = th.linalg.cross(up, forward, dim=-1)
        position = (
            cars.position[index] * scale
            + forward * CAR_OFFSET[0]
            + right * CAR_OFFSET[1]
            + up * CAR_OFFSET[2]
        )
        rendered.append({
            "team":   index,
            "pos":    position.cpu().tolist(),
            "fwd":    forward.cpu().tolist(),
            "rgt":    right.cpu().tolist(),
            "up":     up.cpu().tolist(),
            "demoed": bool(cars.demoed[index]),
        })

    return {
        "ball": {"pos": (ball.position * scale).cpu().tolist()},
        "cars": rendered,
    }


def frame_from_state(
    state:      th.Tensor,
    checkpoint: Path,
    reward:     th.Tensor,
    expert:     CARLObservation,
) -> dict:
    cars = state[9:31].view(1, 22)
    rendered = []

    for index, car in enumerate(cars):
        forward = car[9:12]
        up = car[12:15]
        right = th.linalg.cross(up, forward, dim=-1)
        position = (
            car[:3]
            + forward * CAR_OFFSET[0]
            + right * CAR_OFFSET[1]
            + up * CAR_OFFSET[2]
        )
        rendered.append({
            "team":    index,
            "pos":     position.cpu().tolist(),
            "fwd":     forward.cpu().tolist(),
            "rgt":     right.cpu().tolist(),
            "up":      up.cpu().tolist(),
            "demoed":  bool(car[17]),
        })

    return {
        "checkpoint": checkpoint.name,
        "reward":     reward.cpu().tolist(),
        "ball":       {"pos": state[:3].cpu().tolist()},
        "cars":       rendered,
        "expert":     frame_from_expert(expert),
    }


def publish_frame(
    viewer:     ViewerState,
    base:       CARLTorchVectorEnv,
    replays:    ExpertGoalStates,
    checkpoint: Path,
    reward:     th.Tensor,
) -> None:
    th.cuda.synchronize(base.device)
    raw = th.from_dlpack(base._env.get_state()).clone()[0]
    expert = CARLObservation.from_tensor(replays.current(-1)[0], 1)
    viewer.publish(frame_from_state(raw, checkpoint, reward, expert))


def simulate(viewer: ViewerState, args: argparse.Namespace) -> None:
    base = CARLTorchVectorEnv(
        n_sim=1,
        n_blue=1,
        n_orange=0,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=1_000_000,
        normalize=True,
        synchronize=True,
    )
    replays = ExpertGoalStates(
        str(args.replay_dir),
        n_env=1,
        windows=args.windows,
        obs_limit=args.obs_limit,
        n_cars=1,
        device=base.device,
    )
    env = ExpertLookaheadEnv(
        base,
        replays,
        reward_scale=args.tracking_reward_scale,
        minimum_reward=args.minimum_tracking_reward,
    )

    try:
        checkpoint = newest_checkpoint(args.checkpoint_dir)
        policy = load_policy(checkpoint, env)
        checkpoint_mtime = checkpoint.stat().st_mtime_ns
        observation = env.reset()
        frame_time = args.frameskip / (120.0 * args.fast_forward)
        publish_frame(viewer, base, replays, checkpoint, th.zeros(1, device=env.device))
        viewer.stop.wait(frame_time)
        next_step = time.perf_counter()

        while not viewer.stop.is_set():
            if viewer.reset.is_set():
                viewer.reset.clear()
                observation = env.reset()
                publish_frame(
                    viewer,
                    base,
                    replays,
                    checkpoint,
                    th.zeros(1, device=env.device),
                )
                viewer.stop.wait(frame_time)
                next_step = time.perf_counter()
                continue

            latest = newest_checkpoint(args.checkpoint_dir)
            latest_mtime = latest.stat().st_mtime_ns
            if latest != checkpoint or latest_mtime != checkpoint_mtime:
                policy = load_policy(latest, env)
                checkpoint = latest
                checkpoint_mtime = latest_mtime

            with th.no_grad():
                action = policy.act(
                    observation,
                    deterministic=not args.sample_actions,
                ).action
                observation, reward, _, _, _ = env.step(action)

            publish_frame(viewer, base, replays, checkpoint, reward)

            next_step += frame_time
            delay = next_step - time.perf_counter()
            if delay > 0:
                viewer.stop.wait(delay)
            else:
                next_step = time.perf_counter()

    except Exception as error:
        viewer.publish({"error": f"{type(error).__name__}: {error}"})
    finally:
        env.close()


def make_handler(viewer: ViewerState, frontend: Path, arena: Path):
    class Handler(BaseHTTPRequestHandler):

        def do_POST(self) -> None:
            if self.path != "/reset":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            viewer.reset.set()
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

        def do_GET(self) -> None:
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
            content_type = {
                ".html": "text/html; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".obj": "text/plain",
            }.get(path.suffix, "application/octet-stream")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
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
                    with viewer.condition:
                        viewer.condition.wait_for(
                            lambda: viewer.sequence > sequence,
                            timeout=10,
                        )
                        if viewer.sequence == sequence:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            continue
                        sequence = viewer.sequence
                        payload = json.dumps(viewer.frame, separators=(",", ":"))

                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/tracker"))
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--obs-limit", type=int, default=100_000)
    parser.add_argument("--tracking-reward-scale", type=float, default=1.0)
    parser.add_argument("--minimum-tracking-reward", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast-forward", type=int, default=1)
    parser.add_argument("--sample-actions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    viewer = ViewerState()
    thread = threading.Thread(target=simulate, args=(viewer, args), daemon=True)
    thread.start()

    url = f"http://{args.host}:{args.port}"
    print(f"Viewer: {url}")
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    arena = Path(carl.__file__).resolve().parent / "assets" / "arena.obj"
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(viewer, ROOT / "web" / "checkpoint", arena),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.stop.set()
        thread.join(timeout=5)
        server.server_close()


if __name__ == "__main__":
    main()
