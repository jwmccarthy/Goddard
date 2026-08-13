#!/usr/bin/env python3
"""Watch the newest ASE checkpoint play in a live browser viewer."""

import argparse
import json
import threading
import time
import webbrowser

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch as th
import torch.nn as nn

import carl
from carl.gymnasium import CARLTorchVectorEnv
from jarl.modules import GRU, MLP
from jarl.modules.layer import orthogonal_init

from capture import LatentCapture
from config import ASEConfig
from encoder import LatentEncoder
from expert_dataset import ExpertStateResetProvider, ExpertTrajectoryDataset
from modules import LatentMultiCategoricalPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
    checkpoints = list(directory.rglob("ase_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no ASE checkpoints found under {directory}")

    return max(checkpoints, key=lambda path: path.stat().st_mtime_ns)


def load_policy(path: Path, env, latents):
    payload = th.load(path, map_location="cpu", weights_only=True)
    state = payload.get("policy", payload)
    recurrent = "body.rnn.weight_ih_l0" in state
    body = (
        GRU(hidden_size=state["body.rnn.weight_hh_l0"].shape[1])
        if recurrent
        else MLP(dims=[1024, 1024, 512], func=nn.ReLU)
    )
    policy = LatentMultiCategoricalPolicy(
        foot=LatentEncoder(latents),
        body=body,
        head=MLP(
            dims=[],
            out_init_func=orthogonal_init(0.01)
        ),
        action_codec=env.action_codec
    ).build(env).to(env.device)

    policy.load_state_dict(state)
    policy.eval().requires_grad_(False)

    return policy


def raw_state(env: CARLTorchVectorEnv) -> th.Tensor:
    th.cuda.synchronize(env.device)
    return th.from_dlpack(env._env.get_state()).clone()[0]


def frame_from_state(
    state:      th.Tensor,
    checkpoint: Path,
    latents:    LatentCapture,
    blue_score: int,
    orange_score: int
) -> dict:
    cars = state[9:53].view(2, 22)

    rendered_cars = []
    for index, car in enumerate(cars):
        forward = car[9:12]
        up = car[12:15]
        right = th.linalg.cross(up, forward, dim=-1)
        position = (
            car[0:3]
            + forward * CAR_OFFSET[0]
            + right * CAR_OFFSET[1]
            + up * CAR_OFFSET[2]
        )

        rendered_cars.append({
            "team": index,
            "pos": position.cpu().tolist(),
            "fwd": forward.cpu().tolist(),
            "rgt": right.cpu().tolist(),
            "up": up.cpu().tolist(),
            "demoed": bool(car[17])
        })

    return {
        "checkpoint": checkpoint.name,
        "blue_score": blue_score,
        "orange_score": orange_score,
        "ball": {"pos": state[0:3].cpu().tolist()},
        "cars": rendered_cars,
        "latent_steps": latents.steps.cpu().tolist()
    }


def simulate(state: ViewerState, args: argparse.Namespace) -> None:
    config = ASEConfig()
    env = CARLTorchVectorEnv(
        n_sim=1,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=config.frameskip,
        max_ticks=config.max_ticks,
        normalize=True,
        synchronize=True
    )
    latents = LatentCapture(
        latent_dim=config.latent_dim,
        device=env.device,
        seed=args.seed
    )

    try:
        demos = ExpertTrajectoryDataset(
            args.expert_dir,
            obs_limit=args.expert_limit,
            device=env.device
        )
        env.reset_state_provider = ExpertStateResetProvider(demos, env.device)

        checkpoint = newest_checkpoint(args.checkpoint_dir)
        policy = load_policy(checkpoint, env, latents)
        checkpoint_mtime = checkpoint.stat().st_mtime_ns

        observation = env.reset()
        latents.reset(env.n_envs)
        policy_state = policy.initial_state(env.n_envs)
        blue_score = orange_score = 0
        next_step = time.perf_counter()

        while not state.stop.is_set():
            if state.reset.is_set():
                state.reset.clear()
                observation = env.reset()
                latents.reset(env.n_envs)
                policy_state = policy.initial_state(env.n_envs)
                blue_score = orange_score = 0

            latest = newest_checkpoint(args.checkpoint_dir)
            latest_mtime = latest.stat().st_mtime_ns
            if latest != checkpoint or latest_mtime != checkpoint_mtime:
                policy = load_policy(latest, env, latents)
                checkpoint = latest
                checkpoint_mtime = latest_mtime
                policy_state = policy.initial_state(env.n_envs)

            with th.no_grad():
                output = policy.act(
                    observation,
                    policy_state,
                    deterministic=not args.sample_actions
                )
                observation, reward, terminated, truncated, _ = env.step(output.action)

            done = terminated | truncated
            policy_state = output.next_state
            if policy_state is not None and done.any():
                policy_state = policy_state.clone()
                policy_state[done] = 0
            latents.advance(done)

            goal = int(reward[0].item())
            if goal > 0:
                blue_score += goal
            elif goal < 0:
                orange_score -= goal

            state.publish(frame_from_state(
                raw_state(env),
                checkpoint,
                latents,
                blue_score,
                orange_score
            ))

            next_step += config.frameskip / (120.0 * args.fast_forward)
            delay = next_step - time.perf_counter()
            if delay > 0:
                state.stop.wait(delay)
            else:
                next_step = time.perf_counter()

    except Exception as error:
        state.publish({"error": f"{type(error).__name__}: {error}"})
    finally:
        env.close()


def make_handler(state: ViewerState, frontend: Path, arena: Path):
    class Handler(BaseHTTPRequestHandler):

        def do_POST(self) -> None:
            if self.path != "/reset":
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            state.reset.set()
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/api/stream":
                self._stream()
                return

            paths = {
                "/": frontend / "index.html",
                "/app.js": frontend / "app.js",
                "/arena.obj": arena
            }
            path = paths.get(self.path)
            if path is None or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            payload = path.read_bytes()
            content_type = {
                ".html": "text/html; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".obj": "text/plain"
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
                    with state.condition:
                        state.condition.wait_for(
                            lambda: state.sequence > sequence,
                            timeout=10
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
    config = ASEConfig()
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--checkpoint-dir", type=Path, default=config.checkpoint_dir)
    parser.add_argument("--expert-dir", type=Path, default=config.expert_dir)
    parser.add_argument("--expert-limit", type=int, default=16_000_000)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--fast-forward", type=int, default=1)
    parser.add_argument("--sample-actions", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--open", action="store_true")

    args = parser.parse_args()
    if args.expert_limit < 2 or args.fast_forward < 1:
        parser.error("expert limit and fast-forward must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")

    return args


def main() -> None:
    args = parse_arguments()
    state = ViewerState()
    thread = threading.Thread(target=simulate, args=(state, args), daemon=True)
    thread.start()

    url = f"http://{args.host}:{args.port}"
    print(f"Viewer: {url}")

    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    frontend = PROJECT_ROOT / "web" / "checkpoint"
    arena = Path(carl.__file__).resolve().parent / "assets" / "arena.obj"
    if not frontend.is_dir() or not arena.is_file():
        raise FileNotFoundError("checkpoint spectator assets are missing")

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(state, frontend, arena)
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
