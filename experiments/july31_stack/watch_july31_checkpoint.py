"""Watch July 31 CARL/JARL actor checkpoints play in the existing web arena.

Uses the pinned July 31 simulation and reset corpus. The frontend comes from
the preserved checkpoint viewer; the original actor weights are never edited.
"""

import argparse
import json
import mimetypes
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import carl
import ppo
import torch
from jarl.envs import DatasetResetSampler
from replay_states import load_replay_dataset


CAR_OFFSET = (13.8757, 0.0, 20.755)


class ViewerState:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.reset = threading.Event()
        self.kickoff = threading.Event()
        self.frame = None
        self.sequence = 0
        self.pending_match = None

    def publish(self, frame: dict) -> None:
        with self.condition:
            self.frame = frame
            self.sequence += 1
            self.condition.notify_all()


def checkpoints(root: Path) -> list[dict]:
    return [
        {
            "path": path.name,
            "label": path.stem,
            "step": int(path.stem.removeprefix("policy_")) if path.stem.removeprefix("policy_").isdigit() else 0,
            "modified": path.stat().st_mtime_ns,
            "kind": "basic",
        }
        for path in sorted(root.glob("policy_*.pt"), reverse=True)
    ]


def resolve_checkpoint(root: Path, filename: str) -> Path:
    path = (root / filename).resolve()
    if root.resolve() not in path.parents or not path.is_file() or not path.name.startswith("policy_") or path.suffix != ".pt":
        raise ValueError("invalid checkpoint")
    return path


def load_policy(path: Path, env):
    actor, _ = ppo.build_policy_and_critic(env, SimpleNamespace(hidden_size=256))
    source = torch.load(path, map_location="cpu", weights_only=True)
    weights = source.get("modules", {}).get("policy", source)
    if "policy" in weights:
        weights = weights["policy"]
    if "source_sha256" in source:
        weights = {
            ("foot." + key[5:] if key.startswith("head.") else
             "head." + key[5:] if key.startswith("foot.") else key): value
            for key, value in weights.items()
        }
    actor.load_state_dict(weights, strict=True)
    return actor.eval().requires_grad_(False)


def render_frame(env, root: Path, blue: Path, orange: Path, scores, round_number, tick):
    env._sync()
    raw = torch.from_dlpack(env._env.get_state()).clone()[0].cpu()
    cars = []
    for team, car in enumerate(raw[9:53].view(2, 22)):
        forward, up = car[9:12], car[12:15]
        right = torch.linalg.cross(up, forward, dim=-1)
        center = car[:3] + forward * CAR_OFFSET[0] + right * CAR_OFFSET[1] + up * CAR_OFFSET[2]
        cars.append({
            "team": team,
            "pos": center.tolist(),
            "fwd": forward.tolist(),
            "rgt": right.tolist(),
            "up": up.tolist(),
            "boost": float(car[15]),
            "boosting": bool(car[20]),
            "demoed": bool(car[17]),
        })
    return {
        "tick": tick,
        "round": round_number,
        "blue": {"checkpoint": blue.stem, "path": blue.name, "score": scores[0]},
        "orange": {"checkpoint": orange.stem, "path": orange.name, "score": scores[1]},
        "cars": cars,
        "ball": {"pos": raw[:3].tolist()},
    }


def simulate(state: ViewerState, options) -> None:
    env = None
    try:
        torch.manual_seed(options.seed)
        corpus = load_replay_dataset(options.dataset, device="cuda:0")
        provider = ppo.SyntheticMatchResetProvider(
            DatasetResetSampler(corpus, probability=options.replay_probability, seed=options.seed)
        )
        env = ppo.CARLTorchVectorEnv(
            n_sim=1,
            n_blue=1,
            n_orange=1,
            seed=options.seed,
            frameskip=8,
            max_ticks=36_000,
            no_touch_timeout_seconds=30.0,
            normalize=True,
            synchronize=True,
            reset_state_provider=provider,
        )
        paths = [resolve_checkpoint(options.checkpoint_dir, options.blue),
                 resolve_checkpoint(options.checkpoint_dir, options.orange)]
        actors = [load_policy(path, env) for path in paths]
        states = [actor.initial_state(1) for actor in actors]
        observation = env.reset()
        scores = [0, 0]
        round_number, tick = 1, 0
        next_step = time.perf_counter()

        while not state.stop.is_set():
            with state.condition:
                match, state.pending_match = state.pending_match, None
            if match is not None:
                try:
                    new_paths = [resolve_checkpoint(options.checkpoint_dir, name) for name in match]
                    new_actors = [load_policy(path, env) for path in new_paths]
                except Exception as error:
                    state.publish({"error": f"{type(error).__name__}: {error}"})
                else:
                    paths, actors = new_paths, new_actors
                    state.reset.set()

            if state.reset.is_set() or state.kickoff.is_set():
                kickoff = state.kickoff.is_set()
                state.reset.clear()
                state.kickoff.clear()
                if kickoff:
                    env.reset_state_provider = None
                try:
                    observation = env.reset()
                finally:
                    env.reset_state_provider = provider
                states = [actor.initial_state(1) for actor in actors]
                scores = [0, 0]
                round_number, tick = 1, 0

            with torch.inference_mode():
                outputs = [
                    actor.act(observation[i:i+1], states[i], deterministic=not options.sample)
                    for i, actor in enumerate(actors)
                ]
                states = [output.next_state for output in outputs]
                action = torch.cat([output.action for output in outputs])
            observation, _, terminated, truncated, _ = env.step(action)
            tick += 8
            score_delta = int(env._from_carl(env._env.get_rewards())[0].item())
            scores[0] += max(score_delta, 0)
            scores[1] += max(-score_delta, 0)
            if (terminated | truncated).any():
                states = [actor.initial_state(1) for actor in actors]
                round_number += 1
                tick = 0
            state.publish(render_frame(env, options.checkpoint_dir, *paths, scores, round_number, tick))
            next_step += 8 / 120.0
            delay = next_step - time.perf_counter()
            if delay > 0:
                state.stop.wait(delay)
            else:
                next_step = time.perf_counter()
    except Exception as error:
        traceback.print_exc()
        state.publish({"error": f"{type(error).__name__}: {error}"})
    finally:
        if env is not None:
            env.close()


def make_handler(state: ViewerState, options):
    arena = Path(carl.__file__).resolve().parent / "assets" / "arena.obj"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/api/checkpoints":
                payload = json.dumps(checkpoints(options.checkpoint_dir)).encode()
                content_type = "application/json"
            elif self.path == "/api/stream":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                sequence = 0
                try:
                    while True:
                        with state.condition:
                            state.condition.wait_for(lambda: state.sequence > sequence, timeout=10)
                            if state.sequence == sequence:
                                self.wfile.write(b": keepalive\n\n")
                                self.wfile.flush()
                                continue
                            sequence = state.sequence
                            frame = json.dumps(state.frame, separators=(",", ":"))
                        self.wfile.write(f"data: {frame}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            else:
                path = {
                    "/": options.frontend / "index.html",
                    "/app.js": options.frontend / "app.js",
                    "/arena.obj": arena,
                }.get(self.path)
                if path is None or not path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                payload = path.read_bytes()
                content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if self.path in ("/api/reset", "/api/kickoff"):
                (state.reset if self.path == "/api/reset" else state.kickoff).set()
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return
            if self.path != "/api/match":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                paths = tuple(resolve_checkpoint(options.checkpoint_dir, data[key]).name for key in ("blue", "orange"))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self.send_error(HTTPStatus.BAD_REQUEST, str(error))
                return
            with state.condition:
                state.pending_match = paths
            self.send_response(HTTPStatus.ACCEPTED)
            self.end_headers()

        def log_message(self, format, *args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/ballchasing-ssl-1v1/reset_dataset"))
    parser.add_argument("--frontend", type=Path, default=Path("/Goddard-pre-july31-20260924/web/self_play"))
    parser.add_argument("--blue", default="policy_003014580461.pt")
    parser.add_argument("--orange", default="policy_003014580461.pt")
    parser.add_argument("--replay-probability", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=9210)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--sample", action="store_true", help="sample actions rather than use the policy mode")
    options = parser.parse_args()
    if not 0 <= options.replay_probability <= 1:
        parser.error("replay probability must be between zero and one")
    for filename in (options.blue, options.orange):
        resolve_checkpoint(options.checkpoint_dir, filename)
    state = ViewerState()
    worker = threading.Thread(target=simulate, args=(state, options), daemon=True)
    worker.start()
    server = ThreadingHTTPServer((options.host, options.port), make_handler(state, options))
    print(f"Watching {options.blue} vs {options.orange} on http://{options.host}:{options.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        state.stop.set()
        worker.join(timeout=5)
        server.server_close()


if __name__ == "__main__":
    main()
