#!/usr/bin/env python3
"""Watch deterministic GAIFO self-play from standard 1v1 starting states."""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import carl
import gaifo
import torch as th

from carl.gymnasium import CARLTorchVectorEnv


ROOT = Path(__file__).parent
GAIFO_ARCHITECTURE = gaifo.GAIFO_ARCHITECTURE


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
        for path in self.directory.rglob("gaifo_*.pt"):
            try:
                resolved = path.resolve(strict=True)
                checkpoints.append(CheckpointMetadata(
                    resolved,
                    resolved.relative_to(self.directory).as_posix(),
                    int(resolved.stem.removeprefix("gaifo_")),
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
                f"no GAIFO checkpoints found in {self.directory}"
            )
        newest = checkpoints[0]
        return newest.path, newest.path

    def resolve(self, value: str) -> Path:
        path = (self.directory / value).resolve()
        if (
            self.directory not in path.parents
            or not path.is_file()
            or not path.match("gaifo_*.pt")
        ):
            raise ValueError("invalid checkpoint path")
        return path


def require_gaifo_config(config: dict, path: Path) -> None:
    if config.get("architecture") != GAIFO_ARCHITECTURE:
        raise ValueError(
            f"checkpoint {path} has architecture {config.get('architecture')!r}, "
            f"expected {GAIFO_ARCHITECTURE!r}"
        )


def resolve_frameskip(
    blue_config: dict,
    orange_config: dict,
    explicit: int | None,
) -> int:
    blue_frameskip = int(blue_config["frameskip"])
    orange_frameskip = int(orange_config["frameskip"])
    if blue_frameskip != orange_frameskip:
        raise ValueError(
            f"frameskip mismatch: blue {blue_frameskip} != orange {orange_frameskip}"
        )
    if explicit is not None and blue_frameskip != explicit:
        raise ValueError(
            f"checkpoint frameskip {blue_frameskip} does not match --frameskip {explicit}"
        )
    return blue_frameskip


def require_compatible_checkpoints(
    blue_path: Path,
    blue_payload: dict,
    orange_path: Path,
    orange_payload: dict,
    explicit_frameskip: int | None,
) -> int:
    require_gaifo_config(blue_payload["config"], blue_path)
    require_gaifo_config(orange_payload["config"], orange_path)
    return resolve_frameskip(
        blue_payload["config"],
        orange_payload["config"],
        explicit_frameskip,
    )


def load_policy(path: Path, env: CARLTorchVectorEnv) -> th.nn.Module:
    payload = th.load(path, map_location="cpu", weights_only=True)
    config = payload["config"]
    require_gaifo_config(config, path)
    policy_args = SimpleNamespace(
        policy_hidden=int(config["policy_hidden"]),
    )
    policy = gaifo.build_policy(env, policy_args)
    policy.load_state_dict(payload["policy"])
    return policy.eval().requires_grad_(False)


def select_actions(
    blue_policy: th.nn.Module,
    orange_policy: th.nn.Module,
    observation: th.Tensor,
    blue_state: th.Tensor | None,
    orange_state: th.Tensor | None,
) -> tuple[th.Tensor, th.Tensor | None, th.Tensor | None]:
    with th.inference_mode():
        blue_output = blue_policy.act(
            observation[:1], blue_state, deterministic=True
        )
        orange_output = orange_policy.act(
            observation[1:], orange_state, deterministic=True
        )
        actions = th.cat((blue_output.action, orange_output.action))
    return actions, blue_output.next_state, orange_output.next_state


def simulate(
    state: SpectatorState,
    registry: CheckpointRegistry,
    blue_path: Path,
    orange_path: Path,
    args: argparse.Namespace,
) -> None:
    from watch_checkpoints import raw_state, render_frame

    env = None
    try:
        blue_payload = th.load(blue_path, map_location="cpu", weights_only=True)
        orange_payload = th.load(orange_path, map_location="cpu", weights_only=True)
        frameskip = require_compatible_checkpoints(
            blue_path,
            blue_payload,
            orange_path,
            orange_payload,
            args.frameskip,
        )
        env = CARLTorchVectorEnv(
            n_sim=1,
            n_blue=1,
            n_orange=1,
            seed=args.seed,
            frameskip=frameskip,
            max_ticks=args.max_ticks,
            normalize=True,
            synchronize=True,
            discrete_actions=True,
        )
        blue = load_policy(blue_path, env)
        orange = load_policy(orange_path, env)

        observation = env.reset()
        blue_state = blue.initial_state(1)
        orange_state = orange.initial_state(1)
        blue_score = 0
        orange_score = 0
        round_number = 1
        tick = 0
        next_step = time.perf_counter()

        while not state.stop.is_set():
            pending = state.take_match()
            if pending is not None:
                try:
                    next_blue_path, next_orange_path = pending
                    next_blue_payload = th.load(
                        next_blue_path, map_location="cpu", weights_only=True
                    )
                    next_orange_payload = th.load(
                        next_orange_path, map_location="cpu", weights_only=True
                    )
                    require_compatible_checkpoints(
                        next_blue_path,
                        next_blue_payload,
                        next_orange_path,
                        next_orange_payload,
                        frameskip,
                    )
                    next_blue = load_policy(next_blue_path, env)
                    next_orange = load_policy(next_orange_path, env)
                except Exception as error:
                    state.publish({"error": f"{type(error).__name__}: {error}"})
                else:
                    blue_path, orange_path = next_blue_path, next_orange_path
                    blue_payload, orange_payload = next_blue_payload, next_orange_payload
                    blue, orange = next_blue, next_orange
                    state.reset.set()

            if state.reset.is_set():
                state.reset.clear()
                observation = env.reset()
                blue_state = blue.initial_state(1)
                orange_state = orange.initial_state(1)
                blue_score = orange_score = 0
                round_number = 1
                tick = 0

            actions, blue_state, orange_state = select_actions(
                blue, orange, observation, blue_state, orange_state
            )
            observation, reward, terminated, truncated, _ = env.step(actions)
            tick += frameskip

            score_delta = int(reward[0].item())
            blue_score += max(score_delta, 0)
            orange_score += max(-score_delta, 0)

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
            next_step += frameskip / 120.0
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/gaifo")
    )
    parser.add_argument("--blue")
    parser.add_argument("--orange")
    parser.add_argument("--frameskip", type=int)
    parser.add_argument("--max-ticks", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if (args.blue is None) != (args.orange is None):
        parser.error("--blue and --orange must be provided together")
    if args.frameskip is not None and args.frameskip < 1:
        parser.error("--frameskip must be positive")
    if args.max_ticks < 1:
        parser.error("--max-ticks must be positive")
    if args.port < 1:
        parser.error("--port must be positive")
    return args


def main() -> None:
    from watch_checkpoints import SpectatorState, make_handler

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
