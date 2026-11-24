"""Find starts that repeat a flip-reset goal: uv run python tools/scenario_search.py CHECKPOINT CLIP_DIR --output starts.json"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    CONTACT,
    RESET,
    TOUCH,
    build,
    reset_from,
    sample_actions,
)

from training.checkpoints import load_payload  # noqa: E402
from training.models import Agent  # noqa: E402
from training.tensors import plain_tensor  # noqa: E402
from vis.playback import build_agent  # noqa: E402

FPS = 30            # clip frames per second (4 ticks)
GROUNDED_Z = 40.0
MIN_LEAD, MAX_LEAD = 1.0, 4.5


def candidates(clip_dir: Path) -> list[dict]:
    summary = json.loads((clip_dir / "summary.json").read_text())
    clips = np.load(clip_dir / "clips.npz")
    found = []

    for goal in summary["goals"]:
        frames = clips[f"frames_{goal['clip']}"]
        grounded = np.ones(len(frames), dtype=bool)

        for car in (0, 1):
            base = 9 + 21 * car
            grounded &= (frames[:, base + 16] > 0.5) & (frames[:, base + 2] < GROUNDED_Z)

        for f in range(len(frames) - round(MIN_LEAD * FPS), -1, -1):
            if len(frames) - f > MAX_LEAD * FPS:
                break

            if grounded[f]:
                found.append({
                    "clip":         goal["clip"],
                    "frame":        f,
                    "scorer":       goal["scorer_car"],
                    "lead_seconds": (len(frames) - f) / FPS,
                    "state":        frames[f]
                })
                break

    return found


@torch.no_grad()
def attempts(
    agent:       Agent,
    config:      dict,
    starts:      np.ndarray,
    scorer:      np.ndarray,
    seconds:     float,
    seed:        int,
    temperature: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(starts)
    env = build(config, n, seed)
    device = env.device
    car = torch.tensor(scorer, device=device)
    observation = reset_from(
        env, torch.tensor(starts, device=device, dtype=torch.float32)
    )

    rows = torch.arange(n, device=device)
    live = torch.zeros(n, dtype=torch.bool, device=device)
    got_reset = torch.zeros(n, dtype=torch.bool, device=device)
    last_live_scorer = torch.zeros(n, dtype=torch.bool, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    success = torch.zeros(n, dtype=torch.bool, device=device)
    team_goal = torch.zeros(n, dtype=torch.bool, device=device)
    sign = torch.where(car == 0, 1, -1)

    for _ in range(round(seconds * 120 / config["frameskip"])):
        actions = sample_actions(agent, observation, temperature)
        observation, *_ = env.step(actions)
        observation = plain_tensor(observation)

        flags = env.events.cars.flags.view(n, 2).int()
        mine = flags[rows, car]
        running = ~finished

        live &= (mine & CONTACT) == 0
        reset = ((mine & RESET) != 0) & running
        got_reset |= reset
        live |= reset

        touched = (flags & TOUCH) != 0
        any_touch = touched.any(1)
        scorer_touch = touched[rows, car]
        last_live_scorer = torch.where(
            any_touch, scorer_touch & live, last_live_scorer
        )

        score = env.events.score_delta * sign
        new = (score > 0) & running
        team_goal |= new
        success |= new & last_live_scorer
        finished |= env.events.done

    env.close()

    return got_reset.cpu().numpy(), success.cpu().numpy(), team_goal.cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("clip_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = load_payload(args.checkpoint)
    config = payload["config"]
    found = candidates(args.clip_dir)
    print(f"{len(found)} grounded starts 1-{MAX_LEAD} s before a flip-reset goal", flush=True)

    torch.manual_seed(args.seed)
    probe = build(config, 1, 0)
    agent = build_agent(probe, config, payload)
    probe.close()

    starts = np.repeat(np.stack([c["state"] for c in found]), args.k, 0)
    scorer = np.repeat(np.array([c["scorer"] for c in found]), args.k)
    resets, wins, goals = [], [], []

    for lo in range(0, len(starts), args.batch):
        r, w, g = attempts(
            agent, config, starts[lo:lo + args.batch], scorer[lo:lo + args.batch],
            MAX_LEAD + 2.0, args.seed + lo
        )
        resets.append(r), wins.append(w), goals.append(g)
        print(f"  {min(lo + args.batch, len(starts))}/{len(starts)} attempts", flush=True)

    resets = np.concatenate(resets).reshape(len(found), args.k).mean(1)
    wins = np.concatenate(wins).reshape(len(found), args.k).mean(1)
    goals = np.concatenate(goals).reshape(len(found), args.k).mean(1)
    ranked = sorted(
        (
            {
                "clip":                  c["clip"],
                "frame":                 c["frame"],
                "scorer":                ("blue", "orange")[c["scorer"]],
                "lead_seconds":          c["lead_seconds"],
                "flip_reset_goal_rate":  float(w),
                "flip_reset_rate":       float(r),
                "scorer_team_goal_rate": float(g)
            }
            for c, r, w, g in zip(found, resets, wins, goals)
        ),
        key=lambda x: -x["flip_reset_goal_rate"]
    )

    args.output.write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "k": args.k, "starts": ranked}, indent=1
    ))
    print("mean flip-reset goal rate over starts:", f"{100 * wins.mean():.2f}%")

    for x in ranked[:12]:
        print(f"  clip {x['clip']:4d} frame {x['frame']:3d} {x['scorer']:6s} lead {x['lead_seconds']:.1f}s  "
              f"FR goal {100 * x['flip_reset_goal_rate']:5.1f}%  FR {100 * x['flip_reset_rate']:5.1f}%  "
              f"team goal {100 * x['scorer_team_goal_rate']:5.1f}%")


if __name__ == "__main__":
    main()
