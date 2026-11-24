"""Rates of a skill from one fixed start: uv run python tools/scenario_rates.py CHECKPOINT START.npz --temperatures 1 0.25"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    CONTACT,
    DODGE,
    RESET,
    TOUCH,
    build,
    reset_from,
    sample_actions,
)

from training.checkpoints import load_payload  # noqa: E402
from training.tensors import plain_tensor  # noqa: E402
from vis.playback import build_agent  # noqa: E402

ORANGE = 1


@torch.no_grad()
def run(
    payload:     dict,
    start:       np.ndarray,
    n:           int,
    seconds:     float,
    temperature: float,
    seed:        int
) -> dict:
    config = payload["config"]
    torch.manual_seed(seed)
    env = build(config, n, seed)
    device = env.device
    agent = build_agent(env, config, payload)

    first = torch.tensor(start[None], device=device, dtype=torch.float32)
    observation = reset_from(env, first.expand(n, -1).contiguous())

    live = torch.zeros(n, dtype=torch.bool, device=device)
    got_reset = torch.zeros(n, dtype=torch.bool, device=device)
    dodged_after = torch.zeros(n, dtype=torch.bool, device=device)
    last_orange_live = torch.zeros(n, dtype=torch.bool, device=device)
    last_toucher = torch.full((n,), -1, dtype=torch.long, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    # 0 none, 1 blue goal, 2 orange goal, 3 orange flip-reset goal
    outcome = torch.zeros(n, dtype=torch.long, device=device)

    for _ in range(round(seconds * 120 / config["frameskip"])):
        actions = sample_actions(agent, observation, temperature)
        observation, *_ = env.step(actions)
        observation = plain_tensor(observation)

        flags = env.events.cars.flags.view(n, 2).int()
        running = ~finished
        orange = flags[:, ORANGE]

        live &= (orange & CONTACT) == 0
        reset = ((orange & RESET) != 0) & running
        got_reset |= reset
        dodged_after |= live & ((orange & DODGE) != 0) & running
        live |= reset

        touched = (flags & TOUCH) != 0
        toucher = touched.any(1)
        pick = torch.where(
            touched[:, ORANGE],
            torch.full_like(last_toucher, ORANGE),
            touched.float().argmax(1)
        )
        last_toucher = torch.where(toucher, pick, last_toucher)
        last_orange_live = torch.where(
            toucher, live & (pick == ORANGE), last_orange_live
        )

        score = env.events.score_delta
        new = (score != 0) & running
        outcome = torch.where(new & (score > 0), torch.ones_like(outcome), outcome)
        orange_goal = new & (score < 0)
        outcome = torch.where(
            orange_goal, torch.where(last_orange_live, 3, 2), outcome
        )
        finished |= env.events.done

    env.close()

    return {
        "orange flip reset":        got_reset.float().mean().item(),
        "orange dodge after reset": dodged_after.float().mean().item(),
        "orange flip-reset goal":   (outcome == 3).float().mean().item(),
        "orange goal (any)":        ((outcome == 2) | (outcome == 3)).float().mean().item(),
        "blue goal":                (outcome == 1).float().mean().item(),
        "no goal":                  (outcome == 0).float().mean().item()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("start", type=Path, help="npz with frames[0] = raw CARL start state")
    parser.add_argument("--n-sim", type=int, default=4096)
    parser.add_argument("--seconds", type=float, default=7.0)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[1.0, 0.25])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = load_payload(args.checkpoint)
    start = np.load(args.start)["frames"][0]
    print(f"step {payload['step']}, {args.n_sim} attempts of {args.seconds:.0f} s from {args.start}")

    for t in args.temperatures:
        rates = run(payload, start, args.n_sim, args.seconds, t, args.seed)
        print(f"temperature {t}: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in rates.items()))


if __name__ == "__main__":
    main()
