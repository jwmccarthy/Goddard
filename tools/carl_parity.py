"""Compare a CARL clip with Rocket League replay: tools/carl_parity.py carl|compare PARITY_DIR"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import DODGE, RESET, build, frame, reset_from  # noqa: E402

CHECKPOINTS = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)
TICKS = 4
TAIL_DECISIONS = 45  # 1.5 s of neutral controls after the clip, as in the RLBot test
CONFIG = {
    "team_size":         1,
    "frameskip":         TICKS,
    "max_ticks":         10**9,
    "no_touch_timeout":  1000,
    "previous_controls": False
}


def simulate(first: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, dict]:
    """State-set one CARL arena to `first` and replay controls."""
    env = build(CONFIG, 1, 0)
    device = env.device
    reset_from(env, torch.tensor(first[None], device=device, dtype=torch.float32))

    neutral = np.zeros((TAIL_DECISIONS, 2, 7), dtype=actions.dtype)
    out, events = [], {"orange_flip_reset": None, "orange_dodge": None, "goal": None}

    for k, action in enumerate(np.concatenate([actions, neutral])):
        out.append(frame(env)[0].cpu().numpy())
        env.step(torch.tensor(action, device=device, dtype=torch.long))

        flags = int(env.events.cars.flags.view(-1)[1])
        seconds = (k + 1) * TICKS / 120

        if events["orange_flip_reset"] is None and flags & RESET:
            events["orange_flip_reset"] = seconds

        if events["orange_dodge"] is None and flags & DODGE:
            events["orange_dodge"] = seconds

        if events["goal"] is None and int(env.events.score_delta[0]) != 0:
            score = int(env.events.score_delta[0])
            events["goal"] = f"{seconds:.2f} s ({'blue' if score > 0 else 'orange'} scored)"
            break

    env.close()

    return np.array(out, dtype=np.float32), events


def show_events(events: dict) -> None:
    def fmt(value):
        return "none" if value is None else (value if isinstance(value, str) else f"{value:.2f} s")
    print(f"  orange flip reset {fmt(events['orange_flip_reset'])}, orange dodge "
          f"{fmt(events['orange_dodge'])}, goal {fmt(events['goal'])}")


def carl_baseline(directory: Path, perturbations: int, nudge: float) -> None:
    clip = np.load(directory / "clip.npz")
    frames, actions = clip["frames"], clip["actions"]
    print("Original CARL clip: orange dodges at 3.00 s; orange scores just after "
          f"{len(frames) * TICKS / 120:.2f} s.")
    rerun, events = simulate(frames[0], actions)
    np.save(directory / "carl-rerun.npy", rerun)
    print("\nCARL rerun from the state-set start (default hidden state) vs the original clip:")
    report(rerun, frames)
    show_events(events)
    rng = np.random.default_rng(0)
    for i in range(perturbations):
        first = frames[0].astype(np.float64).copy()
        direction = rng.normal(size=3)
        first[0:3] += nudge * direction / np.linalg.norm(direction)   # nudge the ball only
        nudged, events = simulate(first.astype(np.float32), actions)
        print(f"\nChaos check {i + 1}: ball start moved {nudge} uu, vs the unnudged rerun:")
        report(nudged, rerun)
        show_events(events)


def game_frames(path: Path) -> tuple[np.ndarray, dict]:
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ticks = [x for x in lines[1:] if "tick" in x]
    end = next((x["end"] for x in lines if "end" in x), "unknown")
    out, events = [], {"end": end, "orange_dodge_tick": None, "score_change_tick": None}
    first_score = ticks[0]["score"] if ticks else None
    for x in ticks:
        cars = sorted(x["cars"], key=lambda c: c["team"])
        if events["orange_dodge_tick"] is None and cars[1]["has_dodged"]:
            events["orange_dodge_tick"] = x["tick"]
        if events["score_change_tick"] is None and x["score"] != first_score:
            events["score_change_tick"] = x["tick"]
        if x["tick"] % TICKS:
            continue
        row = [*x["ball"]["location"], *x["ball"]["velocity"], *x["ball"]["angular_velocity"]]
        for c in cars:
            p, y, r = c["rotation"]
            cp, sp, cy, sy, cr, sr = np.cos(p), np.sin(p), np.cos(y), np.sin(y), np.cos(r), np.sin(r)
            fwd = [cp * cy, cp * sy, sp]
            up = [-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr]
            row += [*c["location"], *c["velocity"], *c["angular_velocity"], *fwd, *up,
                    c["boost"], c["air_state"] == 0, c["demolished"], c["has_dodged"],
                    c["has_double_jumped"], 0.0]
        out.append(row)
    return np.array(out, dtype=np.float64), events


def report(actual: np.ndarray, reference: np.ndarray) -> None:
    n = min(len(actual), len(reference))
    a, r = actual[:n], reference[:n]
    errors = {
        "ball":   np.linalg.norm(a[:, 0:3] - r[:, 0:3], axis=1),
        "blue":   np.linalg.norm(a[:, 9:12] - r[:, 9:12], axis=1),
        "orange": np.linalg.norm(a[:, 30:33] - r[:, 30:33], axis=1)
    }
    print("  time    " + "  ".join(f"{k:>8s}" for k in errors) + "   (position error, uu)")
    for t in CHECKPOINTS:
        i = round(t * 120 / TICKS)
        if i < n:
            print(f"  {t:4.2f}s  " + "  ".join(f"{e[i]:8.1f}" for e in errors.values()))
    for k, e in errors.items():
        over = {lim: next((i * TICKS / 120 for i, v in enumerate(e) if v > lim), None) for lim in (10, 50, 200)}
        print(f"  {k:6s} first exceeds 10/50/200 uu at: "
              + ", ".join("never" if v is None else f"{v:.2f}s" for v in over.values()))


def compare(directory: Path) -> None:
    """Rocket League vs the CARL rerun from the same state-set start."""
    clip = np.load(directory / "clip.npz")
    frames = clip["frames"].astype(np.float64)
    rerun_path = directory / "carl-rerun.npy"
    if not rerun_path.exists():
        raise FileNotFoundError("run `carl_parity.py carl DIR` first for the CARL rerun")
    rerun = np.load(rerun_path).astype(np.float64)
    runs = sorted(directory.glob("parity-run-*.jsonl"))
    if not runs:
        raise FileNotFoundError("no parity-run-*.jsonl: run the RLBot parity match first")
    games = []
    for path in runs:
        game, events = game_frames(path)
        games.append(game)
        print(f"\n{path.name}: Rocket League vs CARL rerun ({len(game)} frames, run {events['end']})")
        report(game, rerun)
        dodge = events["orange_dodge_tick"]
        goal = events["score_change_tick"]
        print(f"  orange dodge {'none' if dodge is None else f'{dodge / 120:.2f} s'}, "
              f"score change {'none' if goal is None else f'{goal / 120:.2f} s'} "
              "(CARL rerun: no reset or dodge, orange goal at about 6.2 s)")
    if len(games) > 1:
        print("\nRocket League run-to-run variation (vs run 1):")
        for i, game in enumerate(games[1:], 2):
            print(f" run {i}:")
            report(game, games[0])
    print("\nFor reference, CARL rerun vs original clip (lost hidden state + chaos):")
    report(rerun, frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("carl", "compare"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--perturbations", type=int, default=3)
    parser.add_argument("--nudge", type=float, default=0.1, help="uu, chaos check")
    args = parser.parse_args()
    if args.mode == "carl":
        carl_baseline(args.directory, args.perturbations, args.nudge)
    else:
        compare(args.directory)


if __name__ == "__main__":
    main()
