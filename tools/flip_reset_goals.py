"""Find flip-reset goals in self-play: uv run python tools/flip_reset_goals.py runs/<run>/latest.pt --output DIR"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    CONTACT,
    DODGE,
    DOUBLE_JUMP,
    JUMP,
    RESET,
    TOUCH,
    frame,
    load_sampler,
    read_clip,
    write_summary,
)

from replays.layout import (  # noqa: E402
    BALL_VELOCITY_SCALE,
    CAR_VELOCITY_SCALE,
    POSITION_SCALE,
    team_observation_size,
    team_replay_row_size,
)
from training.tensors import plain_tensor  # noqa: E402

# Parsed replay normalization (replays/parse.py).
NORM_POS = np.array(POSITION_SCALE, dtype=np.float32)
BALL_SCALE = np.r_[NORM_POS, [BALL_VELOCITY_SCALE] * 3, [6.0] * 3].astype(np.float32)
CAR_SCALE = np.r_[NORM_POS, [CAR_VELOCITY_SCALE] * 3, [5.5] * 3, [1.0] * 12].astype(np.float32)
JUMP_HEAD = 6


def write_labeler_clip(
    directory: Path,
    index:     int,
    frames:    np.ndarray,
    actions:   np.ndarray,
    goal:      dict,
    mechanic:  str = "flipreset"
) -> list[str]:
    """Write blue- and orange-POV parsed rows: <player>-<clip>-<mechanic>.npy."""
    frames = frames.astype(np.float32)
    ball, cars = frames[:, :9], frames[:, 9:].reshape(len(frames), 2, 21).copy()

    if cars[..., 15].max() > 1.5:  # CARL boost in 0..100
        cars[..., 15] /= 100.0

    jump = actions[..., JUMP_HEAD] == 1
    names = []

    for pov in (0, 1):
        b, c = ball.copy(), cars[:, [pov, 1 - pov]].copy()

        if pov == 1:  # orange POV: rotate the field 180 degrees about z
            for field in (0, 3, 6):
                b[:, field:field + 2] *= -1

            for field in (0, 3, 6, 9, 12):
                c[..., field:field + 2] *= -1

        rows = np.zeros(
            (len(frames), team_replay_row_size(1)), dtype=np.float32
        )
        rows[:, :9] = b / BALL_SCALE
        rows[:, 9:51] = (c / CAR_SCALE).reshape(len(frames), 42)
        rows[:, team_observation_size(1) + 5] = jump[:, pov]

        stem = f"{'blue' if pov == 0 else 'orange'}-{index:03d}-{mechanic}"
        outcome = (
            "scored" if goal["scorer_team"] == ("blue", "orange")[pov]
            else "conceded"
        )
        meta = {
            "frameskip":     4,
            "team_size":     1,
            "ego_team":      pov,
            "goal_terminal": True,
            "goal_outcome":  outcome,
            "source":        "policy self-play (sampled actions)",
            **goal
        }

        np.save(directory / f"{stem}.npy", rows)
        (directory / f"{stem}.json").write_text(json.dumps(meta, indent=1))
        names.append(stem)

    return names


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-sim", type=int, default=8192)
    parser.add_argument("--minutes", type=float, default=20.0, help="game minutes per arena")
    parser.add_argument("--clip-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labeler-dir", type=Path,
                        help="also write clips as parsed replays for the skill labeler")
    args = parser.parse_args()

    payload, env, agent = load_sampler(args.checkpoint, args.n_sim, args.seed)
    config = payload["config"]
    device = env.device

    n, cars, fs = args.n_sim, env.n_cars, config["frameskip"]
    team = torch.arange(cars, device=device) >= cars // 2           # False blue, True orange
    length = round(args.clip_seconds * 120 / fs)
    observation = plain_tensor(env.reset())
    sample = frame(env)
    ring = torch.zeros(length, n, sample.shape[-1], device=device)
    ring_actions = torch.zeros(length, n, cars, 7, dtype=torch.int8, device=device)

    live = torch.full((n, cars), -1, dtype=torch.long, device=device)   # decision of live reset
    used = torch.zeros(n, cars, dtype=torch.bool, device=device)
    last_car = torch.full((n,), -1, dtype=torch.long, device=device)
    last_reset = torch.full((n,), -1, dtype=torch.long, device=device)  # reset decision at last touch
    last_used = torch.zeros(n, dtype=torch.bool, device=device)
    last_step = torch.zeros(n, dtype=torch.long, device=device)
    filled = torch.zeros(n, dtype=torch.long, device=device)
    arena = torch.arange(n, device=device)

    totals = {"flip_resets": 0, "goals": 0}
    goals_found, clips = [], []
    decisions = round(args.minutes * 60 * 120 / fs)
    started = time.perf_counter()

    for step in range(decisions):
        slot = step % length
        ring[slot] = frame(env)
        actions = agent.get_action(observation, sample=True)
        ring_actions[slot] = actions.view(n, cars, -1).to(torch.int8)
        filled += 1

        observation, *_ = env.step(actions)
        observation = plain_tensor(observation)
        events = env.events
        flags = events.cars.flags.view(n, cars).int()

        # Landing on the arena ends a reset; a new reset this step re-arms.
        landed = (flags & CONTACT) != 0
        live[landed] = -1
        used[landed] = False
        reset = (flags & RESET) != 0
        totals["flip_resets"] += int(reset.sum())
        used |= (live >= 0) & ((flags & (JUMP | DOUBLE_JUMP | DODGE)) != 0)
        live[reset] = step
        used[reset] = False

        touched = (flags & TOUCH) != 0
        toucher = touched.any(1)
        # If both touch in one step, credit the car whose reset is live, else car 0.
        pick = torch.where((touched & (live >= 0)).any(1),
                           (touched & (live >= 0)).float().argmax(1), touched.float().argmax(1))
        last_car = torch.where(toucher, pick, last_car)
        chosen_live = live.gather(1, pick[:, None])[:, 0]
        chosen_used = used.gather(1, pick[:, None])[:, 0]
        last_reset = torch.where(toucher, chosen_live, last_reset)
        last_used = torch.where(toucher, chosen_used, last_used)
        last_step = torch.where(toucher, torch.full_like(last_step, step), last_step)

        score = events.score_delta
        scored = score != 0
        if scored.any():
            totals["goals"] += int(scored.sum())
            scoring_orange = score < 0
            valid = scored & (last_car >= 0)
            by_team = valid & (team[last_car.clamp_min(0)] == scoring_orange)
            hits = by_team & (last_reset >= 0)
            for a in arena[hits].tolist():
                car = int(last_car[a])
                goals_found.append({
                    "clip":                   len(clips),
                    "arena":                  a,
                    "decision":               step,
                    "game_seconds":           step * fs / 120,
                    "scorer_car":             car,
                    "scorer_team":            "orange" if car >= cars // 2 else "blue",
                    "used_reset":             bool(last_used[a]),
                    "seconds_reset_to_touch": (int(last_step[a]) - int(last_reset[a])) * fs / 120,
                    "seconds_touch_to_goal":  (step - int(last_step[a])) * fs / 120
                })
                clips.append(
                    read_clip(ring, ring_actions, a, slot, filled, length)
                )

        done = events.done
        if done.any():
            live[done] = -1
            used[done] = False
            last_car[done] = -1
            last_reset[done] = -1
            last_used[done] = False
            filled[done] = 0

        if step % 3000 == 0:
            print(f"decision {step}/{decisions}  {time.perf_counter() - started:.0f}s  "
                  f"resets {totals['flip_resets']}  goals {totals['goals']}  "
                  f"flip-reset goals {len(goals_found)}", flush=True)

    env.close()

    hours = n * cars * decisions * fs / 120 / 3600
    args.output.mkdir(parents=True, exist_ok=True)

    if clips:
        np.savez_compressed(
            args.output / "clips.npz",
            **{f"frames_{i}": c[0] for i, c in enumerate(clips)},
            **{f"actions_{i}": c[1] for i, c in enumerate(clips)}
        )

    summary = {
        "checkpoint":               str(args.checkpoint.resolve()),
        "step":                     int(payload["step"]),
        "sampled":                  True,
        "n_sim":                    n,
        "game_minutes_per_arena":   args.minutes,
        "car_hours":                hours,
        "frameskip":                fs,
        "frame_layout":             "raw ball pos vel angvel; per car the 21 parsed-replay car fields",
        "flip_resets":              totals["flip_resets"],
        "flip_resets_per_car_hour": totals["flip_resets"] / hours,
        "total_goals":              totals["goals"],
        "flip_reset_goals":         len(goals_found),
        "flip_reset_goals_used":    sum(g["used_reset"] for g in goals_found),
        "goals":                    goals_found
    }

    if args.labeler_dir is not None and clips:
        if config["frameskip"] != 4:
            raise ValueError("labeler clips need one frame per 4 ticks (frameskip 4)")

        args.labeler_dir.mkdir(parents=True, exist_ok=True)

        for goal, (frames, actions) in zip(goals_found, clips):
            goal["labeler_files"] = write_labeler_clip(
                args.labeler_dir, goal["clip"], frames, actions, goal
            )

    write_summary(args.output / "summary.json", summary, "goals")


if __name__ == "__main__":
    main()
