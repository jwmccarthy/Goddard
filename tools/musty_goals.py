"""Find musty flick goals in self-play: uv run python tools/musty_goals.py runs/<run>/latest.pt --output DIR --labeler-dir DIR/parsed"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    DODGE,
    TOUCH,
    frame,
    load_sampler,
    read_clip,
    write_summary,
)
from flip_reset_goals import write_labeler_clip  # noqa: E402

from training.tensors import plain_tensor  # noqa: E402

NOSE_DOWN = -0.3          # forward z at the dodge
ROOF_SIDE = 40.0          # ball offset along the car's up axis (uu)
REACH = 320.0             # ball distance at the dodge (uu)
BACKWARD_RATE = 1.5       # nose-toward-roof rotation (rad/s)
STRIKE_SECONDS = 0.4
SPEED_GAIN = 300.0        # uu/s


def car_view(frames: torch.Tensor, cars: int) -> tuple[torch.Tensor, ...]:
    ball_pos, ball_vel = frames[:, 0:3], frames[:, 3:6]
    car = frames[:, 9:].view(len(frames), cars, 21)
    return ball_pos, ball_vel, car[..., 0:3], car[..., 6:9], car[..., 9:12], car[..., 12:15]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labeler-dir", type=Path)
    parser.add_argument("--n-sim", type=int, default=8192)
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--clip-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    payload, env, agent = load_sampler(args.checkpoint, args.n_sim, args.seed)
    config = payload["config"]
    device = env.device

    n, cars, fs = args.n_sim, env.n_cars, config["frameskip"]
    team = torch.arange(cars, device=device) >= cars // 2
    length = round(args.clip_seconds * 120 / fs)
    window = round(STRIKE_SECONDS * 120 / fs)
    observation = plain_tensor(env.reset())
    current = frame(env)
    ring = torch.zeros(length, n, current.shape[-1], device=device)
    ring_actions = torch.zeros(length, n, cars, 7, dtype=torch.int8, device=device)

    dodged = torch.zeros(n, cars, dtype=torch.bool, device=device)        # dodge last step
    armed_until = torch.full((n, cars), -1, dtype=torch.long, device=device)
    armed_from = torch.full((n, cars), -1, dtype=torch.long, device=device)
    last_car = torch.full((n,), -1, dtype=torch.long, device=device)
    last_musty = torch.zeros(n, dtype=torch.bool, device=device)
    last_step = torch.zeros(n, dtype=torch.long, device=device)
    last_dodge = torch.zeros(n, dtype=torch.long, device=device)
    filled = torch.zeros(n, dtype=torch.long, device=device)
    arena = torch.arange(n, device=device)

    totals = {"dodges": 0, "musty_setups": 0, "musty_touches": 0, "goals": 0}
    found, clips = [], []
    decisions = round(args.minutes * 60 * 120 / fs)
    started = time.perf_counter()

    for step in range(decisions):
        slot = step % length
        ring[slot] = current
        filled += 1

        # Judge last step's dodges from the state right after them.
        if dodged.any():
            ball_pos, _, pos, angvel, fwd, up = car_view(current, cars)
            rel = ball_pos[:, None] - pos
            nose_to_roof = (torch.cross(angvel, fwd, dim=-1) * up).sum(-1)
            setup = (dodged & (fwd[..., 2] < NOSE_DOWN) & ((rel * up).sum(-1) > ROOF_SIDE)
                     & (rel.norm(dim=-1) < REACH) & (nose_to_roof > BACKWARD_RATE))
            totals["musty_setups"] += int(setup.sum())
            armed_until = torch.where(setup, torch.full_like(armed_until, step + window), armed_until)
            armed_from = torch.where(setup, torch.full_like(armed_from, step - 1), armed_from)

        actions = agent.get_action(observation, sample=True)
        ring_actions[slot] = actions.view(n, cars, -1).to(torch.int8)
        observation, *_ = env.step(actions)
        observation = plain_tensor(observation)
        events = env.events
        flags = events.cars.flags.view(n, cars).int()
        after = frame(env)

        dodged = (flags & DODGE) != 0
        totals["dodges"] += int(dodged.sum())
        touched = (flags & TOUCH) != 0
        done = events.done
        # Goal steps reset the arena, so the post-touch speed is unknown; accept them.
        gain = (after[:, 3:6].norm(dim=-1) - current[:, 3:6].norm(dim=-1) >= SPEED_GAIN) | done
        musty = touched & (armed_until >= step) & gain[:, None]
        totals["musty_touches"] += int(musty.sum())
        armed_until[musty] = -1

        toucher = touched.any(1)
        pick = torch.where(musty.any(1), musty.float().argmax(1), touched.float().argmax(1))
        last_car = torch.where(toucher, pick, last_car)
        last_musty = torch.where(toucher, musty.any(1), last_musty)
        last_step = torch.where(toucher, torch.full_like(last_step, step), last_step)
        last_dodge = torch.where(toucher & musty.any(1),
                                 armed_from.gather(1, pick[:, None])[:, 0], last_dodge)

        score = events.score_delta
        scored = score != 0
        if scored.any():
            totals["goals"] += int(scored.sum())
            hits = (scored & (last_car >= 0) & last_musty
                    & (team[last_car.clamp_min(0)] == (score < 0)))
            for a in arena[hits].tolist():
                car = int(last_car[a])
                found.append({
                    "clip":                   len(clips),
                    "arena":                  a,
                    "decision":               step,
                    "game_seconds":           step * fs / 120,
                    "scorer_car":             car,
                    "scorer_team":            "orange" if car >= cars // 2 else "blue",
                    "seconds_dodge_to_touch": (int(last_step[a]) - int(last_dodge[a])) * fs / 120,
                    "seconds_touch_to_goal":  (step - int(last_step[a])) * fs / 120
                })
                clips.append(
                    read_clip(ring, ring_actions, a, slot, filled, length)
                )

        if done.any():
            armed_until[done] = -1
            last_car[done] = -1
            last_musty[done] = False
            filled[done] = 0
            dodged[done] = False
        current = after

        if step % 3000 == 0:
            print(f"decision {step}/{decisions}  {time.perf_counter() - started:.0f}s  "
                  f"setups {totals['musty_setups']}  musty touches {totals['musty_touches']}  "
                  f"goals {totals['goals']}  musty goals {len(found)}", flush=True)

    env.close()

    hours = n * cars * decisions * fs / 120 / 3600
    args.output.mkdir(parents=True, exist_ok=True)

    if args.labeler_dir is not None and clips:
        args.labeler_dir.mkdir(parents=True, exist_ok=True)

        for goal, (frames, actions) in zip(found, clips):
            goal["labeler_files"] = write_labeler_clip(
                args.labeler_dir, goal["clip"], frames, actions, goal, mechanic="musty"
            )

    summary = {
        "checkpoint":             str(args.checkpoint.resolve()),
        "step":                   int(payload["step"]),
        "sampled":                True,
        "n_sim":                  n,
        "game_minutes_per_arena": args.minutes,
        "car_hours":              hours,
        "frameskip":              fs,
        "criteria":               {
            "nose_down_forward_z": NOSE_DOWN,
            "roof_side_uu": ROOF_SIDE,
            "reach_uu": REACH,
            "backward_rate": BACKWARD_RATE,
            "strike_seconds": STRIKE_SECONDS,
            "speed_gain": SPEED_GAIN
        },
        **totals,
        "musty_touches_per_car_hour": totals["musty_touches"] / hours,
        "musty_goals":                len(found),
        "found":                      found
    }

    write_summary(args.output / "summary.json", summary, "found")


if __name__ == "__main__":
    main()
