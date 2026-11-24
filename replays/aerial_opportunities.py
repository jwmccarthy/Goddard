import argparse
from pathlib import Path

import numpy as np

from replays import layout
from replays.aerial_events import (
    BALL_VELOCITY_SCALE,
    CAR_VELOCITY_SCALE,
    EGO,
    MIN_TOUCH_HEIGHT,
    OTHER,
    POSITION_SCALE,
    aerial_touch_rows,
    flight_residual,
    quantiles,
    reset_eligibility,
    scan_files,
)
from replays.timing import GRAVITY, TICK_RATE, run_bounds

KIND_NAMES = ("completed", "attempted", "declined")
SURFACE_NAMES = ("floor", "wall", "ceiling", "ramp")
PHASE_NAMES = ("approach", "takeoff", "early_flight", "late_flight")
COMPLETED, ATTEMPTED, DECLINED = range(3)
APPROACH, TAKEOFF, EARLY, LATE = range(4)

FLOOR_UZ = 0.9
WALL_UZ = 0.3
MIN_FLIGHT_ROWS = 3
JUMP_ROW = 2
MIN_JUMP_SPEED = 150.0
ATTEMPT_DISTANCE = 300.0
DECLINE_LOOKAHEAD = 30
DECLINE_MERGE_GAP = 3
MAX_DECLINED_STATES = 15
STATE_STRIDE = 3
APPROACH_ROWS = 45
TAKEOFF_ROWS = 6
LATE_ROWS = 10

JUMP_IMPULSE = 300.0
REACH_TIMES = np.arange(0.5, 2.5 + 1e-9, 0.1)
REACH_HEIGHT = (250.0, 1900.0)
MAX_ACCELERATION = 1300.0
BOOST_PER_SECOND = 33.3
PROBE_ROWS = 15


def classify_surface(up_z: np.ndarray) -> np.ndarray:
    """Surface code from the car up vector z component."""

    return np.where(
        up_z > FLOOR_UZ,
        0,
        np.where(up_z < -FLOOR_UZ, 2, np.where(np.abs(up_z) < WALL_UZ, 1, 3))
    )


def find_flights(
    grounded: np.ndarray,
    demoed:   np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """First and last rows of airborne runs that start after a grounded row."""

    flying = ~grounded & ~demoed
    first, stops = run_bounds(flying)
    last = stops - 1
    after_ground = (first > 0) & grounded[np.maximum(first - 1, 0)]
    keep = after_ground & (last - first + 1 >= MIN_FLIGHT_ROWS)

    return first[keep], last[keep]


def assign_phase(
    rows:    np.ndarray,
    takeoff: int,
    event:   int
) -> np.ndarray:
    """Phase per state row; takeoff beats late flight, which beats early."""

    phase = np.where(rows >= event - LATE_ROWS, LATE, EARLY)
    phase = np.where(rows < takeoff + TAKEOFF_ROWS, TAKEOFF, phase)

    return np.where(rows < takeoff, APPROACH, phase)


def reachability(
    ego_pos:  np.ndarray,
    ego_vel:  np.ndarray,
    up:       np.ndarray,
    boost:    np.ndarray,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Height window, required acceleration and boost feasibility per time."""

    t = REACH_TIMES[None, :, None]
    drop = 0.5 * GRAVITY * t**2
    ball = ball_pos[:, None] + ball_vel[:, None] * t + drop
    start_vel = ego_vel + JUMP_IMPULSE * up
    gap = ball - ego_pos[:, None] - start_vel[:, None] * t - drop
    need = 2 * np.linalg.norm(gap, axis=-1) / REACH_TIMES**2
    height = (
        (ball[..., 2] >= REACH_HEIGHT[0]) & (ball[..., 2] <= REACH_HEIGHT[1])
    )
    fuel = boost[:, None] >= BOOST_PER_SECOND * REACH_TIMES

    return height, need, fuel


def first_reach_time(feasible: np.ndarray) -> np.ndarray:
    """Smallest feasible look-ahead time per row, NaN if none."""

    found = feasible.any(-1)

    return np.where(found, REACH_TIMES[feasible.argmax(-1)], np.nan)


def state_rows(first: int, event: int) -> np.ndarray:
    """Rows every STATE_STRIDE back from the event row, not before first."""

    return np.arange(event, first - 1, -STATE_STRIDE)[::-1]


def declined_runs(
    reachable: np.ndarray,
    takeoffs:  np.ndarray
) -> list[tuple[int, int]]:
    """Reachable grounded runs that no takeoff follows soon after."""

    starts, stops = run_bounds(reachable)
    runs = []

    for first, last in zip(starts, stops - 1):
        if runs and first - runs[-1][1] - 1 <= DECLINE_MERGE_GAP:
            runs[-1][1] = last
        else:
            runs.append([first, last])

    kept = []

    for first, last in runs:
        nxt = np.searchsorted(takeoffs, last + 1)
        followed = (
            nxt < len(takeoffs) and takeoffs[nxt] <= last + DECLINE_LOOKAHEAD
        )

        if not followed:
            kept.append((int(first), int(last)))

    return kept


def scan_replay(
    path:         Path,
    counterparts: list[Path],
    holdout:      bool,
    max_accel:    float = MAX_ACCELERATION
) -> dict:
    """Aerial opportunities of the ego car in one perspective file."""

    rows = np.load(path, mmap_mode="r", allow_pickle=False)
    rows = np.array(rows, dtype=np.float32)
    count = len(rows)
    index = np.arange(count)
    ego_pos = rows[:, EGO : EGO + 3] * POSITION_SCALE
    ego_vel = rows[:, EGO + 3 : EGO + 6] * CAR_VELOCITY_SCALE
    up = rows[:, EGO + 12 : EGO + 15]
    boost = rows[:, EGO + 15] * 100
    grounded = rows[:, EGO + 16] > 0.5
    demoed = rows[:, EGO + 17] > 0.5
    ball_pos = rows[:, 0:3] * POSITION_SCALE
    ball_vel = rows[:, 3:6] * BALL_VELOCITY_SCALE
    other_pos = rows[:, OTHER : OTHER + 3] * POSITION_SCALE
    ball_gap = np.linalg.norm(ego_pos - ball_pos, axis=-1)
    step = layout.PARSED_FRAMESKIP / TICK_RATE

    surface = classify_surface(up[:, 2])
    height, need, fuel = reachability(
        ego_pos, ego_vel, up, boost, ball_pos, ball_vel
    )
    accel = need <= max_accel
    feasible = height & accel & fuel
    reach_time = np.where(grounded, first_reach_time(feasible), np.nan)
    reachable = grounded & ~np.isnan(reach_time)

    first, last = find_flights(grounded, demoed)
    touches = aerial_touch_rows(rows, ego_pos, grounded, demoed)
    owner = np.searchsorted(first, touches, side="right") - 1
    inside = (owner >= 0) & (touches <= last[np.maximum(owner, 0)])
    first_touch = {}

    for flight, touch in zip(owner[inside], touches[inside]):
        first_touch.setdefault(int(flight), int(touch))

    residual = flight_residual(ego_pos, ego_vel, step)
    last_air = np.maximum.accumulate(np.where(~grounded, index, -1))

    # Episodes as (kind, takeoff, event, run start, state rows, surface, jump).
    episodes = []

    for flight, (start, end) in enumerate(zip(first, last)):
        ground = start - 1
        jump_row = ground + JUMP_ROW
        jumped = jump_row < count and (
            ego_vel[jump_row] @ up[ground] > MIN_JUMP_SPEED
        )
        run_start = last_air[ground] + 1

        if flight in first_touch:
            kind, event = COMPLETED, first_touch[flight]
        elif jumped:
            air = np.arange(start, end + 1)
            air = air[ego_pos[air, 2] > MIN_TOUCH_HEIGHT]

            if not len(air) or ball_gap[air].min() >= ATTEMPT_DISTANCE:
                continue

            kind, event = ATTEMPTED, int(air[ball_gap[air].argmin()])
        else:
            continue

        episodes.append({
            "kind":     kind,
            "takeoff":  int(start),
            "event":    event,
            "rows":     state_rows(max(run_start, start - APPROACH_ROWS), event),
            "surface":  int(surface[ground]),
            "jump":     bool(jumped),
            "residual": residual[start + 1 : event + 1].max(initial=0.0)
        })

    for begin, event in declined_runs(reachable, first):
        candidates = state_rows(begin, event)
        candidates = candidates[reachable[candidates]]

        if len(candidates) > MAX_DECLINED_STATES:
            pick = np.linspace(0, len(candidates) - 1, MAX_DECLINED_STATES)
            candidates = candidates[np.unique(pick.round().astype(int))]

        episodes.append({
            "kind":     DECLINED,
            "takeoff":  -1,
            "event":    event,
            "rows":     candidates,
            "surface":  int(surface[event]),
            "jump":     False,
            "residual": np.nan
        })

    result = {
        "rows":      count,
        "frameskip": layout.PARSED_FRAMESKIP,
        "grounded":  int(grounded.sum()),
        "reachable": int(reachable.sum()),
        "episodes":  len(episodes)
    }

    probes = np.array([
        episode["takeoff"] - PROBE_ROWS for episode in episodes
        if episode["kind"] == COMPLETED
    ], dtype=int)
    probes = probes[(probes >= 0)]
    probes = probes[grounded[probes]]
    missed = probes[~reachable[probes]]
    nearly = height[missed] & fuel[missed]
    result["probe"] = np.array([
        len(probes),
        len(probes) - len(missed),
        (~height[missed]).sum(),
        (~accel[missed]).sum(),
        (~fuel[missed]).sum(),
        missed.size * len(REACH_TIMES)
    ])
    result["probe_accel"] = (
        np.where(nearly, need[missed], np.inf).min(-1)
        if len(missed) else np.zeros(0)
    )

    if not episodes:
        result["arrays"] = None

        return result

    quality = np.load(path.with_suffix(".quality.npz"))
    any_unsafe = quality["unsafe"].any(axis=-1)
    eligible = reset_eligibility(path, rows, counterparts, any_unsafe)
    sizes = np.array([len(episode["rows"]) for episode in episodes])
    state = np.concatenate([episode["rows"] for episode in episodes])

    def each(key: str) -> np.ndarray:
        return np.repeat([episode[key] for episode in episodes], sizes)

    event = each("event")
    kind = each("kind")
    phase = np.concatenate([
        np.zeros(len(episode["rows"]), dtype=int) if episode["kind"] == DECLINED
        else assign_phase(episode["rows"], episode["takeoff"], episode["event"])
        for episode in episodes
    ])
    surface_state = np.where(kind == DECLINED, surface[state], each("surface"))

    result["arrays"] = {
        "row":                       state,
        "local_episode":             np.repeat(np.arange(len(episodes)), sizes),
        "kind":                      kind.astype(np.int8),
        "surface":                   surface_state.astype(np.int8),
        "phase":                     phase.astype(np.int8),
        "time_to_event_s":           (event - state) * step,
        "event_height_uu":           ego_pos[event, 2],
        "closest_distance_uu":       ball_gap[event],
        "jump_takeoff":              each("jump").astype(bool),
        "reachable":                 reachable[state],
        "reach_time_s":              reach_time[state],
        "boost":                     boost[state],
        "ego_z_uu":                  ego_pos[state, 2],
        "ball_z_uu":                 ball_pos[state, 2],
        "opponent_ball_distance_uu": np.linalg.norm(
            other_pos[state] - ball_pos[state], axis=-1
        ),
        "holdout":                np.full(len(state), holdout),
        "reset_ok":               eligible[state],
        "flight_max_residual_uu": each("residual").astype(float)
    }

    return result


def summarise(
    results:   list[dict],
    states:    dict[str, np.ndarray],
    max_accel: float = MAX_ACCELERATION
) -> str:
    """Console summary of the mined opportunities."""

    lines = []
    rows = sum(result["rows"] for result in results)
    ticks = sum(result["rows"] * result["frameskip"] for result in results)
    hours = ticks / TICK_RATE / 3600
    lines.append(f"files scanned: {len(results)}")
    lines.append(
        f"rows: {rows}  ego hours: {hours:.1f}  max accel: {max_accel:.0f}"
    )

    kind, surface, phase = states["kind"], states["surface"], states["phase"]
    _, firsts = np.unique(states["episode_id"], return_index=True)
    lines.append("episodes by kind x surface (takeoff surface; declined: current):")

    for code, name in enumerate(KIND_NAMES):
        pick = kind[firsts] == code
        lines.append(f"  {name}: {pick.sum()}  " + "  ".join(
            f"{surf}: {(pick & (surface[firsts] == i)).sum()}"
            for i, surf in enumerate(SURFACE_NAMES)
        ))

    usable = states["reset_ok"] & ~states["holdout"]
    lines.append(
        f"states: {len(kind)}  usable (reset_ok & not holdout): {usable.sum()}"
    )
    lines.append("states by kind x phase (all / usable):")

    for code, name in enumerate(KIND_NAMES):
        lines.append(f"  {name}: " + "  ".join(
            f"{ph}: {((kind == code) & (phase == j)).sum()}"
            f"/{((kind == code) & (phase == j) & usable).sum()}"
            for j, ph in enumerate(PHASE_NAMES)
        ))

    lines.append("usable states by kind x surface x phase "
                 + "(" + " / ".join(PHASE_NAMES) + "):")

    for code, name in enumerate(KIND_NAMES):
        for i, surf in enumerate(SURFACE_NAMES):
            pick = (kind == code) & (surface == i) & usable
            counts = [int((pick & (phase == j)).sum()) for j in range(4)]

            if any(counts):
                lines.append(f"  {name} {surf}: " + " / ".join(map(str, counts)))

    lines.append("time_to_event_s p10/p50/p90 per phase:")

    for j, ph in enumerate(PHASE_NAMES):
        lines.append(
            f"  {ph}: {quantiles(states['time_to_event_s'][phase == j])}"
        )

    attempted = firsts[kind[firsts] == ATTEMPTED]
    lines.append(
        "attempted closest_distance_uu p10/p50/p90: "
        + quantiles(states["closest_distance_uu"][attempted])
    )
    completed = firsts[kind[firsts] == COMPLETED]
    lines.append(
        "completed jump_takeoff share: "
        f"{states['jump_takeoff'][completed].mean():.3f}"
    )
    lines.append(
        "completed closest_distance_uu p10/p50/p90: "
        + quantiles(states["closest_distance_uu"][completed])
    )

    probe = sum(result["probe"] for result in results)
    total, hit, no_height, no_accel, no_fuel, pairs = probe
    lines.append(
        f"reachability validation: {hit}/{total} completed episodes reachable "
        f"at takeoff-{PROBE_ROWS} rows ({hit / max(total, 1):.3f})"
    )
    lines.append(
        "  missed (episode, time) pairs failing: "
        f"height {no_height / max(pairs, 1):.3f}  "
        f"accel {no_accel / max(pairs, 1):.3f}  "
        f"boost {no_fuel / max(pairs, 1):.3f}"
    )
    need = np.concatenate([result["probe_accel"] for result in results])
    lines.append(
        "  missed: min required accel (height and boost ok) p10/p50/p90: "
        + quantiles(np.where(np.isfinite(need), need, np.nan))
        + f"  none height+boost ok: {(~np.isfinite(need)).sum()}"
    )
    grounded = sum(result["grounded"] for result in results)
    reachable = sum(result["reachable"] for result in results)
    lines.append(
        f"grounded ego rows reachable: {reachable}/{grounded} "
        f"({reachable / max(grounded, 1):.4f})"
    )
    lines.append(
        "reachable share of states: completed approach "
        f"{states['reachable'][(kind == 0) & (phase == 0)].mean():.3f}  "
        "attempted approach "
        f"{states['reachable'][(kind == 1) & (phase == 0)].mean():.3f}"
    )

    return "\n".join(lines)


def mine(
    directory: str | Path,
    limit:     int | None = None,
    workers:   int = 1,
    max_accel: float = MAX_ACCELERATION
) -> tuple[list[Path], list[dict], dict[str, np.ndarray]]:
    paths, results = scan_files(
        scan_replay, directory, limit, workers, max_accel
    )
    shown = [result for result in results if result["arrays"] is not None]
    keys = list(shown[0]["arrays"])
    states = {
        key: np.concatenate([result["arrays"][key] for result in shown])
        for key in keys
    }
    offsets = np.cumsum([0] + [result["episodes"] for result in results])
    states["episode_id"] = np.concatenate([
        result["arrays"]["local_episode"] + offsets[index]
        for index, result in enumerate(results) if result["arrays"] is not None
    ]).astype(np.int64)
    del states["local_episode"]
    states["file_index"] = np.concatenate([
        np.full(len(result["arrays"]["row"]), index, dtype=np.int32)
        for index, result in enumerate(results) if result["arrays"] is not None
    ])

    return paths, results, states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="replays/duels/parsed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-accel", type=float, default=MAX_ACCELERATION)
    args = parser.parse_args()

    paths, results, states = mine(
        args.directory, args.limit, args.workers, args.max_accel
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        paths=np.array([str(path) for path in paths]),
        kind_names=np.array(KIND_NAMES),
        surface_names=np.array(SURFACE_NAMES),
        phase_names=np.array(PHASE_NAMES),
        **states
    )
    text = summarise(results, states, args.max_accel)
    print(text)
    output.with_name(output.name + ".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
