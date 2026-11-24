import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import carl
import numpy as np
import torch
from carl.gymnasium import CARLTorchVectorEnv

from replays import layout
from replays.aerial_events import (
    EGO,
    EGO_TOUCH,
    OTHER,
    OTHER_TOUCH,
    POSITION_SCALE,
    quantiles,
)
from replays.aerial_opportunities import ATTEMPTED, COMPLETED
from replays.dataset import replay_paths
from training.replay_resets import (
    ReplayResetProvider,
    replay_eligibility,
    replay_internal_states,
    replay_key,
)

VERSION = "aerial-bank-v1"
TEAM_SIZE = 1
CHECK_ROWS = 15
CONTACT_DISTANCE = 300.0
BALL_ERROR_LIMIT = 25.0
TICKS_PER_ROW = layout.PARSED_FRAMESKIP
TAGS = (
    "kind", "surface", "phase", "time_to_event_s", "event_height_uu",
    "episode_id", "file_index", "row"
)


def select_states(states: dict[str, np.ndarray]) -> np.ndarray:
    """Indices of completed and attempted, reset-eligible, non-holdout states."""

    kind = states["kind"]
    usable = states["reset_ok"] & ~states["holdout"]

    return np.flatnonzero(((kind == COMPLETED) | (kind == ATTEMPTED)) & usable)


def prepare_file(task: tuple) -> dict:
    """Eligible scene, internal state and replay reference of one file."""

    path, counterparts, ids, picked = task
    rows = np.load(path, mmap_mode="r", allow_pickle=False)
    scene = np.array(
        rows[:, : layout.team_scene_size(TEAM_SIZE)], dtype=np.float32
    )
    internal, _ = replay_internal_states(rows, scene, TEAM_SIZE, counterparts)
    n_cars = layout.team_car_count(TEAM_SIZE)
    eligible = replay_eligibility(path, rows, scene, internal, n_cars)
    keep = eligible[picked]
    ids, picked = ids[keep], picked[keep]

    count = len(rows)
    ball = np.asarray(rows[:, 0:3], dtype=np.float64) * POSITION_SCALE
    distance = np.stack([
        np.linalg.norm(
            rows[:, start : start + 3] * POSITION_SCALE - ball, axis=-1
        )
        for start in (EGO, OTHER)
    ])
    stop = (
        (distance < CONTACT_DISTANCE).any(0)
        | (rows[:, EGO_TOUCH] > 0.5)
        | (rows[:, OTHER_TOUCH] > 0.5)
    )
    first = np.where(stop, np.arange(count), count)
    first = np.minimum.accumulate(first[::-1])[::-1]
    steps = np.minimum(first[picked] - picked - 1, count - 1 - picked)
    steps = np.clip(steps, 0, CHECK_ROWS)
    window = np.minimum(picked[:, None] + np.arange(CHECK_ROWS + 1), count - 1)

    return {
        "ids":       ids,
        "dropped":   int((~keep).sum()),
        "frames":    scene[picked],
        "internal":  internal[picked],
        "reference": ball[window].astype(np.float32),
        "steps":     steps.astype(np.int8),
        "grounded":  np.stack([
            rows[picked, EGO + 16] > 0.5, rows[picked, OTHER + 16] > 0.5
        ], axis=-1)
    }


def simulate(
    frames:    np.ndarray,
    internal:  np.ndarray,
    reference: np.ndarray,
    steps:     np.ndarray,
    grounded:  np.ndarray,
    sims:      int,
    device:    torch.device | str
) -> tuple[np.ndarray, np.ndarray]:
    """Ball error and on-ground mismatches from neutral-action rollouts."""

    seen = []

    def capture(context):
        seen.append((
            context.current.ball_position.clone(),
            context.current.car_on_ground.clone(),
            context.events.done.clone()
        ))

        return torch.zeros_like(context.current.car_position[..., 0])

    n_cars = layout.team_car_count(TEAM_SIZE)
    env = CARLTorchVectorEnv(
        n_sim=sims,
        n_blue=TEAM_SIZE,
        n_orange=TEAM_SIZE,
        seed=0,
        frameskip=1,
        max_ticks=36_000,
        normalize=True,
        reward_funcs=[capture],
        masked_resets=True
    )
    error = np.zeros(len(frames))
    mismatch = np.zeros((len(frames), n_cars), dtype=bool)
    actions = torch.zeros(
        sims * n_cars, len(carl.ACTION_NVECS), device=env.device
    )

    try:
        for begin in range(0, len(frames), sims):
            index = np.arange(begin, begin + sims).clip(max=len(frames) - 1)
            count = min(sims, len(frames) - begin)
            batch = SimpleNamespace(
                frames=torch.as_tensor(frames[index], device=env.device),
                internal_states=torch.as_tensor(
                    internal[index], device=env.device
                ),
                n_cars=n_cars
            )
            provider = ReplayResetProvider(batch)
            ready = torch.ones(sims, dtype=torch.bool, device=env.device)
            order = torch.arange(sims, device=env.device)
            pending = [ready]

            def reset(mask):
                chosen = pending.pop() if pending else torch.zeros_like(mask)

                return provider.reset_state(
                    order,
                    simulation_indices=None,
                    reset_mask=chosen
                )

            env.reset_state_provider = reset
            seen.clear()
            env.reset()
            reference_batch = torch.as_tensor(
                reference[index], device=env.device
            )
            check = torch.as_tensor(steps[index], device=env.device).long()
            error_batch = (
                env._state.ball_position - reference_batch[:, 0]
            ).norm(dim=-1)
            alive = torch.ones(sims, dtype=torch.bool, device=env.device)

            last = max(int(check[:count].max()) * TICKS_PER_ROW, 1)

            for tick in range(1, last + 1):
                env.step(actions)
                ball, on_ground, done = seen[-1]

                if tick == 1:
                    expected = torch.as_tensor(grounded[index], device=env.device)
                    mismatched = (on_ground != expected).cpu().numpy()
                    mismatch[begin : begin + count] = mismatched[:count]

                alive &= ~done

                if tick % TICKS_PER_ROW:
                    continue

                row = tick // TICKS_PER_ROW
                gap = (ball - reference_batch[:, row]).norm(dim=-1)
                gap = torch.where(alive & (check >= row), gap, 0.0)
                error_batch = torch.maximum(error_batch, gap)

            error[begin : begin + count] = error_batch.cpu().numpy()[:count]
    finally:
        env.close()

    return error, mismatch


def build_bank(
    opportunities: dict[str, np.ndarray],
    directory:     str | Path,
    limit:         int | None = None,
    workers:       int = 1,
    sims:          int = 2048,
    threshold:     float = BALL_ERROR_LIMIT,
    device:        torch.device | str = "cuda"
) -> tuple[dict, dict]:
    """Bank payload and the per-candidate check results behind its summary."""

    _, listing = replay_paths(directory, TEAM_SIZE)
    paths = [str(path) for path in opportunities["paths"]]
    known = {str(path): path for path in listing}
    missing = [path for path in paths if path not in known]

    if missing:
        raise ValueError(
            f"{len(missing)} opportunity files not under {directory}"
        )

    groups = {}

    for path in listing:
        groups.setdefault(replay_key(path), []).append(path)

    selected = select_states(opportunities)[:limit]
    candidates = len(selected)
    files = opportunities["file_index"][selected]
    order = np.argsort(files, kind="stable")
    selected, files = selected[order], files[order]
    bounds = np.flatnonzero(np.diff(files)) + 1
    tasks = []

    for ids in np.split(selected, bounds):
        if not len(ids):
            continue

        path = known[paths[opportunities["file_index"][ids[0]]]]
        counterparts = [
            other for other in groups[replay_key(path)] if other != path
        ]
        tasks.append((path, counterparts, ids, opportunities["row"][ids]))

    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(prepare_file, tasks, chunksize=16))
    else:
        results = [prepare_file(task) for task in tasks]

    dropped = sum(result["dropped"] for result in results)

    def join(key: str) -> np.ndarray:
        return np.concatenate([result[key] for result in results])

    ids = join("ids")
    frames, internal = join("frames"), join("internal")
    steps, grounded = join("steps"), join("grounded")
    error, mismatch = simulate(
        frames, internal, join("reference"), steps, grounded, sims, device
    )

    passed = (error <= threshold) & ~mismatch.any(-1)
    tags = {key: opportunities[key][ids] for key in TAGS}

    payload = {
        "version":         VERSION,
        "team_size":       TEAM_SIZE,
        "frames":          torch.as_tensor(frames[passed]),
        "internal_states": torch.as_tensor(internal[passed]),
        "kind":            torch.as_tensor(tags["kind"][passed]).to(torch.int8),
        "surface":         torch.as_tensor(tags["surface"][passed]).to(torch.int8),
        "phase":           torch.as_tensor(tags["phase"][passed]).to(torch.int8),
        "time_to_event_s": torch.as_tensor(tags["time_to_event_s"][passed]).float(),
        "event_height_uu": torch.as_tensor(tags["event_height_uu"][passed]).float(),
        "episode_id":      torch.as_tensor(tags["episode_id"][passed]).long(),
        "file_index":      torch.as_tensor(tags["file_index"][passed]).long(),
        "row":             torch.as_tensor(tags["row"][passed]).long(),
        "ball_error_uu":   torch.as_tensor(error[passed]).float(),
        "kind_names":      [str(name) for name in opportunities["kind_names"]],
        "surface_names":   [str(name) for name in opportunities["surface_names"]],
        "phase_names":     [str(name) for name in opportunities["phase_names"]],
        "paths":           paths
    }
    checks = {
        "selected":  candidates,
        "dropped":   dropped,
        "tags":      tags,
        "error":     error,
        "steps":     steps,
        "mismatch":  mismatch,
        "passed":    passed,
        "threshold": threshold
    }

    return payload, checks


def percentiles(values: np.ndarray) -> str:
    return quantiles(values, (50, 90, 99))


def summarise(
    payload: dict,
    checks:  dict
) -> str:
    """Text summary of kept and rejected states and the check distributions."""

    tags, passed = checks["tags"], checks["passed"]
    error, steps = checks["error"], checks["steps"]
    mismatch = checks["mismatch"].any(-1)
    late = error > checks["threshold"]
    kinds, surfaces = payload["kind_names"], payload["surface_names"]
    phases = payload["phase_names"]
    lines = [
        f"version: {payload['version']}",
        f"candidates: {checks['selected']}  "
        f"dropped by eligibility: {checks['dropped']}  "
        f"simulated: {len(passed)}  kept: {int(passed.sum())}",
        f"ball error threshold: {checks['threshold']:.1f} uu",
        f"rejected: ball error {int((late & ~mismatch).sum())}  "
        f"on-ground mismatch {int((mismatch & ~late).sum())}  "
        f"both {int((late & mismatch).sum())}",
        f"on-ground mismatch rate: {mismatch.mean():.4f}  "
        f"ego {checks['mismatch'][:, 0].mean():.4f}  "
        f"other {checks['mismatch'][:, 1].mean():.4f}",
        "ball error uu p50 / p90 / p99, all: " + percentiles(error),
        "  with replay rows to compare (check rows >= 1): "
        + percentiles(error[steps > 0])
        + f"  count {int((steps > 0).sum())}",
        "  reset round trip only (a car within contact range or a touch at "
        f"the next row): {int((steps == 0).sum())} states, "
        + percentiles(error[steps == 0]),
        "kept / rejected by kind x surface x phase:"
    ]

    for kind, kind_name in enumerate(kinds):
        for surface, surface_name in enumerate(surfaces):
            cells = []

            for phase, phase_name in enumerate(phases):
                pick = (
                    (tags["kind"] == kind) & (tags["surface"] == surface)
                    & (tags["phase"] == phase)
                )

                if pick.any():
                    cells.append(
                        f"{phase_name} {int((pick & passed).sum())}"
                        f"/{int((pick & ~passed).sum())}"
                    )

            if cells:
                lines.append(
                    f"  {kind_name} {surface_name}: " + "  ".join(cells)
                )

    for title, key, names in (
        ("phase", "phase", phases), ("surface", "surface", surfaces)
    ):
        lines.append(
            f"reject rate, ball error p50/p90/p99 and mismatch by {title}:"
        )

        for code, name in enumerate(names):
            pick = tags[key] == code

            if pick.any():
                lines.append(
                    f"  {name}: n {int(pick.sum())}  "
                    f"reject {1 - passed[pick].mean():.4f}"
                    f"  error {percentiles(error[pick])}"
                    f"  mismatch {mismatch[pick].mean():.4f}"
                )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opportunities", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--directory", default="replays/duels/parsed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sims", type=int, default=2048)
    parser.add_argument("--threshold", type=float, default=BALL_ERROR_LIMIT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with np.load(args.opportunities) as stored:
        opportunities = {key: stored[key] for key in stored.files}

    payload, checks = build_bank(
        opportunities,
        args.directory,
        args.limit,
        args.workers,
        args.sims,
        args.threshold,
        args.device
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    text = summarise(payload, checks)
    print(text)
    output.with_suffix(".txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
