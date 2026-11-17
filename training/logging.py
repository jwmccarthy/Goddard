import math
from datetime import datetime
from pathlib import Path

from rich import progress as rich_progress


def imitation_episode_magnitude(directory, recent=100):
    """Prefer exact completed-episode logs; label legacy rate estimates explicitly."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    directory = Path(directory)
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    events.Reload()
    tags = events.Tags()["scalars"]
    exact = ("imitation/episode_abs_sum_mean", "imitation/episode_count")
    legacy = (
        "imitation_reward",
        "gameplay/goals_for_per_1000_steps",
        "gameplay/goals_against_per_1000_steps",
        "gameplay/timeout_fraction",
    )
    tick_rates = (
        "imitation_reward",
        "gameplay/goals_for_per_1000_ticks",
        "gameplay/goals_against_per_1000_ticks",
        "gameplay/timeout_fraction",
        "gameplay/ticks_per_step",
    )
    per_tick = ("imitation/reward_per_tick", *tick_rates[1:])
    names = (exact if all(name in tags for name in exact) else
             per_tick if all(name in tags for name in per_tick) else
             tick_rates if all(name in tags for name in tick_rates) else legacy)

    if not all(name in tags for name in names):
        raise ValueError(f"insufficient imitation episode calibration logs in {directory}")

    values = {name: {event.step: event.value for event in events.Scalars(name)} for name in names}
    all_steps = sorted(set.intersection(*(set(samples) for samples in values.values())))
    steps = all_steps[-recent:]

    if not steps:
        raise ValueError("no aligned imitation episode calibration samples")

    numerator = denominator = 0.0

    for step in steps:
        samples = [values[name][step] for name in names]
        if not all(math.isfinite(value) for value in samples):
            continue

        if names == exact:
            mean, count = samples
            numerator += mean * count
            denominator += count
            continue

        reward, goals_for, goals_against, timeout_fraction = samples[:4]
        ticks_per_step = samples[4] if names in (tick_rates, per_tick) else 1
        if ticks_per_step <= 0:
            continue
        if names == per_tick:
            reward *= ticks_per_step  # Convert both reward and event rates to decisions.
        if not 0 <= timeout_fraction < 1 or goals_for + goals_against <= 0:
            continue

        index = all_steps.index(step)
        work = step - all_steps[index - 1] if index else step
        end_rate = (goals_for + goals_against) * ticks_per_step / (1000 * (1 - timeout_fraction))
        numerator += abs(reward) * work
        denominator += end_rate * work

    if denominator <= 0:
        raise ValueError("no usable completed episodes in calibration logs")

    weight = numerator / denominator
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("episode imitation magnitude must be finite and positive")

    return weight, {
        "directory": str(directory.resolve()),
        "method": "completed-episode-mean" if names == exact else "legacy-rate-estimate",
        "first_step": steps[0],
        "last_step": steps[-1],
        "samples": len(steps),
        "assumptions": None
        if names == exact
        else (
            "stationary recent rewards and episode flow; episode endings are goals or no-touch "
            "timeouts; max-tick truncations and rollout-boundary effects are not measured"
        ),
    }


def run_directory(args, algorithm):
    if getattr(args, "resume", None) and not args.run_name:
        return Path(args.resume).expanduser().resolve().parent
    name = args.run_name or f"{algorithm}-{datetime.now():%Y%m%d-%H%M%S}-{args.seed}"
    path = args.output_dir / name
    path.mkdir(parents=True, exist_ok=True)

    return path


def progress_bar(description, total):
    progress = rich_progress.Progress(
        rich_progress.TextColumn("[bold]{task.description}"),
        rich_progress.BarColumn(bar_width=None),
        rich_progress.TaskProgressColumn(),
        rich_progress.MofNCompleteColumn(),
        rich_progress.TextColumn("{task.fields[rate]}", justify="right"),
        rich_progress.TimeElapsedColumn(),
        rich_progress.TimeRemainingColumn(),
    )
    task = progress.add_task(description, total=total, rate="")
    progress.start()

    return progress, task


def log_metrics(writer, step, metrics, progress=None, task=None, *, frameskip=1):
    for name, value in metrics.items():
        writer.add_scalar(name, value, step)

    if progress is not None:
        rate = metrics.get("physics_ticks_per_second")
        progress.update(
            task,
            completed=min(step, progress.tasks[task].total),
            rate="" if rate is None else f"{rate / frameskip:,.0f} agent-steps/s",
        )
