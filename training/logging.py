import logging
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from rich import progress as rich_progress
from rich.logging import RichHandler
from torch.utils.tensorboard import SummaryWriter

METRICS = {
    "physics_ticks_per_second":               "train/ticks_per_second",
    "reward_per_tick":                        "train/reward",
    "ppo/gamma":                              "train/gamma",
    "policy_loss":                            "ppo/policy_loss",
    "value_loss":                             "ppo/value_loss",
    "entropy":                                "ppo/entropy",
    "approx_kl":                              "ppo/kl",
    "clip_fraction":                          "ppo/clip_fraction",
    "explained_variance":                     "ppo/explained_variance",
    "discriminator/train_loss":               "gaifo/loss",
    "imitation/reward_per_tick":              "gaifo/reward",
    "imitation/goal_weight":                  "gaifo/goal_weight",
    "imitation/ratio_capped_fraction":        "gaifo/capped_fraction",
    "gameplay/goals_for_per_1000_ticks":      "game/goals_per_1000_ticks",
    "gameplay/touches_per_1000_ticks":        "game/touches_per_1000_ticks",
    "gameplay/aerial_touches_per_1000_ticks": (
        "game/aerial_touches_per_1000_ticks"
    ),
    "gameplay/timeout_fraction":              "game/timeout_fraction",
    "gameplay/replay_reset_fraction":         "game/replay_reset_fraction",
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(show_time=False, show_path=False, show_level=False)
        ]
    )


def run_directory(args: Namespace, algorithm: str) -> Path:
    if args.resume and not args.run_name:
        return Path(args.resume).expanduser().resolve().parent

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = args.run_name or f"{algorithm}-{timestamp}-{args.seed}"
    path = args.output_dir / name
    path.mkdir(parents=True, exist_ok=True)

    return path


def progress_bar(
    description: str,
    total:       int
) -> tuple[rich_progress.Progress, int]:
    progress = rich_progress.Progress(
        rich_progress.TextColumn("[bold]{task.description}"),
        rich_progress.BarColumn(bar_width=None),
        rich_progress.TaskProgressColumn(),
        rich_progress.TextColumn("{task.fields[rate]}", justify="right"),
        rich_progress.TimeRemainingColumn()
    )

    task = progress.add_task(description, total=total, rate="")

    progress.start()

    return progress, task


def log_metrics(
    writer:    SummaryWriter,
    step:      int,
    metrics:   dict[str, float],
    progress:  rich_progress.Progress,
    task:      int,
    *,
    frameskip: int
) -> None:
    for name, tag in METRICS.items():
        if name in metrics:
            writer.add_scalar(tag, metrics[name], step)

    rate = metrics["physics_ticks_per_second"]
    progress.update(
        task,
        completed=min(step, progress.tasks[task].total),
        rate=f"{rate / frameskip:,.0f} agent-steps/s"
    )
