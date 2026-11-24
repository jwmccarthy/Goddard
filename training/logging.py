import logging
from argparse import Namespace
from datetime import datetime
from pathlib import Path

from rich import progress as rich_progress
from rich.logging import RichHandler
from torch.utils.tensorboard import SummaryWriter

METRICS = {
    "physics_ticks_per_second":        "train/ticks_per_second",
    "reward_per_tick":                 "train/reward",
    "ppo/gamma":                       "train/gamma",
    "policy_loss":                     "ppo/policy_loss",
    "value_loss":                      "ppo/value_loss",
    "entropy":                         "ppo/entropy",
    "ppo/entropy_coefficient":         "ppo/entropy_coefficient",
    "approx_kl":                       "ppo/kl",
    "clip_fraction":                   "ppo/clip_fraction",
    "explained_variance":              "ppo/explained_variance",
    "value/explained_variance_valid":  "critic/explained_variance_valid",
    "value/return_mean":               "critic/return_mean",
    "value/return_std":                "critic/return_std",
    "value/prediction_mean":           "critic/prediction_mean",
    "value/prediction_std":            "critic/prediction_std",
    "value/bias":                      "critic/bias",
    "value/rmse":                      "critic/rmse",
    "discriminator/train_loss":        "gaifo/loss",
    "imitation/gradient_penalty":      "gaifo/gradient_penalty",
    "imitation/reward_per_tick":       "gaifo/reward",
    "imitation/raw_reward_per_tick":   "gaifo/raw_reward",
    "imitation/reward_weight":         "gaifo/imitation_reward_weight",
    "imitation/goal_weight":           "gaifo/goal_weight",
    "imitation/return_scale":          "gaifo/return_scale",
    "imitation/ratio_capped_fraction": "gaifo/capped_fraction",
    **{
        f"style_potential/{name}": f"style/{name}"
        for name in (
            "expert_reward_mean", "expert_reward_std", "target_mean",
            "target_std", "complete_fraction", "mean_chain_decisions", "fit_loss",
            "policy_potential_mean", "shaping_mean", "shaping_std"
        )
    },
    "gameplay/goals_for_per_1000_ticks":      "game/goals_per_1000_ticks",
    "gameplay/touches_per_1000_ticks":        "game/touches_per_1000_ticks",
    "gameplay/aerial_touches_per_1000_ticks": (
        "game/aerial_touches_per_1000_ticks"
    ),
    "gameplay/low_aerial_touches_per_1000_ticks": (
        "game/low_aerial_touches_per_1000_ticks"
    ),
    "gameplay/high_aerial_touches_per_1000_ticks": (
        "game/high_aerial_touches_per_1000_ticks"
    ),
    "gameplay/timeout_fraction":               "game/timeout_fraction",
    "gameplay/replay_reset_fraction":          "game/replay_reset_fraction",
    "gameplay/bank_aerial_touch_rate":         "game/bank_aerial_touch_rate",
    "gameplay/replay_aerial_touch_rate":       "game/replay_aerial_touch_rate",
    "gameplay/bank_aerial_touch_rate_floor":   "game/bank_aerial_touch_rate_floor",
    "gameplay/bank_aerial_touch_rate_wall":    "game/bank_aerial_touch_rate_wall",
    "gameplay/bank_aerial_touch_rate_ceiling": (
        "game/bank_aerial_touch_rate_ceiling"
    ),
    "gameplay/bank_aerial_touch_rate_ramp": "game/bank_aerial_touch_rate_ramp",
    "gameplay/bank_tick_share":             "game/bank_tick_share",
    "self_play/pool_size":                  "self_play/pool_size",
    "self_play/resident_snapshots":         "self_play/resident_snapshots",
    "self_play/historical_match_fraction":  "self_play/historical_fraction",
    "self_play/latest_snapshot_step":       "self_play/latest_snapshot_step"
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
    writer:   SummaryWriter,
    step:     int,
    metrics:  dict[str, float],
    progress: rich_progress.Progress,
    task:     int,
    *,
    frameskip: int
) -> None:
    ticks = step * frameskip

    for name, tag in METRICS.items():
        if name in metrics:
            writer.add_scalar(tag, metrics[name], ticks)

    rate = metrics["physics_ticks_per_second"]
    progress.update(
        task,
        completed=min(step, progress.tasks[task].total),
        rate=f"{rate / frameskip:,.0f} agent-steps/s"
    )
