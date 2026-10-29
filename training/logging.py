from datetime import datetime

from rich import progress as rich_progress


def run_directory(args, algorithm):
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
        rich_progress.TimeRemainingColumn()
    )
    task = progress.add_task(description, total=total, rate="")
    progress.start()

    return progress, task


def log_metrics(writer, step, metrics, progress=None, task=None):
    for name, value in metrics.items():
        writer.add_scalar(name, value, step)

    if progress is not None:
        rate = metrics.get("transitions_per_second")
        progress.update(
            task,
            completed=min(step, progress.tasks[task].total),
            rate="" if rate is None else f"{rate:,.0f} it/s"
        )
