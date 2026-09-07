from collections.abc import Callable, Sequence
from pathlib import Path

import torch as th


class PeriodicCheckpoint:

    def __init__(
        self,
        modules:  dict[str, th.nn.Module],
        directory: Path,
        interval: int,
        keep:     int,
        config:   dict[str, object] | None = None,
    ) -> None:
        self.modules = modules
        self.directory = directory
        self.interval = interval
        self.keep = keep
        self.config = {} if config is None else dict(config)
        self.step = 0
        self.next_step = interval
        self._written_paths: list[Path] = []
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("tracker_*.pt.tmp"):
            path.unlink()

    def ready(self, step: int) -> bool:
        self.step = step
        return step >= self.next_step

    def run(self) -> None:
        payload = {
            "step": self.step,
            "config": self.config,
            **{name: module.state_dict() for name, module in self.modules.items()},
        }
        path = self.directory / f"tracker_{self.step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)
        if path not in self._written_paths:
            self._written_paths.append(path)

        for old_path in self._written_paths[:-self.keep]:
            old_path.unlink()
        self._written_paths = self._written_paths[-self.keep:]

        self.next_step = self.step + self.interval


class PHCCheckpoint(PeriodicCheckpoint):
    def __init__(
        self,
        directory: Path,
        interval: int,
        keep: int,
        config: dict[str, object],
        assignment_fn: Callable[[], th.Tensor],
    ) -> None:
        super().__init__({}, directory, interval, keep, config)
        self.assignment_fn = assignment_fn
        self.stage = 0
        self.step_offset = 0
        self.policies: Sequence[th.nn.Module] = ()
        self.critic: th.nn.Module | None = None
        self.segment_stats: Sequence[object] = ()

    def set_stage(
        self,
        stage: int,
        policies: Sequence[th.nn.Module],
        critic: th.nn.Module,
        segment_stats: Sequence[object],
        step_offset: int,
    ) -> None:
        self.stage = stage
        self.policies = policies
        self.critic = critic
        self.segment_stats = segment_stats
        self.step_offset = step_offset

    def ready(self, step: int) -> bool:
        return super().ready(self.step_offset + step)

    def run(self) -> None:
        if not self.policies or self.critic is None:
            raise RuntimeError("PHC checkpoint has no active training stage")
        payload = {
            "step": self.step,
            "stage": self.stage,
            "config": self.config,
            "specialists": [policy.state_dict() for policy in self.policies],
            "critic": self.critic.state_dict(),
            "segment_stats": [stats.state_dict() for stats in self.segment_stats],
            "assignments": self.assignment_fn().cpu(),
        }
        path = self.directory / f"tracker_{self.step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)
        if path not in self._written_paths:
            self._written_paths.append(path)

        for old_path in self._written_paths[:-self.keep]:
            old_path.unlink()
        self._written_paths = self._written_paths[-self.keep:]
        self.next_step = self.step + self.interval
