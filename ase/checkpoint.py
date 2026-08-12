from pathlib import Path

import torch as th


class PeriodicCheckpoint:

    def __init__(
        self,
        modules:   dict[str, th.nn.Module],
        directory: Path,
        interval:  int
    ) -> None:
        if interval < 1:
            raise ValueError("checkpoint interval must be positive")

        self.modules = modules
        self.directory = Path(directory)
        self.interval = interval
        self.step = 0
        self.next_step = interval

        self.directory.mkdir(parents=True, exist_ok=True)

    def ready(self, step: int) -> bool:
        self.step = step
        return step >= self.next_step

    def run(self) -> None:
        payload = {
            "step": self.step,
            **{
                name: module.state_dict()
                for name, module in self.modules.items()
            }
        }
        path = self.directory / f"ase_{self.step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")

        th.save(payload, temporary)
        temporary.replace(path)

        self.next_step = self.step + self.interval
