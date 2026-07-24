from dataclasses import asdict
from pathlib import Path

import torch

from jarl.runtime.clock import Clock


class TrainingCheckpointer:
    FORMAT_VERSION = 1

    def __init__(
        self,
        path: Path,
        modules: dict[str, torch.nn.Module],
        optimizers: dict[str, torch.optim.Optimizer],
        stateful: dict[str, object] | None = None,
    ) -> None:
        self.path = path
        self.modules = modules
        self.optimizers = optimizers
        self.stateful = stateful or {}

    def __call__(self, trainer) -> None:
        state = {
            "format_version": self.FORMAT_VERSION,
            "clock": asdict(trainer.clock),
            "modules": {
                name: module.state_dict() for name, module in self.modules.items()
            },
            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.items()
            },
            "stateful": {
                name: value.state_dict() for name, value in self.stateful.items()
            },
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        torch.save(state, temporary)
        temporary.replace(self.path)

    def load(self, path: Path, device: torch.device | str) -> Clock:
        state = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
        if state.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("unsupported training checkpoint format")
        self._require_names("modules", state["modules"], self.modules)
        self._require_names("optimizers", state["optimizers"], self.optimizers)
        self._require_names("stateful", state["stateful"], self.stateful)
        for name, module in self.modules.items():
            module.load_state_dict(state["modules"][name])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
        for name, value in self.stateful.items():
            value.load_state_dict(state["stateful"][name])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(
            [rng.cpu() for rng in state["cuda_rng_state"]]
        )
        return Clock(**state["clock"])

    @classmethod
    def load_modules(
        cls,
        path: Path,
        modules: dict[str, torch.nn.Module],
        device: torch.device | str,
    ) -> None:
        state = torch.load(path, map_location=device, weights_only=False)
        if state.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported training checkpoint format")
        cls._require_names("modules", state["modules"], modules)
        for name, module in modules.items():
            module.load_state_dict(state["modules"][name])

    @staticmethod
    def _require_names(kind: str, saved: dict, current: dict) -> None:
        if saved.keys() != current.keys():
            raise ValueError(
                f"training checkpoint {kind} do not match: "
                f"saved={sorted(saved)}, current={sorted(current)}"
            )
