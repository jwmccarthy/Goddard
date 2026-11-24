from __future__ import annotations

import random
from argparse import Namespace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from training.models import Agent


def load_payload(path: str | Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


@lru_cache(maxsize=1)
def read_checkpoint(path: str | Path) -> dict[str, Any]:
    return load_payload(path)


def save_checkpoint(
    path:      str | Path,
    algorithm: str,
    args:      Namespace,
    agent:     Agent,
    optimizer: torch.optim.Optimizer,
    step:      int,
    *,
    replay_reset_rng: torch.Tensor | None = None,
    opponent_pool:    dict[str, Any] | None = None,
    **extra:          Any
) -> None:
    payload = {
        "format":           "goddard-leanrl-v1",
        "algorithm":        algorithm,
        "config":           vars(args),
        "agent":            agent.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "step":             step,
        "torch_rng":        torch.get_rng_state(),
        "cuda_rng":         torch.cuda.get_rng_state_all(),
        "numpy_rng":        np.random.get_state(),
        "python_rng":       random.getstate(),
        "replay_reset_rng": replay_reset_rng,
        "opponent_pool":    opponent_pool,
        **extra
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    latest = path if path.name == "latest.pt" else None

    if latest is not None:
        path = path.with_name(f"checkpoint_{step}.pt")

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)

    if latest is not None:
        temporary = latest.with_suffix(latest.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(path.name)
        temporary.replace(latest)


def load_checkpoint(
    path:      str | Path,
    agent:     Agent,
    optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    payload = read_checkpoint(path)
    read_checkpoint.cache_clear()
    agent.load_state_dict(payload["agent"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])

    return payload
