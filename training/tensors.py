from contextlib import AbstractContextManager

import torch


def plain_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if type(tensor) is torch.Tensor else tensor.as_subclass(torch.Tensor)


def autocast(device: torch.device, enabled: bool) -> AbstractContextManager[None]:
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled)


def require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(
            f"non-finite {name}; training stopped before corrupting a checkpoint"
        )
