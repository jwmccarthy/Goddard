import torch


def plain_tensor(tensor):
    return tensor if type(tensor) is torch.Tensor else tensor.as_subclass(torch.Tensor)


def autocast(device, enabled):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled)


def require_finite(tensor, name):
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"non-finite {name}; training stopped before corrupting a checkpoint")
