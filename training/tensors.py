import torch


def plain_tensor(tensor):
    return tensor if type(tensor) is torch.Tensor else tensor.as_subclass(torch.Tensor)


def autocast(device, enabled):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled)
