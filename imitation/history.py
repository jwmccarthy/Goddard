import torch


class HistoricalReplayBuffer:

    def __init__(self, capacity, length, scene_size, device, seed=0):
        self.capacity = capacity
        self.length = length
        self.scene_size = scene_size
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.buffer = (
            torch.empty((capacity, length, scene_size), device=device) if capacity > 0 else None
        )
        self.size = 0
        self.start = 0

    def add(self, windows, count):
        if self.buffer is None or count <= 0 or not len(windows):
            return

        if len(windows) > count:
            order = torch.randperm(
                len(windows),
                device=windows.device,
                generator=self.generator
            )
            windows = windows[order[:count]]

        windows = windows.detach()[-self.capacity :]
        filling = min(len(windows), self.capacity - self.size)

        if filling:
            self.buffer[self.size : self.size + filling] = windows[:filling]
            self.size += filling

        remaining = len(windows) - filling

        if remaining:
            positions = (
                self.start + torch.arange(remaining, device=self.device)
            ) % self.capacity
            self.buffer[positions] = windows[filling:]
            self.start = (self.start + remaining) % self.capacity

    def sample(self, count):
        if self.size == 0:
            return torch.empty((0, self.length, self.scene_size), device=self.device)

        count = min(count, self.size)
        order = torch.randperm(
            self.size,
            device=self.device,
            generator=self.generator
        )[:count]
        positions = (self.start + order) % self.capacity

        return self.buffer[positions]
