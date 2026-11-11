import torch


class PolicyWindowHistory:

    def __init__(self, capacity, add_size, interval=1):
        if interval < 1:
            raise ValueError("history interval must be at least one rollout")
        self.capacity = capacity
        self.add_size = add_size
        self.interval = interval
        self.rollouts = 0
        self.windows = None
        self.size = 0
        self.next = 0

    @torch.no_grad()
    def add(self, scenes, indices):
        scheduled = self.rollouts % self.interval == 0
        self.rollouts += 1
        if not scheduled:
            return 0
        count = min(self.add_size, self.capacity, len(indices))
        if not count:
            return 0
        selected = indices[torch.randperm(len(indices), device=indices.device)[:count]]
        windows = scenes.transitions(selected)
        if self.windows is None:
            self.windows = windows.new_empty((self.capacity, *windows.shape[1:]))
        first = min(count, self.capacity - self.next)
        self.windows[self.next:self.next + first].copy_(windows[:first])
        if first < count:
            self.windows[:count - first].copy_(windows[first:])
        self.next = (self.next + count) % self.capacity
        self.size = min(self.size + count, self.capacity)
        return count

    def sample(self, count):
        indices = torch.randint(self.size, (count,), device=self.windows.device)
        return self.windows[indices]
