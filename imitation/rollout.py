import torch


class SceneRollout:

    def __init__(self, observation, steps, length, scene_size):
        agents = len(observation)
        self.length = length
        self.scene_size = scene_size
        self.scenes = observation.new_empty((steps, agents, scene_size))
        self.successors = torch.empty_like(self.scenes)
        self.ages = torch.empty((steps, agents), dtype=torch.long, device=observation.device)
        self.age = torch.zeros(agents, dtype=torch.long, device=observation.device)
        self.history = observation[None, :, :scene_size].expand(length - 2, -1, -1).clone()
        self.timeline = None
        self.valid_indices = None

    def record(self, step, observation, next_observation, final_observation, done):
        self.scenes[step].copy_(observation[:, : self.scene_size])
        successor = torch.where(
            done[:, None],
            final_observation[:, : self.scene_size],
            next_observation[:, : self.scene_size],
        )
        self.successors[step].copy_(successor)
        self.ages[step].copy_(self.age)
        self.age.add_(1).masked_fill_(done, 0)

    def prepare(self):
        self.timeline = torch.cat((self.history, self.scenes))
        self.valid_indices = (self.ages.flatten() >= self.length - 2).nonzero().flatten()

        if self.length > 2:
            self.history.copy_(self.timeline[-(self.length - 2) :])

    def windows(self, indices):
        agents = self.scenes.shape[1]
        step = indices // agents
        agent = indices % agents
        end = step + self.length - 2
        start = end - self.ages[step, agent]
        offsets = torch.arange(2 - self.length, 1, device=indices.device)
        frames = torch.maximum(end[:, None] + offsets, start[:, None])
        prefix = self.timeline[frames, agent[:, None]]

        return torch.cat((prefix, self.successors[step, agent, None]), dim=1)


def split_generated(valid, heldout_sim, heldout_size, n_cars):
    steps, agents = valid.shape
    n_sim = agents // n_cars
    flat_valid = valid.flatten()
    valid_indices = flat_valid.nonzero().flatten()

    if heldout_size == 0 or n_sim < 2:
        return valid_indices, valid_indices[:0], heldout_sim

    if heldout_sim is not None:
        mask = (
            heldout_sim.view(1, n_sim, 1)
            .expand(steps, n_sim, n_cars)
            .reshape(steps, agents)
            .flatten()
        )
        heldout_mask = mask & flat_valid

        return (
            (flat_valid & ~heldout_mask).nonzero().flatten(),
            heldout_mask.nonzero().flatten(),
            heldout_sim,
        )

    per_sim = valid.view(steps, n_sim, n_cars).sum(dim=(0, 2))
    candidates = (per_sim > 0).nonzero().flatten()

    if len(candidates) < 2:
        return valid_indices, valid_indices[:0], heldout_sim

    candidates = candidates[torch.randperm(len(candidates), device=candidates.device)]
    cumulative = per_sim[candidates].cumsum(0)
    count = (
        int(
            torch.searchsorted(
                cumulative,
                torch.tensor(heldout_size, device=cumulative.device),
            ).item()
        )
        + 1
    )
    count = min(count, len(candidates) - 1)
    heldout_sim = torch.zeros(n_sim, dtype=torch.bool, device=valid.device)
    heldout_sim[candidates[:count]] = True
    mask = (
        heldout_sim.view(1, n_sim, 1).expand(steps, n_sim, n_cars).reshape(steps, agents).flatten()
    )
    heldout_mask = mask & flat_valid

    return (
        (flat_valid & ~heldout_mask).nonzero().flatten(),
        heldout_mask.nonzero().flatten(),
        heldout_sim,
    )
