import torch

from imitation.windows import decision_history_length


class TransitionRollout:

    def __init__(self, observation, steps, scene_size, window_length=2, potential_shaping=False,
                 decision_stride=1):
        agents = len(observation)
        self.scene_size = scene_size
        self.window_length = window_length
        self.decision_stride = decision_stride
        self.history_length = decision_history_length(window_length, decision_stride)
        self.scenes = observation.new_empty((steps, agents, scene_size))
        self.successors = torch.empty_like(self.scenes)
        self.dones = torch.zeros((steps, agents), dtype=torch.bool, device=observation.device)
        self.terminated = torch.zeros_like(self.dones)
        self.potential_shaping = potential_shaping
        prefix_length = self.history_length - 2 + int(potential_shaping)
        self.prefix = observation.new_full((prefix_length, agents, scene_size), float("nan"))
        self.prefix_dones = torch.zeros((prefix_length, agents), dtype=torch.bool, device=observation.device)
        self.started = False
        self.cached_frames = None
        self.cached_dones = None

    def record(self, step, observation, next_observation, final_observation, done, terminated=None):
        if step == 0:
            self.cached_frames = None
            self.cached_dones = None
            if self.started and len(self.prefix):
                self.prefix = torch.cat((self.prefix, self.scenes), dim=0)[-len(self.prefix):].clone()
                self.prefix_dones = torch.cat((self.prefix_dones, self.dones), dim=0)[-len(self.prefix_dones):].clone()
            self.started = True
        self.scenes[step].copy_(observation[:, : self.scene_size])
        self.dones[step].copy_(done)
        if terminated is None:
            self.terminated[step].zero_()
        else:
            self.terminated[step].copy_(terminated)
        successor = torch.where(
            done[:, None],
            final_observation[:, : self.scene_size],
            next_observation[:, : self.scene_size]
        )
        self.successors[step].copy_(successor)

    def window_indices(self, indices):
        agents = self.scenes.shape[1]
        step, agent = indices // agents, indices % agents
        if self.cached_dones is None:
            self.cached_dones = torch.cat((self.prefix_dones, self.dones), dim=0)
        dones = self.cached_dones
        offsets = torch.arange(self.history_length - 1, device=indices.device)
        times = step[:, None] + offsets + int(self.potential_shaping)
        previous_done = dones[times, agent[:, None]]
        previous_done[:, -1] = False
        last_reset = torch.where(previous_done, offsets + 1, 0).amax(dim=1)
        # Check resets over the full interval, including unsampled decisions.
        selected = slice(None, None, self.decision_stride)
        history_indices = times[:, selected] * agents + agent[:, None]
        successor_indices = (len(self.prefix) + len(self.scenes)) * agents + indices
        all_indices = torch.cat((history_indices, successor_indices[:, None]), dim=1)
        padding = offsets[None, selected] < last_reset[:, None]
        padding = torch.cat((padding, torch.zeros_like(padding[:, :1])), dim=1)
        return all_indices, padding

    def previous_window_indices(self, indices):
        if not self.potential_shaping:
            raise ValueError("previous windows require potential shaping history")

        agents = self.scenes.shape[1]
        step, agent = indices // agents, indices % agents
        if self.cached_dones is None:
            self.cached_dones = torch.cat((self.prefix_dones, self.dones), dim=0)

        offsets = torch.arange(self.history_length, device=indices.device)
        times = step[:, None] + offsets
        previous_done = self.cached_dones[times, agent[:, None]].clone()
        previous_done[:, -1] = False
        last_reset = torch.where(previous_done, offsets + 1, 0).amax(dim=1)
        selected = slice(None, None, self.decision_stride)
        return (times[:, selected] * agents + agent[:, None],
                offsets[None, selected] < last_reset[:, None])

    def previous_windows(self, indices):
        window_indices, padding = self.previous_window_indices(indices)
        return self.all_frames()[window_indices].masked_fill(padding[..., None], float("nan"))

    def all_frames(self):
        if self.cached_frames is None:
            self.cached_frames = torch.cat((self.prefix, self.scenes, self.successors), dim=0).flatten(0, 1)
        return self.cached_frames

    def transitions(self, indices):
        window_indices, padding = self.window_indices(indices)
        frames = self.all_frames()[window_indices]
        return frames.masked_fill(padding[..., None], float("nan"))

    def endpoints(self, indices):
        return torch.stack((self.scenes.flatten(0, 1)[indices], self.successors.flatten(0, 1)[indices]), dim=1)


def split_generated(transitions, heldout_sim, heldout_size, n_cars, terminated=None):
    steps, agents = transitions.shape
    n_sim = agents // n_cars
    indices = torch.arange(steps * agents, device=transitions.device)
    if terminated is not None:
        indices = indices[~terminated.flatten()]

    if heldout_size == 0 or n_sim < 2:
        return indices, indices[:0], heldout_sim

    if heldout_sim is None:
        per_sim = steps * n_cars
        count = min((heldout_size + per_sim - 1) // per_sim, n_sim - 1)
        selected = torch.randperm(n_sim, device=transitions.device)[:count]
        heldout_sim = torch.zeros(n_sim, dtype=torch.bool, device=transitions.device)
        heldout_sim[selected] = True

    heldout = heldout_sim[(indices % agents) // n_cars]
    return indices[~heldout], indices[heldout], heldout_sim
