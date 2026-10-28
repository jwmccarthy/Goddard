from pathlib import Path

import numpy as np
import torch

from replays.layout import BALL_SIZE, CAR_SIZE, INTERNAL_SIZE, team_observation_size


def replay_key(path):
    path = Path(path)
    parts = path.stem.split("-", 2)

    return path.parent, tuple(parts[1:]) if len(parts) == 3 else (path.stem,)


def replay_internal_states(rows, scenes, team_size, counterparts):
    n_cars = team_size * 2
    cars = scenes[:, BALL_SIZE:].reshape(len(scenes), n_cars, CAR_SIZE)
    internal = np.zeros((len(scenes), n_cars, INTERNAL_SIZE), dtype=np.float32)
    internal_start = team_observation_size(team_size)

    # Missing POVs supply visible flags only; unrecorded timers stay zero.
    for scene_field, internal_field in ((16, 0), (19, 7), (18, 8), (20, 17)):
        internal[..., internal_field] = cars[..., scene_field]

    internal[:, 0] = rows[: len(scenes), internal_start : internal_start + INTERNAL_SIZE]

    for path in counterparts:
        other = np.load(path, mmap_mode="r", allow_pickle=False)

        if other.shape != rows.shape:
            continue

        ego = other[: len(scenes), BALL_SIZE : BALL_SIZE + CAR_SIZE]

        for actor in range(1, n_cars):
            expected = cars[:, actor].copy()

            if actor >= team_size:
                for start in (0, 3, 6, 9, 12):
                    expected[:, start : start + 2] *= -1

            matches = np.isclose(expected, ego, rtol=1e-4, atol=1e-4).all(axis=-1)

            if matches.any():
                recorded = other[: len(scenes), internal_start : internal_start + INTERNAL_SIZE]
                internal[matches, actor] = recorded[matches]

    return internal


class ReplayResetProvider:

    def __init__(self, experts, fraction=0.70, seed=0):
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("replay reset fraction must be in [0, 1]")

        if fraction > 0 and not len(experts.reset_indices):
            raise ValueError("no valid training replay frames available for resets")

        self.experts = experts
        self.fraction = fraction
        self.generator = torch.Generator(device=experts.frames.device).manual_seed(seed)
        self.last_reset_indices = None
        self.selected = 0
        self.total = 0

    @torch.no_grad()
    def __call__(self, reset_mask):
        from carl.gymnasium import CARLBall, CARLCars, CARLResetState

        if self.last_reset_indices is None:
            self.last_reset_indices = torch.full_like(reset_mask, -1, dtype=torch.long)

        self.last_reset_indices[reset_mask] = -1
        simulations = reset_mask.nonzero().flatten()
        self.total += len(simulations)

        if not len(simulations) or self.fraction == 0:
            return None

        selected = (
            torch.rand(len(simulations), device=reset_mask.device, generator=self.generator)
            < self.fraction
        )
        simulations = simulations[selected]

        if not len(simulations):
            return None

        choices = torch.randint(
            len(self.experts.reset_indices),
            (len(simulations),),
            device=reset_mask.device,
            generator=self.generator,
        )
        indices = self.experts.reset_indices[choices]
        self.last_reset_indices[simulations] = indices
        self.selected += len(simulations)
        scenes = self.experts.frames[indices]
        cars = scenes[:, BALL_SIZE:].reshape(-1, self.experts.n_cars, CAR_SIZE)

        return CARLResetState(
            simulation_indices=simulations,
            ball=CARLBall.from_tensor(scenes[:, :BALL_SIZE]),
            cars=CARLCars.from_tensor(cars, self.experts.n_cars),
            car_internal_state=self.experts.internal_states[indices],
            normalized=True,
        )

    def diagnostic_metrics(self):
        if not self.total:
            return {}

        metrics = {"gameplay/replay_reset_fraction": self.selected / self.total}
        self.selected = self.total = 0

        return metrics
