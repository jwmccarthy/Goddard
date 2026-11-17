"""CPU-only actor inference. No training, CARL, or PyTorch dependency."""

import numpy as np
from schema import LEGACY_SCHEMA, observation_size

ACTION_SIZES = (3, 3, 3, 2, 2, 3, 2)
ACTION_OFFSETS = (0, 3, 6, 9, 11, 13, 16)


class Policy:
    def __init__(
        self,
        path,
        sample=True,
        seed=1,
    ):
        with np.load(path, allow_pickle=False) as archive:
            self.layers = [
                (archive[f"weight_{i}"].copy(), archive[f"bias_{i}"].copy())
                for i in range(int(archive["layer_count"].item()))
            ]
            self.pad_positions = archive["pad_positions"].copy()
            self.observation_schema = (str(archive["observation_schema"].item())
                                       if "observation_schema" in archive else LEGACY_SCHEMA)
            self.previous_controls = (bool(archive["previous_controls"].item())
                                      if "previous_controls" in archive else False)
        self.observation_size = observation_size(self.observation_schema, self.previous_controls)
        if (self.layers[0][0].shape[1] != self.observation_size
                or self.layers[-1][0].shape[0] != 18):
            raise ValueError("actor dimensions do not match the exported observation schema / 18 logits")
        self.sample = sample
        self.rng = np.random.default_rng(seed)

    def logits(
        self,
        observation,
    ):
        value = np.asarray(observation, dtype=np.float32)
        if value.ndim < 1 or value.shape[-1] != self.observation_size:
            raise ValueError(f"expected {self.observation_size} observation values")
        for index, (weight, bias) in enumerate(self.layers):
            value = value @ weight.T + bias
            if index + 1 < len(self.layers):
                value = np.maximum(value, 0)
        return value

    def actions(
        self,
        observation,
    ):
        logits = self.logits(observation)
        if logits.shape != (18,) or not np.isfinite(logits).all():
            raise ValueError("invalid policy inputs or logits")
        logits = logits.copy()
        ground = observation[25] > 0.5
        if not ground:
            logits[10] = -1e9
        if observation[24] <= 0:
            logits[12] = -1e9
        if not (ground or (observation[137] > 0.5 and observation[138] > 1 / 120)):
            logits[17] = -1e9
        actions = []
        for offset, size in zip(ACTION_OFFSETS, ACTION_SIZES):
            head = logits[offset:offset + size]
            if self.sample:
                probabilities = np.exp(head - head.max())
                actions.append(int(self.rng.choice(size, p=probabilities / probabilities.sum())))
            else:
                actions.append(int(head.argmax()))
        return actions


def decode_actions(
    actions,
):
    axis = (0.0, -1.0, 1.0)
    horizontal, vertical, throttle, slide, boost, roll, jump = actions
    return {
        "throttle": axis[throttle], "steer": axis[horizontal],
        "yaw": axis[horizontal], "pitch": axis[vertical], "roll": axis[roll],
        "handbrake": bool(slide), "boost": bool(boost), "jump": bool(jump),
    }
