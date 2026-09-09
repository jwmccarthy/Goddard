import math

from pathlib import Path

import numpy as np
import torch as th

from jarl.data import TensorBatch, TensorDataset

from physics_utils import forward_up_to_quat
from replay_safety import infer_unsafe_start_mask
from tracker import (
    BALL_MAX_ANG_SPEED,
    BALL_MAX_SPEED,
    BOOST_MAX,
    CAR_MAX_ANG_SPEED,
    CAR_MAX_SPEED,
    POSITION_SCALE,
)


def load_demonstration_reset_dataset(
    replay_dir: Path,
    device,
    frame_skip: int,
    limit: int | None = None,
    seed: int = 0,
) -> TensorDataset:
    random = np.random.default_rng(seed)
    rows = []
    paths = []

    for path in sorted(replay_dir.glob("*.npy")):
        source = np.load(path, mmap_mode="r")
        if source.ndim == 2 and source.shape[1] == 161:
            paths.append(path)

    if not paths:
        raise ValueError(f"no 1v1 demonstrations found in {replay_dir}")
    quota = None if limit is None else max(1, math.ceil(limit / len(paths)))

    for path in paths:
        source = np.load(path, mmap_mode="r")
        unsafe_path = path.with_suffix(".unsafe-starts.npz")
        if unsafe_path.is_file():
            with np.load(unsafe_path) as stored:
                unsafe = np.asarray(stored["unsafe"], dtype=bool)
                stored_skip = int(stored.get("frame_skip", frame_skip))
            if stored_skip != frame_skip:
                raise ValueError(
                    f"unsafe-start mask for {path.name} uses frame skip "
                    f"{stored_skip}, expected {frame_skip}"
                )
            if unsafe.shape != (len(source),):
                raise ValueError(f"unsafe-start mask for {path.name} has wrong shape")
        else:
            unsafe = infer_unsafe_start_mask(
                source[:, 3:6] * BALL_MAX_SPEED, frame_skip
            )

        cars = source[:, 9:51].reshape(-1, 2, 21)
        invalid = source[:, -4:].astype(bool).any(axis=-1)
        stable = cars[..., 16].astype(bool).all(axis=-1)
        stable &= ~cars[..., 17:21].astype(bool).any(axis=(-2, -1))
        eligible = np.flatnonzero(~unsafe & ~invalid & stable)
        if len(eligible):
            if quota is not None and len(eligible) > quota:
                eligible = random.choice(eligible, size=quota, replace=False)
            rows.append(np.asarray(source[eligible, :51], dtype=np.float32))

    if not rows:
        raise ValueError(f"no safe grounded 1v1 states found in {replay_dir}")

    states = np.concatenate(rows)
    if limit is not None and len(states) > limit:
        selected = random.choice(len(states), size=limit, replace=False)
        states = states[selected]
    state = th.from_numpy(np.ascontiguousarray(states)).to(device)
    ball = state[:, :9]
    cars = state[:, 9:51].reshape(-1, 2, 21)
    position_scale = th.tensor(POSITION_SCALE, device=device)
    data = TensorBatch({
        "ball_position": ball[:, :3] * position_scale,
        "ball_velocity": ball[:, 3:6] * BALL_MAX_SPEED,
        "ball_angular_velocity": ball[:, 6:9] * BALL_MAX_ANG_SPEED,
        "car_position": cars[..., :3] * position_scale,
        "car_rotation": forward_up_to_quat(cars[..., 9:12], cars[..., 12:15]),
        "car_velocity": cars[..., 3:6] * CAR_MAX_SPEED,
        "car_angular_velocity": cars[..., 6:9] * CAR_MAX_ANG_SPEED,
        "car_demoed": cars[..., 17].bool(),
        "car_boost": cars[..., 15] * BOOST_MAX,
    })
    return TensorDataset(data)


__all__ = ["load_demonstration_reset_dataset"]
