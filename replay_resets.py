import math
import re

from pathlib import Path

import numpy as np
import torch as th

from jarl.data import TensorBatch, TensorDataset

from physics_utils import forward_up_to_quat
from replay_safety import infer_unsafe_start_mask, pre_goal_start_mask
from tracker import (
    BALL_MAX_ANG_SPEED,
    BALL_MAX_SPEED,
    BOOST_MAX,
    CAR_MAX_ANG_SPEED,
    CAR_MAX_SPEED,
    POSITION_SCALE,
)


def _sampled_frame_skip(path: Path, fallback: int) -> int:
    """Read the parse marker when an older replay has no safety sidecar."""
    replay_name = path.stem.split("-", 2)[-1]
    skips = set()
    for marker in path.parent.glob(f".{replay_name}.v*-fs*.complete"):
        match = re.search(r"-fs(\d+)(?:-pov-[^.]+)?\.complete$", marker.name)
        if match:
            skips.add(int(match.group(1)))
    return skips.pop() if len(skips) == 1 else fallback


def load_demonstration_reset_dataset(
    replay_dir: Path,
    device,
    frame_skip: int = 4,
    limit: int | None = None,
    seed: int = 0,
    require_frame_skip_match: bool = True,
) -> TensorDataset:
    random = np.random.default_rng(seed)
    rows = []
    paths = []

    for path in sorted(replay_dir.rglob("*.npy")):
        source = np.load(path, mmap_mode="r")
        if source.ndim == 2 and source.shape[1] == 161:
            paths.append(path)

    if not paths:
        raise ValueError(f"no 1v1 demonstrations found in {replay_dir}")
    quota = None if limit is None else max(1, math.ceil(limit / len(paths)))

    for path in paths:
        source = np.load(path, mmap_mode="r")
        unsafe_path = path.with_suffix(".unsafe-starts.npz")
        sampled_frame_skip = frame_skip
        pre_goal = None
        if unsafe_path.is_file():
            with np.load(unsafe_path) as stored:
                unsafe = np.asarray(stored["unsafe"], dtype=bool)
                sampled_frame_skip = int(stored.get("frame_skip", frame_skip))
                if "pre_goal" in stored:
                    pre_goal = np.asarray(stored["pre_goal"], dtype=bool)
            if require_frame_skip_match and sampled_frame_skip != frame_skip:
                raise ValueError(
                    f"unsafe-start mask for {path.name} uses frame skip "
                    f"{sampled_frame_skip}, expected {frame_skip}"
                )
            if unsafe.shape != (len(source),):
                raise ValueError(f"unsafe-start mask for {path.name} has wrong shape")
        else:
            sampled_frame_skip = _sampled_frame_skip(path, frame_skip)
            unsafe = infer_unsafe_start_mask(
                source[:, 3:6] * BALL_MAX_SPEED, sampled_frame_skip
            )

        if pre_goal is None:
            # Older parsed replays lack goal annotations. Conservatively treat
            # every segment end as a goal, using its sampled (not training) skip.
            pre_goal = pre_goal_start_mask(
                len(source), sampled_frame_skip,
                (len(source) - 1) * sampled_frame_skip,
            )
        if pre_goal.shape != (len(source),):
            raise ValueError(f"pre-goal mask for {path.name} has wrong shape")

        invalid = source[:, -4:].astype(bool).any(axis=-1)
        # Keep the full pro-play distribution (aerials, boosting, flips) like the
        # July-31 pipeline; only drop unsafe/invalid frames. The previous
        # ``stable`` filter required both cars grounded and non-mechanical, which
        # stripped exactly the aerial/contest states needed to learn
        # catches/flicks/aerials.
        eligible = np.flatnonzero(~unsafe & ~invalid & ~pre_goal)
        if len(eligible):
            if quota is not None and len(eligible) > quota:
                eligible = random.choice(eligible, size=quota, replace=False)
            rows.append(np.asarray(source[eligible, :51], dtype=np.float32))

    if not rows:
        raise ValueError(f"no valid 1v1 states found in {replay_dir}")

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
