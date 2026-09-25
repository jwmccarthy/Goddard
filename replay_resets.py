import math
import re

from pathlib import Path

import numpy as np
import torch as th

from jarl.data import TensorBatch, TensorDataset

from physics_utils import forward_up_to_quat
from replay_safety import infer_unsafe_start_mask, pre_goal_start_mask


POSITION_SCALE = (4108.0, 6000.0, 2076.0)
BALL_MAX_SPEED = 6000.0
BALL_MAX_ANG_SPEED = 6.0
CAR_MAX_SPEED = 2300.0
CAR_MAX_ANG_SPEED = 5.5
BOOST_MAX = 100.0


# parse_replays.py's 1v1 layout: ball(9), two cars(2 * 21), pads(68),
# relative ball/car(12), goals(6), then the ego's 19 CARL internal fields.
SCENE_SIZE = 51
INTERNAL_START = 137
INTERNAL_SIZE = 19


def _sampled_frame_skip(path: Path, fallback: int) -> int:
    """Read the parse marker when an older replay has no safety sidecar."""
    replay_name = path.stem.split("-", 2)[-1]
    skips = set()
    for marker in path.parent.glob(f".{replay_name}.v*-fs*.complete"):
        match = re.search(r"-fs(\d+)(?:-pov-[^.]+)?\.complete$", marker.name)
        if match:
            skips.add(int(match.group(1)))
    return skips.pop() if len(skips) == 1 else fallback


def _paired_pov_paths(paths: list[Path]) -> dict[Path, Path]:
    """Find unique other-player POVs for the same replay and gameplay period."""
    groups: dict[tuple[Path, str, str], list[Path]] = {}
    for path in paths:
        parts = path.stem.split("-", 2)
        if len(parts) == 3 and parts[1].isdigit():
            groups.setdefault((path.parent, parts[1], parts[2]), []).append(path)
    paired = {}
    for group in groups.values():
        if len(group) == 2:
            paired[group[0]] = group[1]
            paired[group[1]] = group[0]
    return paired


def _paired_opponent_internal(
    source: np.ndarray,
    indices: np.ndarray,
    scene: np.ndarray,
    visible_internal: np.ndarray,
    counterpart: Path | None,
    sampled_frame_skip: int,
    frame_skip: int,
) -> np.ndarray | None:
    """Use a paired POV only when its cadence, scene, and flags all agree."""
    if counterpart is None:
        return None
    paired = np.load(counterpart, mmap_mode="r")
    if paired.shape != source.shape:
        return None

    sidecar = counterpart.with_suffix(".unsafe-starts.npz")
    if sidecar.is_file():
        with np.load(sidecar) as stored:
            paired_skip = int(stored.get("frame_skip", frame_skip))
    else:
        paired_skip = _sampled_frame_skip(counterpart, frame_skip)
    if paired_skip != sampled_frame_skip:
        return None

    # The orange POV rotates the scene 180 degrees and swaps the car order.
    # Check every sampled row before borrowing any otherwise hidden timers.
    mirrored = np.concatenate(
        (scene[:, :9], scene[:, 30:51], scene[:, 9:30]), axis=1
    )
    mirrored[:, (0, 1, 3, 4, 6, 7)] *= -1
    for start in (9, 30):
        for offset in (0, 3, 6, 9, 12):
            mirrored[:, start + offset:start + offset + 2] *= -1
    if not np.allclose(
        paired[indices, :SCENE_SIZE], mirrored, rtol=1e-5, atol=1e-5
    ):
        return None

    internal = np.asarray(
        paired[indices, INTERNAL_START:INTERNAL_START + INTERNAL_SIZE],
        dtype=np.float32,
    )
    if not np.array_equal(
        internal[:, (0, 7, 8, 17)], visible_internal[:, (0, 7, 8, 17)]
    ):
        return None
    return internal


def load_demonstration_reset_dataset(
    replay_dir: Path,
    device,
    frame_skip: int = 4,
    limit: int | None = None,
    seed: int = 0,
    require_frame_skip_match: bool = True,
) -> TensorDataset:
    """Sample safe replay frames with both cars' available CARL control state.

    Unpaired opponent POVs contain only ground/flip/double-jump/boosting flags;
    their unknown jump, flip, and boost timers remain zero. This can give an
    airborne, unspent opponent a fresh dodge window. CARL resets boost pads to
    active: its reset API cannot restore the replay's pad cooldowns.
    """
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
    paired_paths = _paired_pov_paths(paths)

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
        # Retain aerial, boost, and flip states while excluding unsafe frames.
        eligible = np.flatnonzero(~unsafe & ~invalid & ~pre_goal)
        if len(eligible):
            if quota is not None and len(eligible) > quota:
                eligible = random.choice(eligible, size=quota, replace=False)
            scene = np.asarray(source[eligible, :SCENE_SIZE], dtype=np.float32)
            ego_internal = np.asarray(
                source[eligible, INTERNAL_START:INTERNAL_START + INTERNAL_SIZE],
                dtype=np.float32,
            )
            opponent_internal = np.zeros_like(ego_internal)
            opponent = scene[:, 30:51]
            # CARL's setter uses the same 19-field ordering as the parser.
            opponent_internal[:, 0] = opponent[:, 16]   # on_ground
            opponent_internal[:, 7] = opponent[:, 19]   # has_double_jumped
            opponent_internal[:, 8] = opponent[:, 18]   # has_flipped
            opponent_internal[:, 17] = opponent[:, 20]  # is_boosting
            paired_internal = _paired_opponent_internal(
                source, eligible, scene, opponent_internal,
                paired_paths.get(path), sampled_frame_skip, frame_skip,
            )
            if paired_internal is not None:
                opponent_internal = paired_internal
            rows.append(np.concatenate(
                (scene, ego_internal, opponent_internal), axis=1
            ))

    if not rows:
        raise ValueError(f"no valid 1v1 states found in {replay_dir}")

    states = np.concatenate(rows)
    if limit is not None and len(states) > limit:
        selected = random.choice(len(states), size=limit, replace=False)
        states = states[selected]
    state = th.from_numpy(np.ascontiguousarray(states)).to(device)
    ball = state[:, :9]
    cars = state[:, 9:SCENE_SIZE].reshape(-1, 2, 21)
    internal_state = th.stack((
        state[:, SCENE_SIZE:SCENE_SIZE + INTERNAL_SIZE],
        state[:, SCENE_SIZE + INTERNAL_SIZE:],
    ), dim=1)
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
        "car_internal_state": internal_state,
    })
    return TensorDataset(data)


__all__ = ["load_demonstration_reset_dataset"]
