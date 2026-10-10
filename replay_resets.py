import math
import re

from pathlib import Path

import numpy as np
import torch as th

from carl.gymnasium import CARLBall, CARLCars, CARLResetState
from jarl.data import TensorBatch, TensorDataset
from jarl.envs import DatasetResetSampler

from replay_layout import (
    BALL_SIZE, CAR_SIZE, INTERNAL_SIZE, TEAM_SIZES, team_car_count,
    team_scene_size,
)
from replay_safety import infer_unsafe_start_mask, pre_goal_start_mask


BALL_MAX_SPEED = 6000.0


# The standalone demonstration loader below retains its original 1v1 layout.
# ReplayResetProvider accepts every team size through six cars.
SCENE_SIZE = 51
INTERNAL_START = 137
# CARL's jump grace is 0.025 + 0.025 seconds; flip torque and pitch lock
# last 0.65 + 0.3 seconds (RLConstants.cuh). Neither phase is observable.
JUMP_GRACE_SECONDS = 0.05
FLIP_PITCH_LOCK_SECONDS = 0.95


def _invalid_reset_rotations(scenes: np.ndarray, n_cars: int) -> np.ndarray:
    """Find frames whose car axes CARL cannot convert to reset rotations."""
    cars = scenes[:, BALL_SIZE:].reshape(-1, n_cars, CAR_SIZE)
    forward = cars[:, :, 9:12]
    up = cars[:, :, 12:15]
    right = np.cross(up, forward)
    valid = (
        np.isfinite(forward).all(axis=(1, 2))
        & np.isfinite(up).all(axis=(1, 2))
        & (np.square(forward).sum(axis=-1) >= 1e-8).all(axis=1)
        & (np.square(right).sum(axis=-1) >= 1e-8).all(axis=1)
    )
    return ~valid


def observable_replay_reset_mask(
    scenes: th.Tensor,
    internal_states: th.Tensor,
    known_internal: th.Tensor,
) -> th.Tensor:
    """Keep resets whose unobserved jump/flip phases cannot change the next action.

    A missing player POV supplies only visible car flags. Its canonical zero
    control state is safe at a settled ground start, not in an aerial with an
    unknown jump latch, dodge timer, or active flip torque.
    """
    n_frames, n_cars, width = internal_states.shape
    if (width != INTERNAL_SIZE or scenes.shape != (n_frames, BALL_SIZE + n_cars * CAR_SIZE)
            or known_internal.shape != (n_frames, n_cars)):
        raise ValueError("replay reset scenes, internals, and recorded POVs must match")

    cars = scenes[:, BALL_SIZE:].reshape(n_frames, n_cars, CAR_SIZE)
    grounded = cars[..., 16].bool()
    # Active jump, held jump, dodge torque/autoflip, and pitch lock depend on
    # phases absent from CARL's actor observation (which exposes only the
    # available dodge and its remaining window).
    safe_recorded = (
        th.isfinite(internal_states).all(dim=-1)
        & ~internal_states[..., 4].bool()  # is_jumping
        & ~internal_states[..., 5].bool()  # previous jump input (edge latch)
        & ~internal_states[..., 9].bool()  # is_flipping
        & ~internal_states[..., 11].bool()  # is_autoflipping
        & ~(internal_states[..., 3].bool()
            & (internal_states[..., 6] < JUMP_GRACE_SECONDS))
        & ~(internal_states[..., 8].bool()
            & (internal_states[..., 10] < FLIP_PITCH_LOCK_SECONDS))
    )
    safe_unrecorded = grounded & ~cars[..., 18].bool() & ~cars[..., 19].bool()
    return th.where(known_internal.bool(), safe_recorded, safe_unrecorded).all(dim=-1)


def reset_index_dataset(indices: th.Tensor) -> TensorDataset:
    """Sample eligible frame IDs without copying their scenes."""
    return TensorDataset(TensorBatch({"frame_index": indices}))


class ReplayResetProvider:
    """Resolve sampled replay frames to normalized CARL reset requests."""

    def __init__(
        self,
        sampler: DatasetResetSampler,
        frames: th.Tensor,
        internal_states: th.Tensor,
    ) -> None:
        n_cars = next((team_car_count(size) for size in TEAM_SIZES
                       if frames.ndim == 2 and frames.shape[-1] == team_scene_size(size)), None)
        if n_cars is None or internal_states.shape != (len(frames), n_cars, INTERNAL_SIZE):
            raise ValueError("replay reset scenes and car internal states must match")
        self.sampler = sampler
        self.frames = frames
        self.internal_states = internal_states
        self.n_cars = n_cars
        self.scene_size = frames.shape[-1]

    def __call__(self, reset_mask: th.Tensor) -> CARLResetState | None:
        sample = self.sampler(reset_mask)
        if sample is None:
            return None
        indices = sample["frame_index"]
        scenes = self.frames[indices]
        return CARLResetState(
            simulation_indices=sample["simulation_indices"],
            ball=CARLBall.from_tensor(scenes[:, :9]),
            cars=CARLCars.from_tensor(
                scenes[:, 9:self.scene_size].reshape(-1, self.n_cars, 21), self.n_cars,
            ),
            car_internal_state=self.internal_states[indices],
            normalized=True,
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


def load_demonstration_reset_frames(
    replay_dir: Path,
    device,
    frame_skip: int = 4,
    limit: int | None = None,
    seed: int = 0,
    require_frame_skip_match: bool = True,
) -> tuple[th.Tensor, th.Tensor]:
    """Sample safe replay frames with both cars' available CARL control state.

    Return normalized scenes and raw internal states on the requested device.
    Unpaired opponents must be grounded with an unused flip and jump; their
    unknown jump, flip, and boost timers are initialized to zero. CARL resets
    boost pads to active: its reset API cannot restore replay pad cooldowns.
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
        # Retain safe aerial and boost states while excluding unsafe frames.
        eligible = np.flatnonzero(~unsafe & ~invalid & ~pre_goal)
        if len(eligible):
            # Demoed cars can have undefined axes even on otherwise safe frames.
            scene = np.asarray(source[eligible, :SCENE_SIZE], dtype=np.float32)
            eligible = eligible[~_invalid_reset_rotations(scene, 2)]
        if len(eligible):
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
            internal = np.stack((ego_internal, opponent_internal), axis=1)
            known = th.ones((len(eligible), 2), dtype=th.bool)
            known[:, 1] = paired_internal is not None
            safe = observable_replay_reset_mask(
                th.from_numpy(scene), th.from_numpy(internal), known,
            ).numpy()
            if not safe.any():
                continue
            selected = np.flatnonzero(safe)
            if quota is not None and len(selected) > quota:
                selected = random.choice(selected, size=quota, replace=False)
            rows.append(np.concatenate(
                (scene[selected], internal[selected, 0], internal[selected, 1]), axis=1,
            ))

    if not rows:
        raise ValueError(f"no valid 1v1 states found in {replay_dir}")

    states = np.concatenate(rows)
    if limit is not None and len(states) > limit:
        selected = random.choice(len(states), size=limit, replace=False)
        states = states[selected]
    state = th.from_numpy(np.ascontiguousarray(states)).to(device)
    return state[:, :SCENE_SIZE], state[:, SCENE_SIZE:].view(-1, 2, INTERNAL_SIZE)


__all__ = [
    "ReplayResetProvider", "load_demonstration_reset_frames",
    "observable_replay_reset_mask", "reset_index_dataset",
]
