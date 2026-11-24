import hashlib
import json
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch

from replays import layout
from replays.confidence import load_confidence
from replays.jump_actions import load_jump_actions
from replays.quality import motion_discontinuities
from replays.reconstruction import MODES
from replays.results import winning_perspectives

FrameArray = TypeVar("FrameArray", np.ndarray, torch.Tensor)


def split_expert_matches(
    paths:    list[Path],
    fraction: float
) -> tuple[list[Path], list[Path]]:
    """Reserve whole matches, including all players and periods, by hash."""
    train, holdout = [], []

    for path in paths:
        parts = path.stem.split("-", 2)
        identity = parts[2] if len(parts) == 3 else path.stem
        digest = hashlib.sha256(identity.encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        (holdout if value < fraction else train).append(path)

    return train, holdout


def parsed_directory(directory: str | Path) -> Path:
    directory = Path(directory)

    return directory / "parsed" if (directory / "parsed").is_dir() else directory


def replay_paths(
    directory: str | Path,
    team_size: int | None = None
) -> tuple[int, list[Path]]:
    """Select one match format from parsed row layouts."""
    directory = parsed_directory(directory)
    widths = {
        layout.team_replay_row_size(size): size for size in layout.TEAM_SIZES
    }
    groups: dict[int, list[Path]] = {}

    for path in sorted(directory.rglob("*.npy")):
        rows = np.load(path, mmap_mode="r", allow_pickle=False)
        size = widths.get(rows.shape[1]) if rows.ndim == 2 else None

        if size is not None and len(rows):
            groups.setdefault(size, []).append(path)

    if team_size is None:
        if len(groups) > 1:
            formats = ", ".join(f"{size}v{size}" for size in sorted(groups))
            raise ValueError(
                f"mixed replay formats ({formats}); select --team-size"
            )

        team_size = next(iter(groups), None)

    if team_size not in groups:
        requested = "" if team_size is None else f" for {team_size}v{team_size}"
        raise ValueError(f"no parsed replays{requested} in {directory}")

    return team_size, groups[team_size]


def stationary_kickoff_frames(scene: FrameArray, n_cars: int) -> FrameArray:
    """Identify still kickoff poses; pose alone does not imply a countdown."""

    array = scene.new_tensor if isinstance(scene, torch.Tensor) else np.array
    position_scale = array(layout.POSITION_SCALE)
    ball_position = scene[..., :3] * position_scale
    ball_spawn = array([0.0, 0.0, 92.75])
    ball_still = (
        (abs(ball_position - ball_spawn) <= 5).all(-1)
        & ((scene[..., 3:6] ** 2).sum(-1) <= (1 / 6000) ** 2)
        & ((scene[..., 6:9] ** 2).sum(-1) <= (0.05 / 6) ** 2)
    )

    end = layout.BALL_SIZE + n_cars * layout.CAR_SIZE
    cars = scene[..., layout.BALL_SIZE : end].reshape(
        *scene.shape[:-1], n_cars, layout.CAR_SIZE
    )
    positions = cars[..., :3] * position_scale
    spawns = array([[2048.0, 2560.0], [256.0, 3840.0], [0.0, 4608.0]])
    at_spawn = (abs(abs(positions[..., None, :2]) - spawns) <= 5)
    at_spawn = at_spawn.all(-1).any(-1)
    cars_still = (
        at_spawn
        & (abs(positions[..., 2] - 17) <= 5)
        & ((cars[..., 3:6] ** 2).sum(-1) <= (10 / 2300) ** 2)
        & ((cars[..., 6:9] ** 2).sum(-1) <= (0.05 / 5.5) ** 2)
    )

    return ball_still & cars_still.all(-1)


def replay_pairs(
    path:            Path,
    team_size:       int,
    stride:          int,
    limit:           int | None,
    boost_pad_state: bool = False,
    window_length:   int = 2
) -> tuple[np.ndarray, np.ndarray] | None:
    rows = np.load(path, mmap_mode="r", allow_pickle=False)

    if rows.ndim != 2 or rows.shape[1] != layout.team_replay_row_size(team_size):
        return None

    metadata = path.with_suffix(".json")
    meta = json.loads(metadata.read_text())

    if meta["frameskip"] != layout.PARSED_FRAMESKIP:
        raise ValueError(f"parsed replay cadence differs: {path}")

    count = min(len(rows), limit) if limit is not None else len(rows)
    span = stride * (window_length - 1)

    if count <= span:
        return None

    scene_size = layout.team_scene_size(team_size)
    n_cars = layout.team_car_count(team_size)
    width = layout.team_imitation_size(team_size, boost_pad_state)
    scene = np.array(rows[:count, :width], dtype=np.float32)
    frames = scene[:, :scene_size]
    demoed = (frames[:, layout.BALL_SIZE + 17 :: layout.CAR_SIZE] > 0.5)
    demoed = demoed.any(-1)
    affected = demoed | np.concatenate(([False], demoed[:-1]))
    discontinuity = rows[:count, -1] > 0.5

    if affected.any():
        # Existing replay files may flag frozen demo physics as bad motion.
        checked = motion_discontinuities(
            rows[:count, :-1], n_cars, layout.PARSED_FRAMESKIP
        )
        discontinuity &= ~affected | checked

    bad = ~np.isfinite(scene).all(-1) | (rows[:count, -2] > 0.5)
    bad |= discontinuity
    bad |= stationary_kickoff_frames(frames, n_cars)

    reconstruction = meta["reconstruction"]["version"]

    if reconstruction != "legacy":
        if reconstruction not in MODES[1:]:
            raise ValueError(f"unsupported replay reconstruction: {path}")

        quality = load_confidence(path, len(rows), n_cars, reconstruction)
        bad |= quality["unsafe"][:count].any(1)

    if meta["goal_terminal"] and count == len(rows):
        bad[-1] = True

    candidates = np.arange(span, count)
    cumulative = np.concatenate(([0], np.cumsum(bad)))
    valid = cumulative[candidates + 1] == cumulative[candidates - span]

    return (scene, candidates[valid]) if valid.any() else None


class ExpertDataset:
    """Strided scene windows that never cross resets or invalid rows."""

    def __init__(
        self,
        directory:        str | Path,
        team_size:        int,
        device:           torch.device | str,
        frameskip:        int,
        frame_limit:      int | None = None,
        seed:             int = 0,
        boost_pad_state:  bool = False,
        window_length:    int = 2,
        holdout_fraction: float = 0.0,
        holdout:          bool = False,
        jump_enabled:     bool = False,
        action_frameskip: int | None = None
    ) -> None:
        if frameskip < 1 or frameskip % layout.PARSED_FRAMESKIP:
            raise ValueError("discriminator spacing must match replay cadence")

        self.stride = frameskip // layout.PARSED_FRAMESKIP
        self.window_length = window_length
        self.span = self.stride * (window_length - 1)
        self.offsets = torch.arange(-self.span, 1, self.stride, device=device)
        self.boost_pad_state = boost_pad_state
        self.jump_enabled = jump_enabled
        self.action_frameskip = (
            frameskip if action_frameskip is None else action_frameskip
        )

        _, paths = replay_paths(directory, team_size)
        paths = winning_perspectives(paths, parsed_directory(directory))

        if holdout_fraction:
            train, reserved = split_expert_matches(paths, holdout_fraction)
            paths = reserved if holdout else train

        if not paths:
            raise ValueError(
                f"no known winning replay perspectives in {directory}; "
                "match results and player teams are required"
            )

        if frame_limit is not None:
            np.random.default_rng(seed).shuffle(paths)

        chunks, ends = self.load_pairs(paths, team_size, frame_limit)

        if not chunks:
            raise ValueError(f"no valid discriminator windows in {directory}")

        self.frames = torch.from_numpy(np.concatenate(chunks)).to(device)
        self.ends = torch.as_tensor(np.concatenate(ends), device=device)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def load_pairs(
        self,
        paths:       list[Path],
        team_size:   int,
        frame_limit: int | None
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        chunks, ends = [], []
        offset = 0

        for path in paths:
            remaining = frame_limit - offset if frame_limit is not None else None

            if remaining is not None and remaining <= self.span:
                break

            result = replay_pairs(
                path,
                team_size,
                self.stride,
                remaining,
                self.boost_pad_state,
                self.window_length
            )

            if result is None:
                continue

            scene, candidates = result

            if self.jump_enabled:
                jumps = load_jump_actions(path, self.action_frameskip)

                if len(jumps) < len(scene):
                    raise ValueError(f"jump action trace is too short: {path}")

                scene = np.concatenate((scene, jumps[:len(scene), None]), -1)

            chunks.append(scene)
            ends.append(candidates + offset)
            offset += len(scene)

        return chunks, ends

    def sample(self, count: int) -> torch.Tensor:
        choices = torch.randint(
            len(self.ends),
            (count,),
            device=self.frames.device,
            generator=self.generator
        )
        ends = self.ends[choices]
        indices = ends[:, None] + self.offsets

        pairs = self.frames[indices]
        if self.jump_enabled:
            # One action bit conditions the entire window: the ego command
            # during the final policy interval, not all intervening controls.
            pairs[..., -1:] = pairs[:, -1:, -1:].clone()

        return pairs

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "train_rng": self.generator.get_state()
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.generator.set_state(state["train_rng"])
