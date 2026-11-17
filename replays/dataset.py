import json
from pathlib import Path

import numpy as np
import torch

from replays import layout


def stationary_kickoff_frames(scene: np.ndarray, n_cars: int) -> np.ndarray:
    """Identify frozen countdown poses, retaining active kickoff approaches."""
    position_scale = np.array([4108.0, 6000.0, 2076.0])
    ball_position = scene[:, :3] * position_scale
    ball_still = (
        (np.abs(ball_position - [0.0, 0.0, 92.75]) <= 5).all(-1)
        & (np.square(scene[:, 3:6]).sum(-1) <= (1 / 6000) ** 2)
        & (np.square(scene[:, 6:9]).sum(-1) <= (0.05 / 6) ** 2)
    )

    cars = scene[:, layout.BALL_SIZE :].reshape(-1, n_cars, layout.CAR_SIZE)
    positions = cars[..., :3] * position_scale
    spawns = np.array([[2048.0, 2560.0], [256.0, 3840.0], [0.0, 4608.0]])
    at_spawn = (np.abs(np.abs(positions[..., None, :2]) - spawns) <= 5).all(-1).any(-1)
    cars_still = (
        at_spawn
        & (np.abs(positions[..., 2] - 17) <= 5)
        & (np.square(cars[..., 3:6]).sum(-1) <= (10 / 2300) ** 2)
        & (np.square(cars[..., 6:9]).sum(-1) <= (0.05 / 5.5) ** 2)
    )

    return ball_still & cars_still.all(-1)


def replay_pairs(
    path:      Path,
    team_size: int,
    stride:    int,
    limit:     int | None
) -> tuple[np.ndarray, np.ndarray] | None:
    rows = np.load(path, mmap_mode="r", allow_pickle=False)

    if rows.ndim != 2 or rows.shape[1] != layout.team_replay_row_size(team_size):
        return None

    metadata = path.with_suffix(".json")
    meta = json.loads(metadata.read_text()) if metadata.is_file() else {}

    if (
        meta.get("frameskip", layout.PARSED_FRAMESKIP)
        != layout.PARSED_FRAMESKIP
    ):
        raise ValueError(f"parsed replay cadence differs: {path}")

    count = min(len(rows), limit) if limit is not None else len(rows)

    if count <= stride:
        return None

    scene = np.array(rows[:count, : layout.team_scene_size(team_size)], dtype=np.float32)
    bad = ~np.isfinite(scene).all(-1) | (rows[:count, -2:] > 0.5).any(-1)
    bad |= stationary_kickoff_frames(scene, layout.team_car_count(team_size))

    if meta.get("goal_terminal", False) and count == len(rows):
        bad[-1] = True

    candidates = np.arange(stride, count)
    cumulative = np.concatenate(([0], np.cumsum(bad)))
    valid = cumulative[candidates + 1] == cumulative[candidates - stride]

    return (scene, candidates[valid]) if valid.any() else None


class ExpertDataset:
    """Strided scene pairs that never cross resets or invalid rows."""

    def __init__(
        self,
        directory:   str | Path,
        team_size:   int,
        device:      torch.device | str,
        frameskip:   int,
        frame_limit: int | None = None,
        seed:        int = 0
    ) -> None:
        directory = Path(directory)

        if (directory / "parsed").is_dir():
            directory /= "parsed"

        self.stride = frameskip // layout.PARSED_FRAMESKIP
        paths = sorted(directory.rglob("*.npy"))

        if frame_limit is not None:
            np.random.default_rng(seed).shuffle(paths)

        chunks, ends = self.load_pairs(paths, team_size, frame_limit)
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

            if remaining is not None and remaining <= self.stride:
                break

            result = replay_pairs(path, team_size, self.stride, remaining)

            if result is None:
                continue

            scene, candidates = result
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
        indices = torch.stack((ends - self.stride, ends), 1)

        return self.frames[indices]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "train_rng": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.generator.set_state(state["train_rng"])
