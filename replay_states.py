import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from jarl.data import TensorBatch, TensorDataset

from replay_dataset import replay_columns


CHUNK_SIZE = 131_072


@dataclass(frozen=True)
class ReplayStateColumns:
    ball_position:         tuple[int, ...]
    ball_velocity:         tuple[int, ...]
    ball_angular_velocity: tuple[int, ...]
    car_position:          tuple[int, ...]
    car_rotation:          tuple[int, ...]
    car_velocity:          tuple[int, ...]
    car_angular_velocity:  tuple[int, ...]
    car_boost:             tuple[int, ...]
    car_demolished_by:     tuple[int, ...]


def load_replay_dataset(
    dataset_root: Path,
    device:       str | torch.device,
) -> TensorDataset:
    metadata, frames = _load_active_generation(dataset_root)
    columns = _resolve_columns(metadata["columns"])
    valid_indices = _validate_and_select_frames(frames, columns)
    data = _load_state_tensors(frames, valid_indices, columns, device)
    return TensorDataset(TensorBatch(data))


def _load_active_generation(
    dataset_root: Path,
) -> tuple[dict, np.ndarray]:
    generation = _read_current_generation(dataset_root)
    directory = dataset_root / generation
    metadata_path = directory / "metadata.json"
    frames_path = directory / "frames.npy"
    if not metadata_path.is_file() or not frames_path.is_file():
        raise ValueError(f"Replay dataset generation is incomplete: {generation}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_columns = replay_columns()
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported replay dataset schema")
    if metadata.get("columns") != expected_columns:
        raise ValueError("Replay dataset columns do not match the parser schema")

    frames = np.load(frames_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (metadata.get("frame_count"), len(expected_columns))
    if frames.dtype != np.float32 or frames.ndim != 2:
        raise ValueError("Replay frames must be a two-dimensional float32 array")
    if frames.shape != expected_shape:
        raise ValueError(
            f"Replay frame shape {frames.shape} does not match {expected_shape}"
        )

    for start in range(0, len(frames), CHUNK_SIZE):
        if not np.isfinite(frames[start : start + CHUNK_SIZE]).all():
            raise ValueError("Replay frames contain non-finite values")

    return metadata, frames


def _read_current_generation(dataset_root: Path) -> str:
    current_path = dataset_root / "CURRENT"
    if not current_path.is_file():
        raise ValueError(f"Replay dataset has no CURRENT generation: {dataset_root}")

    generation = current_path.read_text(encoding="ascii").strip()
    if not generation or Path(generation).name != generation:
        raise ValueError("Replay dataset CURRENT generation is invalid")
    return generation


def _resolve_columns(columns: list[str]) -> ReplayStateColumns:
    index = {name: position for position, name in enumerate(columns)}
    return ReplayStateColumns(
        ball_position=_field_indices(index, "Ball", "position", 3),
        ball_velocity=_field_indices(index, "Ball", "linear velocity", 3),
        ball_angular_velocity=_field_indices(
            index, "Ball", "angular velocity", 3
        ),
        car_position=_player_field_indices(index, "position", 3),
        car_rotation=_player_field_indices(index, "quaternion", 4),
        car_velocity=_player_field_indices(index, "linear velocity", 3),
        car_angular_velocity=_player_field_indices(index, "angular velocity", 3),
        car_boost=tuple(
            index[f"player {player} - boost level (raw replay units)"]
            for player in range(2)
        ),
        car_demolished_by=tuple(
            index[f"player {player} - player demolished by"]
            for player in range(2)
        ),
    )


def _validate_and_select_frames(
    frames:  np.ndarray,
    columns: ReplayStateColumns,
) -> np.ndarray:
    valid_chunks = []

    for start in range(0, len(frames), CHUNK_SIZE):
        chunk = frames[start : start + CHUNK_SIZE]
        raw_boost = chunk[:, columns.car_boost]
        if (raw_boost < 0).any() or (raw_boost > 255).any():
            raise ValueError("Replay boost values must be between 0 and 255")

        demolished_by = chunk[:, columns.car_demolished_by]
        demolition_valid = np.isin(demolished_by[:, 0], (-1, 1))
        demolition_valid &= np.isin(demolished_by[:, 1], (-1, 0))
        if not demolition_valid.all():
            raise ValueError("Replay demolition values are invalid")

        valid = chunk[:, columns.ball_position[2]] > 0
        valid &= (demolished_by[:, 0] >= 0) | (
            chunk[:, columns.car_position[2]] > 0
        )
        valid &= (demolished_by[:, 1] >= 0) | (
            chunk[:, columns.car_position[5]] > 0
        )
        local_indices = np.flatnonzero(valid)
        if not len(local_indices):
            continue

        rotations = chunk[np.ix_(local_indices, columns.car_rotation)].reshape(
            -1, 2, 4
        )
        rotation_norm = np.linalg.vector_norm(rotations, axis=-1)
        if not np.allclose(rotation_norm, 1.0, atol=1e-3):
            raise ValueError("Replay car quaternions must have unit length")

        valid_chunks.append(local_indices + start)

    if not valid_chunks:
        raise ValueError("Replay dataset contains no valid states")
    return np.concatenate(valid_chunks)


def _load_state_tensors(
    frames:        np.ndarray,
    valid_indices: np.ndarray,
    columns:       ReplayStateColumns,
    device:        str | torch.device,
) -> dict[str, torch.Tensor]:
    count = len(valid_indices)
    ball_position = _extract_tensor(
        frames, valid_indices, columns.ball_position, (count, 3), device
    )
    ball_velocity = _extract_tensor(
        frames, valid_indices, columns.ball_velocity, (count, 3), device
    )
    ball_angular_velocity = _extract_tensor(
        frames,
        valid_indices,
        columns.ball_angular_velocity,
        (count, 3),
        device,
    )
    car_position = _extract_tensor(
        frames, valid_indices, columns.car_position, (count, 2, 3), device
    )
    car_rotation = _extract_tensor(
        frames, valid_indices, columns.car_rotation, (count, 2, 4), device
    )
    car_velocity = _extract_tensor(
        frames, valid_indices, columns.car_velocity, (count, 2, 3), device
    )
    car_angular_velocity = _extract_tensor(
        frames,
        valid_indices,
        columns.car_angular_velocity,
        (count, 2, 3),
        device,
    )
    car_demoed = _extract_tensor(
        frames, valid_indices, columns.car_demolished_by, (count, 2), device
    ).ge(0)
    car_boost = _extract_tensor(
        frames, valid_indices, columns.car_boost, (count, 2), device
    ).mul_(100.0 / 255.0)

    data = {
        "ball_position":         ball_position,
        "ball_velocity":         ball_velocity,
        "ball_angular_velocity": ball_angular_velocity,
        "car_position":          car_position,
        "car_rotation":          car_rotation,
        "car_velocity":          car_velocity,
        "car_angular_velocity":  car_angular_velocity,
        "car_demoed":            car_demoed,
        "car_boost":             car_boost,
    }
    return data


def _extract_tensor(
    frames:         np.ndarray,
    valid_indices:  np.ndarray,
    column_indices: tuple[int, ...],
    shape:          tuple[int, ...],
    device:         str | torch.device,
) -> torch.Tensor:
    output = torch.empty(shape, dtype=torch.float32, device=device)

    for start in range(0, len(valid_indices), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(valid_indices))
        rows = valid_indices[start:stop]
        values = frames[np.ix_(rows, column_indices)]
        values = np.ascontiguousarray(values).reshape((stop - start, *shape[1:]))
        output[start:stop].copy_(torch.from_numpy(values))

    return output


def _field_indices(
    index:  dict[str, int],
    entity: str,
    field:  str,
    width:  int,
) -> tuple[int, ...]:
    axes = ("x", "y", "z", "w")[:width]
    return tuple(index[f"{entity} - {field} {axis}"] for axis in axes)


def _player_field_indices(
    index: dict[str, int],
    field: str,
    width: int,
) -> tuple[int, ...]:
    return tuple(
        position
        for player in range(2)
        for position in _field_indices(index, f"player {player}", field, width)
    )
