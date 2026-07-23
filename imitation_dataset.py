import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch

import subtr_actor
from carl.gymnasium import CARLTorchVectorEnv
from jarl.data import TensorBatch, TensorDataset

from replay_dataset import (
    GLOBAL_FEATURES,
    PLAYER_FEATURES,
    ReplayManifest,
    _pre_goal_mask,
    build_dataset,
)
from replay_states import (
    _load_state_tensors,
    _resolve_columns,
    _validate_and_select_frames,
)


PHYSICS_HZ = 120
SPLIT_SALT = b"goddard-replay-split-v1\0"
EXPERT_SCHEMA_VERSION = 5
EXPERT_GLOBAL_FEATURES = [
    *GLOBAL_FEATURES,
    "CurrentTime",
]


class ObservationEncoder:
    def __init__(
        self,
        batch_size: int,
    ) -> None:
        self.batch_size = batch_size
        self.environment = CARLTorchVectorEnv(
            n_sim=batch_size,
            n_blue=1,
            n_orange=1,
            synchronize=True,
            normalize=True,
        )
        self.indices = torch.arange(batch_size, device=self.environment.device)

    def close(self) -> None:
        self.environment.close()

    def encode(self, states: TensorBatch) -> np.ndarray:
        output = np.empty(
            (len(states), 2, self.environment._env.obs_dim),
            dtype=np.float32,
        )

        for start in range(0, len(states), self.batch_size):
            stop = min(start + self.batch_size, len(states))
            count = stop - start
            batch = states[start:stop]
            indices = self.indices[:count]

            self.environment.reset()
            self.environment.set_ball(
                batch["ball_position"],
                batch["ball_velocity"],
                batch["ball_angular_velocity"],
                simulation_indices=indices,
            )
            observation = self.environment.set_car(
                batch["car_position"],
                batch["car_rotation"],
                batch["car_velocity"],
                batch["car_angular_velocity"],
                batch["car_demoed"],
                boost=batch["car_boost"],
                simulation_indices=indices,
            )
            output[start:stop] = (
                observation.view(self.batch_size, 2, -1)[:count].cpu().numpy()
            )

        return output


def replay_split(
    replay_ids:   list[str],
    expert_count: int,
) -> tuple[list[str], list[str]]:
    if not 0 < expert_count < len(replay_ids):
        raise ValueError("expert count must split the replay set")

    ordered = sorted(
        replay_ids,
        key=lambda replay_id: hashlib.sha256(
            SPLIT_SALT + replay_id.encode()
        ).digest(),
    )
    return ordered[:expert_count], ordered[expert_count:]


def expert_columns() -> list[str]:
    headers = subtr_actor.get_column_headers(
        global_feature_adders=EXPERT_GLOBAL_FEATURES,
        player_feature_adders=PLAYER_FEATURES,
    )
    global_headers = list(headers["global_headers"])
    player_headers = list(headers["player_headers"])
    return global_headers + [
        f"player {index} - {header}"
        for index in range(2)
        for header in player_headers
    ]


def parse_expert_replay(
    replay_path:    Path,
    frameskip:      int,
    sequence_length: int,
    encoder:        ObservationEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    fps = PHYSICS_HZ / frameskip
    metadata, frames = subtr_actor.get_ndarray_with_info_from_replay_filepath(
        str(replay_path),
        global_feature_adders=EXPERT_GLOBAL_FEATURES,
        player_feature_adders=PLAYER_FEATURES,
        fps=fps,
        dtype="float32",
    )
    replay_metadata = metadata["replay_meta"]
    if (
        len(replay_metadata["team_zero"]) != 1
        or len(replay_metadata["team_one"]) != 1
    ):
        raise ValueError(f"Replay is not 1v1: {replay_path.name}")

    columns = expert_columns()
    replay_data = subtr_actor.get_replay_frames_data(str(replay_path))
    goal_times = [event["time"] for event in replay_data["goal_events"]]
    allowed = _pre_goal_mask(frames, columns, goal_times)
    frame_times = frames[:, columns.index("frame time")]
    precedes_goal = np.zeros(len(frames), dtype=np.bool_)
    for goal_time in goal_times:
        precedes_goal |= frame_times <= goal_time - 5.0
    allowed &= precedes_goal
    game_state = frames[:, columns.index("game state")]
    states, counts = np.unique(game_state, return_counts=True)
    live_game_state = states[counts.argmax()]
    allowed &= game_state == live_game_state
    state_columns = _resolve_columns(columns)
    valid_indices = _validate_and_select_frames(frames, state_columns)
    valid_indices = valid_indices[allowed[valid_indices]]

    demolished = frames[np.ix_(valid_indices, state_columns.car_demolished_by)]
    valid_indices = valid_indices[(demolished < 0).all(axis=1)]
    if len(valid_indices) <= sequence_length:
        return _empty_sequences(sequence_length)

    states = TensorDataset(
        TensorBatch(
            _load_state_tensors(
                frames,
                valid_indices,
                state_columns,
                encoder.environment.device,
            )
        )
    )
    observations = encoder.encode(states.data)

    current_time = frames[:, columns.index("current time")]
    pair = np.diff(valid_indices) == 1
    pair &= np.isclose(
        current_time[valid_indices[1:]] - current_time[valid_indices[:-1]],
        frameskip / PHYSICS_HZ,
        atol=2e-4,
        rtol=0,
    )
    starts = []
    run_start = 0
    for index in np.flatnonzero(~pair):
        starts.extend(range(run_start, index - sequence_length + 1, sequence_length))
        run_start = index + 1
    starts.extend(range(run_start, len(pair) - sequence_length + 1, sequence_length))
    if not starts:
        return _empty_sequences(sequence_length)

    sequence_indices = np.asarray(starts)[:, None] + np.arange(sequence_length)
    observation = observations[sequence_indices].transpose(0, 2, 1, 3)
    next_observation = observations[sequence_indices + 1].transpose(0, 2, 1, 3)
    observation = observation.reshape(-1, sequence_length, observations.shape[-1])
    next_observation = next_observation.reshape(
        -1, sequence_length, observations.shape[-1]
    )
    return observation, next_observation


def _empty_sequences(sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    empty = np.empty((0, sequence_length, 119), dtype=np.float32)
    return empty, empty.copy()


def write_expert_shard(
    path:             Path,
    observation:      np.ndarray,
    next_observation: np.ndarray,
    replay_id:        str,
    frameskip:        int,
    sequence_length:  int,
) -> None:
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(
                temporary,
                observation=observation,
                next_obs=next_observation,
                replay_id=np.asarray(replay_id),
                frameskip=np.asarray(frameskip, dtype=np.int64),
                sequence_length=np.asarray(sequence_length, dtype=np.int64),
                schema_version=np.asarray(EXPERT_SCHEMA_VERSION, dtype=np.int64),
            )
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def build_expert_dataset(
    source:          Path,
    replay_ids:      list[str],
    frameskip:       int,
    sequence_length: int,
    batch_size:      int,
) -> Path:
    root = source / "expert_dataset"
    shards = root / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    encoder = ObservationEncoder(batch_size)

    try:
        for index, replay_id in enumerate(replay_ids, start=1):
            shard = shards / f"{replay_id}.npz"
            if not _valid_expert_shard(shard, frameskip, sequence_length):
                observation, next_observation = parse_expert_replay(
                    source / "replays" / f"{replay_id}.replay",
                    frameskip,
                    sequence_length,
                    encoder,
                )
                write_expert_shard(
                    shard,
                    observation,
                    next_observation,
                    replay_id,
                    frameskip,
                    sequence_length,
                )
            print(f"Expert replay {index}/{len(replay_ids)}: {replay_id}")
    finally:
        encoder.close()

    counts = []
    for replay_id in replay_ids:
        with np.load(shards / f"{replay_id}.npz", allow_pickle=False) as shard:
            counts.append(len(shard["observation"]))

    generation = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    directory = root / generation
    directory.mkdir()
    total = sum(counts)
    observation = np.lib.format.open_memmap(
        directory / "observation.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total, sequence_length, 119),
    )
    next_obs = np.lib.format.open_memmap(
        directory / "next_obs.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total, sequence_length, 119),
    )

    offset = 0
    for replay_id, count in zip(replay_ids, counts):
        with np.load(shards / f"{replay_id}.npz", allow_pickle=False) as shard:
            stop = offset + count
            observation[offset:stop] = shard["observation"]
            next_obs[offset:stop] = shard["next_obs"]
            offset = stop
    observation.flush()
    next_obs.flush()
    del observation, next_obs

    metadata = {
        "schema_version":         EXPERT_SCHEMA_VERSION,
        "frameskip":              frameskip,
        "physics_hz":             PHYSICS_HZ,
        "sample_hz":              PHYSICS_HZ / frameskip,
        "sequence_length":        sequence_length,
        "observation_dim":        119,
        "sequence_count":         total,
        "goal_buffer_seconds":    5.0,
        "live_gameplay_only":     True,
        "demoed_states_removed": True,
        "replay_ids":             replay_ids,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / "CURRENT").write_text(generation + "\n")
    return directory


def _valid_expert_shard(
    path: Path,
    frameskip: int,
    sequence_length: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as shard:
            return (
                int(shard["schema_version"]) == EXPERT_SCHEMA_VERSION
                and int(shard["frameskip"]) == frameskip
                and int(shard["sequence_length"]) == sequence_length
                and shard["observation"].dtype == np.float32
                and shard["observation"].shape == shard["next_obs"].shape
                and shard["observation"].shape[1:] == (sequence_length, 119)
            )
    except (KeyError, OSError, ValueError):
        return False


def load_expert_dataset(
    root:            Path,
    frameskip:       int,
    sequence_length: int,
) -> TensorDataset:
    generation = (root / "CURRENT").read_text().strip()
    directory = root / generation
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("frameskip") != frameskip:
        raise ValueError("Expert dataset frameskip does not match the environment")
    if metadata.get("schema_version") != EXPERT_SCHEMA_VERSION:
        raise ValueError("Expert dataset schema is out of date; rebuild it")
    if metadata.get("sequence_length") != sequence_length:
        raise ValueError("Expert dataset sequence length does not match training")
    observation = torch.from_numpy(
        np.load(directory / "observation.npy", mmap_mode="c", allow_pickle=False)
    )
    next_obs = torch.from_numpy(
        np.load(directory / "next_obs.npy", mmap_mode="c", allow_pickle=False)
    )
    return TensorDataset(
        TensorBatch(
            {
                "observation": observation,
                "next_obs":    next_obs,
            }
        )
    )


def build_split(
    source:          Path,
    expert_count:    int,
    frameskip:       int,
    sequence_length: int,
    batch_size:      int,
) -> None:
    manifest = ReplayManifest(source / "manifest.sqlite3")
    try:
        replay_ids = manifest.parsed_replay_ids()
    finally:
        manifest.close()
    expert_ids, reset_ids = replay_split(replay_ids, expert_count)
    split = {
        "salt":        SPLIT_SALT.decode(errors="ignore").rstrip("\0"),
        "frameskip":   frameskip,
        "sequence_length": sequence_length,
        "expert_ids":  expert_ids,
        "reset_ids":   reset_ids,
    }
    (source / "split.json").write_text(json.dumps(split, indent=2) + "\n")

    build_dataset(source, reset_ids, dataset_name="reset_dataset")
    directory = build_expert_dataset(
        source, expert_ids, frameskip, sequence_length, batch_size
    )
    print(f"Reset replays: {len(reset_ids)}")
    print(f"Expert replays: {len(expert_ids)}")
    print(f"Expert dataset: {directory}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build disjoint replay reset and GAIfO datasets"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/ballchasing-ssl-1v1"),
    )
    parser.add_argument("--expert-count", type=int, default=512)
    parser.add_argument("--frameskip", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if min(
        arguments.expert_count,
        arguments.frameskip,
        arguments.sequence_length,
        arguments.batch_size,
    ) < 1:
        raise ValueError("counts, frameskip, and sequence length must be positive")
    build_split(
        arguments.source,
        arguments.expert_count,
        arguments.frameskip,
        arguments.sequence_length,
        arguments.batch_size,
    )


if __name__ == "__main__":
    main()
