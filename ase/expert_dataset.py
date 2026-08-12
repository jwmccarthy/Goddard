from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F

from jarl.data.batch import TensorBatch

from config import REPLAY_LAYOUT, REPLAY_STATE_SIZE


BALL_MAX_SPEED     = 6000.0
BALL_MAX_ANG_SPEED = 6.0
CAR_MAX_SPEED      = 2300.0
CAR_MAX_ANG_SPEED  = 5.5
BOOST_MAX          = 100.0
POSITION_SCALE     = (4108.0, 6000.0, 2076.0)


def _forward_up_to_quaternion(
    forward: th.Tensor,
    up:      th.Tensor
) -> th.Tensor:
    forward = F.normalize(forward, dim=-1)
    up = up - (up * forward).sum(-1, keepdim=True) * forward
    up = F.normalize(up, dim=-1)
    right = th.linalg.cross(up, forward, dim=-1)
    rotation = th.stack((forward, right, up), dim=-1)

    m00 = rotation[..., 0, 0]
    m01 = rotation[..., 0, 1]
    m02 = rotation[..., 0, 2]
    m10 = rotation[..., 1, 0]
    m11 = rotation[..., 1, 1]
    m12 = rotation[..., 1, 2]
    m20 = rotation[..., 2, 0]
    m21 = rotation[..., 2, 1]
    m22 = rotation[..., 2, 2]

    x = th.copysign(
        (1 + m00 - m11 - m22).clamp_min(0).sqrt(),
        m21 - m12
    )
    y = th.copysign(
        (1 - m00 + m11 - m22).clamp_min(0).sqrt(),
        m02 - m20
    )
    z = th.copysign(
        (1 - m00 - m11 + m22).clamp_min(0).sqrt(),
        m10 - m01
    )
    w = (1 + m00 + m11 + m22).clamp_min(0).sqrt()

    quaternion = th.stack((x, y, z, w), dim=-1)

    return F.normalize(quaternion, dim=-1)


class ExpertTrajectoryDataset:

    @staticmethod
    def _transform(replay: np.ndarray) -> np.ndarray:
        if replay.ndim != 2 or replay.shape[1] != REPLAY_STATE_SIZE:
            raise ValueError(
                f"expected replay shaped [frames, {REPLAY_STATE_SIZE}], "
                f"got {tuple(replay.shape)}"
            )

        ball = replay[:, REPLAY_LAYOUT.ball]
        ego = replay[:, REPLAY_LAYOUT.ego]
        other_position = replay[:, REPLAY_LAYOUT.other_position]
        other_velocity = replay[:, REPLAY_LAYOUT.other_velocity]

        relative = np.concatenate(
            (
                (ball[:, 0:3] - ego[:, 0:3]) / 2.0,
                (
                    ball[:, 3:6] * BALL_MAX_SPEED
                    - ego[:, 3:6] * CAR_MAX_SPEED
                ) / (BALL_MAX_SPEED + CAR_MAX_SPEED),
                (other_position - ego[:, 0:3]) / 2.0,
                (other_velocity - ego[:, 3:6]) / 2.0
            ),
            axis=-1
        )

        return np.concatenate((replay, relative), axis=-1)

    def __init__(
        self,
        replay_dir: str,
        obs_limit:  int = 16_000_000,
        seq_len:    int = 2,
        device:     str | th.device = "cpu"
    ) -> None:
        if obs_limit < 1:
            raise ValueError("obs_limit must be positive")
        if seq_len < 1:
            raise ValueError("seq_len must be positive")

        self.replays = []
        self.seq_len = seq_len
        self.device = th.device(device)

        total_obs = 0

        for path in sorted(Path(replay_dir).glob("*.npy")):
            replay = np.load(path, mmap_mode="r")
            replay = replay[:obs_limit - total_obs]
            replay = self._transform(replay)

            if len(replay) >= seq_len:
                self.replays.append(replay)

            if (total_obs := total_obs + len(replay)) >= obs_limit:
                break

        if not self.replays:
            raise ValueError(f"no usable observation files found in {replay_dir}")

    def __len__(self) -> int:
        return sum(len(replay) - self.seq_len + 1 for replay in self.replays)

    def sample(self, count: int, limit: int = None) -> TensorBatch:
        if count < 1:
            raise ValueError("sample count must be positive")

        sequences = []
        seq_len = self.seq_len if limit is None else limit

        for _ in range(count):
            replay = self.replays[np.random.randint(len(self.replays))]
            start = np.random.randint(len(replay) - self.seq_len + 1)
            sequences.append(replay[start : start + seq_len])

        batch = th.from_numpy(np.stack(sequences)).to(self.device)

        return TensorBatch({"observation": batch})


class ExpertStateResetProvider:

    def __init__(
        self,
        demos:  ExpertTrajectoryDataset,
        device: th.device = "cuda:0"
    ) -> None:
        self._demos = demos
        self._device = th.device(device)
        self._position_scale = th.tensor(
            POSITION_SCALE,
            dtype=th.float32,
            device=self._device
        )

    def __call__(self, reset_mask: th.Tensor) -> TensorBatch | None:
        env_idx = reset_mask.nonzero(as_tuple=True)[0]
        if not (count := env_idx.numel()):
            return

        states = self._demos.sample(count, 1)["observation"][:, 0]
        states = states[:, :REPLAY_STATE_SIZE].to(self._device)

        ball = states[:, 0:9]
        cars = states[:, 9:43].reshape(count, 2, 17)

        ball_position = (ball[..., 0:3] * self._position_scale).contiguous()
        ball_velocity = (ball[..., 3:6] * BALL_MAX_SPEED).contiguous()
        ball_angular_velocity = (
            ball[..., 6:9] * BALL_MAX_ANG_SPEED
        ).contiguous()

        car_position = (
            cars[..., 0:3] * self._position_scale
        ).contiguous()
        
        car_velocity = (cars[..., 3:6] * CAR_MAX_SPEED).contiguous()

        car_angular_velocity = (
            cars[..., 6:9] * CAR_MAX_ANG_SPEED
        ).contiguous()

        car_rotation = _forward_up_to_quaternion(
            cars[..., 9:12],
            cars[..., 12:15]
        ).contiguous()

        car_boost = (cars[..., 15] * BOOST_MAX).contiguous()
        car_demoed = cars[..., 16].bool().contiguous()

        score = th.zeros(count, dtype=th.int32, device=self._device)
        ticks = th.zeros(count, dtype=th.int32, device=self._device)

        return TensorBatch({
            "simulation_indices":    env_idx,
            "ball_position":         ball_position,
            "ball_velocity":         ball_velocity,
            "ball_angular_velocity": ball_angular_velocity,
            "car_position":          car_position,
            "car_rotation":          car_rotation,
            "car_velocity":          car_velocity,
            "car_angular_velocity":  car_angular_velocity,
            "car_demoed":            car_demoed,
            "car_boost":             car_boost,
            "blue_score":            score,
            "orange_score":          score.clone(),
            "episode_ticks":         ticks
        })
