"""Replay reset loading and typed CARL requests."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th

from basic import SyntheticMatchResetProvider
from carl.gymnasium import CARLMatchReset, CARLResetState, REGULATION_TICKS
from carl.gymnasium.torch import _forward_up_to_quat
from jarl.envs import DatasetResetSampler
from replay_resets import (
    ReplayResetProvider, load_demonstration_reset_frames, reset_index_dataset,
)


def write_replay(folder: Path) -> np.ndarray:
    rows = np.zeros((16, 161), dtype=np.float32)
    rows[:, 0] = 0.25 + np.arange(16) / 100
    rows[:, 2] = 91.25 / 2076
    rows[:, 3] = 0.1
    for car in (9, 30):
        rows[:, car + 2] = 17 / 2076
        rows[:, car + 9] = 1
        rows[:, car + 14] = 1
        rows[:, car + 15] = 0.5
    rows[:, 30 + 16] = 1
    rows[:, 30 + 18] = 1
    rows[:, 30 + 19] = 1
    rows[:, 30 + 20] = 1
    rows[:, 137] = 1
    rows[:, 137 + 5] = 0.25
    rows[3, -4] = 1  # Invalid parser row.
    unsafe = np.zeros(16, dtype=bool)
    unsafe[2] = True
    np.save(folder / "replay.npy", rows)
    np.savez(
        folder / "replay.unsafe-starts.npz",
        unsafe=unsafe, pre_goal=np.zeros(16, dtype=bool), frame_skip=4,
    )
    return rows


class ReplayResetTests(unittest.TestCase):
    def test_safe_frames_remain_normalized_with_both_cars_internal_state(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = write_replay(folder)
            frames, internal = load_demonstration_reset_frames(folder, "cpu")

            eligible = np.array([index for index in range(16) if index not in (2, 3)])
            th.testing.assert_close(frames, th.from_numpy(rows[eligible, :51]))
            th.testing.assert_close(internal[:, 0], th.from_numpy(rows[eligible, 137:156]))
            th.testing.assert_close(internal[:, 1, (0, 7, 8, 17)], th.ones(14, 4))
            th.testing.assert_close(internal[:, 1, 5], th.zeros(14))

    def test_index_sampler_resolves_typed_request_on_demand(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_replay(folder)
            frames, internal = load_demonstration_reset_frames(folder, "cpu")
            dataset = reset_index_dataset(th.arange(len(frames)))
            self.assertEqual(set(dataset.data), {"frame_index"})
            mask = th.tensor([True, False, True, True])
            expected = DatasetResetSampler(dataset, seed=7)(mask)
            request = ReplayResetProvider(
                DatasetResetSampler(dataset, seed=7), frames, internal,
            )(mask)

            self.assertIsInstance(request, CARLResetState)
            self.assertTrue(request.normalized)
            th.testing.assert_close(request.simulation_indices, th.tensor([0, 2, 3]))
            selected = frames[expected["frame_index"]]
            th.testing.assert_close(request.ball, selected[:, :9])
            th.testing.assert_close(request.cars, selected[:, 9:51].view(-1, 2, 21))
            th.testing.assert_close(
                request.car_internal_state, internal[expected["frame_index"]],
            )
            ball, cars = request.physical()
            th.testing.assert_close(ball.velocity[:, 0], th.full((3,), 600.0))
            th.testing.assert_close(cars.boost, th.full((3, 2), 50.0))
            th.testing.assert_close(request.cars.boost, th.full((3, 2), 0.5))

    def test_invalid_car_rotations_are_excluded_before_reset_sampling(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = write_replay(folder)
            rows[0, 9 + 9:9 + 15] = 0  # Demoed ego without axes.
            rows[0, 9 + 17] = 1
            rows[1, 30 + 12:30 + 15] = rows[1, 30 + 9:30 + 12]  # Parallel opponent axes.
            rows[4, 9 + 9] = 1e-5  # Too short to define a forward direction.
            rows[5, 30 + 12] = np.nan
            rows[6, 9 + 12:9 + 15] = [1, 1e-5, 0]  # Almost parallel to forward.
            np.save(folder / "replay.npy", rows)

            frames, _ = load_demonstration_reset_frames(folder, "cpu")
            th.testing.assert_close(frames, th.from_numpy(rows[7:, :51]))

            limited_frames, limited_internal = load_demonstration_reset_frames(
                folder, "cpu", limit=8, seed=0,
            )
            self.assertEqual(len(limited_frames), 8)
            request = ReplayResetProvider(
                DatasetResetSampler(
                    reset_index_dataset(th.arange(len(limited_frames))), seed=0,
                ), limited_frames, limited_internal,
            )(th.ones(1_024, dtype=th.bool))
            _, cars = request.physical()
            self.assertEqual(_forward_up_to_quat(cars.forward, cars.up).shape, (1_024, 2, 4))

    def test_synthetic_match_state_and_kickoff_fallback(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_replay(folder)
            frames, internal = load_demonstration_reset_frames(folder, "cpu")
            dataset = reset_index_dataset(th.arange(len(frames)))
            mask = th.tensor([True, False, True])
            provider = SyntheticMatchResetProvider(ReplayResetProvider(
                DatasetResetSampler(dataset), frames, internal,
            ))
            request = provider(mask)
            self.assertIsInstance(request.match, CARLMatchReset)
            self.assertEqual(request.match.blue_score.dtype, th.int32)
            self.assertEqual(request.match.orange_score.dtype, th.int32)
            self.assertEqual(request.match.episode_ticks.dtype, th.int32)
            self.assertTrue((
                (request.match.episode_ticks >= 0)
                & (request.match.episode_ticks <= REGULATION_TICKS)
            ).all())
            kickoff = SyntheticMatchResetProvider(ReplayResetProvider(
                DatasetResetSampler(dataset, probability=0), frames, internal,
            ))
            self.assertIsNone(kickoff(mask))


if __name__ == "__main__":
    unittest.main()
