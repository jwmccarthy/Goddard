import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from ballchasing_replays.parse_replays import _parse
from replay_resets import load_demonstration_reset_dataset


def _observation() -> np.ndarray:
    observation = np.zeros(159, dtype=np.float32)
    observation[2] = 100 / 2076
    for start in (9, 30):
        observation[start + 2] = 20 / 2076
        observation[start + 9] = 1
        observation[start + 14] = 1
        observation[start + 16] = 1
    return observation


class ReplayResetTest(unittest.TestCase):
    def test_parser_and_loader_filter_goal_period_but_preserve_no_goal_period(self):
        cars = {
            "blue": SimpleNamespace(team_num=0),
            "orange": SimpleNamespace(team_num=1),
        }

        def samples(seconds):
            return [
                (
                    SimpleNamespace(
                        state=SimpleNamespace(tick_count=second * 120, cars=cars),
                        actions={car: np.zeros(8) for car in cars},
                    ),
                    {},
                )
                for second in seconds
            ]

        replay = SimpleNamespace(
            game_df={"time": SimpleNamespace(to_numpy=lambda: np.arange(16))},
            analyzer={"gameplay_periods": [
                {"start_frame": 0, "goal_frame": 10, "end_frame": 10},
                {"start_frame": 11, "end_frame": 15},
            ]},
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "ballchasing_replays.parse_replays._get_active_frames",
                return_value=[samples(range(10)), samples(range(11, 15))],
            ), patch(
                "ballchasing_replays.parse_replays._build_observation",
                return_value=_observation(),
            ):
                self.assertEqual(_parse(replay, "replay", root, frame_skip=120), 4)

            with np.load(root / "blue-0-replay.unsafe-starts.npz") as goal:
                np.testing.assert_array_equal(
                    goal["pre_goal"], [False] * 6 + [True] * 4
                )
            with np.load(root / "blue-1-replay.unsafe-starts.npz") as no_goal:
                self.assertFalse(no_goal["pre_goal"].any())

            dataset = load_demonstration_reset_dataset(root, "cpu", frame_skip=120)
            # Each car's goal period keeps six starts; the no-goal period keeps all four.
            self.assertEqual(len(dataset), 2 * (6 + 4))

    def test_existing_replays_use_recorded_sampling_skip_for_five_second_tail(self):
        rows = np.zeros((155, 161), dtype=np.float32)
        rows[:, 0] = np.arange(155) / 4108
        rows[:, :159] += _observation()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "replay.npy"
            np.save(path, rows)
            np.savez_compressed(
                path.with_suffix(".unsafe-starts.npz"),
                unsafe=np.zeros(len(rows), dtype=bool),
                frame_skip=4,
            )

            dataset = load_demonstration_reset_dataset(
                root, "cpu", frame_skip=8, require_frame_skip_match=False
            )

        self.assertEqual(len(dataset), 5)
        torch.testing.assert_close(
            dataset[torch.arange(5)]["ball_position"][:, 0],
            torch.arange(5, dtype=torch.float32),
        )

    def test_missing_sidecar_uses_parser_completion_marker(self):
        rows = np.zeros((155, 161), dtype=np.float32)
        rows[:, :159] = _observation()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "blue-0-replay.npy", rows)
            (root / ".replay.v6-fs4.complete").touch()

            dataset = load_demonstration_reset_dataset(root, "cpu", frame_skip=8)

        self.assertEqual(len(dataset), 5)


if __name__ == "__main__":
    unittest.main()
