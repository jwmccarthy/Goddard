import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from replays import layout
from replays.dataset import ExpertDataset, stationary_kickoff_frames


def kickoff_rows(count, team_size=1):
    rows = np.zeros((count, layout.team_replay_row_size(team_size)), dtype=np.float32)
    rows[:, 2] = 92.75 / 2076
    cars = rows[:, 9:layout.team_scene_size(team_size)].reshape(count, 2 * team_size, 21)
    for i in range(2 * team_size):
        sign = -1 if i < team_size else 1
        cars[:, i, :3] = np.array([[2048, 2560, 17], [256, 3840, 17], [0, 4608, 17]][i % team_size]) / [4108, 6000, 2076]
        cars[:, i, 1] *= sign
        cars[:, i, 10] = -sign
        cars[:, i, 14] = 1
    return rows


class ExpertKickoffTests(unittest.TestCase):
    def test_countdown_noise_and_active_play_for_all_team_sizes(self):
        for team in (1, 2, 3):
            scene = kickoff_rows(8, team)[:, :layout.team_scene_size(team)]
            scene[1, 14] = 7.42 / 2300  # Reconstructed suspension velocity.
            scene[1, 15] = .011 / 5.5
            scene[2, 33] = 43 / 2300  # Only the other car has started moving.
            scene[3, 0] = 100 / 4108  # Stationary play away from center.
            scene[4, 9] = 100 / 4108  # A car away from the kickoff spawns.
            scene[5, 3] = 100 / 6000  # Ball moving.
            scene[6, 15] = 1 / 5.5  # Car rotating.
            scene[7, 11] = 100 / 2076  # Car airborne.
            np.testing.assert_array_equal(stationary_kickoff_frames(scene, 2 * team),
                                          [True, True, False, False, False, False, False, False])

    def test_windows_and_phase_pairs_exclude_countdown_even_at_skipped_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = kickoff_rows(80)
            rows[:, 12] = 100 / 2300  # Moving kickoff with ball still at center.
            rows[:8, 12] = 0
            rows[25, 12] = 0  # Interior bad frame, including skipped cadence phases.
            for name in ("one", "two"):
                np.save(Path(directory) / f"{name}.npy", rows)
            dataset = ExpertDataset(directory, 1, "cpu", frameskip=12, window_length=3,
                                    heldout_replays=1, phase_ticks=12)
            expected = list(range(14, 25)) + list(range(32, 80))
            for ends, lengths in ((dataset.train_ends, dataset.train_lengths),
                                  (dataset.heldout_ends, dataset.heldout_lengths)):
                self.assertEqual((ends % 80).tolist(), expected)
                self.assertTrue((lengths == 3).all())
            for windows in (dataset.sample(256), dataset.sample(256, heldout=True),
                            dataset.sample_pairs(256).flatten(0, 1)):
                frames = windows.flatten(0, 1)
                frames = frames[torch.isfinite(frames).all(-1)].numpy()
                self.assertFalse(stationary_kickoff_frames(frames, 2).any())
            np.testing.assert_array_equal(np.load(Path(directory) / "one.npy"), rows)

    def test_all_countdown_corpus_has_no_valid_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            np.save(Path(directory) / "one.npy", kickoff_rows(40))
            with self.assertRaisesRegex(ValueError, "needs valid windows"):
                ExpertDataset(directory, 1, "cpu", window_length=3)


if __name__ == "__main__":
    unittest.main()
