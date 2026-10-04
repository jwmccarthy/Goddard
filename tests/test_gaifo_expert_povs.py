"""Only stored replay POVs may supply expert scenes to GAIFO."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch as th

from gaifo import (
    BLUE_START, ConfidentExpertResetTransform, ExpertSceneDataset,
    ORANGE_START, SCENE_SIZE, nearest_ball_distance, opponent_view,
)


class BallXDiscriminator(th.nn.Module):
    def forward(self, windows: th.Tensor) -> th.Tensor:
        return 50.0 * windows[:, -1, 0]


def write_pov(folder: Path, name: str, ball_x: float, paired: bool) -> None:
    rows = np.zeros((32, 161), dtype=np.float32)
    rows[:, 0] = ball_x
    rows[:, BLUE_START] = -0.8  # The stored focal car is off-ball.
    rows[:, ORANGE_START] = ball_x + 0.01  # The opponent is near the ball.
    np.save(folder / f"100-0-{name}.npy", rows)
    if paired:
        other = rows.copy()
        other[:, :SCENE_SIZE] = opponent_view(th.from_numpy(rows[:, :SCENE_SIZE])).numpy()
        np.save(folder / f"200-0-{name}.npy", other)


class ExpertPOVTests(unittest.TestCase):
    def test_sampling_and_heldout_use_only_stored_focals(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_pov(folder, "single", 0.2, paired=False)
            write_pov(folder, "paired", 0.4, paired=True)
            expert = ExpertSceneDataset(folder, trajectory_length=3)

            scenes = expert.sample(512, "cpu")[:, -1, 0]
            self.assertTrue(th.isclose(scenes, th.tensor(0.2)).any())
            self.assertFalse(th.isclose(scenes, th.tensor(-0.2)).any())
            for ball_x in (0.4, -0.4):
                self.assertTrue(th.isclose(scenes, th.tensor(ball_x)).any())

            heldout = ExpertSceneDataset(folder, trajectory_length=3, heldout_size=8)
            for starts, sampled in (
                (heldout.train_window_starts, heldout.sample(256, "cpu")),
                (heldout.heldout_window_starts, heldout.sample_heldout(256, "cpu")),
            ):
                ball_x = heldout.frames[starts[0], 0].item()
                sampled_x = sampled[:, -1, 0]
                self.assertTrue(th.isclose(sampled_x.abs(), th.tensor(ball_x)).all())
                if np.isclose(ball_x, 0.2):
                    self.assertTrue((sampled_x > 0).all())
                else:
                    self.assertTrue((sampled_x > 0).any())
                    self.assertTrue((sampled_x < 0).any())

    def test_near_ball_and_reset_mining_exclude_unstored_opponent(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_pov(folder, "single", 0.2, paired=False)
            write_pov(folder, "paired", 0.4, paired=True)
            expert = ExpertSceneDataset(folder, trajectory_length=3)

            self.assertEqual(expert.near_total, 32)
            near = expert.sample_near(32, "cpu")
            th.testing.assert_close(near[:, -1, 0], th.full((32,), -0.4))
            self.assertTrue((nearest_ball_distance(near) < 100).all())

            starts = th.stack([segment[0] for segment in expert.segment_window_starts])
            miner = ConfidentExpertResetTransform(
                expert, expert.reset_dataset(), BallXDiscriminator(), microbatch_size=1,
            )
            scores = miner._score(starts)
            self.assertTrue((scores[expert.opponent_pov_available[starts]] > 0.99).all())
            self.assertTrue((scores[~expert.opponent_pov_available[starts]] < 0.01).all())

            with tempfile.TemporaryDirectory(dir="/tmp/opencode") as single_directory:
                single = Path(single_directory)
                write_pov(single, "single", 0.2, paired=False)
                unpaired = ExpertSceneDataset(single, trajectory_length=3)
                self.assertEqual(unpaired.near_total, 0)
                with self.assertRaisesRegex(ValueError, "no near-ball expert windows"):
                    unpaired.sample_near(1, "cpu")


if __name__ == "__main__":
    unittest.main()
