"""Kickoff scenes must be trainable without looking across replay or reset boundaries."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th

from gaifo import (
    BLUE_START, DRIVING_SKILL, KICKOFF_SKILL, ORANGE_START, POSITION_SCALE,
    ConfidentExpertResetTransform,
    ExpertSceneDataset, SceneWindowCapture, opponent_view,
)


def save_period(folder: Path, name: str, marker: int, paired: bool = False) -> None:
    rows = np.zeros((8, 161), dtype=np.float32)
    rows[:, 3] = marker + np.arange(8)
    rows[:, BLUE_START + 14] = rows[:, ORANGE_START + 14] = 1
    rows[:, BLUE_START + 16] = rows[:, ORANGE_START + 16] = 1
    np.save(folder / f"100-0-{name}.npy", rows)
    if paired:
        opponent = rows.copy()
        opponent[:, :51] = opponent_view(th.from_numpy(rows[:, :51])).numpy()
        np.save(folder / f"200-0-{name}.npy", opponent)


class KickoffRetentionTests(unittest.TestCase):
    def test_curated_kickoff_covers_first_challenge_and_respects_replay_safety(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            for name in ("kickoff", "nonkickoff"):
                rows = np.zeros((110, 161), dtype=np.float32)
                rows[:, 2] = 92.75 / POSITION_SCALE[2]
                for car, y in ((BLUE_START, -2_560), (ORANGE_START, 2_560)):
                    rows[:, car + 1] = y / POSITION_SCALE[1]
                    rows[:, car + 2] = 17 / POSITION_SCALE[2]
                    rows[:, car + 9] = rows[:, car + 14] = rows[:, car + 16] = 1
                rows[60:, 0] = (np.arange(60, 110) - 59) * 30 / POSITION_SCALE[0]
                rows[60:, 3] = 900 / 6_000
                if name == "nonkickoff":
                    rows[:, ORANGE_START + 1] = 0  # Not a kickoff formation.
                else:
                    rows[30, -2] = 1  # Parser discontinuity removes intersecting windows.
                path = folder / f"100-0-{name}.npy"
                np.save(path, rows)
                unsafe = np.zeros(len(rows), dtype=bool)
                unsafe[65] = True
                np.savez_compressed(
                    path.with_suffix(".unsafe-starts.npz"), unsafe=unsafe,
                    pre_goal=np.zeros(len(rows), dtype=bool), frame_skip=4,
                )
                if name == "kickoff":
                    opposite = folder / "200-0-kickoff.npy"
                    second = rows.copy()
                    second[:, :51] = opponent_view(th.from_numpy(rows[:, :51])).numpy()
                    np.save(opposite, second)
                    np.savez_compressed(
                        opposite.with_suffix(".unsafe-starts.npz"), unsafe=unsafe,
                        pre_goal=np.zeros(len(rows), dtype=bool), frame_skip=4,
                    )

            expert = ExpertSceneDataset(
                folder, 8, frame_skip=4, reject_discontinuities=True,
                skill_sampling=True,
            )
            start = int(expert.segment_window_starts[0][0])
            kickoff = expert.curated_pools()[KICKOFF_SKILL]
            self.assertEqual(set(kickoff[:, 1].tolist()), {0, 1})
            for actor in (0, 1):
                selected = kickoff[kickoff[:, 1] == actor, 0]
                self.assertEqual(set(selected.tolist()),
                                 set(range(start, start + 76)) - set(range(start + 30, start + 38)))
                self.assertFalse(th.isin(
                    selected, expert.curated_pools()[DRIVING_SKILL][
                        expert.curated_pools()[DRIVING_SKILL][:, 1] == actor, 0,
                    ],
                ).any())
            self.assertTrue(th.isin(kickoff[:, 0], expert.train_window_starts).all())
            self.assertFalse(th.isin(
                expert._curated_reset_pools[KICKOFF_SKILL],
                th.tensor([start + 65 + expert.partition_span]),
            ).any())
            self.assertGreater(len(expert.curated_pools()[DRIVING_SKILL]), 0)

            old_style = ExpertSceneDataset(
                folder, 8, frame_skip=4, reject_discontinuities=True,
                skill_sampling=True, kickoff_fraction=0,
            )
            self.assertFalse(len(old_style.curated_pools()[KICKOFF_SKILL]))
            self.assertTrue(th.isin(
                th.tensor([start]), old_style.curated_pools()[DRIVING_SKILL][:, 0],
            ).all())

            heldout = ExpertSceneDataset(
                folder, 8, heldout_size=16, frame_skip=4,
                reject_discontinuities=True, skill_sampling=True,
            )
            for split, eligible in ((False, heldout.train_window_starts),
                                    (True, heldout.heldout_window_starts)):
                pairs = heldout.curated_pools(heldout=split)[KICKOFF_SKILL]
                self.assertTrue(th.isin(pairs[:, 0], eligible).all())
                self.assertFalse((pairs[:, 1].bool()
                                  & ~heldout.opponent_pov_available[pairs[:, 0]]).any())
            self.assertFalse(th.isin(heldout.reset_indices,
                                     heldout.heldout_window_starts + heldout.partition_span).any())

            # A smaller frame skip must not truncate the kickoff before contact.
            faster = ExpertSceneDataset(
                folder, 8, frame_skip=2, reject_discontinuities=True,
                skill_sampling=True,
            )
            quick_start = int(faster.segment_window_starts[0][0])
            quick_kickoff = faster.curated_pools()[KICKOFF_SKILL]
            self.assertGreater(int(quick_kickoff[:, 0].max()) - quick_start, 140)
            self.assertLess(int(quick_kickoff[:, 0].max()) - quick_start, 160)

    def test_each_stored_pov_scores_kickoff_without_crossing_segments(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            save_period(folder, "single", 10)
            save_period(folder, "paired", 100, paired=True)
            expert = ExpertSceneDataset(folder, trajectory_length=4)
            self.assertEqual(expert.train_total, 16)  # One scored window per physical frame.
            self.assertEqual(len(expert.reset_indices), 16)  # No duplicate reset states.

            first_starts = [int(starts[0]) for starts in expert.segment_window_starts]
            self.assertEqual(len(first_starts), 2)
            for start in first_starts:
                marker = int(expert.frames[start, 3])
                kickoff = expert.frames[start + expert.window_offsets]
                th.testing.assert_close(kickoff[:, 3], th.full((4,), marker, dtype=th.float32))
                first_action = expert.frames[start + 1 + expert.window_offsets]
                th.testing.assert_close(first_action[:, 3], th.tensor([
                    marker, marker, marker, marker + 1,
                ], dtype=th.float32))
                self.assertIn(start + 3, expert.reset_indices.tolist())
                self.assertNotIn(start, expert.reset_indices.tolist())

            starts = th.cat(expert.situation_pools())
            self.assertFalse((starts[:, 1].bool() &
                              ~expert.opponent_pov_available[starts[:, 0]]).any())
            self.assertEqual(int((starts[:, 0] == first_starts[0]).sum()),
                             2 if expert.opponent_pov_available[first_starts[0]] else 1)

            miner = ConfidentExpertResetTransform(
                expert, expert.reset_dataset(), th.nn.Identity(), microbatch_size=4,
            )
            self.assertTrue(th.isin(miner.candidate_starts, expert.reset_indices).all())
            self.assertTrue(set(start + 3 for start in first_starts).issubset(
                miner.candidate_starts.tolist(),
            ))

            heldout = ExpertSceneDataset(folder, trajectory_length=4, heldout_size=4)
            for selected in (heldout.train_window_starts, heldout.heldout_window_starts):
                self.assertEqual(len(selected), 8)
                window = heldout.frames[selected[0] + heldout.window_offsets]
                self.assertTrue((window[:, 3] == window[0, 3]).all())

    def test_generated_first_action_uses_its_own_kickoff_context_after_reset(self):
        capture = SceneWindowCapture(4)
        capture.reset(batch_size=2)

        def capture_step(current: int, following: int, done: bool = False):
            observation = th.zeros(2, 51)
            observation[:, 3] = current
            next_obs = observation.clone()
            next_obs[:, 3] = following
            return capture._capture(SimpleNamespace(
                observation=observation,
                env_step=SimpleNamespace(
                    next_obs=next_obs,
                    done=th.tensor([done, False]),
                ),
            ))

        first = capture_step(0, 1)
        self.assertTrue(first["scene_window_valid"].all())
        th.testing.assert_close(first["scene_window"][:, :, 3],
                                th.tensor([[0., 0., 0., 1.]]).expand(2, -1))
        second = capture_step(1, 2, done=True)
        th.testing.assert_close(second["scene_window"][:, :, 3],
                                th.tensor([[0., 0., 1., 2.]]).expand(2, -1))
        after_reset = capture_step(10, 11)
        self.assertTrue(after_reset["scene_window_valid"].all())
        th.testing.assert_close(after_reset["scene_window"][:, :, 3],
                                th.tensor([[10., 10., 10., 11.]]).expand(2, -1))


if __name__ == "__main__":
    unittest.main()
