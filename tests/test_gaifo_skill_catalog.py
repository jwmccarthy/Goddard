"""Diagnostic skill cataloging does not affect uniform expert or reset sampling."""

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import numpy as np
import torch as th

from gaifo import (
    AERIAL_TOUCH_SKILL, DRIVING_SKILL, KICKOFF_SKILL,
    BLUE_START, ORANGE_START, POSITION_SCALE,
    ConfidentExpertResetTransform,
    ExpertSceneDataset, ReplayResetProvider, SceneGAIFOMinibatches, parse_args,
)
from carl.gymnasium import CARLTorchVectorEnv
from jarl.data import TensorBatch
from jarl.envs import DatasetResetSampler
from watch_checkpoints import load_replay_reset_provider


def _period(folder: Path, kind: str, index: int) -> None:
    rows = np.zeros((48, 161), dtype=np.float32)
    rows[:, 2] = 92 / POSITION_SCALE[2]
    rows[:, BLUE_START + 2] = rows[:, ORANGE_START + 2] = 17 / POSITION_SCALE[2]
    rows[:, BLUE_START + 9] = rows[:, ORANGE_START + 9] = 1
    rows[:, BLUE_START + 14] = rows[:, ORANGE_START + 14] = 1
    rows[:, BLUE_START + 16] = rows[:, ORANGE_START + 16] = 1
    rows[:, ORANGE_START] = 3_000 / POSITION_SCALE[0]
    rows[:, 0] = 2_000 / POSITION_SCALE[0]
    rows[:, 8] = (index + 1) / 10 + np.arange(48) / 10_000
    rows[:, 137] = 1

    if kind in ("aerial", "aerial_touch"):
        rows[8:25, BLUE_START + 16] = 0
        rows[8:25, BLUE_START + 2] = 650 / POSITION_SCALE[2]
        rows[8:12, ORANGE_START + 16] = 0
        rows[8:12, ORANGE_START + 18] = 1
        rows[8:12, ORANGE_START + 2] = 650 / POSITION_SCALE[2]
        rows[8:25, 2] = 750 / POSITION_SCALE[2]
        rows[8:25, 0] = 100 / POSITION_SCALE[0]
        if kind == "aerial":
            rows[8:25, 1] = (np.arange(8, 25) - 8) * 20 / POSITION_SCALE[1]
            rows[[10, 16, 22], 156] = 1  # Three distinct airborne touches.
        else:
            rows[10, 156] = 1
    elif kind in ("dribble", "flick"):
        for step in range(48):
            if 5 <= step < 12:
                rows[step, 2] = 180 / POSITION_SCALE[2]
                rows[step, 3] = 1_000 / 6_000
                rows[step, BLUE_START + 3] = 1_000 / 2_300
                rows[step, 0] = 0
            elif step >= 12:
                rows[step, 0] = 400 / POSITION_SCALE[0]
                rows[step, 2] = (240 if kind == "flick" else 92) / POSITION_SCALE[2]
                rows[step, 3] = (1_700 if kind == "flick" else 400) / 6_000
                rows[step, 5] = (500 if kind == "flick" else 0) / 6_000
                if kind == "flick" and step < 16:
                    rows[step, BLUE_START + 16] = 0
                if kind == "flick" and step >= 14:
                    rows[step, BLUE_START + 18] = 1
    elif kind == "kickoff":
        rows[:, 0] = rows[:, 3] = 0
        rows[:, BLUE_START + 1] = -2_560 / POSITION_SCALE[1]
        rows[:, ORANGE_START + 1] = 2_560 / POSITION_SCALE[1]
        rows[20:, 0] = (np.arange(20, 48) - 19) * 30 / POSITION_SCALE[0]
        rows[20:, 3] = 900 / 6_000  # Kickoff challenge at step 20.

    rows[:, 137] = rows[:, BLUE_START + 16]

    unsafe = np.zeros(len(rows), dtype=bool)
    pre_goal = np.zeros(len(rows), dtype=bool)
    if kind in ("aerial", "aerial_touch", "kickoff"):
        unsafe[12] = True  # Exclude the reset, not its valid touch trajectory.
        pre_goal[13] = True
    if kind == "driving":
        rows[15, -2] = 1  # Parser-marked window is not a valid expert example.
    filename = folder / f"100-0-{kind}.npy"
    np.save(filename, rows)
    np.savez_compressed(
        filename.with_suffix(".unsafe-starts.npz"),
        unsafe=unsafe, pre_goal=pre_goal, frame_skip=4,
    )


def _expert(folder: Path) -> ExpertSceneDataset:
    for index, kind in enumerate(("aerial", "aerial_touch", "dribble", "flick", "driving", "kickoff")):
        _period(folder, kind, index)
    return ExpertSceneDataset(
        folder, trajectory_length=8, reject_discontinuities=True,
        catalog_skills=True,
    )


class UniformExpertSamplingTests(unittest.TestCase):
    def test_checkpoint_viewer_resets_sample_safe_frames_uniformly(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = _expert(Path(directory))
            provider = load_replay_reset_provider(
                Path(directory), "cpu", frame_skip=4, state_limit=24,
                corpus_limit=None, seed=0,
            )
            self.assertEqual(len(provider.frames), 24)
            self.assertTrue(th.isin(
                provider.frames[:, 8], expert.frames[expert.reset_indices, 8],
            ).all())
            self.assertFalse(expert.unsafe_reset_frames[expert.reset_indices].any())
            mask = th.ones(1, dtype=th.bool)
            self.assertIsNone(provider(th.zeros(1, dtype=th.bool)))
            chosen = []
            for _ in range(2_400):
                request = provider(mask)
                chosen.append(int(provider.last_reset_indices[0]))
                self.assertEqual(request.simulation_indices.tolist(), [0])
                self.assertTrue(request.normalized)
            frequencies = th.bincount(th.tensor(chosen), minlength=24).float() / len(chosen)
            th.testing.assert_close(frequencies, th.full((24,), 1 / 24), atol=.02, rtol=0)

    def test_training_defaults_sample_all_expert_windows(self):
        with patch.object(sys, "argv", [
            "gaifo.py", "--replay-dir", "parsed_replays",
        ]):
            args, _ = parse_args()
        self.assertEqual(args.replay_reset_fraction, .70)
        self.assertFalse(hasattr(args, "curated_skill_sampling"))
        self.assertFalse(hasattr(args, "general_driving_fraction"))
        self.assertFalse(hasattr(args, "kickoff_fraction"))

    def test_legacy_checkpoint_reset_fraction_is_preserved_without_skill_sampling(self):
        checkpoint = {"config": {
            "replay_dir": "/tmp/opencode", "replay_reset_fraction": .25,
            "curated_skill_sampling": True, "general_driving_fraction": .1,
            "kickoff_fraction": .05,
        }}
        with patch("gaifo.load_resume_checkpoint", return_value=checkpoint):
            base = ["gaifo.py", "--resume-checkpoint", "/tmp/opencode/old.pt"]
            with patch.object(sys, "argv", base):
                resumed, _ = parse_args()
            self.assertEqual(resumed.replay_reset_fraction, .25)
            self.assertFalse(hasattr(resumed, "curated_skill_sampling"))
            with patch.object(sys, "argv", [
                *base, "--replay-reset-fraction", ".6",
            ]):
                explicit, _ = parse_args()
            self.assertEqual(explicit.replay_reset_fraction, .6)

    def test_cataloged_skills_do_not_filter_training_windows_or_situation_pairs(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            _period(folder, "driving", 0)
            rows = np.load(folder / "100-0-driving.npy")
            rows[20:30, BLUE_START + 13] = .98
            rows[20:30, BLUE_START + 14] = .2
            np.save(folder / "100-0-driving.npy", rows)
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, reject_discontinuities=True,
                catalog_skills=True,
            )
            classified = th.cat(expert.curated_pools())[:, 0]
            unclassified = expert.train_window_starts[
                ~th.isin(expert.train_window_starts, classified)
            ]
            self.assertGreater(len(unclassified), 0)
            with patch.object(expert, "curated_pools", side_effect=AssertionError("training catalog used")):
                povs = expert.sample_povs(8_192)
                scenes = expert.sample(128, "cpu")
                generated = scenes[:16]
                batch = next(SceneGAIFOMinibatches(
                    expert, batch_size=16, epochs=1, noise_std=0, factorize=True,
                ).sample_windows(generated, th.arange(len(generated))))
            self.assertTrue(th.isin(povs[:, 0], expert.train_window_starts).all())
            self.assertAlmostEqual(
                th.isin(povs[:, 0], unclassified).float().mean().item(),
                len(unclassified) / len(expert.train_window_starts), delta=.03,
            )
            self.assertEqual(scenes.shape, (128, 8, 51))
            self.assertNotIn("skill_category", batch)

    def test_uniform_reset_pool_includes_safe_frames_outside_cataloged_skills(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            _period(folder, "driving", 0)
            rows = np.load(folder / "100-0-driving.npy")
            # Grounded on a wall, with valid axes: safe to reset, but not a
            # curated flat-ground driving or aerial-control expert example.
            rows[20:30, BLUE_START + 13] = .98
            rows[20:30, BLUE_START + 14] = .2
            np.save(folder / "100-0-driving.npy", rows)
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, reject_discontinuities=True,
                catalog_skills=True,
            )
            classified = th.cat(expert.curated_pools())[:, 0] + expert.partition_span
            additional = expert.reset_indices[~th.isin(expert.reset_indices, classified)]
            self.assertGreater(len(additional), 0)
            self.assertFalse(expert.unsafe_reset_frames[expert.reset_indices].any())
            self.assertFalse(th.isin(
                expert.segment_frame_indices[0][15], expert.reset_indices,
            ).any())  # Parser-marked correction.

            sampled = DatasetResetSampler(expert.reset_dataset(), seed=13)(
                th.ones(10_000, dtype=th.bool),
            )["frame_index"]
            self.assertTrue(th.isin(sampled, expert.reset_indices).all())
            self.assertAlmostEqual(
                th.isin(sampled, additional).float().mean().item(),
                len(additional) / len(expert.reset_indices), delta=.03,
            )

    def test_legacy_safety_sidecar_without_cadence_or_unsafe_mask(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            _period(folder, "driving", 0)
            path = folder / "100-0-driving.npy"
            rows = np.load(path)
            rows[20, 3] = .05
            rows[21, 3] = .10  # Consecutive impulses are unsafe reset starts.
            np.save(path, rows)
            np.savez_compressed(
                path.with_suffix(".unsafe-starts.npz"),
                pre_goal=np.zeros(len(rows), dtype=bool),
            )
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, frame_skip=4,
                reject_discontinuities=True,
            )
            self.assertGreater(len(expert.reset_indices), 0)
            rejected = expert.segment_frame_indices[0][19:22]
            self.assertFalse(th.isin(rejected, expert.reset_indices).any())
            self.assertTrue(expert.unsafe_reset_frames[rejected].all())

    def test_single_pov_opponent_reset_infers_visible_ground_and_flip_flags(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = _expert(Path(directory))
            for rows in expert.segment_frame_indices:
                self.assertTrue(th.equal(
                    expert.internal_states[rows, 1, 0],
                    expert.frames[rows, ORANGE_START + 16],
                ))
                self.assertTrue(th.equal(
                    expert.internal_states[rows, 1, 8],
                    expert.frames[rows, ORANGE_START + 18],
                ))

    @unittest.skipUnless(th.cuda.is_available(), "CARL reset integration requires CUDA")
    def test_carl_resets_avoid_unrecorded_airborne_opponent_controls(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            for index, kind in enumerate(("aerial", "aerial_touch", "dribble", "flick", "driving")):
                _period(Path(directory), kind, index)
            expert = ExpertSceneDataset(
                Path(directory), trajectory_length=8, device="cuda:0",
                reject_discontinuities=True,
            )
            airborne = expert.reset_indices[
                (expert.frames[expert.reset_indices, BLUE_START + 16] < .5)
                & (expert.frames[expert.reset_indices, ORANGE_START + 16] > .5)
            ]
            self.assertTrue(len(airborne))
            ground = expert.reset_indices[
                (expert.frames[expert.reset_indices, BLUE_START + 16] > .5)
                & (expert.frames[expert.reset_indices, ORANGE_START + 16] > .5)
            ][0]

            class FixedStart:
                def __init__(self, index: th.Tensor) -> None:
                    self.index = index

                def __call__(self, mask: th.Tensor) -> TensorBatch:
                    return TensorBatch({
                        "frame_index": self.index.reshape(1),
                        "simulation_indices": th.zeros(1, dtype=th.long, device=mask.device),
                    })

            fixed = FixedStart(airborne[0])
            env = CARLTorchVectorEnv(
                n_sim=1, n_blue=1, n_orange=1, frameskip=4,
                normalize=True, discrete_actions=True,
                reset_state_provider=ReplayResetProvider(
                    fixed, expert.frames, expert.internal_states,
                ),
            )
            try:
                observations = env.reset()
                self.assertFalse(bool(observations[0, BLUE_START + 16]))
                self.assertTrue(bool(observations[0, ORANGE_START + 16]))
                self.assertTrue(bool(env.action_mask(observations)[1, 17]))
                fixed.index = ground
                observations = env.reset()
                self.assertTrue(bool(observations[0, ORANGE_START + 16]))
                self.assertTrue(bool(env.action_mask(observations)[1, 10]))
            finally:
                env.close()

    def test_opt_in_mined_resets_stay_inside_the_safe_training_pool(self):
        class FavorExpert(th.nn.Module):
            def forward(self, windows: th.Tensor) -> th.Tensor:
                return windows.new_full((len(windows),), -2.0)

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = _expert(Path(directory))
            reset_dataset = expert.reset_dataset()
            miner = ConfidentExpertResetTransform(
                expert, reset_dataset, FavorExpert(), microbatch_size=64,
            )
            miner.ready = True
            sampler = DatasetResetSampler(
                reset_dataset, transforms=(miner,), seed=9,
            )
            sample = sampler(th.ones(1_024, dtype=th.bool))
            self.assertTrue(th.isin(sample["frame_index"], expert.reset_indices).all())
            self.assertNotIn("skill_category", sample)
            self.assertGreater(miner.take_mined_fraction(), .3)

    def test_unmatched_situations_still_receive_uniform_expert_examples(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            _period(folder, "driving", 0)
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, reject_discontinuities=True,
            )
            generated = expert.sample(16, "cpu").clone()
            generated[:, :, BLUE_START + 16] = 0
            generated[:, :, BLUE_START + 2] = 200 / POSITION_SCALE[2]
            samples = list(SceneGAIFOMinibatches(
                expert, batch_size=16, epochs=1, noise_std=0, factorize=True,
            ).sample_windows(generated, th.arange(len(generated))))
            self.assertEqual(len(samples), 1)
            self.assertFalse(samples[0]["situation_matched"].any())
            self.assertNotIn("skill_category", samples[0])

    def test_nearby_but_untouched_flights_do_not_enter_aerial_pool(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            _period(folder, "aerial", 0)
            rows = np.load(folder / "100-0-aerial.npy")
            rows[:, 156] = 0
            np.save(folder / "100-0-aerial.npy", rows)
            expert = ExpertSceneDataset(
                folder, trajectory_length=8, reject_discontinuities=True,
                catalog_skills=True,
            )
            self.assertFalse(len(expert.curated_pools()[AERIAL_TOUCH_SKILL]))
            self.assertFalse(len(expert.curated_pools()[KICKOFF_SKILL]))
            self.assertTrue(len(expert.curated_pools()[DRIVING_SKILL]))

    def test_heldout_cataloged_windows_never_become_reset_or_training_examples(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            for index, kind in enumerate(("aerial", "aerial_touch", "dribble", "flick", "driving")):
                _period(Path(directory), kind, index)
            heldout = ExpertSceneDataset(
                Path(directory), trajectory_length=8, heldout_size=16,
                reject_discontinuities=True, catalog_skills=True,
            )
            for pool in heldout.curated_pools(heldout=True):
                self.assertTrue(th.isin(pool[:, 0], heldout.heldout_window_starts).all())
                self.assertFalse(th.isin(pool[:, 0], heldout.train_window_starts).any())
            for pool in heldout.curated_pools():
                self.assertFalse(th.isin(pool[:, 0], heldout.heldout_window_starts).any())
            self.assertFalse(th.isin(
                heldout.reset_indices, heldout.real_frame_indices[
                    th.isin(heldout.real_frame_indices - heldout.partition_span,
                            heldout.heldout_window_starts)
                ],
            ).any())
            sampled = heldout.sample_heldout(128, "cpu")
            markers = heldout.frames[heldout.heldout_window_starts + 7, 8]
            self.assertTrue(th.isin(sampled[:, -1, 8], markers).all())


if __name__ == "__main__":
    unittest.main()
