"""Discriminator-guided expert replay resets for both GAIFO modes."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch as th

from gaifo import (
    AdaptiveDiscriminatorUpdate,
    BLUE_START,
    ConfidentExpertResetTransform,
    ExpertSceneDataset,
    ORANGE_START,
    POSITION_SCALE,
    SCENE_SIZE,
    SceneDiscriminatorLoss,
    parse_args,
)
from jarl.data import TensorBatch
from jarl.envs import DatasetResetSampler


class ScoreDiscriminator(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = th.nn.Parameter(th.tensor(10.0))
        self.grad_modes = []

    def forward(self, windows: th.Tensor) -> th.Tensor:
        self.grad_modes.append(th.is_grad_enabled())
        return self.scale * windows[:, -1, 5]


class ScoreFactorizedDiscriminator(ScoreDiscriminator):
    factorized = True

    def forward(self, windows: th.Tensor) -> th.Tensor:
        self.grad_modes.append(th.is_grad_enabled())
        return self.scale * th.stack((
            windows[:, -1, BLUE_START + 5], windows[:, -1, 5],
        ), dim=-1)


def make_expert(folder: Path, *, heldout_size: int = 8) -> ExpertSceneDataset:
    for segment in range(3):
        rows = np.zeros((32, 161), dtype=np.float32)
        rows[:, 0] = (32 * segment + np.arange(32)) / 10_000
        rows[:, 2] = 91.25 / POSITION_SCALE[2]
        for car in (BLUE_START, ORANGE_START):
            rows[:, car + 2] = 17 / POSITION_SCALE[2]
            rows[:, car + 9] = 1
            rows[:, car + 14] = 1
        np.save(folder / f"segment{segment}.npy", rows)
    return ExpertSceneDataset(folder, trajectory_length=8, heldout_size=heldout_size)


def training_segments(expert: ExpertSceneDataset) -> list[th.Tensor]:
    return [indices for indices in expert.segment_frame_indices
            if (expert.reset_indices == indices[0]).any()]


def frame_ids(resets: TensorBatch) -> th.Tensor:
    return resets["frame_index"]


class HardPositiveMiningTests(unittest.TestCase):
    def test_flag_is_opt_in(self):
        for flag, expected in ((None, False), ("--hard-positive-mining", True),
                               ("--no-hard-positive-mining", False)):
            with self.subTest(flag=flag):
                flags = ["gaifo.py", "--replay-dir", "parsed_replays"]
                if flag is not None:
                    flags.append(flag)
                with patch.object(sys, "argv", flags):
                    parsed, _ = parse_args()
                self.assertIs(parsed.hard_positive_mining, expected)

    def test_unified_resets_favor_confident_training_window_starts(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = make_expert(Path(directory))
            training = training_segments(expert)
            self.assertEqual(len(training), 2)
            expert.frames[training[0], 5] = -0.9
            expert.frames[training[1], 5] = 0.9
            # More confident held-out scenes must never become reset candidates.
            expert.frames[expert.heldout_window_starts, 5] = -0.99
            dataset = expert.reset_dataset()
            discriminator = ScoreDiscriminator()
            miner = ConfidentExpertResetTransform(expert, dataset, discriminator, 128)
            sampler = DatasetResetSampler(dataset, transforms=(miner,), seed=11)
            uniform = DatasetResetSampler(dataset, seed=11)
            mask = th.ones(4_096, dtype=th.bool)

            before = sampler(mask)
            ordinary = uniform(mask)
            for key in before:
                th.testing.assert_close(before[key], ordinary[key])
            self.assertEqual(miner.take_mined_fraction(), 0)
            self.assertEqual(discriminator.grad_modes, [])

            miner.ready = True
            ordinary = uniform(mask)
            selected = sampler(mask)
            self.assertGreater(
                (expert.frames[frame_ids(selected), 5] < 0).float().mean().item(),
                (expert.frames[frame_ids(ordinary), 5] < 0).float().mean().item() + 0.15,
            )
            self.assertTrue(th.isin(frame_ids(selected), expert.reset_indices).all())
            changed = frame_ids(selected) != frame_ids(ordinary)
            self.assertTrue(th.isin(frame_ids(selected)[changed], expert.train_window_starts).all())
            self.assertGreater(miner.take_mined_fraction(), 0.4)
            self.assertTrue(discriminator.training)
            self.assertIsNone(discriminator.scale.grad)
            self.assertFalse(any(discriminator.grad_modes))

    def test_factorized_resets_prioritize_near_ball_control(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = make_expert(Path(directory))
            near, far = training_segments(expert)
            expert.frames[near, 5] = -0.4
            expert.frames[far, 5] = -0.9
            for car in (BLUE_START, ORANGE_START):
                expert.frames[far, car] = 3_000 / POSITION_SCALE[0]
                expert.frames[:, car + 5] = 0  # The car head is equally undecided.
            dataset = expert.reset_dataset()
            discriminator = ScoreFactorizedDiscriminator()
            miner = ConfidentExpertResetTransform(expert, dataset, discriminator, 128)
            scores = miner._score(th.stack((near[0], far[0])))
            self.assertGreater(scores[0].item(), 0.7)
            self.assertAlmostEqual(scores[1].item(), 0.5)

            miner.ready = True
            sampler = DatasetResetSampler(dataset, transforms=(miner,), seed=7)
            ordinary = DatasetResetSampler(dataset, seed=7)
            mask = th.ones(4_096, dtype=th.bool)
            selected = sampler(mask)
            baseline = ordinary(mask)
            near_selected = (
                expert.frames[frame_ids(selected), BLUE_START] * POSITION_SCALE[0] < 500
            ).float().mean().item()
            near_baseline = (
                expert.frames[frame_ids(baseline), BLUE_START] * POSITION_SCALE[0] < 500
            ).float().mean().item()
            self.assertGreater(near_selected, near_baseline + 0.15)
            self.assertTrue(th.isin(frame_ids(selected), expert.reset_indices).all())
            self.assertGreater(miner.take_mined_fraction(), 0.4)
            self.assertFalse(any(discriminator.grad_modes))

    def test_no_confident_candidates_preserves_uniform_reset_sampling(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = make_expert(Path(directory))
            expert.frames[:, 5] = 0.9
            dataset = expert.reset_dataset()
            miner = ConfidentExpertResetTransform(expert, dataset, ScoreDiscriminator(), 128)
            miner.ready = True
            sampler = DatasetResetSampler(dataset, transforms=(miner,), seed=13)
            ordinary = DatasetResetSampler(dataset, seed=13)
            mask = th.ones(128, dtype=th.bool)
            selected, baseline = sampler(mask), ordinary(mask)
            for key in selected:
                th.testing.assert_close(selected[key], baseline[key])
            self.assertEqual(miner.take_mined_fraction(), 0)

    def test_discriminator_update_enables_mining_without_weighting_its_loss(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            expert = make_expert(Path(directory), heldout_size=0)
            expert.frames[:, 5] = -0.9
            dataset = expert.reset_dataset()
            discriminator = ScoreDiscriminator()
            miner = ConfidentExpertResetTransform(expert, dataset, discriminator, 16)
            update = AdaptiveDiscriminatorUpdate(
                expert=expert, history=None, batch_size=2, epochs=1,
                noise_std=0, heldout_size=0, accuracy_target=1,
                history_add_size=0, history_mix_fraction=0, max_grad_norm=1,
                discriminator=discriminator,
                optimizer=th.optim.SGD(discriminator.parameters(), lr=0),
                loss=SceneDiscriminatorLoss(discriminator),
                microbatch_size=2, reset_miner=miner,
            )
            windows = th.zeros(1, 2, 8, SCENE_SIZE)
            windows[..., 5] = 0.9
            rollout = TensorBatch({
                "scene_window": windows,
                "scene_window_valid": th.ones(1, 2, dtype=th.bool),
            })
            _, first = update.run(rollout)
            self.assertTrue(miner.ready)
            self.assertEqual(first["Discriminator"]["reset_mined_fraction"], 0)
            self.assertEqual(first["Discriminator"]["minibatches"], 1)

            sampler = DatasetResetSampler(dataset, transforms=(miner,), seed=0)
            sampler(th.ones(128, dtype=th.bool))
            _, second = update.run(rollout)
            self.assertGreater(second["Discriminator"]["reset_mined_fraction"], 0.3)
            self.assertNotIn("train_expert_mining_max_weight", second["Discriminator"])


if __name__ == "__main__":
    unittest.main()
