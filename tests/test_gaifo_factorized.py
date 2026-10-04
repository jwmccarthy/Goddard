"""Factorized GAIFO credit assignment without changing unified-mode behavior."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch as th

from gaifo import (
    AdaptiveDiscriminatorUpdate,
    BALL_NEAR_DISTANCE,
    BLUE_START,
    GAIFOCheckpoints,
    ExpertSceneDataset,
    FactorizedSceneDiscriminator,
    SceneDiscriminator,
    SceneDiscriminatorLoss,
    SceneDiscriminatorReward,
    SceneGAIFOMinibatches,
    ball_responsibility,
    build_discriminator,
    load_resume_checkpoint,
    nearest_ball_distance,
    opponent_view,
    parse_args,
    restore_training_checkpoint,
    train_discriminator_minibatch,
    validate_resume_args,
)
from jarl.data import TensorBatch
from jarl.store import RolloutBuffer
from jarl.transform import PrepareContext


class CoordinateHeads(th.nn.Module):
    factorized = True

    def forward(self, windows: th.Tensor) -> th.Tensor:
        return th.stack((windows[:, -1, BLUE_START + 15], windows[:, -1, 3]), dim=-1)


class TrainableCoordinateHeads(th.nn.Module):
    factorized = True

    def __init__(self):
        super().__init__()
        self.car_scale = th.nn.Parameter(th.tensor(0.1))
        self.ball_scale = th.nn.Parameter(th.tensor(1.0))

    def forward(self, windows: th.Tensor) -> th.Tensor:
        return th.stack((
            self.car_scale * windows[:, -1, BLUE_START + 15],
            self.ball_scale * windows[:, -1, 3],
        ), dim=-1)


class NearMistakeHeads(th.nn.Module):
    factorized = True

    def __init__(self):
        super().__init__()
        self.scale = th.nn.Parameter(th.tensor(1.0))

    def forward(self, windows: th.Tensor) -> th.Tensor:
        marker = self.scale * windows[:, -1, BLUE_START + 15]
        near = nearest_ball_distance(windows) <= BALL_NEAR_DISTANCE
        return th.stack((marker, th.where(near, -marker, marker)), dim=-1)


class FactorizedGAIFOTests(unittest.TestCase):
    def test_car_head_uses_start_context_but_not_future_ball_motion(self):
        th.manual_seed(2)
        model = FactorizedSceneDiscriminator(8, 8, 16)
        windows = th.randn(3, 8, 51) * 0.1
        changed = windows.clone()
        changed[:, 1:, :9] += 0.5
        original_logits = model(windows)
        changed_logits = model(changed)
        self.assertEqual(original_logits.shape, (3, 2))
        th.testing.assert_close(original_logits[:, 0], changed_logits[:, 0], rtol=0, atol=0)
        self.assertGreater((original_logits[:, 1] - changed_logits[:, 1]).abs().max(), 1e-5)

    def test_ball_gate_uses_physical_proximity_and_remembers_recent_contact(self):
        windows = th.zeros(2, 3, 51)
        windows[1, :, BLUE_START] = 3_000 / 4_108
        self.assertAlmostEqual(ball_responsibility(windows)[0].item(), 1.0)
        self.assertLess(ball_responsibility(windows)[1].item(), 0.2)
        windows[1, 0, BLUE_START] = 0
        self.assertAlmostEqual(ball_responsibility(windows)[1].item(), 1.0)

        canonical = th.zeros(1, 3, 51)
        canonical[..., BLUE_START] = 3_000 / 4_108
        self.assertLess(ball_responsibility(canonical).item(), 0.2)
        self.assertAlmostEqual(ball_responsibility(opponent_view(canonical)).item(), 1.0)

    def test_only_ball_imitation_is_gated_and_goal_still_pays_on_invalid_windows(self):
        windows = th.zeros(1, 5, 2, 51)
        windows[0, [1, 3], :, BLUE_START] = 2_000 / 4_108
        windows[0, :4, -1, BLUE_START + 15] = th.tensor([-2., -2., 2., 2.])
        windows[0, :4, -1, 3] = th.tensor([-2., 2., -2., 2.])
        valid = th.tensor([[True, True, True, True, False]])
        batch = TensorBatch({
            "observation": th.zeros(1, 5, 51),
            "scene_window": windows, "scene_window_valid": valid,
            "reward": th.tensor([[0., 0., 0., 0., 2.]]),
        })
        result = SceneDiscriminatorReward(
            CoordinateHeads(), noise_std=0, trajectory_length=2,
            batch_size=2,
        )(batch, PrepareContext())
        th.testing.assert_close(result["car_imitation_reward"][0, :4],
                                th.tensor([0.5, 0.5, -0.5, -0.5]))
        th.testing.assert_close(result["ball_imitation_reward"][0, [0, 2]],
                                th.tensor([0.5, 0.5]))
        self.assertLess(result["ball_imitation_reward"][0, 1].abs(), 0.2)
        self.assertGreater(result["car_imitation_reward"][0, 1].abs(), 0.4)
        self.assertEqual(result["training_reward"][0, 4].item(), 2.0)
        self.assertEqual(result["imitation_reward"][0, 4].item(), 0.0)
        self.assertTrue(result["learner_mask"].all())

    def test_exp_log_odds_rewards_keep_factorized_ball_proximity_gate(self):
        windows = th.zeros(1, 2, 2, 51)
        windows[0, :, :, BLUE_START + 15] = th.log(th.tensor(2.0))
        windows[0, :, :, 3] = -th.log(th.tensor(2.0))
        windows[0, 1, :, BLUE_START] = 3_000 / 4_108
        batch = TensorBatch({
            "observation": th.zeros(1, 2, 51),
            "scene_window": windows,
            "scene_window_valid": th.ones(1, 2, dtype=th.bool),
            "reward": th.zeros(1, 2),
        })
        result = SceneDiscriminatorReward(
            CoordinateHeads(), noise_std=0, trajectory_length=2,
            exp_log_odds_reward=True,
        )(batch, PrepareContext())
        th.testing.assert_close(result["car_imitation_reward"], th.full((1, 2), 0.25))
        th.testing.assert_close(result["ball_imitation_reward"],
                                result["ball_proximity"])
        self.assertGreater(result["ball_imitation_reward"][0, 0].item(), 0.9)
        self.assertLess(result["ball_imitation_reward"][0, 1].item(), 0.2)

    def test_balanced_near_ball_expert_sampling_and_both_heads_train(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = np.zeros((32, 161), dtype=np.float32)
            rows[:, BLUE_START] = 2_500 / 4_108
            rows[:, 30] = 3_000 / 4_108
            rows[:, BLUE_START + 14] = rows[:, 30 + 14] = 1
            rows[:, BLUE_START + 16] = rows[:, 30 + 16] = 1
            rows[8:13, BLUE_START] = 0
            np.save(folder / "replay.npy", rows)
            expert = ExpertSceneDataset(folder, 2, device="cpu", frame_skip=4, heldout_size=4)
            self.assertGreater(expert.near_total, 0)
            self.assertTrue((nearest_ball_distance(expert.sample_near(4, "cpu"))
                             <= BALL_NEAR_DISTANCE).all())

            generated = th.zeros(4, 2, 51)
            generated[:, :, BLUE_START] = 2_500 / 4_108
            generated[:, :, BLUE_START + 14] = 1
            generated[:, :, BLUE_START + 16] = 1
            generated[:1, :, BLUE_START] = 0
            untouched = generated.clone()
            sampler = SceneGAIFOMinibatches(expert, 4, 1, 0, factorize=True)
            sample = next(sampler.sample_windows(generated, th.arange(4)))
            th.testing.assert_close(generated, untouched)
            self.assertEqual(int(sample["situation_matched"][:4].sum()), 1)
            self.assertTrue((nearest_ball_distance(sample["window"][:1])
                             <= BALL_NEAR_DISTANCE).all())
            self.assertTrue((nearest_ball_distance(sample["window"][4:5])
                             <= BALL_NEAR_DISTANCE).all())

            discriminator = FactorizedSceneDiscriminator(8, 8, 16)
            output = SceneDiscriminatorLoss(discriminator)(sample)
            output.loss.backward()
            self.assertTrue(th.isfinite(output.loss))
            for head in (discriminator.car_head, discriminator.ball_head):
                self.assertIsNotNone(head.weight.grad)
                self.assertGreater(head.weight.grad.abs().sum().item(), 0)
            self.assertIn("near_ball_fraction", output.metrics)

    def test_ball_weighting_agrees_across_microbatch_sizes(self):
        windows = th.zeros(8, 2, 51)
        windows[[2, 3, 6, 7], :, BLUE_START] = 3_000 / 4_108
        windows[:4, -1, 3] = th.tensor([-2., -2., 2., 2.])
        windows[4:, -1, 3] = th.tensor([2., 2., -2., -2.])
        windows[:4, -1, BLUE_START + 15] = -1
        windows[4:, -1, BLUE_START + 15] = 1
        batch = TensorBatch({
            "window": windows,
            "is_agent": th.tensor([1., 1., 1., 1., 0., 0., 0., 0.]),
        })

        outcomes = []
        for size in (4, 1):
            discriminator = TrainableCoordinateHeads()
            optimizer = th.optim.SGD(discriminator.parameters(), lr=0)
            metrics = train_discriminator_minibatch(
                batch, discriminator, optimizer, SceneDiscriminatorLoss(discriminator),
                microbatch_size=size, max_grad_norm=100,
            )
            outcomes.append((metrics, discriminator.ball_scale.grad.clone()))
        th.testing.assert_close(outcomes[0][0]["ball_loss"], outcomes[1][0]["ball_loss"])
        th.testing.assert_close(outcomes[0][0]["loss"], outcomes[1][0]["loss"])
        th.testing.assert_close(outcomes[0][1], outcomes[1][1])
        self.assertGreater(outcomes[0][0]["ball_loss"].item(), 1.6)

    def test_near_ball_heldout_accuracy_prevents_early_stop_on_far_examples(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            rows = np.zeros((32, 161), dtype=np.float32)
            rows[:, BLUE_START] = 3_000 / 4_108
            rows[:, 30] = 3_000 / 4_108
            rows[8:13, BLUE_START] = 0
            rows[:, BLUE_START + 15] = -1
            rows[:, 30 + 15] = -1
            for name in ("first", "second"):
                np.save(folder / f"{name}.npy", rows)
            expert = ExpertSceneDataset(folder, 2, device="cpu", frame_skip=4, heldout_size=8)
            self.assertGreater(expert.heldout_near_total, 0)

            generated = th.zeros(8, 2, 51)
            generated[:, :, BLUE_START] = 3_000 / 4_108
            generated[0, :, BLUE_START] = 0
            generated[:, -1, BLUE_START + 15] = 1
            discriminator = NearMistakeHeads()
            update = AdaptiveDiscriminatorUpdate(
                expert=expert, history=None, batch_size=4, epochs=1, noise_std=0,
                heldout_size=8, accuracy_target=0.8, history_add_size=0,
                history_mix_fraction=0, max_grad_norm=1, discriminator=discriminator,
                optimizer=th.optim.Adam(discriminator.parameters()),
                loss=SceneDiscriminatorLoss(discriminator),
            )
            result = update._evaluate(generated, generated[:1])
            self.assertEqual(result["car_heldout_accuracy"], 1.0)
            self.assertEqual(result["ball_near_heldout_accuracy"], 0.0)
            self.assertEqual(result["heldout_accuracy"], 0.0)

    def test_flag_defaults_to_unified_and_checkpoint_resume_keeps_mode(self):
        with patch.object(sys, "argv", ["gaifo.py", "--replay-dir", "parsed_replays"]):
            default, _ = parse_args()
        self.assertFalse(default.factorize)
        self.assertFalse(default.hard_positive_mining)
        self.assertIsInstance(build_discriminator(default), SceneDiscriminator)

        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            flags = [
                "gaifo.py", "--replay-dir", "parsed_replays", "--factorize",
                "--hard-positive-mining",
                "--n-sim", "1", "--rollout", "4", "--policy-hidden", "16",
                "--critic-hidden", "16", "--discriminator-hidden", "16",
                "--frame-embedding", "8", "--temporal-hidden", "8",
            ]
            with patch.object(sys, "argv", flags):
                args, _ = parse_args()
            discriminator = build_discriminator(args)
            self.assertIsInstance(discriminator, FactorizedSceneDiscriminator)
            modules = {"policy": th.nn.Linear(1, 1), "critic": th.nn.Linear(1, 1),
                       "discriminator": discriminator}
            optimizers = {name: th.optim.Adam(module.parameters())
                          for name, module in modules.items()}
            checkpointer = GAIFOCheckpoints(
                Path(directory), 10, 2, modules["policy"], modules["critic"], discriminator,
                optimizers["policy"], optimizers["critic"], optimizers["discriminator"],
                RolloutBuffer(4, 2, th.device("cpu")), args,
            )
            checkpointer.save(0, force=True)
            path = Path(directory) / "gaifo_000000000000.pt"
            payload = load_resume_checkpoint(path)
            self.assertTrue(payload["config"]["factorize"])
            self.assertTrue(payload["config"]["hard_positive_mining"])
            with patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path)]):
                resumed, _ = parse_args()
            self.assertTrue(resumed.factorize)
            self.assertTrue(resumed.hard_positive_mining)
            validate_resume_args(resumed, payload)
            with patch.object(sys, "argv", ["gaifo.py", "--resume-checkpoint", str(path),
                                            "--no-factorize"]):
                mismatch, _ = parse_args()
            with self.assertRaisesRegex(ValueError, "--factorize must match"):
                validate_resume_args(mismatch, payload)

            restored = {"policy": th.nn.Linear(1, 1), "critic": th.nn.Linear(1, 1),
                        "discriminator": build_discriminator(resumed)}
            restored_optimizers = {name: th.optim.Adam(module.parameters())
                                   for name, module in restored.items()}
            restore_training_checkpoint(payload, resumed, restored, restored_optimizers)
            for key, value in discriminator.state_dict().items():
                th.testing.assert_close(value, restored["discriminator"].state_dict()[key])


if __name__ == "__main__":
    unittest.main()
