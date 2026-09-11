import unittest

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from carl.gymnasium.action import CARLActionCodec
from distill import (
    ACTION_SIZES,
    ActionDecoder,
    ConditionalPrior,
    DeterministicTeacher,
    DistillCheckpoints,
    DistillRolloutTransform,
    DistillationRunner,
    GaussianEncoder,
    PulseLoss,
    PulsePolicy,
    TrajectoryChunkMinibatches,
    categorical_distillation_loss,
    diagonal_gaussian_kl,
    encode_action_factors,
    exact_action_accuracy,
    factor_actions,
    kl_coefficient,
    masked_logits,
    validate_control_manifest,
    validate_args,
)
from jarl.data.batch import TensorBatch
from jarl.data.records import PolicyOutput
from jarl.store.rollout import RolloutBuffer
from tracker import GOAL_STATE_SIZE


class AllValidActionCodec:
    def mask(self, state: th.Tensor) -> th.Tensor:
        return th.ones(
            (*state.shape[:-1], sum(ACTION_SIZES)),
            dtype=th.bool,
            device=state.device,
        )


class RecordingDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[th.Tensor, th.Tensor]] = []

    def forward(self, state: th.Tensor, latent: th.Tensor) -> th.Tensor:
        self.calls.append((state.detach().clone(), latent.detach().clone()))
        return th.zeros(
            state.shape[0], sum(ACTION_SIZES), device=state.device, dtype=state.dtype
        )


class LatentBlindDecoder(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.model = nn.Linear(state_dim, sum(ACTION_SIZES))

    def forward(self, state: th.Tensor, latent: th.Tensor) -> th.Tensor:
        return self.model(state)


class GaussianEncoderTest(unittest.TestCase):
    def test_segment_returns_one_posterior_per_segment(self):
        encoder = GaussianEncoder(30, 4, [16])
        observations = th.randn(3, 5, 30)
        valid = th.ones(3, 5, dtype=th.bool)

        mean, log_variance = encoder.segment(observations, valid)

        self.assertEqual(mean.shape, (3, 4))
        self.assertEqual(log_variance.shape, (3, 4))
        self.assertTrue((log_variance >= -5.0).all() and (log_variance <= 2.0).all())

    def test_segment_ignores_padded_observations(self):
        encoder = GaussianEncoder(30, 4, [16])
        prefix = th.randn(3, 30)
        observations = th.stack((
            th.cat((prefix, th.randn(2, 30))),
            th.cat((prefix, th.randn(2, 30))),
        ))
        valid = th.tensor([[True, True, True, False, False]] * 2)

        mean, log_variance = encoder.segment(observations, valid)

        th.testing.assert_close(mean[0], mean[1])
        th.testing.assert_close(log_variance[0], log_variance[1])

    def test_segment_encoding_preserves_temporal_order(self):
        encoder = GaussianEncoder(30, 4, [16])
        observations = th.randn(1, 4, 30)
        reversed_observations = observations.flip(1)
        valid = th.ones(1, 4, dtype=th.bool)

        mean, _ = encoder.segment(observations, valid)
        reversed_mean, _ = encoder.segment(reversed_observations, valid)

        self.assertFalse(th.allclose(mean, reversed_mean))

    def test_segment_is_action_conditioned(self):
        encoder = GaussianEncoder(GOAL_STATE_SIZE, 4, [16])
        observations = th.randn(1, 4, GOAL_STATE_SIZE)
        valid = th.ones(1, 4, dtype=th.bool)
        action_a = th.zeros(1, 4, 7, dtype=th.long)
        action_b = th.ones(1, 4, 7, dtype=th.long)

        mean_a, _ = encoder.segment(observations, valid, action_a)
        mean_b, _ = encoder.segment(observations, valid, action_b)

        self.assertFalse(th.allclose(mean_a, mean_b))

    def test_forward_uses_zero_action_context_fallback(self):
        encoder = GaussianEncoder(GOAL_STATE_SIZE, 4, [16])
        observation = th.randn(2, GOAL_STATE_SIZE)

        mean, log_variance = encoder.forward(observation)

        self.assertEqual(mean.shape, (2, 4))
        self.assertEqual(log_variance.shape, (2, 4))
        self.assertTrue((log_variance >= -5.0).all() and (log_variance <= 2.0).all())


class ConditionalPriorTest(unittest.TestCase):
    def test_old_style_construction_and_forward(self):
        prior = ConditionalPrior(GOAL_STATE_SIZE, 3, [8])
        state = th.randn(2, GOAL_STATE_SIZE)

        mean, log_variance = prior(state)

        self.assertEqual(mean.shape, (2, 3))
        self.assertEqual(log_variance.shape, (2, 3))

    def test_empty_hidden_dimensions_raise(self):
        with self.assertRaises(ValueError):
            ConditionalPrior(GOAL_STATE_SIZE, 3, [])

    def test_invalid_max_duration_raises(self):
        with self.assertRaises(ValueError):
            ConditionalPrior(GOAL_STATE_SIZE, 3, [8], max_duration=0)

    def test_unconditioned_prior_ignores_duration(self):
        prior = ConditionalPrior(GOAL_STATE_SIZE, 3, [8])
        state = th.randn(2, GOAL_STATE_SIZE)

        mean_a, variance_a = prior(state, th.tensor([1, 10]))
        mean_b, variance_b = prior(state, th.tensor([5, 5]))

        th.testing.assert_close(mean_a, mean_b)
        th.testing.assert_close(variance_a, variance_b)

    def test_duration_conditioning_changes_output(self):
        prior = ConditionalPrior(GOAL_STATE_SIZE, 3, [8], max_duration=10)
        state = th.randn(1, GOAL_STATE_SIZE)

        mean_a, _ = prior(state, th.tensor([1]))
        mean_b, _ = prior(state, th.tensor([10]))

        self.assertFalse(th.allclose(mean_a, mean_b))


class DeterministicTeacherTest(unittest.TestCase):
    def test_forces_deterministic_recurrent_actions(self):
        class Teacher:
            device = th.device("cpu")

            @staticmethod
            def initial_state(batch_size):
                return th.zeros(batch_size, 1)

            @staticmethod
            def act(observation, state, *, deterministic=False):
                return PolicyOutput(
                    action=th.full((len(observation), 7), int(deterministic)),
                    next_state=state + 1,
                )

        teacher = DeterministicTeacher(Teacher())
        state = teacher.initial_state(2)

        output = teacher.act(th.zeros((2, GOAL_STATE_SIZE)), state)

        th.testing.assert_close(output.action, th.ones((2, 7), dtype=th.long))
        th.testing.assert_close(output.next_state, th.ones((2, 1)))


class CategoricalDistillationLossTest(unittest.TestCase):
    def test_real_action_mask_excludes_illegal_ground_controls(self):
        state = th.zeros((1, GOAL_STATE_SIZE))
        state[:, 25] = 1
        logits = th.zeros((1, sum(ACTION_SIZES)))
        logits[:, [4, 5, 12, 14, 15]] = 100

        action = factor_actions(masked_logits(logits, state, CARLActionCodec()))

        self.assertEqual(action[0, 1].item(), 0)
        self.assertEqual(action[0, 4].item(), 0)
        self.assertEqual(action[0, 5].item(), 0)

    def test_factorized_action_loss_and_argmax(self):
        target = th.tensor([[2, 1, 0, 1, 0, 2, 1]])
        logits = th.full((1, sum(ACTION_SIZES)), -5.0)
        offset = 0
        for size, value in zip(ACTION_SIZES, target[0]):
            logits[0, offset + value] = 5.0
            offset += size

        loss, accuracy = categorical_distillation_loss(logits, target)

        th.testing.assert_close(factor_actions(logits), target)
        self.assertLess(loss.item(), 0.001)
        self.assertEqual(accuracy.item(), 1.0)

    def test_valid_mask_ignores_invalid_frames(self):
        target = th.zeros((4, 7), dtype=th.long)
        logits = th.zeros((4, sum(ACTION_SIZES)))

        masked_loss, masked_accuracy = categorical_distillation_loss(
            logits, target, valid=th.tensor([True, True, False, False])
        )
        full_loss, full_accuracy = categorical_distillation_loss(
            logits[:2], target[:2]
        )

        th.testing.assert_close(masked_loss, full_loss)
        th.testing.assert_close(masked_accuracy, full_accuracy)

    def test_all_invalid_frames_return_zero(self):
        target = th.zeros((4, 7), dtype=th.long)
        logits = th.zeros((4, sum(ACTION_SIZES)))

        loss, accuracy = categorical_distillation_loss(
            logits, target, valid=th.zeros(4, dtype=th.bool)
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(accuracy.item(), 0.0)


class ExactActionAccuracyTest(unittest.TestCase):
    def test_perfect_predictions_have_exact_accuracy_one(self):
        target = th.tensor([[2, 1, 0, 1, 0, 2, 1]])
        logits = th.full((1, sum(ACTION_SIZES)), -5.0)
        offset = 0
        for size, value in zip(ACTION_SIZES, target[0]):
            logits[0, offset + value.item()] = 5.0
            offset += size

        accuracy = exact_action_accuracy(logits, target)

        self.assertEqual(accuracy.item(), 1.0)

    def test_single_wrong_factor_reduces_exact_accuracy(self):
        target = th.tensor([[2, 1, 0, 1, 0, 2, 1]])
        logits = th.full((1, sum(ACTION_SIZES)), -5.0)
        offset = 0
        for size, value in zip(ACTION_SIZES, target[0]):
            logits[0, offset + value.item()] = 5.0
            offset += size
        logits[0, 1] = 5.0

        accuracy = exact_action_accuracy(logits, target)

        self.assertEqual(accuracy.item(), 0.0)

    def test_valid_mask_ignores_invalid_frames(self):
        target = th.zeros((4, 7), dtype=th.long)
        logits = th.zeros((4, sum(ACTION_SIZES)))
        logits[:2, 0] = 5.0

        masked_accuracy = exact_action_accuracy(
            logits, target, valid=th.tensor([True, True, False, False])
        )
        full_accuracy = exact_action_accuracy(logits[:2], target[:2])

        th.testing.assert_close(masked_accuracy, full_accuracy)


class PulseLossTest(unittest.TestCase):
    def test_shared_latent_decodes_at_every_valid_frame(self):
        th.manual_seed(0)
        latent_size = 2
        encoder = GaussianEncoder(GOAL_STATE_SIZE, latent_size, [16])
        decoder = RecordingDecoder()
        action_codec = AllValidActionCodec()
        policy = PulsePolicy(encoder, decoder, action_codec)
        prior = ConditionalPrior(GOAL_STATE_SIZE, latent_size, [16], max_duration=4)
        loss = PulseLoss(
            policy,
            prior,
            action_codec,
            kl_weight=0.1,
            prior_action_weight=0.5,
        )

        teacher_action = th.stack(
            [th.randint(0, size, (2, 3)) for size in ACTION_SIZES], dim=-1
        )
        batch = TensorBatch(
            {
                "observation": th.randn(2, 3, GOAL_STATE_SIZE),
                "teacher_action": teacher_action,
                "valid": th.tensor([[True, True, False], [True, False, False]]),
                "duration": th.tensor([2, 1]),
                "planned_duration": th.tensor([3, 4]),
            }
        )

        output = loss(batch)

        self.assertEqual(output.loss.shape, ())
        self.assertIn("action_loss", output.metrics)
        self.assertIn("prior_action_loss", output.metrics)
        self.assertIn("action_accuracy", output.metrics)
        self.assertIn("prior_action_accuracy", output.metrics)
        self.assertIn("action_exact_accuracy", output.metrics)
        self.assertIn("prior_action_exact_accuracy", output.metrics)
        self.assertIn("kl", output.metrics)
        self.assertIn("latent_contrast_loss", output.metrics)
        self.assertIn("latent_action_change_rate", output.metrics)
        self.assertIn("prior_duration_sensitivity", output.metrics)

        self.assertEqual(len(decoder.calls), 3)
        for state, latent in decoder.calls:
            self.assertEqual(state.shape, (3, GOAL_STATE_SIZE))
            self.assertEqual(latent.shape, (3, latent_size))
            th.testing.assert_close(latent[0], latent[1])

    def test_contrastive_loss_zero_for_single_segment(self):
        encoder = GaussianEncoder(GOAL_STATE_SIZE, 2, [16])
        decoder = LatentBlindDecoder(GOAL_STATE_SIZE)
        action_codec = AllValidActionCodec()
        policy = PulsePolicy(encoder, decoder, action_codec)
        prior = ConditionalPrior(GOAL_STATE_SIZE, 2, [16], max_duration=4)
        loss = PulseLoss(
            policy,
            prior,
            action_codec,
            kl_weight=0.0,
            prior_action_weight=0.0,
            latent_contrast_weight=1.0,
            latent_contrast_margin=0.5,
        )
        batch = TensorBatch(
            {
                "observation": th.randn(1, 3, GOAL_STATE_SIZE),
                "teacher_action": th.zeros(1, 3, 7, dtype=th.long),
                "valid": th.ones(1, 3, dtype=th.bool),
                "duration": th.tensor([3]),
                "planned_duration": th.tensor([3]),
            }
        )

        output = loss(batch)

        self.assertTrue(th.isfinite(output.loss))
        self.assertEqual(output.metrics["latent_contrast_loss"].item(), 0.0)
        self.assertEqual(output.metrics["latent_action_change_rate"].item(), 0.0)

    def test_contrastive_loss_positive_when_decoder_ignores_latent(self):
        encoder = GaussianEncoder(GOAL_STATE_SIZE, 2, [16])
        decoder = LatentBlindDecoder(GOAL_STATE_SIZE)
        action_codec = AllValidActionCodec()
        policy = PulsePolicy(encoder, decoder, action_codec)
        prior = ConditionalPrior(GOAL_STATE_SIZE, 2, [16], max_duration=4)
        loss = PulseLoss(
            policy,
            prior,
            action_codec,
            kl_weight=0.0,
            prior_action_weight=0.0,
            latent_contrast_weight=1.0,
            latent_contrast_margin=0.5,
        )
        teacher_action = th.tensor(
            [
                [[0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0]],
                [[2, 2, 2, 0, 0, 2, 0], [0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0]],
            ],
            dtype=th.long,
        )
        batch = TensorBatch(
            {
                "observation": th.randn(2, 3, GOAL_STATE_SIZE),
                "teacher_action": teacher_action,
                "valid": th.ones(2, 3, dtype=th.bool),
                "duration": th.tensor([3, 3]),
                "planned_duration": th.tensor([3, 3]),
            }
        )

        output = loss(batch)

        self.assertAlmostEqual(
            output.metrics["latent_contrast_loss"].item(), 0.5, places=5
        )
        self.assertEqual(output.metrics["latent_action_change_rate"].item(), 0.0)

    def test_prior_duration_sensitivity_zero_for_unconditioned_prior(self):
        encoder = GaussianEncoder(GOAL_STATE_SIZE, 2, [16])
        decoder = LatentBlindDecoder(GOAL_STATE_SIZE)
        action_codec = AllValidActionCodec()
        policy = PulsePolicy(encoder, decoder, action_codec)
        prior = ConditionalPrior(GOAL_STATE_SIZE, 2, [16])
        loss = PulseLoss(
            policy,
            prior,
            action_codec,
            kl_weight=0.0,
            prior_action_weight=0.0,
        )
        batch = TensorBatch(
            {
                "observation": th.randn(2, 3, GOAL_STATE_SIZE),
                "teacher_action": th.zeros(2, 3, 7, dtype=th.long),
                "valid": th.ones(2, 3, dtype=th.bool),
                "duration": th.tensor([3, 3]),
                "planned_duration": th.tensor([3, 3]),
            }
        )

        output = loss(batch)

        self.assertEqual(output.metrics["prior_duration_sensitivity"].item(), 0.0)

    def test_prior_duration_sensitivity_positive_for_conditioned_prior(self):
        encoder = GaussianEncoder(GOAL_STATE_SIZE, 2, [16])
        decoder = LatentBlindDecoder(GOAL_STATE_SIZE)
        action_codec = AllValidActionCodec()
        policy = PulsePolicy(encoder, decoder, action_codec)
        prior = ConditionalPrior(GOAL_STATE_SIZE, 2, [16], max_duration=10)
        loss = PulseLoss(
            policy,
            prior,
            action_codec,
            kl_weight=0.0,
            prior_action_weight=0.0,
        )
        batch = TensorBatch(
            {
                "observation": th.randn(2, 3, GOAL_STATE_SIZE),
                "teacher_action": th.zeros(2, 3, 7, dtype=th.long),
                "valid": th.ones(2, 3, dtype=th.bool),
                "duration": th.tensor([3, 3]),
                "planned_duration": th.tensor([3, 3]),
            }
        )

        output = loss(batch)

        self.assertGreater(output.metrics["prior_duration_sensitivity"].item(), 0.0)


class TrajectoryChunkMinibatchesTest(unittest.TestCase):
    def _rollout(self, time: int = 6, envs: int = 2) -> TensorBatch:
        return TensorBatch(
            {
                "observation": th.arange(time * envs * 30, dtype=th.float32).reshape(
                    time, envs, 30
                ),
                "teacher_action": th.zeros(time, envs, 7, dtype=th.long),
                "terminated": th.zeros(time, envs, dtype=th.bool),
                "truncated": th.zeros(time, envs, dtype=th.bool),
            }
        )

    def test_chunk_durations_within_range(self):
        sampler = TrajectoryChunkMinibatches(horizon=3, jitter=1, batch_size=8)
        batch = next(iter(sampler(self._rollout())))

        self.assertTrue((batch.duration >= 1).all())
        self.assertTrue((batch.duration <= 4).all())
        self.assertTrue((batch.planned_duration >= 2).all())
        self.assertTrue((batch.planned_duration <= 4).all())
        self.assertTrue((batch.duration <= batch.planned_duration).all())
        self.assertEqual(batch.data["observation"].shape[1], 4)
        self.assertEqual(batch.valid.shape[1], 4)

    def test_non_crossing_chunks(self):
        data = self._rollout(time=6, envs=2)
        data = data.replace_fields(
            terminated=data["terminated"].clone().scatter_(0, th.tensor([[2], [5]]), True)
        )
        sampler = TrajectoryChunkMinibatches(horizon=2, jitter=0, batch_size=16)

        chunks = []
        for batch in sampler(data):
            for index in range(len(batch.duration)):
                chunks.append(
                    (
                        batch.data["observation"][index],
                        batch.duration[index],
                        batch.valid[index],
                    )
                )

        for observation, duration, valid in chunks:
            valid_frames = observation[valid]
            indices = (valid_frames[:, 0] / 30.0).long()
            env_id = (indices[0].item() % 2)
            self.assertTrue(((indices % 2) == env_id).all())
            th.testing.assert_close(
                indices[1:] - indices[:-1], th.full((len(indices) - 1,), 2)
            )

    def test_epochs_yield_separate_batches(self):
        sampler = TrajectoryChunkMinibatches(
            horizon=2, jitter=0, batch_size=12, epochs=2
        )
        batches = list(sampler(self._rollout()))
        self.assertEqual(len(batches), 2)

    def test_chunk_plan_is_stable_across_optimizer_epochs(self):
        th.manual_seed(3)
        sampler = TrajectoryChunkMinibatches(
            horizon=2, jitter=1, batch_size=128, epochs=2
        )
        first, second = list(sampler(self._rollout()))

        def plan(batch):
            starts = batch.data["observation"][:, 0, 0].tolist()
            return sorted(zip(starts, batch.duration.tolist(), batch.planned_duration.tolist()))

        self.assertEqual(plan(first), plan(second))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            TrajectoryChunkMinibatches(horizon=0, jitter=1, batch_size=4)
        with self.assertRaises(ValueError):
            TrajectoryChunkMinibatches(horizon=2, jitter=-1, batch_size=4)
        with self.assertRaises(ValueError):
            TrajectoryChunkMinibatches(horizon=2, jitter=0, batch_size=0)
        with self.assertRaises(ValueError):
            TrajectoryChunkMinibatches(horizon=2, jitter=0, batch_size=4, epochs=0)


class DistillRolloutTransformTest(unittest.TestCase):
    def test_keeps_useful_metrics(self):
        observation = th.arange(18, dtype=th.float32).reshape(3, 2, 3)
        terminated = th.tensor([[False, False], [True, False], [False, False]])
        truncated = th.tensor([[False, True], [False, False], [False, False]])
        action = th.zeros((3, 2, 7), dtype=th.long)
        teacher_action = action.clone()
        teacher_action[2, 1, 0] = 1
        batch = TensorBatch(
            {
                "observation": observation,
                "terminated": terminated,
                "truncated": truncated,
                "action": action,
                "teacher_action": teacher_action,
                "student_controlled": th.tensor([[True, False], [False, True], [False, False]]),
            }
        )

        transformed = DistillRolloutTransform()(batch, None)

        self.assertIn("action_agreement", transformed)
        self.assertIn("rollout_action_agreement", transformed)
        self.assertIn("reset_fraction", transformed)
        self.assertIn("student_controlled", transformed)
        self.assertAlmostEqual(
            transformed["action_agreement"][2, 1].item(), 6 / 7
        )
        self.assertEqual(
            transformed["rollout_action_agreement"][2, 1].item(), 0.0
        )
        th.testing.assert_close(
            transformed["reset_fraction"],
            (terminated | truncated).float(),
        )
        th.testing.assert_close(
            transformed["student_controlled"],
            batch["student_controlled"].float(),
        )


class ArgumentValidationTest(unittest.TestCase):
    def _valid_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            n_sim=256,
            frameskip=4,
            latent_size=32,
            rollout=32,
            batch_size=16,
            epochs=6,
            timesteps=1000,
            checkpoint_interval=10,
            checkpoint_keep=2,
            minimum_tracking_frames=32,
            minimum_remaining_frames=128,
            skill_horizon=16,
            skill_horizon_jitter=4,
            prior_action_weight=0.0,
            kl_initial=0.001,
            kl_final=0.0001,
            kl_anneal_start=0,
            kl_anneal_end=10,
            latent_contrast_weight=0.1,
            latent_contrast_margin=0.1,
            student_rollout_fraction=0.25,
            student_rollout_warmup=100_000_000,
            tracker_checkpoint=Path("tracker.pt"),
            resume=None,
        )

    def test_valid_args_pass(self):
        args = self._valid_args()
        args.tracker_checkpoint = Path(__file__)
        validate_args(args)

    def test_skill_horizon_jitter_lower_duration_must_be_at_least_one(self):
        args = self._valid_args()
        args.skill_horizon = 2
        args.skill_horizon_jitter = 4
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_prior_action_weight_must_be_finite_and_non_negative(self):
        args = self._valid_args()
        args.prior_action_weight = -0.1
        with self.assertRaises(ValueError):
            validate_args(args)
        args = self._valid_args()
        args.prior_action_weight = float("inf")
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_latent_contrast_weight_must_be_finite_and_non_negative(self):
        args = self._valid_args()
        args.latent_contrast_weight = -0.1
        with self.assertRaises(ValueError):
            validate_args(args)
        args = self._valid_args()
        args.latent_contrast_weight = float("nan")
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_latent_contrast_margin_must_be_finite_and_non_negative(self):
        args = self._valid_args()
        args.latent_contrast_margin = -0.1
        with self.assertRaises(ValueError):
            validate_args(args)
        args = self._valid_args()
        args.latent_contrast_margin = float("inf")
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_skill_horizon_must_be_positive(self):
        args = self._valid_args()
        args.skill_horizon = 0
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_skill_horizon_jitter_must_be_non_negative(self):
        args = self._valid_args()
        args.skill_horizon_jitter = -1
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_student_rollout_fraction_must_be_in_zero_one(self):
        args = self._valid_args()
        args.student_rollout_fraction = -0.1
        with self.assertRaises(ValueError):
            validate_args(args)
        args = self._valid_args()
        args.student_rollout_fraction = 1.1
        with self.assertRaises(ValueError):
            validate_args(args)

    def test_student_rollout_warmup_must_be_positive(self):
        args = self._valid_args()
        args.student_rollout_warmup = 0
        with self.assertRaises(ValueError):
            validate_args(args)


class DistillTest(unittest.TestCase):
    def test_checkpoint_waits_for_rollout_boundary(self):
        checkpoint = DistillCheckpoints.__new__(DistillCheckpoints)
        checkpoint.step = 0
        checkpoint.next_step = 10
        checkpoint.buffer = SimpleNamespace(position=1)

        self.assertFalse(checkpoint.ready(10))
        self.assertEqual(checkpoint.step, 10)

        checkpoint.buffer.position = 0
        self.assertTrue(checkpoint.ready(11))

    def test_matching_gaussians_have_zero_kl(self):
        mean = th.randn(4, 3)
        log_variance = th.randn(4, 3).clamp(-5, 2)

        loss = diagonal_gaussian_kl(mean, log_variance, mean, log_variance)

        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_kl_is_summed_over_latent_dimensions(self):
        posterior_mean = th.ones(2, 3)
        zeros = th.zeros_like(posterior_mean)

        loss = diagonal_gaussian_kl(posterior_mean, zeros, zeros, zeros)

        self.assertAlmostEqual(loss.item(), 1.5, places=6)

    def test_kl_coefficient_anneals_linearly(self):
        values = [
            kl_coefficient(step, 0.01, 0.001, 100, 200)
            for step in (0, 100, 150, 200, 300)
        ]

        self.assertEqual(values, [0.01, 0.01, 0.0055, 0.001, 0.001])

    def test_resume_requires_matching_opponent_context_manifest(self):
        validate_control_manifest(("replay:a",), ("replay:a",))
        with self.assertRaisesRegex(ValueError, "opponent-context"):
            validate_control_manifest(("replay:a",), ("replay:b",))
        with self.assertRaisesRegex(ValueError, "opponent-context"):
            validate_control_manifest(None, ("replay:a",))


class FakeActionCodec:
    def mask(self, state: th.Tensor) -> th.Tensor:
        return th.ones(
            (*state.shape[:-1], sum(ACTION_SIZES)),
            dtype=th.bool,
            device=state.device,
        )


class FakeDistillEnv:
    def __init__(
        self,
        n_envs: int,
        obs_dim: int,
        done_steps: dict[int, int] | None = None,
        device: str = "cpu",
    ) -> None:
        self.n_envs = n_envs
        self.device = th.device(device)
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (obs_dim,), np.float32
        )
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space, n_envs
        )
        self.single_action_space = gym.spaces.MultiDiscrete([2] * 7)
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, n_envs
        )
        self.action_codec = FakeActionCodec()
        self._done_steps = done_steps or {}
        self._step = 0
        self._obs = th.zeros(n_envs, obs_dim, device=self.device)

    def reset(self):
        self._step = 0
        self._obs = (
            th.arange(self.n_envs * self._obs.shape[-1], dtype=th.float32, device=self.device)
            .reshape(self._obs.shape)
        )
        return self._obs

    def step(self, action):
        self._step += 1
        self._obs = self._obs + 1.0
        reward = th.ones(self.n_envs, device=self.device)
        terminated = th.zeros(self.n_envs, dtype=th.bool, device=self.device)
        for env_id, step in self._done_steps.items():
            if self._step == step:
                terminated[env_id] = True
        truncated = th.zeros(self.n_envs, dtype=th.bool, device=self.device)
        info = {}
        return self._obs, reward, terminated, truncated, info

    def close(self):
        pass


class FakeRecurrentTeacher(nn.Module):
    def __init__(self, action: th.Tensor, state_dim: int = 1) -> None:
        super().__init__()
        self.device = action.device
        self._action = action
        self._state_dim = state_dim

    def initial_state(self, batch_size: int):
        return th.zeros(batch_size, self._state_dim, device=self.device)

    def act(self, observation, state, *, deterministic=False):
        action = self._action.expand(observation.shape[0], -1)
        return PolicyOutput(action=action, next_state=state + 1)


class FakePrior(nn.Module):
    def __init__(self, latent_size: int, state_dim: int = GOAL_STATE_SIZE) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.mean = nn.Linear(1, latent_size)

    def forward(self, state, duration):
        latent = state[:, :1].expand(-1, self.mean.out_features)
        return latent, th.zeros_like(latent)


class RecordingFakePulsePolicy(nn.Module):
    def __init__(self, latent_size: int) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.decoder_calls: list[tuple[th.Tensor, th.Tensor]] = []

    def decoder(self, state: th.Tensor, latent: th.Tensor) -> th.Tensor:
        self.decoder_calls.append((state.detach().clone(), latent.detach().clone()))
        logits = th.zeros(
            state.shape[0], sum(ACTION_SIZES), device=state.device, dtype=state.dtype
        )
        # Deterministic action depends on first latent component so we can
        # observe latent values through the chosen action.
        logits[:, 0] = latent[:, 0]
        return logits


class DistillationRunnerTest(unittest.TestCase):
    def _build_runner(
        self,
        n_envs: int = 2,
        student_fraction: float = 1.0,
        skill_horizon: int = 2,
        skill_horizon_jitter: int = 0,
        latent_size: int = 3,
        done_steps: dict[int, int] | None = None,
        teacher_action: th.Tensor | None = None,
    ):
        obs_dim = GOAL_STATE_SIZE + 2
        env = FakeDistillEnv(n_envs, obs_dim, done_steps=done_steps)
        if teacher_action is None:
            teacher_action = th.ones(n_envs, 7, dtype=th.long)
        teacher = DeterministicTeacher(
            FakeRecurrentTeacher(teacher_action, state_dim=1)
        )
        policy = RecordingFakePulsePolicy(latent_size)
        prior = FakePrior(latent_size)
        buffer = RolloutBuffer(horizon=16, num_envs=n_envs, device="cpu")
        runner = DistillationRunner(
            env,
            teacher,
            policy,
            prior,
            buffer,
            student_fraction=student_fraction,
            skill_horizon=skill_horizon,
            skill_horizon_jitter=skill_horizon_jitter,
            seed=0,
        )
        return runner, env, policy, buffer

    def test_teacher_labels_differ_from_student_actions(self):
        runner, env, policy, buffer = self._build_runner(n_envs=2, student_fraction=1.0)
        runner.reset()
        for _ in range(4):
            runner.step()

        rollout = buffer.finish().steps
        # Teacher always emits ones; student decoder returns zeros.
        self.assertTrue((rollout["teacher_action"] == 1).all())
        self.assertTrue((rollout["action"] == 0).all())
        self.assertTrue((rollout["student_controlled"]).all())

    def test_student_latents_held_for_duration(self):
        runner, env, policy, buffer = self._build_runner(
            n_envs=2, student_fraction=1.0, skill_horizon=2, skill_horizon_jitter=0
        )
        runner.reset()
        for _ in range(4):
            runner.step()

        # Decoder is invoked once per step for all student envs.
        self.assertEqual(len(policy.decoder_calls), 4)
        for env_id in range(2):
            latents = [
                latent[env_id, 0].item()
                for _, latent in policy.decoder_calls
            ]
            # Duration is exactly two, so latents pair up: (a, a, b, b).
            self.assertEqual(latents[0], latents[1])
            self.assertEqual(latents[2], latents[3])
            self.assertNotEqual(latents[0], latents[2])

    def test_done_resets_teacher_state_and_mode_machinery(self):
        teacher_action = th.empty(2, 7, dtype=th.long)
        teacher_action[0] = 0
        teacher_action[1] = 1
        runner, env, policy, buffer = self._build_runner(
            n_envs=2,
            student_fraction=1.0,
            skill_horizon=2,
            skill_horizon_jitter=0,
            done_steps={0: 2},
            teacher_action=teacher_action,
        )
        runner.reset()
        # Step 1: teacher state increments for both envs.
        runner.step()
        self.assertTrue((runner._teacher_state == 1).all())
        # Step 2: env 0 reaches done and its teacher state is zeroed; env 1
        # continues and accumulates another step.
        runner.step()
        self.assertEqual(runner._teacher_state[0].item(), 0)
        self.assertEqual(runner._teacher_state[1].item(), 2)
        # Env 0 is reset as student. Its next latent is selected from the new
        # episode observation when the next primitive step begins.
        self.assertTrue(runner._student_controlled[0].item())
        self.assertEqual(runner._elapsed[0].item(), 0)
        self.assertEqual(runner._duration[0].item(), 2)
        th.testing.assert_close(
            runner._held_latent[0], th.zeros_like(runner._held_latent[0])
        )
        # Env 1 was never done and remains a student with a valid skill duration.
        self.assertTrue(runner._student_controlled[1].item())
        self.assertEqual(runner._duration[1].item(), 2)

    def test_rollout_exact_agreement_catches_one_wrong_factor(self):
        observation = th.arange(18, dtype=th.float32).reshape(3, 2, 3)
        terminated = th.zeros(3, 2, dtype=th.bool)
        truncated = th.zeros(3, 2, dtype=th.bool)
        action = th.zeros((3, 2, 7), dtype=th.long)
        teacher_action = action.clone()
        teacher_action[2, 1, 0] = 1
        batch = TensorBatch(
            {
                "observation": observation,
                "terminated": terminated,
                "truncated": truncated,
                "action": action,
                "teacher_action": teacher_action,
                "student_controlled": th.zeros(3, 2, dtype=th.bool),
            }
        )

        transformed = DistillRolloutTransform()(batch, None)

        # Per-factor agreement is 6/7 for the mismatched frame.
        self.assertAlmostEqual(
            transformed["action_agreement"][2, 1].item(), 6 / 7
        )
        # Exact agreement is zero for the mismatched frame.
        self.assertEqual(
            transformed["rollout_action_agreement"][2, 1].item(), 0.0
        )
        # All other frames agree exactly.
        self.assertEqual(
            transformed["rollout_action_agreement"][:2].sum().item(), 4.0
        )


if __name__ == "__main__":
    unittest.main()
