import unittest

from pathlib import Path
from types import SimpleNamespace

import torch as th
import torch.nn as nn

from distill import (
    ACTION_SIZES,
    ActionDecoder,
    ConditionalPrior,
    DistillCheckpoints,
    DistillRolloutTransform,
    GaussianEncoder,
    PulseLoss,
    PulsePolicy,
    TrajectoryChunkMinibatches,
    categorical_distillation_loss,
    diagonal_gaussian_kl,
    factor_actions,
    kl_coefficient,
    validate_args,
)
from jarl.data.batch import TensorBatch
from tracker import GOAL_STATE_SIZE


class AllValidActionCodec:
    def mask(self, state: th.Tensor) -> th.Tensor:
        return th.ones(
            (*state.shape[:-1], 18), dtype=th.bool, device=state.device
        )


class RecordingDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[th.Tensor, th.Tensor]] = []

    def forward(self, state: th.Tensor, latent: th.Tensor) -> th.Tensor:
        self.calls.append((state.detach().clone(), latent.detach().clone()))
        return th.zeros(
            state.shape[0], 18, device=state.device, dtype=state.dtype
        )


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


class CategoricalDistillationLossTest(unittest.TestCase):
    def test_factorized_action_loss_and_argmax(self):
        target = th.tensor([[2, 1, 0, 1, 0, 2, 1]])
        logits = th.full((1, 18), -5.0)
        offset = 0
        for size, value in zip((3, 3, 3, 2, 2, 3, 2), target[0]):
            logits[0, offset + value] = 5.0
            offset += size

        loss, accuracy = categorical_distillation_loss(logits, target)

        th.testing.assert_close(factor_actions(logits), target)
        self.assertLess(loss.item(), 0.001)
        self.assertEqual(accuracy.item(), 1.0)

    def test_valid_mask_ignores_invalid_frames(self):
        target = th.tensor([[2, 1, 0, 1, 0, 2, 1]])
        logits = th.full((4, 18), -5.0)
        for row, value in enumerate(target):
            offset = 0
            for size, action in zip((3, 3, 3, 2, 2, 3, 2), target[0]):
                logits[row, offset + action] = 5.0
                offset += size

        masked_loss, masked_accuracy = categorical_distillation_loss(
            logits, target.repeat(4, 1), valid=th.tensor([True, True, False, False])
        )
        full_loss, full_accuracy = categorical_distillation_loss(
            logits[:2], target.repeat(2, 1)
        )

        th.testing.assert_close(masked_loss, full_loss)
        th.testing.assert_close(masked_accuracy, full_accuracy)

    def test_all_invalid_frames_return_zero(self):
        target = th.zeros((4, 7), dtype=th.long)
        logits = th.zeros((4, 18))

        loss, accuracy = categorical_distillation_loss(
            logits, target, valid=th.zeros(4, dtype=th.bool)
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(accuracy.item(), 0.0)


class PulseLossTest(unittest.TestCase):
    def test_shared_latent_decodes_at_every_valid_frame(self):
        th.manual_seed(0)
        latent_size = 2
        encoder = GaussianEncoder(GOAL_STATE_SIZE, latent_size, [16])
        decoder = RecordingDecoder()
        policy = PulsePolicy(encoder, decoder, AllValidActionCodec())
        prior = ConditionalPrior(GOAL_STATE_SIZE, latent_size, [16], max_duration=4)
        loss = PulseLoss(
            policy,
            prior,
            AllValidActionCodec(),
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
        self.assertIn("kl", output.metrics)

        self.assertEqual(len(decoder.calls), 2)
        for state, latent in decoder.calls:
            self.assertEqual(state.shape, (3, GOAL_STATE_SIZE))
            self.assertEqual(latent.shape, (3, latent_size))
            th.testing.assert_close(latent[0], latent[1])


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
            }
        )

        transformed = DistillRolloutTransform()(batch, None)

        self.assertIn("action_agreement", transformed)
        self.assertIn("reset_fraction", transformed)
        self.assertAlmostEqual(
            transformed["action_agreement"][2, 1].item(), 6 / 7
        )
        th.testing.assert_close(
            transformed["reset_fraction"],
            (terminated | truncated).float(),
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
            minimum_remaining_frames=128,
            ball_outcome_weight=0.1,
            skill_horizon=16,
            skill_horizon_jitter=4,
            prior_action_weight=1.0,
            kl_anneal_start=0,
            kl_anneal_end=10,
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

    def test_prior_action_weight_must_be_non_negative(self):
        args = self._valid_args()
        args.prior_action_weight = -0.1
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


if __name__ == "__main__":
    unittest.main()
