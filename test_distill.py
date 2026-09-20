import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn

from distill import (
    ACTION_SIZES,
    ActionDecoder,
    ConditionalPrior,
    ConsecutiveFrameMinibatches,
    DeterministicTeacher,
    DistillCheckpoints,
    DistillRolloutTransform,
    DistillationRunner,
    GaussianEncoder,
    PulseLoss,
    PulsePolicy,
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
from tracker import CONTROL_STATE_SIZE, GOAL_STATE_SIZE


class AllValidActionCodec:
    def mask(self, state: th.Tensor) -> th.Tensor:
        return th.ones(
            (*state.shape[:-1], sum(ACTION_SIZES)),
            dtype=th.bool,
            device=state.device,
        )


class GaussianEncoderTest(unittest.TestCase):
    def test_forward_returns_per_frame_diagonal_gaussian(self):
        encoder = GaussianEncoder(30, 4, [16])
        mean, log_variance = encoder(th.randn(7, 30))

        self.assertEqual(mean.shape, (7, 4))
        self.assertEqual(log_variance.shape, (7, 4))
        self.assertTrue((log_variance <= 2.0).all())
        self.assertTrue((log_variance >= -5.0).all())

    def test_encoder_is_feed_forward(self):
        encoder = GaussianEncoder(30, 4, [16])

        self.assertFalse(hasattr(encoder, "segment_gru"))
        self.assertFalse(hasattr(encoder, "segment"))

    def test_encoder_has_no_action_conditioning(self):
        encoder = GaussianEncoder(30, 4, [16])
        mean, _ = encoder(th.zeros(1, 30))

        self.assertEqual(mean.shape, (1, 4))


class ConditionalPriorTest(unittest.TestCase):
    def test_forward_returns_per_state_diagonal_gaussian(self):
        prior = ConditionalPrior(GOAL_STATE_SIZE, 3, [8])
        mean, log_variance = prior(th.randn(5, GOAL_STATE_SIZE))

        self.assertEqual(mean.shape, (5, 3))
        self.assertEqual(log_variance.shape, (5, 3))

    def test_empty_hidden_dimensions_raise(self):
        with self.assertRaisesRegex(ValueError, "hidden"):
            ConditionalPrior(GOAL_STATE_SIZE, 3, [])

    def test_prior_depends_on_each_frame_state(self):
        prior = ConditionalPrior(GOAL_STATE_SIZE, 3, [8])
        state = th.zeros(4, GOAL_STATE_SIZE)
        state[1, 0] = 1.0
        mean, _ = prior(state)

        self.assertFalse(th.allclose(mean[0], mean[1]))


class ActionDecoderTest(unittest.TestCase):
    def test_decoder_concatenates_state_and_latent(self):
        decoder = ActionDecoder(CONTROL_STATE_SIZE, 4, [16])
        logits = decoder(th.randn(3, CONTROL_STATE_SIZE), th.randn(3, 4))

        self.assertEqual(logits.shape, (3, sum(ACTION_SIZES)))


class PulsePolicyTest(unittest.TestCase):
    def test_student_action_uses_explicit_control_state(self):
        decoder = ActionDecoder(CONTROL_STATE_SIZE, 3, [8])
        policy = PulsePolicy(
            GaussianEncoder(CONTROL_STATE_SIZE + 8, 3, [8]),
            decoder,
            AllValidActionCodec(),
        )
        observation = th.randn(2, CONTROL_STATE_SIZE + 8)
        control_state = th.randn(2, CONTROL_STATE_SIZE)

        action = policy.student_action(observation, control_state)

        self.assertEqual(action.shape, (2, 7))

    def test_act_slices_control_state_from_observation(self):
        decoder = ActionDecoder(GOAL_STATE_SIZE, 3, [8])
        policy = PulsePolicy(
            GaussianEncoder(GOAL_STATE_SIZE, 3, [8]),
            decoder,
            AllValidActionCodec(),
        )

        output = policy.act(th.randn(2, GOAL_STATE_SIZE), deterministic=True)

        self.assertEqual(output.action.shape, (2, 7))

    def test_act_rejects_recurrent_state(self):
        policy = PulsePolicy(
            GaussianEncoder(GOAL_STATE_SIZE, 3, [8]),
            ActionDecoder(GOAL_STATE_SIZE, 3, [8]),
            AllValidActionCodec(),
        )
        with self.assertRaisesRegex(ValueError, "recurrent"):
            policy.act(th.randn(2, GOAL_STATE_SIZE), state=th.zeros(2, 4))


class CategoricalDistillationLossTest(unittest.TestCase):
    def test_factorized_action_loss_and_argmax(self):
        logits = th.zeros(2, sum(ACTION_SIZES))
        target = th.zeros(2, 7, dtype=th.long)
        for offset in (0, 8, 17, 25, 30, 35, 41):
            logits[:, offset] = 10.0

        loss, accuracy = categorical_distillation_loss(logits, target)

        self.assertLess(loss.item(), 1e-3)
        self.assertEqual(accuracy.item(), 1.0)

    def test_exact_accuracy_requires_every_factor(self):
        logits = th.zeros(3, sum(ACTION_SIZES))
        target = th.zeros(3, 7, dtype=th.long)
        for offset in (0, 8, 17, 25, 30, 35, 41):
            logits[:, offset] = 10.0
        logits[1, 10] = 10.0

        self.assertAlmostEqual(exact_action_accuracy(logits, target).item(), 2 / 3)

    def test_action_factors_round_trip_through_one_hot(self):
        action = th.tensor([[0, 1, 2, 0, 1, 0, 1]])

        encoded = encode_action_factors(action)

        self.assertEqual(encoded.shape, (1, sum(ACTION_SIZES)))
        self.assertEqual(factor_actions(encoded).tolist(), action.tolist())

    def test_masked_logits_blocks_illegal_actions(self):
        logits = th.zeros(1, sum(ACTION_SIZES))
        state = th.zeros(1, CONTROL_STATE_SIZE)
        codec = AllValidActionCodec()
        codec.mask = lambda state: th.zeros_like(logits, dtype=th.bool)

        masked = masked_logits(logits, state, codec)

        self.assertTrue((masked <= th.finfo(masked.dtype).min / 2).all())


class KlTest(unittest.TestCase):
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


class PulseLossTest(unittest.TestCase):
    @staticmethod
    def _loss(kl_weight: float = 0.0, regu_weight: float = 0.0) -> PulseLoss:
        policy = PulsePolicy(
            GaussianEncoder(CONTROL_STATE_SIZE, 3, [16]),
            ActionDecoder(CONTROL_STATE_SIZE, 3, [16]),
            AllValidActionCodec(),
        )
        prior = ConditionalPrior(CONTROL_STATE_SIZE, 3, [16])
        return PulseLoss(policy, prior, AllValidActionCodec(), kl_weight, regu_weight)

    @staticmethod
    def _pair_batch(n: int = 4, shift: float = 0.0) -> TensorBatch:
        observation = th.randn(2, n, CONTROL_STATE_SIZE)
        observation[1] = observation[0] + shift
        return TensorBatch({
            "observation": observation,
            "control_state": observation.clone(),
            "teacher_action": th.zeros(2, n, 7, dtype=th.long),
        })

    def test_loss_decodes_every_frame_with_its_own_latent(self):
        loss = self._loss()
        batch = self._pair_batch()

        output = loss(batch)

        self.assertEqual(output.metrics["action_accuracy"].shape, th.Size(()))
        self.assertTrue(th.isfinite(output.loss))

    def test_loss_requires_consecutive_frame_pairs(self):
        loss = self._loss()
        flat = TensorBatch({
            "observation": th.randn(8, CONTROL_STATE_SIZE),
            "teacher_action": th.zeros(8, 7, dtype=th.long),
        })

        with self.assertRaisesRegex(ValueError, "pairs"):
            loss(flat)

    def test_kl_weight_scales_the_total_loss(self):
        batch = self._pair_batch()
        zero = self._loss(kl_weight=0.0)(batch)
        th.testing.assert_close(zero.loss, zero.metrics["action_loss"])

        weighted = self._loss(kl_weight=0.01)(batch)
        th.testing.assert_close(
            weighted.loss,
            weighted.metrics["action_loss"] + 0.01 * weighted.metrics["kl"],
        )

    def test_regu_penalizes_consecutive_latent_deviation(self):
        batch = self._pair_batch()
        identical = self._pair_batch(shift=0.0)
        shifted = self._pair_batch(shift=1.0)
        plain = self._loss(kl_weight=0.0, regu_weight=0.005)

        smooth = plain(identical)
        deviating = plain(shifted)

        th.testing.assert_close(
            smooth.metrics["latent_regu_loss"],
            th.zeros(()),
        )
        th.testing.assert_close(
            smooth.loss,
            smooth.metrics["action_loss"],
        )
        self.assertGreater(
            deviating.metrics["latent_regu_loss"].item(),
            smooth.metrics["latent_regu_loss"].item(),
        )
        th.testing.assert_close(
            deviating.loss,
            deviating.metrics["action_loss"]
            + 0.005 * deviating.metrics["latent_regu_loss"],
        )

    def test_loss_backpropagates_into_encoder_decoder_and_prior(self):
        loss = self._loss(kl_weight=0.1, regu_weight=0.005)
        batch = self._pair_batch()

        output = loss(batch)
        output.loss.backward()

        for parameter in loss.policy.encoder.parameters():
            self.assertIsNotNone(parameter.grad)
        for parameter in loss.policy.decoder.parameters():
            self.assertIsNotNone(parameter.grad)
        for parameter in loss.prior.parameters():
            self.assertIsNotNone(parameter.grad)


class DistillRolloutTransformTest(unittest.TestCase):
    def test_keeps_useful_metrics(self):
        observation = th.arange(18, dtype=th.float32).reshape(3, 2, 3)
        terminated = th.zeros(3, 2, dtype=th.bool)
        truncated = th.zeros(3, 2, dtype=th.bool)
        action = th.zeros((3, 2, 7), dtype=th.long)
        teacher_action = action.clone()
        teacher_action[2, 1, 0] = 1
        batch = TensorBatch({
            "observation": observation,
            "terminated": terminated,
            "truncated": truncated,
            "action": action,
            "teacher_action": teacher_action,
        })

        transformed = DistillRolloutTransform()(batch, None)

        self.assertAlmostEqual(
            transformed["action_agreement"][2, 1].item(), 6 / 7
        )
        self.assertEqual(
            transformed["rollout_action_agreement"][2, 1].item(), 0.0
        )
        self.assertEqual(
            transformed["rollout_action_agreement"][:2].sum().item(), 4.0
        )


class ConsecutiveFrameMinibatchesTest(unittest.TestCase):
    @staticmethod
    def _rollout(
        time: int = 3,
        num_envs: int = 2,
        terminated: dict[tuple[int, int], bool] | None = None,
    ) -> TensorBatch:
        observation = (
            th.arange(time * num_envs, dtype=th.float32)
            .reshape(time, num_envs, 1)
        )
        done = th.zeros(time, num_envs, dtype=th.bool)
        for (t, env), value in (terminated or {}).items():
            done[t, env] = value
        return TensorBatch({
            "observation": observation,
            "terminated": done,
            "truncated": th.zeros(time, num_envs, dtype=th.bool),
        })

    def test_pairs_are_consecutive_within_an_environment(self):
        sampler = ConsecutiveFrameMinibatches(batch_size=16, epochs=1)

        batches = list(sampler(self._rollout(time=3, num_envs=2)))

        self.assertEqual(len(batches), 1)
        pair = batches[0]["observation"]
        self.assertEqual(pair.shape, (2, 4, 1))
        th.testing.assert_close(pair[1] - pair[0], th.full((4, 1), 10.0))
        th.testing.assert_close(pair[0] % 10.0, pair[1] % 10.0)

    def test_pairs_never_cross_episode_boundaries(self):
        sampler = ConsecutiveFrameMinibatches(batch_size=16, epochs=1)

        batches = list(sampler(self._rollout(
            time=3,
            num_envs=2,
            terminated={(1, 0): True},
        )))

        pair = batches[0]["observation"]
        self.assertEqual(pair.shape[1], 3)
        # Frame (t=1, env=0) ends its episode, so it must never be a pair start.
        self.assertEqual((pair[0] == 10.0).sum().item(), 0)

    def test_epochs_yield_separate_batches(self):
        sampler = ConsecutiveFrameMinibatches(batch_size=4, epochs=2)

        batches = list(sampler(self._rollout(time=3, num_envs=2)))

        self.assertEqual(len(batches), 2)
        for pair in batches:
            self.assertEqual(pair.shape, (2, 4, 1))

    def test_all_done_rollout_raises(self):
        sampler = ConsecutiveFrameMinibatches(batch_size=4, epochs=1)
        done = {(t, env): True for t in range(2) for env in range(2)}

        with self.assertRaisesRegex(RuntimeError, "no consecutive frame pairs"):
            list(sampler(self._rollout(time=2, num_envs=2, terminated=done)))

    def test_single_step_rollout_raises(self):
        sampler = ConsecutiveFrameMinibatches(batch_size=4, epochs=1)

        with self.assertRaisesRegex(RuntimeError, "no consecutive frame pairs"):
            list(sampler(self._rollout(time=1, num_envs=2)))

    def test_invalid_arguments(self):
        with self.assertRaisesRegex(ValueError, "minibatch"):
            ConsecutiveFrameMinibatches(batch_size=0, epochs=1)
        with self.assertRaisesRegex(ValueError, "minibatch"):
            ConsecutiveFrameMinibatches(batch_size=4, epochs=0)


class ArgumentValidationTest(unittest.TestCase):
    def _args(self, **overrides):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        checkpoint = Path(directory.name) / "tracker.pt"
        checkpoint.touch()
        args = SimpleNamespace(
            tracker_checkpoint=checkpoint,
            resume=None,
            n_sim=2,
            frameskip=4,
            latent_size=4,
            rollout=4,
            batch_size=8,
            epochs=1,
            minimum_tracking_frames=1,
            minimum_remaining_frames=1,
            timesteps=100,
            checkpoint_interval=10,
            checkpoint_keep=1,
            kl_initial=0.01,
            kl_final=0.001,
            kl_anneal_start=0,
            kl_anneal_end=100,
            regu_weight=0.005,
        )
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    def test_valid_args_pass(self):
        validate_args(self._args())

    def test_kl_weight_must_be_finite_and_non_negative(self):
        with self.assertRaisesRegex(ValueError, "kl-initial"):
            validate_args(self._args(kl_initial=-1.0))
        with self.assertRaisesRegex(ValueError, "kl-final"):
            validate_args(self._args(kl_final=float("nan")))

    def test_regu_weight_must_be_finite_and_non_negative(self):
        with self.assertRaisesRegex(ValueError, "regu-weight"):
            validate_args(self._args(regu_weight=-0.001))
        with self.assertRaisesRegex(ValueError, "regu-weight"):
            validate_args(self._args(regu_weight=float("inf")))

    def test_anneal_window_must_be_ordered_and_within_budget(self):
        with self.assertRaisesRegex(ValueError, "kl-anneal-end"):
            validate_args(self._args(kl_anneal_start=50, kl_anneal_end=50))
        with self.assertRaisesRegex(ValueError, "kl-anneal-end"):
            validate_args(self._args(kl_anneal_end=101))

    def test_positive_arguments_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "latent-size"):
            validate_args(self._args(latent_size=0))
        with self.assertRaisesRegex(ValueError, "rollout"):
            validate_args(self._args(rollout=0))

    def test_missing_tracker_checkpoint_raises(self):
        args = self._args()
        args.tracker_checkpoint = Path("does-not-exist.pt")
        with self.assertRaises(FileNotFoundError):
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

    def test_resume_requires_matching_opponent_context_manifest(self):
        validate_control_manifest(("replay:a",), ("replay:a",))
        with self.assertRaisesRegex(ValueError, "replay segments"):
            validate_control_manifest(("replay:a",), ("replay:b",))
        with self.assertRaisesRegex(ValueError, "replay segments"):
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
            th.arange(
                self.n_envs * self._obs.shape[-1],
                dtype=th.float32,
                device=self.device,
            ).reshape(self._obs.shape)
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
        return


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


class RecordingFakePulsePolicy(nn.Module):
    def __init__(self, latent_size: int) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.decoder_calls: list[tuple[th.Tensor, th.Tensor]] = []

    def student_action(self, observation, control_state, *, deterministic=False):
        latent = th.zeros(
            observation.shape[0],
            self.latent_size,
            device=observation.device,
            dtype=observation.dtype,
        )
        self.decoder_calls.append(
            (control_state.detach().clone(), latent.clone())
        )
        return th.zeros(
            observation.shape[0],
            7,
            dtype=th.long,
            device=observation.device,
        )


class DistillationRunnerTest(unittest.TestCase):
    def _build_runner(
        self,
        n_envs: int = 2,
        latent_size: int = 3,
        done_steps: dict[int, int] | None = None,
        teacher_action: th.Tensor | None = None,
    ):
        env = FakeDistillEnv(n_envs, CONTROL_STATE_SIZE, done_steps=done_steps)
        if teacher_action is None:
            teacher_action = th.ones(n_envs, 7, dtype=th.long)
        teacher = DeterministicTeacher(
            FakeRecurrentTeacher(teacher_action, state_dim=1)
        )
        policy = RecordingFakePulsePolicy(latent_size)
        buffer = RolloutBuffer(horizon=16, num_envs=n_envs, device="cpu")
        runner = DistillationRunner(
            env,
            teacher,
            policy,
            buffer,
            seed=0,
        )
        return runner, env, policy, buffer

    def test_student_controls_every_step_and_teacher_labels_are_stored(self):
        runner, env, policy, buffer = self._build_runner(n_envs=2)
        runner.reset()
        for _ in range(4):
            runner.step()

        rollout = buffer.finish().steps
        self.assertTrue((rollout["teacher_action"] == 1).all())
        self.assertTrue((rollout["action"] == 0).all())
        self.assertEqual(rollout["control_state"].shape, (4, 2, CONTROL_STATE_SIZE))
        self.assertEqual(len(policy.decoder_calls), 4)

    def test_done_resets_teacher_state(self):
        teacher_action = th.empty(2, 7, dtype=th.long)
        teacher_action[0] = 0
        teacher_action[1] = 1
        runner, env, policy, buffer = self._build_runner(
            n_envs=2,
            done_steps={0: 2},
            teacher_action=teacher_action,
        )
        runner.reset()
        runner.step()
        self.assertTrue((runner._teacher_state == 1).all())
        runner.step()
        self.assertEqual(runner._teacher_state[0].item(), 0)
        self.assertEqual(runner._teacher_state[1].item(), 2)


if __name__ == "__main__":
    unittest.main()
