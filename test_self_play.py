import argparse
import copy
import math
import tempfile
import unittest

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th

from gymnasium.vector.utils import batch_space

from carl.gymnasium.state import CarlEvents, CarlState, RewardContext
from distill import ACTION_FORMAT, ActionDecoder, ConditionalPrior, GOAL_STATE_SIZE
from jarl.data.records import PolicyOutput
from jarl.modules import GRU, MLP
from jarl.modules.encoder import LinearEncoder
from rewards import AnnealedNextoReward, nexto_shaping_scale

from self_play import (
    FixedGaussianPolicy,
    FrozenPulseController,
    PulseLatentEnv,
    RaggedRolloutBuffer,
    SemiMarkovSelfPlayRunner,
    build_policy,
    load_demonstration_reset_dataset,
    policy_observation,
    primitive_discount,
    validate_args,
)


class AllValidActionCodec:
    def mask(self, state: th.Tensor) -> th.Tensor:
        return th.ones((*state.shape[:-1], 18), dtype=th.bool, device=state.device)


class FakeEnv:
    def __init__(self) -> None:
        self.n_envs = 2
        self.n_sim = 1
        self.device = th.device("cpu")
        self.single_observation_space = gym.spaces.Box(
            -1.0, 1.0, (GOAL_STATE_SIZE,), dtype="float32"
        )
        self.observation_space = batch_space(
            self.single_observation_space, self.n_envs
        )
        self.last_action = None

    def reset(self, **kwargs):
        return th.zeros((self.n_envs, GOAL_STATE_SIZE))

    def step(self, action):
        self.last_action = action
        observation = th.ones((self.n_envs, GOAL_STATE_SIZE))
        reward = th.zeros(self.n_envs)
        done = th.zeros(self.n_envs, dtype=th.bool)
        return observation, reward, done, done, {}

    def close(self):
        return


def make_controller(
    latent_size: int = 3,
    max_duration: int | None = None,
) -> FrozenPulseController:
    return FrozenPulseController(
        ConditionalPrior(
            GOAL_STATE_SIZE, latent_size, [8], max_duration=max_duration
        ),
        ActionDecoder(GOAL_STATE_SIZE, latent_size, [8]),
        AllValidActionCodec(),
    )


def make_reward_context(
    score_delta: int = 0,
    demoed_car: int | None = None,
) -> RewardContext:
    raw = th.zeros((1, 53))
    raw[:, 2] = 100.0
    raw[:, 9 + 9] = 1.0
    raw[:, 31 + 9] = 1.0
    current_raw = raw.clone()
    if demoed_car is not None:
        current_raw[:, 9 + 22 * demoed_car + 17] = 1.0
    team_sign = th.tensor([1.0, -1.0])
    previous = CarlState(raw, 2, th.empty((0, 3)), team_sign)
    current = CarlState(current_raw, 2, th.empty((0, 3)), team_sign)
    events = CarlEvents(
        score_delta=th.tensor([score_delta]),
        done=th.tensor([False]),
        terminated=th.tensor([False]),
        truncated=th.tensor([False]),
    )
    return RewardContext(
        current,
        previous,
        None,
        None,
        events,
        None,
        th.tensor([score_delta]),
        th.tensor([0]),
        th.tensor([False]),
    )


class SelfPlayTest(unittest.TestCase):
    def test_nexto_reward_keeps_weighted_zero_sum_goals_without_shaping(self):
        reward = AnnealedNextoReward(1, 1, shaping_scale=0.0)

        th.testing.assert_close(
            reward(make_reward_context(score_delta=1)), th.tensor([[10.0, -10.0]])
        )
        th.testing.assert_close(
            reward(make_reward_context(score_delta=0)), th.zeros((1, 2))
        )

    def test_nexto_shaping_is_not_opponent_centered(self):
        value = AnnealedNextoReward(1, 1)(make_reward_context())

        self.assertGreater(value.sum().item(), 0.0)

    def test_competitive_shaping_components_remain_zero_sum(self):
        baseline = AnnealedNextoReward(1, 1)(make_reward_context())
        demo = AnnealedNextoReward(1, 1)(make_reward_context(demoed_car=1))
        demo_delta = demo - baseline
        reward = AnnealedNextoReward(1, 1)
        context = make_reward_context(score_delta=1)
        win_progress = reward._win_probability_progress(
            context, context.current.team_sign[None, :]
        )

        th.testing.assert_close(demo_delta, th.tensor([[0.5, -0.5]]))
        th.testing.assert_close(win_progress.sum(dim=-1), th.zeros(1))

    def test_nexto_shaping_schedule_spans_the_full_training_run(self):
        self.assertEqual(nexto_shaping_scale(0, 1.0, 1000), 1.0)
        self.assertEqual(nexto_shaping_scale(500, 1.0, 1000), 0.5)
        self.assertEqual(nexto_shaping_scale(1000, 1.0, 1000), 0.0)

    def test_fixed_gaussian_policy_uses_requested_standard_deviation(self):
        env = PulseLatentEnv(FakeEnv(), make_controller())
        policy = FixedGaussianPolicy(
            LinearEncoder(8), MLP(dims=[8]), MLP(dims=[]), std=0.22
        ).build(env)
        observation = env.reset()

        output = policy.act(observation)
        evaluation = policy.evaluate_actions(observation, output.action)

        self.assertEqual(output.action.shape, (2, 3))
        self.assertEqual(output.log_prob.shape, (2,))
        self.assertEqual(evaluation.entropy.shape, (2,))
        th.testing.assert_close(policy.log_std.exp(), th.full((3,), 0.22))
        self.assertFalse(policy.log_std.requires_grad)

    def test_fixed_gaussian_policy_carries_state_across_32_step_sequences(self):
        env = PulseLatentEnv(FakeEnv(), make_controller())
        policy = FixedGaussianPolicy(
            LinearEncoder(8), GRU(hidden_size=4), MLP(dims=[]), std=0.22
        ).build(env)
        state = policy.initial_state(2)

        output = policy.act(th.ones((2, GOAL_STATE_SIZE)), state)
        observations = th.ones((32, 2, GOAL_STATE_SIZE))
        actions = th.zeros((32, 2, 3))
        evaluation = policy.evaluate_actions(
            observations,
            actions,
            state,
            reset=th.zeros((32, 2), dtype=th.bool),
        )

        self.assertEqual(state.shape, (2, 1, 4))
        self.assertEqual(output.next_state.shape, state.shape)
        self.assertEqual(evaluation.log_prob.shape, (32, 2))
        self.assertEqual(evaluation.entropy.shape, (32, 2))

    def test_compact_recurrent_policy_uses_requested_dimensions(self):
        env = PulseLatentEnv(FakeEnv(), make_controller())
        policy = build_policy(
            env,
            exploration_std=0.22,
            gru_hidden_size=7,
            gru_input_size=11,
        )

        self.assertEqual(policy.foot.feats, 11)
        self.assertEqual(policy.body.input_size, 11)
        self.assertEqual(policy.body.hidden_size, 7)

    def test_controller_is_frozen_and_decodes_latent_residuals(self):
        controller = make_controller()
        observation = th.zeros((2, GOAL_STATE_SIZE))
        captured = []
        hook = controller.decoder.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[1].clone())
        )

        residual = th.full((2, 3), 0.25)
        action = controller.decode(observation, residual)
        hook.remove()
        prior_mean, _ = controller.prior(observation)

        self.assertEqual(action.shape, (2, 7))
        th.testing.assert_close(captured[0], prior_mean + residual)
        self.assertTrue(all(not parameter.requires_grad for parameter in controller.parameters()))

    def test_demonstration_dataset_loads_safe_grounded_1v1_states(self):
        rows = np.zeros((3, 161), dtype=np.float32)
        rows[:, 2] = 100 / 2076
        cars = rows[:, 9:51].reshape(3, 2, 21)
        cars[..., 2] = 20 / 2076
        cars[..., 9] = 1
        cars[..., 14] = 1
        cars[..., 15] = 0.5
        cars[..., 16] = 1

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npy"
            np.save(path, rows)
            np.savez_compressed(
                path.with_suffix(".unsafe-starts.npz"),
                unsafe=np.zeros(3, dtype=bool),
                frame_skip=4,
            )

            dataset = load_demonstration_reset_dataset(
                Path(directory), "cpu", frame_skip=4
            )

        self.assertEqual(len(dataset), 3)
        sample = dataset[th.tensor([0])]
        th.testing.assert_close(sample["ball_position"][0, 2], th.tensor(100.0))
        th.testing.assert_close(sample["car_boost"], th.full((1, 2), 50.0))

    def test_controller_loads_distillation_artifact(self):
        source = make_controller()
        payload = {
            "prior": source.prior.state_dict(),
            "decoder": source.decoder.state_dict(),
            "config": {
                "action_format": ACTION_FORMAT,
                "latent_size": 3,
                "encoder_hidden": [8],
                "decoder_hidden": [8],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "distill.pt"
            th.save(payload, checkpoint)

            loaded = FrozenPulseController.load(
                checkpoint, AllValidActionCodec(), "cpu", bf16=True
            )

        self.assertTrue(loaded.bf16)
        for expected, actual in zip(source.parameters(), loaded.parameters()):
            th.testing.assert_close(expected, actual)

    def test_latent_environment_steps_with_decoded_actions(self):
        base_env = FakeEnv()
        env = PulseLatentEnv(base_env, make_controller())
        env.reset()

        observation, reward, terminated, truncated, info = env.step(
            th.zeros((2, 3))
        )

        self.assertEqual(base_env.last_action.shape, (2, 7))
        self.assertEqual(observation.shape, (2, GOAL_STATE_SIZE))
        self.assertEqual(reward.shape, (2,))
        self.assertFalse(terminated.any())
        self.assertFalse(truncated.any())
        self.assertEqual(info, {})

    def test_policy_observation_legacy_returns_physical(self):
        physical = th.randn(GOAL_STATE_SIZE)
        self.assertIs(policy_observation(physical, 5, None), physical)

    def test_policy_observation_appends_normalized_duration(self):
        physical = th.randn(GOAL_STATE_SIZE)
        observation = policy_observation(physical, 3, 8)

        self.assertEqual(observation.shape, (GOAL_STATE_SIZE + 1,))
        th.testing.assert_close(observation[:-1], physical)
        th.testing.assert_close(observation[-1], th.tensor(3.0 / 8.0))

    def test_new_artifact_observation_space_appends_duration(self):
        env = PulseLatentEnv(FakeEnv(), make_controller(max_duration=12))

        self.assertEqual(
            env.single_observation_space.shape, (GOAL_STATE_SIZE + 1,)
        )
        self.assertEqual(env.observation_space.shape, (2, GOAL_STATE_SIZE + 1))

    def test_primitive_discount(self):
        self.assertAlmostEqual(
            primitive_discount(4, 1.0), 2.0 ** (-4.0 / 120.0), places=6
        )
        self.assertAlmostEqual(
            primitive_discount(8, 2.0), 2.0 ** (-8.0 / 240.0), places=6
        )

    def test_controller_loads_duration_conditioned_artifact(self):
        source = make_controller(max_duration=12)
        payload = {
            "prior": source.prior.state_dict(),
            "decoder": source.decoder.state_dict(),
            "config": {
                "action_format": ACTION_FORMAT,
                "latent_size": 3,
                "encoder_hidden": [8],
                "decoder_hidden": [8],
                "frameskip": 4,
                "skill_horizon": 8,
                "skill_horizon_jitter": 4,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "distill.pt"
            th.save(payload, checkpoint)

            loaded = FrozenPulseController.load(
                checkpoint, AllValidActionCodec(), "cpu", frame_skip=4
            )

        self.assertEqual(loaded.max_duration, 12)
        self.assertEqual(loaded.skill_horizon, 8)
        self.assertEqual(loaded.skill_horizon_jitter, 4)
        for expected, actual in zip(source.parameters(), loaded.parameters()):
            th.testing.assert_close(expected, actual)

    def test_new_artifact_full_latent_remains_exact_across_observations(self):
        controller = make_controller(latent_size=3, max_duration=12)
        env = PulseLatentEnv(FakeEnv(), controller)
        observation = env.reset()
        residual = th.tensor([0.25, -0.10, 0.05])
        duration = 8
        full_latent = controller.select_latent(observation, residual, duration)

        captured = []
        hook = controller.decoder.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[1].clone())
        )
        env.step(full_latent)
        env.step(full_latent)
        hook.remove()

        self.assertEqual(len(captured), 2)
        for latent in captured:
            th.testing.assert_close(latent, full_latent)

    def test_ragged_rollout_buffer_pads_and_preserves_per_actor_order(self):
        buffer = RaggedRolloutBuffer(horizon=4, num_envs=2, device="cpu")
        buffer.append([0, 1], {"x": th.tensor([[1.0], [2.0]])})
        buffer.append([0], {"x": th.tensor([[3.0]])})

        rollout = buffer.finish()
        steps = rollout.steps

        self.assertEqual(steps["x"].shape, (2, 2, 1))
        th.testing.assert_close(steps["x"][:, 0], th.tensor([[1.0], [3.0]]))
        th.testing.assert_close(steps["x"][:, 1], th.tensor([[2.0], [0.0]]))
        expected_valid = th.tensor([[True, True], [True, False]])
        self.assertTrue(steps["valid"].equal(expected_valid))
        self.assertTrue(steps["learner_mask"].equal(expected_valid))

    def test_ragged_rollout_buffer_full_when_any_actor_reaches_horizon(self):
        buffer = RaggedRolloutBuffer(horizon=2, num_envs=2, device="cpu")
        self.assertFalse(buffer.full)
        self.assertEqual(buffer.position, 0)

        buffer.append([0, 1], {"x": th.tensor([[1.0], [2.0]])})
        self.assertFalse(buffer.full)

        buffer.append([0], {"x": th.tensor([[3.0]])})
        self.assertTrue(buffer.full)
        self.assertEqual(buffer.position, 2)

    def test_ragged_rollout_buffer_allows_faster_actor_to_exceed_horizon(self):
        buffer = RaggedRolloutBuffer(horizon=2, num_envs=2, device="cpu")
        buffer.append([0, 1], {"x": th.tensor([[1.0], [2.0]])})
        buffer.append([0], {"x": th.tensor([[3.0]])})
        self.assertTrue(buffer.full)
        buffer.append([0], {"x": th.tensor([[4.0]])})
        buffer.append([1], {"x": th.tensor([[5.0]])})

        self.assertTrue(buffer.full)
        self.assertEqual(buffer.position, 3)
        th.testing.assert_close(
            buffer.finish().steps["x"][:, 0],
            th.tensor([[1.0], [3.0], [4.0]]),
        )

    def test_legacy_distill_artifact_rejected_for_new_training(self):
        source = make_controller()
        payload = {
            "prior": source.prior.state_dict(),
            "decoder": source.decoder.state_dict(),
            "config": {
                "action_format": ACTION_FORMAT,
                "latent_size": 3,
                "encoder_hidden": [8],
                "decoder_hidden": [8],
                "frameskip": 4,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "distill.pt"
            th.save(payload, checkpoint)

            loaded = FrozenPulseController.load(
                checkpoint, AllValidActionCodec(), "cpu", frame_skip=4
            )

        self.assertIsNone(loaded.skill_horizon)
        self.assertIsNone(loaded.skill_horizon_jitter)
        args = argparse.Namespace(skill_horizon=16, skill_horizon_jitter=4)
        with self.assertRaises(ValueError):
            self._assert_skill_semantics_match(loaded, args)

    def test_continuous_action_artifact_is_rejected(self):
        source = make_controller()
        payload = {
            "prior": source.prior.state_dict(),
            "decoder": source.decoder.state_dict(),
            "config": {
                "action_format": "mixed-continuous-v1",
                "latent_size": 3,
                "encoder_hidden": [8],
                "decoder_hidden": [8],
                "frameskip": 4,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "distill.pt"
            th.save(payload, checkpoint)

            with self.assertRaisesRegex(RuntimeError, "incompatible action format"):
                FrozenPulseController.load(
                    checkpoint, AllValidActionCodec(), "cpu", frame_skip=4
                )

    def test_distill_artifact_skill_semantics_must_match_args(self):
        source = make_controller(max_duration=12)
        payload = {
            "prior": source.prior.state_dict(),
            "decoder": source.decoder.state_dict(),
            "config": {
                "action_format": ACTION_FORMAT,
                "latent_size": 3,
                "encoder_hidden": [8],
                "decoder_hidden": [8],
                "frameskip": 4,
                "skill_horizon": 8,
                "skill_horizon_jitter": 4,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "distill.pt"
            th.save(payload, checkpoint)

            loaded = FrozenPulseController.load(
                checkpoint, AllValidActionCodec(), "cpu", frame_skip=4
            )

        matching = argparse.Namespace(skill_horizon=8, skill_horizon_jitter=4)
        self._assert_skill_semantics_match(loaded, matching)

        mismatch = argparse.Namespace(skill_horizon=16, skill_horizon_jitter=4)
        with self.assertRaises(ValueError):
            self._assert_skill_semantics_match(loaded, mismatch)

    @staticmethod
    def _assert_skill_semantics_match(controller, args):
        if (
            controller.skill_horizon is None
            or controller.skill_horizon_jitter is None
        ):
            raise ValueError(
                "legacy distill artifact without skill_horizon and skill_horizon_jitter"
            )
        if (
            controller.skill_horizon != args.skill_horizon
            or controller.skill_horizon_jitter != args.skill_horizon_jitter
        ):
            raise ValueError(
                "distill artifact skill_horizon/skill_horizon_jitter do not match args"
            )

    def test_primitive_discount_uses_half_life_seconds(self):
        self.assertAlmostEqual(
            primitive_discount(4, 10.0), 2.0 ** (-4.0 / 1200.0), places=6
        )

    def test_validate_args_rejects_invalid_skill_options(self):
        with tempfile.TemporaryDirectory() as directory:
            distill = Path(directory) / "distill.pt"
            distill.touch()
            replay = Path(directory) / "replays"
            replay.mkdir()

            base = self._base_args(distill, replay)
            base.skill_horizon = 16
            base.skill_horizon_jitter = 4
            base.discount_half_life_seconds = 10.0
            validate_args(base)

            non_positive = copy.deepcopy(base)
            non_positive.skill_horizon = 0
            with self.assertRaises(ValueError):
                validate_args(non_positive)

            too_small_jitter = copy.deepcopy(base)
            too_small_jitter.skill_horizon = 4
            too_small_jitter.skill_horizon_jitter = 4
            with self.assertRaises(ValueError):
                validate_args(too_small_jitter)

    @staticmethod
    def _base_args(distill_checkpoint, replay_dir):
        return argparse.Namespace(
            n_sim=1,
            frameskip=4,
            max_ticks=1,
            rollout=8,
            batch_size=8,
            epochs=1,
            sequence_length=8,
            gru_input_size=8,
            gru_hidden_size=8,
            lr=1e-4,
            exploration_std=0.1,
            max_grad_norm=0.5,
            snapshot_interval=1,
            snapshot_pool_size=4,
            historical_policies=2,
            reset_state_limit=1,
            timesteps=16,
            checkpoint_interval=1,
            checkpoint_keep=1,
            skill_horizon=8,
            skill_horizon_jitter=2,
            discount_half_life_seconds=10.0,
            entropy_coef=0.0,
            bf16=False,
            current_fraction=0.5,
            demonstration_reset_fraction=0.0,
            nexto_shaping_scale=0.0,
            goal_reward_scale=1.0,
            distill_checkpoint=distill_checkpoint,
            replay_dir=replay_dir,
        )

    def test_ragged_rollout_buffer_preserves_learner_mask(self):
        buffer = RaggedRolloutBuffer(horizon=3, num_envs=2, device="cpu")
        buffer.append(
            [0, 1],
            {
                "x": th.tensor([[1.0], [2.0]]),
                "learner_mask": th.tensor([True, False]),
            },
        )
        buffer.append(
            [0],
            {
                "x": th.tensor([[3.0]]),
                "learner_mask": th.tensor([True]),
            },
        )

        steps = buffer.finish().steps
        expected_valid = th.tensor([[True, True], [True, False]])
        expected_learner_mask = th.tensor([[True, False], [True, False]])
        self.assertTrue(steps["valid"].equal(expected_valid))
        self.assertTrue(steps["learner_mask"].equal(expected_learner_mask))


class FakeSemiMarkovMatchmaker:
    def __init__(self, n_envs, device="cpu") -> None:
        self.n_envs = n_envs
        self.num_matches = n_envs
        self.team_sizes = (1, 1)
        self.players_per_match = 1
        self.device = th.device(device)
        self.current_fraction = 1.0
        self.learner_mask = th.ones(n_envs, dtype=th.bool, device=self.device)
        self.opponent_ids = th.full(
            (n_envs,), -1, dtype=th.int64, device=self.device
        )
        self.learner_count = n_envs

    def rematch(self, done=None):
        return


class TrackedFakePolicy:
    def __init__(self, latent_size, device="cpu") -> None:
        self.device = th.device(device)
        self.latent_size = latent_size
        self._calls = []

    @property
    def call_count(self):
        return len(self._calls)

    def initial_state(self, batch_size):
        return th.zeros((batch_size, 2, 4), device=self.device)

    def act(self, observation, state=None):
        self._calls.append(observation.shape[0])
        action = observation[:, : self.latent_size]
        next_state = state + 1 if state is not None else None
        return PolicyOutput(
            action=action,
            log_prob=th.zeros(observation.shape[0], device=self.device),
            next_state=next_state,
        )


class FakeCritic:
    def __init__(self, device="cpu") -> None:
        self.device = th.device(device)

    def initial_state(self, batch_size):
        return th.zeros((batch_size, 2, 4), device=self.device)

    def body_features(self, observation, state=None):
        features = observation.sum(dim=-1)
        next_state = state + 1 if state is not None else None
        return features, next_state

    def value_from_features(self, features):
        return features


class FakeSemiMarkovController:
    def __init__(self, latent_size, device="cpu") -> None:
        self.latent_size = latent_size
        self.device = th.device(device)
        self.max_duration = None

    def select_latent(self, observation, residual, duration=None):
        return residual + 1.0


class FakeSemiMarkovEnv:
    def __init__(self, n_envs, latent_size, done_at=None, device="cpu") -> None:
        self.n_envs = n_envs
        self.n_sim = n_envs
        self.device = th.device(device)
        self.single_observation_space = gym.spaces.Box(
            -1.0, 1.0, (GOAL_STATE_SIZE,), dtype="float32"
        )
        self.observation_space = batch_space(self.single_observation_space, n_envs)
        self.single_action_space = gym.spaces.Box(
            -math.inf, math.inf, (latent_size,), dtype="float32"
        )
        self.action_space = batch_space(self.single_action_space, n_envs)
        self.done_at = done_at or {}
        self._observation = None
        self._step_count = 0
        self.step_actions = []

    def reset(self, **kwargs):
        self._observation = th.zeros(
            (self.n_envs, GOAL_STATE_SIZE), dtype=th.float32, device=self.device
        )
        self._step_count = 0
        return self._observation

    def step(self, action):
        self.step_actions.append(action.clone())
        self._step_count += 1
        self._observation = self._observation + 0.1
        reward = th.ones(self.n_envs, dtype=th.float32, device=self.device)
        terminated = th.zeros(self.n_envs, dtype=th.bool, device=self.device)
        for index, at in self.done_at.items():
            if self._step_count == at:
                terminated[index] = True
        truncated = th.zeros(self.n_envs, dtype=th.bool, device=self.device)
        return self._observation, reward, terminated, truncated, {}

    def close(self):
        return


class TestSemiMarkovSelfPlayRunner(unittest.TestCase):
    def _make_runner(
        self, n_envs=2, horizon=4, jitter=0, done_at=None, gamma=0.9, seed=0
    ):
        latent_size = 3
        env = FakeSemiMarkovEnv(n_envs=n_envs, latent_size=latent_size, done_at=done_at)
        policy = TrackedFakePolicy(latent_size, device=env.device)
        critic = FakeCritic(device=env.device)
        controller = FakeSemiMarkovController(latent_size, device=env.device)
        buffer = RaggedRolloutBuffer(horizon=20, num_envs=n_envs, device=env.device)
        matchmaker = FakeSemiMarkovMatchmaker(n_envs, device=env.device)
        runner = SemiMarkovSelfPlayRunner(
            env,
            policy,
            critic,
            controller,
            buffer,
            gamma=gamma,
            skill_horizon=horizon,
            skill_horizon_jitter=jitter,
            seed=seed,
            matchmaker=matchmaker,
        )
        return runner, env, policy, critic, controller, buffer

    def test_policy_called_only_at_skill_boundaries(self):
        runner, env, policy, critic, controller, buffer = self._make_runner(
            n_envs=2, horizon=3, jitter=0, gamma=0.9
        )
        runner.reset()

        for _ in range(7):
            runner.step()

        # With horizon=3, policy is invoked at the first step of each skill.
        self.assertEqual(policy.call_count, 3)
        # Actions within a skill are identical; they change at the next boundary.
        self.assertTrue(th.allclose(env.step_actions[0], env.step_actions[1]))
        self.assertTrue(th.allclose(env.step_actions[1], env.step_actions[2]))
        self.assertFalse(th.allclose(env.step_actions[2], env.step_actions[3]))
        self.assertTrue(th.allclose(env.step_actions[3], env.step_actions[4]))
        self.assertTrue(th.allclose(env.step_actions[4], env.step_actions[5]))
        self.assertFalse(th.allclose(env.step_actions[5], env.step_actions[6]))

    def test_discounted_reward_aggregated_over_skill(self):
        runner, env, policy, critic, controller, buffer = self._make_runner(
            n_envs=2, horizon=2, jitter=0, gamma=0.9
        )
        runner.reset()

        for _ in range(2):
            runner.step()

        rollout = buffer.finish()
        steps = rollout.steps
        # Reward is 1 per primitive step; two-step skill => 1 + 0.9*1 = 1.9
        expected = th.tensor([1.9, 1.9])
        th.testing.assert_close(steps["reward"][0], expected)

    def test_rollout_cut_closes_partial_skills_before_update(self):
        runner, env, policy, critic, controller, buffer = self._make_runner(
            n_envs=2, horizon=2, jitter=0, gamma=0.9
        )
        buffer.horizon = 1
        runner.reset()
        runner._planned_duration[:] = th.tensor([1, 2])

        runner.step()

        self.assertTrue(buffer.full)
        th.testing.assert_close(buffer.counts, th.tensor([1, 1]))
        th.testing.assert_close(runner._elapsed, th.zeros(2, dtype=th.int64))
        steps = buffer.finish().steps
        self.assertEqual(steps["duration"][0, 1].item(), 1)
        self.assertFalse(steps["terminated"][0, 1])
        self.assertFalse(steps["truncated"][0, 1])

    def test_early_done_records_realized_duration(self):
        runner, env, policy, critic, controller, buffer = self._make_runner(
            n_envs=2, horizon=4, jitter=0, gamma=0.9, done_at={0: 2}
        )
        runner.reset()

        for _ in range(2):
            runner.step()

        rollout = buffer.finish()
        steps = rollout.steps
        # Actor 0 terminated early after 2 primitive steps; actor 1 not finished.
        self.assertEqual(steps["duration"][0, 0].item(), 2)
        self.assertFalse(steps["valid"][0, 1].item())

    def test_queued_duration_used_at_next_boundary(self):
        runner, env, policy, critic, controller, buffer = self._make_runner(
            n_envs=1, horizon=4, jitter=0, gamma=0.9
        )
        runner.reset()

        for _ in range(4):
            runner.step()

        # First skill completed at step 3; next boundary is step 4.
        rollout = buffer.finish()
        steps = rollout.steps
        max_duration = 4
        queued_duration = int(steps["next_obs"][0, 0, -1].item() * max_duration)
        self.assertGreaterEqual(queued_duration, 1)
        self.assertLessEqual(queued_duration, max_duration)

        # The next boundary uses the queued duration as its planned duration.
        runner.step()
        self.assertEqual(runner._planned_duration[0].item(), queued_duration)

    def test_queued_duration_reset_on_done(self):
        runner, env, policy, critic, controller, buffer = self._make_runner(
            n_envs=1, horizon=4, jitter=0, gamma=0.9, done_at={0: 2}
        )
        runner.reset()

        runner.step()
        runner.step()

        # After a done, the queued/planned durations are reset to sentinel values.
        self.assertEqual(runner._queued_duration[0].item(), -1)
        self.assertEqual(runner._planned_duration[0].item(), -1)
        self.assertEqual(runner._elapsed[0].item(), 0)

    def test_duration_sampling_stays_within_skill_jitter_range(self):
        runner, env, policy, critic, controller, buffer = self._make_runner(
            n_envs=8, horizon=8, jitter=3, gamma=0.9, seed=7
        )
        runner.reset()

        for _ in range(20):
            runner.step()

        rollout = buffer.finish()
        steps = rollout.steps
        observed_durations = steps["duration"][steps["valid"]]
        self.assertTrue((observed_durations >= 5).all().item())
        self.assertTrue((observed_durations <= 11).all().item())


if __name__ == "__main__":
    unittest.main()
