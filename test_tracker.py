import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th
import gymnasium as gym

from carl.gymnasium import CARLObservation
from carl.gymnasium.action import ACTION_NVECS, CARLActionCodec
from jarl.data import PolicyOutput, TensorBatch

from ballchasing_replays.parse_replays import _project_carl_actions
from watch_demonstrations import frame_from_state, publish_frame
from tracker_checkpoint import PHCCheckpoint, PeriodicCheckpoint

from tracker import (
    ACTION_FACTORS,
    BALL_MAX_ANG_SPEED,
    BALL_MAX_SPEED,
    CONTROL_STATE_SIZE,
    DEFAULT_TRACKER_WINDOWS,
    EXPERT_TOUCH_INDEX,
    ExpertGoalStates,
    ExpertLookaheadEnv,
    GOAL_STATE_SIZE,
    INTERNAL_STATE_SIZE,
    OPPONENT_STATE_INDEX,
    OPPONENT_STATE_SIZE,
    PHC_TRACKER_ARCHITECTURE,
    POSITION_SCALE,
    RAW_ACTION_INDEX,
    RAW_ACTION_SIZE,
    RAW_JUMP_INDEX,
    RoutedTrackerPolicy,
    SegmentScores,
    StatelessCriticCapture,
    STORED_REPLAY_SIZE,
    TrackingReward,
    annealed_value,
    build_tracker_policy,
    evaluate_tracker_policy,
    load_tracker_policy,
    set_learning_rate,
    specialist_assignments,
    validated_replay_assignments,
    validate_args,
    _expert_jump_loss,
)


class TrackerTest(unittest.TestCase):
    def test_hyperparameter_schedule_caps_at_configured_steps(self):
        self.assertEqual(annealed_value(0.0, 1.0, 0.1, 10_000, 1_000), 1.0)
        self.assertAlmostEqual(
            annealed_value(0.05, 1.0, 0.1, 10_000, 1_000), 0.55
        )
        self.assertAlmostEqual(
            annealed_value(0.5, 1.0, 0.1, 10_000, 1_000), 0.1
        )

    def test_tracker_hyperparameter_validation(self):
        args = SimpleNamespace(
            n_sim=256,
            frameskip=2,
            rollout=64,
            batch_size=16_384,
            epochs=4,
            sequence_length=64,
            timesteps=6_000_000,
            stage_timesteps=1_000_000,
            hard_negative_fraction=0.8,
            schedule_timesteps=1_000,
            gamma=0.997,
            gae_lambda=0.98,
            lr=1e-4,
            lr_final=1e-5,
            entropy_coef=1e-3,
            entropy_coef_final=1e-4,
            jump_imitation_weight=0.1,
            second_jump_weight=8.0,
            clip=0.2,
            clip_final=0.1,
            max_grad_norm=0.5,
            tracking_progress_scale=4.0,
        )
        validate_args(args)

        args.gae_lambda = 0
        with self.assertRaisesRegex(ValueError, "gae-lambda"):
            validate_args(args)

        args.gae_lambda = 0.98
        args.tracking_progress_scale = -1
        with self.assertRaisesRegex(ValueError, "tracking-progress-scale"):
            validate_args(args)

        args.tracking_progress_scale = 4.0
        args.timesteps = 6_000_001
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_args(args)

    def test_detects_airborne_second_jump_press_edges(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._replays = th.zeros((3, STORED_REPLAY_SIZE))
        replays._offsets = th.tensor([0, 3])
        replays._demo_id = th.tensor([0])
        replays._cursors = th.tensor([2])
        replays._replays[1, GOAL_STATE_SIZE + 3] = 1
        replays._replays[1, RAW_ACTION_INDEX + RAW_JUMP_INDEX] = 1

        jump, second_jump = replays.current_jump_supervision(offset=-1)

        th.testing.assert_close(jump, th.tensor([1]))
        th.testing.assert_close(second_jump, th.tensor([True]))

    def test_jump_imitation_loss_upweights_second_jump_edges(self):
        logits = th.zeros((2, sum(ACTION_NVECS)))
        logits[1, -1] = 5
        logits.requires_grad_()

        loss, accuracy, recall, sample_rate = _expert_jump_loss(
            logits,
            th.ones_like(logits, dtype=th.bool),
            th.tensor([0, 1]),
            th.tensor([False, True]),
            th.tensor([True, True]),
            ACTION_NVECS,
            second_jump_weight=8.0,
        )

        self.assertLess(loss.item(), np.log(2))
        self.assertEqual(accuracy.item(), 1.0)
        self.assertEqual(recall.item(), 1.0)
        self.assertEqual(sample_rate.item(), 0.5)

    def test_learning_rate_schedule_updates_all_optimizers(self):
        actor = th.nn.Linear(2, 2)
        critic = th.nn.Linear(2, 1)
        optimizers = (th.optim.Adam(actor.parameters()), th.optim.Adam(critic.parameters()))

        set_learning_rate(optimizers, 2.5e-5)

        self.assertTrue(all(
            group["lr"] == 2.5e-5
            for optimizer in optimizers
            for group in optimizer.param_groups
        ))

    def test_default_tracker_lookahead_extends_to_two_seconds(self):
        self.assertEqual(DEFAULT_TRACKER_WINDOWS, (1, 2, 4, 8, 16, 32, 64))

    def test_legacy_tracker_checkpoint_has_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.pt"
            th.save({
                "policy": {},
                "config": {
                    "architecture": "hybrid-beta-gru-v1",
                    "windows": [1, 2],
                    "frameskip": 4,
                },
            }, path)

            with self.assertRaisesRegex(RuntimeError, "legacy tracker checkpoint"):
                load_tracker_policy(path, SimpleNamespace(device="cpu"), (1, 2), 4)

    def test_phc_checkpoint_rejects_different_replay_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.pt"
            th.save({
                "assignments": th.tensor([0]),
                "specialists": [{}],
                "config": {
                    "architecture": PHC_TRACKER_ARCHITECTURE,
                    "windows": [1, 2],
                    "frameskip": 4,
                    "replay_manifest": ["different"],
                },
            }, path)
            env = SimpleNamespace(
                device="cpu",
                replays=SimpleNamespace(demo_manifest=("expected",)),
            )

            with self.assertRaisesRegex(ValueError, "replay segments"):
                load_tracker_policy(path, env, (1, 2), 4)

    def test_phc_routes_accept_a_verified_replay_prefix(self):
        assignments = validated_replay_assignments(
            th.tensor([2, 1, 0]),
            ("first", "second", "third"),
            ("first", "second"),
        )

        th.testing.assert_close(assignments, th.tensor([2, 1]))

    def test_tracker_policy_uses_all_categorical_action_factors(self):
        observation_size = GOAL_STATE_SIZE + 21 * len(DEFAULT_TRACKER_WINDOWS)
        env = SimpleNamespace(
            action_codec=CARLActionCodec(),
            device="cpu",
            single_action_space=gym.spaces.MultiDiscrete(ACTION_NVECS),
            single_observation_space=gym.spaces.Box(
                -np.inf,
                np.inf,
                (observation_size,),
                np.float32,
            ),
        )
        policy = build_tracker_policy(env, DEFAULT_TRACKER_WINDOWS)
        observation = th.zeros((2, observation_size))

        output = policy.act(observation)
        evaluation = policy.evaluate_actions(observation, output.action)

        self.assertEqual(output.action.dtype, th.int64)
        self.assertEqual(output.action.shape, (2, ACTION_FACTORS))
        self.assertEqual(evaluation.extras["factor_entropy"].shape, (2, ACTION_FACTORS))
        th.testing.assert_close(output.log_prob, evaluation.log_prob)

    def test_checkpoint_retention_does_not_delete_legacy_high_step_files(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            legacy = directory / "tracker_999999999999.pt"
            th.save({"policy": {}}, legacy)
            checkpoint = PeriodicCheckpoint(
                {"policy": th.nn.Linear(1, 1)},
                directory,
                interval=1,
                keep=1,
            )

            checkpoint.run()
            checkpoint.step = 1
            checkpoint.run()

            self.assertTrue(legacy.exists())
            self.assertFalse((directory / "tracker_000000000000.pt").exists())
            self.assertTrue((directory / "tracker_000000000001.pt").exists())

    def test_low_deterministic_rewards_drive_hard_negative_sampling(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._base_sampling_probabilities = th.full((3,), 1 / 3)
        replays._sampling_probabilities = replays._base_sampling_probabilities.clone()
        scores = SegmentScores(3, "cpu")
        scores.set(th.tensor([0.9, 0.1, 0.8]))

        replays.focus_hard_negatives(scores, 1.0)

        self.assertGreater(
            replays._sampling_probabilities[1],
            replays._sampling_probabilities[0],
        )
        th.testing.assert_close(replays._sampling_probabilities.sum(), th.tensor(1.0))

    def test_specialist_assignment_selects_highest_deterministic_reward(self):
        first = SegmentScores(3, "cpu")
        first.mean_rewards[:] = th.tensor([0.9, 0.2, th.nan])
        second = SegmentScores(3, "cpu")
        second.mean_rewards[:] = th.tensor([0.7, 0.8, th.nan])

        assignments = specialist_assignments((first, second))

        th.testing.assert_close(assignments, th.tensor([0, 1, 1]))

    def test_segment_evaluation_uses_deterministic_mean_reward(self):
        class Replays:
            n_demos = 3

            def queue_demo_ids(self, demo_ids):
                self.demo_ids = demo_ids

            @staticmethod
            def clear_queued_demo_ids():
                return

        class Environment:
            device = th.device("cpu")
            n_envs = 2
            minimum_reward = 0.1

            def __init__(self):
                self.replays = Replays()
                self.reward = SimpleNamespace(value=None)

            def reset(self):
                self.demo_ids = self.replays.demo_ids
                self.elapsed = th.zeros(self.n_envs)
                return th.zeros((self.n_envs, 1))

            def step(self, action):
                self.elapsed += 1
                self.reward.value = (self.demo_ids.float() + 1) / 5
                terminated = self.elapsed >= self.demo_ids + 1
                return (
                    th.zeros((self.n_envs, 1)),
                    self.reward.value[:, None],
                    terminated,
                    th.zeros(self.n_envs, dtype=th.bool),
                    {},
                )

        class Policy(th.nn.Module):
            def __init__(self):
                super().__init__()
                self.deterministic = []

            @staticmethod
            def initial_state(batch_size):
                return th.zeros((batch_size, 1))

            def act(self, observation, state=None, *, deterministic=False):
                self.deterministic.append(deterministic)
                return PolicyOutput(
                    action=th.zeros((len(observation), ACTION_FACTORS), dtype=th.long),
                    next_state=state,
                )

        env = Environment()
        policy = Policy()

        scores = evaluate_tracker_policy(env, policy)

        th.testing.assert_close(scores, th.tensor([0.2, 0.4, 0.6]))
        self.assertTrue(all(policy.deterministic))
        self.assertEqual(env.minimum_reward, 0.1)
        self.assertTrue(policy.training)

    def test_routed_tracker_dispatches_each_replay_segment(self):
        class Specialist(th.nn.Module):
            def __init__(self, action: int) -> None:
                super().__init__()
                self.value = th.nn.Parameter(th.tensor(0.0))
                self.action_value = action

            @property
            def device(self):
                return self.value.device

            def initial_state(self, batch_size):
                return th.zeros((batch_size, 1))

            def act(self, observation, state=None, *, deterministic=False):
                batch = len(observation)
                return PolicyOutput(
                    action=th.full((batch, ACTION_FACTORS), self.action_value),
                    next_state=th.full((batch, 1), float(self.action_value)),
                    log_prob=th.full((batch,), float(self.action_value)),
                )

        replays = SimpleNamespace(
            n_demos=3,
            current_demo_ids=lambda: th.tensor([0, 1, 2]),
        )
        policy = RoutedTrackerPolicy(
            (Specialist(0), Specialist(1)),
            th.tensor([0, 1, 0]),
            replays,
        )

        output = policy.act(th.zeros((3, 2)))

        th.testing.assert_close(output.action[:, 0], th.tensor([0, 1, 0]))
        th.testing.assert_close(output.next_state[:, 0], th.tensor([0.0, 1.0, 0.0]))

    def test_phc_checkpoint_serializes_specialist_scores_and_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            scores = SegmentScores(2, "cpu")
            policies = [th.nn.Linear(1, 1), th.nn.Linear(1, 1)]
            checkpoint = PHCCheckpoint(
                Path(directory),
                interval=10,
                keep=2,
                config={"architecture": "test"},
                assignment_fn=lambda: th.tensor([0, 1]),
            )
            checkpoint.set_stage(
                1,
                policies,
                th.nn.Linear(1, 1),
                (scores, scores),
                step_offset=100,
            )
            checkpoint.step = 200

            checkpoint.run()

            payload = th.load(
                Path(directory) / "tracker_000000000200.pt",
                weights_only=True,
            )
            self.assertEqual(len(payload["specialists"]), 2)
            self.assertEqual(len(payload["segment_scores"]), 2)
            th.testing.assert_close(payload["assignments"], th.tensor([0, 1]))

    def test_parser_projection_uses_carl_axis_class_order(self):
        raw = np.zeros((3, 8), dtype=np.float32)
        raw[:, 0] = raw[:, 1] = raw[:, 2] = raw[:, 3] = raw[:, 4] = [
            -1.0,
            0.0,
            1.0,
        ]

        projected = _project_carl_actions(raw)

        np.testing.assert_array_equal(projected[:, 0], [1, 0, 2])
        np.testing.assert_array_equal(projected[:, 1], [1, 0, 2])
        np.testing.assert_array_equal(projected[:, 2], [1, 0, 2])
        np.testing.assert_array_equal(projected[:, 5], [1, 0, 2])

    def test_dataset_loading_keeps_segments_without_ball_touches(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._min_len = 30
        replays.minimum_remaining_frames = 1
        demo = np.zeros((30, 161), dtype=np.float32)

        loaded = replays._filter(demo, np.zeros(30, dtype=bool))

        self.assertEqual(len(loaded), 1)
        self.assertEqual(len(loaded[0][0]), 30)
        self.assertEqual(loaded[0][0].shape[1], STORED_REPLAY_SIZE)

    def test_dataset_loading_preserves_expert_ego_touch_timing(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._min_len = 30
        replays.minimum_remaining_frames = 1
        demo = np.zeros((30, 161), dtype=np.float32)
        demo[7, -5] = 1.0
        loaded, _ = replays._filter(demo, np.zeros(30, dtype=bool))[0]
        replays._replays = loaded
        replays._cursors = th.tensor([7])
        replays._demo_id = th.tensor([0])
        replays._offsets = th.tensor([0, len(loaded)])

        self.assertTrue(replays.current_ego_touch().item())
        self.assertEqual(loaded.shape[1], STORED_REPLAY_SIZE)

    def test_dataset_loading_guards_non_ego_touch_transitions_in_all_modes(self):
        for width in (161, 215, 269):
            with self.subTest(width=width):
                replays = ExpertGoalStates.__new__(ExpertGoalStates)
                replays._min_len = 3
                replays.minimum_remaining_frames = 1
                demo = np.zeros((12, width), dtype=np.float32)
                demo[:, 9] = np.arange(12)
                demo[5, -4] = 1.0

                loaded = replays._filter(demo, np.zeros(12, dtype=bool))

                self.assertEqual(len(loaded), 2)
                np.testing.assert_array_equal(
                    loaded[0][0][:, 9].numpy(), np.arange(4)
                )
                np.testing.assert_array_equal(
                    loaded[1][0][:, 9].numpy(), np.arange(7, 12)
                )

    def test_dataset_loading_does_not_guard_expert_ego_touches(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._min_len = 3
        replays.minimum_remaining_frames = 1
        demo = np.zeros((12, 215), dtype=np.float32)
        demo[5, -5] = 1.0

        loaded = replays._filter(demo, np.zeros(12, dtype=bool))

        self.assertEqual(len(loaded), 1)
        self.assertEqual(len(loaded[0][0]), 12)

    def test_dataset_loading_guards_resets_around_expert_touches(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._min_len = 12
        replays.minimum_remaining_frames = 1
        demo = np.zeros((12, 161), dtype=np.float32)
        demo[0, -5] = 1.0

        _, start_map = replays._filter(demo, np.zeros(12, dtype=bool))[0]

        self.assertEqual(start_map[0].item(), 2)

    def test_random_starts_leave_the_configured_number_of_frames(self):
        count = 512
        length = 200
        minimum = 128
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays.n_cars = 1
        replays.device = th.device("cpu")
        replays.balance = False
        replays.start_at_beginning = False
        replays.minimum_remaining_frames = minimum
        replays._selected_demo = 0
        replays._n_demos = 1
        replays._demo_id = th.zeros(count, dtype=th.long)
        replays._offsets = th.tensor([0, length])
        replays._safe_cursors = th.arange(length)
        replays._cursors = th.zeros(count, dtype=th.long)
        replays._replays = th.zeros((length, STORED_REPLAY_SIZE))

        replays.reset(th.ones(count, dtype=th.bool))

        remaining = length - replays._cursors - 1
        self.assertTrue((remaining >= minimum).all())
        self.assertTrue((replays._cursors <= length - minimum - 1).all())

    def test_queued_segment_evaluation_starts_at_earliest_safe_frame(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays.n_cars = 1
        replays.device = th.device("cpu")
        replays.start_at_beginning = False
        replays.minimum_remaining_frames = 1
        replays._n_demos = 2
        replays._demo_id = th.zeros(2, dtype=th.long)
        replays._offsets = th.tensor([0, 3, 6])
        replays._safe_cursors = th.tensor([1, 1, 2, 4, 4, 5])
        replays._cursors = th.zeros(2, dtype=th.long)
        replays._replays = th.zeros((6, STORED_REPLAY_SIZE))
        replays.queue_demo_ids(th.tensor([0, 1]))

        replays.reset(th.ones(2, dtype=th.bool))

        th.testing.assert_close(replays._demo_id, th.tensor([0, 1]))
        th.testing.assert_close(replays._cursors, th.tensor([1, 4]))

    def test_filter_rejects_segments_with_no_safe_early_start(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._min_len = 129
        replays.minimum_remaining_frames = 128
        demo = np.zeros((150, 161), dtype=np.float32)
        unsafe = np.ones(150, dtype=bool)
        unsafe[50] = False

        self.assertEqual(replays._filter(demo, unsafe), [])

    def test_goals_include_internal_state_and_relative_ball_and_car_state(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._windows = th.tensor([[1, 2]])
        replays._cursors = th.tensor([0])
        replays._demo_id = th.tensor([0])
        replays._offsets = th.tensor([0, 3])
        replays._replays = th.zeros((3, STORED_REPLAY_SIZE))
        replays._replays[0, GOAL_STATE_SIZE:EXPERT_TOUCH_INDEX] = th.arange(
            INTERNAL_STATE_SIZE
        )
        replays._replays[1, :9] = 100.0
        replays._replays[2, :9] = 200.0
        replays._replays[1, 9:GOAL_STATE_SIZE] = 1.0
        replays._replays[2, 9:GOAL_STATE_SIZE] = 2.0
        observation = th.zeros((1, GOAL_STATE_SIZE))
        observation[:, :9] = 10.0

        goal_observation, end = replays.next_goals(observation)

        internal_end = GOAL_STATE_SIZE + INTERNAL_STATE_SIZE
        self.assertEqual(
            goal_observation.shape,
            (1, GOAL_STATE_SIZE + INTERNAL_STATE_SIZE + 2 * GOAL_STATE_SIZE),
        )
        th.testing.assert_close(goal_observation[:, :GOAL_STATE_SIZE], observation)
        th.testing.assert_close(
            goal_observation[:, GOAL_STATE_SIZE:internal_end],
            th.arange(INTERNAL_STATE_SIZE, dtype=th.float32).expand(1, -1),
        )
        th.testing.assert_close(
            goal_observation[:, internal_end:internal_end + 9],
            th.full((1, 9), 90.0),
        )
        th.testing.assert_close(
            goal_observation[:, internal_end + 9:internal_end + GOAL_STATE_SIZE],
            th.ones(1, GOAL_STATE_SIZE - 9),
        )
        th.testing.assert_close(
            goal_observation[
                :, internal_end + GOAL_STATE_SIZE:internal_end + GOAL_STATE_SIZE + 9
            ],
            th.full((1, 9), 190.0),
        )
        th.testing.assert_close(
            goal_observation[:, internal_end + GOAL_STATE_SIZE + 9:],
            th.full((1, GOAL_STATE_SIZE - 9), 2.0),
        )
        self.assertFalse(end.item())

    def test_tracking_reward_ignores_ball_state_before_touch(self):
        target_tensor = th.zeros((1, GOAL_STATE_SIZE))
        actual_tensor = target_tensor.clone()
        actual_tensor[:, :9] = 100.0
        target = CARLObservation.from_tensor(target_tensor, 1)
        actual = CARLObservation.from_tensor(actual_tensor, 1)
        replays = SimpleNamespace(device=th.device("cpu"), current=lambda: target)
        reward = TrackingReward(replays)

        value = reward(self._tracking_context(actual))

        th.testing.assert_close(value, th.ones((1, 1)))

    def test_tracking_progress_is_signed_and_does_not_reward_oscillation(self):
        target_tensor = th.zeros((1, GOAL_STATE_SIZE))
        target = CARLObservation.from_tensor(target_tensor, 1)
        far_tensor = target_tensor.clone()
        far_tensor[:, 9] = 0.02
        close_tensor = target_tensor.clone()
        close_tensor[:, 9] = 0.01
        far = CARLObservation.from_tensor(far_tensor, 1)
        close = CARLObservation.from_tensor(close_tensor, 1)
        replays = SimpleNamespace(device=th.device("cpu"), current=lambda: target)
        reward = TrackingReward(replays, progress_scale=4.0)

        toward = reward(self._tracking_context(close, previous=far))
        gain = reward.progress.clone()
        away = reward(self._tracking_context(far, previous=close))
        loss = reward.progress.clone()

        self.assertGreater(gain.item(), 0)
        th.testing.assert_close(loss, -gain)
        self.assertGreater(toward.item(), reward.value.item())
        self.assertLess(away.item(), reward.value.item())

    def test_ball_outcome_multiplies_car_reward_after_touch_and_latches(self):
        target_tensor = th.zeros((1, GOAL_STATE_SIZE))
        target = CARLObservation.from_tensor(target_tensor, 1)
        missed_ball_tensor = target_tensor.clone()
        missed_ball_tensor[:, :9] = 100.0
        missed_ball = CARLObservation.from_tensor(missed_ball_tensor, 1)
        replays = SimpleNamespace(device=th.device("cpu"), current=lambda: target)
        reward = TrackingReward(replays)

        before_contact = reward(self._tracking_context(missed_ball))
        contact = reward(self._tracking_context(missed_ball, touched=True))
        follow_up = reward(self._tracking_context(missed_ball))

        th.testing.assert_close(before_contact, th.ones((1, 1)))
        self.assertLess(contact.item(), 1e-3)
        th.testing.assert_close(follow_up, contact)
        th.testing.assert_close(reward.value[:, None], contact)

    def test_ball_outcome_scores_ball_position_relative_to_car(self):
        target_tensor = th.zeros((1, GOAL_STATE_SIZE))
        target = CARLObservation.from_tensor(target_tensor, 1)
        aligned_tensor = target_tensor.clone()
        aligned_tensor[:, 0] = 0.005
        aligned_tensor[:, 9] = 0.005
        opposed_tensor = aligned_tensor.clone()
        opposed_tensor[:, 9] = -0.005
        replays = SimpleNamespace(device=th.device("cpu"), current=lambda: target)

        aligned = TrackingReward(replays)(
            self._tracking_context(
                CARLObservation.from_tensor(aligned_tensor, 1),
                touched=True,
            )
        )
        opposed = TrackingReward(replays)(
            self._tracking_context(
                CARLObservation.from_tensor(opposed_tensor, 1),
                touched=True,
            )
        )

        self.assertGreater(aligned.item(), opposed.item())

    def test_ball_outcome_latch_resets_with_replay_segment(self):
        target_tensor = th.zeros((1, GOAL_STATE_SIZE))
        target = CARLObservation.from_tensor(target_tensor, 1)
        missed_ball_tensor = target_tensor.clone()
        missed_ball_tensor[:, :9] = 100.0
        missed_ball = CARLObservation.from_tensor(missed_ball_tensor, 1)
        replays = SimpleNamespace(device=th.device("cpu"), current=lambda: target)
        reward = TrackingReward(replays)
        reward(self._tracking_context(missed_ball, touched=True))

        reward.reset(th.tensor([True]))
        after_reset = reward(self._tracking_context(missed_ball))

        th.testing.assert_close(after_reset, th.ones((1, 1)))

    def test_ball_outcome_latch_clears_after_native_done(self):
        target_tensor = th.zeros((1, GOAL_STATE_SIZE))
        target = CARLObservation.from_tensor(target_tensor, 1)
        missed_ball_tensor = target_tensor.clone()
        missed_ball_tensor[:, :9] = 100.0
        missed_ball = CARLObservation.from_tensor(missed_ball_tensor, 1)
        replays = SimpleNamespace(device=th.device("cpu"), current=lambda: target)
        reward = TrackingReward(replays)

        terminal = reward(
            self._tracking_context(missed_ball, touched=True, done=True)
        )
        next_episode = reward(self._tracking_context(missed_ball))

        self.assertLess(terminal.item(), 1e-3)
        th.testing.assert_close(next_episode, th.ones((1, 1)))

    def test_ball_is_anchored_to_expert_before_touch(self):
        wrapper, environment, observation = self._anchor_fixture()

        anchored = wrapper._anchor_ball(observation, th.tensor([False]))

        self.assertEqual(len(environment.calls), 1)
        position, velocity, angular_velocity, indices = environment.calls[0]
        th.testing.assert_close(position, th.tensor([[410.8, 1200.0, 622.8]]))
        th.testing.assert_close(velocity, th.tensor([[2400.0, 3000.0, 3600.0]]))
        th.testing.assert_close(angular_velocity, th.tensor([[4.2, 4.8, 5.4]]))
        th.testing.assert_close(indices, th.tensor([0]))
        th.testing.assert_close(anchored, th.ones_like(observation))

    def test_simulated_touch_releases_without_overwriting_ball(self):
        wrapper, environment, observation = self._anchor_fixture()
        wrapper.reward.touched[:] = True

        returned = wrapper._anchor_ball(observation, th.tensor([False]))

        self.assertFalse(wrapper._ball_anchored.item())
        self.assertEqual(environment.calls, [])
        self.assertIs(returned, observation)

    def test_missed_expert_touch_releases_without_applying_expert_impulse(self):
        wrapper, environment, observation = self._anchor_fixture(
            expert_touch=True
        )

        returned = wrapper._anchor_ball(observation, th.tensor([False]))

        self.assertFalse(wrapper._ball_anchored.item())
        self.assertEqual(environment.calls, [])
        self.assertIs(returned, observation)

    def test_upcoming_expert_touch_releases_before_replay_impulse(self):
        wrapper, environment, observation = self._anchor_fixture(
            upcoming_expert_touch=True
        )

        returned = wrapper._anchor_ball(observation, th.tensor([False]))

        self.assertFalse(wrapper._ball_anchored.item())
        self.assertEqual(environment.calls, [])
        self.assertIs(returned, observation)

    def test_expert_touch_lookahead_stays_within_replay_segment(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._replays = th.zeros((4, STORED_REPLAY_SIZE))
        replays._replays[2, EXPERT_TOUCH_INDEX] = 1
        replays._cursors = th.tensor([1])
        replays._demo_id = th.tensor([0])
        replays._offsets = th.tensor([0, 2, 4])

        self.assertFalse(replays.current_ego_touch(offset=1).item())

        replays._replays[1, EXPERT_TOUCH_INDEX] = 1
        replays._cursors[0] = 0
        self.assertTrue(replays.current_ego_touch(offset=1).item())

    def test_step_anchors_before_advancing_replay_cursor(self):
        wrapper, environment, observation = self._anchor_fixture()
        events = []

        class Replays:
            cursor = 1
            goal_size = INTERNAL_STATE_SIZE + len(DEFAULT_TRACKER_WINDOWS) * GOAL_STATE_SIZE

            def current_raw_action(self, offset=0):
                events.append(("raw_action", self.cursor + offset))
                return th.full((1, 8), float(self.cursor + offset))

            @staticmethod
            def current_jump_supervision(offset=0):
                return th.zeros(1, dtype=th.long), th.zeros(1, dtype=th.bool)

            def current_ego_touch(self, offset=0):
                events.append(("touch", self.cursor + offset))
                return th.tensor([False])

            def current(self):
                events.append(("current", self.cursor))
                expert = th.zeros((1, GOAL_STATE_SIZE))
                expert[:, 0] = self.cursor
                return CARLObservation.from_tensor(expert, 1)

            def next_goals(self, obs, mask=None):
                events.append(("next", self.cursor))
                self.cursor += 1
                return th.nn.functional.pad(obs, (0, self.goal_size)), th.tensor([False])

        wrapper.replays = Replays()
        wrapper.minimum_reward = 0.1
        wrapper.minimum_tracking_frames = 1
        wrapper._low_reward_frames = th.zeros(1, dtype=th.long)
        wrapper.reward.value = th.ones(1)
        environment.step = lambda action: (
            observation,
            th.zeros(1),
            th.tensor([False]),
            th.tensor([False]),
            {},
        )

        wrapper.step(th.zeros((1, 7), dtype=th.long))

        self.assertEqual(
            events,
            [
                ("raw_action", 0),
                ("touch", 1),
                ("touch", 2),
                ("current", 1),
                ("next", 1),
            ],
        )
        self.assertEqual(wrapper.replays.cursor, 2)
        th.testing.assert_close(wrapper.last_raw_expert_action, th.zeros((1, 8)))

    def test_native_final_observation_padding_defers_action_hints(self):
        wrapper = ExpertLookaheadEnv.__new__(ExpertLookaheadEnv)
        wrapper.replays = SimpleNamespace(
            goal_size=INTERNAL_STATE_SIZE + 7 * GOAL_STATE_SIZE
        )

        padded = wrapper._pad_goals(th.zeros((2, GOAL_STATE_SIZE)))

        self.assertEqual(
            padded.shape,
            (2, GOAL_STATE_SIZE + INTERNAL_STATE_SIZE + 7 * GOAL_STATE_SIZE),
        )

    def test_mixed_native_and_tracking_resets_keep_final_observation_width(self):
        goal_size = INTERNAL_STATE_SIZE + 7 * GOAL_STATE_SIZE

        class Environment:
            def step(self, action):
                return (
                    th.zeros((2, GOAL_STATE_SIZE)),
                    th.zeros(2),
                    th.tensor([True, False]),
                    th.zeros(2, dtype=th.bool),
                    {
                        "final_obs": th.zeros((2, GOAL_STATE_SIZE)),
                        "_final_obs": th.tensor([True, False]),
                    },
                )

            @staticmethod
            def _apply_reset_state(mask):
                return

            @staticmethod
            def _clear_sim_stats(mask):
                return

            @staticmethod
            def _observe():
                return th.zeros((2, GOAL_STATE_SIZE))

        class Replays:
            goal_size = INTERNAL_STATE_SIZE + 7 * GOAL_STATE_SIZE

            @staticmethod
            def current_raw_action(offset=0):
                return th.zeros((2, 8))

            @staticmethod
            def current_jump_supervision(offset=0):
                return th.zeros(2, dtype=th.long), th.zeros(2, dtype=th.bool)

            @staticmethod
            def next_goals(obs, mask=None):
                count = len(obs)
                return th.nn.functional.pad(obs, (0, goal_size)), th.zeros(
                    count, dtype=th.bool
                )

        wrapper = ExpertLookaheadEnv.__new__(ExpertLookaheadEnv)
        wrapper.env = Environment()
        wrapper.replays = Replays()
        wrapper.reward = SimpleNamespace(value=th.tensor([1.0, 0.0]))
        wrapper.minimum_reward = 0.5
        wrapper.minimum_tracking_frames = 1
        wrapper._low_reward_frames = th.zeros(2, dtype=th.long)
        wrapper._anchor_ball = lambda obs, native: obs

        observation, _, _, _, info = wrapper.step(
            th.zeros((2, ACTION_FACTORS), dtype=th.long)
        )

        expected_width = GOAL_STATE_SIZE + goal_size
        self.assertEqual(observation.shape, (2, expected_width))
        self.assertEqual(info["final_obs"].shape, (2, expected_width))

    def test_stateless_critic_capture_records_current_and_next_values(self):
        critic = SimpleNamespace(value=lambda observation: observation.sum(-1))
        context = SimpleNamespace(
            observation=th.tensor([[1.0, 2.0]]),
            env_step=SimpleNamespace(next_obs=th.tensor([[3.0, 4.0]])),
        )

        captured = StatelessCriticCapture(critic)(context)

        th.testing.assert_close(captured["baseline_value"], th.tensor([3.0]))
        th.testing.assert_close(captured["baseline_next_value"], th.tensor([7.0]))

    def test_demonstration_frame_includes_raw_expert_actions(self):
        raw_expert_action = th.tensor([[0.25, -0.5, 0.75, 1.0, 0.0, 1.0, 0.0, 1.0]])

        frame = frame_from_state(
            th.zeros(31),
            Path("tracker.pt"),
            th.tensor([1.0]),
            th.zeros(GOAL_STATE_SIZE),
            "demo",
            th.zeros((1, ACTION_FACTORS)),
            raw_expert_action,
        )

        self.assertEqual(frame["raw_expert_action"], raw_expert_action[0].tolist())

    def test_watcher_publishes_expert_row_matching_eager_replay_cursor(self):
        class Replays:
            @staticmethod
            def current_tensor(offset=0):
                self.assertEqual(offset, -1)
                return th.zeros((1, GOAL_STATE_SIZE))

            @staticmethod
            def current_demo_name():
                return "demo"

        viewer = SimpleNamespace(publish=lambda frame: setattr(viewer, "frame", frame))
        base = SimpleNamespace(
            device=th.device("cpu"),
            _env=SimpleNamespace(get_state=lambda: np.zeros((1, 31), dtype=np.float32)),
        )
        with unittest.mock.patch("torch.cuda.synchronize"):
            publish_frame(
                viewer,
                base,
                Replays(),
                Path("tracker.pt"),
                th.zeros(1),
                th.zeros((1, ACTION_FACTORS)),
                th.zeros((1, 8)),
            )

        self.assertEqual(viewer.frame["demo"], "demo")

    def test_filter_appends_first_non_ego_car_state_without_shifting_existing_indices(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._min_len = 3
        replays.minimum_remaining_frames = 1
        demo = np.zeros((10, 161), dtype=np.float32)
        demo[:, :GOAL_STATE_SIZE] = np.arange(GOAL_STATE_SIZE)
        demo[:, GOAL_STATE_SIZE:CONTROL_STATE_SIZE] = np.arange(OPPONENT_STATE_SIZE) + 1000
        internal_start = 83 + 27 * 2
        demo[:, internal_start:internal_start + INTERNAL_STATE_SIZE] = (
            np.arange(INTERNAL_STATE_SIZE) + 200
        )
        demo[3, -5] = 1.0
        raw_actions = np.zeros((10, RAW_ACTION_SIZE), dtype=np.float32)
        raw_actions[:] = np.arange(RAW_ACTION_SIZE) + 300

        loaded, _ = replays._filter(demo, np.zeros(10, dtype=bool), raw_actions)[0]

        self.assertEqual(loaded.shape[1], STORED_REPLAY_SIZE)
        np.testing.assert_array_equal(
            loaded[:, :GOAL_STATE_SIZE].numpy(),
            demo[:, :GOAL_STATE_SIZE],
        )
        np.testing.assert_array_equal(
            loaded[:, GOAL_STATE_SIZE:EXPERT_TOUCH_INDEX].numpy(),
            demo[:, internal_start:internal_start + INTERNAL_STATE_SIZE],
        )
        np.testing.assert_array_equal(loaded[:, EXPERT_TOUCH_INDEX].numpy(), demo[:, -5])
        np.testing.assert_array_equal(
            loaded[:, RAW_ACTION_INDEX:RAW_ACTION_INDEX + RAW_ACTION_SIZE].numpy(),
            raw_actions,
        )
        np.testing.assert_array_equal(
            loaded[:, OPPONENT_STATE_INDEX:].numpy(),
            demo[:, GOAL_STATE_SIZE:CONTROL_STATE_SIZE],
        )

    def test_filter_skips_teammates_when_selecting_opponent_context(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._min_len = 3
        replays.minimum_remaining_frames = 1
        demo = np.zeros((10, 215), dtype=np.float32)
        teammate = slice(30, 51)
        opponent = slice(51, 72)
        demo[:, teammate] = 1.0
        demo[:, opponent] = 2.0

        loaded, _ = replays._filter(
            demo,
            np.zeros(10, dtype=bool),
            np.zeros((10, RAW_ACTION_SIZE), dtype=np.float32),
        )[0]

        np.testing.assert_array_equal(
            loaded[:, OPPONENT_STATE_INDEX:].numpy(), demo[:, opponent]
        )

    def test_current_opponent_state_clamps_to_segment_boundaries(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._demo_id = th.tensor([0, 1])
        replays._offsets = th.tensor([0, 2, 4])
        replays._cursors = th.tensor([0, 2])
        replays._replays = th.zeros((4, STORED_REPLAY_SIZE))
        replays._replays[0, OPPONENT_STATE_INDEX:] = th.arange(OPPONENT_STATE_SIZE) + 1.0
        replays._replays[1, OPPONENT_STATE_INDEX:] = th.arange(OPPONENT_STATE_SIZE) + 2.0
        replays._replays[2, OPPONENT_STATE_INDEX:] = th.arange(OPPONENT_STATE_SIZE) + 3.0
        replays._replays[3, OPPONENT_STATE_INDEX:] = th.arange(OPPONENT_STATE_SIZE) + 4.0

        at_start = replays.current_opponent_state(offset=-1)
        self.assertEqual(at_start.shape, (2, OPPONENT_STATE_SIZE))
        th.testing.assert_close(at_start[0], th.arange(OPPONENT_STATE_SIZE) + 1.0)
        th.testing.assert_close(at_start[1], th.arange(OPPONENT_STATE_SIZE) + 3.0)

        at_end = replays.current_opponent_state(offset=5)
        th.testing.assert_close(at_end[0], th.arange(OPPONENT_STATE_SIZE) + 2.0)
        th.testing.assert_close(at_end[1], th.arange(OPPONENT_STATE_SIZE) + 4.0)

    def test_tracker_observation_space_unchanged_by_opponent_context(self):
        env = SimpleNamespace(
            n_cars=1,
            n_envs=2,
            n_sim=2,
            device="cpu",
            action_space=gym.vector.utils.batch_space(
                gym.spaces.MultiDiscrete(ACTION_NVECS), 2
            ),
            single_action_space=gym.spaces.MultiDiscrete(ACTION_NVECS),
            register_reward=lambda reward: None,
        )
        opponent = th.arange(OPPONENT_STATE_SIZE, dtype=th.float32)[None, :].expand(2, -1) + 1000
        replays = SimpleNamespace(
            goal_size=INTERNAL_STATE_SIZE + len(DEFAULT_TRACKER_WINDOWS) * GOAL_STATE_SIZE,
            current_opponent_state=lambda offset=0: opponent if offset == -1 else None,
            device="cpu",
        )
        wrapper = ExpertLookaheadEnv(env, replays)
        expected_size = GOAL_STATE_SIZE + replays.goal_size
        self.assertEqual(wrapper.single_observation_space.shape, (expected_size,))

        observation = th.arange(GOAL_STATE_SIZE, dtype=th.float32)[None, :].expand(2, -1) + 1
        control = wrapper.control_state(observation)
        self.assertEqual(control.shape, (2, CONTROL_STATE_SIZE))
        th.testing.assert_close(control[:, :GOAL_STATE_SIZE], observation)
        th.testing.assert_close(control[:, GOAL_STATE_SIZE:], opponent)

    @staticmethod
    def _anchor_fixture(
        expert_touch: bool = False,
        upcoming_expert_touch: bool = False,
    ):
        expert_tensor = th.zeros((1, GOAL_STATE_SIZE))
        expert_tensor[:, :9] = th.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        expert = CARLObservation.from_tensor(expert_tensor, 1)

        class FakeEnvironment:
            def __init__(self):
                self.calls = []

            def set_ball(
                self,
                position,
                velocity,
                angular_velocity,
                *,
                simulation_indices,
            ):
                self.calls.append((
                    position,
                    velocity,
                    angular_velocity,
                    simulation_indices,
                ))
                return th.ones((1, GOAL_STATE_SIZE))

        environment = FakeEnvironment()
        wrapper = ExpertLookaheadEnv.__new__(ExpertLookaheadEnv)
        wrapper.env = environment
        wrapper.replays = SimpleNamespace(
            current=lambda: expert,
            current_ego_touch=lambda offset=0: th.tensor([
                expert_touch if offset == 0 else upcoming_expert_touch
            ]),
        )
        wrapper.reward = SimpleNamespace(touched=th.tensor([False]))
        wrapper._ball_anchored = th.tensor([True])
        wrapper._pos_scale = th.tensor(POSITION_SCALE)
        observation = th.zeros((1, GOAL_STATE_SIZE))
        return wrapper, environment, observation

    @staticmethod
    def _tracking_context(
        observation,
        *,
        previous=None,
        touched=False,
        done=False,
    ):
        if previous is None:
            previous = observation
        return SimpleNamespace(
            current_observation=observation,
            previous_observation=previous,
            current=SimpleNamespace(
                car_ball_touches=th.tensor([[touched]], dtype=th.bool)
            ),
            events=SimpleNamespace(done=th.tensor([done], dtype=th.bool)),
        )


if __name__ == "__main__":
    unittest.main()
