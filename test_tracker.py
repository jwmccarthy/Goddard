import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as th

from carl.gymnasium import CARLObservation
from jarl.data.batch import TensorBatch

from ballchasing_replays.parse_replays import _project_carl_actions
from watch_demonstrations import frame_from_state
from tracker_checkpoint import PeriodicCheckpoint

from tracker import (
    ACTION_FACTORS,
    BALL_MAX_ANG_SPEED,
    BALL_MAX_SPEED,
    DEFAULT_TRACKER_WINDOWS,
    EXPERT_TOUCH_INDEX,
    ExpertGoalStates,
    ExpertLookaheadEnv,
    GOAL_STATE_SIZE,
    POSITION_SCALE,
    StatelessCriticCapture,
    STORED_REPLAY_SIZE,
    TrackingReward,
    _expert_action_loss,
    _expert_action_labels,
    load_tracker_policy,
)


class TrackerTest(unittest.TestCase):
    def test_default_tracker_lookahead_extends_to_two_seconds(self):
        self.assertEqual(DEFAULT_TRACKER_WINDOWS, (1, 2, 4, 8, 16, 32, 64))

    def test_legacy_tracker_checkpoint_has_explicit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.pt"
            th.save({"policy": {}}, path)

            with self.assertRaisesRegex(RuntimeError, "legacy tracker checkpoint"):
                load_tracker_policy(path, SimpleNamespace(device="cpu"), (1, 2), 4)

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

    def test_expert_action_labels_only_supervise_direct_controls(self):
        raw = np.zeros((3, 8), dtype=np.float32)
        raw[:, 0] = [-1.0, 0.0, 1.0]
        raw[:, 1:5] = np.asarray([-1.0, 0.0, 1.0])[:, None]
        raw[1, 5:] = [1.0, 1.0, 1.0]

        labels, valid = _expert_action_labels(raw)

        np.testing.assert_array_equal(labels[:, 2], [1, 0, 2])
        np.testing.assert_array_equal(labels[:, 0], [1, 0, 2])
        np.testing.assert_array_equal(labels[:, 1], [1, 0, 2])
        np.testing.assert_array_equal(labels[:, 5], [1, 0, 2])
        np.testing.assert_array_equal(labels[1, [3, 4, 6]], [1, 1, 1])
        self.assertFalse(valid[:, [0, 1, 5]].any())
        self.assertTrue(valid[:, [2, 3, 4, 6]].all())

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

        self.assertTrue(replays.current_ego_touch().item())
        self.assertEqual(loaded.shape[1], STORED_REPLAY_SIZE)

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

    def test_filter_rejects_segments_with_no_safe_early_start(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._min_len = 129
        replays.minimum_remaining_frames = 128
        demo = np.zeros((150, 161), dtype=np.float32)
        unsafe = np.ones(150, dtype=bool)
        unsafe[50] = False

        self.assertEqual(replays._filter(demo, unsafe), [])

    def test_goals_contain_only_relative_car_state(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._windows = th.tensor([[1, 2]])
        replays._cursors = th.tensor([0])
        replays._demo_id = th.tensor([0])
        replays._offsets = th.tensor([0, 3])
        replays._replays = th.zeros((3, GOAL_STATE_SIZE))
        replays._replays[1, :9] = 100.0
        replays._replays[2, :9] = 200.0
        replays._replays[1, 9:] = 1.0
        replays._replays[2, 9:] = 2.0
        observation = th.zeros((1, GOAL_STATE_SIZE))

        goal_observation, end = replays.next_goals(observation)

        self.assertEqual(goal_observation.shape, (1, GOAL_STATE_SIZE + 42))
        th.testing.assert_close(goal_observation[:, :GOAL_STATE_SIZE], observation)
        th.testing.assert_close(goal_observation[:, GOAL_STATE_SIZE:30 + 21], th.ones(1, 21))
        th.testing.assert_close(goal_observation[:, 30 + 21:], th.full((1, 21), 2.0))
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

    def test_ball_outcome_reward_activates_on_touch_and_latches(self):
        target_tensor = th.zeros((1, GOAL_STATE_SIZE))
        target = CARLObservation.from_tensor(target_tensor, 1)
        replays = SimpleNamespace(device=th.device("cpu"), current=lambda: target)
        reward = TrackingReward(replays, ball_outcome_weight=0.1)

        contact = reward(self._tracking_context(target, touched=True))
        follow_up = reward(self._tracking_context(target))

        th.testing.assert_close(contact, th.tensor([[1.1]]))
        th.testing.assert_close(follow_up, th.tensor([[1.1]]))
        th.testing.assert_close(reward.value, th.ones(1))

    def test_ball_outcome_latch_resets_with_replay_segment(self):
        target_tensor = th.zeros((1, GOAL_STATE_SIZE))
        target = CARLObservation.from_tensor(target_tensor, 1)
        replays = SimpleNamespace(device=th.device("cpu"), current=lambda: target)
        reward = TrackingReward(replays, ball_outcome_weight=0.1)
        reward(self._tracking_context(target, touched=True))

        reward.reset(th.tensor([True]))
        after_reset = reward(self._tracking_context(target))

        th.testing.assert_close(after_reset, th.ones((1, 1)))

    def test_ball_outcome_latch_clears_after_native_done(self):
        target_tensor = th.zeros((1, GOAL_STATE_SIZE))
        target = CARLObservation.from_tensor(target_tensor, 1)
        replays = SimpleNamespace(device=th.device("cpu"), current=lambda: target)
        reward = TrackingReward(replays, ball_outcome_weight=0.1)

        terminal = reward(self._tracking_context(target, touched=True, done=True))
        next_episode = reward(self._tracking_context(target))

        th.testing.assert_close(terminal, th.tensor([[1.1]]))
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

    def test_step_anchors_before_advancing_replay_cursor(self):
        wrapper, environment, observation = self._anchor_fixture()
        events = []

        class Replays:
            cursor = 1

            def current_expert_action(self, offset=0):
                events.append(("action", self.cursor + offset))
                return (
                    th.full((1, ACTION_FACTORS), self.cursor + offset),
                    th.ones((1, ACTION_FACTORS), dtype=th.bool),
                )

            def current_ego_touch(self):
                events.append(("touch", self.cursor))
                return th.tensor([False])

            def current(self):
                events.append(("current", self.cursor))
                expert = th.zeros((1, GOAL_STATE_SIZE))
                expert[:, 0] = self.cursor
                return CARLObservation.from_tensor(expert, 1)

            def next_goals(self, obs, mask=None):
                events.append(("next", self.cursor))
                self.cursor += 1
                return obs, th.tensor([False])

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
            [("action", 0), ("touch", 1), ("current", 1), ("next", 1)],
        )
        self.assertEqual(wrapper.replays.cursor, 2)
        th.testing.assert_close(
            wrapper.last_expert_action,
            th.zeros((1, ACTION_FACTORS), dtype=th.long),
        )

    def test_expert_action_loss_uses_only_valid_legal_targets(self):
        logits = th.zeros((2, 18), requires_grad=True)
        action_mask = th.ones((2, 18), dtype=th.bool)
        action_mask[1, 12] = False
        expert_action = th.zeros((2, ACTION_FACTORS), dtype=th.long)
        expert_action[:, 2] = th.tensor([1, 2])
        expert_action[:, 4] = 1
        valid = th.zeros((2, ACTION_FACTORS), dtype=th.bool)
        valid[:, 2] = True
        valid[:, 4] = True
        loss, _, _ = _expert_action_loss(
            logits,
            action_mask,
            expert_action,
            valid,
            th.ones(2, dtype=th.bool),
            (3, 3, 3, 2, 2, 3, 2),
            inferred_weight=0,
        )

        expected_expert_loss = (np.log(3) + np.log(2)) / 2
        self.assertAlmostEqual(loss.item(), expected_expert_loss, places=6)

    def test_expert_action_loss_trains_inferred_factors_on_valid_sequence_steps(self):
        loss, _, _ = _expert_action_loss(
            th.zeros((2, 1, 18), requires_grad=True),
            th.ones((2, 1, 18), dtype=th.bool),
            th.zeros((2, 1, ACTION_FACTORS), dtype=th.long),
            th.zeros((2, 1, ACTION_FACTORS), dtype=th.bool),
            th.tensor([[True], [False]]),
            (3, 3, 3, 2, 2, 3, 2),
            inferred_weight=0.1,
        )

        self.assertAlmostEqual(loss.item(), np.log(3), places=6)

    def test_stateless_critic_capture_records_current_and_next_values(self):
        critic = SimpleNamespace(value=lambda observation: observation.sum(-1))
        context = SimpleNamespace(
            observation=th.tensor([[1.0, 2.0]]),
            env_step=SimpleNamespace(next_obs=th.tensor([[3.0, 4.0]])),
        )

        captured = StatelessCriticCapture(critic)(context)

        th.testing.assert_close(captured["baseline_value"], th.tensor([3.0]))
        th.testing.assert_close(captured["baseline_next_value"], th.tensor([7.0]))

    def test_demonstration_frame_includes_expert_actions_and_confidence(self):
        expert_action = th.tensor([[1, 2, 0, 1, 0, 2, 1]])
        expert_valid = th.tensor([[False, False, True, True, True, False, True]])

        frame = frame_from_state(
            th.zeros(31),
            Path("tracker.pt"),
            th.tensor([1.0]),
            th.zeros(GOAL_STATE_SIZE),
            "demo",
            th.zeros((1, ACTION_FACTORS), dtype=th.long),
            expert_action,
            expert_valid,
        )

        self.assertEqual(frame["expert_action"], expert_action[0].tolist())
        self.assertEqual(frame["expert_action_valid"], expert_valid[0].tolist())

    @staticmethod
    def _anchor_fixture(expert_touch: bool = False):
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
            current_ego_touch=lambda: th.tensor([expert_touch]),
        )
        wrapper.reward = SimpleNamespace(touched=th.tensor([False]))
        wrapper._ball_anchored = th.tensor([True])
        wrapper._pos_scale = th.tensor(POSITION_SCALE)
        observation = th.zeros((1, GOAL_STATE_SIZE))
        return wrapper, environment, observation

    @staticmethod
    def _tracking_context(observation, *, touched=False, done=False):
        return SimpleNamespace(
            current_observation=observation,
            current=SimpleNamespace(
                car_ball_touches=th.tensor([[touched]], dtype=th.bool)
            ),
            events=SimpleNamespace(done=th.tensor([done], dtype=th.bool)),
        )


if __name__ == "__main__":
    unittest.main()
