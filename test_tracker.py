import unittest

from types import SimpleNamespace

import numpy as np
import torch as th

from carl.gymnasium import CARLObservation

from tracker import (
    BALL_MAX_ANG_SPEED,
    BALL_MAX_SPEED,
    EXPERT_TOUCH_INDEX,
    ExpertGoalStates,
    ExpertLookaheadEnv,
    GOAL_STATE_SIZE,
    POSITION_SCALE,
    TrackingReward,
)


class TrackerTest(unittest.TestCase):
    def test_dataset_loading_keeps_segments_without_ball_touches(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._min_len = 30
        demo = np.zeros((30, 161), dtype=np.float32)

        loaded = replays._filter(demo, np.zeros(30, dtype=bool))

        self.assertEqual(len(loaded), 1)
        self.assertEqual(len(loaded[0][0]), 30)
        self.assertEqual(loaded[0][0].shape[1], EXPERT_TOUCH_INDEX + 1)

    def test_dataset_loading_preserves_expert_ego_touch_timing(self):
        replays = ExpertGoalStates.__new__(ExpertGoalStates)
        replays._min_len = 30
        demo = np.zeros((30, 161), dtype=np.float32)
        demo[7, -5] = 1.0
        loaded, _ = replays._filter(demo, np.zeros(30, dtype=bool))[0]
        replays._replays = loaded
        replays._cursors = th.tensor([7])

        self.assertTrue(replays.current_ego_touch().item())

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

    def test_tracking_reward_ignores_ball_state(self):
        target_tensor = th.zeros((1, GOAL_STATE_SIZE))
        actual_tensor = target_tensor.clone()
        actual_tensor[:, :9] = 100.0
        target = CARLObservation.from_tensor(target_tensor, 1)
        actual = CARLObservation.from_tensor(actual_tensor, 1)
        replays = SimpleNamespace(device=th.device("cpu"), current=lambda: target)
        reward = TrackingReward(replays)

        value = reward(SimpleNamespace(
            current_observation=actual,
            current=SimpleNamespace(car_ball_touches=th.zeros((1, 1), dtype=th.bool)),
        ))

        th.testing.assert_close(value, th.ones((1, 1)))

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

    def test_expert_touch_anchors_post_touch_ball_then_releases(self):
        wrapper, environment, observation = self._anchor_fixture(expert_touch=True)

        wrapper._anchor_ball(observation, th.tensor([False]))

        self.assertFalse(wrapper._ball_anchored.item())
        self.assertEqual(len(environment.calls), 1)

    def test_step_anchors_before_advancing_replay_cursor(self):
        wrapper, environment, observation = self._anchor_fixture()
        events = []

        class Replays:
            cursor = 1

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

        self.assertEqual(events, [("touch", 1), ("current", 1), ("next", 1)])
        self.assertEqual(wrapper.replays.cursor, 2)

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


if __name__ == "__main__":
    unittest.main()
