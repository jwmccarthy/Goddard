import unittest

from types import SimpleNamespace

import torch as th

from carl.gymnasium import CARLObservation

from tracker import ExpertGoalStates, GOAL_STATE_SIZE, TrackingReward


class TrackerTest(unittest.TestCase):
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

        value = reward(SimpleNamespace(current_observation=actual))

        th.testing.assert_close(value, th.ones((1, 1)))


if __name__ == "__main__":
    unittest.main()
