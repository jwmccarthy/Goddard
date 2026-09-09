import unittest

import torch as th

from simple import GoalOnlyReward
from test_self_play import make_reward_context


class SimpleSelfPlayTest(unittest.TestCase):
    def test_goal_reward_is_zero_sum_plus_or_minus_ten(self):
        reward = GoalOnlyReward()

        th.testing.assert_close(
            reward(make_reward_context(score_delta=1)),
            th.tensor([[10.0, -10.0]]),
        )
        th.testing.assert_close(
            reward(make_reward_context(score_delta=-1)),
            th.tensor([[-10.0, 10.0]]),
        )

    def test_non_goal_events_have_zero_reward(self):
        reward = GoalOnlyReward()

        th.testing.assert_close(
            reward(make_reward_context(touched_car=0, truncated=True)),
            th.zeros((1, 2)),
        )


if __name__ == "__main__":
    unittest.main()
