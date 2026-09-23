import unittest

import torch as th

from carl.gymnasium.state import CarlEvents, CarlState, RewardContext
from rewards import (
    DifferentialReward,
    DifferentialRewardWeights,
    SeerReward,
    SeerRewardWeights,
)


TEAM_SIGN = th.tensor([1.0, -1.0])


def zero_weights(**overrides) -> DifferentialRewardWeights:
    return DifferentialRewardWeights(
        **{
            name: 0.0
            for name in DifferentialRewardWeights.__dataclass_fields__
        }
        | overrides
    )


def zero_seer_weights(**overrides) -> SeerRewardWeights:
    return SeerRewardWeights(
        **{
            name: 0.0
            for name in SeerRewardWeights.__dataclass_fields__
        }
        | overrides
    )


def make_context(
    ball_y: float = 0.0,
    ball_z: float = 100.0,
    previous_ball_y: float = 0.0,
    previous_ball_z: float = 100.0,
    ball_speed: float = 0.0,
    previous_ball_speed: float = 0.0,
    touched_car: int | tuple[int, ...] | None = None,
    demoed_car: int | None = None,
    score_delta: int = 0,
    truncated: bool = False,
) -> RewardContext:
    previous_raw = th.zeros((1, 53))
    previous_raw[:, 9 + 9] = 1.0
    previous_raw[:, 31 + 9] = 1.0
    previous_raw[:, 1] = previous_ball_y
    previous_raw[:, 2] = previous_ball_z
    previous_raw[:, 4] = previous_ball_speed
    current_raw = previous_raw.clone()
    current_raw[:, 1] = ball_y
    current_raw[:, 2] = ball_z
    current_raw[:, 4] = ball_speed
    if touched_car is not None:
        for car in (touched_car,) if isinstance(touched_car, int) else touched_car:
            current_raw[:, 9 + 22 * car + 21] = 1.0
    if demoed_car is not None:
        current_raw[:, 9 + 22 * demoed_car + 17] = 1.0
    previous = CarlState(previous_raw, 2, th.empty((0, 3)), TEAM_SIGN)
    current = CarlState(current_raw, 2, th.empty((0, 3)), TEAM_SIGN)
    events = CarlEvents(
        score_delta=th.tensor([score_delta]),
        done=th.tensor([truncated]),
        terminated=th.tensor([False]),
        truncated=th.tensor([truncated]),
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


class DifferentialRewardTest(unittest.TestCase):
    def test_goal_and_touch_rewards_are_explicit(self):
        reward = DifferentialReward(1, 1, shaping_scale=0.0)
        th.testing.assert_close(
            reward(make_context(score_delta=1)), th.tensor([[10.0, -10.0]])
        )
        th.testing.assert_close(
            reward(make_context(touched_car=0)), th.tensor([[0.1, 0.0]])
        )

    def test_ball_goal_progress_rewards_the_scoring_team(self):
        reward = DifferentialReward(
            1,
            1,
            weights=zero_weights(ball_goal_progress=5.0),
        )
        value = reward(make_context(ball_y=500.0))
        self.assertGreater(float(value[0, 0]), 0.0)
        self.assertLess(float(value[0, 1]), 0.0)

    def test_own_goal_clearance_rewards_moving_away(self):
        weights = zero_weights(own_goal_clearance=2.5)
        reward = DifferentialReward(1, 1, weights=weights)
        away = reward(make_context(ball_y=500.0))
        toward = reward(make_context(ball_y=-500.0))
        self.assertGreater(float(away[0, 0]), 0.0)
        self.assertLess(float(toward[0, 0]), 0.0)

    def test_team_common_progress_rewards_both_cars(self):
        weights = zero_weights(ball_height_progress=1.0)
        reward = DifferentialReward(1, 1, weights=weights)
        value = reward(make_context(ball_z=300.0, previous_ball_z=100.0))
        self.assertTrue(bool((value > 0).all()))

    def test_distance_player_ball_level_rewards_proximity(self):
        weights = zero_weights(distance_player_ball=0.5)
        reward = DifferentialReward(1, 1, weights=weights)
        near = reward(make_context(ball_y=200.0))
        far = reward(make_context(ball_y=4000.0))
        self.assertGreater(float(near[0, 0]), float(far[0, 0]))
        self.assertGreater(float(far[0, 0]), 0.0)

    def test_no_touch_timeout_penalty(self):
        reward = DifferentialReward(1, 1, shaping_scale=0.0, no_touch_timeout_steps=1)
        th.testing.assert_close(
            reward(make_context(truncated=True)), th.tensor([[-1.0, -1.0]])
        )


class SeerRewardTest(unittest.TestCase):
    def test_goals_and_win_probability_remain_zero_sum(self):
        reward = SeerReward(
            1, 1, normalize=False,
            weights=zero_seer_weights(
                goal_scored=10.0, goal_speed_bonus=2.5,
                goal_distance_bonus=2.5, goal_time_bonus=1.0,
                win_probability=10.0,
            ),
        )
        value = reward(make_context(score_delta=1, previous_ball_speed=1000.0))

        self.assertGreater(float(value[0, 0]), 20.0)
        th.testing.assert_close(value[0, 0], -value[0, 1])

    def test_touch_shaping_is_local_even_with_default_normalization(self):
        context = make_context(touched_car=0, ball_speed=2300.0)
        weights = zero_seer_weights(touch_acceleration=1.0)

        raw = SeerReward(1, 1, normalize=False, weights=weights)(context)
        th.testing.assert_close(raw, th.tensor([[1.0, 0.0]]))

        normalized = SeerReward(1, 1, weights=weights)(context)
        self.assertGreater(float(normalized[0, 0]), 0.0)
        th.testing.assert_close(normalized[0, 1], th.tensor(0.0))

    def test_both_cars_can_earn_shaping_at_once(self):
        reward = SeerReward(
            1, 1, normalize=False,
            weights=zero_seer_weights(touch_acceleration=1.0),
        )
        value = reward(make_context(touched_car=(0, 1), ball_speed=2300.0))

        th.testing.assert_close(value, th.ones((1, 2)))

    def test_goal_and_touch_shaping_remain_distinct(self):
        reward = SeerReward(
            1, 1, normalize=False,
            weights=zero_seer_weights(goal_scored=10.0, touch_acceleration=1.0),
        )

        th.testing.assert_close(
            reward(make_context(score_delta=1, touched_car=0, ball_speed=2300.0)),
            th.tensor([[11.0, -10.0]]),
        )

    def test_demo_rewards_behavior_without_penalizing_victim(self):
        reward = SeerReward(
            1, 1, normalize=False,
            weights=zero_seer_weights(demo=5.0),
        )

        th.testing.assert_close(
            reward(make_context(demoed_car=1)), th.tensor([[5.0, 0.0]])
        )


if __name__ == "__main__":
    unittest.main()
