import math
import unittest
from dataclasses import replace

import torch as th

from carl.gymnasium.state import CarlEvents, CarlState, RewardContext
from rewards import (
    BALL_RADIUS,
    CAR_MAX_SPEED,
    CEILING_Z,
    GOAL_Y,
    MATCH_TICKS,
    DifferentialReward,
    DifferentialRewardWeights,
    SeerReward,
    SeerRewardWeights,
)


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
    ball_x: float = 0.0,
    ball_y: float = 0.0,
    ball_z: float = 100.0,
    previous_ball_x: float = 0.0,
    previous_ball_y: float = 0.0,
    previous_ball_z: float = 100.0,
    ball_speed: float = 0.0,
    previous_ball_speed: float = 0.0,
    ball_vz: float = 0.0,
    previous_ball_vz: float = 0.0,
    car_z: float = 0.0,
    touched_car: int | tuple[int, ...] | None = None,
    demoed_car: int | None = None,
    score_delta: int = 0,
    episode_ticks: int = 0,
    truncated: bool = False,
    n_blue: int = 1,
    n_orange: int = 1,
) -> RewardContext:
    n_cars = n_blue + n_orange
    previous_raw = th.zeros((1, 9 + 22 * n_cars))
    for car in range(n_cars):
        previous_raw[:, 9 + 22 * car + 9] = 1.0
    previous_raw[:, 9 + 2] = car_z
    previous_raw[:, 0] = previous_ball_x
    previous_raw[:, 1] = previous_ball_y
    previous_raw[:, 2] = previous_ball_z
    previous_raw[:, 4] = previous_ball_speed
    previous_raw[:, 5] = previous_ball_vz
    current_raw = previous_raw.clone()
    current_raw[:, 0] = ball_x
    current_raw[:, 1] = ball_y
    current_raw[:, 2] = ball_z
    current_raw[:, 4] = ball_speed
    current_raw[:, 5] = ball_vz
    if touched_car is not None:
        for car in (touched_car,) if isinstance(touched_car, int) else touched_car:
            current_raw[:, 9 + 22 * car + 21] = 1.0
    if demoed_car is not None:
        current_raw[:, 9 + 22 * demoed_car + 17] = 1.0
    team_sign = th.tensor([1.0] * n_blue + [-1.0] * n_orange)
    previous = CarlState(previous_raw, n_cars, th.empty((0, 3)), team_sign)
    current = CarlState(current_raw, n_cars, th.empty((0, 3)), team_sign)
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
        th.tensor([episode_ticks]),
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
    def test_original_weight_defaults(self):
        nonzero = {
            "goal_scored": 10.0,
            "goal_speed_bonus": 2.5,
            "goal_distance_bonus": 2.5,
            "boost_gain": 1.0,
            "boost_loss": 0.5,
            "demo": 5.0,
            "ball_goal_progress": 5.0,
            "player_ball_progress": 0.75,
            "alignment_progress": 0.5,
            "touch_acceleration": 0.25,
            "aerial_touch": 1.0,
            "angular_velocity": 0.01,
            "flip_reset": 10.0,
            "touch_grass": 0.005,
            "win_probability": 10.0,
            "goal_time_bonus": 1.0,
            "air_dribble_start": 0.5,
            "air_dribble_progress": 1.0,
            "air_dribble_complete": 1.0,
        }
        zero = (
            "ball_touch", "ball_height", "ball_velocity", "distance_player_ball",
            "distance_ball_goal", "facing_ball", "align_ball_goal", "closest_to_ball",
            "touched_last", "behind_ball", "velocity_player_ball", "kickoff",
            "velocity", "boost_amount", "forward_velocity",
        )
        self.assertEqual(vars(SeerRewardWeights()), nonzero | dict.fromkeys(zero, 0.0))

    def test_default_angular_velocity_and_touch_grass_still_pay(self):
        context = make_context()
        context.current.raw[0, 9 + 6] = 5.5
        context.current.raw[0, 9 + 16] = 1.0

        reward = SeerReward(1, 1, normalize=False)
        th.testing.assert_close(reward(context), th.tensor([[0.005, -0.005]]))

    def test_default_occupancy_weights_do_not_pay_for_stationary_possession(self):
        context = make_context(
            ball_z=500.0, previous_ball_z=500.0,
            ball_speed=1800.0, previous_ball_speed=1800.0,
            touched_car=0,
        )
        for state in (context.previous, context.current):
            state.raw[0, 9 + 0] = 200.0
            state.raw[0, 9 + 3] = 500.0
            state.raw[0, 9 + 15] = 50.0
        weights = replace(SeerRewardWeights(), aerial_touch=0.0)
        reward = SeerReward(1, 1, normalize=False, weights=weights)

        th.testing.assert_close(reward(context), th.zeros((1, 2)))

    def test_full_opponent_subtraction_of_touch_shaping_for_both_sides(self):
        weights = zero_seer_weights(touch_acceleration=1.0)
        reward = SeerReward(1, 1, normalize=False, weights=weights)
        th.testing.assert_close(
            reward(make_context(touched_car=0, ball_speed=2300.0)),
            th.tensor([[1.0, -1.0]]),
        )
        th.testing.assert_close(
            reward(make_context(touched_car=1, ball_speed=2300.0)),
            th.tensor([[-1.0, 1.0]]),
        )
        th.testing.assert_close(
            reward(make_context(touched_car=(0, 1), ball_speed=2300.0)),
            th.zeros((1, 2)),
        )

    def test_full_opponent_team_mean_in_two_vs_two(self):
        reward = SeerReward(
            2, 2, normalize=False,
            weights=zero_seer_weights(touch_acceleration=1.0),
        )
        th.testing.assert_close(
            reward(make_context(n_blue=2, n_orange=2, touched_car=0, ball_speed=2300.0)),
            th.tensor([[1.0, 0.0, -0.5, -0.5]]),
        )
        th.testing.assert_close(
            reward(make_context(n_blue=2, n_orange=2, touched_car=2, ball_speed=2300.0)),
            th.tensor([[-0.5, -0.5, 1.0, 0.0]]),
        )

    def test_goal_and_touch_shaping_are_both_opponent_subtracted(self):
        reward = SeerReward(
            1, 1, normalize=False,
            weights=zero_seer_weights(goal_scored=10.0, touch_acceleration=1.0),
        )
        th.testing.assert_close(
            reward(make_context(score_delta=1, touched_car=0, ball_speed=2300.0)),
            th.tensor([[11.0, -11.0]]),
        )

    def test_demo_uses_half_opponent_minus_self_before_zero_sum(self):
        reward = SeerReward(
            1, 1, normalize=False, log_diagnostics=True,
            weights=zero_seer_weights(demo=5.0),
        )
        blue_demo = reward(make_context(demoed_car=1, truncated=True))
        th.testing.assert_close(blue_demo.reward, th.tensor([[5.0, -5.0]]))
        self.assertEqual(blue_demo.info["seer/component/demo"], [2.5, -2.5])

        orange_demo = reward(make_context(demoed_car=0, truncated=True))
        th.testing.assert_close(orange_demo.reward, th.tensor([[-5.0, 5.0]]))
        self.assertEqual(orange_demo.info["seer/component/demo"], [-2.5, 2.5])

    def test_goal_without_any_touch_pays_original_bonuses(self):
        reward = SeerReward(
            1, 1, normalize=False,
            weights=zero_seer_weights(
                goal_scored=10.0, goal_speed_bonus=2.5,
                goal_distance_bonus=2.5, goal_time_bonus=1.0,
                win_probability=10.0,
            ),
        )
        value = reward(make_context(score_delta=1, previous_ball_speed=1000.0))

        def win_probability(diff):
            return 0.5 * (1.0 + math.erf((diff - 0.5) / math.sqrt(20.0)))

        expected = (
            20.0 + 2.5 * 1000.0 / 6000.0
            + 2.5 * (1.0 - math.exp(-100.0 / CAR_MAX_SPEED))
            + 20.0 * (win_probability(1) - win_probability(0))
        )
        th.testing.assert_close(value, th.tensor([[expected, -expected]]))

    def test_late_win_probability_uses_score_difference_minus_half(self):
        reward = SeerReward(
            1, 1, normalize=False,
            weights=zero_seer_weights(win_probability=10.0),
        )
        one_second_left = MATCH_TICKS - 120
        blue_scores = reward(make_context(score_delta=1, episode_ticks=one_second_left))
        orange_scores = reward(make_context(score_delta=-1, episode_ticks=one_second_left))

        variance = 2.0 / 60.0

        def win_probability(diff):
            return 0.5 * (1.0 + math.erf((diff - 0.5) / math.sqrt(2.0 * variance)))

        blue_value = 20.0 * (win_probability(1) - win_probability(0))
        orange_value = 20.0 * (win_probability(0) - win_probability(-1))
        th.testing.assert_close(blue_scores, th.tensor([[blue_value, -blue_value]]))
        th.testing.assert_close(orange_scores, th.tensor([[-orange_value, orange_value]]))
        self.assertGreater(blue_value, orange_value)

    def test_goal_transition_keeps_original_ball_progress(self):
        weights = zero_seer_weights(ball_goal_progress=5.0)
        context = dict(previous_ball_y=GOAL_Y, ball_y=0.0)
        goal = SeerReward(1, 1, normalize=False, weights=weights)(
            make_context(score_delta=1, **context)
        )
        no_goal = SeerReward(1, 1, normalize=False, weights=weights)(
            make_context(**context)
        )
        th.testing.assert_close(goal, no_goal)
        self.assertLess(float(goal[0, 0]), 0.0)
        th.testing.assert_close(goal[0, 0], -goal[0, 1])

    def test_ball_touch_only_pays_on_forward_progress(self):
        weights = zero_seer_weights(ball_touch=1.0)
        forward = SeerReward(1, 1, normalize=False, weights=weights)(
            make_context(ball_y=500.0, touched_car=0)
        )
        backward = SeerReward(1, 1, normalize=False, weights=weights)(
            make_context(ball_y=-500.0, touched_car=0)
        )
        self.assertGreater(float(forward[0, 0]), 0.0)
        th.testing.assert_close(forward[0, 0], -forward[0, 1])
        th.testing.assert_close(backward, th.zeros((1, 2)))

    def test_flip_reset_and_aerial_touch(self):
        context = make_context(ball_z=500.0, previous_ball_z=500.0, car_z=400.0, touched_car=0)
        context.previous.raw[0, 9 + 18] = 1.0
        context.current.raw[0, 9 + 14] = -1.0
        reward = SeerReward(
            1, 1, normalize=False,
            weights=zero_seer_weights(flip_reset=10.0, aerial_touch=1.0),
        )
        expected = 10.0 + 500.0 / 2250.0
        th.testing.assert_close(reward(context), th.tensor([[expected, -expected]]))

    def test_short_air_dribble_starts_and_pays_for_positive_progress(self):
        reward = SeerReward(
            1, 1, normalize=False,
            weights=zero_seer_weights(air_dribble_start=0.5, air_dribble_progress=1.0),
        )
        start = reward(make_context(
            ball_z=400.0, previous_ball_z=300.0, ball_vz=230.0,
            car_z=300.0, touched_car=0,
        ))
        air_height = (400.0 - 2 * BALL_RADIUS) / (CEILING_Z - 2 * BALL_RADIUS)
        expected_start = 0.5 * air_height + 100.0 / (CEILING_Z - 2 * BALL_RADIUS) + 0.1
        th.testing.assert_close(start, th.tensor([[expected_start, -expected_start]]))

        steady = reward(make_context(
            ball_z=400.0, previous_ball_z=400.0,
            ball_vz=230.0, previous_ball_vz=230.0,
            car_z=300.0, episode_ticks=8,
        ))
        th.testing.assert_close(steady, th.zeros((1, 2)))

        rising = reward(make_context(
            ball_z=450.0, previous_ball_z=400.0,
            ball_vz=230.0, previous_ball_vz=230.0,
            car_z=300.0, episode_ticks=16,
        ))
        expected_rising = 50.0 / (CEILING_Z - 2 * BALL_RADIUS)
        th.testing.assert_close(rising, th.tensor([[expected_rising, -expected_rising]]))

        expired = reward(make_context(
            ball_z=500.0, previous_ball_z=450.0,
            car_z=300.0, episode_ticks=47,
        ))
        th.testing.assert_close(expired, th.zeros((1, 2)))

    def test_short_air_dribble_completes_on_landing_after_two_contacts(self):
        reward = SeerReward(
            1, 1, normalize=False,
            weights=zero_seer_weights(air_dribble_complete=1.0),
        )
        reward(make_context(
            ball_z=400.0, previous_ball_z=400.0,
            car_z=300.0, touched_car=0, episode_ticks=0,
        ))
        reward(make_context(
            ball_z=420.0, previous_ball_z=400.0,
            car_z=300.0, touched_car=0, episode_ticks=8,
        ))
        landing = make_context(
            ball_y=700.0, previous_ball_y=0.0, ball_z=100.0,
            previous_ball_z=420.0, car_z=300.0, episode_ticks=16,
        )
        th.testing.assert_close(reward(landing), th.tensor([[1.0, -1.0]]))
        th.testing.assert_close(reward(landing), th.zeros((1, 2)))

    def test_short_air_dribble_does_not_start_along_wall(self):
        reward = SeerReward(
            1, 1, normalize=False,
            weights=zero_seer_weights(air_dribble_start=0.5, air_dribble_progress=1.0),
        )
        value = reward(make_context(
            ball_x=3700.0, previous_ball_x=3700.0,
            ball_z=400.0, previous_ball_z=300.0,
            car_z=300.0, touched_car=0,
        ))
        th.testing.assert_close(value, th.zeros((1, 2)))

    def test_no_touch_steps_have_no_penalty(self):
        reward = SeerReward(1, 1, normalize=False)
        for _ in range(35):
            th.testing.assert_close(reward(make_context()), th.zeros((1, 2)))

    def test_normalization_centers_using_running_variance(self):
        reward = SeerReward(1, 1)
        th.testing.assert_close(
            reward._normalize(th.tensor([[0.0, 2.0]])),
            th.tensor([[-1.0, 1.0]]),
        )
        th.testing.assert_close(
            reward._normalize(th.tensor([[0.0, 4.0]])),
            th.tensor([[-1.5, 2.5]]) / math.sqrt(2.75),
        )
        self.assertEqual(reward._count, 4)

    def test_default_normalization_keeps_opponent_rewards_opposite(self):
        reward = SeerReward(
            1, 1, weights=zero_seer_weights(touch_acceleration=1.0)
        )
        th.testing.assert_close(
            reward(make_context(touched_car=0, ball_speed=2300.0)),
            th.tensor([[1.0, -1.0]]),
        )
        th.testing.assert_close(
            reward(make_context(touched_car=1, ball_speed=4600.0)),
            th.tensor([[-2.0, 2.0]]) / math.sqrt(2.5),
        )

    def test_public_scales_keep_goal_reward_separate_from_shaping(self):
        reward = SeerReward(
            1, 1, normalize=False,
            weights=zero_seer_weights(goal_scored=10.0, touch_acceleration=1.0),
        )
        context = make_context(score_delta=1, touched_car=0, ball_speed=2300.0)
        th.testing.assert_close(reward(context), th.tensor([[11.0, -11.0]]))
        reward.set_shaping_scale(0.0)
        th.testing.assert_close(reward(context), th.tensor([[10.0, -10.0]]))
        reward.set_goal_scored_weight(7.0)
        th.testing.assert_close(reward(context), th.tensor([[7.0, -7.0]]))
        with self.assertRaises(ValueError):
            reward.set_shaping_scale(1.1)
        with self.assertRaises(ValueError):
            reward.set_goal_scored_weight(0.0)

    def test_diagnostics_report_per_episode_means_and_zero_sum(self):
        reward = SeerReward(
            1, 1, normalize=False, log_diagnostics=True,
            weights=zero_seer_weights(touch_acceleration=1.0),
        )
        reward(make_context(touched_car=0, ball_speed=2300.0))
        reward(make_context(touched_car=1, ball_speed=4600.0))
        result = reward(make_context(truncated=True))

        info = result.info
        for name in ("seer/component/touch_acceleration", "seer/aggregate/raw"):
            self.assertAlmostEqual(info[name][0], 1.0 / 3.0, places=6)
            self.assertAlmostEqual(info[name][1], 2.0 / 3.0, places=6)
        for name in ("seer/aggregate/zero_sum", "seer/aggregate/normalized"):
            self.assertAlmostEqual(info[name][0], -1.0 / 3.0, places=6)
            self.assertAlmostEqual(info[name][1], 1.0 / 3.0, places=6)
        self.assertAlmostEqual(info["seer/scale/raw"][0], math.sqrt(1.0 / 3.0), places=6)
        self.assertAlmostEqual(info["seer/scale/zero_sum"][0], math.sqrt(5.0 / 3.0), places=6)
        self.assertNotIn("seer/aggregate/outcome_adjusted", info)


if __name__ == "__main__":
    unittest.main()
