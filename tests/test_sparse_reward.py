import math
import unittest
from dataclasses import replace

import torch

from carl.gymnasium.state import (
    BOOST_PAD_POSITIONS,
    CARLObservation,
    CarlEvents,
    CarlState,
    RewardContext,
)
from reward_spec import (
    BALL_RADIUS,
    CAR_MAX_SPEED,
    GOAL_HEIGHT,
    GOAL_Y,
    RewardSpec,
)


def sparse_context() -> RewardContext:
    n_sim = 7
    raw = torch.zeros(n_sim, 9 + 22 * 2 + len(BOOST_PAD_POSITIONS))
    cars = raw[:, 9:53].view(n_sim, 2, 22)
    raw[:, 2] = BALL_RADIUS
    cars[:, :, 2] = 17.0
    cars[:, 0, 1] = -300.0
    cars[:, 1, 1] = 300.0
    cars[:, :, 16] = 1.0

    # Ordinary ball motion and uneven positioning must pay nothing in sparse mode.
    raw[0, 4] = 1000.0
    cars[0, 0, 1] = -150.0
    cars[0, 1, 1] = 1000.0

    # A ground pop from behind the ball begins an air-dribble setup.
    raw[1, 2] = 120.0
    raw[1, 5] = 750.0
    cars[1, 0, 1] = -100.0
    cars[1, 0, 21] = 1.0

    # A directed ball touch accelerates toward the opponent goal.
    raw[2, 2] = 200.0
    raw[2, 4] = 1450.0
    cars[2, 0, 1] = -100.0
    cars[2, 0, 21] = 1.0

    # An orange aerial touch is above crossbar height, but too slow to be a shot.
    raw[3, 2] = 700.0
    raw[3, 4] = -600.0
    cars[3, 1, 1] = 100.0
    cars[3, 1, 2] = 600.0
    cars[3, 1, 16] = 0.0
    cars[3, 1, 21] = 1.0

    # Fast flight toward a nearby airborne ball earns a small velocity bonus.
    raw[4, 2] = 550.0
    cars[4, 0, 2] = 300.0
    cars[4, 0, 16] = 0.0
    cars[4, 0, 4] = 1750.0
    cars[4, 0, 5] = 1250.0

    cars[5, 1, 17] = 1.0  # Orange is newly demolished.

    previous = raw.clone()
    previous[:, 9:53].view(n_sim, 2, 22)[:, :, 21] = 0.0
    previous[1, 5] = 0.0
    previous[2, 4] = 100.0
    previous[3, 4] = 0.0
    previous[:, 9:53].view(n_sim, 2, 22)[5, 1, 17] = 0.0
    pads = torch.tensor(BOOST_PAD_POSITIONS)
    signs = torch.tensor([1.0, -1.0])
    observation = CARLObservation.from_tensor(torch.zeros(n_sim * 2, 51), 2)
    return RewardContext(
        current=CarlState.from_raw(raw, 2, pads, signs),
        previous=CarlState.from_raw(previous, 2, pads, signs),
        current_observation=observation,
        previous_observation=observation,
        events=CarlEvents(
            score_delta=torch.tensor([0.0] * 6 + [1.0]),
            done=torch.ones(n_sim, dtype=torch.bool),
            terminated=torch.zeros(n_sim, dtype=torch.bool),
            truncated=torch.zeros(n_sim, dtype=torch.bool),
        ),
        actions=torch.zeros(n_sim * 2, 7),
        score_difference=torch.zeros(n_sim),
        episode_ticks=torch.zeros(n_sim),
        overtime=torch.zeros(n_sim, dtype=torch.bool),
    )


class SparseRewardTests(unittest.TestCase):
    def test_only_complex_events_and_fast_air_approaches_are_rewarded(self):
        context = sparse_context()
        reward_spec = RewardSpec(normalize=False, log_diagnostics=True, sparse=True)
        result = reward_spec(context)
        reward = result.reward

        for name in (
            "boost_gain", "distance_player_ball", "ball_goal_progress",
            "player_ball_progress", "win_probability", "velocity",
        ):
            self.assertEqual(getattr(reward_spec.weights, name), 0.0)
        self.assertEqual(reward[0].tolist(), [0.0, 0.0])

        setup_score = 750.0 / CAR_MAX_SPEED
        shot_score = 2.0 * 1350.0 * GOAL_Y / (
            math.hypot(GOAL_Y, GOAL_HEIGHT / 2 - 200.0) * CAR_MAX_SPEED
        )
        aerial_score = 600.0 * GOAL_Y / (
            math.hypot(GOAL_Y, GOAL_HEIGHT / 2 - 700.0) * CAR_MAX_SPEED
        )
        approach_speed = (1750.0 * 300.0 + 1250.0 * 250.0) / math.hypot(300, 250)
        velocity_score = 0.05 * (approach_speed / CAR_MAX_SPEED - 0.6) / 0.4
        for index, expected in (
            (1, setup_score), (2, shot_score), (3, -aerial_score),
            (4, velocity_score), (5, 5.0), (6, 10.0),
        ):
            with self.subTest(simulation=index):
                self.assertAlmostEqual(reward[index, 0].item(), expected, places=5)
                self.assertAlmostEqual(reward[index, 1].item(), -expected, places=5)

        self.assertEqual(set(
            key.removeprefix("reward_spec/component/")
            for key in result.info if key.startswith("reward_spec/component/")
        ), {
            "goal_scored", "shot", "air_dribble_setup", "car_velocity",
            "aerial_touch", "demo", "aerial_carry_progress",
            "aerial_speed_progress", "speed_progress", "boost_free_speed_progress",
            "soft_lift", "flip_reset",
        })
        self.assertGreater(result.info["reward_spec/component/shot"][2 * 2], 0)
        self.assertGreater(result.info["reward_spec/component/aerial_touch"][3 * 2 + 1], 0)

        dense = RewardSpec(normalize=False)(context)
        self.assertNotEqual(dense[0, 0].item(), 0.0)
        reward_spec.set_goal_scored_weight(7.0)
        self.assertEqual(reward_spec(context).reward[6].tolist(), [7.0, -7.0])

    def test_negative_or_small_impulses_do_not_pay_and_goals_remain_zero_sum(self):
        context = sparse_context()
        raw = context.current.raw.clone()
        raw[3, 4] = 600.0  # A high touch away from orange's goal.
        raw[4, 9:53].view(2, 22)[0, 4:6] *= -1  # Moving away from the ball.
        changed = replace(context, current=replace(context.current, raw=raw))

        reward = RewardSpec(normalize=False, sparse=True)(changed)
        torch.testing.assert_close(reward[3:5], torch.zeros(2, 2))
        torch.testing.assert_close(reward.sum(dim=-1), torch.zeros(len(reward)))

        normalized = RewardSpec(sparse=True)(changed)
        torch.testing.assert_close(normalized[0], torch.zeros(2))
        torch.testing.assert_close(normalized.sum(dim=-1), torch.zeros(len(normalized)))

        raw[3, 4] = -200.0  # Too small an impulse for a high aerial touch.
        small = RewardSpec(normalize=False, sparse=True)(changed)
        torch.testing.assert_close(small[3], torch.zeros(2))

        raw[3, 2] = GOAL_HEIGHT  # Exactly at crossbar height is not above it.
        raw[3, 4] = -600.0
        boundary = RewardSpec(normalize=False, sparse=True)(changed)
        torch.testing.assert_close(boundary[3], torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
