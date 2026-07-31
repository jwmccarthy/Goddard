import unittest
from unittest.mock import Mock, patch

import torch

from carl.gymnasium import CarlEvents, CarlState, RewardContext
from jarl.data import TensorBatch

from gaifo import parse_arguments
from imitation import AddImitationReward, EveryNUpdates
from rewards import (
    BALL_MAX_SPEED,
    CEILING_Z,
    GOAL_HEIGHT,
    SeerReward,
    SeerRewardWeights,
)


class HybridRewardTests(unittest.TestCase):
    @staticmethod
    def reward_context(
        previous_raw,
        current_raw,
        *,
        score_delta=0,
        score_difference=0,
        episode_ticks=0,
        overtime=False,
    ):
        team_sign = torch.tensor([1.0, -1.0])
        boost_positions = torch.zeros(34, 3)
        previous = CarlState.from_raw(
            previous_raw, 2, boost_positions, team_sign
        )
        current = CarlState.from_raw(current_raw, 2, boost_positions, team_sign)
        events = CarlEvents(
            score_delta=torch.tensor([score_delta], dtype=torch.float32),
            done=torch.tensor([True]),
            terminated=torch.tensor([bool(score_delta)]),
            truncated=torch.tensor([not bool(score_delta)]),
        )
        return RewardContext(
            current=current,
            previous=previous,
            events=events,
            actions=torch.zeros(2, 7),
            score_difference=torch.tensor([score_difference]),
            episode_ticks=torch.tensor([episode_ticks]),
            overtime=torch.tensor([overtime]),
        )

    @staticmethod
    def raw_state():
        raw = torch.zeros(1, 9 + 2 * 22 + 34)
        raw[:, 2] = 225.0
        for index in range(2):
            car = 9 + index * 22
            raw[:, car + 9] = 1.0
            raw[:, car + 14] = 1.0
        return raw

    def test_ball_state_rewards_are_normalized_and_attributed(self):
        weights = SeerRewardWeights()
        self.assertEqual(weights.ball_height, 0.00025)
        self.assertEqual(weights.ball_velocity, 0.00025)
        height, velocity = SeerReward._ball_state_rewards(
            torch.tensor([[[0.0, 0.0, CEILING_Z]]]),
            torch.tensor([[BALL_MAX_SPEED]]),
            torch.tensor([[1.0, 0.0]]),
        )

        torch.testing.assert_close(height, torch.tensor([[1.0, 0.0]]))
        torch.testing.assert_close(velocity, torch.tensor([[1.0, 0.0]]))

    def test_nexto_mechanics_reward_components(self):
        previous = self.raw_state()
        current = previous.clone()
        blue = 9
        orange = 9 + 22
        previous[:, blue + 15] = 0.0
        current[:, blue + 15] = 100.0
        previous[:, orange + 15] = 100.0
        current[:, orange + 15] = 0.0
        previous[:, blue + 18] = 1.0
        current[:, blue + 2] = 325.0
        current[:, blue + 21] = 1.0
        current[:, orange + 2] = 17.0
        current[:, orange + 16] = 1.0
        current[:, orange + 17] = 1.0

        result = SeerReward(1, 1, normalize=False, log_diagnostics=True)(
            self.reward_context(previous, current)
        )

        self.assertAlmostEqual(result.info["seer/component/boost_gain"][0], 1.0)
        self.assertAlmostEqual(
            result.info["seer/component/boost_loss"][1],
            -0.5 * (1.0 - 17.0 / GOAL_HEIGHT),
        )
        self.assertAlmostEqual(result.info["seer/component/aerial_touch"][0], 0.1)
        self.assertAlmostEqual(result.info["seer/component/flip_reset"][0], 10.0)
        self.assertAlmostEqual(result.info["seer/component/demo"][0], 2.5)
        self.assertAlmostEqual(result.info["seer/component/demo"][1], -2.5)
        self.assertAlmostEqual(result.info["seer/component/touch_grass"][1], -0.005)

    def test_goal_uses_transition_state_and_previous_ball_speed(self):
        previous = self.raw_state()
        current = previous.clone()
        previous[:, 3] = 3000.0
        current[:, 9 + 22] = 2300.0

        result = SeerReward(1, 1, normalize=False, log_diagnostics=True)(
            self.reward_context(
                previous,
                current,
                score_delta=1,
                score_difference=1,
                episode_ticks=5 * 60 * 120,
                overtime=True,
            )
        )

        self.assertAlmostEqual(result.info["seer/component/goal_scored"][0], 10.0)
        self.assertAlmostEqual(result.info["seer/component/goal_speed_bonus"][0], 1.25)
        self.assertGreater(result.info["seer/component/goal_distance_bonus"][0], 0.0)
        self.assertAlmostEqual(result.info["seer/component/win_probability"][0], 5.0)
        self.assertAlmostEqual(result.info["seer/component/win_probability"][1], -5.0)

    def test_seer_reward_is_disabled_by_default(self):
        with patch("sys.argv", ["gaifo.py"]):
            arguments = parse_arguments()

        self.assertFalse(arguments.seer_reward)

    def test_periodic_stage_runs_first_and_every_n_updates(self):
        stage = Mock()
        stage.run.side_effect = lambda experience: (experience, {"ran": {}})
        periodic = EveryNUpdates(stage, interval=3)

        metrics = [periodic.run("rollout")[1] for _ in range(7)]

        self.assertEqual(stage.run.call_count, 3)
        self.assertEqual(
            metrics,
            [{"ran": {}}, {}, {}, {"ran": {}}, {}, {}, {"ran": {}}],
        )

    def test_shaping_scale_does_not_change_goal_reward(self):
        reward = SeerReward(1, 1, normalize=False)
        reward.set_shaping_scale(0.25)

        components = reward._scale_components(
            {
                "goal_scored": torch.tensor([2.0]),
                "ball_touch": torch.tensor([2.0]),
            }
        )

        torch.testing.assert_close(components["goal_scored"], torch.tensor([2.0]))
        torch.testing.assert_close(components["ball_touch"], torch.tensor([0.5]))

    def test_imitation_is_added_at_full_strength(self):
        batch = TensorBatch(
            {
                "reward": torch.tensor([[1.0, 2.0]]),
                "imitation_reward": torch.tensor([[0.25, 0.5]]),
            }
        )

        combined = AddImitationReward()(batch, None)

        torch.testing.assert_close(
            combined["reward"], torch.tensor([[1.25, 2.5]])
        )


if __name__ == "__main__":
    unittest.main()
