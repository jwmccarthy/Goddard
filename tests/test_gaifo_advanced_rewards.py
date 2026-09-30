import math
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch as th

from carl.gymnasium.state import (
    BOOST_PAD_POSITIONS,
    CARLObservation,
    CarlEvents,
    CarlState,
    RewardContext,
)
from gaifo import (
    AdvancedTouchCapture,
    GameplayDiagnostics,
    SceneDiscriminatorReward,
    advanced_touch_events,
    parse_args,
)
from jarl.data import TensorBatch
from jarl.transform import PrepareContext
from reward_spec import CAR_MAX_SPEED, GOAL_HEIGHT, GOAL_Y


class ZeroDiscriminator(th.nn.Module):
    def forward(self, windows):
        return th.zeros(len(windows), device=windows.device)


class PositionDiscriminator(th.nn.Module):
    def forward(self, windows):
        return windows[:, -1, 0]


def touch_context() -> RewardContext:
    n_sim = 4
    raw = th.zeros(n_sim, 9 + 22 * 2 + len(BOOST_PAD_POSITIONS))
    cars = raw[:, 9:53].view(n_sim, 2, 22)

    # Blue scores while accelerating a high ball toward the opponent goal.
    raw[0, 2] = 700.0
    raw[0, 4] = 1200.0
    cars[0, 0, 2] = 600.0
    cars[0, 0, 21] = 1.0

    # Touching a high ball from the ground is not an aerial touch.
    raw[1, 2] = 700.0
    raw[1, 4] = 1200.0
    cars[1, 0, 2] = 600.0
    cars[1, 0, 16] = 1.0
    cars[1, 0, 21] = 1.0

    # Orange recovers a spent flip below crossbar height; no aerial bonus.
    raw[2, 2] = 350.0
    raw[2, 4] = -1200.0
    cars[2, 1, 2] = 500.0
    cars[2, 1, 14] = 1.0
    cars[2, 1, 16] = 1.0  # A wheel-contact ground flag does not suppress a reset.
    cars[2, 1, 21] = 1.0

    # Orange accelerates a high ball toward its goal, without a flip reset.
    raw[3, 2] = 650.0
    raw[3, 4] = -1200.0
    cars[3, 1, 2] = 500.0
    cars[3, 1, 14] = 1.0
    cars[3, 1, 21] = 1.0

    previous = raw.clone()
    previous[:, 4] = th.tensor([300.0, 300.0, -300.0, -300.0])
    previous[:, 9:53].view(n_sim, 2, 22)[2:, 1, 18] = 1.0
    boost_pads = th.tensor(BOOST_PAD_POSITIONS)
    team_sign = th.tensor([1.0, -1.0])
    observation = CARLObservation.from_tensor(th.zeros(n_sim * 2, 51), 2)
    return RewardContext(
        current=CarlState.from_raw(raw, 2, boost_pads, team_sign),
        previous=CarlState.from_raw(previous, 2, boost_pads, team_sign),
        current_observation=observation,
        previous_observation=observation,
        events=CarlEvents(
            score_delta=th.tensor([1.0, 0.0, 0.0, 0.0]),
            done=th.tensor([True, False, False, False]),
            terminated=th.tensor([True, False, False, False]),
            truncated=th.zeros(n_sim, dtype=th.bool),
        ),
        actions=th.zeros(n_sim * 2, 7),
        score_difference=th.zeros(n_sim),
        episode_ticks=th.zeros(n_sim),
        overtime=th.zeros(n_sim, dtype=th.bool),
    )


class AdvancedGAIFORewardTests(unittest.TestCase):
    def test_physical_events_feed_zero_sum_ppo_rewards(self):
        context = touch_context()
        gameplay = GameplayDiagnostics(4, th.device("cpu"), 900)
        goal_only = gameplay(context)
        th.testing.assert_close(goal_only[0], th.tensor([1.0, -1.0]))
        th.testing.assert_close(goal_only[1:], th.zeros(3, 2))

        captured = AdvancedTouchCapture(gameplay)(None)
        aerial = captured["aerial_touch_score"].view(4, 2)
        flip = captured["flip_reset_event"].view(4, 2)
        blue_score = (
            900.0 * GOAL_Y
            / math.hypot(GOAL_Y, GOAL_HEIGHT / 2 - 700.0)
            / CAR_MAX_SPEED
        )
        orange_score = (
            900.0 * GOAL_Y
            / math.hypot(GOAL_Y, GOAL_HEIGHT / 2 - 650.0)
            / CAR_MAX_SPEED
        )
        self.assertAlmostEqual(aerial[0, 0].item(), blue_score)
        self.assertAlmostEqual(aerial[0, 1].item(), -blue_score)
        th.testing.assert_close(aerial[1], th.zeros(2))
        th.testing.assert_close(aerial[2], th.zeros(2))
        self.assertAlmostEqual(aerial[3, 1].item(), orange_score)
        self.assertAlmostEqual(aerial[3, 0].item(), -orange_score)
        th.testing.assert_close(flip[2], th.tensor([-1.0, 1.0]))
        th.testing.assert_close(flip[3], th.zeros(2))

        batch = TensorBatch({
            "observation": th.zeros(1, 8, 51),
            "reward": goal_only.reshape(1, 8),
            "scene_window": th.zeros(1, 8, 2, 51),
            "scene_window_valid": th.zeros(1, 8, dtype=th.bool),
            **{name: value.unsqueeze(0) for name, value in captured.items()},
        })
        transform = SceneDiscriminatorReward(
            discriminator=ZeroDiscriminator(),
            noise_std=0.0,
            trajectory_length=2,
            goal_reward_weight=3.0,
            aerial_touch_reward_weight=0.5,
            flip_reset_reward_weight=1.0,
        )
        result = transform(batch, PrepareContext())
        expected = (
            goal_only.reshape(1, 8) * 3.0
            + aerial.reshape(1, 8) * 0.5
            + flip.reshape(1, 8)
        )
        th.testing.assert_close(result["training_reward"], expected)
        th.testing.assert_close(result["imitation_reward"], th.zeros(1, 8))
        th.testing.assert_close(
            result["training_reward"].view(4, 2).sum(-1), th.zeros(4)
        )
        th.testing.assert_close(
            result["learner_mask"].view(4, 2).any(-1),
            th.tensor([True, False, True, True]),
        )

        metrics = gameplay.diagnostic_metrics()["Gameplay"]
        self.assertEqual(metrics["aerial_touches_per_1000_steps"], 250.0)
        self.assertEqual(metrics["flip_resets_per_1000_steps"], 125.0)

        # Continuing ball contact cannot award a second flip reset.
        gameplay(replace(context, previous=context.current))
        self.assertFalse(gameplay.last_flip_reset.any())

    def test_short_window_reward_and_terminal_event_on_invalid_window(self):
        windows = th.zeros(1, 4, 2, 51)
        windows[0, :, -1, 0] = th.tensor([-2.0, 0.0, 3.0, 0.0])
        batch = TensorBatch({
            "observation": th.zeros(1, 4, 51),
            "scene_window": windows,
            "scene_window_valid": th.tensor([[True, False, True, False]]),
            "reward": th.tensor([[0.0, 0.0, 0.0, 2.0]]),
        })
        reward = SceneDiscriminatorReward(
            PositionDiscriminator(), noise_std=0.0, trajectory_length=2,
            batch_size=1,
        )(batch, PrepareContext())
        th.testing.assert_close(
            reward["imitation_reward"], th.tensor([[1.0, 0.0, -1.0, 0.0]])
        )
        th.testing.assert_close(
            reward["training_reward"], th.tensor([[1.0, 0.0, -1.0, 2.0]])
        )
        th.testing.assert_close(
            reward["learner_mask"], th.tensor([[True, False, True, True]])
        )
        self.assertNotIn("long_imitation_reward", reward)

    def test_aerial_bonus_requires_height_and_goalward_acceleration(self):
        context = touch_context()
        raw = context.current.raw.clone()
        raw[0, 4] = context.previous.ball_velocity[0, 1]  # No velocity gain.
        raw[3, 4] = 300.0  # Velocity gain away from orange's goal.
        current = replace(context.current, raw=raw)
        scores, touches, _ = advanced_touch_events(replace(context, current=current))
        self.assertTrue(touches[0, 0])
        self.assertTrue(touches[3, 1])
        th.testing.assert_close(scores, th.zeros(4, 2))

        raw[3, 2] = GOAL_HEIGHT  # Exactly at the crossbar is not above it.
        raw[3, 4] = -1200.0
        scores, touches, _ = advanced_touch_events(replace(context, current=current))
        self.assertFalse(touches[3, 1])
        th.testing.assert_close(scores[3], th.zeros(2))

    def test_reward_weight_flags(self):
        with patch.object(sys, "argv", [
            "gaifo.py", "--replay-dir", "parsed_replays",
            "--aerial-touch-reward-weight", "0.25",
            "--flip-reset-reward-weight", "2.0",
        ]):
            args, _ = parse_args()
        self.assertEqual(args.aerial_touch_reward_weight, 0.25)
        self.assertEqual(args.flip_reset_reward_weight, 2.0)


if __name__ == "__main__":
    unittest.main()
