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
    EgoBallTouchCapture,
    GameplayDiagnostics,
    SceneDiscriminatorReward,
    advanced_touch_events, build_ppo_transforms, parse_args,
)
from jarl.data import TensorBatch
from jarl.transform import PrepareContext
from replay_layout import team_scene_size
from reward_spec import BALL_RADIUS, CEILING_Z, GOAL_HEIGHT


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

    # A grounded touch is not an aerial touch.
    raw[1, 2] = BALL_RADIUS
    raw[1, 4] = 1200.0
    cars[1, 0, 2] = 17.0
    cars[1, 0, 16] = 1.0
    cars[1, 0, 21] = 1.0

    # Orange resets a spent flip with its wheels against a low airborne ball.
    raw[2, 2] = 350.0
    raw[2, 4] = -1200.0
    cars[2, 1, 2] = 500.0
    cars[2, 1, 14] = 1.0
    cars[2, 1, 16] = 1.0  # A wheel-contact ground flag does not suppress a reset.
    cars[2, 1, 21] = 1.0

    # Orange accelerates an aerial ball toward its goal, without a flip reset.
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
    def test_differential_short_window_rewards_improvement_not_expiring_history(self):
        windows = th.zeros(2, 6, 3, team_scene_size(3))
        windows[0, :, 0, 0] = -100
        windows[1, :, 0, 0] = 100  # A new game cannot change the prefix's baseline.
        windows[0, :, 1, 0] = 1
        windows[0, :, 2, 0] = th.tensor([0., 2., 3., 1., 0., 2.])
        windows[1, :, 1, 0] = 5
        windows[1, :, 2, 0] = th.tensor([4., 6., 5., 3., 7., 5.])
        valid = th.ones(2, 6, dtype=th.bool)
        valid[0, 5] = False
        ended = th.zeros_like(valid)
        ended[0, 2] = True

        class TeamPositionJudge(PositionDiscriminator):
            scene_size = team_scene_size(3)

        reward = SceneDiscriminatorReward(
            TeamPositionJudge(), noise_std=0, trajectory_length=3,
            differential=True, max_magnitude=1,
        )
        th.testing.assert_close(reward._score_windows(windows, valid, ended), th.tensor([
            [1., -1., -1., 0., 1., 0.],
            [1., -1., 0., 1., -1., 0.],
        ]))

        class FirstFrameJudge(th.nn.Module):
            scene_size = team_scene_size(3)

            def forward(self, scenes):
                return scenes[:, 0, 0]

        constant = SceneDiscriminatorReward(
            FirstFrameJudge(), noise_std=0.5, trajectory_length=3, differential=True,
        )
        th.testing.assert_close(
            constant._score_windows(windows, valid, ended), th.zeros(2, 6),
        )

    def test_physical_events_feed_zero_sum_ppo_rewards(self):
        context = touch_context()
        gameplay = GameplayDiagnostics(4, th.device("cpu"), 900)
        goal_only = gameplay(context)
        th.testing.assert_close(goal_only[0], th.tensor([1.0, -1.0]))
        th.testing.assert_close(goal_only[1:], th.zeros(3, 2))

        captured = AdvancedTouchCapture(gameplay)(None)
        aerial = captured["aerial_touch_score"].view(4, 2)
        flip = captured["flip_reset_event"].view(4, 2)
        self.assertGreater(aerial[0, 0].item(), aerial[3, 1].item())
        self.assertGreater(aerial[3, 1].item(), aerial[2, 1].item())
        self.assertGreater(aerial[2, 1].item(), 0.5)
        self.assertAlmostEqual(aerial[0, 1].item(), -aerial[0, 0].item())
        th.testing.assert_close(aerial[1], th.zeros(2))
        self.assertAlmostEqual(aerial[2, 0].item(), -aerial[2, 1].item())
        self.assertAlmostEqual(aerial[3, 0].item(), -aerial[3, 1].item())
        th.testing.assert_close(flip[2], th.tensor([-1.0, 1.0]))
        th.testing.assert_close(flip[3], th.zeros(2))
        touches = EgoBallTouchCapture(gameplay)(None)
        th.testing.assert_close(
            touches["ego_ball_touch"].view(4, 2),
            th.tensor([[True, False], [True, False], [False, True], [False, True]]),
        )
        th.testing.assert_close(
            touches["opponent_ball_touch"].view(4, 2),
            touches["ego_ball_touch"].view(4, 2).flip(-1),
        )

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
        self.assertTrue(result["learner_mask"].all())

        metrics = gameplay.diagnostic_metrics()["Gameplay"]
        self.assertEqual(metrics["aerial_touches_per_1000_steps"], 375.0)
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
        self.assertTrue(reward["learner_mask"].all())
        # Self-play still excludes non-learner actors, regardless of D context.
        masked = batch.with_fields(learner_mask=th.tensor([[True, False, True, False]]))
        replayed = SceneDiscriminatorReward(
            PositionDiscriminator(), noise_std=0.0, trajectory_length=2,
            batch_size=1,
        )(masked, PrepareContext())
        th.testing.assert_close(replayed["learner_mask"], masked["learner_mask"])
        self.assertNotIn("long_imitation_reward", reward)

    def test_exp_log_odds_reward_is_bounded_without_batch_centering(self):
        windows = th.zeros(1, 5, 2, 51)
        windows[0, :, -1, 0] = th.tensor([
            0.0, math.log(2.0), -math.log(2.0), -100.0, 0.0,
        ])
        batch = TensorBatch({
            "observation": th.zeros(1, 5, 51),
            "scene_window": windows,
            "scene_window_valid": th.tensor([[True, True, True, True, False]]),
            "reward": th.tensor([[0.0, 0.0, 0.0, 0.0, 2.0]]),
        })
        result = SceneDiscriminatorReward(
            PositionDiscriminator(), noise_std=0, trajectory_length=2,
            batch_size=2, max_magnitude=10.0, exp_log_odds_reward=True,
        )(batch, PrepareContext())
        th.testing.assert_close(
            result["imitation_reward"], th.tensor([[1.0, 0.5, 2.0, 10.0, 0.0]]),
        )
        th.testing.assert_close(
            result["training_reward"], th.tensor([[1.0, 0.5, 2.0, 10.0, 2.0]]),
        )
        self.assertTrue(result["learner_mask"].all())

    def test_replay_window_credit_scales_only_heads_with_borrowed_context(self):
        windows = th.zeros(1, 4, 2, 51)
        windows[0, :, -1, 0] = th.tensor([0., math.log(2), 0., 0.])
        batch = TensorBatch({
            "observation": th.zeros(1, 4, 51),
            "scene_window": windows,
            "scene_window_valid": th.tensor([[True, True, False, False]]),
            "scene_window_agent_fraction": th.tensor([[.25, 1., .25, 1.]]),
            "terminated": th.zeros(1, 4, dtype=th.bool),
            "reward": th.tensor([[0., 0., 2., 0.]]),
        })
        short = SceneDiscriminatorReward(
            PositionDiscriminator(), noise_std=0, trajectory_length=2,
            exp_log_odds_reward=True,
        )(batch, PrepareContext())
        th.testing.assert_close(short["imitation_reward"], th.tensor([[.25, .5, 0., 0.]]))
        th.testing.assert_close(short["training_reward"], th.tensor([[.25, .5, 2., 0.]]))

        class ConstantRecurrent(th.nn.Module):
            recurrent_global = True

            def score_sequence(self, scenes, reset, initial_state=None):
                return (scenes.new_zeros(scenes.shape[:2]),
                        scenes.new_zeros((1, scenes.shape[1], 1)))

        class FactorizedRecurrent(th.nn.Module):
            factorized = True
            recurrent_global = True

            def __init__(self):
                super().__init__()
                self.global_discriminator = ConstantRecurrent()

            def specialist_logits(self, scenes):
                return scenes.new_zeros((len(scenes), 2))

        global_only = SceneDiscriminatorReward(
            ConstantRecurrent(), noise_std=0, trajectory_length=2,
            exp_log_odds_reward=True,
        )(batch, PrepareContext())
        th.testing.assert_close(global_only["imitation_reward"], th.tensor([[1., 1., 0., 0.]]))

        combined = SceneDiscriminatorReward(
            FactorizedRecurrent(), noise_std=0, trajectory_length=2,
            exp_log_odds_reward=True,
        )(batch, PrepareContext())
        th.testing.assert_close(combined["near_imitation_reward"],
                                th.tensor([[.125, .5, 0., 0.]]))
        th.testing.assert_close(combined["global_imitation_reward"],
                                th.tensor([[.5, .5, 0., 0.]]))
        th.testing.assert_close(combined["training_reward"],
                                th.tensor([[.625, 1., 2., 0.]]))

        class ConstantTransformer(th.nn.Module):
            transformer_global = True

            def score_context(self, scenes, ages, *, return_previous=False):
                current = scenes.new_ones(len(scenes))
                return (current, current.new_zeros(len(scenes))) if return_previous else current

        transformer = SceneDiscriminatorReward(
            ConstantTransformer(), noise_std=0, trajectory_length=2,
        )(batch, PrepareContext())
        th.testing.assert_close(transformer["imitation_reward"],
                                th.tensor([[-1., -1., 0., 0.]]))

        with self.assertRaisesRegex(ValueError, "fraction must match"):
            SceneDiscriminatorReward(
                PositionDiscriminator(), noise_std=0, trajectory_length=2,
            )(TensorBatch({**batch, "scene_window_agent_fraction": th.ones(1, 2)}),
              PrepareContext())

    def test_state_rewards_scale_with_elapsed_ticks_but_events_and_changes_do_not(self):
        windows = th.zeros(1, 2, 2, 51)
        windows[0, 1, -1, 0] = math.log(2)
        batch = TensorBatch({
            "observation": th.zeros(1, 2, 51),
            "scene_window": windows,
            "scene_window_valid": th.ones(1, 2, dtype=th.bool),
            "scene_window_agent_fraction": th.tensor([[.25, 1.]]),
            "reward": th.tensor([[0., 2.]]),
            "aerial_touch_score": th.tensor([[0., .8]]),
            "flip_reset_event": th.tensor([[0., 1.]]),
        })
        for frameskip in (2, 4, 8):
            with self.subTest(frameskip=frameskip):
                scale = frameskip / 4
                result = SceneDiscriminatorReward(
                    PositionDiscriminator(), noise_std=0, trajectory_length=2,
                    exp_log_odds_reward=True, frameskip=frameskip,
                    goal_reward_weight=3, aerial_touch_reward_weight=.5,
                    flip_reset_reward_weight=1,
                )(batch, PrepareContext())
                th.testing.assert_close(result["imitation_reward"],
                                        th.tensor([[.25 * scale, .5 * scale]]))
                th.testing.assert_close(result["goal_reward"], th.tensor([[0., 6.]]))
                th.testing.assert_close(result["aerial_touch_reward"],
                                        th.tensor([[0., .4]]))
                th.testing.assert_close(result["flip_reset_reward"],
                                        th.tensor([[0., 1.]]))
                th.testing.assert_close(result["training_reward"],
                                        th.tensor([[.25 * scale, 7.4 + .5 * scale]]))

                changed = windows.clone()
                changed[0, 0, 0, 0] = math.log(2)
                changed[0, 0, 1, 0] = 0
                changed_batch = TensorBatch({
                    "observation": th.zeros(1, 2, 51),
                    "scene_window": changed,
                    "scene_window_valid": th.ones(1, 2, dtype=th.bool),
                    "reward": th.zeros(1, 2),
                })
                delta = SceneDiscriminatorReward(
                    PositionDiscriminator(), noise_std=0, trajectory_length=2,
                    exp_log_odds_reward=True, differential=True,
                    gamma=.9, frameskip=frameskip,
                )(changed_batch, PrepareContext())
                self.assertAlmostEqual(
                    float(delta["imitation_reward"][0, 0]),
                    1 - .5 * .9 ** scale, places=6,
                )

        with self.assertRaisesRegex(ValueError, "frame skip must be positive"):
            SceneDiscriminatorReward(
                PositionDiscriminator(), noise_std=0, trajectory_length=2, frameskip=0,
            )

    def test_transformer_global_change_stays_unscaled_while_specialists_are_state_rewards(self):
        class ConstantTransformer(th.nn.Module):
            def score_context(self, scenes, ages, *, return_previous=False):
                current = scenes.new_full((len(scenes),), math.log(2))
                return (current, current.new_zeros(len(scenes))) if return_previous else current

        class FactorizedTransformer(th.nn.Module):
            factorized = True
            transformer_global = True

            def __init__(self):
                super().__init__()
                self.global_discriminator = ConstantTransformer()

            def specialist_logits(self, windows):
                return windows.new_zeros((len(windows), 2))

        batch = TensorBatch({
            "observation": th.zeros(1, 2, 51),
            "scene_window": th.zeros(1, 2, 2, 51),
            "scene_window_valid": th.ones(1, 2, dtype=th.bool),
            "terminated": th.zeros(1, 2, dtype=th.bool),
            "reward": th.zeros(1, 2),
        })
        for frameskip in (2, 8):
            with self.subTest(frameskip=frameskip):
                output = SceneDiscriminatorReward(
                    FactorizedTransformer(), noise_std=0, trajectory_length=2,
                    exp_log_odds_reward=True, frameskip=frameskip,
                )(batch, PrepareContext())
                th.testing.assert_close(output["near_imitation_reward"],
                                        th.full((1, 2), .5 * frameskip / 4))
                th.testing.assert_close(output["global_imitation_reward"],
                                        th.full((1, 2), -.25))

    def test_ppo_discounts_and_differential_odds_share_frame_skip_timing(self):
        for frameskip in (2, 4, 8):
            with self.subTest(frameskip=frameskip), patch.object(sys, "argv", [
                "gaifo.py", "--replay-dir", "parsed_replays",
                "--frameskip", str(frameskip), "--gamma", "0.81", "--lambda", "0.64",
                "--differential", "--exp-log-odds-reward",
            ]):
                args, _ = parse_args()
                imitation, gae, _ = build_ppo_transforms(args, PositionDiscriminator())
                expected_gamma = .81 ** (frameskip / 4)
                expected_lambda = .64 ** (frameskip / 4)
                self.assertAlmostEqual(imitation.gamma, expected_gamma)
                self.assertAlmostEqual(gae.gamma, expected_gamma)
                self.assertAlmostEqual(gae.lambda_, expected_lambda)
                self.assertEqual(gae.reward_field, "training_reward")
                self.assertAlmostEqual(imitation.state_reward_scale, frameskip / 4)
                self.assertAlmostEqual(
                    float(imitation._score_change(th.zeros(1), th.zeros(1))[0]),
                    1 - expected_gamma,
                )
                transitions = TensorBatch({
                    "baseline_value": th.zeros(2, 2),
                    "baseline_next_value": th.zeros(2, 2),
                    "training_reward": th.tensor([[0., 0.], [1., 1.]]),
                    "terminated": th.zeros(2, 2, dtype=th.bool),
                    "truncated": th.zeros(2, 2, dtype=th.bool),
                })
                advantages = gae(transitions, PrepareContext())["advantage"]
                th.testing.assert_close(
                    advantages[0], th.full((2,), expected_gamma * expected_lambda),
                )

    def test_aerial_bonus_increases_with_height_without_goalward_acceleration(self):
        context = touch_context()
        raw = context.current.raw.clone()
        raw[0, 4] = context.previous.ball_velocity[0, 1]  # No velocity gain.
        raw[0, 2] = 250.0
        raw[0, 9 + 2] = 210.0
        raw[3, 4] = 300.0  # Ball velocity gain away from orange's goal.
        raw[3, 2] = GOAL_HEIGHT
        current = replace(context.current, raw=raw)
        scores, touches, _ = advanced_touch_events(replace(context, current=current))
        self.assertTrue(touches[0, 0])
        self.assertTrue(touches[3, 1])
        self.assertAlmostEqual(scores[0, 0].item(),
                               0.5 + 0.5 * (250 - BALL_RADIUS) / (CEILING_Z - BALL_RADIUS))
        self.assertGreater(scores[3, 1].item(), scores[0, 0].item())

        raw[0, 9 + 2] = 2 * BALL_RADIUS  # Exactly at the minimum is not airborne.
        raw[3, 9 + 22 + 0] = 4_070.0
        raw[3, 9 + 22 + 14] = 0.0
        raw[3, 9 + 22 + 16] = 1.0  # A side-wall touch is not an aerial touch.
        scores, touches, _ = advanced_touch_events(replace(context, current=current))
        self.assertFalse(touches[0, 0])
        self.assertFalse(touches[3, 1])
        th.testing.assert_close(scores[3], th.zeros(2))

        raw[0, 2] = CEILING_Z
        raw[0, 9 + 2] = CEILING_Z - 100
        scores, touches, _ = advanced_touch_events(replace(context, current=current))
        self.assertTrue(touches[0, 0])
        self.assertEqual(scores[0, 0].item(), 1.0)

    def test_reward_weight_flags(self):
        with patch.object(sys, "argv", [
            "gaifo.py", "--replay-dir", "parsed_replays",
            "--aerial-touch-reward-weight", "0.25",
            "--flip-reset-reward-weight", "2.0",
            "--exp-log-odds-reward", "--recency-replay",
            "--history-reservoir-fraction", "0.3",
        ]):
            args, _ = parse_args()
        self.assertEqual(args.aerial_touch_reward_weight, 0.25)
        self.assertEqual(args.flip_reset_reward_weight, 2.0)
        self.assertTrue(args.exp_log_odds_reward)
        self.assertTrue(args.recency_replay)
        self.assertEqual(args.history_reservoir_fraction, 0.3)
        with patch.object(sys, "argv", [
            "gaifo.py", "--replay-dir", "parsed_replays",
        ]):
            defaults, _ = parse_args()
        self.assertFalse(defaults.exp_log_odds_reward)
        self.assertFalse(defaults.recency_replay)


if __name__ == "__main__":
    unittest.main()
