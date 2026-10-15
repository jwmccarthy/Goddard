import math
import sys
import unittest
from dataclasses import asdict, replace
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
    REWARD_PLUS_COMPONENTS,
    RewardPlusWeights,
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


def physical_state(n_sim: int, n_cars: int = 2) -> th.Tensor:
    raw = th.zeros(n_sim, 9 + 22 * n_cars + len(BOOST_PAD_POSITIONS))
    raw[:, 2] = BALL_RADIUS
    cars = raw[:, 9:9 + 22 * n_cars].view(n_sim, n_cars, 22)
    cars[:, :n_cars // 2, 1] = -120.0
    cars[:, n_cars // 2:, 1] = 120.0
    cars[:, :, 2] = 17.0
    cars[:, :, 14] = 1.0
    cars[:, :, 15] = 50.0
    cars[:, :, 16] = 1.0
    return raw


def physical_context(
    current: th.Tensor, previous: th.Tensor, *, n_cars: int = 2,
    actions: th.Tensor | None = None, done: th.Tensor | None = None,
    episode_ticks: int = 120,
) -> RewardContext:
    n_sim = len(current)
    pads = th.tensor(BOOST_PAD_POSITIONS)
    team_size = n_cars // 2
    teams = th.tensor([1.0] * team_size + [-1.0] * team_size)
    observation = CARLObservation.from_tensor(th.zeros(n_sim * n_cars, 51), n_cars)
    if done is None:
        done = th.zeros(n_sim, dtype=th.bool)
    if actions is None:
        actions = th.zeros(n_sim * n_cars, 7, dtype=th.int32)
    return RewardContext(
        current=CarlState.from_raw(current, n_cars, pads, teams),
        previous=CarlState.from_raw(previous, n_cars, pads, teams),
        current_observation=observation,
        previous_observation=observation,
        events=CarlEvents(
            score_delta=th.zeros(n_sim), done=done, terminated=done,
            truncated=th.zeros(n_sim, dtype=th.bool),
        ),
        actions=actions,
        score_difference=th.zeros(n_sim),
        episode_ticks=th.full((n_sim,), episode_ticks),
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
        th.testing.assert_close(touches["other_ball_touch"],
                                touches["opponent_ball_touch"])

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

    def test_reward_plus_speed_rewards_boost_free_gains_and_charges_losses(self):
        current = physical_state(8)
        previous = current.clone()
        cars = current[:, 9:53].view(8, 2, 22)
        old_cars = previous[:, 9:53].view(8, 2, 22)
        cars[:, 0, 4] = 1_000.0
        old_cars[:, 0, 4] = 600.0
        actions = th.zeros(16, 7, dtype=th.int32)
        actions.view(8, 2, 7)[1, 0, 4] = 1  # Boost held despite no net boost loss.
        cars[2, 0, 15] = 45.0  # Boost spent.
        cars[3, 0, 15] = 62.0  # Boost pad picked up.
        cars[4, 0, 20] = 1.0  # Still boosting.
        cars[5, 0, 17] = 1.0  # Demoed car cannot farm a speed change.
        cars[6, 0, 1] = 1_000.0  # Teleport cannot farm a speed change.
        done = th.zeros(8, dtype=th.bool)
        done[7] = True

        gameplay = GameplayDiagnostics(8, th.device("cpu"), 900, reward_plus=True)
        gameplay(physical_context(current, previous, actions=actions, done=done))
        speed = AdvancedTouchCapture(gameplay)(None)["reward_plus_speed"].view(8, 2)
        delta = 400.0 / 2300.0
        self.assertAlmostEqual(speed[0, 0].item(), 37.5 * delta, places=5)
        th.testing.assert_close(speed[1:5, 0], th.full((4,), 12.5 * delta))
        th.testing.assert_close(speed[5:], th.zeros(3, 2))
        th.testing.assert_close(speed.sum(dim=-1), th.zeros(8))

        reversed_actions = actions[:2].clone()
        reversed_actions[0, 4] = 1
        single_gameplay = GameplayDiagnostics(1, th.device("cpu"), 900, reward_plus=True)
        single_gameplay(physical_context(
            previous[:1], current[:1], actions=reversed_actions,
        ))
        reversed_speed = AdvancedTouchCapture(single_gameplay)(None)["reward_plus_speed"].view(1, 2)
        self.assertAlmostEqual(reversed_speed[0, 0].item(), -37.5 * delta, places=5)
        single_gameplay(physical_context(current[:1], current[:1]))
        th.testing.assert_close(
            AdvancedTouchCapture(single_gameplay)(None)["reward_plus_speed"], th.zeros(2),
        )

    def test_reward_plus_boost_gain_pays_for_pad_refills_but_not_resets_or_demos(self):
        current = physical_state(8)
        previous = current.clone()
        cars = current[:, 9:53].view(8, 2, 22)
        old_cars = previous[:, 9:53].view(8, 2, 22)
        old_cars[:, 0, 15] = 0.0
        cars[:, 0, 15] = 100.0
        cars[0, 0, 15] = 12.0  # A small pad is worth more when nearly empty.
        old_cars[2, 0, 15] = 25.0
        old_cars[3, 0, 15] = 100.0
        cars[3, 0, 15] = 75.0  # Spending boost is not a gain.
        cars[4, 0, 17] = 1.0  # Demolition refills do not pay.
        old_cars[5, 0, 17] = 1.0
        cars[6, 0, 1] = 1_000.0  # A discontinuity is not a pad pickup.
        done = th.zeros(8, dtype=th.bool)
        done[7] = True

        context = physical_context(current, previous, done=done)
        gameplay = GameplayDiagnostics(8, th.device("cpu"), 900, reward_plus=True)
        gameplay(context)
        bonus = AdvancedTouchCapture(gameplay)(None)["reward_plus_boost_gain"].view(8, 2)
        th.testing.assert_close(
            bonus[:, 0], th.tensor([12.5 * math.sqrt(.12), 12.5, 6.25, 0., 0., 0., 0., 0.]),
        )
        th.testing.assert_close(bonus.sum(dim=-1), th.zeros(8))

        legacy = GameplayDiagnostics(
            8, th.device("cpu"), 900, reward_plus=True,
            reward_plus_weights=RewardPlusWeights.legacy(),
        )
        legacy(context)
        th.testing.assert_close(
            AdvancedTouchCapture(legacy)(None)["reward_plus_boost_gain"], th.zeros(16),
        )

    def test_reward_plus_aerial_lifts_and_shots_require_controlled_touch_changes(self):
        current = physical_state(7)
        current[:, 2] = 800.0
        cars = current[:, 9:53].view(7, 2, 22)
        cars[:, 0, 2] = 650.0
        cars[:, 0, 16] = 0.0
        cars[:, 0, 21] = 1.0
        current[:, 4] = 1_100.0
        current[:, 5] = 400.0
        current[1, 4:6] = th.tensor([3_300.0, 1_500.0])  # Hard upward hit.
        current[2, 4:6] = th.tensor([100.0, -500.0])  # Downward hit.
        cars[3, 0, 21] = 0.0  # Same change without a touch.
        cars[4, 0, 2] = 17.0
        cars[4, 0, 16] = 1.0  # Ground touch.
        current[5, 2] = 600.0  # Below the crossbar.
        cars[5, 0, 2] = 500.0
        current[6, 4:6] = th.tensor([100.0, 0.0])  # Spoils a fast aerial shot.

        previous = current.clone()
        previous[:, 4] = 600.0
        previous[:, 5] = 0.0
        previous[2, 5] = 300.0
        previous[6, 4:6] = th.tensor([1_500.0, -100.0])
        gameplay = GameplayDiagnostics(7, th.device("cpu"), 900, reward_plus=True)
        gameplay(physical_context(current, previous))
        bonus = AdvancedTouchCapture(gameplay)(None)
        lift = bonus["reward_plus_aerial_lift"].view(7, 2)
        shot = bonus["reward_plus_aerial_shot"].view(7, 2)
        legacy = GameplayDiagnostics(
            7, th.device("cpu"), 900, reward_plus=True,
            reward_plus_weights=RewardPlusWeights.legacy(),
        )
        legacy(physical_context(current, previous))
        old_bonus = AdvancedTouchCapture(legacy)(None)
        th.testing.assert_close(lift, old_bonus["reward_plus_aerial_lift"].view(7, 2) * 50)
        th.testing.assert_close(shot, old_bonus["reward_plus_aerial_shot"].view(7, 2) * 50)
        self.assertGreater(lift[0, 0], 0)
        self.assertEqual(lift[1, 0], 0)  # Excessive relative-speed increase removes lift.
        self.assertLess(lift[2, 0], 0)
        th.testing.assert_close(lift[3:5], th.zeros(2, 2))
        self.assertGreater(lift[5, 0], 0)
        self.assertLess(lift[5, 0], lift[0, 0])  # The same lift is worth less when lower.
        self.assertGreater(shot[0, 0], 0)
        th.testing.assert_close(shot[3:6], th.zeros(3, 2))
        self.assertLess(shot[6, 0], 0)
        th.testing.assert_close(lift.sum(-1), th.zeros(7))
        th.testing.assert_close(shot.sum(-1), th.zeros(7))

    def test_reward_plus_aerial_control_waits_until_ball_is_1_5_times_car_speed(self):
        current = physical_state(4)
        current[:, 2] = 800.0
        cars = current[:, 9:53].view(4, 2, 22)
        cars[:, 0, 2] = 650.0
        cars[:, 0, 4] = 600.0
        cars[:, 0, 16] = 0.0
        cars[:, 0, 21] = 1.0
        previous = current.clone()
        previous[:, 4] = 600.0
        current[:, 4] = th.tensor([500.0, 700.0, 1_000.0, 1_300.0])
        current[:, 5] = 400.0

        gameplay = GameplayDiagnostics(4, th.device("cpu"), 900, reward_plus=True)
        gameplay(physical_context(current, previous))
        lift = AdvancedTouchCapture(gameplay)(None)["reward_plus_aerial_lift"].view(4, 2)
        self.assertGreater(lift[0, 0], 0)
        th.testing.assert_close(lift[0, 0], lift[1, 0])  # Below 1.5x car speed.
        self.assertGreater(lift[1, 0], lift[2, 0])
        self.assertGreater(lift[2, 0], lift[3, 0])
        th.testing.assert_close(lift.sum(-1), th.zeros(4))

    def test_reward_plus_underside_is_discounted_progress_and_terminally_debited(self):
        current = physical_state(9)
        current[:, 2] = 800.0
        cars = current[:, 9:53].view(9, 2, 22)
        cars[:, 0, 1] = 0.0
        cars[:, 0, 2] = 650.0
        cars[:, 0, 14] = -1.0  # Wheels face the underside of the ball.
        cars[:, 0, 16] = 0.0
        previous = current.clone()
        old_cars = previous[:, 9:53].view(9, 2, 22)
        old_cars[[0, 4, 5, 7], 0, 14] = 1.0
        cars[2, 0, 14] = 1.0
        current[4, 2] = previous[4, 2] = 350.0
        cars[4, 0, 2] = old_cars[4, 0, 2] = 220.0
        cars[5, 0, 14] = 1.0
        cars[6, 0, 2] = old_cars[6, 0, 2] = 900.0  # Car above the ball.
        cars[6, 0, 14] = old_cars[6, 0, 14] = 1.0
        old_cars[7, 0, 0] = 2_000.0  # Teleport to alignment is not progress.
        cars[8, 0, 17] = 1.0  # Demo loses the potential rather than farming it.
        done = th.zeros(9, dtype=th.bool)
        done[3] = True

        gameplay = GameplayDiagnostics(
            9, th.device("cpu"), 900, reward_plus=True, frameskip=8, gamma=.81,
        )
        gameplay(physical_context(current, previous, done=done))
        underside = AdvancedTouchCapture(gameplay)(None)["reward_plus_underside"].view(9, 2)
        effective_gamma = .81 ** 2
        self.assertAlmostEqual(underside[0, 0].item(), 25 * effective_gamma, places=5)
        self.assertAlmostEqual(underside[1, 0].item(), 25 * (effective_gamma - 1), places=5)
        for index in (2, 3, 8):
            self.assertAlmostEqual(underside[index, 0].item(), -25)
        self.assertGreater(underside[4, 0], 0)
        self.assertLess(underside[4, 0], underside[0, 0])
        th.testing.assert_close(underside[[5, 6, 7]], th.zeros(3, 2))
        th.testing.assert_close(underside.sum(-1), th.zeros(9))

    def test_reward_plus_underside_requires_no_actionable_dodge(self):
        current = physical_state(4)
        current[:, 2] = 800.0
        cars = current[:, 9:53].view(4, 2, 22)
        cars[:, 0, 1] = 0.0
        cars[:, 0, 2] = 650.0
        cars[:, 0, 14] = -1.0
        cars[:, 0, 16] = 0.0
        previous = current.clone()
        previous_cars = previous[:, 9:53].view(4, 2, 22)
        previous_cars[[0, 1, 3], 0, 14] = 1.0
        previous_cars[2, 0, 18] = 1.0  # A flip reset restores the spent dodge.
        cars[2, 0, 21] = 1.0

        before = th.zeros(8, 51)
        after = th.zeros_like(before)
        before[0, -2:] = after[0, -2:] = th.tensor([1.0, 1.25])
        before[2, -2:] = after[2, -2:] = th.tensor([1.0, 0.005])  # Dodge expired.
        after[4, -2:] = th.tensor([1.0, 1.25])  # Reset gives the dodge back.
        before[6, -2:] = th.tensor([1.0, 1.25])
        after[6, -2:] = th.tensor([1.0, 0.005])

        context = replace(
            physical_context(current, previous),
            current_observation=CARLObservation.from_tensor(after, 2),
            previous_observation=CARLObservation.from_tensor(before, 2),
        )
        gameplay = GameplayDiagnostics(4, th.device("cpu"), 900, reward_plus=True)
        gameplay(context)
        underside = AdvancedTouchCapture(gameplay)(None)["reward_plus_underside"].view(4, 2)
        self.assertEqual(underside[0, 0].item(), 0)  # Aligning with a dodge pays nothing.
        self.assertGreater(underside[1, 0], 0)  # Expired, though no flip was spent.
        self.assertAlmostEqual(underside[2, 0].item(), -25)  # Restore debits potential.
        self.assertGreater(underside[3, 0], 0)
        th.testing.assert_close(underside.sum(-1), th.zeros(4))

    def test_reward_plus_kickoff_speed_rewards_only_closest_challengers(self):
        for team_size in (1, 2, 3):
            with self.subTest(team_size=team_size):
                n_cars = 2 * team_size
                spawn = physical_state(1, n_cars=n_cars)
                cars = spawn[:, 9:9 + 22 * n_cars].view(1, n_cars, 22)
                cars[:, :team_size, 1] = -3_500.0
                cars[:, team_size:, 1] = 3_500.0
                blue, orange = team_size - 1, n_cars - 1
                cars[0, blue, 1] = -2_500.0
                cars[0, orange, 1] = 2_500.0

                moved = spawn.clone()
                moved_cars = moved[:, 9:9 + 22 * n_cars].view(1, n_cars, 22)
                moved_cars[0, blue, 1] += 100.0
                moved_cars[0, orange, 1] += 100.0  # Retreat costs the orange team.
                if team_size > 1:
                    moved_cars[0, 0, 1] += 100.0  # A non-challenger cannot farm.
                    moved_cars[0, team_size, 1] -= 100.0

                gameplay = GameplayDiagnostics(
                    1, th.device("cpu"), 900, n_cars=n_cars, reward_plus=True,
                )
                capture = AdvancedTouchCapture(gameplay)
                gameplay(physical_context(moved, spawn, n_cars=n_cars, episode_ticks=4))
                speed = capture(None)["reward_plus_kickoff_speed"]
                th.testing.assert_close(speed, th.tensor(
                    [1.0] * team_size + [-1.0] * team_size,
                ), atol=2e-6, rtol=0)

                gameplay(physical_context(spawn, moved, n_cars=n_cars))
                th.testing.assert_close(
                    speed + capture(None)["reward_plus_kickoff_speed"],
                    th.zeros(n_cars), atol=2e-6, rtol=0,
                )

                touched = spawn.clone()
                touched_cars = touched[:, 9:9 + 22 * n_cars].view(1, n_cars, 22)
                touched_cars[0, blue, 1] += 100.0
                touched_cars[0, blue, 21] = 1.0
                gameplay(physical_context(touched, spawn, n_cars=n_cars))
                th.testing.assert_close(
                    capture(None)["reward_plus_kickoff_speed"], th.tensor(
                        [0.5] * team_size + [-0.5] * team_size,
                    ), atol=2e-6, rtol=0,
                )
                th.testing.assert_close(
                    capture(None)["reward_plus_kickoff"], th.tensor(
                        [25.0] * team_size + [-25.0] * team_size,
                    ),
                )
                gameplay(physical_context(moved, touched, n_cars=n_cars))
                th.testing.assert_close(
                    capture(None)["reward_plus_kickoff_speed"], th.zeros(n_cars),
                )

                new_kickoff = GameplayDiagnostics(
                    1, th.device("cpu"), 900, n_cars=n_cars, reward_plus=True,
                )
                new_kickoff(physical_context(moved, spawn, n_cars=n_cars, episode_ticks=4))
                new_kickoff(physical_context(spawn, moved, n_cars=n_cars,
                                             episode_ticks=500))
                th.testing.assert_close(
                    AdvancedTouchCapture(new_kickoff)(None)["reward_plus_kickoff_speed"],
                    th.zeros(n_cars),
                )

                replay = spawn.clone()
                replay[0, 0] = 500.0  # A replay reset, not a centered kickoff.
                replay_moved = replay.clone()
                replay_moved[:, 9:9 + 22 * n_cars].view(1, n_cars, 22)[0, blue, 1] += 100.0
                new_kickoff(physical_context(replay_moved, replay, n_cars=n_cars,
                                             episode_ticks=4))
                th.testing.assert_close(
                    AdvancedTouchCapture(new_kickoff)(None)["reward_plus_kickoff_speed"],
                    th.zeros(n_cars),
                )

    def test_reward_plus_kickoff_first_touch_is_once_per_real_kickoff(self):
        spawn = physical_state(1, n_cars=4)
        cars = spawn[:, 9:97].view(1, 4, 22)
        cars[:, :2, 1] = -2_500.0
        cars[:, 2:, 1] = 2_500.0
        gameplay = GameplayDiagnostics(1, th.device("cpu"), 900, n_cars=4,
                                       reward_plus=True)
        capture = AdvancedTouchCapture(gameplay)

        gameplay(physical_context(spawn, spawn, n_cars=4, episode_ticks=4))
        touch = spawn.clone()
        touch[:, 9:97].view(1, 4, 22)[0, 1, 21] = 1.0
        gameplay(physical_context(touch, spawn, n_cars=4))
        th.testing.assert_close(capture(None)["reward_plus_kickoff"],
                                th.tensor([25., 25., -25., -25.]))
        opponent_touch = spawn.clone()
        opponent_touch[:, 9:97].view(1, 4, 22)[0, 2, 21] = 1.0
        gameplay(physical_context(opponent_touch, touch, n_cars=4))
        th.testing.assert_close(capture(None)["reward_plus_kickoff"], th.zeros(4))

        gameplay(physical_context(spawn, spawn, n_cars=4,
                                  done=th.ones(1, dtype=th.bool)))
        gameplay(physical_context(touch, spawn, n_cars=4, episode_ticks=4))
        th.testing.assert_close(capture(None)["reward_plus_kickoff"],
                                th.tensor([25., 25., -25., -25.]))

        gameplay(physical_context(spawn, spawn, n_cars=4,
                                  done=th.ones(1, dtype=th.bool)))
        double_touch = touch.clone()
        double_touch[:, 9:97].view(1, 4, 22)[0, 0, 21] = 1.0
        gameplay(physical_context(double_touch, spawn, n_cars=4, episode_ticks=4))
        th.testing.assert_close(capture(None)["reward_plus_kickoff"],
                                th.tensor([25., 25., -25., -25.]))

        gameplay(physical_context(spawn, spawn, n_cars=4,
                                  done=th.ones(1, dtype=th.bool)))
        gameplay(physical_context(spawn, spawn, n_cars=4, episode_ticks=4))
        gameplay(physical_context(touch, spawn, n_cars=4, episode_ticks=500))
        th.testing.assert_close(capture(None)["reward_plus_kickoff"], th.zeros(4))

        gameplay(physical_context(spawn, spawn, n_cars=4,
                                  done=th.ones(1, dtype=th.bool)))
        replay_start = spawn.clone()
        replay_start[:, 9:97].view(1, 4, 22)[:, :2, 1] = -120.0
        replay_start[:, 9:97].view(1, 4, 22)[:, 2:, 1] = 120.0
        gameplay(physical_context(replay_start, replay_start, n_cars=4,
                                  episode_ticks=4))
        gameplay(physical_context(touch, replay_start, n_cars=4))
        th.testing.assert_close(capture(None)["reward_plus_kickoff"], th.zeros(4))

    def test_reward_plus_adds_opt_in_events_without_doubling_goals_or_resets(self):
        context = touch_context()
        gameplay = GameplayDiagnostics(4, th.device("cpu"), 900, reward_plus=True)
        goal_only = gameplay(context)
        events = AdvancedTouchCapture(gameplay)(None)
        self.assertTrue(all(name in events for name in REWARD_PLUS_COMPONENTS))
        batch = TensorBatch({
            "observation": th.zeros(1, 8, 51),
            "scene_window": th.zeros(1, 8, 2, 51),
            "scene_window_valid": th.zeros(1, 8, dtype=th.bool),
            "reward": goal_only.reshape(1, 8),
            **{name: value.unsqueeze(0) for name, value in events.items()},
        })
        args = dict(noise_std=0, trajectory_length=2, goal_reward_weight=3,
                    aerial_touch_reward_weight=.5, flip_reset_reward_weight=1)
        baseline = SceneDiscriminatorReward(ZeroDiscriminator(), **args)(
            batch, PrepareContext(),
        )
        with_plus = SceneDiscriminatorReward(
            ZeroDiscriminator(), **args, reward_plus=True,
        )(batch, PrepareContext())
        additional = sum((events[name] for name in REWARD_PLUS_COMPONENTS), th.zeros(8))
        th.testing.assert_close(with_plus["training_reward"],
                                baseline["training_reward"] + additional.unsqueeze(0))
        th.testing.assert_close(with_plus["reward_plus_reward"], additional.unsqueeze(0))
        th.testing.assert_close(with_plus["goal_reward"], goal_only.reshape(1, 8) * 3)
        th.testing.assert_close(with_plus["flip_reset_reward"],
                                events["flip_reset_event"].unsqueeze(0))
        th.testing.assert_close(with_plus["training_reward"].view(4, 2).sum(-1), th.zeros(4))
        with self.assertRaisesRegex(ValueError, "missing reward_plus_speed"):
            SceneDiscriminatorReward(ZeroDiscriminator(), **args, reward_plus=True)(
                TensorBatch({name: value for name, value in batch.items()
                             if name not in REWARD_PLUS_COMPONENTS}), PrepareContext(),
            )

    def test_physical_reward_scale_multiplies_all_mechanics_but_not_imitation_or_goals(self):
        windows = th.zeros(1, 2, 2, 51)
        windows[0, 1, -1, 0] = math.log(2)
        components = {name: th.zeros(1, 2) for name in REWARD_PLUS_COMPONENTS}
        components["reward_plus_speed"] = th.tensor([[.25, -.25]])
        batch = TensorBatch({
            "observation": th.zeros(1, 2, 51),
            "scene_window": windows,
            "scene_window_valid": th.ones(1, 2, dtype=th.bool),
            "reward": th.tensor([[0., 2.]]),
            "aerial_touch_score": th.tensor([[1., -.5]]),
            "flip_reset_event": th.tensor([[0., 1.]]),
            **components,
        })
        options = dict(noise_std=0, trajectory_length=2, exp_log_odds_reward=True,
                       goal_reward_weight=3, aerial_touch_reward_weight=.5,
                       flip_reset_reward_weight=1, reward_plus=True)
        baseline = SceneDiscriminatorReward(PositionDiscriminator(), **options)(
            batch, PrepareContext(),
        )
        for scale in (0., 3.):
            with self.subTest(scale=scale):
                transform = SceneDiscriminatorReward(
                    PositionDiscriminator(), **options, physical_reward_scale=scale,
                )
                result = transform(batch, PrepareContext())
                for name in ("imitation_reward", "goal_reward"):
                    th.testing.assert_close(result[name], baseline[name])
                for name in ("aerial_touch_reward", "flip_reset_reward", "reward_plus_reward"):
                    th.testing.assert_close(result[name], baseline[name] * scale)
                th.testing.assert_close(
                    result["training_reward"],
                    baseline["imitation_reward"] + baseline["goal_reward"]
                    + scale * (baseline["aerial_touch_reward"]
                               + baseline["flip_reset_reward"] + baseline["reward_plus_reward"]),
                )
                scales = transform.diagnostic_metrics()["RewardSpec"]
                self.assertAlmostEqual(scales["rms/imitation"], math.sqrt(.625), places=5)
                self.assertAlmostEqual(scales["rms/goal"], math.sqrt(18), places=5)
                self.assertAlmostEqual(scales["rms/physical"], scale * math.sqrt(.40625),
                                       places=5)
                self.assertAlmostEqual(scales["rms/reward_plus"], scale * .25, places=5)
                self.assertAlmostEqual(scales["rms/reward_plus_speed"], scale * .25,
                                       places=5)
                self.assertEqual(transform.diagnostic_metrics(), {})
        learner_only = batch.with_fields(learner_mask=th.tensor([[False, True]]))
        transform = SceneDiscriminatorReward(
            PositionDiscriminator(), **options, physical_reward_scale=3,
        )
        transform(learner_only, PrepareContext())
        scales = transform.diagnostic_metrics()["RewardSpec"]
        self.assertAlmostEqual(scales["rms/imitation"], .5)
        self.assertAlmostEqual(scales["rms/physical"], 1.5)
        self.assertAlmostEqual(scales["rms/reward_plus_speed"], .75)
        for invalid in (-1., float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "physical reward scale must be finite and non-negative",
            ):
                SceneDiscriminatorReward(
                    PositionDiscriminator(), **options, physical_reward_scale=invalid,
                )

    def test_reward_plus_rms_separates_canceling_components_on_learner_actions(self):
        components = {name: th.zeros(1, 4) for name in REWARD_PLUS_COMPONENTS}
        components["reward_plus_speed"] = th.tensor([[.5, -100., -.5, 100.]])
        components["reward_plus_wall_touch"] = -components["reward_plus_speed"]
        batch = TensorBatch({
            "observation": th.zeros(1, 4, 51),
            "scene_window": th.zeros(1, 4, 2, 51),
            "scene_window_valid": th.ones(1, 4, dtype=th.bool),
            "learner_mask": th.tensor([[True, False, True, False]]),
            "reward": th.zeros(1, 4),
            **components,
        })
        transform = SceneDiscriminatorReward(
            ZeroDiscriminator(), noise_std=0, trajectory_length=2,
            exp_log_odds_reward=True, reward_plus=True, physical_reward_scale=2,
        )
        result = transform(batch, PrepareContext())
        th.testing.assert_close(result["reward_plus_reward"], th.zeros(1, 4))
        th.testing.assert_close(result["training_reward"], th.ones(1, 4))
        scales = transform.diagnostic_metrics()["RewardSpec"]
        self.assertEqual(scales["rms/imitation"], 1)
        self.assertEqual(scales["rms/physical"], 0)
        self.assertEqual(scales["rms/reward_plus"], 0)
        self.assertEqual(scales["rms/reward_plus_speed"], 1)
        self.assertEqual(scales["rms/reward_plus_wall_touch"], 1)
        self.assertTrue(all(scales[f"rms/{name}"] == 0 for name in REWARD_PLUS_COMPONENTS
                            if name not in ("reward_plus_speed", "reward_plus_wall_touch")))

    def test_physical_rms_cap_limits_each_bonus_and_their_sum_not_goals(self):
        components = {name: th.zeros(1, 4) for name in REWARD_PLUS_COMPONENTS}
        components["reward_plus_speed"] = th.tensor([[2., -100., -2., 100.]])
        batch = TensorBatch({
            "observation": th.zeros(1, 4, 51),
            "scene_window": th.zeros(1, 4, 2, 51),
            "scene_window_valid": th.ones(1, 4, dtype=th.bool),
            "learner_mask": th.tensor([[True, False, True, False]]),
            "reward": th.tensor([[1., 0., 0., 0.]]),
            **components,
        })
        for sign, expected_cap in ((1, .25), (-1, .5)):
            with self.subTest(sign=sign):
                opposite = components["reward_plus_speed"] * sign
                transform = SceneDiscriminatorReward(
                    ZeroDiscriminator(), noise_std=0, trajectory_length=2,
                    exp_log_odds_reward=True, reward_plus=True,
                    goal_reward_weight=75, cap_physical_rewards=True,
                )
                result = transform(batch.replace_fields(
                    reward_plus_wall_touch=opposite,
                ), PrepareContext())
                scales = transform.diagnostic_metrics()["RewardSpec"]
                self.assertEqual(scales["scale/physical_rms_cap"], expected_cap)
                self.assertEqual(scales["rms/imitation"], 1)
                self.assertEqual(scales["rms/reward_plus_speed"], 2 * expected_cap)
                self.assertEqual(scales["rms/reward_plus_wall_touch"], 2 * expected_cap)
                self.assertEqual(scales["rms/physical"],
                                 (2 + 2 * sign) * expected_cap)
                self.assertAlmostEqual(scales["rms/goal"], 75 / math.sqrt(2), places=4)
                self.assertLessEqual(max(
                    scales[f"rms/{name}"] for name in (
                        "physical", "aerial_touch", "flip_reset", *REWARD_PLUS_COMPONENTS,
                    )
                ), scales["rms/imitation"])
                th.testing.assert_close(result["goal_reward"], batch["reward"] * 75)
                th.testing.assert_close(result["training_reward"],
                                        result["imitation_reward"] + result["goal_reward"]
                                        + result["reward_plus_reward"])

        # The reward-plus subtotal must be capped even if touch and reset
        # bonuses cancel it in the combined physical reward.
        transform = SceneDiscriminatorReward(
            ZeroDiscriminator(), noise_std=0, trajectory_length=2,
            exp_log_odds_reward=True, reward_plus=True,
            aerial_touch_reward_weight=1, flip_reset_reward_weight=1,
            cap_physical_rewards=True,
        )
        transform(batch.with_fields(
            aerial_touch_score=-components["reward_plus_speed"],
            flip_reset_event=-components["reward_plus_speed"],
        ).replace_fields(
            reward_plus_wall_touch=components["reward_plus_speed"],
        ), PrepareContext())
        subtotals = transform.diagnostic_metrics()["RewardSpec"]
        self.assertEqual(subtotals["scale/physical_rms_cap"], .25)
        self.assertEqual(subtotals["rms/reward_plus"], 1)
        self.assertEqual(subtotals["rms/physical"], 0)

        # A rollout with no valid imitation window cannot pay any physical bonus.
        empty = batch.replace_fields(scene_window_valid=th.zeros(1, 4, dtype=th.bool))
        transform = SceneDiscriminatorReward(
            ZeroDiscriminator(), noise_std=0, trajectory_length=2,
            exp_log_odds_reward=True, reward_plus=True, goal_reward_weight=75,
            cap_physical_rewards=True,
        )
        result = transform(empty, PrepareContext())
        th.testing.assert_close(result["training_reward"], empty["reward"] * 75)
        self.assertEqual(transform.diagnostic_metrics()["RewardSpec"]["scale/physical_rms_cap"],
                         0)

    def test_reward_plus_diagnostics_report_components_before_zero_sum(self):
        current = physical_state(2)
        previous = current.clone()
        current[0, 9 + 4] = 400.0
        gameplay = GameplayDiagnostics(2, th.device("cpu"), 900, reward_plus=True)
        gameplay(physical_context(current, previous))
        self.assertEqual(gameplay.last_reward_plus["reward_plus_speed"].sum().item(), 0)

        report = gameplay.diagnostic_metrics()
        components = report["RewardSpec"]
        speed = (400.0 / 2300.0) * 37.5
        self.assertAlmostEqual(components["component/speed"], speed / 4)
        self.assertAlmostEqual(components["aggregate/reward_plus"], speed / 4)
        self.assertAlmostEqual(components["scale/reward_plus"], speed / 2 ** 0.5)
        for name in REWARD_PLUS_COMPONENTS[1:]:
            self.assertEqual(components[f"component/{name.removeprefix('reward_plus_')}"], 0)
        self.assertNotIn("RewardSpec", gameplay.diagnostic_metrics())

        gameplay(physical_context(current, current))
        quiet = gameplay.diagnostic_metrics()["RewardSpec"]
        self.assertTrue(all(value == 0 for value in quiet.values()))

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
            "--physical-reward-scale", "3.0",
            "--exp-log-odds-reward", "--recency-replay",
            "--reward-plus",
            "--history-reservoir-fraction", "0.3",
        ]):
            args, _ = parse_args()
        self.assertEqual(args.aerial_touch_reward_weight, 0.25)
        self.assertEqual(args.flip_reset_reward_weight, 2.0)
        self.assertEqual(args.physical_reward_scale, 3.0)
        self.assertEqual(
            build_ppo_transforms(args, PositionDiscriminator())[0].physical_reward_scale, 3.0,
        )
        self.assertTrue(args.exp_log_odds_reward)
        self.assertTrue(args.recency_replay)
        self.assertTrue(args.reward_plus)
        self.assertEqual(args.history_reservoir_fraction, 0.3)
        with patch.object(sys, "argv", [
            "gaifo.py", "--replay-dir", "parsed_replays",
        ]):
            defaults, _ = parse_args()
        self.assertFalse(defaults.exp_log_odds_reward)
        self.assertFalse(defaults.recency_replay)
        self.assertFalse(defaults.reward_plus)
        self.assertTrue(defaults.cap_physical_rewards)
        self.assertEqual(defaults.physical_reward_scale, 1.0)
        self.assertEqual(defaults.goal_reward_weight, 75.0)
        self.assertEqual(defaults.aerial_touch_reward_weight, 12.5)
        self.assertEqual(defaults.flip_reset_reward_weight, 12.5)
        self.assertEqual(defaults.reward_plus_weights, asdict(RewardPlusWeights()))
        transform = build_ppo_transforms(defaults, PositionDiscriminator())[0]
        self.assertEqual((transform.goal_reward_weight, transform.aerial_touch_reward_weight,
                          transform.flip_reset_reward_weight), (75.0, 12.5, 12.5))
        self.assertTrue(transform.cap_physical_rewards)
        old = asdict(RewardPlusWeights.legacy())
        self.assertEqual(defaults.reward_plus_weights, {
            **{name: weight * 50 for name, weight in old.items() if name != "boost_gain"},
            "boost_gain": 12.5,
        })


if __name__ == "__main__":
    unittest.main()
