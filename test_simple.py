import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch as th

from simple import MinimalReward, SimpleCheckpoints, parse_args, validate_args


def make_reward_context(score_delta: int):
    def state():
        return SimpleNamespace(
            team_sign=th.tensor([1.0, -1.0]),
            ball_position=th.zeros((1, 3)),
            ball_velocity=th.zeros((1, 3)),
            car_position=th.zeros((1, 2, 3)),
            car_up=th.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]),
            car_ball_touches=th.zeros((1, 2), dtype=th.bool),
            car_has_flipped=th.zeros((1, 2), dtype=th.bool),
            car_has_double_jumped=th.zeros((1, 2), dtype=th.bool),
        )

    return SimpleNamespace(
        events=SimpleNamespace(
            score_delta=th.tensor([score_delta]),
            done=th.tensor([False]),
        ),
        current=state(),
        previous=state(),
    )


def component_reward(**scales):
    disabled = {
        "touch_scale": 0.0,
        "ball_velocity_scale": 0.0,
        "flip_reset_scale": 0.0,
        "ball_goal_progress_scale": 0.0,
        "player_ball_progress_scale": 0.0,
        "ball_height_progress_scale": 0.0,
        "gravity_lift_scale": 0.0,
    }
    return MinimalReward(4, **(disabled | scales))


class SimpleSelfPlayTest(unittest.TestCase):
    def test_goal_reward_is_zero_sum_plus_or_minus_ten(self):
        reward = MinimalReward(4)

        th.testing.assert_close(
            reward(make_reward_context(score_delta=1)),
            th.tensor([[10.0, -10.0]]),
        )
        th.testing.assert_close(
            reward(make_reward_context(score_delta=-1)),
            th.tensor([[-10.0, 10.0]]),
        )

    def test_non_goal_events_have_zero_reward(self):
        reward = MinimalReward(4)

        th.testing.assert_close(
            reward(make_reward_context(score_delta=0)),
            th.zeros((1, 2)),
        )

    def test_touch_and_touch_velocity_change_rewards_are_small(self):
        reward = component_reward(touch_scale=0.05, ball_velocity_scale=0.05)
        context = make_reward_context(0)
        context.current.car_ball_touches[0, 0] = True
        context.current.ball_velocity[0, 0] = 6000.0

        th.testing.assert_close(reward(context), th.tensor([[0.1, 0.0]]))

    def test_ball_goal_and_player_ball_progress_are_signed(self):
        reward = component_reward(
            ball_goal_progress_scale=1.0,
            player_ball_progress_scale=1.0,
        )
        context = make_reward_context(0)
        context.previous.ball_position[0, 1] = 0.0
        context.current.ball_position[0, 1] = 100.0
        context.previous.car_position[0, :, 0] = -1000.0
        context.current.car_position[0, 0, 0] = -500.0
        context.current.car_position[0, 1, 0] = -1500.0

        value = reward(context)

        self.assertGreater(value[0, 0].item(), 0)
        self.assertLess(value[0, 1].item(), 0)

    def test_flip_reset_rewards_the_touching_player(self):
        reward = component_reward(flip_reset_scale=1.0)
        context = make_reward_context(0)
        context.current.car_ball_touches[0, 0] = True
        context.previous.car_has_flipped[0, 0] = True
        context.current.car_position[0, 0, 2] = 400.0
        context.current.ball_position[0, 2] = 300.0

        th.testing.assert_close(reward(context), th.tensor([[1.0, 0.0]]))

    def test_ball_lift_is_credited_to_last_toucher(self):
        reward = component_reward(ball_height_progress_scale=0.1)
        touch = make_reward_context(0)
        touch.current.car_ball_touches[0, 0] = True
        reward(touch)
        lift = make_reward_context(0)
        lift.current.ball_position[0, 2] = 100.0

        value = reward(lift)

        self.assertGreater(value[0, 0].item(), 0)
        self.assertEqual(value[0, 1].item(), 0)

    def test_gravity_compensated_lift_ignores_ballistic_motion(self):
        reward = component_reward(gravity_lift_scale=0.1)
        touch = make_reward_context(0)
        touch.current.car_ball_touches[0, 0] = True
        reward(touch)
        ballistic = make_reward_context(0)
        dt = 4 / 120.0
        ballistic.previous.ball_position[0, 2] = 500.0
        ballistic.previous.ball_velocity[0, 2] = 300.0
        ballistic.current.ball_position[0, 2] = 500.0 + 300.0 * dt - 325.0 * dt**2

        th.testing.assert_close(
            reward(ballistic), th.zeros((1, 2)), atol=1e-7, rtol=0
        )

    def test_cli_requires_replays_but_has_no_distillation_input(self):
        with patch("sys.argv", ["simple.py", "--replay-dir", "replays"]):
            args = parse_args()

        self.assertFalse(hasattr(args, "distill_checkpoint"))
        self.assertEqual(args.replay_dir, Path("replays"))
        self.assertEqual(args.replay_reset_fraction, 0.7)
        self.assertEqual(args.checkpoint_dir, Path("checkpoints/simple"))
        self.assertEqual(args.timesteps, 10_000_000_000)
        self.assertIsNone(args.gamma)
        self.assertEqual(args.discount_half_life, 10.0)
        self.assertEqual(args.discount_half_life_end, 20.0)
        self.assertEqual(args.gae_lambda, 0.99)
        self.assertEqual(args.n_sim, 1024)
        self.assertEqual(args.frameskip, 8)
        self.assertEqual(args.rollout, 512)
        self.assertEqual(args.sequence_length, 16)
        self.assertEqual(args.policy_hidden, 256)
        self.assertEqual(args.critic_hidden, 256)
        self.assertEqual(args.current_fraction, 0.8)

    def test_kickoff_fraction_sets_complementary_replay_fraction(self):
        with patch(
            "sys.argv",
            [
                "simple.py",
                "--replay-dir",
                "replays",
                "--kickoff-reset-fraction",
                "0.35",
                "--no-touch-timeout-seconds",
                "12",
            ],
        ):
            args = parse_args()

        self.assertAlmostEqual(args.replay_reset_fraction, 0.65)
        self.assertEqual(args.no_touch_timeout, 12.0)

    def test_lambda_alias_is_configurable(self):
        with patch(
            "sys.argv",
            ["simple.py", "--replay-dir", "replays", "--lambda", "0.997"],
        ):
            args = parse_args()

        self.assertEqual(args.gae_lambda, 0.997)

    def test_periodic_checkpoint_waits_for_completed_rollout(self):
        checkpoint = SimpleCheckpoints.__new__(SimpleCheckpoints)
        checkpoint.next_step = 10
        checkpoint.buffer = SimpleNamespace(position=1)

        self.assertFalse(checkpoint.ready(10))
        checkpoint.buffer.position = 0
        self.assertTrue(checkpoint.ready(10))

    def test_validation_rejects_nonfinite_optimizer_settings(self):
        with patch("sys.argv", ["simple.py", "--replay-dir", "."]):
            args = parse_args()
        args.lr = float("inf")

        with self.assertRaisesRegex(ValueError, "--lr"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
