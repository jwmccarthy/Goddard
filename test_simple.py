import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch as th

from simple import GoalOnlyReward, SimpleCheckpoints, parse_args, validate_args


def make_reward_context(score_delta: int):
    return SimpleNamespace(
        events=SimpleNamespace(score_delta=th.tensor([score_delta])),
        current=SimpleNamespace(team_sign=th.tensor([1.0, -1.0])),
    )


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
            reward(make_reward_context(score_delta=0)),
            th.zeros((1, 2)),
        )

    def test_cli_requires_replays_but_has_no_distillation_input(self):
        with patch("sys.argv", ["simple.py", "--replay-dir", "replays"]):
            args = parse_args()

        self.assertFalse(hasattr(args, "distill_checkpoint"))
        self.assertEqual(args.replay_dir, Path("replays"))
        self.assertEqual(args.replay_reset_fraction, 0.8)
        self.assertEqual(args.checkpoint_dir, Path("checkpoints/simple"))
        self.assertEqual(args.timesteps, 10_000_000_000)
        self.assertEqual(args.gamma, 0.9997)
        self.assertEqual(args.gae_lambda, 0.999)

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
