import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gaifo import build_entropy_scheduler, parse_args
from jarl.learn import PPOConfig


class GAIFOEntropyScheduleTests(unittest.TestCase):
    def test_default_keeps_entropy_constant(self):
        with patch.object(sys, "argv", [
            "gaifo.py", "--replay-dir", "parsed_replays",
        ]):
            args, resume = parse_args()

        self.assertIsNone(resume)
        self.assertIsNone(args.entropy_end)
        loss = SimpleNamespace(config=PPOConfig(entropy_coef=args.entropy))
        self.assertIsNone(build_entropy_scheduler(args, loss))
        self.assertEqual(loss.config.entropy_coef, args.entropy)

    def test_linear_decay_and_resumed_progress(self):
        with patch.object(sys, "argv", [
            "gaifo.py", "--replay-dir", "parsed_replays",
            "--entropy", "0.02", "--entropy-end", "0.005",
        ]):
            args, _ = parse_args()

        loss = SimpleNamespace(config=PPOConfig(entropy_coef=args.entropy))
        scheduler = build_entropy_scheduler(args, loss)
        scheduler.start(100)
        scheduler.advance(40)
        self.assertAlmostEqual(loss.config.entropy_coef, 0.014)
        self.assertAlmostEqual(
            scheduler.metrics()["Schedule"]["entropy_coef"], 0.014
        )

        # A new process rebuilds the schedule from checkpoint hyperparameters
        # and advances it using the restored training clock.
        resumed_loss = SimpleNamespace(config=PPOConfig(entropy_coef=args.entropy))
        resumed_scheduler = build_entropy_scheduler(args, resumed_loss)
        resumed_scheduler.start(100)
        resumed_scheduler.advance(40)
        self.assertEqual(resumed_loss.config, loss.config)
        resumed_scheduler.advance(100)
        self.assertAlmostEqual(resumed_loss.config.entropy_coef, 0.005)
        resumed_scheduler.advance(125)
        self.assertAlmostEqual(resumed_loss.config.entropy_coef, 0.005)


if __name__ == "__main__":
    unittest.main()
