import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch as th

from gaifo import (
    build_entropy_scheduler, build_training_scheduler, parse_args,
    restore_training_checkpoint,
)
from jarl.learn import PPOConfig


class GAIFOEntropyScheduleTests(unittest.TestCase):
    def test_default_keeps_entropy_constant(self):
        with patch.object(sys, "argv", [
            "gaifo.py", "--replay-dir", "parsed_replays",
        ]):
            args, resume = parse_args()

        self.assertIsNone(resume)
        self.assertIsNone(args.entropy_end)
        self.assertIsNone(args.ppo_lr_end)
        self.assertIsNone(args.discriminator_lr_end)
        loss = SimpleNamespace(config=PPOConfig(entropy_coef=args.entropy))
        self.assertIsNone(build_entropy_scheduler(args, loss))
        optimizers = [th.optim.Adam(th.nn.Linear(1, 1).parameters(), lr=args.ppo_lr)
                      for _ in range(3)]
        self.assertIsNone(build_training_scheduler(args, loss, *optimizers))
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

    def test_learning_rates_anneal_and_restore_from_training_clock(self):
        with patch.object(sys, "argv", [
            "gaifo.py", "--replay-dir", "parsed_replays",
            "--ppo-lr", "0.001", "--ppo-lr-end", "0.0002",
            "--discriminator-lr", "0.002", "--discriminator-lr-end", "0.0005",
            "--entropy", "0.02", "--entropy-end", "0.005",
        ]):
            args, _ = parse_args()

        modules = {name: th.nn.Linear(2, 1) for name in (
            "policy", "critic", "discriminator",
        )}
        optimizers = {
            name: th.optim.Adam(module.parameters(), lr=(
                args.discriminator_lr if name == "discriminator" else args.ppo_lr
            )) for name, module in modules.items()
        }
        loss = SimpleNamespace(config=PPOConfig(entropy_coef=args.entropy))
        scheduler = build_training_scheduler(args, loss, *optimizers.values())
        scheduler.start(100)
        scheduler.advance(40)
        self.assertAlmostEqual(loss.config.entropy_coef, 0.014)
        for name, expected in (("policy", 0.00068), ("critic", 0.00068),
                               ("discriminator", 0.0014)):
            self.assertAlmostEqual(optimizers[name].param_groups[0]["lr"], expected)
        self.assertAlmostEqual(scheduler.metrics()["Schedule"]["ppo_lr"], 0.00068)

        # Checkpoint restoration resets optimizer LRs to their configured starts;
        # the trainer then reapplies each schedule at the restored clock step.
        checkpoint = {
            "step": 40, "config": {"n_sim": 1, "rollout": 8},
            **{name: module.state_dict() for name, module in modules.items()},
            **{f"{name}_optimizer": optimizer.state_dict()
               for name, optimizer in optimizers.items()},
        }
        restored_modules = {name: th.nn.Linear(2, 1) for name in modules}
        restored_optimizers = {
            name: th.optim.Adam(module.parameters(), lr=args.ppo_lr)
            for name, module in restored_modules.items()
        }
        clock = restore_training_checkpoint(
            checkpoint, args, restored_modules, restored_optimizers,
        )
        self.assertEqual(clock.env_steps, 40)
        self.assertAlmostEqual(restored_optimizers["policy"].param_groups[0]["lr"], 0.001)
        self.assertAlmostEqual(restored_optimizers["discriminator"].param_groups[0]["lr"], 0.002)
        resumed_loss = SimpleNamespace(config=PPOConfig(entropy_coef=args.entropy))
        resumed = build_training_scheduler(
            args, resumed_loss, *restored_optimizers.values(),
        )
        resumed.start(100)
        resumed.advance(clock.env_steps)
        self.assertEqual(resumed.metrics(), scheduler.metrics())
        for name in modules:
            self.assertEqual(
                restored_optimizers[name].param_groups[0]["lr"],
                optimizers[name].param_groups[0]["lr"],
            )
        resumed.advance(125)
        self.assertAlmostEqual(restored_optimizers["policy"].param_groups[0]["lr"], 0.0002)
        self.assertAlmostEqual(restored_optimizers["critic"].param_groups[0]["lr"], 0.0002)
        self.assertAlmostEqual(restored_optimizers["discriminator"].param_groups[0]["lr"], 0.0005)


if __name__ == "__main__":
    unittest.main()
