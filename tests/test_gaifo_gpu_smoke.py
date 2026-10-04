"""Opt-in CARL/GPU smoke: GODDARD_GPU_SMOKE=1 python -m unittest ..."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch as th

from gaifo import (
    BLUE_START, ConfidentExpertResetTransform, ORANGE_START, POSITION_SCALE,
    load_resume_checkpoint, main,
)


@unittest.skipUnless(
    os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CUDA/CARL integration smoke",
)
class GAIFOGpuSmokeTests(unittest.TestCase):
    def _short_window_training(
        self, factorize: bool, hard_positive_mining: bool = False,
        exp_log_odds_reward: bool = False, recency_replay: bool = False,
    ):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replays = root / "replays"
            replays.mkdir()
            rows = np.zeros((48, 161), np.float32)
            rows[:, 2] = 91.25 / POSITION_SCALE[2]
            for car, y in ((BLUE_START, -1_200), (ORANGE_START, 1_200)):
                rows[:, car + 1] = y / POSITION_SCALE[1]
                rows[:, car + 2] = 17 / POSITION_SCALE[2]
                rows[:, car + 9] = 1
                rows[:, car + 14] = 1
                rows[:, car + 16] = 1
            rows[:, 137] = 1
            np.save(replays / "replay.npy", rows)
            np.savez_compressed(
                replays / "replay.unsafe-starts.npz",
                unsafe=np.zeros(len(rows), dtype=bool),
                pre_goal=np.zeros(len(rows), dtype=bool), frame_skip=4,
            )
            flags = [
                "gaifo.py", "--replay-dir", str(replays),
                "--replay-reset-fraction", "1",
                "--n-sim", "2",
                "--rollout", "8", "--trajectory-length", "8",
                "--timesteps", "64", "--ppo-batch", "8", "--ppo-epochs", "1",
                "--policy-hidden", "16", "--critic-hidden", "16",
                "--discriminator-hidden", "16", "--frame-embedding", "8",
                "--temporal-hidden", "8",
                "--discriminator-batch", "4" if recency_replay else "2",
                "--discriminator-microbatch", "2",
                "--discriminator-heldout-size", "4",
                "--discriminator-accuracy-target", "1.0",
                "--discriminator-update-interval", "1",
                "--history-capacity", "8", "--history-add-size", "4",
                "--log-dir", str(root / "runs"),
                "--checkpoint-dir", str(root / "checkpoints"),
            ]
            if factorize:
                flags.append("--factorize")
            if hard_positive_mining:
                flags.extend(("--hard-positive-mining", "--no-touch-timeout", "0.3"))
            if exp_log_odds_reward:
                flags.append("--exp-log-odds-reward")
            if recency_replay:
                flags.append("--recency-replay")
            output = io.StringIO()
            ready_resets = []
            original_reset = ConfidentExpertResetTransform.__call__

            def record_reset(miner, sample, context):
                ready_resets.append(miner.ready)
                return original_reset(miner, sample, context)

            with (patch.object(sys, "argv", flags), redirect_stdout(output),
                  patch.object(ConfidentExpertResetTransform, "__call__", record_reset)):
                main()

            checkpoints = list((root / "checkpoints").rglob("gaifo_*.pt"))
            self.assertGreaterEqual(len(checkpoints), 2)
            return load_resume_checkpoint(max(checkpoints)), output.getvalue(), ready_resets

    def test_short_window_training_and_terminal_metrics(self):
        saved, output, _ = self._short_window_training(False)
        self.assertEqual(saved["step"], 64)
        self.assertFalse(saved["config"]["factorize"])
        self.assertNotIn("long_discriminator", saved)
        self.assertNotIn("long_discriminator_optimizer", saved)
        self.assertIn("D heldout accuracy", output)
        self.assertNotIn("short D", output)
        self.assertNotIn("long D", output)

    def test_factorized_short_window_training(self):
        saved, output, _ = self._short_window_training(True)
        self.assertEqual(saved["step"], 64)
        self.assertTrue(saved["config"]["factorize"])
        self.assertTrue(any(key.startswith("car_encoder.")
                            for key in saved["discriminator"]))
        self.assertTrue(any(key.startswith("ball_encoder.")
                            for key in saved["discriminator"]))
        self.assertIn("D car accuracy", output)
        self.assertIn("D ball accuracy", output)

    def test_hard_positive_mining_in_unified_and_factorized_modes(self):
        for factorize in (False, True):
            with self.subTest(factorize=factorize):
                saved, output, resets = self._short_window_training(
                    factorize, hard_positive_mining=True,
                )
                self.assertEqual(saved["step"], 64)
                self.assertTrue(saved["config"]["hard_positive_mining"])
                self.assertEqual(saved["config"]["factorize"], factorize)
                self.assertIn(False, resets)
                self.assertIn(True, resets)
                self.assertIn("D mined reset frac", output)

    def test_exp_log_odds_and_recency_replay_in_both_modes(self):
        for factorize in (False, True):
            with self.subTest(factorize=factorize):
                saved, output, _ = self._short_window_training(
                    factorize, exp_log_odds_reward=True, recency_replay=True,
                )
                self.assertEqual(saved["step"], 64)
                self.assertTrue(saved["config"]["exp_log_odds_reward"])
                self.assertTrue(saved["config"]["recency_replay"])
                self.assertIn("D heldout accuracy", output)


if __name__ == "__main__":
    unittest.main()
