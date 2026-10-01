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

from gaifo import load_resume_checkpoint, main


@unittest.skipUnless(
    os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CUDA/CARL integration smoke",
)
class GAIFOGpuSmokeTests(unittest.TestCase):
    def _short_window_training(self, factorize: bool):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replays = root / "replays"
            replays.mkdir()
            np.save(replays / "replay.npy", np.zeros((48, 161), np.float32))
            flags = [
                "gaifo.py", "--replay-dir", str(replays),
                "--replay-reset-fraction", "0", "--n-sim", "2",
                "--rollout", "8", "--trajectory-length", "8",
                "--timesteps", "64", "--ppo-batch", "8", "--ppo-epochs", "1",
                "--policy-hidden", "16", "--critic-hidden", "16",
                "--discriminator-hidden", "16", "--frame-embedding", "8",
                "--temporal-hidden", "8", "--discriminator-batch", "2",
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
            output = io.StringIO()
            with patch.object(sys, "argv", flags), redirect_stdout(output):
                main()

            checkpoints = list((root / "checkpoints").rglob("gaifo_*.pt"))
            self.assertGreaterEqual(len(checkpoints), 2)
            return load_resume_checkpoint(max(checkpoints)), output.getvalue()

    def test_short_window_training_and_terminal_metrics(self):
        saved, output = self._short_window_training(False)
        self.assertEqual(saved["step"], 64)
        self.assertFalse(saved["config"]["factorize"])
        self.assertNotIn("long_discriminator", saved)
        self.assertNotIn("long_discriminator_optimizer", saved)
        self.assertIn("D heldout accuracy", output)
        self.assertNotIn("short D", output)
        self.assertNotIn("long D", output)

    def test_factorized_short_window_training(self):
        saved, output = self._short_window_training(True)
        self.assertEqual(saved["step"], 64)
        self.assertTrue(saved["config"]["factorize"])
        self.assertTrue(any(key.startswith("car_encoder.")
                            for key in saved["discriminator"]))
        self.assertTrue(any(key.startswith("ball_encoder.")
                            for key in saved["discriminator"]))
        self.assertIn("D car accuracy", output)
        self.assertIn("D ball accuracy", output)


if __name__ == "__main__":
    unittest.main()
