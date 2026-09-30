"""Opt-in full CARL/GPU smoke: GODDARD_GPU_SMOKE=1 python -m unittest ..."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch as th

from smp import load_resume_checkpoint, main


@unittest.skipUnless(
    os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CUDA/CARL integration smoke",
)
class SMPGpuSmokeTests(unittest.TestCase):
    def test_eight_frame_prior_and_two_ppo_updates(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replays = root / "replays"
            replays.mkdir()
            np.save(replays / "replay.npy", np.zeros((48, 161), np.float32))
            flags = [
                "smp.py", "--replay-dir", str(replays),
                "--replay-reset-fraction", "0", "--n-sim", "2",
                "--rollout", "8", "--trajectory-length", "8",
                "--timesteps", "64", "--ppo-batch", "8", "--ppo-epochs", "1",
                "--policy-hidden", "16", "--critic-hidden", "16",
                "--score-hidden", "8", "--diffusion-steps", "16",
                "--score-timesteps", "2", "5", "10",
                "--prior-updates", "2", "--prior-batch", "4",
                "--prior-microbatch", "2", "--prior-eval-interval", "1",
                "--prior-heldout-size", "4", "--prior-calibration-size", "4",
                "--agent-score-updates", "1", "--agent-score-batch", "4",
                "--agent-score-microbatch", "2", "--history-add-size", "4",
                "--history-capacity", "8", "--reward-batch", "2",
                "--log-dir", str(root / "runs"),
                "--checkpoint-dir", str(root / "checkpoints"),
            ]
            with patch.object(sys, "argv", flags):
                main()
            prior_files = list((root / "checkpoints" / "priors").glob("prior_*.pt"))
            self.assertEqual(len(prior_files), 1)
            checkpoints = list((root / "checkpoints").rglob("smp_*.pt"))
            self.assertGreaterEqual(len(checkpoints), 2)
            final = max(checkpoints)
            payload = load_resume_checkpoint(final)
            self.assertEqual(payload["step"], 64)
            self.assertTrue(payload["agent_ready"])
            self.assertEqual(payload["agent_rollouts"], 2)

            # Policy training can be resumed after the reservoir is rebuilt.
            with patch.object(sys, "argv", [
                "smp.py", "--resume-checkpoint", str(final), "--timesteps", "96",
            ]):
                main()
            resumed = max((root / "checkpoints").rglob("smp_*.pt"))
            self.assertEqual(load_resume_checkpoint(resumed)["step"], 96)

            # A frozen prior also works without access to the original replays.
            reuse_flags = [
                "smp.py", "--prior-checkpoint", str(prior_files[0]),
                "--replay-reset-fraction", "0", "--n-sim", "2",
                "--rollout", "8", "--trajectory-length", "8", "--timesteps", "32",
                "--score-hidden", "8", "--diffusion-steps", "16",
                "--score-timesteps", "2", "5", "10",
                "--ppo-batch", "8", "--ppo-epochs", "1",
                "--policy-hidden", "16", "--critic-hidden", "16",
                "--agent-score-updates", "1", "--agent-score-batch", "4",
                "--agent-score-microbatch", "2", "--reward-batch", "2",
                "--log-dir", str(root / "reuse-runs"),
                "--checkpoint-dir", str(root / "reuse-checkpoints"),
            ]
            with patch.object(sys, "argv", reuse_flags):
                main()
            reused = max((root / "reuse-checkpoints").rglob("smp_*.pt"))
            self.assertEqual(load_resume_checkpoint(reused)["step"], 32)


if __name__ == "__main__":
    unittest.main()
