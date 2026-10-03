"""Opt-in CUDA/CARL smoke for replay-backed ASE GAIFO, resume and viewer."""

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
import torch.nn.functional as F

from carl.gymnasium import CARLTorchVectorEnv
from gaifo import (
    BLUE_START, GAIFO_ASE_ARCHITECTURE, ORANGE_START,
    POSITION_SCALE, load_resume_checkpoint, main,
)
from gaifo_ase import stable_categorical_kl
from torch.distributions import Categorical
from watch_checkpoints import load_match


@unittest.skipUnless(
    os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CUDA/CARL integration smoke",
)
class ASEGAIFOGpuSmokeTests(unittest.TestCase):
    def test_cuda_ase_kl_with_confident_masked_actions_at_training_batch_size(self):
        mask_logit = th.finfo(th.float32).min
        logits = th.zeros(16_384, 4, 3, device="cuda:0", requires_grad=True)
        other = th.zeros_like(logits)
        other[..., 1] = -120.0
        other[..., 2] = mask_logit
        masked = logits.masked_fill(
            th.tensor([False, False, True], device="cuda:0"), mask_logit,
        )
        action_kl = stable_categorical_kl(
            Categorical(logits=masked), Categorical(logits=other),
        ).sum(dim=-1)
        loss = 2 * F.huber_loss(action_kl / 0.5, th.ones_like(action_kl), delta=4.0)
        self.assertTrue(th.isfinite(loss))
        loss.backward()
        self.assertTrue(th.isfinite(logits.grad).all())

    def test_1v1_training_resume_and_viewer(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replays = root / "parsed_replays" / "pro_1v1_fs4"
            replays.mkdir(parents=True)
            rows = np.zeros((48, 161), dtype=np.float32)
            rows[:, 2] = 91.25 / POSITION_SCALE[2]
            for car, y in ((BLUE_START, -1_200), (ORANGE_START, 1_200)):
                rows[:, car + 1] = y / POSITION_SCALE[1]
                rows[:, car + 2] = 17 / POSITION_SCALE[2]
                rows[:, car + 9] = 1
                rows[:, car + 14] = 1
            np.save(replays / "replay.npy", rows)

            flags = [
                "gaifo.py", "--replay-dir", str(replays.parent),
                "--ase-diversity", "--factorize", "--n-sim", "2",
                "--replay-reset-fraction", "1", "--rollout", "8",
                "--trajectory-length", "8", "--timesteps", "64",
                "--ppo-batch", "8", "--ppo-epochs", "1",
                "--policy-hidden", "16", "--critic-hidden", "16",
                "--discriminator-hidden", "16", "--frame-embedding", "8",
                "--temporal-hidden", "8", "--discriminator-batch", "2",
                "--discriminator-microbatch", "2",
                "--discriminator-heldout-size", "4", "--history-capacity", "8",
                "--history-add-size", "4", "--ase-skill-dim", "4",
                "--ase-skill-steps", "2", "--ase-encoder-hidden", "16",
                "--ase-encoder-batch", "8", "--ase-encoder-steps", "2",
                "--ase-diversity-batch", "4", "--checkpoint-interval", "64",
                "--log-dir", str(root / "runs"),
                "--checkpoint-dir", str(root / "checkpoints"),
            ]
            output = io.StringIO()
            with patch.object(sys, "argv", flags), redirect_stdout(output):
                main()

            first_path = max((root / "checkpoints").rglob("gaifo_*.pt"))
            saved = load_resume_checkpoint(first_path)
            self.assertEqual(saved["step"], 64)
            self.assertEqual(saved["config"]["architecture"], GAIFO_ASE_ARCHITECTURE)
            self.assertTrue(saved["config"]["factorize"])
            self.assertIn("skill_encoder_optimizer", saved)
            self.assertTrue(saved["skill_encoder_optimizer"]["state"])
            self.assertIn("ASE diversity loss", output.getvalue())
            self.assertIn("ASE reward", output.getvalue())

            with patch.object(sys, "argv", [
                "gaifo.py", "--resume-checkpoint", str(first_path),
                "--timesteps", "128",
            ]), redirect_stdout(io.StringIO()):
                main()
            final_path = next((root / "checkpoints").rglob("gaifo_000000000128.pt"))
            resumed = load_resume_checkpoint(final_path)
            self.assertEqual(resumed["step"], 128)
            restored = load_resume_checkpoint(
                final_path.parent / "gaifo_000000000064.pt"
            )
            for name in ("policy", "critic", "discriminator", "skill_encoder"):
                for key, weight in saved[name].items():
                    th.testing.assert_close(weight, restored[name][key])
            th.testing.assert_close(saved["skill_rng_state"], restored["skill_rng_state"])

            base = CARLTorchVectorEnv(
                n_sim=1, n_blue=1, n_orange=1, frameskip=4,
                normalize=True, discrete_actions=True,
            )
            try:
                env, blue, orange = load_match(first_path, final_path, base, 4, None)
                self.assertFalse(th.allclose(blue.skill, orange.skill))
                observation = env.reset()
                with th.inference_mode():
                    action = th.cat((
                        blue.act(observation[:1], deterministic=True).action,
                        orange.act(observation[1:], deterministic=True).action,
                    ))
                self.assertEqual(env.step(action)[0].shape, observation.shape)
            finally:
                base.close()


if __name__ == "__main__":
    unittest.main()
