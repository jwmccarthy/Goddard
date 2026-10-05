"""Opt-in CUDA/CARL smoke for replay-backed ASE GAIFO, resume and viewer."""

import copy
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
from evaluate_skill_conditioning import evaluate_skill_conditioning
from gaifo import (
    BLUE_START, GAIFO_ASE_ARCHITECTURE, ORANGE_START,
    POSITION_SCALE, load_resume_checkpoint, main,
)
from gaifo_ase import (
    SkillDiscoveryReward, SkillEncoder, SkillEncoderUpdate, SkillGRUEncoder,
    SkillSequenceEncoder, SkillStreamContext, stable_categorical_kl,
)
from jarl.data import TensorBatch
from jarl.store.rollout import Rollout
from jarl.transform import PrepareContext
from torch.distributions import Categorical
from watch_checkpoints import load_match


@unittest.skipUnless(
    os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CUDA/CARL integration smoke",
)
class ASEGAIFOGpuSmokeTests(unittest.TestCase):
    def test_cuda_streaming_gru_scores_and_trains_with_owned_touches(self):
        length, n_envs, dim = 64, 512, 16
        encoder = SkillGRUEncoder(dim, 64, sequence_length=32).cuda()
        optimizer = th.optim.Adam(encoder.parameters(), lr=3e-4)
        stream = SkillStreamContext()
        skill = th.nn.functional.normalize(
            th.randn(2, n_envs, dim, device="cuda:0"), dim=-1,
        ).repeat_interleave(32, dim=0)
        scene = th.randn(length, n_envs, 51, device="cuda:0") * 0.01
        observation = th.cat((scene, skill), dim=-1)
        next_observation = observation.clone()
        next_observation[..., 9] += 0.01
        done = th.zeros(length, n_envs, dtype=th.bool, device="cuda:0")
        touches = done.clone()
        touches[16, ::8] = True
        opponent = done.clone()
        opponent[20, ::16] = True
        batch = TensorBatch({
            "observation": observation, "next_obs": next_observation,
            "training_reward": th.zeros_like(done, dtype=th.float32),
            "learner_mask": done.clone(), "terminated": done,
            "truncated": done.clone(), "ego_ball_touch": touches,
            "opponent_ball_touch": opponent,
        })
        reward = SkillDiscoveryReward(
            encoder, 0.5, stream_context=stream,
        )(batch, PrepareContext())
        self.assertTrue(th.isfinite(reward["skill_reward"]).all())
        self.assertTrue(reward["learner_mask"].all())
        self.assertTrue(th.equal(stream.state.age, th.full_like(stream.state.age, 32)))
        update = SkillEncoderUpdate(
            encoder, optimizer, batch_size=4_096, steps=1,
            max_grad_norm=0.5, seed=0, device=th.device("cuda:0"),
            stream_context=stream,
        )
        _, metrics = update.run(Rollout(batch))
        self.assertTrue(all(np.isfinite(x) for x in metrics["Skill"].values()))
        self.assertGreater(metrics["Skill"]["ball_credit"], 0)
        self.assertTrue(th.isfinite(encoder.car_head.weight).all())
        self.assertTrue(th.isfinite(encoder.ball_head.weight).all())

    def test_cuda_skill_sequences_score_and_train_in_bounded_chunks(self):
        length, n_envs, dim = 64, 512, 16
        encoder = SkillSequenceEncoder(dim, 64, sequence_length=32).cuda()
        optimizer = th.optim.Adam(encoder.parameters(), lr=3e-4)
        skill = th.nn.functional.normalize(
            th.randn(2, n_envs, dim, device="cuda:0"), dim=-1,
        ).repeat_interleave(32, dim=0)
        scene = th.randn(length, n_envs, 51, device="cuda:0") * 0.01
        observation = th.cat((scene, skill), dim=-1)
        next_observation = observation.clone()
        next_observation[..., 9] += 0.01
        done = th.zeros(length, n_envs, dtype=th.bool, device="cuda:0")
        batch = TensorBatch({
            "observation": observation, "next_obs": next_observation,
            "training_reward": th.zeros_like(done, dtype=th.float32),
            "learner_mask": done.clone(),
            "terminated": done, "truncated": done.clone(),
            "ego_ball_touch": done.clone(), "opponent_ball_touch": done.clone(),
        })
        reward = SkillDiscoveryReward(encoder, 0.5, batch_size=4_096)(
            batch, PrepareContext(),
        )
        self.assertTrue(th.isfinite(reward["skill_reward"]).all())
        self.assertTrue(reward["learner_mask"].all())
        update = SkillEncoderUpdate(
            encoder, optimizer, batch_size=4_096, steps=1,
            max_grad_norm=0.5, seed=0, device=th.device("cuda:0"),
        )
        _, metrics = update.run(Rollout(batch))
        self.assertTrue(all(np.isfinite(x) for x in metrics["Skill"].values()))
        self.assertTrue(th.isfinite(encoder.sequence_head.weight).all())

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
            rows = np.zeros((192, 161), dtype=np.float32)
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
            self.assertEqual(saved["config"]["ase_sequence_length"], 2)
            self.assertEqual(saved["config"]["ase_encoder_type"], "gru")
            self.assertTrue(any(
                key.startswith("car_gru.") for key in saved["skill_encoder"]
            ))
            self.assertTrue(saved["config"]["factorize"])
            self.assertFalse(saved["config"]["recurrent_global"])
            self.assertIn("skill_encoder_optimizer", saved)
            self.assertTrue(saved["skill_encoder_optimizer"]["state"])
            self.assertIn("ASE diversity loss", output.getvalue())
            self.assertIn("ASE reward", output.getvalue())
            diagnostic = evaluate_skill_conditioning(
                first_path, replays.parent, skills=3, starts=1, steps=4,
                reset_state_limit=8,
            )
            self.assertTrue(all(np.isfinite(value) for value in diagnostic.values()))
            self.assertGreaterEqual(diagnostic["first_action_disagreement"], 0)

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

            legacy = copy.deepcopy(saved)
            legacy["config"].pop("ase_encoder_type")
            base_state = {
                key: value for key, value in legacy["skill_encoder"].items()
                if key.startswith(("car_model.", "ball_model."))
            }
            old_encoder = SkillSequenceEncoder(4, 16, 2)
            old_encoder.load_state_dict(base_state, strict=False)
            legacy["skill_encoder"] = old_encoder.state_dict()
            legacy["skill_encoder_optimizer"] = th.optim.Adam(
                old_encoder.parameters(), lr=3e-4,
            ).state_dict()
            legacy_path = root / "legacy_ase.pt"
            th.save(legacy, legacy_path)
            upgraded_dir = root / "upgraded_checkpoints"
            with patch.object(sys, "argv", [
                "gaifo.py", "--resume-checkpoint", str(legacy_path),
                "--timesteps", "128", "--checkpoint-dir", str(upgraded_dir),
            ]), redirect_stdout(io.StringIO()):
                main()
            upgraded = load_resume_checkpoint(
                next(upgraded_dir.rglob("gaifo_000000000128.pt"))
            )
            self.assertEqual(upgraded["config"]["ase_sequence_length"], 2)
            self.assertEqual(upgraded["config"]["ase_encoder_type"], "gru")
            self.assertIn("car_gru.weight_ih", upgraded["skill_encoder"])


if __name__ == "__main__":
    unittest.main()
