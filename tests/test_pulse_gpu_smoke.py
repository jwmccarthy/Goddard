"""Opt-in end-to-end PULSE smoke: GODDARD_GPU_SMOKE=1 python -m unittest ..."""

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

from carl.gymnasium import CARLResetState, CARLTorchVectorEnv
from distill import main as distill_main
from jarl.envs import DatasetResetSampler
from pulse import main as pulse_main
from replay_resets import (
    ReplayResetProvider, load_demonstration_reset_frames, reset_index_dataset,
)
from tracker import ExpertLookaheadEnv, main as tracker_main
from watch_checkpoints import load_match


@unittest.skipUnless(
    os.environ.get("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CUDA/CARL integration smoke",
)
class PulseGpuSmokeTests(unittest.TestCase):
    def test_tracker_distillation_and_self_play_use_typed_replay_resets(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replay_dir = root / "replays"
            replay_dir.mkdir()
            rows = np.zeros((48, 161), dtype=np.float32)
            rows[:, 2] = 91.25 / 2076
            for car, y in ((9, -1200), (30, 1200)):
                rows[:, car + 1] = y / 6000
                rows[:, car + 2] = 17 / 2076
                rows[:, car + 9] = 1
                rows[:, car + 14] = 1
                rows[:, car + 15] = 0.5
                rows[:, car + 16] = 1
            rows[:, 137] = 1
            rows[16, -5] = 1
            np.save(replay_dir / "replay.npy", rows)
            np.savez(
                replay_dir / "replay.unsafe-starts.npz",
                unsafe=np.zeros(48, dtype=bool),
                pre_goal=np.zeros(48, dtype=bool), frame_skip=4,
            )

            tracker_resets = []
            original_tracker_reset = ExpertLookaheadEnv._reset_state

            def record_tracker_reset(env, mask):
                request = original_tracker_reset(env, mask)
                tracker_resets.append(request)
                return request

            tracker_dir = root / "tracker"
            with (
                patch.object(sys, "argv", [
                    "tracker.py", "--replay-dir", str(replay_dir),
                    "--n-sim", "2", "--windows", "1", "2",
                    "--minimum-remaining-frames", "2",
                    "--minimum-tracking-frames", "1",
                    "--rollout", "2", "--ppo-batch", "4", "--ppo-epochs", "1",
                    "--timesteps", "8", "--stage-timesteps", "8",
                    "--schedule-timesteps", "8", "--checkpoint-interval", "8",
                    "--log-dir", str(root / "logs"),
                    "--checkpoint-dir", str(tracker_dir),
                ]),
                patch.object(ExpertLookaheadEnv, "_reset_state", record_tracker_reset),
                redirect_stdout(io.StringIO()),
            ):
                tracker_main()
            self.assertTrue(tracker_resets)
            self.assertTrue(all(isinstance(state, CARLResetState) for state in tracker_resets))
            tracker_path = max(tracker_dir.glob("tracker_*.pt"))

            distill_dir = root / "distill"
            with (
                patch.object(sys, "argv", [
                    "distill.py", "--replay-dir", str(replay_dir),
                    "--tracker-checkpoint", str(tracker_path),
                    "--n-sim", "2", "--windows", "1", "2",
                    "--minimum-remaining-frames", "2",
                    "--minimum-tracking-frames", "1",
                    "--latent-size", "3", "--encoder-hidden", "8",
                    "--decoder-hidden", "8", "--rollout", "2",
                    "--distill-batch", "2", "--distill-epochs", "1",
                    "--timesteps", "8", "--kl-anneal-end", "8",
                    "--checkpoint-interval", "8", "--log-dir", str(root / "logs"),
                    "--checkpoint-dir", str(distill_dir),
                ]),
                redirect_stdout(io.StringIO()),
            ):
                distill_main()
            distill_path = max(distill_dir.glob("distill_*.pt"))
            distilled = th.load(distill_path, map_location="cpu", weights_only=True)
            self.assertEqual(distilled["config"]["control_state_size"], 51)

            replay_resets = []
            original_replay_reset = ReplayResetProvider.__call__

            def record_replay_reset(provider, mask):
                request = original_replay_reset(provider, mask)
                replay_resets.append(request)
                return request

            pulse_dir = root / "pulse"
            with (
                patch.object(sys, "argv", [
                    "pulse.py", "--replay-dir", str(replay_dir),
                    "--distill-checkpoint", str(distill_path),
                    "--n-sim", "2", "--rollout", "2", "--ppo-batch", "4",
                    "--ppo-epochs", "1", "--feature-size", "8",
                    "--policy-hidden", "8", "--critic-hidden", "8",
                    "--replay-reset-fraction", "1", "--reset-state-limit", "16",
                    "--no-bf16", "--timesteps", "8", "--checkpoint-interval", "8",
                    "--log-dir", str(root / "logs"),
                    "--checkpoint-dir", str(pulse_dir),
                ]),
                patch.object(ReplayResetProvider, "__call__", record_replay_reset),
                redirect_stdout(io.StringIO()),
            ):
                pulse_main()
            self.assertTrue(replay_resets)
            self.assertTrue(all(isinstance(state, CARLResetState) for state in replay_resets))
            saved = max(pulse_dir.rglob("pulse_*.pt"))
            payload = th.load(saved, map_location="cpu", weights_only=True)
            self.assertGreaterEqual(payload["step"], 8)
            self.assertEqual(payload["config"]["architecture"], "pulse-latent-mlp-v1")
            self.assertTrue((saved.parent / payload["pulse_artifact"]).is_file())

            with (
                patch.object(sys, "argv", [
                    "pulse.py", "--resume-checkpoint", str(saved),
                    "--timesteps", "16", "--run-name", "standard-resume",
                ]),
                redirect_stdout(io.StringIO()),
            ):
                pulse_main()
            standard_resume = pulse_dir / "standard-resume"
            restored = th.load(
                standard_resume / f"pulse_{payload['step']:012d}.pt",
                map_location="cpu", weights_only=True,
            )
            self.assertEqual(restored["clock"]["env_steps"], payload["step"])
            for name, value in payload["policy"].items():
                th.testing.assert_close(value, restored["policy"][name])
            self.assertNotIn("discriminator", restored)
            self.assertGreaterEqual(
                th.load(max(standard_resume.glob("pulse_*.pt")),
                        map_location="cpu", weights_only=True)["step"], 16,
            )

            gaifo_dir = root / "pulse_gaifo"
            with (
                patch.object(sys, "argv", [
                    "pulse.py", "--replay-dir", str(replay_dir),
                    "--distill-checkpoint", str(distill_path),
                    "--gaifo-imitation", "--n-sim", "2", "--rollout", "2",
                    "--ppo-batch", "4", "--feature-size", "8",
                    "--policy-hidden", "8", "--critic-hidden", "8",
                    "--trajectory-length", "2", "--discriminator-batch", "2",
                    "--discriminator-microbatch", "2",
                    "--discriminator-heldout-size", "4",
                    "--discriminator-hidden", "8", "--frame-embedding", "8",
                    "--temporal-hidden", "8", "--history-capacity", "8",
                    "--history-add-size", "4", "--replay-reset-fraction", "1",
                    "--reset-state-limit", "16", "--no-bf16", "--timesteps", "8",
                    "--checkpoint-interval", "8", "--log-dir", str(root / "logs"),
                    "--checkpoint-dir", str(gaifo_dir),
                ]),
                redirect_stdout(io.StringIO()),
            ):
                pulse_main()
            gaifo_saved = max(gaifo_dir.rglob("pulse_*.pt"))
            gaifo_payload = th.load(gaifo_saved, map_location="cpu", weights_only=True)
            self.assertGreaterEqual(gaifo_payload["step"], 8)
            self.assertTrue(gaifo_payload["config"]["gaifo_imitation"])
            self.assertIn("gaifo_history", gaifo_payload)
            self.assertIn("discriminator_optimizer", gaifo_payload)
            self.assertTrue(gaifo_payload["gaifo_update"]["has_updated"])

            with (
                patch.object(sys, "argv", [
                    "pulse.py", "--resume-checkpoint", str(gaifo_saved),
                    "--timesteps", "16", "--run-name", "gaifo-resume",
                ]),
                redirect_stdout(io.StringIO()),
            ):
                pulse_main()
            gaifo_resume = gaifo_dir / "gaifo-resume"
            gaifo_restored = th.load(
                gaifo_resume / f"pulse_{gaifo_payload['step']:012d}.pt",
                map_location="cpu", weights_only=True,
            )
            self.assertEqual(gaifo_restored["clock"]["env_steps"], gaifo_payload["step"])
            self.assertEqual(
                gaifo_restored["distill_sha256"], gaifo_payload["distill_sha256"]
            )
            for name, value in gaifo_payload["discriminator"].items():
                th.testing.assert_close(value, gaifo_restored["discriminator"][name])
            self.assertEqual(
                gaifo_restored["gaifo_history"]["size"],
                gaifo_payload["gaifo_history"]["size"],
            )
            gaifo_latest = max(gaifo_resume.glob("pulse_*.pt"))
            self.assertGreaterEqual(
                th.load(gaifo_latest, map_location="cpu", weights_only=True)["step"], 16
            )

            advanced_dir = root / "pulse_gaifo_advanced"
            with (
                patch.object(sys, "argv", [
                    "pulse.py", "--replay-dir", str(replay_dir),
                    "--distill-checkpoint", str(distill_path),
                    "--gaifo-imitation", "--factorize", "--recency-replay",
                    "--exp-log-odds-reward", "--n-sim", "2", "--rollout", "2",
                    "--ppo-batch", "4", "--feature-size", "8",
                    "--policy-hidden", "8", "--critic-hidden", "8",
                    "--trajectory-length", "2", "--discriminator-batch", "2",
                    "--discriminator-microbatch", "2",
                    "--discriminator-heldout-size", "4",
                    "--discriminator-hidden", "8", "--frame-embedding", "8",
                    "--temporal-hidden", "8", "--history-capacity", "8",
                    "--history-add-size", "4", "--replay-reset-fraction", "1",
                    "--reset-state-limit", "16", "--no-bf16", "--timesteps", "8",
                    "--checkpoint-interval", "8", "--log-dir", str(root / "logs"),
                    "--checkpoint-dir", str(advanced_dir),
                ]),
                redirect_stdout(io.StringIO()),
            ):
                pulse_main()
            advanced = th.load(
                max(advanced_dir.rglob("pulse_*.pt")), map_location="cpu",
                weights_only=True,
            )
            self.assertTrue(advanced["config"]["factorize"])
            self.assertTrue(advanced["config"]["exp_log_odds_reward"])
            self.assertEqual(advanced["gaifo_history"]["type"], "recency")
            self.assertIn("ball_encoder.0.weight", advanced["discriminator"])

            frames, internal = load_demonstration_reset_frames(
                replay_dir, "cuda:0", limit=16,
            )
            provider = ReplayResetProvider(
                DatasetResetSampler(
                    reset_index_dataset(th.arange(len(frames), device=frames.device)),
                    probability=1.0,
                ),
                frames, internal,
            )
            base = CARLTorchVectorEnv(
                n_sim=1, n_blue=1, n_orange=1, frameskip=4,
                normalize=True, discrete_actions=True,
                reset_state_provider=provider,
            )
            try:
                viewer_env, blue, orange = load_match(saved, saved, base, 4, None)
                scene = viewer_env.reset()
                latent = th.cat((
                    blue.act(scene[:1], deterministic=True).action,
                    orange.act(scene[1:], deterministic=True).action,
                ))
                observation, _, _, _, _ = viewer_env.step(latent)
                self.assertEqual(observation.shape, scene.shape)
                mixed, _, _ = load_match(gaifo_saved, gaifo_latest, base, 4, None)
                self.assertEqual(mixed.reset().shape, scene.shape)
            finally:
                base.close()


if __name__ == "__main__":
    unittest.main()
