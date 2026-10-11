"""PCIL contrastive learning and on-policy replay integration."""

import argparse
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from jarl.data import TensorBatch

from gaifo import ExpertSceneDataset, HistoricalReplayBuffer, build_env
from pcil import (
    PCIL_ARCHITECTURE, PCILContrastiveLoss, PCILContrastiveUpdate,
    PCILEncoder, PCILReward, load_resume_checkpoint, main,
    sample_expert_pairs,
)
from replay_layout import team_replay_row_size
from watch_checkpoints import checkpoint_kind, load_policy_checkpoint


def write_expert(folder: Path) -> None:
    rows = np.zeros((48, team_replay_row_size(1)), dtype=np.float32)
    rows[:, 0] = np.linspace(0.05, 0.55, len(rows))
    rows[:, 2] = 91.25 / 2076
    for car, y in ((9, -1200), (30, 1200)):
        rows[:, car + 1] = y / 6000
        rows[:, car + 2] = 17 / 2076
        rows[:, car + 9] = rows[:, car + 14] = rows[:, car + 16] = 1
    rows[:, 51:85] = 1
    rows[:, 137] = 1
    np.save(folder / "replay.npy", rows)
    np.savez_compressed(folder / "replay.unsafe-starts.npz", frame_skip=4)


class AxisEncoder(nn.Module):
    trajectory_length = 2
    scene_size = 2

    def forward(self, windows: th.Tensor) -> th.Tensor:
        return F.normalize(windows[..., -1, :], dim=-1)


class AxisExpert:
    def sample(self, n: int, device: th.device) -> th.Tensor:
        windows = th.zeros(n, 2, 2, device=device)
        windows[:, -1, 0] = 1
        return windows


class PCILTests(unittest.TestCase):
    def setUp(self):
        th.manual_seed(0)

    def test_info_nce_uses_one_expert_positive_and_all_agent_negatives(self):
        encoder = PCILEncoder(3, 5, hidden_size=12, embedding_size=4)
        windows = th.randn(6, 3, 5)
        anchors, positives, negatives = windows[:2], windows[2:4], windows[4:6]
        encoded = encoder(windows)
        th.testing.assert_close(encoded.norm(dim=-1), th.ones(6))

        objective = PCILContrastiveLoss(encoder, temperature=0.07, gradient_penalty_weight=0)
        actual = objective(anchors, positives, negatives)
        anchor, positive, negative = encoder(anchors), encoder(positives), encoder(negatives)
        logits = th.cat(((anchor * positive).sum(-1, keepdim=True), anchor @ negative.T), dim=1)
        expected = F.cross_entropy(logits / 0.07, th.zeros(2, dtype=th.long))
        th.testing.assert_close(actual.loss, expected)
        actual.loss.backward()
        self.assertGreater(encoder.model[0].weight.grad.abs().sum().item(), 0)

        encoder.zero_grad(set_to_none=True)
        penalized = PCILContrastiveLoss(encoder, gradient_penalty_weight=10)(
            anchors, positives, negatives,
        )
        th.testing.assert_close(
            penalized.loss,
            penalized.metrics["contrastive_loss"] + 10 * penalized.metrics["gradient_penalty"],
        )
        self.assertGreaterEqual(penalized.metrics["gradient_penalty"].item(), 0)
        penalized.loss.backward()
        self.assertTrue(th.isfinite(encoder.model[0].weight.grad).all())

    def test_expert_positive_is_a_different_recorded_window(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_expert(folder)
            expert = ExpertSceneDataset(
                folder, 4, frame_skip=4, reject_discontinuities=True,
            )
            anchors, positives = sample_expert_pairs(expert, 128)
            self.assertEqual(anchors.shape, (128, 4, 51))
            self.assertTrue((anchors[:, -1, 0] != positives[:, -1, 0]).all())

    def test_cosine_reward_respects_validity_replay_fraction_and_frame_skip(self):
        windows = th.zeros(2, 2, 2, 2)
        windows[:, :, -1, 0] = th.tensor([[1., -1.], [-1., 1.]])
        batch = TensorBatch({
            "scene_window": windows,
            "scene_window_valid": th.tensor([[True, False], [True, True]]),
            "scene_window_agent_fraction": th.tensor([[1., 1.], [.25, 0.]]),
            "reward": th.tensor([[0., 1.], [0., -1.]]),
        })
        result = PCILReward(
            AxisExpert(), AxisEncoder(), batch_size=2, frameskip=8,
            goal_reward_weight=0.5,
        )(batch, None)
        th.testing.assert_close(result["imitation_reward"], th.tensor([[2., 0.], [-.5, 0.]]))
        th.testing.assert_close(result["training_reward"], th.tensor([[2., .5], [-.5, -.5]]))
        self.assertTrue(result["learner_mask"].all())

    def test_update_excludes_borrowed_windows_from_negative_history(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            folder = Path(directory)
            write_expert(folder)
            expert = ExpertSceneDataset(
                folder, 4, frame_skip=4, reject_discontinuities=True,
            )
            encoder = PCILEncoder(4, 51, hidden_size=12, embedding_size=4)
            optimizer = th.optim.Adam(encoder.parameters(), lr=1e-3)
            history = HistoricalReplayBuffer(8, 4, "cpu", scene_size=51)
            updater = PCILContrastiveUpdate(
                expert, encoder, optimizer, batch_size=2, epochs=2,
                loss=PCILContrastiveLoss(encoder, gradient_penalty_weight=0),
                max_grad_norm=1, history=history, history_add_size=8,
            )
            batch = TensorBatch({
                "scene_window": th.randn(2, 3, 4, 51),
                "scene_window_valid": th.tensor([[True, True, False], [True, True, True]]),
                "scene_window_agent_fraction": th.tensor([[1., .5, 1.], [1., 1., .25]]),
            })
            before = encoder.model[0].weight.clone()
            returned, metrics = updater.run(batch)
            self.assertIs(returned, batch)
            self.assertEqual(metrics["PCIL"]["minibatches"], 4)
            self.assertEqual(history.size, 3)
            self.assertGreater((encoder.model[0].weight - before).abs().sum().item(), 0)
            self.assertTrue(np.isfinite(metrics["PCIL"]["loss"]))


@unittest.skipUnless(
    os.getenv("GODDARD_GPU_SMOKE") == "1" and th.cuda.is_available(),
    "opt-in CUDA/CARL PCIL training smoke",
)
class PCILGPUSmokeTests(unittest.TestCase):
    def test_checkpointed_training_resume_and_viewer(self):
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            root = Path(directory)
            replays = root / "replays"
            replays.mkdir()
            write_expert(replays)
            flags = [
                "--replay-dir", str(replays),
                "--n-sim", "2", "--rollout", "8", "--timesteps", "64",
                "--trajectory-length", "4", "--replay-reset-fraction", "1",
                "--policy-hidden", "16", "--critic-hidden", "16",
                "--encoder-hidden", "16", "--embedding-size", "8",
                "--contrastive-batch", "4", "--contrastive-epochs", "1",
                "--gradient-penalty-weight", "1",
                "--ppo-batch", "8", "--ppo-epochs", "1", "--sequence-length", "4",
                "--history-capacity", "16", "--history-add-size", "8",
                "--checkpoint-interval", "32", "--checkpoint-keep", "3",
                "--log-dir", str(root / "runs"),
                "--checkpoint-dir", str(root / "checkpoints"),
            ]
            output = io.StringIO()
            with redirect_stdout(output):
                main(flags)
            checkpoints = sorted((root / "checkpoints").rglob("pcil_*.pt"))
            self.assertGreaterEqual(len(checkpoints), 2)
            saved = load_resume_checkpoint(checkpoints[-1])
            self.assertEqual(saved["step"], 64)
            self.assertEqual(saved["config"]["architecture"], PCIL_ARCHITECTURE)
            self.assertTrue(saved["config"]["gru"])
            self.assertTrue(saved["encoder_optimizer"]["state"])
            self.assertIn("InfoNCE", output.getvalue())
            self.assertEqual(checkpoint_kind(checkpoints[-1]), "pcil")

            env = build_env(argparse.Namespace(
                team_size=1, n_sim=1, seed=0, frameskip=4,
                max_ticks=1_000_000, no_touch_timeout=30.0,
            ))
            try:
                policy, signature = load_policy_checkpoint(checkpoints[-1], env, 4, None)
                self.assertEqual(signature[0], "pcil")
                self.assertEqual(policy.episode_limits.max_ticks, 1_000_000)
            finally:
                env.close()

            with redirect_stdout(output):
                main(["--resume-checkpoint", str(checkpoints[-1]), "--timesteps", "128"])
            resumed = next((root / "checkpoints").rglob("pcil_000000000128.pt"))
            self.assertEqual(load_resume_checkpoint(resumed)["step"], 128)


if __name__ == "__main__":
    unittest.main()
