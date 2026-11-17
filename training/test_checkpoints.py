import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from training.checkpoints import save_checkpoint


class CheckpointRetentionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.latest = Path(self.directory.name) / "latest.pt"
        self.args = SimpleNamespace(observation_schema="ego-controller-v3", previous_controls=False)
        self.agent = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.agent.parameters())
        rng = patch("torch.cuda.get_rng_state_all", return_value=[])
        rng.start()
        self.addCleanup(rng.stop)

    def save(self, step):
        save_checkpoint(self.latest, "ppo", self.args, self.agent, self.optimizer, step)

    def load(self, path):
        return torch.load(path, map_location="cpu", weights_only=False)

    def test_retains_decision_steps_and_updates_latest(self):
        # Existing runs start with a regular latest.pt file.
        torch.save({"step": 0}, self.latest)
        self.save(524288)
        first = self.latest.resolve()
        expected = self.load(first)["agent"]["weight"].clone()
        with torch.no_grad():
            self.agent.weight.add_(1)
        self.save(1048576)

        self.assertEqual(first.name, "checkpoint_524288.pt")
        self.assertTrue(self.latest.is_symlink())
        self.assertEqual(self.latest.readlink(), Path("checkpoint_1048576.pt"))
        self.assertEqual(self.load(first)["step"], 524288)
        self.assertEqual(self.load(self.latest)["step"], 1048576)
        torch.testing.assert_close(self.load(first)["agent"]["weight"], expected)
        torch.testing.assert_close(self.load(self.latest)["agent"]["weight"], expected + 1)

    def test_failed_save_preserves_latest_and_previous_checkpoint(self):
        self.save(524288)

        def failed_save(payload, path):
            Path(path).write_bytes(b"partial checkpoint")
            raise OSError("write failed")

        with patch("training.checkpoints.torch.save", side_effect=failed_save):
            with self.assertRaisesRegex(OSError, "write failed"):
                self.save(1048576)

        self.assertEqual(self.latest.readlink(), Path("checkpoint_524288.pt"))
        self.assertEqual(self.load(self.latest)["step"], 524288)
        self.assertFalse((self.latest.parent / "checkpoint_1048576.pt").exists())


if __name__ == "__main__":
    unittest.main()
