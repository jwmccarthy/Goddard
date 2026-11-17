import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from deployment.export_rlbot import export
from training.models import mlp


@unittest.skipUnless(importlib.util.find_spec("rlbot"), "requires standalone RLBot test dependencies")
class RLBotExportTests(unittest.TestCase):
    def test_exported_bots_validate_all_layouts_and_frameskips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for schema, controls, inputs, frameskip in (
                ("ego-controller-v1", False, 166, 8),
                ("ego-controller-v3", False, 144, 4),
                ("ego-controller-v3", True, 152, 8),
            ):
                with self.subTest(schema=schema, controls=controls):
                    checkpoint = root / "checkpoint.pt"
                    torch.manual_seed(345)
                    actor = mlp(inputs, 18, 16, 2, .01)
                    config = dict(team_size=1, observation_schema=schema, previous_controls=controls,
                                  frameskip=frameskip, hidden_size=16, layers=2, seed=1)
                    torch.save(dict(format="goddard-leanrl-v1", algorithm="ppo", config=config,
                                    step=4, agent={"actor." + k: v for k, v in actor.state_dict().items()}),
                               checkpoint)
                    destination = root / f"bot-{inputs}"
                    export(checkpoint, destination)
                    metadata = json.loads((destination / "export.json").read_text())
                    self.assertEqual(metadata["observation_size"], inputs)
                    self.assertEqual(metadata["previous_controls"], controls)
                    result = subprocess.run([sys.executable, str(destination / "self_check.py")],
                                            cwd=destination, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    # Settings cannot silently relabel a model with a different input schema.
                    settings_path = destination / "settings.json"
                    settings = json.loads(settings_path.read_text())
                    settings["previous_controls"] = not controls
                    settings_path.write_text(json.dumps(settings))
                    result = subprocess.run([sys.executable, str(destination / "self_check.py")],
                                            cwd=destination, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("settings observation layout differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
