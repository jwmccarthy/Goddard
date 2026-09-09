import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch as th

from simple import SIMPLE_ARCHITECTURE, build_policy
from test_watch_gaifo import FakeEnv
from watch_checkpoints import (
    CheckpointRegistry,
    load_simple_checkpoint,
    simulate,
)


class SimpleCheckpointWatcherTest(unittest.TestCase):
    def test_registry_discovers_nested_simple_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "simple-run"
            run.mkdir()
            checkpoint = run / "simple_000000000123.pt"
            checkpoint.touch()
            registry = CheckpointRegistry(root)

            listed = registry.list()

            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0].step, 123)
            self.assertEqual(registry.resolve(listed[0].relative_path), checkpoint)

    def test_loads_direct_action_simple_policy(self):
        env = FakeEnv()
        policy = build_policy(env, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simple_000000000001.pt"
            th.save({
                "policy": policy.state_dict(),
                "config": {
                    "architecture": SIMPLE_ARCHITECTURE,
                    "policy_hidden": 16,
                    "frameskip": 4,
                },
            }, path)

            loaded, config = load_simple_checkpoint(path, env)

        self.assertEqual(config["architecture"], SIMPLE_ARCHITECTURE)
        for key, value in policy.state_dict().items():
            th.testing.assert_close(loaded.state_dict()[key], value)

    def test_simulation_dispatches_simple_checkpoints_without_pulse(self):
        payload = {"config": {"architecture": SIMPLE_ARCHITECTURE}}
        state = SimpleNamespace(publish=MagicMock())
        registry = SimpleNamespace()

        with (
            patch("watch_checkpoints.th.load", return_value=payload),
            patch("watch_checkpoints._simulate_simple") as direct,
            patch("watch_checkpoints._simulate_pulse") as pulse,
        ):
            simulate(
                state,
                registry,
                Path("blue.pt"),
                Path("orange.pt"),
                SimpleNamespace(),
            )

        direct.assert_called_once()
        pulse.assert_not_called()
        state.publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
