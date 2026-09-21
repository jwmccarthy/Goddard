import tempfile
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from watch_checkpoints import (
    CheckpointRegistry,
    file_sha256,
    resolve_pulse_artifact,
    simulate,
)


class SelfPlayCheckpointWatcherTest(unittest.TestCase):
    def test_registry_discovers_nested_self_play_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "self-play-run"
            run.mkdir()
            checkpoint = run / "self_play_000000000123.pt"
            checkpoint.touch()
            registry = CheckpointRegistry(root)

            listed = registry.list()

            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0].step, 123)
            self.assertEqual(registry.resolve(listed[0].relative_path), checkpoint)

    def test_registry_discovers_nested_difo_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "difo-run"
            run.mkdir()
            checkpoint = run / "difo_000000000032.pt"
            checkpoint.touch()
            registry = CheckpointRegistry(root)

            listed = registry.list()

            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0].step, 32)
            self.assertEqual(listed[0].kind, "difo")
            self.assertEqual(registry.resolve(listed[0].relative_path), checkpoint)

    def test_registry_ignores_other_checkpoint_kinds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "simple_000000000001.pt").touch()
            registry = CheckpointRegistry(root)

            self.assertEqual(registry.list(), [])

    def test_resolve_pulse_artifact_verifies_embedded_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "frozen_pulse.pt"
            artifact.touch()
            payload = {
                "distill_sha256": "abc",
                "pulse_artifact": "frozen_pulse.pt",
                "pulse_sha256": file_sha256(artifact),
            }
            checkpoint = root / "self_play_000000000001.pt"
            checkpoint.touch()

            resolved = resolve_pulse_artifact(None, checkpoint, payload, payload)

            self.assertEqual(resolved, artifact)

    def test_resolve_pulse_artifact_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "frozen_pulse.pt"
            artifact.touch()
            payload = {
                "distill_sha256": "abc",
                "pulse_artifact": "frozen_pulse.pt",
                "pulse_sha256": "wrong",
            }
            checkpoint = root / "self_play_000000000001.pt"
            checkpoint.touch()

            with self.assertRaisesRegex(ValueError, "verification"):
                resolve_pulse_artifact(None, checkpoint, payload, payload)

    def test_simulation_uses_the_pulse_pipeline(self):
        state = SimpleNamespace(publish=MagicMock())
        registry = SimpleNamespace()

        with patch("watch_checkpoints._simulate_pulse") as pulse:
            simulate(
                state,
                registry,
                Path("blue.pt"),
                Path("orange.pt"),
                SimpleNamespace(),
            )

        pulse.assert_called_once()
        state.publish.assert_not_called()

    def test_simulation_uses_the_difo_pipeline(self):
        state = SimpleNamespace(publish=MagicMock())
        registry = SimpleNamespace()

        with patch("watch_checkpoints._simulate_difo") as difo:
            simulate(
                state,
                registry,
                Path("difo_000000000001.pt"),
                Path("difo_000000000001.pt"),
                SimpleNamespace(),
            )

        difo.assert_called_once()
        state.publish.assert_not_called()

    def test_simulation_rejects_mixed_checkpoint_kinds(self):
        state = SimpleNamespace(publish=MagicMock())
        registry = SimpleNamespace()

        simulate(
            state,
            registry,
            Path("self_play_000000000001.pt"),
            Path("difo_000000000001.pt"),
            SimpleNamespace(),
        )

        state.publish.assert_called_once()
        self.assertIn("mix", state.publish.call_args.args[0]["error"])

    def test_simulation_publishes_pipeline_errors(self):
        state = SimpleNamespace(publish=MagicMock())
        registry = SimpleNamespace()

        with patch(
            "watch_checkpoints._simulate_pulse", side_effect=ValueError("bad")
        ):
            simulate(
                state,
                registry,
                Path("blue.pt"),
                Path("orange.pt"),
                SimpleNamespace(),
            )

        state.publish.assert_called_once()
        self.assertIn("bad", state.publish.call_args.args[0]["error"])


if __name__ == "__main__":
    unittest.main()
