import tempfile
import unittest

from pathlib import Path

from watch_demonstrations import newest_checkpoint


class TrackerCheckpointDiscoveryTest(unittest.TestCase):
    def test_finds_tracker_checkpoints_in_nested_run_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "tracker-run"
            run.mkdir()
            checkpoint = run / "tracker_000000000001.pt"
            checkpoint.touch()

            self.assertEqual(newest_checkpoint(root), checkpoint)


if __name__ == "__main__":
    unittest.main()
