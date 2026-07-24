import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from imitation import EveryNUpdates
from jarl.runtime.clock import Clock
from training_checkpoint import TrainingCheckpointer


class TrainingCheckpointTests(unittest.TestCase):
    def test_round_trip_restores_training_state(self):
        module = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
        optimizer.zero_grad()
        module(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        cadence = EveryNUpdates(SimpleNamespace(run=lambda value: value), 3)
        cadence.update_count = 7
        expected_weight = module.weight.detach().clone()

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_latest.pt"
            checkpointer = TrainingCheckpointer(
                path,
                modules={"model": module},
                optimizers={"optimizer": optimizer},
                stateful={"cadence": cadence},
            )
            checkpointer(SimpleNamespace(clock=Clock(env_steps=123, learner_updates=4)))

            with torch.no_grad():
                module.weight.zero_()
            optimizer.state.clear()
            cadence.update_count = 0
            clock = checkpointer.load(path, "cpu")

        torch.testing.assert_close(module.weight, expected_weight)
        self.assertTrue(optimizer.state)
        self.assertEqual(cadence.update_count, 7)
        self.assertEqual(clock.env_steps, 123)
        self.assertEqual(clock.learner_updates, 4)

    def test_module_preload_restores_weights(self):
        source = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(source.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training_latest.pt"
            TrainingCheckpointer(
                path,
                modules={"model": source},
                optimizers={"optimizer": optimizer},
            )(SimpleNamespace(clock=Clock()))
            target = torch.nn.Linear(2, 1)
            TrainingCheckpointer.load_modules(path, {"model": target}, "cpu")

        torch.testing.assert_close(target.weight, source.weight)
        torch.testing.assert_close(target.bias, source.bias)


if __name__ == "__main__":
    unittest.main()
