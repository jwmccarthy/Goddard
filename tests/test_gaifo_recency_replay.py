"""Recency-favored replay for generated discriminator windows."""

import unittest

import torch as th

from gaifo import RecencyReplayBuffer


def marked_windows(start: int, count: int) -> th.Tensor:
    windows = th.zeros(count, 2, 51)
    windows[:, -1, 0] = th.arange(start, start + count)
    return windows


class RecencyReplayTests(unittest.TestCase):
    def test_new_mode_is_favored_without_forgetting_old_windows(self):
        history = RecencyReplayBuffer(40, 2, "cpu", seed=2)
        history.add(marked_windows(0, 200), 200)
        history.add(marked_windows(200, 200), 200)

        self.assertEqual(history.size, 40)
        self.assertEqual(history.seen, 400)
        self.assertEqual(history.recent.size, 30)
        sampled = th.cat([history.sample(20, "cpu") for _ in range(20)])[:, -1, 0]
        self.assertTrue((sampled < 400).all())
        old = (sampled < 200).sum().item()
        new = (sampled >= 200).sum().item()
        self.assertGreater(old, 0)
        self.assertGreater(new, 3 * old)

    def test_small_or_empty_samples_and_fraction_validation(self):
        history = RecencyReplayBuffer(4, 2, "cpu", seed=5)
        self.assertEqual(history.sample(2, "cpu").shape, (0, 2, 51))
        history.add(marked_windows(0, 3), 3)
        self.assertEqual(history.sample(2, "cpu").shape, (2, 2, 51))
        history.add(marked_windows(3, 3), 3)
        self.assertEqual(history.sample(4, "cpu").shape, (4, 2, 51))
        with self.assertRaisesRegex(ValueError, "capacity"):
            RecencyReplayBuffer(1, 2, "cpu")
        with self.assertRaisesRegex(ValueError, "fraction"):
            RecencyReplayBuffer(4, 2, "cpu", reservoir_fraction=0.75)


if __name__ == "__main__":
    unittest.main()
