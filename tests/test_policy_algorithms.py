import unittest
from unittest.mock import Mock

import torch

from jarl.learn import PPOLoss, SPOLoss

from ppo import SyntheticMatchResetProvider, build_policy_loss


class PolicyAlgorithmTests(unittest.TestCase):
    def test_synthetic_match_reset_preserves_default_resets(self):
        provider = Mock(return_value=None)
        reset_mask = torch.tensor([True, False])

        state = SyntheticMatchResetProvider(provider)(reset_mask)

        self.assertIsNone(state)
        provider.assert_called_once_with(reset_mask)

    def test_builds_ppo_loss(self):
        self.assertIsInstance(build_policy_loss("ppo", None, None, 0.01), PPOLoss)

    def test_builds_spo_loss(self):
        self.assertIsInstance(build_policy_loss("spo", None, None, 0.01), SPOLoss)

    def test_rejects_unknown_algorithm(self):
        with self.assertRaisesRegex(ValueError, "unknown policy optimization"):
            build_policy_loss("unknown", None, None, 0.01)


if __name__ == "__main__":
    unittest.main()
