import unittest

from jarl.learn import PPOLoss, SPOLoss

from ppo import build_policy_loss


class PolicyAlgorithmTests(unittest.TestCase):
    def test_builds_ppo_loss(self):
        self.assertIsInstance(build_policy_loss("ppo", None, None, 0.01), PPOLoss)

    def test_builds_spo_loss(self):
        self.assertIsInstance(build_policy_loss("spo", None, None, 0.01), SPOLoss)

    def test_rejects_unknown_algorithm(self):
        with self.assertRaisesRegex(ValueError, "unknown policy optimization"):
            build_policy_loss("unknown", None, None, 0.01)


if __name__ == "__main__":
    unittest.main()
