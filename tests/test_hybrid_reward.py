import unittest
from unittest.mock import Mock, patch

import torch

from jarl.data import TensorBatch

from gaifo import parse_arguments
from imitation import AddImitationReward, EveryNUpdates
from rewards import SeerReward


class HybridRewardTests(unittest.TestCase):
    def test_seer_reward_is_disabled_by_default(self):
        with patch("sys.argv", ["gaifo.py"]):
            arguments = parse_arguments()

        self.assertFalse(arguments.seer_reward)

    def test_periodic_stage_runs_first_and_every_n_updates(self):
        stage = Mock()
        stage.run.side_effect = lambda experience: (experience, {"ran": {}})
        periodic = EveryNUpdates(stage, interval=3)

        metrics = [periodic.run("rollout")[1] for _ in range(7)]

        self.assertEqual(stage.run.call_count, 3)
        self.assertEqual(
            metrics,
            [{"ran": {}}, {}, {}, {"ran": {}}, {}, {}, {"ran": {}}],
        )

    def test_shaping_scale_does_not_change_goal_reward(self):
        reward = SeerReward(1, 1, normalize=False)
        reward.set_shaping_scale(0.25)

        components = reward._scale_components(
            {
                "goal_scored": torch.tensor([2.0]),
                "ball_touch": torch.tensor([2.0]),
            }
        )

        torch.testing.assert_close(components["goal_scored"], torch.tensor([2.0]))
        torch.testing.assert_close(components["ball_touch"], torch.tensor([0.5]))

    def test_imitation_is_added_at_full_strength(self):
        batch = TensorBatch(
            {
                "reward": torch.tensor([[1.0, 2.0]]),
                "imitation_reward": torch.tensor([[0.25, 0.5]]),
            }
        )

        combined = AddImitationReward()(batch, None)

        torch.testing.assert_close(
            combined["reward"], torch.tensor([[1.25, 2.5]])
        )


if __name__ == "__main__":
    unittest.main()
