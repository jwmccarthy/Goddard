import unittest

from types import SimpleNamespace

import torch as th

from distill import (
    DistillCheckpoints,
    DistillRolloutTransform,
    categorical_distillation_loss,
    diagonal_gaussian_kl,
    factor_actions,
    kl_coefficient,
)
from jarl.data.batch import TensorBatch


class DistillTest(unittest.TestCase):
    def test_checkpoint_waits_for_rollout_boundary(self):
        checkpoint = DistillCheckpoints.__new__(DistillCheckpoints)
        checkpoint.step = 0
        checkpoint.next_step = 10
        checkpoint.buffer = SimpleNamespace(position=1)

        self.assertFalse(checkpoint.ready(10))
        self.assertEqual(checkpoint.step, 10)

        checkpoint.buffer.position = 0
        self.assertTrue(checkpoint.ready(11))

    def test_matching_gaussians_have_zero_kl(self):
        mean = th.randn(4, 3)
        log_variance = th.randn(4, 3).clamp(-5, 2)

        loss = diagonal_gaussian_kl(mean, log_variance, mean, log_variance)

        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_kl_is_summed_over_latent_dimensions(self):
        posterior_mean = th.ones(2, 3)
        zeros = th.zeros_like(posterior_mean)

        loss = diagonal_gaussian_kl(posterior_mean, zeros, zeros, zeros)

        self.assertAlmostEqual(loss.item(), 1.5, places=6)

    def test_kl_coefficient_anneals_linearly(self):
        values = [
            kl_coefficient(step, 0.01, 0.001, 100, 200)
            for step in (0, 100, 150, 200, 300)
        ]

        self.assertEqual(values, [0.01, 0.01, 0.0055, 0.001, 0.001])

    def test_factorized_action_loss_and_argmax(self):
        target = th.tensor([[2, 1, 0, 1, 0, 2, 1]])
        logits = th.full((1, 18), -5.0)
        offset = 0
        for size, value in zip((3, 3, 3, 2, 2, 3, 2), target[0]):
            logits[0, offset + value] = 5.0
            offset += size

        loss, accuracy = categorical_distillation_loss(logits, target)

        th.testing.assert_close(factor_actions(logits), target)
        self.assertLess(loss.item(), 0.001)
        self.assertEqual(accuracy.item(), 1.0)

    def test_temporal_pairs_do_not_cross_resets(self):
        observation = th.arange(18, dtype=th.float32).reshape(3, 2, 3)
        terminated = th.tensor([[False, False], [True, False], [False, False]])
        truncated = th.tensor([[False, True], [False, False], [False, False]])
        action = th.zeros((3, 2, 7), dtype=th.long)
        teacher_action = action.clone()
        teacher_action[2, 1, 0] = 1
        batch = TensorBatch(
            {
                "observation": observation,
                "terminated": terminated,
                "truncated": truncated,
                "action": action,
                "teacher_action": teacher_action,
            }
        )

        transformed = DistillRolloutTransform()(batch, None)

        th.testing.assert_close(
            transformed["previous_observation"][1:], observation[:-1]
        )
        th.testing.assert_close(
            transformed["smooth_pair"],
            th.tensor([[False, False], [True, False], [False, True]]),
        )
        self.assertAlmostEqual(
            transformed["action_agreement"][2, 1].item(), 6 / 7
        )


if __name__ == "__main__":
    unittest.main()
