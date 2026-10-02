"""Confidence-weighted expert examples for both GAIFO discriminator modes."""

import sys
import unittest
from unittest.mock import patch

import torch as th
import torch.nn.functional as F

from gaifo import (
    BLUE_START,
    SceneDiscriminatorLoss,
    confident_expert_weights,
    parse_args,
    train_discriminator_minibatch,
)
from jarl.data import TensorBatch


class ScoreDiscriminator(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = th.nn.Parameter(th.tensor(1.0))

    def forward(self, windows: th.Tensor) -> th.Tensor:
        return self.scale * windows[:, -1, 3]


class ScoreFactorizedDiscriminator(th.nn.Module):
    factorized = True

    def __init__(self):
        super().__init__()
        self.car_scale = th.nn.Parameter(th.tensor(1.0))
        self.ball_scale = th.nn.Parameter(th.tensor(1.0))

    def forward(self, windows: th.Tensor) -> th.Tensor:
        return th.stack((
            self.car_scale * windows[:, -1, BLUE_START + 15],
            self.ball_scale * windows[:, -1, 3],
        ), dim=-1)


def example_batch() -> TensorBatch:
    windows = th.zeros(8, 2, 51)
    windows[:, -1, 3] = th.tensor([0.7, -0.7, 1.5, -1.5, -3., -0.5, 0.6, -2.])
    windows[:, -1, BLUE_START + 15] = th.tensor([1., -1., 2., -2., -1., -3., 0.6, 1.])
    windows[[1, 3, 5, 7], :, BLUE_START] = 3_000 / 4_108
    return TensorBatch({
        "window": windows,
        "is_agent": th.tensor([1., 1., 1., 1., 0., 0., 0., 0.]),
    })


class HardPositiveMiningTests(unittest.TestCase):
    def test_flag_is_opt_in(self):
        for flag, expected in ((None, False), ("--hard-positive-mining", True),
                               ("--no-hard-positive-mining", False)):
            with self.subTest(flag=flag):
                flags = ["gaifo.py", "--replay-dir", "parsed_replays"]
                if flag is not None:
                    flags.append(flag)
                with patch.object(sys, "argv", flags):
                    parsed, _ = parse_args()
                self.assertIs(parsed.hard_positive_mining, expected)

    def test_only_confident_expert_examples_gain_weight_without_weight_gradients(self):
        logits = th.tensor([-8., -2., 0., 2.], requires_grad=True)
        weights = confident_expert_weights(logits)
        self.assertFalse(weights.requires_grad)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])
        th.testing.assert_close(weights[2:], th.ones(2))
        self.assertLessEqual(weights.max().item(), 3.0)

        batch = example_batch()
        discriminator = ScoreDiscriminator()
        logits = discriminator(batch["window"])
        expert_weights = confident_expert_weights(logits[4:])
        expert_weights /= expert_weights.mean()
        self.assertGreater(expert_weights[0], expert_weights[2])
        th.testing.assert_close(expert_weights.mean(), th.tensor(1.0))
        expected = 0.5 * (
            F.binary_cross_entropy_with_logits(logits[:4], batch["is_agent"][:4])
            + (F.binary_cross_entropy_with_logits(
                logits[4:], batch["is_agent"][4:], reduction="none",
            ) * expert_weights).mean()
        )
        mined = SceneDiscriminatorLoss(discriminator, hard_positive_mining=True)(batch)
        ordinary = SceneDiscriminatorLoss(discriminator)(batch)
        th.testing.assert_close(mined.loss, expected)
        self.assertLess(mined.loss.item(), ordinary.loss.item())

    def test_mining_is_stable_across_microbatch_sizes_in_both_modes(self):
        batch = example_batch()
        for discriminator_type in (ScoreDiscriminator, ScoreFactorizedDiscriminator):
            with self.subTest(mode=discriminator_type.__name__):
                reference = discriminator_type()
                expected = SceneDiscriminatorLoss(reference, hard_positive_mining=True)(batch)
                if getattr(reference, "factorized", False):
                    expert_logits = reference(batch["window"])[4:]
                    car_favorite = confident_expert_weights(expert_logits[:, 0]).argmax()
                    ball_favorite = confident_expert_weights(expert_logits[:, 1]).argmax()
                    self.assertNotEqual(car_favorite.item(), ball_favorite.item())
                    ordinary = SceneDiscriminatorLoss(reference)(batch)
                    self.assertNotAlmostEqual(expected.metrics["ball_loss"].item(),
                                              ordinary.metrics["ball_loss"].item(), places=4)

                outcomes = []
                for size in (4, 1):
                    discriminator = discriminator_type()
                    optimizer = th.optim.SGD(discriminator.parameters(), lr=0)
                    metrics = train_discriminator_minibatch(
                        batch, discriminator, optimizer,
                        SceneDiscriminatorLoss(discriminator, hard_positive_mining=True),
                        microbatch_size=size, max_grad_norm=100,
                    )
                    outcomes.append((metrics, [parameter.grad.clone()
                                               for parameter in discriminator.parameters()]))
                    th.testing.assert_close(metrics["loss"], expected.loss)
                    self.assertGreater(metrics["expert_mining_max_weight"].item(), 1.0)
                for first, second in zip(outcomes[0][1], outcomes[1][1]):
                    th.testing.assert_close(first, second)
                th.testing.assert_close(outcomes[0][0]["loss"], outcomes[1][0]["loss"])


if __name__ == "__main__":
    unittest.main()
