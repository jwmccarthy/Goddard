import unittest

import torch
import torch.nn as nn

from jarl.data import TensorBatch, TensorDataset

from imitation import (
    SequenceDiscriminator,
    SequenceDiscriminatorReward,
    SequenceGAIFOMinibatches,
    _sequence_chunks,
)


class SumDiscriminator(nn.Module):
    def forward(self, sequence):
        observation, _ = sequence
        return observation[..., 0].sum(dim=1)


class SequenceGAIFOTests(unittest.TestCase):
    def test_chunks_keep_each_environment_sequence_contiguous(self):
        values = torch.arange(8).reshape(4, 2)

        chunks = _sequence_chunks(values, 2)

        torch.testing.assert_close(
            chunks,
            torch.tensor([[0, 2], [1, 3], [4, 6], [5, 7]]),
        )

    def test_discriminator_returns_one_logit_per_sequence(self):
        discriminator = SequenceDiscriminator(hidden_size=8, noise_std=0.0)
        observation = torch.randn(3, 4, 119)

        score = discriminator((observation, observation))

        self.assertEqual(score.shape, (3,))

    def test_sampler_filters_terminal_sequences_and_balances_classes(self):
        observation = torch.randn(4, 2, 119)
        terminal = torch.zeros(4, 2, dtype=torch.bool)
        terminal[0, 0] = True
        rollout = TensorBatch(
            {
                "observation": observation,
                "next_obs": observation.clone(),
                "learner_mask": torch.ones(4, 2, dtype=torch.bool),
                "terminated": terminal,
                "truncated": torch.zeros_like(terminal),
            }
        )
        expert = TensorDataset(
            TensorBatch(
                {
                    "observation": torch.randn(3, 2, 119),
                    "next_obs": torch.randn(3, 2, 119),
                }
            )
        )
        sampler = SequenceGAIFOMinibatches(
            expert, sequence_length=2, batch_size=2
        )

        batch = next(iter(sampler(rollout)))

        self.assertEqual(batch["observation"].shape, (4, 2, 119))
        torch.testing.assert_close(
            batch["is_agent"], torch.tensor([1.0, 1.0, 0.0, 0.0])
        )

    def test_reward_is_emitted_at_valid_sequence_final_step(self):
        observation = torch.zeros(4, 2, 119)
        observation[..., 0] = torch.tensor(
            [[0.0, 1.0], [10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
        )
        terminal = torch.zeros(4, 2, dtype=torch.bool)
        terminal[2, 1] = True
        batch = TensorBatch(
            {
                "observation": observation,
                "next_obs": observation.clone(),
                "terminated": terminal,
                "truncated": torch.zeros_like(terminal),
            }
        )
        transform = SequenceDiscriminatorReward(SumDiscriminator(), 2)

        reward = transform(batch, None)["imitation_reward"]

        torch.testing.assert_close(
            reward,
            torch.tensor([[0.0, 0.0], [-10.0, -12.0], [0.0, 0.0], [-50.0, 0.0]]),
        )


if __name__ == "__main__":
    unittest.main()
