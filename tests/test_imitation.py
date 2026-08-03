import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from jarl.data import TensorBatch, TensorDataset

from imitation import (
    SequenceDiscriminator,
    SequenceDiscriminatorReward,
    SequenceGAIFOMinibatches,
    _sequence_chunks,
    _sequence_grid,
)
from imitation_dataset import reset_replay_ids
from imitation_dataset import random_source_replay_ids


class SumDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, observation):
        self.batch_sizes.append(len(observation))
        return observation[..., 0].sum(dim=1)

    def forward_steps(self, observation):
        self.batch_sizes.append(len(observation))
        return observation[..., 0].cumsum(dim=1)


class SequenceGAIFOTests(unittest.TestCase):
    def test_random_source_replays_are_seeded_and_unique(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            replay_directory = source / "replays"
            replay_directory.mkdir()
            for replay_id in ("a", "b", "c", "d"):
                (replay_directory / f"{replay_id}.replay").touch()

            selected = random_source_replay_ids(source, 3, seed=7)

            self.assertEqual(selected, random_source_replay_ids(source, 3, seed=7))
            self.assertEqual(len(selected), len(set(selected)))

    def test_reset_replay_ids_reads_active_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generation = root / "generation"
            generation.mkdir()
            (root / "CURRENT").write_text("generation\n")
            (generation / "metadata.json").write_text(
                json.dumps(
                    {
                        "replays": [
                            {"replay_id": "first"},
                            {"replay_id": "second"},
                        ]
                    }
                )
            )

            self.assertEqual(reset_replay_ids(root), ["first", "second"])

    def test_chunks_keep_each_environment_sequence_contiguous(self):
        values = torch.arange(8).reshape(4, 2)

        chunks = _sequence_chunks(values, 2)

        torch.testing.assert_close(
            chunks,
            torch.tensor([[0, 2], [1, 3], [4, 6], [5, 7]]),
        )

    def test_sequence_grid_is_a_view_of_rollout_storage(self):
        values = torch.arange(8).reshape(4, 2)

        grid = _sequence_grid(values, 2)

        self.assertEqual(grid.shape, (2, 2, 2))
        self.assertEqual(grid.data_ptr(), values.data_ptr())

    def test_discriminator_returns_one_logit_per_sequence(self):
        discriminator = SequenceDiscriminator(hidden_size=8, noise_std=0.0)
        observation = torch.randn(3, 4, 137)

        score = discriminator(observation)

        self.assertEqual(score.shape, (3,))

    def test_discriminator_projects_relative_positions(self):
        discriminator = SequenceDiscriminator(hidden_size=8, noise_std=0.0)
        observation = torch.arange(137, dtype=torch.float32)

        projected = discriminator.project(observation)

        self.assertEqual(projected.shape, (53,))
        torch.testing.assert_close(projected[-12:], observation[[
            *range(119, 122),
            *range(125, 128),
            *range(131, 137),
        ]])

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
        expert_observation = torch.zeros(3, 4, 119)
        expert_observation[..., 0] = torch.arange(4)
        expert = TensorDataset(TensorBatch({"observation": expert_observation}))
        sampler = SequenceGAIFOMinibatches(
            expert, sequence_length=2, batch_size=2
        )

        batch = next(iter(sampler(rollout)))

        self.assertEqual(batch["observation"].shape, (4, 2, 119))
        self.assertNotIn("next_obs", batch)
        torch.testing.assert_close(
            batch["observation"][2:, 1, 0] - batch["observation"][2:, 0, 0],
            torch.ones(2),
        )
        torch.testing.assert_close(
            batch["is_agent"], torch.tensor([1.0, 1.0, 0.0, 0.0])
        )

    def test_sampler_uses_overlapping_agent_windows(self):
        observation = torch.zeros(4, 1, 119)
        observation[:, 0, 0] = torch.arange(4)
        rollout = TensorBatch(
            {
                "observation": observation,
                "learner_mask": torch.ones(4, 1, dtype=torch.bool),
                "terminated": torch.zeros(4, 1, dtype=torch.bool),
                "truncated": torch.zeros(4, 1, dtype=torch.bool),
            }
        )
        expert = TensorDataset(
            TensorBatch({"observation": torch.zeros(3, 2, 119)})
        )
        sampler = SequenceGAIFOMinibatches(
            expert, sequence_length=2, batch_size=3
        )

        batch = next(iter(sampler(rollout)))
        windows = {
            tuple(window.tolist()) for window in batch["observation"][:3, :, 0]
        }

        self.assertEqual(windows, {(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)})

    def test_reward_is_emitted_at_each_valid_sequence_step(self):
        observation = torch.zeros(4, 2, 119)
        observation[..., 0] = torch.tensor(
            [[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]
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

        expected = torch.zeros(4, 2)
        expected[0] = 0.35 * (1.0 - torch.tensor([0.0, 0.1]) / 10.0)
        expected[1] = 0.35 * (1.0 - torch.tensor([1.0, 1.2]) / 10.0)
        expected[2, 0] = 0.35 * (1.0 - torch.tensor(3.0) / 10.0)
        expected[3, 0] = 0.35 * (1.0 - torch.tensor(5.0) / 10.0)
        expected[3, 1] = 0.35 * (1.0 - torch.tensor(3.1) / 10.0)
        torch.testing.assert_close(reward, expected)

    def test_reward_inference_respects_batch_size(self):
        discriminator = SumDiscriminator()
        observation = torch.ones(4, 3, 119)
        batch = TensorBatch(
            {
                "observation": observation,
                "next_obs": observation.clone(),
                "learner_mask": torch.ones(4, 3, dtype=torch.bool),
                "terminated": torch.zeros(4, 3, dtype=torch.bool),
                "truncated": torch.zeros(4, 3, dtype=torch.bool),
            }
        )
        transform = SequenceDiscriminatorReward(
            discriminator, sequence_length=2, batch_size=2
        )

        transform(batch, None)

        self.assertTrue(all(size <= 2 for size in discriminator.batch_sizes))
        self.assertEqual(sum(discriminator.batch_sizes), 12)


if __name__ == "__main__":
    unittest.main()
