import torch
import torch.nn as nn
import torch.nn.functional as F

from jarl.data import TensorBatch
from jarl.learn import LossOutput


def _sequence_chunks(value: torch.Tensor, sequence_length: int) -> torch.Tensor:
    horizon, num_envs = value.shape[:2]
    if horizon % sequence_length:
        raise ValueError("rollout horizon must be divisible by sequence length")
    return (
        value.reshape(
            horizon // sequence_length,
            sequence_length,
            num_envs,
            *value.shape[2:],
        )
        .transpose(1, 2)
        .reshape(-1, sequence_length, *value.shape[2:])
    )


class SequenceDiscriminator(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        noise_std: float = 0.01,
    ) -> None:
        super().__init__()
        if noise_std < 0:
            raise ValueError("noise standard deviation cannot be negative")
        self.noise_std = noise_std
        self.recurrent = nn.GRU(82, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def project(self, observation: torch.Tensor) -> torch.Tensor:
        ball = observation[..., :9]
        own_car = observation[..., 9:25]
        opponent = observation[..., 30:46]
        return torch.cat((ball, own_car, opponent), dim=-1)

    def forward(self, sequence) -> torch.Tensor:
        observation, next_observation = sequence
        observation = self.project(observation)
        next_observation = self.project(next_observation)
        if self.training and self.noise_std:
            observation = observation + torch.randn_like(observation) * self.noise_std
            next_observation = (
                next_observation + torch.randn_like(next_observation) * self.noise_std
            )
        transitions = torch.cat((observation, next_observation), dim=-1)
        _, final_state = self.recurrent(transitions)
        return self.output(final_state[-1]).squeeze(-1)


class SequenceGAIFOMinibatches:
    def __init__(
        self,
        expert_dataset,
        sequence_length: int,
        batch_size: int,
        epochs: int = 1,
    ) -> None:
        if min(sequence_length, batch_size, epochs) < 1:
            raise ValueError("sequence length, batch size, and epochs must be positive")
        self.expert_dataset = expert_dataset
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs

    def __call__(self, rollout: TensorBatch):
        observation = _sequence_chunks(
            rollout["observation"], self.sequence_length
        )
        next_obs = _sequence_chunks(rollout["next_obs"], self.sequence_length)
        valid = torch.ones(len(observation), dtype=torch.bool, device=rollout.device)
        if "learner_mask" in rollout:
            valid &= _sequence_chunks(
                rollout["learner_mask"], self.sequence_length
            ).bool().all(dim=1)
        for field in ("terminated", "truncated"):
            if field in rollout:
                valid &= ~_sequence_chunks(
                    rollout[field], self.sequence_length
                ).bool().any(dim=1)

        agent_sequences = TensorBatch(
            {"observation": observation[valid], "next_obs": next_obs[valid]}
        )
        if len(agent_sequences) < self.batch_size:
            raise RuntimeError(
                "not enough valid rollout sequences for a discriminator minibatch"
            )

        for _ in range(self.epochs):
            indices = torch.randperm(len(agent_sequences), device=rollout.device)
            for start in range(0, len(agent_sequences), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                if len(batch_indices) == self.batch_size:
                    yield self._build_batch(agent_sequences[batch_indices])

    def _build_batch(self, agent_sequences: TensorBatch) -> TensorBatch:
        expert = self.expert_dataset.sample(self.batch_size).to(agent_sequences.device)
        return TensorBatch(
            {
                "observation": torch.cat(
                    (agent_sequences["observation"], expert["observation"])
                ),
                "next_obs": torch.cat(
                    (agent_sequences["next_obs"], expert["next_obs"])
                ),
                "is_agent": torch.cat(
                    (
                        torch.ones(self.batch_size, device=agent_sequences.device),
                        torch.zeros(self.batch_size, device=agent_sequences.device),
                    )
                ),
            }
        )


class SequenceGAIFOLoss:
    def __init__(self, discriminator: SequenceDiscriminator) -> None:
        self.discriminator = discriminator

    def __call__(self, batch: TensorBatch) -> LossOutput:
        score = self.discriminator((batch["observation"], batch["next_obs"]))
        target = batch["is_agent"]
        loss = F.binary_cross_entropy_with_logits(score, target)
        is_agent = target.bool()
        return LossOutput(
            loss,
            {
                "loss": loss.item(),
                "agent_score": score[is_agent].mean().item(),
                "expert_score": score[~is_agent].mean().item(),
            },
        )


class SequenceDiscriminatorReward:
    def __init__(
        self,
        discriminator: SequenceDiscriminator,
        sequence_length: int,
        output_field: str = "imitation_reward",
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence length must be positive")
        self.discriminator = discriminator
        self.sequence_length = sequence_length
        self.output_field = output_field

    @torch.no_grad()
    def __call__(self, batch: TensorBatch, context) -> TensorBatch:
        observation = _sequence_chunks(batch["observation"], self.sequence_length)
        next_obs = _sequence_chunks(batch["next_obs"], self.sequence_length)
        was_training = self.discriminator.training
        self.discriminator.eval()
        try:
            sequence_reward = -self.discriminator((observation, next_obs))
        finally:
            self.discriminator.train(was_training)

        valid = torch.ones_like(sequence_reward, dtype=torch.bool)
        if "learner_mask" in batch:
            valid &= _sequence_chunks(
                batch["learner_mask"], self.sequence_length
            ).bool().all(dim=1)
        for field in ("terminated", "truncated"):
            if field in batch:
                valid &= ~_sequence_chunks(
                    batch[field], self.sequence_length
                ).bool().any(dim=1)
        sequence_reward = sequence_reward.masked_fill(~valid, 0.0)

        horizon, num_envs = batch["observation"].shape[:2]
        reward = torch.zeros(
            (horizon // self.sequence_length, num_envs, self.sequence_length),
            dtype=sequence_reward.dtype,
            device=sequence_reward.device,
        )
        reward[..., -1] = sequence_reward.reshape(
            horizon // self.sequence_length, num_envs
        )
        reward = reward.transpose(1, 2).reshape(horizon, num_envs)
        return batch.with_fields(**{self.output_field: reward})
