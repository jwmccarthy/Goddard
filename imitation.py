import torch
import torch.nn as nn
import torch.nn.functional as F

from jarl.data import TensorBatch
from jarl.learn import LossOutput


def _sequence_grid(value: torch.Tensor, sequence_length: int) -> torch.Tensor:
    horizon, num_envs = value.shape[:2]
    if horizon % sequence_length:
        raise ValueError("rollout horizon must be divisible by sequence length")
    return value.reshape(
        horizon // sequence_length,
        sequence_length,
        num_envs,
        *value.shape[2:],
    )


def _sequence_chunks(value: torch.Tensor, sequence_length: int) -> torch.Tensor:
    grid = _sequence_grid(value, sequence_length)
    return (
        grid
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
        self.recurrent = nn.GRU(41, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def project(self, observation: torch.Tensor) -> torch.Tensor:
        ball = observation[..., :9]
        own_car = observation[..., 9:25]
        opponent = observation[..., 30:46]
        return torch.cat((ball, own_car, opponent), dim=-1)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        observation = self.project(observation)
        if self.training and self.noise_std:
            observation = observation + torch.randn_like(observation) * self.noise_std
        _, final_state = self.recurrent(observation)
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
        observation = _sequence_grid(rollout["observation"], self.sequence_length)
        valid = torch.ones(
            observation.shape[0],
            observation.shape[2],
            dtype=torch.bool,
            device=rollout.device,
        )
        if "learner_mask" in rollout:
            valid &= _sequence_grid(
                rollout["learner_mask"], self.sequence_length
            ).bool().all(dim=1)
        for field in ("terminated", "truncated"):
            if field in rollout:
                valid &= ~_sequence_grid(
                    rollout[field], self.sequence_length
                ).bool().any(dim=1)

        coordinates = valid.nonzero()
        if len(coordinates) < self.batch_size:
            raise RuntimeError(
                "not enough valid rollout sequences for a discriminator minibatch"
            )

        for _ in range(self.epochs):
            indices = torch.randperm(len(coordinates), device=rollout.device)
            for start in range(0, len(coordinates), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                if len(batch_indices) == self.batch_size:
                    selected = coordinates[batch_indices]
                    chunk, environment = selected.unbind(dim=1)
                    agent_sequences = TensorBatch(
                        {
                            "observation": observation[chunk, :, environment],
                        }
                    )
                    yield self._build_batch(agent_sequences)

    def _build_batch(self, agent_sequences: TensorBatch) -> TensorBatch:
        expert = self.expert_dataset.sample(self.batch_size)["observation"]
        history_length = expert.shape[1]
        if history_length < self.sequence_length:
            raise ValueError("expert history is shorter than rollout sequences")
        if history_length > self.sequence_length:
            starts = torch.randint(
                history_length - self.sequence_length + 1,
                (self.batch_size,),
                device=expert.device,
            )
            offsets = torch.arange(self.sequence_length, device=expert.device)
            expert = expert[
                torch.arange(self.batch_size, device=expert.device)[:, None],
                starts[:, None] + offsets,
            ]
        expert = expert.to(agent_sequences.device)
        return TensorBatch(
            {
                "observation": torch.cat(
                    (agent_sequences["observation"], expert)
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
        score = self.discriminator(batch["observation"])
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
        batch_size: int = 4096,
        output_field: str = "imitation_reward",
    ) -> None:
        if sequence_length < 1 or batch_size < 1:
            raise ValueError("sequence length and batch size must be positive")
        self.discriminator = discriminator
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.output_field = output_field

    @torch.no_grad()
    def __call__(self, batch: TensorBatch, context) -> TensorBatch:
        observation = _sequence_grid(batch["observation"], self.sequence_length)
        valid = torch.ones(
            observation.shape[0],
            observation.shape[2],
            dtype=torch.bool,
            device=batch.device,
        )
        if "learner_mask" in batch:
            valid &= _sequence_grid(
                batch["learner_mask"], self.sequence_length
            ).bool().all(dim=1)
        for field in ("terminated", "truncated"):
            if field in batch:
                valid &= ~_sequence_grid(
                    batch[field], self.sequence_length
                ).bool().any(dim=1)

        sequence_reward = torch.zeros(
            valid.shape,
            dtype=batch["observation"].dtype,
            device=batch.device,
        )
        coordinates = valid.nonzero()
        was_training = self.discriminator.training
        self.discriminator.eval()
        try:
            for start in range(0, len(coordinates), self.batch_size):
                selected = coordinates[start : start + self.batch_size]
                chunk, environment = selected.unbind(dim=1)
                score = self.discriminator(observation[chunk, :, environment])
                sequence_reward[chunk, environment] = F.softplus(-score)
        finally:
            self.discriminator.train(was_training)

        horizon, num_envs = batch["observation"].shape[:2]
        reward = torch.zeros(
            (horizon, num_envs),
            dtype=sequence_reward.dtype,
            device=sequence_reward.device,
        )
        reward.reshape(-1, self.sequence_length, num_envs)[:, -1] = sequence_reward
        return batch.with_fields(**{self.output_field: reward})


class AddImitationReward:
    def __init__(self, imitation_field: str = "imitation_reward") -> None:
        self.imitation_field = imitation_field

    def __call__(self, batch: TensorBatch, context) -> TensorBatch:
        return batch.replace_fields(
            reward=batch["reward"] + batch[self.imitation_field]
        )


class EveryNUpdates:
    def __init__(self, stage, interval: int) -> None:
        if interval < 1:
            raise ValueError("update interval must be positive")
        self.stage = stage
        self.interval = interval
        self.update_count = 0

    def run(self, experience):
        should_run = self.update_count % self.interval == 0
        self.update_count += 1
        if should_run:
            return self.stage.run(experience)
        return experience, {}
