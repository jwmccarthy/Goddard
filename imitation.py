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
        self.recurrent = nn.GRU(53, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def project(self, observation: torch.Tensor) -> torch.Tensor:
        ball = observation[..., :9]
        own_car = observation[..., 9:25]
        opponent = observation[..., 30:46]
        own_to_ball = observation[..., 119:122]
        own_to_opponent = observation[..., 125:128]
        ball_to_goals = observation[..., 131:137]
        return torch.cat(
            (
                ball,
                own_car,
                opponent,
                own_to_ball,
                own_to_opponent,
                ball_to_goals,
            ),
            dim=-1,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.forward_steps(observation)[..., -1]

    def forward_steps(self, observation: torch.Tensor) -> torch.Tensor:
        observation = self.project(observation)
        if self.training and self.noise_std:
            observation = observation + torch.randn_like(observation) * self.noise_std
        output, _ = self.recurrent(observation)
        return self.output(output).squeeze(-1)


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
        horizon, num_envs = rollout["observation"].shape[:2]
        window_count = horizon - self.sequence_length + 1
        if window_count < 1:
            raise ValueError("rollout horizon must be at least the sequence length")
        valid = torch.ones(
            window_count,
            num_envs,
            dtype=torch.bool,
            device=rollout.device,
        )
        for start in range(window_count):
            stop = start + self.sequence_length
            if "learner_mask" in rollout:
                valid[start] &= rollout["learner_mask"][start:stop].bool().all(dim=0)
            for field in ("terminated", "truncated"):
                if field in rollout:
                    valid[start] &= ~rollout[field][start:stop].bool().any(dim=0)

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
                    start_index, environment = selected.unbind(dim=1)
                    offsets = torch.arange(
                        self.sequence_length, device=rollout.device
                    )
                    agent_sequences = TensorBatch(
                        {
                            "observation": rollout["observation"][
                                start_index[:, None] + offsets,
                                environment[:, None],
                            ],
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
    def __init__(
        self,
        discriminator: SequenceDiscriminator,
        label_smoothing: float = 0.1,
    ) -> None:
        if not 0.0 <= label_smoothing < 0.5:
            raise ValueError("label smoothing must be in [0, 0.5)")
        self.discriminator = discriminator
        self.label_smoothing = label_smoothing

    def __call__(self, batch: TensorBatch) -> LossOutput:
        score = self.discriminator.forward_steps(batch["observation"])
        target = batch["is_agent"]
        smoothed_target = target.lerp(
            torch.full_like(target, 0.5),
            2.0 * self.label_smoothing,
        )
        smoothed_target = smoothed_target[:, None].expand_as(score)
        loss = F.binary_cross_entropy_with_logits(score, smoothed_target)
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
        reward_scale: float = 0.7,
        logit_clip: float = 10.0,
    ) -> None:
        if min(sequence_length, batch_size, reward_scale, logit_clip) <= 0:
            raise ValueError("lengths, batch size, and reward scales must be positive")
        self.discriminator = discriminator
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.output_field = output_field
        self.reward_scale = reward_scale
        self.logit_clip = logit_clip

    @torch.no_grad()
    def __call__(self, batch: TensorBatch, context) -> TensorBatch:
        reward = torch.zeros(
            batch["observation"].shape[:2],
            dtype=batch["observation"].dtype,
            device=batch.device,
        )
        horizon, num_envs = reward.shape
        age = torch.zeros(num_envs, dtype=torch.int64, device=batch.device)
        ages = torch.empty_like(reward, dtype=torch.int64)
        valid = torch.ones_like(reward, dtype=torch.bool)
        boundary = torch.zeros_like(reward, dtype=torch.bool)
        for field in ("terminated", "truncated"):
            if field in batch:
                boundary |= batch[field].bool()
        if "learner_mask" in batch:
            valid &= batch["learner_mask"].bool()
        valid &= ~boundary
        for step in range(horizon):
            if step:
                age = torch.where(boundary[step - 1], 0, age)
            age = (age + 1).clamp_max(self.sequence_length)
            ages[step] = age

        was_training = self.discriminator.training
        self.discriminator.eval()
        try:
            for length in range(1, self.sequence_length + 1):
                coordinates = (valid & ages.eq(length)).nonzero()
                offsets = torch.arange(length, device=batch.device)
                for start in range(0, len(coordinates), self.batch_size):
                    selected = coordinates[start : start + self.batch_size]
                    step, environment = selected.unbind(dim=1)
                    observation = batch["observation"][
                        step[:, None] - length + 1 + offsets,
                        environment[:, None],
                    ]
                    score = self.discriminator(observation).clamp(
                        -self.logit_clip, self.logit_clip
                    )
                    reward[step, environment] = (
                        self.reward_scale / self.sequence_length
                    ) * (1.0 - score / self.logit_clip)
        finally:
            self.discriminator.train(was_training)

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

    def state_dict(self) -> dict[str, int]:
        return {"update_count": self.update_count}

    def load_state_dict(self, state: dict[str, int]) -> None:
        update_count = state.get("update_count")
        if not isinstance(update_count, int) or update_count < 0:
            raise ValueError("invalid periodic update count")
        self.update_count = update_count
