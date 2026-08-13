import torch as th
import torch.nn.functional as F

from torch.distributions import kl_divergence

from jarl.data.batch import TensorBatch
from jarl.learn import LossOutput, PPOLoss


class ASEReward:

    def __init__(
        self,
        discriminator,
        skill_encoder,
        beta:  float = 0.5,
        kappa: float = 1.0
    ) -> None:
        self.discriminator = discriminator
        self.skill_encoder = skill_encoder

        self.beta = beta
        self.kappa = kappa

    @th.no_grad()
    def __call__(self, batch: TensorBatch, context) -> TensorBatch:
        transition = (batch["observation"], batch["next_obs"])
        done = batch["terminated"] | batch["truncated"]
        reset = th.zeros_like(done)
        reset[1:] = done[:-1]

        imitation_reward = F.softplus(
            self.discriminator(transition, reset=reset)
        )
        skill_direction = self.skill_encoder(transition, reset=reset)
        skill_reward = self.kappa * (skill_direction * batch["latent"]).sum(-1)

        reward = imitation_reward + self.beta * skill_reward

        return batch.replace_fields(reward=reward).with_fields(
            imitation_reward=imitation_reward,
            skill_reward=skill_reward
        )


class DiscriminatorMinibatches:

    def __init__(
        self,
        expert_dataset,
        batch_size: int,
        epochs:     int = 1
    ) -> None:
        self.expert_dataset = expert_dataset
        self.batch_size = batch_size
        self.epochs = epochs

        self._epoch_callback = None

    def set_epoch_callback(self, callback) -> None:
        self._epoch_callback = callback

    def __call__(self, data: TensorBatch):
        agent = data.flatten(0, 1)
        learner = agent.get("learner_mask")

        if learner is not None:
            agent = agent[learner.bool()]

        for _ in range(self.epochs):
            order = th.randperm(len(agent), device=agent.device)

            for left in range(0, len(agent), self.batch_size):
                indices = order[left:left + self.batch_size]
                count = len(indices)

                if not count:
                    continue

                expert = self.expert_dataset.sample(count)["observation"]

                yield TensorBatch({
                    "agent_observation":  agent["observation"][indices],
                    "agent_next_obs":     agent["next_obs"][indices],
                    "expert_observation": expert[:, 0],
                    "expert_next_obs":    expert[:, 1]
                })

            if self._epoch_callback is not None:
                self._epoch_callback()


class RecurrentDiscriminatorMinibatches(DiscriminatorMinibatches):

    def __init__(self, expert_dataset, sequence_length, batch_size, epochs=1):
        super().__init__(expert_dataset, batch_size, epochs)
        self.sequence_length = sequence_length

    def __call__(self, data: TensorBatch):
        time, environments = data.shape[:2]
        chunks = time // self.sequence_length
        if chunks < 1:
            return

        agent = data["observation"][:chunks * self.sequence_length]
        next_obs = data["next_obs"][:chunks * self.sequence_length]
        done = (
            data["terminated"][:chunks * self.sequence_length]
            | data["truncated"][:chunks * self.sequence_length]
        )
        agent = agent.reshape(chunks, self.sequence_length, environments, -1)
        next_obs = next_obs.reshape(chunks, self.sequence_length, environments, -1)
        done = done.reshape(chunks, self.sequence_length, environments)
        agent = agent.swapaxes(1, 2).reshape(-1, self.sequence_length, agent.shape[-1])
        next_obs = next_obs.swapaxes(1, 2).reshape(-1, self.sequence_length, next_obs.shape[-1])
        done = done.swapaxes(1, 2).reshape(-1, self.sequence_length)

        for _ in range(self.epochs):
            order = th.randperm(len(agent), device=agent.device)
            for left in range(0, len(agent), self.batch_size):
                indices = order[left:left + self.batch_size]
                expert = self.expert_dataset.sample(
                    len(indices), self.sequence_length + 1
                )["observation"]
                agent_reset = th.zeros_like(done[indices])
                agent_reset[:, 1:] = done[indices, :-1]
                yield TensorBatch({
                    "agent_observation": agent[indices].swapaxes(0, 1),
                    "agent_next_obs": next_obs[indices].swapaxes(0, 1),
                    "agent_reset": agent_reset.swapaxes(0, 1),
                    "expert_observation": expert[:, :-1].swapaxes(0, 1),
                    "expert_next_obs": expert[:, 1:].swapaxes(0, 1),
                    "expert_reset": th.zeros_like(agent_reset).swapaxes(0, 1)
                })

            if self._epoch_callback is not None:
                self._epoch_callback()


class DiscriminatorLoss:

    def __init__(self, discriminator, gradient_penalty: float = 5.0) -> None:
        self.discriminator = discriminator
        self.gradient_penalty = gradient_penalty

    def after_update(self) -> None:
        return

    def __call__(self, batch: TensorBatch) -> LossOutput:
        expert_observation = batch["expert_observation"].detach().requires_grad_(True)
        expert_next_obs = batch["expert_next_obs"].detach().requires_grad_(True)

        if hasattr(self.discriminator.body, "initial_state"):
            expert_input = self.discriminator.expert_foot(
                (expert_observation, expert_next_obs)
            )
            expert_features, _ = self.discriminator.body(
                expert_input,
                reset=batch["expert_reset"]
            )
            expert_logits = self.discriminator.head(expert_features).squeeze(-1)
            noise_scale = 0.01
            noisy_features, _ = self.discriminator.body(
                expert_input + noise_scale * th.randn_like(expert_input),
                reset=batch["expert_reset"]
            )
            noisy_logits = self.discriminator.head(noisy_features).squeeze(-1)
            gradient_penalty = (
                (noisy_logits - expert_logits) / noise_scale
            ).pow(2).mean()
        else:
            expert_logits = self.discriminator.forward_expert(
                (expert_observation, expert_next_obs)
            )
            gradients = th.autograd.grad(
                expert_logits.sum(),
                (expert_observation, expert_next_obs),
                create_graph=True
            )
            gradient_penalty = sum(
                gradient.flatten(1).pow(2).sum(-1)
                for gradient in gradients
            ).mean()

        agent_logits = self.discriminator(
            (batch["agent_observation"], batch["agent_next_obs"]),
            reset=batch.get("agent_reset")
        )

        expert_loss = F.softplus(-expert_logits).mean()
        agent_loss = F.softplus(agent_logits).mean()

        loss = expert_loss + agent_loss + self.gradient_penalty * gradient_penalty

        return LossOutput(loss, {
            "loss":             loss,
            "expert_loss":      expert_loss,
            "agent_loss":       agent_loss,
            "gradient_penalty": gradient_penalty,
            "expert_accuracy":  expert_logits.gt(0).float().mean(),
            "agent_accuracy":   agent_logits.lt(0).float().mean()
        })


class SkillEncoderLoss:

    def __init__(self, skill_encoder, kappa: float = 1.0) -> None:
        self.skill_encoder = skill_encoder
        self.kappa = kappa

    def after_update(self) -> None:
        return

    def __call__(self, batch: TensorBatch) -> LossOutput:
        valid = None
        reset = None
        if hasattr(batch, "steps"):
            valid = batch.valid
            reset = batch.reset
            batch = batch.steps

        direction = self.skill_encoder(
            (batch["observation"], batch["next_obs"]),
            reset=reset
        )

        similarity = (direction * batch["latent"].detach()).sum(-1)
        if valid is not None:
            similarity = similarity[valid]
        loss = -self.kappa * similarity.mean()

        return LossOutput(loss, {
            "loss": loss,
            "cosine_similarity": similarity.mean()
        })


class ASEPPOLoss(PPOLoss):

    def __init__(self, *args, diversity: float = 0.01, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.diversity = diversity

    def _evaluate(self, batch, state, critic_state, reset):
        observation = (batch["observation"], batch["latent"])
        evaluation = self.policy.evaluate_actions(
            observation,
            batch["action"],
            state,
            reset=reset
        )

        value = self.critic.evaluate_values(
            observation,
            critic_state,
            reset=reset
        )
        return evaluation, value

    def __call__(self, sample) -> LossOutput:
        output = super().__call__(sample)

        batch, state, _, reset, valid = self._unpack_sample(sample)
        diversity_loss = self._diversity_loss(
            batch["observation"],
            state,
            reset,
            valid
        )

        loss = output.loss + self.diversity * diversity_loss

        return LossOutput(loss, output.metrics | {
            "diversity_loss": diversity_loss
        })

    def _diversity_loss(self, observation, state, reset, valid) -> th.Tensor:
        latent_dim = self.policy.foot.latents.latent_dim
        latent_a = F.normalize(
            th.randn(
                *observation.shape[:-1],
                latent_dim,
                device=observation.device
            ),
            dim=-1
        )

        latent_b = F.normalize(
            th.randn_like(latent_a),
            dim=-1
        )

        dist_a = self._dist(observation, latent_a, state, reset)
        dist_b = self._dist(observation, latent_b, state, reset)

        policy_distance = sum(
            kl_divergence(a, b).sum(-1)
            for (_, a), (_, b) in zip(dist_a, dist_b)
        )

        latent_distance = 0.5 * (1.0 - (latent_a * latent_b).sum(-1))

        loss = (
            (policy_distance / latent_distance.clamp_min(1e-4)) - 1.0
        ).pow(2)
        return loss[valid].mean()

    def _dist(self, observation, latent, state, reset):
        features, _ = self.policy.body_features(
            (observation, latent),
            state,
            reset
        )
        logits = self.policy.head(features)

        return self.policy._grouped_distributions(logits, observation)
