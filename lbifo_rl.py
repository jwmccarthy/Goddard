"""Online, behavior-conditioned skill RL with a frozen-EMA tracking reward."""

import torch as th
import torch.nn as nn

from gaifo import N_CARS, SCENE_SIZE
from lbifo_repr import RelationalEncoder


class EmbeddingTrackingReward:
    """Compare each actor's realized request prefix to its target embedding.

    The two cars may change requests at different times, so their histories
    have independent starts even though each encoder input is a joint scene.
    Histories and potentials persist across collection-batch boundaries.
    """

    def __init__(
        self, encoder: RelationalEncoder, n_sim: int, max_duration: int,
        latent_dim: int, device: th.device, match_weight: float = 1.0,
        progress_weight: float = 1.0, batch_limit: int = 1024,
    ) -> None:
        if n_sim < 1 or max_duration < 2 or batch_limit < 1 or (
            match_weight < 0 or progress_weight < 0
        ):
            raise ValueError("invalid embedding tracking configuration")
        self.encoder = encoder
        self.n_sim = n_sim
        self.max_duration = max_duration
        self.latent_dim = latent_dim
        self.match_weight = match_weight
        self.progress_weight = progress_weight
        self.history = th.zeros(n_sim, N_CARS, max_duration + 1, SCENE_SIZE, device=device)
        self.previous = th.zeros(n_sim, N_CARS, device=device)
        self.batch_limit = batch_limit

    @th.no_grad()
    def _similarity(
        self, lengths: th.Tensor, requests: th.Tensor, active: th.Tensor,
    ) -> th.Tensor:
        cosine = self.previous.new_zeros(self.previous.shape)
        for length in lengths[active].unique().tolist():
            sims, cars = (active & (lengths == length)).nonzero(as_tuple=True)
            for start in range(0, len(sims), self.batch_limit):
                selected, entity = sims[start:start + self.batch_limit], cars[start:start + self.batch_limit]
                windows = self.history[selected, entity, :length]
                directions, _ = self.encoder(windows)
                achieved = directions[th.arange(len(selected), device=selected.device), -1, entity]
                cosine[selected, entity] = (achieved * requests[selected, entity]).sum(-1)
        return cosine

    @th.no_grad()
    def refresh(self, requests: th.Tensor, ages: th.Tensor, active: th.Tensor) -> None:
        """Refresh the old potential after the EMA changes between rollouts."""
        current = active & (ages > 0)
        self.previous = self._similarity(ages + 1, requests, current)

    @th.no_grad()
    def step(
        self, before: th.Tensor, after: th.Tensor, requests: th.Tensor,
        ages: th.Tensor, active: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        if before.shape != (self.n_sim, SCENE_SIZE) or after.shape != before.shape or (
            requests.shape != (self.n_sim, N_CARS, self.latent_dim)
            or ages.shape != (self.n_sim, N_CARS) or active.shape != ages.shape
        ):
            raise ValueError("tracking reward needs matching joint scenes, requests, and ages")
        if bool((active & (ages >= self.max_duration)).any()):
            raise ValueError("active behavior exceeded its maximum encoder duration")
        starting, car = (active & (ages == 0)).nonzero(as_tuple=True)
        self.history[starting, car, 0] = before[starting]
        sims, car = active.nonzero(as_tuple=True)
        self.history[sims, car, ages[sims, car] + 1] = after[sims]
        cosine = self._similarity(ages + 2, requests, active)
        progress = th.where(active & (ages > 0), cosine - self.previous, 0)
        reward = self.match_weight * cosine + self.progress_weight * progress
        self.previous = th.where(active, cosine, th.zeros_like(cosine))
        return reward, cosine, progress


class SkillCritic(nn.Module):
    """Value of tracking a requested behavior, separate from task-plan value."""

    def __init__(self, observation_size: int, latent_dim: int, hidden: int) -> None:
        super().__init__()
        self.observation_size = observation_size
        self.latent_dim = latent_dim
        self.network = nn.Sequential(
            nn.Linear(observation_size + latent_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def forward(self, observation: th.Tensor, latent: th.Tensor, age: th.Tensor) -> th.Tensor:
        if observation.shape[-1] != self.observation_size or (
            latent.shape != (*observation.shape[:-1], self.latent_dim)
            or age.shape != observation.shape[:-1]
        ):
            raise ValueError("skill critic needs matching observations, requests, and ages")
        inputs = th.cat((observation, latent, (age.float() / 32)[..., None]), dim=-1)
        return self.network(inputs).squeeze(-1)


def tracking_gae(
    reward: th.Tensor, value: th.Tensor, ended: th.Tensor,
    active: th.Tensor, bootstrap: th.Tensor, gamma: float, lambda_: float,
) -> tuple[th.Tensor, th.Tensor]:
    """Bootstrap unfinished skills; never bridge a request change or episode end."""
    if reward.shape != value.shape or ended.shape != reward.shape or active.shape != reward.shape or (
        bootstrap.shape != reward.shape[1:]
    ):
        raise ValueError("tracking GAE requires [time, simulation, entity] fields")
    if not 0 <= gamma <= 1 or not 0 <= lambda_ <= 1:
        raise ValueError("invalid tracking discount or GAE lambda")
    advantage = th.zeros_like(reward)
    next_advantage = th.zeros_like(bootstrap)
    next_value = bootstrap
    for tick in range(len(reward) - 1, -1, -1):
        continuation = (~ended[tick]).to(reward.dtype)
        delta = reward[tick] + gamma * continuation * next_value - value[tick]
        next_advantage = (delta + gamma * lambda_ * continuation * next_advantage) * active[tick]
        advantage[tick] = next_advantage
        next_value = value[tick]
    return advantage, advantage + value
