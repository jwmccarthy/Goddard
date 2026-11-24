"""Episode-persistent historical opponents; only live-policy cars train PPO."""

from __future__ import annotations

import copy
from argparse import Namespace
from typing import TYPE_CHECKING

import torch

from training.residual import ResidualActor, ResidualOpponents

if TYPE_CHECKING:
    from training.models import Agent


class OpponentPool:

    def __init__(
        self,
        agent:  Agent,
        args:   Namespace,
        device: torch.device | str,
        step:   int = 0
    ) -> None:
        self.agent = agent
        self.capacity = args.opponent_pool_size
        self.interval = args.opponent_snapshot_interval
        self.fraction = args.opponent_historical_fraction
        self.team_size = args.team_size
        self.n_cars = 2 * self.team_size
        self.device = torch.device(device)
        self.residual = (
            ResidualOpponents(agent.actor, args)
            if isinstance(agent.actor, ResidualActor) else None
        )
        self.policy_state = (
            torch.zeros(
                args.n_sim * self.n_cars,
                args.policy_hidden_size,
                device=self.device
            )
            if agent.gru_policy
            else None
        )
        self.generator = torch.Generator(device=self.device).manual_seed(args.seed + 9173)
        self.assignments = torch.full(
            (args.n_sim,),
            -1,
            dtype=torch.long,
            device=self.device
        )

        self.learner_sides = torch.zeros(
            args.n_sim,
            dtype=torch.long,
            device=self.device
        )

        self.snapshots = {}
        self.published = []
        self.next_id = 0
        self.last_snapshot_step = step
        self.add_snapshot(step)

    def add_snapshot(self, step: int) -> None:
        actor = self.copy_actor()
        identity = self.next_id
        self.next_id += 1
        self.snapshots[identity] = {
            "actor": actor,
            "step":  step
        }

        self.published.append(identity)
        self.published = self.published[-self.capacity :]
        self.last_snapshot_step = step
        self.prune_snapshots()
        self.refresh_residual()

    def prune_snapshots(self) -> bool:
        # Removing a snapshot from future matchmaking must never change an
        # opponent in an unfinished episode. Release retired models only when
        # no simulation still references them.
        retired = set(self.snapshots) - set(self.published)

        if not retired:
            return False

        occupied = set(self.assignments.cpu().tolist())
        unused = retired - occupied

        for old in unused:
            del self.snapshots[old]

        return bool(unused)

    def copy_actor(self) -> torch.nn.Module:
        # Residual snapshots differ only in their correction weights.
        memo = (
            {id(self.agent.actor.base): self.agent.actor.base}
            if self.residual is not None else None
        )

        return copy.deepcopy(self.agent.actor, memo).eval().requires_grad_(False)

    def refresh_residual(self) -> None:
        if self.residual is not None:
            self.residual.refresh({
                identity: snapshot["actor"]
                for identity, snapshot in self.snapshots.items()
            })

    def maybe_snapshot(self, step: int) -> None:
        if step - self.last_snapshot_step >= self.interval:
            self.add_snapshot(step)
        elif self.prune_snapshots():
            self.refresh_residual()

    @torch.no_grad()
    def resample(self, reset_mask: torch.Tensor) -> None:
        simulations = reset_mask.nonzero().flatten()
        count = len(simulations)

        if not count:
            return

        if self.policy_state is not None:
            self.policy_state.view(len(reset_mask), self.n_cars, -1)[
                simulations
            ] = 0

        historical = (
            torch.rand(count, device=self.device, generator=self.generator)
            < self.fraction
        )

        choices = torch.randint(
            len(self.published), (count,),
            device=self.device, generator=self.generator
        )

        identities = torch.tensor(self.published, device=self.device)[choices]
        self.assignments[simulations] = torch.where(historical, identities, -1)

        self.learner_sides[simulations] = torch.randint(
            2, (count,), device=self.device, generator=self.generator
        )

    def learner_mask(self) -> torch.Tensor:
        teams = torch.arange(self.n_cars, device=self.device) // self.team_size
        current = teams[None] == self.learner_sides[:, None]

        return ((self.assignments[:, None] < 0) | current).flatten()

    @torch.no_grad()
    def apply_actions(
        self,
        observations: torch.Tensor,
        actions:      torch.Tensor
    ) -> None:
        opponents = ~self.learner_mask()
        identities = self.assignments.repeat_interleave(self.n_cars)

        if self.residual is not None:
            indices = opponents.nonzero().flatten()

            if not len(indices):
                return

            batch = observations[indices]
            state = (
                self.policy_state[indices]
                if self.policy_state is not None else None
            )
            logits, state = self.residual(batch, identities[indices], state)

            if self.policy_state is not None:
                self.policy_state[indices] = state

            actions[indices] = self.agent.sample_action_logits(logits, batch)[0]

            return

        for identity, snapshot in self.snapshots.items():
            indices = ((identities == identity) & opponents).nonzero().flatten()

            if not len(indices):
                continue

            batch = observations[indices]

            if self.policy_state is None:
                logits = snapshot["actor"](batch)
            else:
                logits, state = snapshot["actor"](
                    batch, self.policy_state[indices]
                )
                self.policy_state[indices] = state

            actions[indices] = self.agent.sample_action_logits(
                logits.float(), batch
            )[0]

    def state_dict(self) -> dict[str, object]:
        # Simulator episodes are not checkpointed. Resume starts fresh episodes,
        # so only published snapshots (not retired in-flight models) are needed.
        return {
            "format":             "historical-opponents-v1",
            "next_id":            self.next_id,
            "last_snapshot_step": self.last_snapshot_step,
            "rng":                self.generator.get_state(),
            "snapshots":          [
                {
                    "id":    identity,
                    "step":  self.snapshots[identity]["step"],
                    "actor": {
                        key: value.detach().cpu().clone()
                        for key, value in self.snapshots[identity]["actor"].state_dict().items()
                    }
                }
                for identity in self.published
            ]
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        snapshots = state["snapshots"][-self.capacity :]

        if self.residual is not None:
            base = {
                f"base.{key}": value.detach().cpu()
                for key, value in self.agent.actor.base.state_dict().items()
            }

            for snapshot in snapshots:
                saved = {
                    key: value for key, value in snapshot["actor"].items()
                    if key.startswith("base.")
                }

                if saved.keys() != base.keys() or any(
                    not torch.equal(saved[key].cpu(), value)
                    for key, value in base.items()
                ):
                    raise ValueError("opponent snapshot has a different base")

        self.snapshots.clear()
        self.published.clear()

        for snapshot in snapshots:
            actor = self.copy_actor()

            if self.residual is not None:
                correction = {
                    key.removeprefix("correction."): value
                    for key, value in snapshot["actor"].items()
                    if key.startswith("correction.")
                }
                actor.correction.load_state_dict(correction, strict=True)
            else:
                actor.load_state_dict(snapshot["actor"], strict=True)

            identity = snapshot["id"]
            self.snapshots[identity] = {
                "actor": actor,
                "step":  snapshot["step"]
            }

            self.published.append(identity)

        self.next_id = state["next_id"]

        if self.next_id <= max(self.published):
            raise ValueError("opponent pool snapshot IDs are inconsistent")

        self.last_snapshot_step = state["last_snapshot_step"]
        self.generator.set_state(state["rng"])
        self.assignments.fill_(-1)

        if self.policy_state is not None:
            self.policy_state.zero_()

        self.refresh_residual()

    def diagnostic_metrics(self) -> dict[str, int | float]:
        return {
            "self_play/pool_size":                 len(self.published),
            "self_play/resident_snapshots":        len(self.snapshots),
            "self_play/historical_match_fraction": (self.assignments >= 0).float().mean().item(),
            "self_play/latest_snapshot_step":      self.last_snapshot_step
        }
