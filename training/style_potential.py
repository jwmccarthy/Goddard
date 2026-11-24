"""Expert style potential: shaping toward states where expert style is still missing.

The potential Phi(s) estimates the discounted forward-KL imitation reward that
expert play collects from state s onward, centered on the mean expert reward:

    Phi(s_t) ~ E_expert[ sum_k gamma^k (r(window ending at s_{t+k+1}) - mean) ]

The forward-KL reward is the capped expert/policy density ratio, so Phi is high
just before expert behavior the policy rarely produces (rare mechanics), and
near zero where experts do what the policy already does. Rollouts receive

    F = w * (gamma * Phi(s') - Phi(s)),  Phi(terminal) = 0

which is potential-based shaping (Ng, Harada & Russell 1999): it leaves the
optimal policy of the imitation objective unchanged and only speeds credit
assignment. Phi is refit periodically against the current target discriminator,
and one snapshot scores both s and s' of every transition in a rollout.
"""

from __future__ import annotations

import math
from argparse import Namespace
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

from replays import layout
from training.discriminator import (
    args_motion_noise,
    args_rewards,
    mirror_scene_pairs,
)
from training.tensors import autocast

if TYPE_CHECKING:
    from replays.dataset import ExpertDataset
    from training.imitation import TransitionPairs
    from training.rollout import Rollout

HIDDEN = 256
LEARNING_RATE = 1e-3
CHAINS = 4096
FIT_STEPS = 64
FIT_BATCH = 4096
EVAL_BATCH = 65_536


class StylePotential(nn.Module):
    """Scalar potential over one ego-perspective scene frame."""

    def __init__(self, scene_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(scene_size, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1)
        )
        # Start at Phi = 0 so the first rollouts are unshaped.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, scenes: torch.Tensor) -> torch.Tensor:
        return self.net(scenes).squeeze(-1)


class StylePotentialShaping:

    def __init__(
        self,
        args:    Namespace,
        experts: ExpertDataset,
        device:  torch.device
    ) -> None:
        self.args = args
        self.experts = experts
        self.n_cars = 2 * args.team_size
        self.scene_size = layout.team_scene_size(args.team_size)
        self.model = StylePotential(self.scene_size).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE
        )
        self.iteration = 0

        # Replay frames are 4 ticks apart; one policy decision spans several.
        self.decision = args.frameskip // layout.PARSED_FRAMESKIP
        valid = torch.zeros(
            len(experts.frames), dtype=torch.bool, device=device
        )
        valid[experts.ends] = True
        self.valid = valid
        self.generator = torch.Generator(device=device).manual_seed(
            args.seed + 7
        )

    @torch.no_grad()
    def potential(self, scenes: torch.Tensor) -> torch.Tensor:
        shape = scenes.shape[:-1]
        flat = scenes.reshape(-1, scenes.shape[-1])[:, : self.scene_size]
        values = [
            self.model(batch).float() for batch in flat.split(EVAL_BATCH)
        ]

        return torch.cat(values).view(shape)

    @torch.no_grad()
    def expert_targets(
        self,
        reward_model: nn.Module,
        gamma:        float
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Bootstrapped H-step centered expert returns from sampled states."""

        horizon = self.args.style_potential_horizon
        experts = self.experts
        starts = experts.ends[
            torch.randint(
                len(experts.ends), (CHAINS,),
                device=experts.ends.device, generator=self.generator
            )
        ]

        # Transition k scores the window ending at s_{t+k+1}.
        steps = torch.arange(1, horizon + 1, device=starts.device)
        ends = starts[:, None] + steps * self.decision
        inside = ends < len(experts.frames)
        ends = ends.clamp_max(len(experts.frames) - 1)
        alive = (self.valid[ends] & inside).long().cumprod(1).bool()

        rewards = torch.zeros(ends.shape, device=starts.device)
        live = ends[alive]
        values = []
        size = self.args.discriminator_inference_microbatch_size

        for batch in live.split(size):
            windows = args_motion_noise(
                experts.frames[batch[:, None] + experts.offsets], self.args
            )

            with autocast(windows.device, self.args.bf16):
                logits = reward_model(windows)

            values.append(args_rewards(logits, self.args).float())

        raw = torch.cat(values)
        mean = raw.mean()
        rewards[alive] = raw - mean

        discounts = gamma ** torch.arange(
            horizon, device=starts.device, dtype=torch.float32
        )
        targets = (rewards * discounts).sum(1)

        # Bootstrap on the current potential at the last state reached. Most
        # chain breaks are reconstruction gaps rather than episode ends, so a
        # break truncates the return instead of terminating it.
        reached = alive.sum(1)
        complete = reached == horizon
        bootstrapped = reached > 0
        if bootstrapped.any():
            tail = experts.frames[
                ends[bootstrapped, reached[bootstrapped] - 1], : self.scene_size
            ]
            targets[bootstrapped] += (
                gamma ** reached[bootstrapped].float() * self.potential(tail)
            )

        states = experts.frames[starts, : self.scene_size]
        fraction = complete.float().mean()
        decisions = reached.float().mean()
        metrics = {
            "style_potential/expert_reward_mean":   mean.item(),
            "style_potential/expert_reward_std":    raw.std().item(),
            "style_potential/target_mean":          targets.mean().item(),
            "style_potential/target_std":           targets.std().item(),
            "style_potential/complete_fraction":    fraction.item(),
            "style_potential/mean_chain_decisions": decisions.item()
        }

        return states, targets, metrics

    def fit(
        self,
        states:  torch.Tensor,
        targets: torch.Tensor
    ) -> float:
        total = torch.zeros((), device=states.device)
        # Squared error keeps Phi an expected return: a median-type loss
        # would discount exactly the rare high-ratio windows Phi exists to
        # find, and bootstrapping compounds that bias. Standardizing only
        # conditions the optimization.
        scale = targets.std().clamp_min(1e-6)

        for _ in range(FIT_STEPS):
            index = torch.randint(
                len(states), (min(FIT_BATCH, len(states)),),
                device=states.device, generator=self.generator
            )
            batch = states[index]

            if self.args.discriminator_mirror:
                batch = mirror_scene_pairs(batch[:, None], self.n_cars)[:, 0]

            loss = F.mse_loss(
                self.model(batch) / scale, targets[index] / scale
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.detach()

        return (total / FIT_STEPS).item()

    def refresh(
        self,
        reward_model: nn.Module,
        gamma:        float
    ) -> dict[str, float]:
        # Called on the eager module, never the compiled wrapper, and without
        # touching its train/eval flag, so compiled guards are unaffected.
        states, targets, metrics = self.expert_targets(reward_model, gamma)
        metrics["style_potential/fit_loss"] = self.fit(states, targets)

        return metrics

    @torch.no_grad()
    def shape(
        self,
        scenes:  TransitionPairs,
        rollout: Rollout,
        gamma:   float
    ) -> dict[str, float]:
        current = self.potential(scenes.scenes)
        following = self.potential(scenes.successors).masked_fill(
            scenes.terminated, 0
        )
        weight = (
            self.args.style_potential_weight * self.args.imitation_reward_weight
        )
        shaping = weight * (gamma * following - current)
        rollout.rewards.add_(shaping.view_as(rollout.rewards))

        return {
            "style_potential/policy_potential_mean": current.mean().item(),
            "style_potential/shaping_mean":          shaping.mean().item(),
            "style_potential/shaping_std":           shaping.std().item()
        }

    def update(
        self,
        reward_model: nn.Module,
        scenes:       TransitionPairs,
        rollout:      Rollout,
        gamma:        float | torch.Tensor
    ) -> dict[str, float]:
        gamma = float(gamma)
        metrics = {}

        # Refit before shaping, so one snapshot covers this whole rollout.
        if self.iteration % self.args.style_potential_interval == 0:
            metrics.update(self.refresh(reward_model, gamma))

        self.iteration += 1
        metrics.update(self.shape(scenes, rollout, gamma))

        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iteration,
            "generator": self.generator.get_state()
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.iteration = state["iteration"]
        self.generator.set_state(state["generator"])


def check_style_potential_args(args: Namespace) -> None:
    weight = args.style_potential_weight

    if not math.isfinite(weight) or weight < 0:
        raise ValueError("style potential weight must be finite and nonnegative")

    if args.style_potential_interval < 1:
        raise ValueError("style potential interval must be positive")

    if args.style_potential_horizon < 1:
        raise ValueError("style potential horizon must be positive")

    if args.frameskip % layout.PARSED_FRAMESKIP:
        raise ValueError("style potential needs a frameskip on the replay cadence")

    if args.discriminator_jump:
        raise ValueError("style potential does not support jump-conditioned windows")
