import torch
from carl.gymnasium.state import RewardContext

from rewards.events import advanced_touch_events
from rewards.shaping import RewardPlus


class GameplayDiagnostics:

    def __init__(
        self,
        n_sim: int,
        n_cars: int,
        device: torch.device,
        frameskip: int,
        reward_plus: bool = False,
        no_touch_timeout_steps: int = 0,
        goal_weight: float = 1.0
    ) -> None:
        self.n_cars = n_cars
        self.team_size = n_cars // 2
        self.no_touch_timeout_steps = no_touch_timeout_steps
        self.device = device
        self.goal_weight = goal_weight
        self.reward_plus = RewardPlus(n_sim, n_cars, device, frameskip) if reward_plus else None
        self.touch_steps = torch.zeros(n_sim, dtype=torch.long, device=device)
        self.component_sums = {}
        self.counts = {
            name: torch.zeros((), device=device)
            for name in (
                "steps",
                "touches",
                "aerial_touches",
                "flip_resets",
                "demos",
                "demoed",
                "goals_for",
                "goals_against",
                "episodes",
                "timeouts"
            )
        }

    def _team_difference(self, events: torch.Tensor) -> torch.Tensor:
        scores = events.reshape(-1, 2, self.team_size).sum(dim=-1, keepdim=True)

        return (scores - scores.flip(dims=(1,))).expand(-1, -1, self.team_size).reshape(-1)

    @torch.no_grad()
    def __call__(self, context: RewardContext) -> torch.Tensor:
        current, previous = context.current, context.previous
        touches = current.car_ball_touches
        score_for_actor = context.events.score_delta[:, None] * current.team_sign

        _, aerial, flip_reset = advanced_touch_events(context)
        if self.reward_plus is not None:
            components = self.reward_plus.components(context, aerial)
            for name, value in components.items():
                if name not in self.component_sums:
                    self.component_sums[name] = torch.zeros((), device=self.device)
                self.component_sums[name] += value.sum()

        demoed = current.car_demoed & ~previous.car_demoed

        self.touch_steps += 1
        self.touch_steps[touches.any(dim=-1)] = 0

        timeout = context.events.truncated & (self.touch_steps >= self.no_touch_timeout_steps)
        self.touch_steps[context.events.done] = 0

        self.counts["steps"] += touches.numel()
        self.counts["touches"] += touches.sum()
        self.counts["aerial_touches"] += aerial.sum()
        self.counts["flip_resets"] += flip_reset.sum()
        self.counts["demos"] += demoed.sum()
        self.counts["demoed"] += current.car_demoed.sum()
        self.counts["goals_for"] += (score_for_actor > 0).sum()
        self.counts["goals_against"] += (score_for_actor < 0).sum()
        self.counts["episodes"] += context.events.done.sum() * self.n_cars
        self.counts["timeouts"] += timeout.sum() * self.n_cars

        if self.reward_plus is None:
            return self.goal_weight * score_for_actor

        weights = self.reward_plus.weights
        kickoff = components["kickoff"] + components["kickoff_speed"]
        shaping = sum((value for name, value in components.items()
                       if name not in ("kickoff", "kickoff_speed")), torch.zeros_like(score_for_actor))
        demo_reward = self._team_difference(-demoed.float()).reshape_as(score_for_actor)
        goal_reward = weights.goal * score_for_actor
        flip_reward = weights.flip_reset * flip_reset.float()
        total = shaping + self._team_difference(kickoff).reshape_as(score_for_actor)

        for name, value in (("goal", goal_reward), ("flip_reset", flip_reward), ("demo", demo_reward * weights.demo)):
            if name not in self.component_sums:
                self.component_sums[name] = torch.zeros((), device=self.device)
            self.component_sums[name] += value.abs().sum()

        return total + goal_reward + flip_reward + weights.demo * demo_reward

    def diagnostic_metrics(self) -> dict[str, float]:
        metrics = {}
        steps = self.counts["steps"].item()

        if steps > 0:
            for name in (
                "touches",
                "aerial_touches",
                "flip_resets",
                "demos",
                "goals_for",
                "goals_against"
            ):
                metrics[f"gameplay/{name}_per_1000_steps"] = self.counts[name].item() / steps * 1000

            metrics["gameplay/demoed_fraction"] = self.counts["demoed"].item() / steps

            if self.reward_plus is not None:
                metrics.update(
                    {
                        f"reward_plus/{name}": total.item() / steps
                        for name, total in self.component_sums.items()
                    }
                )

                for total in self.component_sums.values():
                    total.zero_()

            for name in (
                "steps",
                "touches",
                "aerial_touches",
                "flip_resets",
                "demos",
                "demoed",
                "goals_for",
                "goals_against"
            ):
                self.counts[name].zero_()

        episodes = self.counts["episodes"].item()

        if episodes > 0:
            metrics["gameplay/timeout_fraction"] = self.counts["timeouts"].item() / episodes
            self.counts["episodes"].zero_()
            self.counts["timeouts"].zero_()

        return metrics
