import torch
from carl.gymnasium.state import StepContext

from rewards.shaping import NextoReward


class GameplayDiagnostics:
    """Nexto rewards and the gameplay counters shown in training logs."""

    def __init__(
        self,
        n_sim:                  int,
        n_cars:                 int,
        device:                 torch.device,
        frameskip:              int,
        no_touch_timeout_steps: int = 0
    ) -> None:
        self.n_cars = n_cars
        self.frameskip = frameskip
        self.no_touch_timeout_steps = no_touch_timeout_steps
        self.reward = NextoReward(n_cars)
        self.touch_steps = torch.zeros(n_sim, dtype=torch.long, device=device)

        self.counts = {
            name: torch.zeros((), device=device)
            for name in (
                "steps",
                "touches",
                "aerial_touches",
                "goals_for",
                "episodes",
                "timeouts"
            )
        }

    @torch.no_grad()
    def __call__(self, context: StepContext) -> torch.Tensor:
        current = context.current
        touches = current.car_ball_touches
        score = context.events.score_delta[:, None] * current.team_sign

        self.touch_steps += 1
        self.touch_steps[touches.any(-1)] = 0
        timeout = context.events.truncated & (self.touch_steps >= self.no_touch_timeout_steps)
        self.touch_steps[context.events.done] = 0

        self.counts["steps"] += touches.numel()
        self.counts["touches"] += touches.sum()
        self.counts["aerial_touches"] += (
            touches & ~current.car_on_ground & current.car_position[..., 2].gt(182.5)
        ).sum()
        self.counts["goals_for"] += (score > 0).sum()
        self.counts["episodes"] += context.events.done.sum() * self.n_cars
        self.counts["timeouts"] += timeout.sum() * self.n_cars

        return self.reward(context)

    def diagnostic_metrics(self) -> dict[str, float]:
        metrics = {}
        steps = self.counts["steps"].item()

        if steps:
            ticks = steps * self.frameskip

            for name in ("touches", "aerial_touches", "goals_for"):
                metrics[f"gameplay/{name}_per_1000_ticks"] = self.counts[name].item() * 1000 / ticks

        episodes = self.counts["episodes"].item()

        if episodes:
            metrics["gameplay/timeout_fraction"] = self.counts["timeouts"].item() / episodes

        for value in self.counts.values():
            value.zero_()

        return metrics
