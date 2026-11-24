from __future__ import annotations

import torch
from carl.gymnasium.state import StepContext

from rewards.shaping import NextoReward

TOUCH_WINDOW_TICKS = 8
CROSSBAR_HEIGHT = 642.775


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
        self.fixed_touch_windows = frameskip in (1, 2, 4, 8)
        self.window_ticks = torch.zeros(
            n_sim, dtype=torch.long, device=device
        )
        self.window_touches = torch.zeros(
            (n_sim, n_cars), dtype=torch.bool, device=device
        )

        self.counts = {
            name: torch.zeros((), device=device)
            for name in (
                "steps",
                "touches",
                "aerial_touches",
                "touch_windows",
                "low_aerial_touches",
                "high_aerial_touches",
                "goals_for",
                "episodes",
                "timeouts"
            )
        }

    @torch.no_grad()
    def __call__(self, context: StepContext) -> torch.Tensor:
        current = context.current
        touches = current.car_ball_touches
        aerial = (
            ~current.car_on_ground
            & current.car_position[..., 2].gt(182.5)
        )
        score = context.events.score_delta[:, None] * current.team_sign

        self.touch_steps += 1
        self.touch_steps[touches.any(-1)] = 0

        timeout = (
            context.events.truncated
            & (self.touch_steps >= self.no_touch_timeout_steps)
        )

        self.touch_steps[context.events.done] = 0
        self.counts["steps"] += touches.numel()

        self.track_touch_windows(context, aerial)

        self.counts["goals_for"] += (score > 0).sum()
        self.counts["episodes"] += context.events.done.sum() * self.n_cars
        self.counts["timeouts"] += timeout.sum() * self.n_cars

        return self.reward(context)

    def track_touch_windows(
        self,
        context: StepContext,
        aerial:  torch.Tensor
    ) -> None:
        """Count touches per fixed window."""
        if not self.fixed_touch_windows:
            return

        current = context.current

        self.window_ticks.add_(self.frameskip)
        self.window_touches.logical_or_(current.car_ball_touches)

        complete = self.window_ticks == TOUCH_WINDOW_TICKS
        contacted = self.window_touches & complete[:, None]

        self.counts["touch_windows"] += complete.sum() * self.n_cars
        self.counts["touches"] += contacted.sum()

        # Match FS8: contact anywhere in the window, aerial at its end.
        aerial_contact = contacted & aerial
        high = current.car_position[..., 2].gt(CROSSBAR_HEIGHT)

        self.counts["aerial_touches"] += aerial_contact.sum()
        self.counts["low_aerial_touches"] += (
            aerial_contact & ~high
        ).sum()
        self.counts["high_aerial_touches"] += (
            aerial_contact & high
        ).sum()

        # Count complete terminal windows using the final state. Discard
        # incomplete ones so contacts never leak into a new episode.
        clear = complete | context.events.done

        self.window_ticks.masked_fill_(clear, 0)
        self.window_touches.masked_fill_(clear[:, None], False)

    def diagnostic_metrics(self) -> dict[str, float]:
        metrics = {}
        steps = self.counts["steps"].item()

        if steps:
            ticks = steps * self.frameskip

            metrics["gameplay/goals_for_per_1000_ticks"] = (
                self.counts["goals_for"].item() * 1000 / ticks
            )

        windows = self.counts["touch_windows"].item()

        if windows:
            ticks = windows * TOUCH_WINDOW_TICKS

            for name in (
                "touches", "aerial_touches",
                "low_aerial_touches", "high_aerial_touches"
            ):
                metrics[f"gameplay/{name}_per_1000_ticks"] = (
                    self.counts[name].item() * 1000 / ticks
                )

        episodes = self.counts["episodes"].item()

        if episodes:
            metrics["gameplay/timeout_fraction"] = (
                self.counts["timeouts"].item() / episodes
            )

        # Pending contact windows span reporting/rollout boundaries.
        for value in self.counts.values():
            value.zero_()

        return metrics
