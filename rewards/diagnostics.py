from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from carl.gymnasium.state import StepContext

from rewards.shaping import NextoReward

if TYPE_CHECKING:
    from training.replay_resets import ReplayResetProvider

TOUCH_WINDOW_TICKS = 8
CROSSBAR_HEIGHT = 642.775
OPENING_TICKS = 360
REPLAY_SOURCE, BANK_SOURCE = 1, 2


class GameplayDiagnostics:
    """Nexto rewards and the gameplay counters shown in training logs."""

    def __init__(
        self,
        n_sim:                  int,
        n_cars:                 int,
        device:                 torch.device,
        frameskip:              int,
        no_touch_timeout_steps: int = 0,
        resets:                 ReplayResetProvider | None = None
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

        has_bank = resets is not None and resets.bank is not None
        self.resets = resets if has_bank else None

        if self.resets is not None:
            self.surfaces = self.resets.bank.surface_names
            self.armed = torch.ones(n_sim, dtype=torch.bool, device=device)
            self.opening_source = torch.zeros(n_sim, dtype=torch.int8, device=device)
            self.opening_surface = torch.zeros(n_sim, dtype=torch.long, device=device)
            self.opening_ticks = torch.zeros(n_sim, dtype=torch.long, device=device)
            self.opening_touched = torch.zeros(n_sim, dtype=torch.bool, device=device)
            self.opening_counted = torch.zeros(n_sim, dtype=torch.bool, device=device)

            for name in ("arena_ticks", "bank_ticks", "replay_resets", "replay_aerial"):
                self.counts[name] = torch.zeros((), device=device)

            for name in ("bank", *(f"bank_{surface}" for surface in self.surfaces)):
                self.counts[f"{name}_resets"] = torch.zeros((), device=device)
                self.counts[f"{name}_aerial"] = torch.zeros((), device=device)

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
        aerial_touch = torch.zeros_like(context.events.done)

        if self.fixed_touch_windows:
            self.window_ticks.add_(self.frameskip)
            self.window_touches.logical_or_(touches)
            complete = self.window_ticks == TOUCH_WINDOW_TICKS
            contacted = self.window_touches & complete[:, None]
            self.counts["touch_windows"] += complete.sum() * self.n_cars
            self.counts["touches"] += contacted.sum()
            # Match FS8: contact anywhere in the window, aerial at its end.
            aerial_contact = contacted & aerial
            high = current.car_position[..., 2].gt(CROSSBAR_HEIGHT)
            aerial_touch = aerial_contact.any(-1)
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

        if self.resets is not None:
            self.track_openings(aerial_touch, context.events.done)

        self.counts["goals_for"] += (score > 0).sum()
        self.counts["episodes"] += context.events.done.sum() * self.n_cars
        self.counts["timeouts"] += timeout.sum() * self.n_cars

        return self.reward(context)

    def track_openings(
        self,
        aerial: torch.Tensor,
        done:   torch.Tensor
    ) -> None:
        """Count aerial touches in the opening seconds after each reset."""
        resets = self.resets
        fresh = self.armed
        surface = resets.bank.surface[resets.bank_indices.clamp(min=0)].long()
        self.opening_source.copy_(
            torch.where(fresh, resets.sources, self.opening_source)
        )
        self.opening_surface.copy_(
            torch.where(fresh, surface, self.opening_surface)
        )
        self.opening_ticks.masked_fill_(fresh, 0)
        self.opening_touched.masked_fill_(fresh, False)
        self.opening_counted.masked_fill_(fresh, False)

        source = self.opening_source
        remaining = (
            (OPENING_TICKS - self.opening_ticks).clamp(0, self.frameskip)
        )
        self.counts["arena_ticks"] += self.armed.numel() * self.frameskip
        self.counts["bank_ticks"] += (
            torch.where(source == BANK_SOURCE, remaining, 0).sum()
        )

        self.opening_touched.logical_or_(
            aerial & (self.opening_ticks < OPENING_TICKS)
        )
        self.opening_ticks += self.frameskip

        due = (
            ((self.opening_ticks >= OPENING_TICKS) | done)
            & ~self.opening_counted
        )

        for name, pick in (
            ("replay", source == REPLAY_SOURCE),
            ("bank", source == BANK_SOURCE)
        ):
            pick = pick & due
            self.counts[f"{name}_resets"] += pick.sum()
            self.counts[f"{name}_aerial"] += (pick & self.opening_touched).sum()

        for index, name in enumerate(self.surfaces):
            pick = due & (source == BANK_SOURCE) & (self.opening_surface == index)
            self.counts[f"bank_{name}_resets"] += pick.sum()
            self.counts[f"bank_{name}_aerial"] += (
                pick & self.opening_touched
            ).sum()

        self.opening_counted.logical_or_(due)
        self.armed.copy_(done)

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

        if self.resets is not None:
            ticks = self.counts["arena_ticks"].item()

            if ticks:
                metrics["gameplay/bank_tick_share"] = (
                    self.counts["bank_ticks"].item() / ticks
                )

            if self.fixed_touch_windows:
                for name, metric in (
                    ("replay", "replay_aerial_touch_rate"),
                    ("bank", "bank_aerial_touch_rate"),
                    *(
                        (f"bank_{surface}", f"bank_aerial_touch_rate_{surface}")
                        for surface in self.surfaces
                    )
                ):
                    resets = self.counts[f"{name}_resets"].item()

                    if resets:
                        metrics[f"gameplay/{metric}"] = (
                            self.counts[f"{name}_aerial"].item() / resets
                        )

        # Pending contact windows span reporting/rollout boundaries.
        for value in self.counts.values():
            value.zero_()

        return metrics
