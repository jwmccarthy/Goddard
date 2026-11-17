import torch

from rewards.shaping import NextoReward


class GameplayDiagnostics:
    """Nexto rewards plus lightweight GPU-resident gameplay counters."""

    def __init__(self, n_sim, n_cars, device, frameskip, no_touch_timeout_steps=0,
                 reward_plus=False, include_reward_plus_goal=True):
        self.n_cars = n_cars
        self.frameskip = frameskip
        self.no_touch_timeout_steps = no_touch_timeout_steps
        self.reward_tag = "reward_plus" if reward_plus else "reward_nexto"
        self.include_reward_plus_goal = include_reward_plus_goal
        if reward_plus:
            from rewards.reward_plus import RewardPlus

            self.reward = RewardPlus(n_sim, n_cars, device, frameskip)
        else:
            self.reward = NextoReward(n_cars)
        self.touch_steps = torch.zeros(n_sim, dtype=torch.long, device=device)
        self.components = {}
        self.counts = {name: torch.zeros((), device=device) for name in (
            "steps", "touches", "aerial_touches", "demos", "demoed", "goals_for", "goals_against", "episodes", "timeouts"
        )}

    @torch.no_grad()
    def __call__(self, context):
        current, previous = context.current, context.previous
        touches = current.car_ball_touches
        score = context.events.score_delta[:, None] * current.team_sign
        components = self.reward.components(context)
        if self.reward_tag == "reward_plus" and not self.include_reward_plus_goal:
            components.pop("goal")  # GAIfO applies its explicit goal term separately.
        for name, value in components.items():
            if name not in self.components:
                self.components[name] = value.new_zeros(())
            # Zero-sum components cancel globally. Log magnitudes instead of
            # misleading near-zero averages, separately from learner reward.
            self.components[name] += value.abs().sum()
        self.touch_steps += 1
        self.touch_steps[touches.any(-1)] = 0
        timeout = context.events.truncated & (self.touch_steps >= self.no_touch_timeout_steps)
        self.touch_steps[context.events.done] = 0
        self.counts["steps"] += touches.numel()
        self.counts["touches"] += touches.sum()
        self.counts["aerial_touches"] += (touches & ~current.car_on_ground
                                          & current.car_position[..., 2].gt(182.5)).sum()
        self.counts["demos"] += (current.car_demoed & ~previous.car_demoed).sum()
        self.counts["demoed"] += current.car_demoed.sum()
        self.counts["goals_for"] += (score > 0).sum()
        self.counts["goals_against"] += (score < 0).sum()
        self.counts["episodes"] += context.events.done.sum() * self.n_cars
        self.counts["timeouts"] += timeout.sum() * self.n_cars
        return sum(components.values())

    def diagnostic_metrics(self):
        metrics = {}
        steps = self.counts["steps"].item()
        if steps:
            ticks = steps * self.frameskip
            for name in ("touches", "aerial_touches", "demos", "goals_for", "goals_against"):
                metrics[f"gameplay/{name}_per_1000_ticks"] = self.counts[name].item() * 1000 / ticks
            metrics["gameplay/ticks_per_step"] = self.frameskip
            metrics["gameplay/demoed_fraction"] = self.counts["demoed"].item() * self.frameskip / ticks
            metrics.update({f"{self.reward_tag}/{name}_abs_per_tick": value.item() / ticks
                             for name, value in self.components.items()})
        episodes = self.counts["episodes"].item()
        if episodes:
            metrics["gameplay/timeout_fraction"] = self.counts["timeouts"].item() / episodes
        for value in (*self.counts.values(), *self.components.values()):
            value.zero_()
        return metrics
