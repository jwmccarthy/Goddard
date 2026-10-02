"""PULSE self-play reward built from Goddard's shared shaping components."""

import torch as th

from carl.gymnasium.state import RewardContext

from reward_spec import RewardSpec


class PulseReward(RewardSpec):
    """Keep BASIC dense shaping alongside annealed Nexto and gameplay rewards."""

    def __init__(
        self,
        shaping_scale: float = 1.0,
        goal_scale: float = 10.0,
        touch_scale: float = 0.1,
        no_touch_penalty: float = 1.0,
        no_touch_timeout_steps: int | None = None,
        basic_shaping_scale: float = 1.0,
    ) -> None:
        # Keep fixed goal/touch rewards in PULSE's units rather than normalizing
        # the entire reward as BASIC does.
        super().__init__(normalize=False)
        self.shaping_scale = shaping_scale
        self.basic_shaping_scale = basic_shaping_scale
        self.goal_scale = goal_scale
        self.touch_scale = touch_scale
        self.no_touch_penalty = no_touch_penalty
        self.no_touch_timeout_steps = no_touch_timeout_steps
        self._steps_since_touch: th.Tensor | None = None
        self.last_touches: th.Tensor | None = None
        self.last_score_for_actor: th.Tensor | None = None
        self.last_no_touch_timeout: th.Tensor | None = None

    def __call__(self, context: RewardContext) -> th.Tensor:
        touches = context.current.car_ball_touches
        if (
            self._steps_since_touch is None
            or self._steps_since_touch.shape[0] != touches.shape[0]
        ):
            self._steps_since_touch = th.zeros(
                touches.shape[0], dtype=th.long, device=touches.device
            )
        self._steps_since_touch += 1
        self._steps_since_touch[touches.any(dim=-1)] = 0
        self.last_touches = touches.detach().clone()
        self.last_score_for_actor = (
            context.events.score_delta[:, None] * context.current.team_sign[None, :]
        ).detach().clone()
        self.last_no_touch_timeout = (
            context.events.truncated
            & (self._steps_since_touch >= self.no_touch_timeout_steps)
            if self.no_touch_timeout_steps is not None
            else th.zeros_like(context.events.truncated)
        )
        return super().__call__(context)

    def _finish_reward(
        self, context: RewardContext, components: dict[str, th.Tensor]
    ) -> th.Tensor:
        dense = sum(
            value for name, value in components.items() if name != "goal_scored"
        )
        # BASIC centers its dense reward against the opposing player. Its goal
        # component is omitted here because PULSE already awards goals below.
        basic_shaping = dense - self._opponent_team_mean(dense)
        # Nexto keeps shaping local, doubles BASIC's demo term, and only gives
        # kickoff shaping in the first second.
        nexto_shaping = (
            dense + components["demo"] - components["kickoff"]
            * (context.episode_ticks[:, None] >= 120)
        ) / 10.0
        reward = (
            self.goal_scale * self.last_score_for_actor
            + self.touch_scale * self.last_touches
            - self.no_touch_penalty * self.last_no_touch_timeout[:, None]
            + self.basic_shaping_scale * basic_shaping
            + self.shaping_scale * nexto_shaping
        )
        done = context.events.done
        self._last_touch[done] = False
        self._steps_since_touch[done] = 0
        return reward
