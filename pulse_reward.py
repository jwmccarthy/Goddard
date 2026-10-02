"""PULSE self-play reward built from Goddard's shared shaping components."""

import torch as th

from carl.gymnasium.state import RewardContext

from reward_spec import RewardSpec, RewardWeights


class PulseReward(RewardSpec):
    """Anneal local Nexto shaping while retaining goals and touch/timeout rewards."""

    def __init__(
        self,
        shaping_scale: float = 1.0,
        goal_scale: float = 10.0,
        touch_scale: float = 0.1,
        no_touch_penalty: float = 1.0,
        no_touch_timeout_steps: int | None = None,
    ) -> None:
        # The shared reward halves the demo difference before weighting it.
        # PULSE's original Nexto shaping used the full difference.
        super().__init__(normalize=False, weights=RewardWeights(demo=10.0))
        self.shaping_scale = shaping_scale
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
        # The original PULSE reward kept shaping local to each player instead
        # of centering it against the opponent, and scaled it against a goal of 10.
        shaping = sum(
            value for name, value in components.items() if name != "goal_scored"
        ) / 10.0
        # Kickoff shaping in PULSE only applied during the first second.
        shaping = shaping - components["kickoff"] * (
            context.episode_ticks[:, None] >= 120
        ) / 10.0
        reward = (
            self.goal_scale * self.last_score_for_actor
            + self.touch_scale * self.last_touches
            - self.no_touch_penalty * self.last_no_touch_timeout[:, None]
            + self.shaping_scale * shaping
        )
        done = context.events.done
        self._last_touch[done] = False
        self._steps_since_touch[done] = 0
        return reward
