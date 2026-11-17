"""Default vectorized Nexto-era reward; optional custom shaping is in reward_plus.

Reference: Rolv-Arild/Necto, 5c56a79910decdfe1298c7cb234d08101a0bf70a,
training/reward.py (before Necto-v3/Tecko experiments). See NEXTO_SOURCE.md.
This adaptation is licensed CC BY-NC-SA 4.0; it is not an official Nexto release.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class NextoWeights:
    team_spirit: float = .6
    goal: float = 10.
    goal_distance: float = 10.
    goal_bonus: float = 2.5
    demo: float = 5.
    player_distance: float = .5
    player_alignment: float = .5
    boost_gain: float = 1.
    boost_loss: float = .9
    ground: float = .005
    touch_height: float = 1.
    touch_acceleration: float = .25
    opponent_punish: float = 1.


class NextoReward:
    def __init__(self, n_cars, weights=None):
        if n_cars < 2 or n_cars % 2:
            raise ValueError("Nexto reward requires equal, nonempty teams")
        self.n_cars = n_cars
        self.team_size = n_cars // 2
        self.weights = weights or NextoWeights()

    def qualities(self, state):
        """Published state/player qualities; signed, undiscounted differences."""
        ball, cars = state.ball_position, state.car_position
        # Reference RLGym goal geometry, deliberately not a new tuned target.
        blue = ball.new_tensor([0., -5560., 321.3875])
        orange = ball.new_tensor([0., 5560., 321.3875])
        state_quality = .25 * self.weights.goal_distance * (
            torch.exp(-(orange - ball).norm(dim=-1) / 2300.)
            - torch.exp(-(blue - ball).norm(dim=-1) / 2300.)
        )
        blue_back = ball.new_tensor([0., -6000., 321.3875])
        orange_back = ball.new_tensor([0., 6000., 321.3875])
        to_ball = ball[:, None] - cars
        alignment = .5 * (
            F.cosine_similarity(to_ball, orange_back - cars, dim=-1, eps=1e-8)
            - F.cosine_similarity(to_ball, blue_back - cars, dim=-1, eps=1e-8)
        ) * state.team_sign
        player_quality = (self.weights.player_distance * torch.exp(-to_ball.norm(dim=-1) / 1410.)
                          + self.weights.player_alignment * alignment)
        return state_quality, player_quality

    def distribute(self, reward):
        teams = reward.reshape(-1, 2, self.team_size)
        means = teams.mean(-1, keepdim=True)
        return ((1 - self.weights.team_spirit) * teams + self.weights.team_spirit * means
                - self.weights.opponent_punish * means.flip(1)).reshape_as(reward)

    @torch.no_grad()
    def components(self, context):
        current, previous = context.current, context.previous
        w = self.weights
        state_quality, player_quality = self.qualities(current)
        old_state_quality, old_player_quality = self.qualities(previous)
        touches = current.car_ball_touches.float()
        height = .5 * (current.ball_position[:, None, 2] + current.car_position[..., 2]) / 2044.
        acceleration = (current.ball_velocity - previous.ball_velocity).norm(dim=-1) / 2300.
        # CARL stores boost in [0,100]; the reference uses [0,1].
        boost_delta = ((current.car_boost / 100.).clamp(0, 1).sqrt()
                       - (previous.car_boost / 100.).clamp(0, 1).sqrt())
        demoed = (current.car_demoed & ~previous.car_demoed).float()
        victim_counts = demoed.reshape(-1, 2, self.team_size).sum(-1, keepdim=True)
        # CARL does not identify the demolisher. Share attacker credit across
        # the opposing team; this is exact in 1v1, an adaptation in 2v2/3v3.
        demo_credit = victim_counts.flip(1).expand(-1, -1, self.team_size).reshape_as(demoed) / self.team_size
        base = {
            "ball_position": (state_quality - old_state_quality)[:, None] * current.team_sign,
            "player_position": player_quality - old_player_quality,
            "touch_height": w.touch_height * touches * (2 - current.car_on_ground.float()) * height,
            "touch_acceleration": w.touch_acceleration * touches * acceleration[:, None],
            "boost_gain": w.boost_gain * boost_delta.clamp_min(0),
            "boost_loss": w.boost_loss * boost_delta.clamp_max(0),
            "ground": -w.ground * current.car_on_ground.float(),
            "demo": .5 * w.demo * (demo_credit - demoed),
        }
        score = context.events.score_delta[:, None] * current.team_sign
        goal = context.events.score_delta.ne(0)[:, None]
        speed = previous.ball_velocity.norm(dim=-1)[:, None]
        defender_distance = (current.car_position - previous.ball_position[:, None]).norm(dim=-1)
        # The reference REPLACES shaping on goals, rather than adding a goal
        # reward to it. It uses goal_dist_bonus_w for BOTH bonuses (source quirk).
        goals = torch.where(score > 0, w.goal * score + w.goal_bonus * speed / 6000.,
                            -w.goal_bonus * (1 - torch.exp(-defender_distance / 2300.)))
        components = {name: self.distribute(value.masked_fill(goal, 0)) for name, value in base.items()}
        components["goal"] = self.distribute(goals.masked_fill(~goal, 0))
        return components

    def __call__(self, context):
        return sum(self.components(context).values())
