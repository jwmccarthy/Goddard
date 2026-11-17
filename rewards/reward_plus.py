"""Optional custom RewardPlus shaping, adapted from the former implementation.

Kept separate from the default Nexto reward. GAIfO excludes this module's goal
term and uses its existing explicit goal reward, avoiding double counting.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

BALL_RADIUS = 91.25
BALL_MAX_SPEED = 6000.0
CAR_MAX_SPEED = 2300.0
CEILING_Z = 2044.0
GOAL_HEIGHT = 642.775
GOAL_Y = 5124.25
TICKS_PER_SECOND = 120


@dataclass(frozen=True)
class RewardPlusWeights:
    goal: float = 200.0
    flip_reset: float = 12.5
    demo: float = 5.0
    speed_progress: float = 12.5
    boost_free_speed_progress: float = 25.0
    boost_gain: float = 12.5
    aerial_lift: float = 100.0
    aerial_shot: float = 50.0
    ground_pop: float = 25.0
    air_dribble_setup: float = 25.0
    kickoff_speed: float = 25.0
    kickoff: float = 25.0
    wall_touch: float = 10.0
    wall_control: float = 15.0
    inverted_ground: float = 10.0


def wall_contact(on_ground, position, up_z):
    x, y, z = position[..., 0].abs(), position[..., 1].abs(), position[..., 2]
    near = (x > 3900) | (y > 4900) | ((x > 3000) & (y > 4300))
    return on_ground & (z < 1700) & (up_z.abs() < .5) & near


def advanced_touch_events(context):
    current, previous = context.current, context.previous
    touches = current.car_ball_touches
    relative = current.ball_position[:, None] - current.car_position
    surface = (current.car_on_ground & (current.car_position[..., 2] < 2 * BALL_RADIUS)
               & (current.car_up[..., 2] > .5)) | wall_contact(
                   current.car_on_ground, current.car_position, current.car_up[..., 2])
    shape = current.car_on_ground.shape
    def available(observation):
        return (observation.ego_has_flip_or_jump
                & (observation.ego_flip_window_remaining > 1 / TICKS_PER_SECOND)).reshape(shape)
    native_restored = ~available(context.previous_observation) & available(context.current_observation)
    flip_available = ~(current.car_has_flipped | current.car_has_double_jumped)
    restored = (previous.car_has_flipped | previous.car_has_double_jumped | native_restored) & flip_available
    ball_reset = (touches & restored & ~surface & (current.car_position[..., 2] > 3 * BALL_RADIUS)
                  & (relative.square().sum(-1) < (2 * BALL_RADIUS) ** 2)
                  & (F.cosine_similarity(relative, -current.car_up, dim=-1, eps=1e-6) > .9))
    ceiling_reset = (native_restored & flip_available & (current.car_position[..., 2] > 1850)
                     & (current.car_up[..., 2] < -.7) & ~current.car_demoed & ~previous.car_demoed)
    aerial = touches & (current.car_position[..., 2] > 2 * BALL_RADIUS) & ~surface
    return aerial, ball_reset | ceiling_reset


class RewardPlus:
    def __init__(self, n_sim, n_cars, device, frameskip, weights=None):
        if n_cars < 2 or n_cars % 2 or frameskip < 1:
            raise ValueError("RewardPlus requires equal nonempty teams and a positive frameskip")
        self.frameskip, self.team_size = frameskip, n_cars // 2
        self.weights = weights or RewardPlusWeights()
        self.kickoff_pending = torch.zeros(n_sim, dtype=torch.bool, device=device)
        self.kickoff_challenger = torch.zeros(n_sim, n_cars, dtype=torch.bool, device=device)
        self.ground_setup_ticks = torch.zeros(n_sim, n_cars, dtype=torch.int32, device=device)
        self.wall_ticks = torch.zeros_like(self.ground_setup_ticks)

    def team_difference(self, value):
        total = value.reshape(-1, 2, self.team_size).sum(-1, keepdim=True)
        return (total - total.flip(1)).expand(-1, -1, self.team_size).reshape_as(value)

    @staticmethod
    def shot_quality(velocity, toward_goal):
        speed = (velocity[:, None] * toward_goal).sum(-1)
        alignment = F.cosine_similarity(velocity[:, None], toward_goal, dim=-1, eps=1e-6)
        return ((speed - 400) / 1200).clamp(0, 1) * ((alignment - .8) / .2).clamp(0, 1)

    def kickoff_rewards(self, context, valid):
        current, previous = context.current, context.previous
        spawn = ((previous.ball_position[:, :2].norm(dim=-1) < 2 * BALL_RADIUS)
                 & (previous.ball_velocity.norm(dim=-1) < 250)
                 & ((previous.ball_position[:, 2] - BALL_RADIUS).abs() < 60)
                 & (previous.car_position[:, 0, 1] < -2000)
                 & (previous.car_position[:, self.team_size, 1] > 2000)
                 & (previous.car_position[..., 2] < 120).all(-1))
        first = context.episode_ticks <= self.frameskip
        self.kickoff_pending = torch.where(first, spawn, self.kickoff_pending)
        target = previous.ball_position[:, None, :2]
        distance = (previous.car_position[..., :2] - target).norm(dim=-1)
        closest = distance.reshape(-1, 2, self.team_size).argmin(-1)
        challengers = F.one_hot(closest, self.team_size).reshape_as(valid).bool()
        self.kickoff_challenger = torch.where(first[:, None], challengers & spawn[:, None], self.kickoff_challenger)
        self.kickoff_pending &= context.episode_ticks <= 96 * 4
        next_distance = (current.car_position[..., :2] - target).norm(dim=-1)
        approach = ((1 - next_distance / 5000).clamp(0, 1)
                    - (1 - distance / 5000).clamp(0, 1)).clamp_min(0)
        speed = self.weights.kickoff_speed * approach * (self.kickoff_pending[:, None] & self.kickoff_challenger & valid)
        touches = current.car_ball_touches
        first_touch = (touches & self.kickoff_pending[:, None]).to(previous.raw.dtype)
        self.kickoff_pending &= ~(touches.any(-1) | context.events.done)
        touch = self.weights.kickoff * first_touch / first_touch.sum(-1, keepdim=True).clamp_min(1)
        return speed, touch

    def ground_air_rewards(self, context, valid):
        current, previous = context.current, context.previous
        relative = current.ball_position[:, None] - current.car_position
        distance = relative.norm(dim=-1)
        relative_speed = (current.ball_velocity[:, None] - current.car_velocity).norm(dim=-1)
        control = (1 - relative_speed / 1200).clamp(0, 1)
        goal_side = current.team_sign[None] * relative[..., 1] > -BALL_RADIUS
        not_backwards = current.team_sign[None] * current.ball_velocity[:, None, 1] > -200
        touches = current.car_ball_touches
        fresh = touches & ~previous.car_ball_touches & (touches.sum(-1, keepdim=True) == 1)
        floor_start = (previous.car_on_ground & (previous.car_position[..., 2] < 120)
                       & (previous.car_up[..., 2] > .65) & (current.car_position[..., 2] < 2 * BALL_RADIUS)
                       & (current.car_up[..., 2] > .5))
        rise = current.ball_velocity[:, None, 2] - previous.ball_velocity[:, None, 2]
        peak = current.ball_position[:, 2] + current.ball_velocity[:, 2].clamp_min(0).square() / (2 * 650)
        pop = (valid & fresh & floor_start & (previous.ball_position[:, None, 2] < 3 * BALL_RADIUS)
               & (peak[:, None] > 3 * BALL_RADIUS) & (distance < 280) & goal_side & not_backwards
               & (current.ball_velocity[:, None, 2] > 350) & (rise > 300) & (self.ground_setup_ticks == 0))
        pop = pop * ((rise - 300) / 500).clamp(0, 1) * control
        other_touch = touches.sum(-1, keepdim=True) > touches.long()
        seconds = self.ground_setup_ticks.clamp_max(self.frameskip).to(current.raw.dtype) / TICKS_PER_SECOND
        setup = ((self.ground_setup_ticks > 0) & valid & ~other_touch & ~current.car_on_ground
                 & (current.car_position[..., 2] > 120) & (current.ball_position[:, None, 2] > 2.5 * BALL_RADIUS)
                 & (relative[..., 2] > 0) & goal_side & not_backwards)
        setup = setup * ((400 - distance) / 300).clamp(0, 1) * control * seconds
        remaining = torch.where(pop > 0, 180, (self.ground_setup_ticks - self.frameskip).clamp_min(0))
        landed = ((self.ground_setup_ticks > 0) & (current.ball_position[:, None, 2] <= 2 * BALL_RADIUS)
                  & ((current.ball_velocity[:, None, 2] <= 0) | (previous.ball_velocity[:, None, 2] < 0)))
        self.ground_setup_ticks = torch.where(valid & ~other_touch & ~landed, remaining, 0)
        return pop, setup

    def mechanical_rewards(self, context, valid):
        current, previous = context.current, context.previous
        touches = current.car_ball_touches & valid
        prior_touches = previous.car_ball_touches & valid
        distance = (current.ball_position[:, None] - current.car_position).norm(dim=-1)
        near_wall = (current.ball_position[:, 0].abs() > 3500) | (current.ball_position[:, 1].abs() > 4500)
        event = (touches & ~prior_touches & (touches.sum(-1, keepdim=True) == 1)
                 & wall_contact(current.car_on_ground, current.car_position, current.car_up[..., 2])
                 & near_wall[:, None] & (distance < 280))
        relative_speed = (current.ball_velocity[:, None] - current.car_velocity).norm(dim=-1)
        control = (1 - relative_speed / 1200).clamp(0, 1)
        goalward = (current.team_sign[None] * current.ball_velocity[:, None, 1] / 600).clamp(0, 1)
        touch = event * goalward * control
        other_touch = touches.any(-1, keepdim=True) & ~touches
        seconds = self.wall_ticks.clamp_max(self.frameskip).to(current.raw.dtype) / TICKS_PER_SECOND
        carry = ((self.wall_ticks > 0) & valid & ~other_touch) * ((400 - distance) / 300).clamp(0, 1) * goalward * control * seconds
        self.wall_ticks = torch.where(touch > 0, 180, (self.wall_ticks - self.frameskip).clamp_min(0))
        self.wall_ticks = torch.where(valid & ~other_touch, self.wall_ticks, 0)
        inverted = ((current.car_position[..., 2] < 65) & (current.car_up[..., 2] < -.7)
                    & (current.car_velocity[..., 2].abs() < 120))
        return {"wall_touch": touch, "wall_control": carry,
                "inverted_ground": -(valid & inverted).to(current.raw.dtype) * self.frameskip / TICKS_PER_SECOND}

    @torch.no_grad()
    def components(self, context):
        current, previous = context.current, context.previous
        w = self.weights
        ball_distance = (current.ball_position - previous.ball_position).norm(dim=-1)
        car_distance = (current.car_position - previous.car_position).norm(dim=-1)
        maximum = CAR_MAX_SPEED * self.frameskip / TICKS_PER_SECOND + 150
        continuous = (~previous.car_demoed & (ball_distance[:, None] <= BALL_MAX_SPEED * self.frameskip / TICKS_PER_SECOND + 2 * BALL_RADIUS)
                      & ((car_distance <= maximum) | current.car_demoed))
        valid = continuous & (car_distance <= maximum) & ~current.car_demoed & ~context.events.done[:, None]
        speed_change = (current.car_velocity.norm(dim=-1) - previous.car_velocity.norm(dim=-1)) / CAR_MAX_SPEED
        forward_change = ((current.car_velocity * current.car_forward).sum(-1)
                          - (previous.car_velocity * previous.car_forward).sum(-1)) / CAR_MAX_SPEED
        boost_action = context.actions.reshape(len(current.raw), current.n_cars, -1)[..., 4] > 0
        no_boost = (~boost_action & ~current.car_is_boosting & ~previous.car_is_boosting
                    & ((current.car_boost - previous.car_boost).abs() < .01))
        boost_free = torch.where(no_boost, speed_change, speed_change.clamp_max(0))
        speed = valid * (w.speed_progress * (speed_change + forward_change).clamp_min(0)
                         + w.boost_free_speed_progress * boost_free.clamp_min(0))
        boost_gain = w.boost_gain * valid * ((current.car_boost / 100).clamp(0, 1).sqrt()
                                            - (previous.car_boost / 100).clamp(0, 1).sqrt()).clamp_min(0)
        aerial_touches, flip_reset = advanced_touch_events(context)
        touches = current.car_ball_touches
        aerial = valid & touches & (touches.sum(-1, keepdim=True) == 1) & aerial_touches & (current.ball_position[:, None, 2] > 2 * BALL_RADIUS)
        height = ((current.ball_position[:, None, 2] - BALL_RADIUS) / (CEILING_Z - BALL_RADIUS)).clamp(0, 1)
        upward = ((current.ball_velocity[:, None, 2] - previous.ball_velocity[:, None, 2]) / 400).clamp(-1, 1)
        relative_change = ((current.ball_velocity[:, None] - current.car_velocity).norm(dim=-1)
                           - (previous.ball_velocity[:, None] - previous.car_velocity).norm(dim=-1)).clamp_min(0)
        excess_speed = (current.ball_velocity.norm(dim=-1, keepdim=True) - 1.5 * current.car_velocity.norm(dim=-1)).clamp_min(0)
        control = (1 - torch.minimum(relative_change, excess_speed) / 1200).clamp(0, 1)
        separation = (current.ball_velocity[:, None] - current.car_velocity).norm(dim=-1)
        lift = aerial * height * upward.clamp_min(0) * control * (.5 + .5 * (separation / 300).clamp(0, 1)) * (current.ball_velocity[:, None, 2] > 0)
        goal = torch.zeros_like(current.car_position)
        goal[..., 1] = current.team_sign[None] * GOAL_Y;goal[..., 2] = GOAL_HEIGHT / 2
        direction = F.normalize(goal - current.ball_position[:, None], dim=-1, eps=1e-6)
        shot = (aerial & (current.ball_position[:, None, 2] > GOAL_HEIGHT)) * (self.shot_quality(current.ball_velocity, direction) - self.shot_quality(previous.ball_velocity, direction)).clamp_min(0)
        pop, setup = self.ground_air_rewards(context, valid)
        kickoff_speed, kickoff = self.kickoff_rewards(context, valid)
        mechanical = self.mechanical_rewards(context, valid)
        demoed = current.car_demoed & ~previous.car_demoed
        return {"speed": speed, "boost_gain": boost_gain, "aerial_lift": w.aerial_lift * lift,
                "aerial_shot": w.aerial_shot * shot, "ground_pop": w.ground_pop * pop,
                "air_dribble_setup": w.air_dribble_setup * setup,
                "kickoff_speed": self.team_difference(kickoff_speed),
                "kickoff": self.team_difference(kickoff),
                **{name: getattr(w, name) * value for name, value in mechanical.items()},
                "goal": w.goal * context.events.score_delta[:, None] * current.team_sign,
                "flip_reset": w.flip_reset * flip_reset.float(),
                "demo": w.demo * self.team_difference(-demoed.float())}

    def __call__(self, context):
        return sum(self.components(context).values())
