import torch as th
import torch.nn.functional as F

from dataclasses import dataclass

from carl.gymnasium import RewardContext
from jarl.data.batch import TensorBatch

from curriculum.physics_utils import (
    rotation_matrix_to_quaternion,
    quaternion_forward
)

# Arena
SIDE_WALL_X      = 4096
GOAL_Y            = 5124.25
MATCH_TICKS       = 5 * 60 * 120
BALL_RADIUS = 91.25
BALL_MAX_SPEED = 6000
BALL_WALL_X    = SIDE_WALL_X - BALL_RADIUS

# Offense car reset
CAR_HALF_HEIGHT       = 19.32955
MAX_SUSPENSION_TRAVEL = 12.0
CAR_LATERAL_RANGE     = 100
CAR_BEHIND_MIN, CAR_BEHIND_MAX = 300, 500
ROTATION_PERTURBATION = 0.1

# Ball reset
LENGTH_MIN, LENGTH_MAX = -1000, 1000
HEIGHT_MIN, HEIGHT_MAX =   500,  600
SPEED_MIN, SPEED_MAX   = 1000, 1250
ANGLE_MIN, ANGLE_MAX   =   55,   90

# Defense car reset
DEFENSE_X_LIMIT = 3000
DEFENSE_Y_MIN, DEFENSE_Y_MAX = 1000, 4000
CAR_GROUND_Z = CAR_HALF_HEIGHT

# Reward
TOUCH_RADIUS                = 180.0
TOUCH_REWARD_WEIGHT         = 0.25
BALL_APPROACH_REWARD_WEIGHT = 0.05
HEIGHT_REWARD_SCALE         = 500.0
AIR_POSITION_REWARD_WEIGHT  = 0.02
AIR_BOOST_REWARD_WEIGHT     = 0.005
AIR_DRIBBLE_MIN_HEIGHT      = 700.0
AIR_DRIBBLE_CTRL_DIST       = 250.0
AIR_DRIBBLE_MAX_TOUCH_GAP   = 30
AIR_DRIBBLE_MIN_GOAL_SPEED  = 500.0
AIR_DRIBBLE_START_REWARD    = 0.10
AIR_DRIBBLE_TOUCH_REWARD    = 0.20
AIR_DRIBBLE_CARRY_REWARD    = 0.02
GOAL_TIME_WEIGHT            = 1
GOAL_SPEED_WEIGHT           = 2.5
BALL_PROG_WEIGHT            = 5.0


class WallToAirResetProvider:

    def __init__(
        self,
        device: th.device = "cuda:0", 
        seed:   int = 0
    ) -> None:
        self._device = device
        self._generator = th.Generator(device=device).manual_seed(seed)

        # Combine ranges into tensors
        self.BALL_POS_MINS = th.tensor([LENGTH_MIN, HEIGHT_MIN]).to(device)
        self.BALL_POS_MAXS = th.tensor([LENGTH_MAX, HEIGHT_MAX]).to(device)

    def _ball_position(self, count: int, wall_sign: th.Tensor) -> th.Tensor:
        ball_x = wall_sign * BALL_WALL_X

        return th.cat([
            ball_x,
            th.rand(
                count, 2,
                device=self._device,
                generator=self._generator
            ) * (self.BALL_POS_MAXS - self.BALL_POS_MINS) + self.BALL_POS_MINS
        ], dim=1)

    def _ball_velocity(self, count: int, team_sign: th.Tensor) -> th.Tensor:
        speed = th.rand(
            count,
            device=self._device,
            generator=self._generator
        ) * (SPEED_MAX - SPEED_MIN) + SPEED_MIN

        angle = th.rand(
            count,
            device=self._device,
            generator=self._generator
        ) * (ANGLE_MAX - ANGLE_MIN) + ANGLE_MIN
        angle = th.deg2rad(angle)

        return th.stack([
            th.zeros_like(speed),
            speed * th.cos(angle) * team_sign.squeeze(-1),
            speed * th.sin(angle)
        ], dim=1)

    def _ball_angular_velocity(self, pos: th.Tensor, vel: th.Tensor) -> th.Tensor:
        s = th.sign(pos[:, 0])

        ang = th.zeros_like(vel)
        ang[:, 1] = -s * vel[:, 2]
        ang[:, 2] =  s * vel[:, 1]

        return ang / BALL_RADIUS

    def _offense_car_position(
        self,
        ball_pos:  th.Tensor,
        ball_vel:  th.Tensor,
        wall_sign: th.Tensor,
    ) -> th.Tensor:
        batch_size = ball_pos.shape[0]

        # To set direction of car behind ball
        direction = ball_vel[:, 1:]
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # For random perturbation off direct line
        perp = th.stack((-direction[:, 1], direction[:, 0]), dim=-1)

        dist = th.rand(
            batch_size,
            device=self._device,
            generator=self._generator,
        ) * (CAR_BEHIND_MAX - CAR_BEHIND_MIN) + CAR_BEHIND_MIN

        lateral = (
            th.rand(
                batch_size,
                device=self._device,
                generator=self._generator,
            ) * 2 - 1
        ) * CAR_LATERAL_RANGE

        car_pos = ball_pos.clone()
        car_pos[:, 1:] += (
            -direction * dist[:, None]
            + perp * lateral[:, None]
        )
        car_pos[:, 0] = wall_sign.squeeze(-1) * (
            SIDE_WALL_X - CAR_HALF_HEIGHT - MAX_SUSPENSION_TRAVEL
        )

        return car_pos

    def _defense_car_position(
        self,
        batch_size: int,
        team_sign:  th.Tensor,
    ) -> th.Tensor:
        random_values = th.rand(
            batch_size,
            2,
            device=self._device,
            generator=self._generator,
        )

        car_x = (2 * random_values[:, 0] - 1) * DEFENSE_X_LIMIT
        car_y = team_sign.squeeze(-1) * (
            DEFENSE_Y_MIN
            + random_values[:, 1] * (DEFENSE_Y_MAX - DEFENSE_Y_MIN)
        )

        return th.stack((
            car_x,
            car_y,
            th.full_like(car_x, CAR_GROUND_Z),
        ), dim=-1)

    def _offense_car_rotation(
        self,
        car_pos:   th.Tensor,
        ball_pos:  th.Tensor,
        wall_sign: th.Tensor,
    ) -> th.Tensor:
        wall_normal = th.zeros_like(car_pos)
        wall_normal[:, 0] = -wall_sign.squeeze(-1)

        # Forward must remain tangent to the wall while pointing toward the ball
        forward = ball_pos - car_pos
        forward -= (forward * wall_normal).sum(-1, keepdim=True) * wall_normal
        forward = F.normalize(forward, dim=-1)

        tangent = th.cross(wall_normal, forward, dim=-1)
        perturbation = (
            th.rand(
                car_pos.shape[0], 1,
                device=self._device,
                generator=self._generator,
            ) * 2.0 - 1.0
        ) * ROTATION_PERTURBATION

        # Columns are the car's local forward, right, and up axes.
        forward = F.normalize(forward + tangent * perturbation, dim=-1)
        right = th.cross(wall_normal, forward, dim=-1)
        rotation = th.stack((forward, right, wall_normal), dim=-1)

        return rotation_matrix_to_quaternion(rotation)

    def _defense_car_rotation(self, batch_size: int) -> th.Tensor:
        car_half_yaw = th.rand(
            batch_size,
            device=self._device,
            generator=self._generator,
        ) * th.pi

        car_rotation_zero = th.zeros_like(car_half_yaw)

        return th.stack((
            car_rotation_zero,
            car_rotation_zero,
            th.sin(car_half_yaw),
            th.cos(car_half_yaw),
        ), dim=-1)

    def _offense_car_velocity(
        self,
        ball_vel: th.Tensor,
        car_rot: th.Tensor,
    ) -> th.Tensor:
        speed = ball_vel.norm(dim=-1, keepdim=True)

        speed *= 1 + th.rand(
            ball_vel.shape[0],
            1,
            device=self._device,
            generator=self._generator,
        )

        return F.normalize(quaternion_forward(car_rot), dim=-1) * speed

    def __call__(self, reset_mask: th.Tensor) -> TensorBatch | None:
        env_idx = reset_mask.nonzero(as_tuple=True)[0]
        if not (count := env_idx.numel()):
            return

        # Attacking team (1 = blue)
        team_sign = th.randint(
            0, 2, (count,1),
            device=self._device,
            generator=self._generator
        ) * 2 - 1

        # Left wall positive
        wall_sign = th.randint(
            0, 2, (count, 1),
            device=self._device,
            generator=self._generator,
        ) * 2 - 1

        # Ball rolling upward at angle on wall
        ball_pos = self._ball_position(count, wall_sign)
        ball_vel = self._ball_velocity(count, team_sign)
        ball_ang = self._ball_angular_velocity(ball_pos, ball_vel)

        # Offense car physics state
        off_car_pos = self._offense_car_position(ball_pos, ball_vel, wall_sign)
        off_car_rot = self._offense_car_rotation(off_car_pos, ball_pos, wall_sign)
        off_car_vel = self._offense_car_velocity(ball_vel, off_car_rot)
        off_car_ang = th.zeros_like(off_car_pos)

        # Defense car physics state
        def_car_pos = self._defense_car_position(count, team_sign)
        def_car_rot = self._defense_car_rotation(count)
        def_car_vel = th.zeros_like(off_car_pos)
        def_car_ang = th.zeros_like(off_car_pos)

        # Player index 0 is blue and index 1 is orange
        attacking_is_blue = (team_sign == 1).view(count, 1, 1)

        car_pos = th.where(
            attacking_is_blue,
            th.stack((off_car_pos, def_car_pos), dim=1),
            th.stack((def_car_pos, off_car_pos), dim=1),
        )
        car_rot = th.where(
            attacking_is_blue,
            th.stack((off_car_rot, def_car_rot), dim=1),
            th.stack((def_car_rot, off_car_rot), dim=1),
        )
        car_vel = th.where(
            attacking_is_blue,
            th.stack((off_car_vel, def_car_vel), dim=1),
            th.stack((def_car_vel, off_car_vel), dim=1),
        )
        car_ang = th.where(
            attacking_is_blue,
            th.stack((off_car_ang, def_car_ang), dim=1),
            th.stack((def_car_ang, off_car_ang), dim=1),
        )

        return TensorBatch({
            "simulation_indices":    env_idx,
            "ball_position":         ball_pos,
            "ball_velocity":         ball_vel,
            "ball_angular_velocity": ball_ang,
            "car_position":          car_pos,
            "car_rotation":          car_rot,
            "car_velocity":          car_vel,
            "car_angular_velocity":  car_ang,
            "car_demoed":            th.zeros((count, 2), dtype=th.int32, device=self._device),
            "car_boost":             th.full((count, 2), 100.0, device=self._device),
            "blue_score":            th.zeros(count, dtype=th.int32, device=self._device),
            "orange_score":          th.zeros(count, dtype=th.int32, device=self._device),
            "episode_ticks":         th.zeros(count, dtype=th.int32, device=self._device)
        })


@dataclass
class _RewardState:
    curr:           th.Tensor
    prev:           th.Tensor
    team_sign:      th.Tensor
    opp_goal:       th.Tensor
    curr_ball_dist: th.Tensor
    prev_ball_dist: th.Tensor
    new_touch:       th.Tensor
    ball_goal_speed: th.Tensor
    height_mult:    th.Tensor
    airborne:       th.Tensor


@dataclass
class _AirDribbleState:
    active:      th.Tensor
    touch_age:   th.Tensor
    touch_count: th.Tensor


class WallToAirReward:

    def __init__(self) -> None:
        self._air_dribble: _AirDribbleState | None = None

    def __call__(self, context: RewardContext) -> th.Tensor:
        state = self._state(context)
        goal_reward = self._goal_reward(context, state)
        goal_reward = goal_reward - goal_reward.mean(dim=-1, keepdim=True)

        return (
            (
                self._ball_goal_progress(state)
                + self._touch_reward(state)
                + self._approach_reward(state)
                + self._air_dribble_reward(context, state)
            ) * state.height_mult
            + goal_reward
            + self._air_position_reward(state)
            + self._air_boost_reward(state)
        )

    def _state(self, context: RewardContext) -> _RewardState:
        curr = context.current
        prev = context.previous
        team_sign = curr.team_sign[None, :]

        opp_goal = th.zeros_like(curr.car_position)
        opp_goal[..., 1] = team_sign * GOAL_Y

        curr_ball_dist = (
            curr.car_position - curr.ball_position[:, None, :]
        ).norm(dim=-1)
        
        prev_ball_dist = (
            prev.car_position - prev.ball_position[:, None, :]
        ).norm(dim=-1)
        new_touch = (
            curr_ball_dist <= TOUCH_RADIUS
        ) & (prev_ball_dist > TOUCH_RADIUS)

        ball_goal_speed = (
            curr.ball_velocity[:, None, 1] * team_sign
        ).clamp_min(0.0)

        height_mult = th.exp(
            (curr.ball_position[:, None, 2] - HEIGHT_MIN)
            / HEIGHT_REWARD_SCALE
        ).clamp_max(10.0)
        airborne = (~curr.car_on_ground).float()

        return _RewardState(
            curr=curr,
            prev=prev,
            team_sign=team_sign,
            opp_goal=opp_goal,
            curr_ball_dist=curr_ball_dist,
            prev_ball_dist=prev_ball_dist,
            new_touch=new_touch,
            ball_goal_speed=ball_goal_speed,
            height_mult=height_mult,
            airborne=airborne,
        )

    def _ball_goal_progress(self, state: _RewardState) -> th.Tensor:
        ball_to_goal = state.opp_goal - state.curr.ball_position[:, None, :]
        prev_ball_pos = state.prev.ball_position[:, None, :]
        prev_ball_to_goal = state.opp_goal - prev_ball_pos

        height_loss = (
            prev_ball_pos[..., 2] - state.curr.ball_position[:, None, 2]
        ).clamp_min(0.0)

        height_discount = th.exp(-height_loss / 300.0)

        ball_goal_prog = (
            th.exp(-ball_to_goal.norm(dim=-1) / BALL_MAX_SPEED)
            - th.exp(-prev_ball_to_goal.norm(dim=-1) / BALL_MAX_SPEED)
        )

        return th.where(
            ball_goal_prog >= 0,
            ball_goal_prog * height_discount,
            ball_goal_prog,
        ) * BALL_PROG_WEIGHT

    def _goal_reward(
        self,
        context: RewardContext,
        state: _RewardState,
    ) -> th.Tensor:
        scored = context.events.score_delta[:, None] * state.team_sign
        scored = scored.clamp_min(0)

        goal_speed_bonus = (
            scored
            * state.prev.ball_velocity.norm(dim=-1, keepdim=True)
            / BALL_MAX_SPEED
        )

        t_remaining = (
            (MATCH_TICKS - context.episode_ticks[:, None]) / 120.0
        ).clamp_min(0.0)

        goal_time_bonus = scored * (
            t_remaining / (5.0 * 60.0)
        ).clamp(0.0, 1.0)

        return (
            scored
            + goal_speed_bonus * GOAL_SPEED_WEIGHT
            + goal_time_bonus * GOAL_TIME_WEIGHT
        )

    def _touch_reward(self, state: _RewardState) -> th.Tensor:
        ball_toward_goal = state.ball_goal_speed / BALL_MAX_SPEED

        return state.new_touch * ball_toward_goal * TOUCH_REWARD_WEIGHT

    def _approach_reward(self, state: _RewardState) -> th.Tensor:
        return (
            state.prev_ball_dist - state.curr_ball_dist
        ) / BALL_MAX_SPEED * BALL_APPROACH_REWARD_WEIGHT

    def _air_position_reward(self, state: _RewardState) -> th.Tensor:
        altitude = (
            (state.curr.car_position[..., 2] - CAR_HALF_HEIGHT) / 1000.0
        ).clamp_min(0.0)

        wall_clearance = (
            state.curr.car_position[..., 0].abs() - 1000.0
        ).clamp_min(0.0) / 1000.0

        return (
            state.airborne
            * altitude
            * wall_clearance
            * AIR_POSITION_REWARD_WEIGHT
        )

    def _air_boost_reward(self, state: _RewardState) -> th.Tensor:
        boost_used = (
            state.prev.car_boost - state.curr.car_boost
        ).clamp_min(0.0) / 100.0

        return state.airborne * boost_used * AIR_BOOST_REWARD_WEIGHT

    def _air_dribble_reward(
        self,
        context: RewardContext,
        state:   _RewardState,
    ) -> th.Tensor:
        air_dribble = self._air_dribble_state(state)
        ball_high = state.curr.ball_position[:, None, 2] >= AIR_DRIBBLE_MIN_HEIGHT
        goalward = state.ball_goal_speed >= AIR_DRIBBLE_MIN_GOAL_SPEED
        controlled = (
            state.airborne.bool()
            & ball_high
            & (state.curr_ball_dist <= AIR_DRIBBLE_CTRL_DIST)
            & goalward
        )
        recent = air_dribble.active & (
            air_dribble.touch_age <= AIR_DRIBBLE_MAX_TOUCH_GAP
        )
        start_touch = state.new_touch & controlled & ~recent
        continued_touch = state.new_touch & controlled & recent

        air_dribble.touch_age += 1
        air_dribble.touch_age[state.new_touch] = 0
        air_dribble.touch_count = th.where(
            continued_touch,
            air_dribble.touch_count + 1,
            th.where(
                start_touch,
                th.ones_like(air_dribble.touch_count),
                air_dribble.touch_count,
            ),
        )
        air_dribble.active = (air_dribble.active & controlled) | start_touch | continued_touch

        start_reward = start_touch * AIR_DRIBBLE_START_REWARD
        touch_reward = (
            continued_touch
            * air_dribble.touch_count.float()
            * AIR_DRIBBLE_TOUCH_REWARD
        )
        carry_reward = (
            air_dribble.active
            * state.ball_goal_speed
            / BALL_MAX_SPEED
            * AIR_DRIBBLE_CARRY_REWARD
        )

        done = context.events.done
        if done.any():
            air_dribble.active[done] = False
            air_dribble.touch_age[done] = 0
            air_dribble.touch_count[done] = 0

        return start_reward + touch_reward + carry_reward

    def _air_dribble_state(self, state: _RewardState) -> _AirDribbleState:
        shape = state.curr_ball_dist.shape
        device = state.curr_ball_dist.device

        if (
            self._air_dribble is None
            or self._air_dribble.active.shape != shape
            or self._air_dribble.active.device != device
        ):
            self._air_dribble = _AirDribbleState(
                active=th.zeros(shape, dtype=th.bool, device=device),
                touch_age=th.zeros(shape, dtype=th.int64, device=device),
                touch_count=th.zeros(shape, dtype=th.int64, device=device),
            )

        return self._air_dribble
