import argparse
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.state import RewardContext
from jarl.collect import (
    CaptureContext,
    CriticCapture,
    LogProbCapture,
    RecurrentCriticCapture,
    RecurrentStateCapture,
    Runner,
)
from jarl.collect.capture import CaptureBase
from jarl.data import TensorBatch, TensorDataset
from jarl.envs import DatasetResetSampler, ResetContext
from jarl.learn import (
    Algorithm,
    IndependentOptimizerSteps,
    LossOutput,
    OptimizerStep,
    PPOConfig,
    PPOLoss,
    Update,
)
from jarl.log.logger import Logger
from jarl.modules import GRU, MLP, orthogonal_init
from jarl.modules.encoder import LinearEncoder
from jarl.modules.operator import Critic
from jarl.modules.policy import MultiCategoricalPolicy
from jarl.runtime import (
    Clock,
    LinearSchedule,
    OnPolicySchedule,
    ScheduledValue,
    Trainer,
    ValueScheduler,
)
from jarl.sample import RecurrentRolloutMinibatches, RolloutMinibatches
from jarl.store import RolloutBuffer
from jarl.store.rollout import Rollout
from jarl.transform import GAE, PrepareContext

from gaifo_ase import (
    ASEPPOLoss, FiniteIndependentOptimizerSteps, SkillConditionedEnv,
    SkillDiscoveryReward, SkillEncoder, SkillEncoderUpdate, SkillGRUEncoder,
    SkillSequenceEncoder, SkillStreamContext,
)
from replay_resets import (
    ReplayResetProvider, _sampled_frame_skip, reset_index_dataset,
)
from replay_safety import (
    GOAL_EXCLUSION_SECONDS, TICKS_PER_SECOND,
    infer_unsafe_start_mask, pre_goal_start_mask,
)
from reward_spec import CEILING_Z


SCENE_SIZE = 51
GAIFO_ARCHITECTURE = "scene-marl-gaifo-1v1-v3"
GAIFO_GRU_ARCHITECTURE = "scene-marl-gaifo-1v1-v3-gru"
GAIFO_ASE_ARCHITECTURE = "scene-marl-gaifo-1v1-v3-ase"
BALL_SIZE = 9
CAR_SIZE = 21
N_CARS = 2
DOUBLES_N_CARS = 4
DOUBLES_SCENE_SIZE = BALL_SIZE + DOUBLES_N_CARS * CAR_SIZE
BLUE_START = 9
ORANGE_START = 30
CAR_BOOL_START = 16
CAR_BOOL_END = 21
INTERNAL_STATE_START = 137
INTERNAL_STATE_SIZE = 19
POSITION_SCALE = (4108.0, 6000.0, 2076.0)
BALL_MAX_SPEED = 6000.0
BALL_RADIUS = 91.25
GOAL_Y = 5124.25
GOAL_HEIGHT = 642.775
CAR_MAX_SPEED = 2300.0
INTERNAL_BOOL_INDICES = (0, 2, 3, 4, 5, 7, 8, 9, 11, 17)
BALL_NEAR_DISTANCE = 1_500.0
BALL_CLOSE_DISTANCE = 350.0
SITUATION_MATCH_FRACTION = 0.25
GROUND_RANDOM_FRACTION = 0.05
DISTANCE_BANDS = ("close", "approach", "far")
CAR_SITUATIONS = ("grounded", "wall", "low_air", "mid_air", "high_air", "ceiling")
GROUND_MANEUVERS = ("dribble", "flick")
GROUND_MANEUVER_START = len(DISTANCE_BANDS) * len(CAR_SITUATIONS)
N_SITUATIONS = GROUND_MANEUVER_START + len(GROUND_MANEUVERS)
SKILL_CATEGORIES = (
    "aerial_touch", "aerial_maneuver", "dribble", "flick", "driving", "kickoff",
)
(
    AERIAL_TOUCH_SKILL, AERIAL_MANEUVER_SKILL, DRIBBLE_SKILL,
    FLICK_SKILL, DRIVING_SKILL, KICKOFF_SKILL,
) = range(len(SKILL_CATEGORIES))
SKILL_AERIAL_NEAR_STEPS = 4
SKILL_AERIAL_MANEUVER_CONTACTS = 3
SKILL_AERIAL_TOUCH_SEPARATION = 3
KICKOFF_MAX_STEPS = 96  # At frame skip 4, covers the challenge after the approach.
KICKOFF_FOLLOW_THROUGH_STEPS = 16
LOW_AIR_HEIGHT = 350.0
MID_AIR_HEIGHT = 900.0
NEAR_CEILING_HEIGHT = 1_700.0
CEILING_AIR_HEIGHT = 1_850.0
AERIAL_SETUP_STEPS = 16
AERIAL_RECOVERY_STEPS = 16
AERIAL_MIN_CONTEXT_STEPS = 8
MANEUVER_SETUP_STEPS = 8
MANEUVER_RECOVERY_STEPS = 8
MANEUVER_MAX_TRACKED_STEPS = 256
GROUND_MAX_TRACKED_STEPS = 256
MANEUVER_MAX_PENDING = 2_048
MANEUVER_ARCHIVE_PER_SITUATION = 32
DRIBBLE_MIN_STEPS = 5
DRIBBLE_MAX_HORIZONTAL_DISTANCE = 180.0
DRIBBLE_MAX_RELATIVE_SPEED = 1_200.0
FLICK_MIN_VELOCITY_CHANGE = 500.0
BALL_GATE_RADIUS = 200.0
BALL_GATE_SCALE = 1_000.0
BALL_GATE_FLOOR = 0.1
RESET_MINING_CANDIDATES = 8
RESET_MINING_FRACTION = 0.5
RESET_MINING_MIN_CONFIDENCE = 0.6


def scene_car_count(scenes: th.Tensor) -> int:
    if scenes.shape[-1] not in (SCENE_SIZE, DOUBLES_SCENE_SIZE):
        raise ValueError("physical scenes must contain two or four cars")
    return (scenes.shape[-1] - BALL_SIZE) // CAR_SIZE


def noise_mask(device: str | th.device = "cpu", n_cars: int = N_CARS) -> th.Tensor:
    """Boolean mask that is True for continuous scene features and False for car booleans."""
    if n_cars not in (N_CARS, DOUBLES_N_CARS):
        raise ValueError("noise mask needs two or four cars")
    mask = th.ones(BALL_SIZE + n_cars * CAR_SIZE, dtype=th.bool, device=device)
    for car in range(n_cars):
        start = BALL_SIZE + car * CAR_SIZE
        mask[start + CAR_BOOL_START : start + CAR_BOOL_END] = False
    return mask


def add_scene_noise(windows: th.Tensor, std: float) -> th.Tensor:
    """Apply symmetric Gaussian noise only to continuous scene features."""
    if not math.isfinite(std) or std < 0.0:
        raise ValueError("scene noise standard deviation must be finite and nonnegative")
    n_cars = scene_car_count(windows)
    if std <= 0.0:
        return windows
    mask = noise_mask(windows.device, n_cars).view(
        *((1,) * (windows.ndim - 1)), windows.shape[-1]
    )
    noise = th.randn_like(windows) * std * mask
    return windows + noise


def nearest_ball_distance(windows: th.Tensor) -> th.Tensor:
    """Closest ego-car/ball distance in physical units over each causal window."""
    if windows.ndim < 3:
        raise ValueError("ball proximity needs scene windows")
    scene_car_count(windows)
    return _ball_distances(windows, BLUE_START).amin(dim=-1)


def _ball_distances(windows: th.Tensor, car_start: int) -> th.Tensor:
    relative = windows[..., :3] - windows[..., car_start:car_start + 3]
    return th.linalg.vector_norm(relative * windows.new_tensor(POSITION_SCALE), dim=-1)


def scene_situation_ids(windows: th.Tensor, car_start: int = BLUE_START) -> th.Tensor:
    """Distance and ego-car situation at the closest approach in each 1v1 window.

    Six car situations are crossed with close (<350), approach (<1500), and
    far (>=1500) ball-distance bands. Near-roof airborne play counts as ceiling;
    a car with its wheels on a vertical surface counts as wall instead.
    """
    if windows.ndim != 3 or windows.shape[-1] != SCENE_SIZE:
        raise ValueError("situation labels need [batch, frames, 1v1 scene] windows")
    if car_start not in (BLUE_START, ORANGE_START):
        raise ValueError("situation car must be one of the stored 1v1 actors")
    distance = _ball_distances(windows, car_start)
    closest = distance.argmin(dim=1)
    rows = th.arange(len(windows), device=windows.device)
    car = windows[rows, closest, car_start:car_start + CAR_SIZE]
    height = car[:, 2] * POSITION_SCALE[2]
    up_z = car[:, 14]
    grounded = car[:, CAR_BOOL_START] > 0.5
    wall_contact = grounded & (up_z < 0.5) & (up_z > -0.5)
    ceiling = (height >= NEAR_CEILING_HEIGHT) & (
        (up_z <= -0.5) | ((height >= CEILING_AIR_HEIGHT) & ~wall_contact)
    )

    situation = th.zeros_like(closest)
    situation[grounded & (up_z < 0.5)] = 1
    situation[~grounded & (height < LOW_AIR_HEIGHT)] = 2
    situation[~grounded & (height >= LOW_AIR_HEIGHT)
              & (height < MID_AIR_HEIGHT)] = 3
    situation[~grounded & (height >= MID_AIR_HEIGHT)] = 4
    situation[ceiling] = 5
    approach_distance = distance[rows, closest]
    band = th.where(approach_distance < BALL_CLOSE_DISTANCE, 0,
                    th.where(approach_distance < BALL_NEAR_DISTANCE, 1, 2))
    return situation * len(DISTANCE_BANDS) + band


@dataclass(frozen=True)
class SceneManeuver:
    """An actor's setup, active ball/car control, and recovery window indices."""

    situation: int
    setup_start: int
    action_start: int
    action_stop: int
    recovery_stop: int
    actor: int = 0
    stride: int = 1
    goal_terminal: bool = False


@dataclass(frozen=True)
class AirManeuver(SceneManeuver):
    skill_category: int = AERIAL_TOUCH_SKILL

    @property
    def takeoff(self) -> int:
        return self.action_start

    @property
    def landing(self) -> int:
        return self.action_stop


@dataclass(frozen=True)
class GroundManeuver(SceneManeuver):
    @property
    def carry_start(self) -> int:
        return self.action_start

    @property
    def release(self) -> int:
        """First recovery frame, after the flick impulse when present."""
        return self.action_stop


def _arena_surface_contact(
    on_ground: th.Tensor, position: th.Tensor, up_z: th.Tensor,
) -> th.Tensor:
    """Distinguish floor and side walls from wheel contact with a ball or roof."""
    x = position[..., 0].abs()
    y = position[..., 1].abs()
    z = position[..., 2]
    floor = (z < 2 * BALL_RADIUS) & (up_z > 0.5)
    near_wall = (
        (x > 3_900) | (y > 4_900) | ((x > 3_000) & (y > 4_300))
    )
    wall = near_wall & (z < NEAR_CEILING_HEIGHT) & (up_z.abs() < 0.5)
    return on_ground & (floor | wall)


def recovery_surface_contact(
    scenes: th.Tensor, previous: th.Tensor, car_start: int = BLUE_START,
) -> th.Tensor:
    """Only floor and side-wall contacts end a flight, not ball or roof contacts."""
    car = scenes[..., car_start:car_start + CAR_SIZE]
    prior = previous[..., car_start:car_start + CAR_SIZE]
    position = car[..., :3] * scenes.new_tensor(POSITION_SCALE)
    on_surface = _arena_surface_contact(
        car[..., CAR_BOOL_START] > 0.5, position, car[..., 14],
    )

    # Wall contact restores a spent flip too. A nearby ball doing the restoring
    # is a flip reset, however, and must not truncate the active flight.
    spent_flip = (prior[..., 18] > 0.5) | (prior[..., 19] > 0.5)
    flip_available = (car[..., 18] < 0.5) & (car[..., 19] < 0.5)
    ball_reset = (
        spent_flip & flip_available & (position[..., 2] > 3 * BALL_RADIUS)
        & (_ball_distances(scenes, car_start) < 2 * BALL_RADIUS)
    )
    return on_surface & ~ball_reset


def _maneuver_runs(
    valid: np.ndarray, goal_ends: np.ndarray | None,
) -> list[np.ndarray]:
    if goal_ends is not None and goal_ends.shape != valid.shape:
        raise ValueError("goal endings must align with maneuver frames")
    active = np.flatnonzero(valid)
    if not len(active):
        return []
    boundaries = np.diff(active) != 1
    if goal_ends is not None:
        # A new episode can start immediately after a scored goal.
        boundaries |= goal_ends[active[:-1]]
    return np.split(active, np.flatnonzero(boundaries) + 1)


def _aerial_contact_count(touches: np.ndarray) -> int:
    """Debounce nearby sampled flags from the same physical contact."""
    starts = np.flatnonzero(
        np.diff(np.pad(touches.astype(np.int8), (1, 0))) == 1
    )
    return (1 + int((np.diff(starts) >= SKILL_AERIAL_TOUCH_SEPARATION).sum())
            if len(starts) else 0)


def air_maneuvers(
    valid: np.ndarray,
    on_surface: np.ndarray,
    height: np.ndarray,
    up_z: np.ndarray,
    ball_distance: np.ndarray,
    *,
    setup_steps: int = AERIAL_SETUP_STEPS,
    recovery_steps: int = AERIAL_RECOVERY_STEPS,
    goal_ends: np.ndarray | None = None,
    touches: np.ndarray | None = None,
    allow_partial: bool = False,
) -> list[AirManeuver]:
    """Trace flights; three contacts also qualify without setup or recovery."""
    if (not all(len(values) == len(valid) for values in
        (on_surface, height, up_z, ball_distance))
            or setup_steps < 1 or recovery_steps < 1):
        raise ValueError("maneuver timelines must agree and include setup/recovery")
    if (touches is not None and touches.shape != valid.shape
            or allow_partial and touches is None):
        raise ValueError("partial aerials require aligned touch events")
    flights: list[AirManeuver] = []
    for run in _maneuver_runs(valid, goal_ends):
        goal_at_end = goal_ends is not None and bool(goal_ends[run[-1]])
        grounded = on_surface[run]
        takeoffs = np.flatnonzero(grounded[:-1] & ~grounded[1:]) + 1
        landings = np.flatnonzero(grounded)
        previous_landing = int(landings[0]) if len(landings) else 0
        spans = []
        if allow_partial and not grounded[0]:
            landing = int(landings[0]) if len(landings) else len(run)
            if _aerial_contact_count(touches[run[:landing]]) >= SKILL_AERIAL_MANEUVER_CONTACTS:
                next_takeoff = int(takeoffs[0]) if len(takeoffs) else len(run)
                spans.append((0, 0, landing, min(len(run), landing + recovery_steps,
                                                 next_takeoff)))
        for index, takeoff in enumerate(takeoffs):
            next_landing = np.searchsorted(landings, takeoff)
            if next_landing == len(landings):
                if (not goal_at_end and not (
                    allow_partial and _aerial_contact_count(touches[run[takeoff:]])
                    >= SKILL_AERIAL_MANEUVER_CONTACTS
                )):
                    break  # An incomplete airborne span cannot provide recovery.
                landing = len(run)
            else:
                landing = int(landings[next_landing])
            if landing - takeoff < 2:
                previous_landing = landing
                continue
            next_takeoff = int(takeoffs[index + 1]) if index + 1 < len(takeoffs) else len(run)
            setup_start = max(previous_landing, takeoff - setup_steps)
            recovery_stop = min(len(run), landing + recovery_steps, next_takeoff)
            spans.append((setup_start, takeoff, landing, recovery_stop))
            previous_landing = landing
        for setup_start, takeoff, landing, recovery_stop in spans:
            air = run[takeoff:landing]
            peak = float(height[air].max())
            if peak <= 2 * BALL_RADIUS:
                continue  # A ground-level flick is not an aerial maneuver.
            roof = ((height[air] >= NEAR_CEILING_HEIGHT)
                    & ((up_z[air] <= -0.5) | (height[air] >= CEILING_AIR_HEIGHT)))
            if roof.any():
                situation = 5
            elif peak >= MID_AIR_HEIGHT:
                situation = 4
            elif peak >= LOW_AIR_HEIGHT:
                situation = 3
            else:
                situation = 2
            closest = float(ball_distance[air].min())
            band = 0 if closest < BALL_CLOSE_DISTANCE else (
                1 if closest < BALL_NEAR_DISTANCE else 2
            )
            flights.append(AirManeuver(
                situation=situation * len(DISTANCE_BANDS) + band,
                setup_start=int(run[setup_start]),
                action_start=int(run[takeoff]),
                action_stop=int(run[landing]) if landing < len(run) else int(run[-1]) + 1,
                recovery_stop=int(run[recovery_stop - 1]) + 1,
                goal_terminal=goal_at_end and recovery_stop == len(run),
            ))
    return flights


def aerial_skill_category(
    touches: np.ndarray, car_height: np.ndarray,
    ball_position: np.ndarray, ball_distance: np.ndarray,
) -> int | None:
    """Three distinct airborne contacts define a maneuver, with no carry gate."""
    if _aerial_contact_count(touches) >= SKILL_AERIAL_MANEUVER_CONTACTS:
        return AERIAL_MANEUVER_SKILL
    ball_height = ball_position[:, 2] * POSITION_SCALE[2]
    close = ((car_height > 2 * BALL_RADIUS) & (ball_height > 250)
             & (ball_distance < BALL_CLOSE_DISTANCE))
    edges = np.flatnonzero(np.diff(np.pad(close.astype(np.int8), (1, 1))))
    if not len(edges) or max(edges[1::2] - edges[::2]) < SKILL_AERIAL_NEAR_STEPS:
        return None

    touch_edges = np.flatnonzero(np.diff(np.pad(touches.astype(np.int8), (1, 1))))
    return (AERIAL_TOUCH_SKILL if any(
        close[begin:end].any()
        for begin, end in zip(touch_edges[::2], touch_edges[1::2])
    ) else None)


def generated_air_skill(
    windows: th.Tensor, flight: AirManeuver, touches: np.ndarray,
) -> int:
    """Label a generated flight from actual ball-contact events, when available."""
    active = slice(flight.action_start, flight.action_stop)
    if not touches[active].any():
        return AERIAL_TOUCH_SKILL
    scored = windows[active, -1]
    category = aerial_skill_category(
        touches[active],
        (scored[:, BLUE_START + 2] * POSITION_SCALE[2]).cpu().numpy(),
        scored[:, :3].cpu().numpy(),
        _ball_distances(scored, BLUE_START).cpu().numpy(),
    )
    return AERIAL_TOUCH_SKILL if category is None else category


def ground_feature_indices(car_start: int) -> tuple[int, ...]:
    """Scene columns needed to recognize an ego-car carry and its release."""
    return (*range(6), *range(car_start, car_start + 5),
            car_start + 14, car_start + 16, car_start + 18)


def dribble_control_mask(features: np.ndarray) -> np.ndarray:
    """Upright ground car carrying a nearby elevated ball at similar speed."""
    if features.shape[-1] != 14:
        raise ValueError("ground ball-control features must contain 14 values")
    ball = features[..., :3] * POSITION_SCALE
    car = features[..., 6:9] * POSITION_SCALE
    horizontal = np.linalg.norm(ball[..., :2] - car[..., :2], axis=-1)
    relative_velocity = np.linalg.norm(
        features[..., 3:5] * BALL_MAX_SPEED
        - features[..., 9:11] * CAR_MAX_SPEED, axis=-1,
    )
    return (
        (features[..., 12] > 0.5) & (features[..., 11] > 0.65)
        & (car[..., 2] < 130) & (ball[..., 2] >= 120)
        & (ball[..., 2] < 320) & (ball[..., 2] - car[..., 2] >= 100)
        & (ball[..., 2] - car[..., 2] < 270)
        & (horizontal < DRIBBLE_MAX_HORIZONTAL_DISTANCE)
        & (relative_velocity < DRIBBLE_MAX_RELATIVE_SPEED)
    )


def ground_maneuvers(
    valid: np.ndarray,
    features: np.ndarray,
    *,
    setup_steps: int = MANEUVER_SETUP_STEPS,
    recovery_steps: int = MANEUVER_RECOVERY_STEPS,
    goal_ends: np.ndarray | None = None,
) -> list[GroundManeuver]:
    """Trace sustained dribbles and flicks through recovery or a scored goal."""
    if features.shape != (len(valid), 14) or setup_steps < 1 or recovery_steps < 1:
        raise ValueError("ground control needs aligned scenes and setup/recovery")
    raw = dribble_control_mask(features) & valid
    carrying = raw.copy()
    carrying[1:-1] |= raw[:-2] & raw[2:] & valid[1:-1]
    maneuvers: list[GroundManeuver] = []
    for run in _maneuver_runs(valid, goal_ends):
        goal_at_end = goal_ends is not None and bool(goal_ends[run[-1]])
        in_control = carrying[run]
        changes = np.flatnonzero(np.diff(np.pad(in_control.astype(np.int8), (1, 1))))
        starts, stops = changes[::2], changes[1::2]
        previous_stop = 0
        for index, (start, stop) in enumerate(zip(starts, stops)):
            next_start = int(starts[index + 1]) if index + 1 < len(starts) else len(run)
            setup_start = max(previous_stop, int(start) - setup_steps)
            previous_stop = int(stop)
            if stop - start < DRIBBLE_MIN_STEPS or setup_start == start:
                continue
            if stop == len(run):
                if goal_at_end:
                    maneuvers.append(GroundManeuver(
                        situation=GROUND_MANEUVER_START,
                        setup_start=int(run[setup_start]),
                        action_start=int(run[start]),
                        action_stop=int(run[-1]) + 1,
                        recovery_stop=int(run[-1]) + 1,
                        goal_terminal=True,
                    ))
                continue
            release_end = min(len(run), int(stop) + recovery_steps, next_start)
            flip_begin = max(0, int(stop) - 3)
            flips = features[run[flip_begin:release_end], 13] > 0.5
            flip_frames = np.flatnonzero(flips[1:] & ~flips[:-1]) + flip_begin + 1
            post = features[run[stop:release_end]]
            if not len(post):
                continue
            previous_velocity = features[run[stop - 1:release_end - 1], 3:6]
            launch = np.linalg.norm(
                (post[:, 3:6] - previous_velocity) * BALL_MAX_SPEED, axis=-1,
            )
            impulse_frames = np.flatnonzero(
                (launch >= FLICK_MIN_VELOCITY_CHANGE)
                & (post[:, 5] * BALL_MAX_SPEED > 100)
            ) + int(stop)
            jumped = (post[:4, 12] < 0.5).any()
            release_action = next((
                max(int(flip), int(impulse)) + 1
                for flip in flip_frames for impulse in impulse_frames
                if abs(int(flip) - int(impulse)) <= 2
            ), None) if jumped else None
            action_stop = release_action if release_action is not None else int(stop)
            recovery_stop = min(len(run), action_stop + recovery_steps, next_start)
            if (recovery_stop < action_stop
                    or (recovery_stop == action_stop and not goal_at_end)):
                continue
            maneuvers.append(GroundManeuver(
                situation=GROUND_MANEUVER_START + int(release_action is not None),
                setup_start=int(run[setup_start]),
                action_start=int(run[start]),
                action_stop=int(run[action_stop]) if action_stop < len(run)
                else int(run[-1]) + 1,
                recovery_stop=int(run[recovery_stop - 1]) + 1,
                goal_terminal=goal_at_end and recovery_stop == len(run),
            ))
    return maneuvers


def ball_responsibility(windows: th.Tensor) -> th.Tensor:
    """Keep a small off-ball signal and credit recent proximity after contact."""
    separation = (nearest_ball_distance(windows) - BALL_GATE_RADIUS).clamp_min(0)
    return BALL_GATE_FLOOR + (1 - BALL_GATE_FLOOR) * th.exp(
        -0.5 * (separation / BALL_GATE_SCALE).square()
    )


def actor_view(scenes: th.Tensor, actor_index: int) -> th.Tensor:
    """Make one actor the ego, followed by teammates and then opponents."""
    n_cars = scene_car_count(scenes)
    if not 0 <= actor_index < n_cars:
        raise ValueError("actor index must identify a car in the scene")
    team_size = n_cars // 2
    view = scenes.clone()
    if actor_index >= team_size:
        # Orange actors attack in the opposite direction in the stored view.
        view[..., [0, 1, 3, 4, 6, 7]] *= -1.0
        for car in range(n_cars):
            start = BALL_SIZE + car * CAR_SIZE
            for offset in (0, 3, 6, 9, 12):
                view[..., start + offset : start + offset + 2] *= -1.0

    own = range((actor_index // team_size) * team_size,
                (actor_index // team_size + 1) * team_size)
    order = [actor_index, *(car for car in own if car != actor_index),
             *(car for car in range(n_cars) if car // team_size != actor_index // team_size)]
    cars = view[..., BALL_SIZE:].reshape(*view.shape[:-1], n_cars, CAR_SIZE)
    return th.cat((view[..., :BALL_SIZE], cars[..., order, :].flatten(-2)), dim=-1)


def opponent_view(scenes: th.Tensor) -> th.Tensor:
    """Use the first opposing player as the ego (rotating the field)."""
    return actor_view(scenes, scene_car_count(scenes) // 2)


def teammate_view(scenes: th.Tensor) -> th.Tensor:
    """Use the other player on the ego's team as the ego in a 2v2 scene."""
    if scene_car_count(scenes) != DOUBLES_N_CARS:
        raise ValueError("teammate view needs a four-car scene")
    return actor_view(scenes, 1)


def actor_views(scenes: th.Tensor) -> th.Tensor:
    """Stack all actors' ego views without mixing the two teams' car roles."""
    if scenes.ndim < 2:
        raise ValueError("actor views need a window of scenes")
    return th.stack(
        [actor_view(scenes, actor) for actor in range(scene_car_count(scenes))],
        dim=-3,
    )


def _resample_coordinates(
    length: int,
    source_frame_skip: int,
    target_frame_skip: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_frame_skip < 1 or target_frame_skip < 1:
        raise ValueError("source and target frame skips must be positive")
    source_ticks = np.arange(length, dtype=np.float64) * source_frame_skip
    target_ticks = np.arange(
        0.0,
        source_ticks[-1] + 1.0,
        target_frame_skip,
    )
    right = np.searchsorted(source_ticks, target_ticks).clip(0, length - 1)
    left = (right - 1).clip(0, length - 1)
    span = source_ticks[right] - source_ticks[left]
    alpha = np.divide(
        target_ticks - source_ticks[left],
        span,
        out=np.zeros_like(target_ticks),
        where=span > 0,
    ).astype(np.float32)
    return left, right, alpha


def resample_scene(
    scene: np.ndarray,
    source_frame_skip: int,
    target_frame_skip: int,
) -> np.ndarray:
    """Resample normalized physical scenes onto the simulator's time cadence."""
    if len(scene) < 2 or source_frame_skip == target_frame_skip:
        return np.asarray(scene, dtype=np.float32)
    left, right, alpha = _resample_coordinates(
        len(scene), source_frame_skip, target_frame_skip
    )
    output = (
        scene[left] * (1.0 - alpha[:, None])
        + scene[right] * alpha[:, None]
    ).astype(np.float32)

    nearest = np.where(alpha < 0.5, left, right)
    for car_start in (BLUE_START, ORANGE_START):
        bool_slice = slice(
            car_start + CAR_BOOL_START,
            car_start + CAR_BOOL_END,
        )
        output[:, bool_slice] = scene[nearest, bool_slice]
        forward = output[:, car_start + 9:car_start + 12]
        up = output[:, car_start + 12:car_start + 15]
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True).clip(1e-6)
        up -= forward * np.sum(forward * up, axis=-1, keepdims=True)
        up /= np.linalg.norm(up, axis=-1, keepdims=True).clip(1e-6)

    return output


def resample_internal_state(
    state: np.ndarray,
    source_frame_skip: int,
    target_frame_skip: int,
) -> np.ndarray:
    if len(state) < 2 or source_frame_skip == target_frame_skip:
        return np.asarray(state, dtype=np.float32)
    left, right, alpha = _resample_coordinates(
        len(state), source_frame_skip, target_frame_skip
    )
    output = (
        state[left] * (1.0 - alpha[:, None])
        + state[right] * alpha[:, None]
    ).astype(np.float32)
    nearest = np.where(alpha < 0.5, left, right)
    output[:, INTERNAL_BOOL_INDICES] = state[nearest[:, None], INTERNAL_BOOL_INDICES]
    return output


def extract_scene_observations(
    observation: th.Tensor,
) -> th.Tensor:
    """Return every actor's own canonical 51-feature physical scene prefix."""
    if observation.shape[-1] < SCENE_SIZE:
        raise ValueError(f"actor observations require at least {SCENE_SIZE} features")
    return observation[..., :SCENE_SIZE].contiguous()


def _unsafe_replay_reset_frames(
    path: Path, stored: np.ndarray, frame_skip: int,
) -> np.ndarray:
    """Reject annotated unsafe starts, pre-goal starts, and contact/correction rows."""
    sidecar = path.with_suffix(".unsafe-starts.npz")
    if sidecar.is_file():
        with np.load(sidecar) as metadata:
            unsafe = np.asarray(metadata["unsafe"], dtype=bool)
            pre_goal = np.asarray(metadata.get(
                "pre_goal", pre_goal_start_mask(
                    len(stored), frame_skip, (len(stored) - 1) * frame_skip,
                ),
            ), dtype=bool)
    else:
        unsafe = infer_unsafe_start_mask(stored[:, 3:6] * BALL_MAX_SPEED, frame_skip)
        pre_goal = pre_goal_start_mask(
            len(stored), frame_skip, (len(stored) - 1) * frame_skip,
        )
    if unsafe.shape != (len(stored),) or pre_goal.shape != (len(stored),):
        raise ValueError(f"invalid safety mask for {path.name}")
    # The first touch column is the ego's contact. An opponent's contact is
    # immediately available from the paired POV; neither is a safe reset tick.
    return unsafe | pre_goal | np.asarray(stored[:, -5:], dtype=bool).any(axis=-1)


def _replay_goal_scorer(path: Path, stored: np.ndarray) -> int | None:
    """Identify the scorer at a period boundary, including legacy replay files.

    Older safety sidecars omit goal annotations. Their last physical ball state
    can still establish a goal when it has crossed the goal line inside the
    mouth. An explicit non-goal annotation takes precedence over that fallback.
    """
    sidecar = path.with_suffix(".unsafe-starts.npz")
    if sidecar.is_file():
        with np.load(sidecar) as metadata:
            if "pre_goal" in metadata:
                pre_goal = metadata["pre_goal"]
                if pre_goal.shape != (len(stored),):
                    raise ValueError(f"invalid goal mask for {path.name}")
                if not bool(pre_goal[-1]):
                    return None
                return 0 if stored[-1, 1] > 0 else 1

    ball = stored[-1, :3] * np.asarray(POSITION_SCALE)
    if (abs(ball[1]) >= GOAL_Y and abs(ball[0]) < 900
            and 0 < ball[2] < GOAL_HEIGHT):
        return 0 if ball[1] > 0 else 1
    return None


def advanced_touch_events(
    context: RewardContext,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Reward every elevated aerial touch more at height; detect flip resets."""
    current = context.current
    previous = context.previous
    touches = current.car_ball_touches
    ball_to_car = current.ball_position[:, None, :] - current.car_position
    ball_height = current.ball_position[:, None, 2]
    car_height = current.car_position[..., 2]

    spent_flip = previous.car_has_flipped | previous.car_has_double_jumped
    flip_available = ~(current.car_has_flipped | current.car_has_double_jumped)
    flip_reset = (
        touches
        & spent_flip
        & flip_available
        & car_height.gt(3.0 * BALL_RADIUS)
        & ball_to_car.square().sum(dim=-1).lt((2.0 * BALL_RADIUS) ** 2)
        & F.cosine_similarity(
            ball_to_car, -current.car_up, dim=-1, eps=1e-6
        ).gt(0.9)
    )
    on_surface = _arena_surface_contact(
        current.car_on_ground, current.car_position, current.car_up[..., 2],
    )
    aerial = touches & car_height.gt(2.0 * BALL_RADIUS) & (~on_surface | flip_reset)
    # A low aerial earns a nonzero bonus; the configured weight caps it at the roof.
    height_fraction = ((ball_height - BALL_RADIUS) / (CEILING_Z - BALL_RADIUS)).clamp(0, 1)
    aerial_score = aerial.to(current.raw.dtype) * (0.5 + 0.5 * height_fraction)
    return aerial_score, aerial, flip_reset


class GameplayDiagnostics:
    """Record physical events and goal-only episode rewards before CARL auto-resets."""

    def __init__(self, n_sim: int, device: th.device, no_touch_timeout_steps: int) -> None:
        self.no_touch_timeout_steps = no_touch_timeout_steps
        self.touch_steps = th.zeros(n_sim, dtype=th.long, device=device)
        self.last_aerial_touch_score = th.zeros(n_sim * N_CARS, device=device)
        self.last_flip_reset = th.zeros(n_sim * N_CARS, device=device)
        self.last_ego_ball_touch = th.zeros(n_sim * N_CARS, dtype=th.bool, device=device)
        self.last_opponent_ball_touch = th.zeros(n_sim * N_CARS, dtype=th.bool, device=device)
        self.counts = {
            name: th.zeros((), dtype=th.float32, device=device)
            for name in (
                "steps",
                "touches",
                "aerial_touches",
                "flip_resets",
                "goals_for",
                "goals_against",
                "episodes",
                "timeouts",
            )
        }

    @th.no_grad()
    def __call__(self, context: RewardContext) -> th.Tensor:
        touches = context.current.car_ball_touches
        self.last_ego_ball_touch = touches.reshape(-1)
        self.last_opponent_ball_touch = touches.flip(dims=(-1,)).reshape(-1)
        score_for_actor = context.events.score_delta[:, None] * context.current.team_sign
        aerial_score, aerial, flip_reset = advanced_touch_events(context)
        self.last_aerial_touch_score = (
            aerial_score - aerial_score.flip(dims=(-1,))
        ).reshape(-1)
        flip_reset_score = flip_reset.to(aerial_score.dtype)
        self.last_flip_reset = (
            flip_reset_score - flip_reset_score.flip(dims=(-1,))
        ).reshape(-1)

        self.touch_steps += 1
        self.touch_steps[touches.any(dim=-1)] = 0
        timeout = context.events.truncated & (
            self.touch_steps >= self.no_touch_timeout_steps
        )
        self.touch_steps[context.events.done] = 0

        self.counts["steps"] += touches.numel()
        self.counts["touches"] += touches.sum()
        self.counts["aerial_touches"] += aerial.sum()
        self.counts["flip_resets"] += flip_reset.sum()
        self.counts["goals_for"] += (score_for_actor > 0).sum()
        self.counts["goals_against"] += (score_for_actor < 0).sum()
        self.counts["episodes"] += context.events.done.sum() * N_CARS
        self.counts["timeouts"] += timeout.sum() * N_CARS

        # GAIFO's imitation reward is computed separately; preserve CARL's
        # default goal-only reward for episode statistics and rollout records.
        return score_for_actor

    def diagnostic_metrics(self) -> dict[str, dict[str, float]]:
        metrics = {}
        steps = self.counts["steps"].item()
        if steps > 0:
            metrics.update({
                "touches_per_1000_steps": self.counts["touches"].item() / steps * 1000,
                "aerial_touches_per_1000_steps": self.counts["aerial_touches"].item() / steps * 1000,
                "flip_resets_per_1000_steps": self.counts["flip_resets"].item() / steps * 1000,
                "goals_for_per_1000_steps": self.counts["goals_for"].item() / steps * 1000,
                "goals_against_per_1000_steps": self.counts["goals_against"].item() / steps * 1000,
            })
            for name in (
                "steps", "touches", "aerial_touches", "flip_resets",
                "goals_for", "goals_against",
            ):
                self.counts[name].zero_()

        episodes = self.counts["episodes"].item()
        if episodes > 0:
            metrics["timeout_fraction"] = self.counts["timeouts"].item() / episodes
            self.counts["episodes"].zero_()
            self.counts["timeouts"].zero_()

        return {"Gameplay": metrics} if metrics else {}


class AdvancedTouchCapture(CaptureBase):
    """Keep physical touch rewards separate from CARL's goal-only episode return."""

    def __init__(self, gameplay: GameplayDiagnostics) -> None:
        self.gameplay = gameplay

    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        return {
            "aerial_touch_score": self.gameplay.last_aerial_touch_score,
            "flip_reset_event": self.gameplay.last_flip_reset,
        }


class ASEBallTouchCapture(CaptureBase):
    """Capture which player caused a ball touch on the scored transition."""

    def __init__(self, gameplay: GameplayDiagnostics) -> None:
        self.gameplay = gameplay

    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        return {
            "ego_ball_touch": self.gameplay.last_ego_ball_touch,
            "opponent_ball_touch": self.gameplay.last_opponent_ball_touch,
        }


class SceneWindowCapture(CaptureBase):
    """Capture short scene windows across rollout boundaries, resetting on done."""

    def __init__(self, trajectory_length: int) -> None:
        if trajectory_length < 2:
            raise ValueError("trajectory length must be at least 2")
        self.trajectory_length = trajectory_length
        self.distances = np.arange(trajectory_length - 1, -1, -1)

        self.n_envs = 0
        self.history: th.Tensor | None = None
        self.history_age: th.Tensor | None = None
        self.history_pos: th.Tensor | None = None

    def reset(self, batch_size: int) -> None:
        if batch_size % N_CARS:
            raise ValueError("1v1 collection requires two actors per simulation")
        self.n_envs = batch_size
        self.history = None
        self.history_age = None
        self.history_pos = None

    def _capture(self, context: CaptureContext) -> dict[str, th.Tensor]:
        observation = context.observation
        next_obs = th.as_tensor(
            context.env_step.next_obs,
            dtype=observation.dtype,
            device=observation.device,
        )
        if len(observation) != self.n_envs:
            raise ValueError("scene capture batch changed after reset")

        current_scene = extract_scene_observations(observation)
        next_scene = extract_scene_observations(next_obs)
        n_envs = len(current_scene)
        capacity = self.trajectory_length - 1

        if self.history is None:
            self.history = th.zeros(
                n_envs,
                capacity,
                SCENE_SIZE,
                dtype=observation.dtype,
                device=observation.device,
            )
            self.history_age = th.zeros(
                n_envs,
                dtype=th.long,
                device=observation.device,
            )
            self.history_pos = th.zeros(
                n_envs,
                dtype=th.long,
                device=observation.device,
            )
        assert self.history_age is not None
        assert self.history_pos is not None

        # A new episode begins at its kickoff state. Left-pad causal context
        # with that state so its first actions can receive imitation reward.
        fresh = self.history_age == 0
        if fresh.any():
            self.history[fresh] = current_scene[fresh, None].expand(-1, capacity, -1)
            self.history_pos[fresh] = 0
            self.history_age[fresh] = capacity - 1

        env_indices = th.arange(n_envs, device=observation.device)
        self.history[env_indices, self.history_pos] = current_scene
        self.history_pos = (self.history_pos + 1) % capacity
        self.history_age = self.history_age + 1

        valid = self.history_age >= capacity
        result: dict[str, th.Tensor] = {
            "scene_window": self._gather_window(current_scene, next_scene),
            "scene_window_valid": valid,
        }

        done = th.as_tensor(
            context.env_step.done,
            dtype=th.bool,
            device=observation.device,
        )
        n_sim = n_envs // N_CARS
        simulation_done = done.view(n_sim, N_CARS).any(dim=1)
        env_done = simulation_done.repeat_interleave(N_CARS)
        self.history[env_done] = 0
        self.history_age[env_done] = 0
        self.history_pos[env_done] = 0

        return result

    def _gather_window(
        self,
        current_scene: th.Tensor,
        next_scene: th.Tensor,
    ) -> th.Tensor:
        """Build [n_envs, trajectory_length, SCENE_SIZE] windows from circular history.

        Distances are chronological offsets from ``next_scene`` (0 = newest),
        ordered oldest-to-newest so the window ends with the scored transition's
        next observation.
        """
        assert self.history is not None
        assert self.history_pos is not None
        n_envs = len(current_scene)
        capacity = self.history.shape[1]
        samples = len(self.distances)
        window = th.empty(
            n_envs,
            samples,
            SCENE_SIZE,
            dtype=current_scene.dtype,
            device=current_scene.device,
        )

        zero_mask = self.distances == 0
        if zero_mask.any():
            window[:, zero_mask] = next_scene[:, None]

        non_zero = self.distances[~zero_mask]
        if len(non_zero):
            distance_tensor = th.as_tensor(
                non_zero, dtype=th.long, device=self.history_pos.device
            )
            positions = (
                self.history_pos.unsqueeze(1) - distance_tensor
            ) % capacity
            env_indices = th.arange(
                n_envs, device=self.history_pos.device
            )[:, None].expand_as(positions)
            window[:, ~zero_mask] = self.history[env_indices, positions]
        return window


class ExpertSceneDataset:
    """Expert windows from stored 1v1 POVs, never from an unstored opponent."""

    def __init__(
        self,
        replay_dir: Path,
        trajectory_length: int,
        limit: int | None = None,
        seed: int = 0,
        frame_skip: int | None = None,
        device: str | th.device = "cpu",
        heldout_size: int = 0,
        reject_discontinuities: bool = False,
        skill_sampling: bool = False,
        driving_fraction: float = 0.10,
        kickoff_fraction: float = 0.05,
    ) -> None:
        if trajectory_length < 2:
            raise ValueError("trajectory length must be at least 2")
        if limit is not None and limit < trajectory_length:
            raise ValueError("expert frame limit must fit one trajectory")
        if frame_skip is not None and frame_skip < 1:
            raise ValueError("expert frame skip must be positive")
        if heldout_size < 0:
            raise ValueError("heldout size must be non-negative")
        if not math.isfinite(driving_fraction) or not 0.0 <= driving_fraction < 1.0:
            raise ValueError("general-driving fraction must be in [0, 1)")
        if not math.isfinite(kickoff_fraction) or not 0.0 <= kickoff_fraction < 1.0:
            raise ValueError("kickoff fraction must be in [0, 1)")
        if driving_fraction + kickoff_fraction >= 1.0:
            raise ValueError("driving and kickoff fractions must sum to less than one")
        if skill_sampling and not reject_discontinuities:
            raise ValueError("skill-filtered expert clips require discontinuity filtering")
        self.trajectory_length = trajectory_length
        self.heldout_size = heldout_size
        self.partition_span = trajectory_length - 1
        self.frame_skip = frame_skip if frame_skip is not None else 4
        self.kickoff_max_steps = max(1, round(KICKOFF_MAX_STEPS * 4 / self.frame_skip))
        self.kickoff_follow_through_steps = max(
            1, round(KICKOFF_FOLLOW_THROUGH_STEPS * 4 / self.frame_skip),
        )
        self.skill_sampling = skill_sampling
        self.driving_fraction = driving_fraction
        self.kickoff_fraction = kickoff_fraction

        rng = np.random.default_rng(seed)
        paths = sorted(Path(replay_dir).glob("*.npy"))
        groups: dict[tuple[str, ...], list[Path]] = {}
        for path in paths:
            source = np.load(path, mmap_mode="r")
            if source.ndim != 2 or source.shape[1] != 161:
                continue
            key = self._dedup_key(path)
            groups.setdefault(key, []).append(path)

        selected = [sorted(groups[key]) for key in sorted(groups)]
        if not selected:
            raise ValueError(f"no 1v1 replay files found in {replay_dir}")

        if limit is not None:
            rng.shuffle(selected)

        frames: list[th.Tensor] = []
        internal_states: list[th.Tensor] = []
        opponent_povs: list[th.Tensor] = []
        invalid_frames: list[th.Tensor] = []
        contact_frames: list[th.Tensor] = []
        ego_touches: list[th.Tensor] = []
        unsafe_reset_frames: list[th.Tensor] = []
        lengths: list[int] = []
        goal_actors: list[int | None] = []
        total = 0
        for group in selected:
            path = group[0]
            if frame_skip is not None:
                metadata_path = path.with_suffix(".unsafe-starts.npz")
                if metadata_path.is_file():
                    with np.load(metadata_path) as metadata:
                        stored_frame_skip = int(metadata.get("frame_skip", -1))
                else:
                    stored_frame_skip = _sampled_frame_skip(path, frame_skip)
            stored = np.load(path, mmap_mode="r")
            goal_actor = _replay_goal_scorer(path, stored)
            if skill_sampling:
                source_skip = stored_frame_skip if frame_skip is not None else 4
                unsafe_reset = _unsafe_replay_reset_frames(path, stored, source_skip)
                touches = np.zeros((len(stored), N_CARS), dtype=bool)
                touches[:, 0] = stored[:, 156] > 0.5
            if reject_discontinuities:
                # The final two columns flag parser corrections and implausible
                # physics jumps. The preceding columns are real touch/bump
                # events and must remain in the motion prior's training data.
                invalid = np.asarray(stored[:, -2:], dtype=bool).any(axis=-1)
                contact = np.asarray(stored[:, -5:-2], dtype=bool).any(axis=-1)
            source = np.array(
                stored[:, :SCENE_SIZE], dtype=np.float32, copy=True
            )
            ego_internal = np.array(
                stored[
                    :, INTERNAL_STATE_START:INTERNAL_STATE_START + INTERNAL_STATE_SIZE
                ],
                dtype=np.float32,
                copy=True,
            )
            opponent_internal = np.zeros_like(ego_internal)
            if len(group) > 1:
                opponent_path = group[1]
                opponent = np.load(opponent_path, mmap_mode="r")
                if len(opponent) != len(stored):
                    raise ValueError(f"paired POV rows differ for {path.name}")
                if skill_sampling:
                    unsafe_reset |= _unsafe_replay_reset_frames(
                        opponent_path, opponent, source_skip,
                    )
                    touches[:, 1] = opponent[:, 156] > 0.5
                if reject_discontinuities:
                    invalid |= np.asarray(opponent[:, -2:], dtype=bool).any(axis=-1)
                    contact |= np.asarray(opponent[:, -5:-2], dtype=bool).any(axis=-1)
                opponent_internal = np.array(
                    opponent[
                        :,
                        INTERNAL_STATE_START:INTERNAL_STATE_START + INTERNAL_STATE_SIZE,
                    ],
                    dtype=np.float32,
                    copy=True,
                )
                if frame_skip is not None:
                    opponent_metadata_path = opponent_path.with_suffix(
                        ".unsafe-starts.npz"
                    )
                    if opponent_metadata_path.is_file():
                        with np.load(opponent_metadata_path) as metadata:
                            opponent_frame_skip = int(metadata.get("frame_skip", -1))
                    else:
                        opponent_frame_skip = _sampled_frame_skip(
                            opponent_path, frame_skip
                        )
                    if opponent_frame_skip != stored_frame_skip:
                        raise ValueError(f"paired POV cadence differs for {path.name}")
            else:
                # CARL uses the internal state, rather than the scene flags, to
                # determine grounded/flip availability on a replay reset.
                for scene_field, internal_field in (
                    (16, 0), (19, 7), (18, 8), (20, 17),
                ):
                    opponent_internal[:, internal_field] = source[
                        :, ORANGE_START + scene_field
                    ]
            if reject_discontinuities and frame_skip is not None and stored_frame_skip != frame_skip:
                left, right, _ = _resample_coordinates(
                    len(stored), stored_frame_skip, frame_skip
                )
                invalid = invalid[left] | invalid[right]
                event_prefix = np.pad(contact.astype(np.int64).cumsum(0), (1, 0))
                previous_right = np.concatenate(([-1], right[:-1]))
                contact = (event_prefix[right + 1] - event_prefix[previous_right + 1]) > 0
            if skill_sampling and frame_skip is not None and stored_frame_skip != frame_skip:
                left, right, _ = _resample_coordinates(
                    len(stored), stored_frame_skip, frame_skip,
                )
                unsafe_reset = unsafe_reset[left] | unsafe_reset[right]
                touches = touches[left] | touches[right]
            if frame_skip is not None:
                source = resample_scene(source, stored_frame_skip, frame_skip)
                ego_internal = resample_internal_state(
                    ego_internal, stored_frame_skip, frame_skip
                )
                opponent_internal = resample_internal_state(
                    opponent_internal, stored_frame_skip, frame_skip
                )
            internal = np.stack((ego_internal, opponent_internal), axis=1)
            full_length = len(source)
            if limit is not None and total + len(source) > limit:
                keep = max(0, limit - total)
                if keep == 0:
                    break
                source = source[:keep]
                internal = internal[:keep]
                if reject_discontinuities:
                    invalid = invalid[:keep]
                    contact = contact[:keep]
                if skill_sampling:
                    unsafe_reset = unsafe_reset[:keep]
                    touches = touches[:keep]
            real_length = len(source)
            # Every kickoff belongs to a causal window, including the first
            # frame of each replay period. Repeating its initial state provides
            # history without borrowing frames from another segment or the future.
            pad = trajectory_length - 1
            source = np.concatenate((np.repeat(source[:1], pad, axis=0), source))
            internal = np.concatenate((np.repeat(internal[:1], pad, axis=0), internal))
            if reject_discontinuities:
                invalid = np.concatenate((np.repeat(invalid[:1], pad), invalid))
                contact = np.concatenate((np.repeat(contact[:1], pad), contact))
            if skill_sampling:
                unsafe_reset = np.concatenate((
                    np.ones(pad, dtype=bool), unsafe_reset,
                ))
                touches = np.concatenate((
                    np.zeros((pad, N_CARS), dtype=bool), touches,
                ))
            frames.append(th.from_numpy(source))
            internal_states.append(th.from_numpy(internal))
            opponent_povs.append(th.full((len(source),), len(group) > 1, dtype=th.bool))
            if reject_discontinuities:
                invalid_frames.append(th.from_numpy(invalid.copy()))
                contact_frames.append(th.from_numpy(contact.copy()))
            if skill_sampling:
                unsafe_reset_frames.append(th.from_numpy(unsafe_reset.copy()))
                ego_touches.append(th.from_numpy(touches.copy()))
            lengths.append(len(source))
            goal_actors.append(goal_actor if real_length == full_length else None)
            total += real_length
            if limit is not None and total >= limit:
                break

        if not frames:
            raise ValueError(f"no expert frames loaded from {replay_dir}")

        self.frames = th.cat(frames).to(device)
        self.internal_states = th.cat(internal_states).to(device)
        self.opponent_pov_available = th.cat(opponent_povs).to(device)
        self.contact_frames = (
            th.cat(contact_frames).to(device) if reject_discontinuities else None
        )
        self.ego_touches = th.cat(ego_touches).to(device) if skill_sampling else None
        self.unsafe_reset_frames = (
            th.cat(unsafe_reset_frames).to(device) if skill_sampling else None
        )
        self.lengths = lengths
        self.segment_goal_actors = goal_actors
        window_starts = []
        segment_window_starts = []
        segment_frame_indices = []
        offset = 0
        for length in lengths:
            count = max(0, length - trajectory_length + 1)
            # Prefix copies are context, not additional physical reset states.
            segment_frame_indices.append(th.arange(
                offset + self.partition_span, offset + length,
            ))
            starts = th.arange(offset, offset + count)
            segment_window_starts.append(starts)
            if count:
                window_starts.append(starts)
            offset += length
        if not window_starts:
            raise ValueError(
                f"expert files are too short to build windows of length {trajectory_length}"
            )
        self.window_starts = th.cat(window_starts).to(device)
        self.segment_window_starts = [starts.to(device) for starts in segment_window_starts]
        self.segment_frame_indices = [indices.to(device) for indices in segment_frame_indices]
        self.real_frame_indices = th.cat(segment_frame_indices).to(device)
        self.window_offsets = th.arange(trajectory_length, device=device)
        self.total_windows = len(self.window_starts)

        self._split_heldout(device, seed)
        if reject_discontinuities:
            invalid = th.cat(invalid_frames).to(device)
            prefix = F.pad(invalid.long().cumsum(0), (1, 0))

            def safe_starts(starts: th.Tensor) -> th.Tensor:
                return starts[
                    prefix[starts + self.trajectory_length] == prefix[starts]
                ]

            self.train_window_starts = safe_starts(self.train_window_starts)
            self.heldout_window_starts = safe_starts(self.heldout_window_starts)
            self.reset_indices = self.reset_indices[~invalid[self.reset_indices]]
        if skill_sampling:
            self.reset_indices = self.reset_indices[
                ~self.unsafe_reset_frames[self.reset_indices]
            ]
        self._near_frames: th.Tensor | None = None
        self._train_near_pairs: th.Tensor | None = None
        self._heldout_near_pairs: th.Tensor | None = None
        self._train_situation_pools: tuple[th.Tensor, ...] | None = None
        self._heldout_situation_pools: tuple[th.Tensor, ...] | None = None
        self._train_maneuvers: tuple[list[SceneManeuver], ...] | None = None
        self._heldout_maneuvers: tuple[list[SceneManeuver], ...] | None = None
        self._train_grounded_choices: tuple[th.Tensor, th.Tensor] | None = None
        self._curated_pools: dict[bool, tuple[th.Tensor, ...]] = {}
        self._curated_maneuvers: dict[bool, tuple[list[SceneManeuver], ...]] = {}
        self._curated_labeled_pools: dict[bool, tuple[tuple[th.Tensor, ...], ...]] = {}
        self._curated_reset_pools: tuple[th.Tensor, ...] | None = None
        if skill_sampling:
            self.curated_pools()

    def _split_heldout(self, device: str | th.device, seed: int) -> None:
        split_rng = th.Generator(device=device).manual_seed(seed)
        eligible = [
            index for index, indices in enumerate(self.segment_frame_indices)
            if len(indices) > self.partition_span
        ]
        if self.heldout_size > 0 and len(eligible) > 1:
            order = th.tensor(eligible, device=device)[th.randperm(
                len(eligible), device=device, generator=split_rng
            )]
            cumulative = th.tensor(
                [len(self.segment_window_starts[index]) for index in order],
                device=device,
            ).cumsum(0)
            count = int(
                th.searchsorted(
                    cumulative,
                    th.tensor(self.heldout_size, device=device),
                ).item()
            ) + 1
            count = min(count, len(order) - 1)
            heldout_segments = set(order[:count].tolist())
            self.heldout_window_starts = th.cat([
                starts
                for index, starts in enumerate(self.segment_window_starts)
                if index in heldout_segments
            ])
            self.train_window_starts = th.cat([
                starts
                for index, starts in enumerate(self.segment_window_starts)
                if index not in heldout_segments
            ])
            self.reset_indices = th.cat([
                indices
                for index, indices in enumerate(self.segment_frame_indices)
                if index not in heldout_segments
            ])
            self._train_generator = th.Generator(device=device).manual_seed(seed)
            self._heldout_generator = th.Generator(device=device).manual_seed(seed + 1)
            return

        n_heldout = min(
            max(0, self.heldout_size),
            max(0, self.total_windows - self.partition_span - 1),
        )
        if n_heldout:
            first_heldout = self.window_starts[-n_heldout]
            train_stop = first_heldout - self.partition_span
            self.heldout_window_starts = self.window_starts[-n_heldout:]
            self.train_window_starts = self.window_starts[
                self.window_starts < train_stop
            ]
            self.reset_indices = self.real_frame_indices[
                self.real_frame_indices < first_heldout
            ]
        else:
            self.heldout_window_starts = self.window_starts[:0]
            self.train_window_starts = self.window_starts
            self.reset_indices = self.real_frame_indices
        self._train_generator = th.Generator(device=device).manual_seed(seed)
        self._heldout_generator = th.Generator(device=device).manual_seed(seed + 1)

    @property
    def train_total(self) -> int:
        return len(self.train_window_starts)

    @property
    def heldout_total(self) -> int:
        return len(self.heldout_window_starts)

    @staticmethod
    def _dedup_key(path: Path) -> tuple[str, ...]:
        """POV copies share {period}-{replay}; keep one canonical file."""
        parts = path.stem.split("-", 2)
        if len(parts) >= 3:
            return (parts[1], parts[2])
        return (path.stem,)

    def _sample_windows(
        self,
        starts: th.Tensor,
        n: int,
        device: str | th.device,
        generator: th.Generator,
    ) -> th.Tensor:
        if n < 1:
            raise ValueError("sample count must be positive")
        if len(starts) == 0:
            raise RuntimeError("no expert windows available")

        requested_device = th.device(device)
        if requested_device != self.frames.device:
            raise ValueError(
                "expert scenes and generated scenes must reside on the same device"
            )
        selected = th.randint(
            len(starts),
            (n,),
            device=self.frames.device,
            generator=generator,
        )
        chosen_starts = starts[selected]
        windows = self.frames[chosen_starts[:, None] + self.window_offsets]
        paired = self.opponent_pov_available[chosen_starts]
        if paired.any():
            opponent = paired & (th.rand(n, device=self.frames.device, generator=generator) < 0.5)
            if opponent.any():
                windows[opponent] = opponent_view(windows[opponent])
        return windows

    def _sample_curated(
        self, n: int, device: str | th.device, *, heldout: bool = False,
    ) -> th.Tensor:
        if n < 1 or th.device(device) != self.frames.device:
            raise ValueError("curated samples require a positive count on the expert device")
        weights = self.curated_weights(heldout=heldout)
        generator = self._heldout_generator if heldout else self._train_generator
        categories = th.multinomial(weights, n, replacement=True, generator=generator)
        pairs = th.empty((n, 2), dtype=th.long, device=self.frames.device)
        for category, pool in enumerate(self.curated_pools(heldout=heldout)):
            selected = (categories == category).nonzero(as_tuple=True)[0]
            if len(selected):
                pairs[selected] = pool[th.randint(
                    len(pool), (len(selected),),
                    generator=generator, device=pool.device,
                )]
        return self._windows_for_povs(pairs)

    def sample(self, n: int, device: str | th.device) -> th.Tensor:
        """Sample ``n`` training scene windows without crossing file boundaries.

        A second ego viewpoint is eligible only if that replay segment has a
        second stored expert POV. Physical scenes are shared between paired POVs.
        """
        if self.skill_sampling:
            return self._sample_curated(n, device)
        return self._sample_windows(
            self.train_window_starts, n, device, self._train_generator
        )

    def sample_heldout(self, n: int, device: str | th.device) -> th.Tensor:
        """Sample ``n`` held-out expert windows for discriminator evaluation."""
        if self.skill_sampling:
            return self._sample_curated(n, device, heldout=True)
        return self._sample_windows(
            self.heldout_window_starts, n, device, self._heldout_generator
        )

    def _near_pairs(self, heldout: bool = False) -> th.Tensor:
        """Cache near-ball windows only for actors with a stored expert POV."""
        cached = self._heldout_near_pairs if heldout else self._train_near_pairs
        if cached is not None:
            return cached
        if self.skill_sampling:
            groups = self.curated_labeled_pools(heldout=heldout)
            pools = [
                group[label] for group in groups
                for label in range(GROUND_MANEUVER_START) if label % 3 != 2
                and len(group[label])
            ]
            starts = self.heldout_window_starts if heldout else self.train_window_starts
            pairs = th.cat(pools) if pools else starts.new_empty((0, 2))
            if heldout:
                self._heldout_near_pairs = pairs
            else:
                self._train_near_pairs = pairs
            return pairs
        if self._near_frames is None:
            cars = th.stack((
                self.frames[:, BLUE_START:BLUE_START + 3],
                self.frames[:, ORANGE_START:ORANGE_START + 3],
            ), dim=1)
            distance = (self.frames[:, None, :3] - cars) * self.frames.new_tensor(POSITION_SCALE)
            self._near_frames = distance.square().sum(dim=-1) <= BALL_NEAR_DISTANCE ** 2
        starts = self.heldout_window_starts if heldout else self.train_window_starts
        found = []
        for chunk in starts.split(32_768):
            near = self._near_frames[chunk[:, None] + self.window_offsets].any(dim=1)
            near[:, 1] &= self.opponent_pov_available[chunk]
            choices = near.nonzero(as_tuple=False)
            if len(choices):
                found.append(th.stack((chunk[choices[:, 0]], choices[:, 1]), dim=-1))
        pairs = th.cat(found) if found else starts.new_empty((0, 2))
        if heldout:
            self._heldout_near_pairs = pairs
        else:
            self._train_near_pairs = pairs
        return pairs

    @property
    def near_total(self) -> int:
        return len(self._near_pairs())

    @property
    def heldout_near_total(self) -> int:
        return len(self._near_pairs(heldout=True))

    def sample_near(
        self, n: int, device: str | th.device, *, heldout: bool = False,
    ) -> th.Tensor:
        """Sample near-ball windows only from stored POVs in the requested split."""
        if n < 1 or th.device(device) != self.frames.device:
            raise ValueError("near-ball sample count must be positive and use the expert device")
        pairs = self._near_pairs(heldout)
        if not len(pairs):
            raise ValueError("no near-ball expert windows in this split")
        generator = self._heldout_generator if heldout else self._train_generator
        chosen = pairs[th.randint(len(pairs), (n,), device=self.frames.device, generator=generator)]
        return self._windows_for_povs(chosen)

    def _windows_for_povs(self, chosen: th.Tensor) -> th.Tensor:
        windows = self.frames[chosen[:, 0, None] + self.window_offsets]
        opponent = chosen[:, 1].bool()
        if opponent.any():
            windows[opponent] = opponent_view(windows[opponent])
        return windows

    def situation_pools(self, *, heldout: bool = False) -> tuple[th.Tensor, ...]:
        """Eligible (window start, focal actor) pairs for each situation band."""
        cached = self._heldout_situation_pools if heldout else self._train_situation_pools
        if cached is not None:
            return cached
        starts = self.heldout_window_starts if heldout else self.train_window_starts
        groups: list[list[th.Tensor]] = [[] for _ in range(N_SITUATIONS)]
        for chunk in starts.split(8_192):
            scenes = self.frames[chunk[:, None] + self.window_offsets]
            for actor, car_start in ((0, BLUE_START), (1, ORANGE_START)):
                focal_starts = chunk
                focal_scenes = scenes
                if actor:
                    available = self.opponent_pov_available[chunk]
                    if not available.any():
                        continue
                    focal_starts = chunk[available]
                    focal_scenes = scenes[available]
                labels = scene_situation_ids(focal_scenes, car_start)
                on_surface = focal_scenes[:, -1, car_start + CAR_BOOL_START] > 0.5
                for label in labels.unique().tolist():
                    if label < 2 * len(DISTANCE_BANDS):
                        selected = focal_starts[(labels == label) & on_surface]
                    else:
                        selected = focal_starts[labels == label]
                    if not len(selected):
                        continue
                    groups[label].append(th.stack((
                        selected, th.full_like(selected, actor),
                    ), dim=-1))
        pools = tuple(
            th.cat(group) if group else starts.new_empty((0, 2))
            for group in groups
        )
        if heldout:
            self._heldout_situation_pools = pools
        else:
            self._train_situation_pools = pools
        return pools

    def _grounded_choices(self) -> tuple[th.Tensor, th.Tensor]:
        """Cache physical training frames and eligible flat-ground POV flags."""
        if self._train_grounded_choices is None:
            starts = self.train_window_starts
            scored = starts + self.trajectory_length - 1
            blue = ((self.frames[scored, BLUE_START + CAR_BOOL_START] > 0.5)
                    & (self.frames[scored, BLUE_START + 14] > 0.65))
            orange = ((self.frames[scored, ORANGE_START + CAR_BOOL_START] > 0.5)
                      & (self.frames[scored, ORANGE_START + 14] > 0.65)
                      & self.opponent_pov_available[starts])
            eligible = blue | orange
            self._train_grounded_choices = (
                starts[eligible], th.stack((blue[eligible], orange[eligible]), dim=-1),
            )
        return self._train_grounded_choices

    def random_grounded_povs(self, n: int) -> th.Tensor:
        """Choose physical training frames uniformly, then a stored grounded POV."""
        starts, available = self._grounded_choices()
        if not len(starts):
            return starts.new_empty((0, 2))
        selected = th.randint(
            len(starts), (n,), device=starts.device, generator=self._train_generator,
        )
        choices = available[selected]
        actor = (~choices[:, 0]) | (
            choices[:, 1] & (th.rand(
                n, device=starts.device, generator=self._train_generator,
            ) < 0.5)
        )
        return th.stack((starts[selected], actor.long()), dim=-1)

    def maneuver_pools(self, *, heldout: bool = False) -> tuple[list[SceneManeuver], ...]:
        """Complete aerial, dribble, and flick sequences from stored POVs."""
        cached = self._heldout_maneuvers if heldout else self._train_maneuvers
        if cached is not None:
            return cached
        allowed = np.zeros(len(self.frames), dtype=bool)
        starts = self.heldout_window_starts if heldout else self.train_window_starts
        allowed[starts.cpu().numpy()] = True
        stored_opponent = self.opponent_pov_available.cpu().numpy()
        frame_distance = {
            actor: _ball_distances(self.frames, car_start).cpu().numpy()
            for actor, car_start in ((0, BLUE_START), (1, ORANGE_START))
        }
        frame_pose = {
            actor: (
                np.pad(recovery_surface_contact(
                    self.frames[1:], self.frames[:-1], car_start,
                ).cpu().numpy(), (1, 0)),
                self.frames[:, car_start + 2].cpu().numpy() * POSITION_SCALE[2],
                self.frames[:, car_start + 14].cpu().numpy(),
            )
            for actor, car_start in ((0, BLUE_START), (1, ORANGE_START))
        }
        control_features = {
            actor: self.frames[:, list(ground_feature_indices(car_start))].cpu().numpy()
            for actor, car_start in ((0, BLUE_START), (1, ORANGE_START))
        }
        touches = self.ego_touches.cpu().numpy() if self.ego_touches is not None else None
        contacts = (self.contact_frames.cpu().numpy()
                    if self.contact_frames is not None else None)
        goal_tail_steps = round(GOAL_EXCLUSION_SECONDS * TICKS_PER_SECOND / self.frame_skip)
        groups: list[list[SceneManeuver]] = [[] for _ in range(N_SITUATIONS)]
        offset = 0
        for length, goal_actor in zip(self.lengths, self.segment_goal_actors):
            count = max(0, length - self.trajectory_length + 1)
            if count and allowed[offset:offset + count].any():
                actors = (0, 1) if stored_opponent[offset] else (0,)
                for actor in actors:
                    goal_ends = np.zeros(count, dtype=bool)
                    goal_ends[-1] = actor == goal_actor
                    surface, height, up_z = frame_pose[actor]
                    scored = slice(offset + self.trajectory_length - 1, offset + length)
                    frame_dist = frame_distance[actor][offset:offset + length]
                    distance = frame_dist[:count].copy()
                    for step in range(1, self.trajectory_length):
                        np.minimum(distance, frame_dist[step:step + count], out=distance)
                    maneuvers = [*air_maneuvers(
                        allowed[offset:offset + count], surface[scored],
                        height[scored], up_z[scored], distance,
                        goal_ends=goal_ends,
                        touches=touches[scored, actor] if touches is not None else None,
                        allow_partial=touches is not None,
                    ), *ground_maneuvers(
                        allowed[offset:offset + count], control_features[actor][scored],
                        goal_ends=goal_ends,
                    )]
                    if (contacts is not None and goal_ends[-1]
                            and allowed[offset + count - 1]
                            and not any(maneuver.goal_terminal for maneuver in maneuvers)):
                        # A shot may land or release before the ball crosses the
                        # line. Keep its safe, untouched follow-through to goal.
                        candidates = [
                            maneuver for maneuver in maneuvers
                            if (0 < count - maneuver.recovery_stop <= goal_tail_steps
                                and count - maneuver.action_stop <= goal_tail_steps
                                and allowed[offset + maneuver.recovery_stop:
                                            offset + count].all()
                                and not contacts[
                                    offset + maneuver.action_stop + self.partition_span:
                                    offset + count + self.partition_span
                                ].any())
                        ]
                        if candidates:
                            latest = max(candidates, key=lambda clip: clip.action_stop)
                            maneuvers = [
                                replace(maneuver, recovery_stop=count, goal_terminal=True)
                                if maneuver is latest else maneuver
                                for maneuver in maneuvers
                            ]
                    for maneuver in maneuvers:
                        groups[maneuver.situation].append(replace(
                            maneuver,
                            setup_start=offset + maneuver.setup_start,
                            action_start=offset + maneuver.action_start,
                            action_stop=offset + maneuver.action_stop,
                            recovery_stop=offset + maneuver.recovery_stop,
                            actor=actor,
                        ))
            offset += length
        result = tuple(groups)
        if heldout:
            self._heldout_maneuvers = result
        else:
            self._train_maneuvers = result
        return result

    def _kickoff_window_mask(self, ball_position: np.ndarray) -> th.Tensor:
        """Take each real kickoff through first ball movement and brief follow-through."""
        mask = np.zeros(len(self.frames), dtype=bool)
        segments = []
        offset = 0
        for length in self.lengths:
            if length - self.partition_span >= self.kickoff_follow_through_steps:
                segments.append((offset, length))
            offset += length
        initial = self.frames[
            [offset + self.partition_span for offset, _ in segments]
        ].cpu().numpy()
        ball_velocity = self.frames[:, 3:6].cpu().numpy()
        for (offset, length), scene in zip(segments, initial):
            count = min(length - self.partition_span, self.kickoff_max_steps)
            first = offset + self.partition_span
            at_center = np.linalg.norm(scene[:2] * POSITION_SCALE[:2]) < 2 * BALL_RADIUS
            stationary = np.linalg.norm(scene[3:6] * BALL_MAX_SPEED) < 250
            spawn = (scene[BLUE_START + 1] * POSITION_SCALE[1] < -2_000
                     and scene[ORANGE_START + 1] * POSITION_SCALE[1] > 2_000
                     and scene[BLUE_START + 2] * POSITION_SCALE[2] < 120
                     and scene[ORANGE_START + 2] * POSITION_SCALE[2] < 120)
            height = scene[2] * POSITION_SCALE[2]
            if at_center and stationary and spawn and BALL_RADIUS - 20 < height < 150:
                flight = ball_position[first:first + count]
                velocity = ball_velocity[first:first + count]
                moved = (
                    (np.linalg.norm(flight[:, :2] * POSITION_SCALE[:2], axis=1)
                     > 1.5 * BALL_RADIUS)
                    | (np.linalg.norm(velocity * BALL_MAX_SPEED, axis=1) > 350)
                )
                if moved.any():
                    stop = min(
                        int(np.flatnonzero(moved)[0]) + self.kickoff_follow_through_steps,
                        count,
                    )
                    mask[offset:offset + stop] = True
        return th.from_numpy(mask).to(self.frames.device)

    def curated_pools(self, *, heldout: bool = False) -> tuple[th.Tensor, ...]:
        """The same complete skill clips and limited driving used by D and resets.

        Aerial touches need a real elevated contact; an aerial maneuver needs
        separate contacts while carrying a moving ball through the air.
        Touches remain *inside* expert discriminator windows, but cannot be reset
        targets. Each pair is a window start and a stored focal-player viewpoint.
        """
        if not self.skill_sampling:
            raise ValueError("curated replay pools require skill sampling")
        if heldout in self._curated_pools:
            return self._curated_pools[heldout]
        starts = self.heldout_window_starts if heldout else self.train_window_starts
        pools: list[list[th.Tensor]] = [[] for _ in SKILL_CATEGORIES]
        kept_clips: list[list[SceneManeuver]] = [[] for _ in range(N_SITUATIONS)]
        used = np.zeros((len(self.frames), N_CARS), dtype=bool)
        touches = self.ego_touches.cpu().numpy()
        ball_position = self.frames[:, :3].cpu().numpy()
        distances = {
            actor: _ball_distances(self.frames, car).cpu().numpy()
            for actor, car in enumerate((BLUE_START, ORANGE_START))
        }
        heights = {
            actor: self.frames[:, car + 2].cpu().numpy() * POSITION_SCALE[2]
            for actor, car in enumerate((BLUE_START, ORANGE_START))
        }
        for label, clips in enumerate(self.maneuver_pools(heldout=heldout)):
            for clip in clips:
                actor = clip.actor
                if isinstance(clip, AirManeuver):
                    scored = np.arange(clip.action_start, clip.action_stop) + self.partition_span
                    category = aerial_skill_category(
                        touches[scored, actor], heights[actor][scored],
                        ball_position[scored], distances[actor][scored],
                    )
                    if category is None:
                        continue
                    # A three-touch aerial can begin or end at a replay, goal,
                    # or valid-window boundary. Single touches still need both
                    # grounded setup and recovery (unless a goal ends the play).
                    if (category != AERIAL_MANEUVER_SKILL
                            and (clip.action_start - clip.setup_start
                                 < AERIAL_MIN_CONTEXT_STEPS
                                 or (not clip.goal_terminal
                                     and clip.recovery_stop - clip.action_stop
                                     < AERIAL_MIN_CONTEXT_STEPS))):
                        continue
                    clip = replace(clip, skill_category=category)
                else:
                    category = DRIBBLE_SKILL + label - GROUND_MANEUVER_START
                kept_clips[label].append(clip)
                indices = th.arange(clip.setup_start, clip.recovery_stop)
                used[indices.numpy(), actor] = True
                pools[category].append(th.stack((
                    indices, th.full_like(indices, actor),
                ), dim=-1))

        # Kickoff has its own quota, reaching beyond the initial approach to
        # the first challenge. Neither kickoff nor general driving duplicates
        # a selected aerial or ground-control skill phase.
        kickoff_starts = (
            self._kickoff_window_mask(ball_position) if self.kickoff_fraction
            else th.zeros(len(self.frames), dtype=th.bool, device=self.frames.device)
        )
        for actor, car in enumerate((BLUE_START, ORANGE_START)):
            eligible = starts
            if actor:
                eligible = eligible[self.opponent_pov_available[eligible]]
            scored = eligible + self.partition_span
            unused = ~th.from_numpy(used[:, actor]).to(eligible.device)[eligible]
            kickoff = eligible[kickoff_starts[eligible] & unused]
            if len(kickoff):
                pools[KICKOFF_SKILL].append(th.stack((
                    kickoff, th.full_like(kickoff, actor),
                ), dim=-1))
            driving = (
                (self.frames[scored, car + CAR_BOOL_START] > 0.5)
                & (self.frames[scored, car + 14] > 0.65)
                & unused & ~kickoff_starts[eligible]
            )
            chosen = eligible[driving].cpu()
            if len(chosen):
                pools[DRIVING_SKILL].append(th.stack((
                    chosen, th.full_like(chosen, actor),
                ), dim=-1))

        result = tuple(
            th.cat(group).to(self.frames.device)
            if group else th.empty((0, 2), dtype=th.long, device=self.frames.device)
            for group in pools
        )
        self._curated_pools[heldout] = result
        self._curated_maneuvers[heldout] = tuple(kept_clips)
        if not heldout:
            eligible = th.zeros(len(self.frames), dtype=th.bool, device=self.frames.device)
            eligible[self.reset_indices] = True
            reset_pools = []
            for pairs in result:
                frames = pairs[:, 0] + self.partition_span
                reset_pools.append(th.unique(frames[eligible[frames]]))
            self._curated_reset_pools = tuple(reset_pools)
            if any(len(pool) for pool in reset_pools):
                self.reset_indices = th.unique(th.cat(reset_pools))
            else:
                self.reset_indices = self.reset_indices[:0]
        return result

    def curated_labeled_pools(
        self, *, heldout: bool = False,
    ) -> tuple[tuple[th.Tensor, ...], ...]:
        """Match every generated window to a curated expert of the same scene bin."""
        if heldout in self._curated_labeled_pools:
            return self._curated_labeled_pools[heldout]
        categories = []
        for pairs in self.curated_pools(heldout=heldout):
            groups: list[list[th.Tensor]] = [[] for _ in range(GROUND_MANEUVER_START)]
            for chunk in pairs.split(8_192):
                windows = self.frames[chunk[:, 0, None] + self.window_offsets]
                labels = scene_situation_ids(windows)
                opposite = chunk[:, 1].bool()
                if opposite.any():
                    labels[opposite] = scene_situation_ids(windows[opposite], ORANGE_START)
                for label in labels.unique().tolist():
                    groups[label].append(chunk[labels == label])
            categories.append(tuple(
                th.cat(group) if group else pairs[:0] for group in groups
            ))
        result = tuple(categories)
        self._curated_labeled_pools[heldout] = result
        return result

    def curated_weights(self, *, heldout: bool = False) -> th.Tensor:
        """Split the remaining share 3:3:2:1 across aerials, dribble and flick."""
        driving = self.driving_fraction
        kickoff = self.kickoff_fraction
        skills = 1 - driving - kickoff
        weights = th.tensor((
            skills / 3, skills / 3, skills * 2 / 9, skills / 9,
            driving, kickoff,
        ), device=self.frames.device)
        available = self.curated_pools(heldout=heldout)
        present = th.tensor(
            [bool(len(pool)) for pool in available], device=weights.device,
        )
        # Keep the total aerial share when only one type is represented.
        if present[:2].any():
            weights[:2] *= 2 / present[:2].sum()
        weights *= present
        if not bool(weights.sum()):
            raise ValueError("no eligible curated aerial, dribble, flick, driving, or kickoff windows")
        return weights / weights.sum()

    def reset_dataset(self) -> TensorDataset:
        """Sample eligible training frame IDs without copying expert scenes."""
        return reset_index_dataset(self.reset_indices)


class HistoricalReplayBuffer:
    """Bounded FIFO replay buffer for generated scene windows on the learner device."""

    def __init__(
        self,
        capacity: int,
        trajectory_length: int,
        device: str | th.device,
        seed: int = 0,
    ) -> None:
        if capacity < 0:
            raise ValueError("history capacity must be non-negative")
        if trajectory_length < 2:
            raise ValueError("trajectory length must be at least 2")
        self.capacity = capacity
        self.trajectory_length = trajectory_length
        self.device = th.device(device)
        self.buffer: th.Tensor | None = None
        self.size = 0
        self.start = 0
        self.rng = th.Generator(device=self.device).manual_seed(seed)

    def add(self, windows: th.Tensor, add_size: int) -> None:
        """Store a bounded random subset of ``windows`` (detached)."""
        if add_size <= 0 or len(windows) == 0 or self.capacity == 0:
            return
        if windows.shape[1:] != (self.trajectory_length, SCENE_SIZE):
            raise ValueError("historical windows have the wrong shape")
        windows = windows.detach().to(self.device, non_blocking=False)
        if len(windows) > add_size:
            perm = th.randperm(len(windows), device=windows.device, generator=self.rng)
            windows = windows[perm[:add_size]]
        if len(windows) > self.capacity:
            windows = windows[-self.capacity :]

        if self.buffer is None:
            self.buffer = th.empty(
                self.capacity,
                self.trajectory_length,
                SCENE_SIZE,
                dtype=windows.dtype,
                device=self.device,
            )

        available = self.capacity - self.size
        filling = min(len(windows), available)
        if filling:
            self.buffer[self.size : self.size + filling] = windows[:filling]
            self.size += filling

        remaining = len(windows) - filling
        if remaining:
            positions = (
                self.start
                + th.arange(remaining, device=self.device)
            ) % self.capacity
            self.buffer[positions] = windows[filling:]
            self.start = (self.start + remaining) % self.capacity

    def sample(self, n: int, device: str | th.device) -> th.Tensor:
        """Sample ``n`` historical windows, or fewer if the buffer is not full."""
        if self.size == 0:
            return th.empty(
                0,
                self.trajectory_length,
                SCENE_SIZE,
                device=device,
            )
        n = min(n, self.size)
        perm = th.randperm(self.size, device=self.device, generator=self.rng)[:n]
        pos = (self.start + perm) % self.capacity
        return self.buffer[pos].to(device, non_blocking=False)


class RecencyReplayBuffer:
    """Sample a recent FIFO plus a uniform reservoir of all past windows."""

    def __init__(
        self,
        capacity: int,
        trajectory_length: int,
        device: str | th.device,
        seed: int = 0,
        reservoir_fraction: float = 0.25,
    ) -> None:
        if capacity < 2:
            raise ValueError("recency replay capacity must be at least two")
        if not math.isfinite(reservoir_fraction) or not 0 < reservoir_fraction <= 0.5:
            raise ValueError("reservoir fraction must be in (0, 0.5]")
        self.reservoir_fraction = reservoir_fraction
        self.reservoir_capacity = max(1, round(capacity * reservoir_fraction))
        self.recent = HistoricalReplayBuffer(
            capacity - self.reservoir_capacity, trajectory_length, device, seed,
        )
        self.trajectory_length = trajectory_length
        self.device = th.device(device)
        self.reservoir: th.Tensor | None = None
        self.reservoir_size = 0
        self.seen = 0
        self.rng = th.Generator(device=self.device).manual_seed(seed + 1)

    @property
    def size(self) -> int:
        return self.recent.size + self.reservoir_size

    def add(self, windows: th.Tensor, add_size: int) -> None:
        if add_size <= 0 or len(windows) == 0:
            return
        if windows.shape[1:] != (self.trajectory_length, SCENE_SIZE):
            raise ValueError("historical windows have the wrong shape")
        windows = windows.detach().to(self.device, non_blocking=False)
        if len(windows) > add_size:
            chosen = th.randperm(len(windows), device=self.device, generator=self.rng)
            windows = windows[chosen[:add_size]]

        self.recent.add(windows, len(windows))
        if self.reservoir is None:
            self.reservoir = windows.new_empty(
                (self.reservoir_capacity, self.trajectory_length, SCENE_SIZE)
            )
        filling = min(self.reservoir_capacity - self.reservoir_size, len(windows))
        if filling:
            self.reservoir[
                self.reservoir_size:self.reservoir_size + filling
            ] = windows[:filling]
            self.reservoir_size += filling

        remaining = windows[filling:]
        if len(remaining):
            # Reservoir sampling: each later window gets one uniformly random
            # slot in the stream so far. The last replacement wins per slot.
            counts = th.arange(
                self.seen + filling + 1, self.seen + len(windows) + 1,
                dtype=th.float64, device=self.device,
            )
            slots = (
                th.rand(len(remaining), dtype=th.float64, device=self.device,
                        generator=self.rng) * counts
            ).long()
            admitted = (slots < self.reservoir_capacity).nonzero(as_tuple=True)[0]
            last = th.full(
                (self.reservoir_capacity,), -1, dtype=th.long, device=self.device,
            )
            last.scatter_reduce_(0, slots[admitted], admitted, reduce="amax")
            replaced = (last >= 0).nonzero(as_tuple=True)[0]
            self.reservoir[replaced] = remaining[last[replaced]]
        self.seen += len(windows)

    def sample(self, n: int, device: str | th.device) -> th.Tensor:
        if self.size == 0:
            return th.empty(0, self.trajectory_length, SCENE_SIZE, device=device)
        n = min(n, self.size)
        n_old = min(int(n * self.reservoir_fraction), self.reservoir_size)
        if n > 1 and self.reservoir_size:
            n_old = max(1, n_old)
        n_recent = min(n - n_old, self.recent.size)
        n_old += min(n - n_recent - n_old, self.reservoir_size - n_old)

        recent = self.recent.sample(n_recent, device)
        if n_old == 0:
            return recent
        indices = th.randperm(
            self.reservoir_size, device=self.device, generator=self.rng,
        )[:n_old]
        old = self.reservoir[indices].to(device, non_blocking=False)
        return th.cat((recent, old), dim=0) if n_recent else old


class SceneDiscriminator(nn.Module):
    """Structured scene-level discriminator for short 1v1 trajectory windows."""

    def __init__(
        self,
        frame_embedding: int,
        temporal_hidden: int,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        if min(frame_embedding, temporal_hidden, hidden_size) < 1:
            raise ValueError("discriminator dimensions must be positive")
        self.ball_encoder = nn.Sequential(
            nn.Linear(BALL_SIZE, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU(),
        )
        self.car_encoder = nn.Sequential(
            nn.Linear(CAR_SIZE + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            frame_embedding * 3, temporal_hidden, batch_first=True
        )
        self.head = nn.Linear(temporal_hidden, 1)

    def forward(self, windows: th.Tensor) -> th.Tensor:
        if windows.ndim != 3 or windows.shape[-1] != SCENE_SIZE:
            raise ValueError("unified discriminator requires two-car scene windows")
        B, T, _ = windows.shape
        ball = windows[..., :BALL_SIZE]
        cars = windows[..., BALL_SIZE:SCENE_SIZE].view(B, T, N_CARS, CAR_SIZE)
        blue, orange = cars[:, :, 0], cars[:, :, 1]

        sign = th.tensor([1.0, -1.0], device=windows.device, dtype=windows.dtype)
        blue_in = th.cat([blue, sign[0].expand_as(blue[..., :1])], dim=-1)
        orange_in = th.cat([orange, sign[1].expand_as(orange[..., :1])], dim=-1)

        ball_emb = self.ball_encoder(ball)
        blue_emb = self.car_encoder(blue_in)
        orange_emb = self.car_encoder(orange_in)

        frame_embedding = th.cat([ball_emb, blue_emb, orange_emb], dim=-1)
        gru_out, _ = self.gru(frame_embedding)
        return self.head(gru_out[:, -1]).squeeze(-1)


class FactorizedSceneDiscriminator(nn.Module):
    """Score ego motion and ball control with fixed initial other-car context.

    Only the first frame of teammates and opponents is visible to either head;
    their subsequent actions cannot become a shortcut for imitation reward.
    Use ``actor_view`` to score each player's own window in four-car scenes.
    """

    factorized = True

    def __init__(
        self, frame_embedding: int, temporal_hidden: int, hidden_size: int = 128,
        *, n_cars: int = N_CARS,
    ) -> None:
        super().__init__()
        if min(frame_embedding, temporal_hidden, hidden_size) < 1:
            raise ValueError("discriminator dimensions must be positive")
        if n_cars not in (N_CARS, DOUBLES_N_CARS):
            raise ValueError("factorized discriminator needs two or four cars")
        self.n_cars = n_cars
        self.scene_size = BALL_SIZE + n_cars * CAR_SIZE
        self.other_context_size = (n_cars - 1) * (CAR_SIZE + 6)
        self.car_encoder = nn.Sequential(
            nn.Linear(CAR_SIZE + 6 + self.other_context_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding), nn.ReLU(),
        )
        self.ball_encoder = nn.Sequential(
            nn.Linear(BALL_SIZE + 6 + self.other_context_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding), nn.ReLU(),
        )
        self.car_gru = nn.GRU(frame_embedding, temporal_hidden, batch_first=True)
        self.ball_gru = nn.GRU(frame_embedding, temporal_hidden, batch_first=True)
        self.car_head = nn.Linear(temporal_hidden, 1)
        self.ball_head = nn.Linear(temporal_hidden, 1)

    def forward(self, windows: th.Tensor) -> th.Tensor:
        if windows.ndim != 3 or windows.shape[-1] != self.scene_size:
            raise ValueError("factorized discriminator needs [batch, frames, scene] windows")
        ball = windows[..., :BALL_SIZE]
        ego = windows[..., BLUE_START:BLUE_START + CAR_SIZE]
        relative_position = ball[..., :3] - ego[..., :3]
        relative_velocity = (
            ball[..., 3:6] - ego[..., 3:6] * (CAR_MAX_SPEED / BALL_MAX_SPEED)
        )
        others = windows[:, 0, BLUE_START + CAR_SIZE:].reshape(
            -1, self.n_cars - 1, CAR_SIZE,
        )
        relative_others = others[..., :6] - ego[:, 0, None, :6]
        initial_others = th.cat((others, relative_others), dim=-1).flatten(1)
        context = initial_others[:, None].expand(-1, windows.shape[1], -1)
        # Hold context fixed so the car head cannot classify subsequent ball motion.
        initial_context = th.cat((relative_position[:, 0], relative_velocity[:, 0]), dim=-1)
        car_input = th.cat((
            ego, initial_context[:, None].expand(-1, windows.shape[1], -1), context,
        ), dim=-1)
        ball_input = th.cat((ball, relative_position, relative_velocity, context), dim=-1)
        car_features, _ = self.car_gru(self.car_encoder(car_input))
        ball_features, _ = self.ball_gru(self.ball_encoder(ball_input))
        return th.cat((
            self.car_head(car_features[:, -1]), self.ball_head(ball_features[:, -1]),
        ), dim=-1)


def load_discriminator_state(discriminator: nn.Module, state: dict[str, th.Tensor]) -> bool:
    """Expand legacy factorized inputs without changing their initial predictions."""
    upgraded = False
    if isinstance(discriminator, FactorizedSceneDiscriminator):
        layers = (
            ("car_encoder.0.weight", discriminator.car_encoder[0].weight, CAR_SIZE + 6),
            ("ball_encoder.0.weight", discriminator.ball_encoder[0].weight, BALL_SIZE + 6),
        )
        if all(key in state and state[key].shape == (weight.shape[0], old_width)
               for key, weight, old_width in layers):
            state = state.copy()
            for key, weight, old_width in layers:
                state[key] = F.pad(state[key], (0, weight.shape[1] - old_width))
            upgraded = True
    discriminator.load_state_dict(state)
    return upgraded


def expand_factorized_optimizer_state(
    optimizer: th.optim.Optimizer, discriminator: FactorizedSceneDiscriminator,
) -> None:
    """Preserve old Adam moments and initialize new opponent columns to zero."""
    for layer, old_width in (
        (discriminator.car_encoder[0], CAR_SIZE + 6),
        (discriminator.ball_encoder[0], BALL_SIZE + 6),
    ):
        for name, value in optimizer.state[layer.weight].items():
            if isinstance(value, th.Tensor) and value.ndim == 2:
                if value.shape == (layer.weight.shape[0], old_width):
                    optimizer.state[layer.weight][name] = F.pad(
                        value, (0, layer.weight.shape[1] - old_width),
                    )
                elif value.shape != layer.weight.shape:
                    raise ValueError(f"incompatible factorized discriminator optimizer {name}")


class CuratedReplayResetTransform:
    """Draw safe states from balanced aerial-touch, carry and ground skill clips."""

    def __init__(self, expert: ExpertSceneDataset) -> None:
        if not expert.skill_sampling or expert._curated_reset_pools is None:
            raise ValueError("skill-filtered reset sampling requires curated replay pools")
        self.expert = expert
        self.pools = expert._curated_reset_pools
        weights = expert.curated_weights().clone()
        weights *= th.tensor(
            [bool(len(pool)) for pool in self.pools], device=weights.device,
        )
        if not bool(weights.sum()):
            raise ValueError("no safe curated replay reset frames")
        self.weights = weights / weights.sum()
        self.counts = th.zeros(len(SKILL_CATEGORIES), dtype=th.long, device=weights.device)

    def __call__(self, sample: TensorBatch, context: ResetContext) -> TensorBatch:
        categories = th.multinomial(
            self.weights, len(sample), replacement=True, generator=context.generator,
        )
        self.counts += th.bincount(categories, minlength=len(SKILL_CATEGORIES))
        indices = sample["frame_index"].clone()
        for category, pool in enumerate(self.pools):
            chosen = (categories == category).nonzero(as_tuple=True)[0]
            if len(chosen):
                indices[chosen] = pool[th.randint(
                    len(pool), (len(chosen),), device=pool.device,
                    generator=context.generator,
                )]
        return sample.replace_fields(frame_index=indices).with_fields(
            skill_category=categories,
        )

    def take_skill_fractions(self) -> dict[str, float]:
        total = int(self.counts.sum())
        if not total:
            return {}
        fractions = (self.counts.float() / total).tolist()
        self.counts.zero_()
        return {
            f"reset_{name}_fraction": fraction
            for name, fraction in zip(SKILL_CATEGORIES, fractions)
        }


class ConfidentExpertResetTransform:
    """Favor confidently expert window starts at reset, without scanning the corpus."""

    def __init__(
        self,
        expert: ExpertSceneDataset,
        dataset: TensorDataset,
        discriminator: SceneDiscriminator | FactorizedSceneDiscriminator,
        microbatch_size: int,
    ) -> None:
        if microbatch_size < 1:
            raise ValueError("reset scoring microbatch size must be positive")
        if len(dataset) != len(expert.reset_indices) or dataset.device != expert.frames.device:
            raise ValueError("reset dataset must match the expert training frames")
        self.expert = expert
        self.dataset = dataset
        self.discriminator = discriminator
        self.microbatch_size = microbatch_size
        resettable = th.zeros(len(expert.frames), dtype=th.bool, device=expert.frames.device)
        resettable[expert.reset_indices] = True
        if expert.skill_sampling:
            self.category_starts = tuple(
                pool - expert.partition_span for pool in expert._curated_reset_pools
            )
            self.candidate_starts = th.cat(self.category_starts)
        else:
            self.category_starts = None
            self.candidate_starts = expert.train_window_starts[
                resettable[expert.train_window_starts]
            ]
        self.ready = False
        self._total = 0
        self._mined = 0

    def _confidence(self, windows: th.Tensor) -> th.Tensor:
        logits = self.discriminator(windows)
        if getattr(self.discriminator, "factorized", False):
            expert_probability = th.sigmoid(-logits)
            near = nearest_ball_distance(windows) <= BALL_NEAR_DISTANCE
            return th.where(
                near, expert_probability.mean(dim=-1), expert_probability[:, 0],
            )
        return th.sigmoid(-logits)

    def _score(self, starts: th.Tensor) -> th.Tensor:
        scores = []
        was_training = self.discriminator.training
        with th.no_grad():
            self.discriminator.eval()
            try:
                for chunk in starts.split(self.microbatch_size):
                    windows = self.expert.frames[chunk[:, None] + self.expert.window_offsets]
                    confidence = self._confidence(windows)
                    paired = self.expert.opponent_pov_available[chunk]
                    if paired.any():
                        confidence[paired] = th.maximum(
                            confidence[paired],
                            self._confidence(opponent_view(windows[paired])),
                        )
                    scores.append(confidence)
            finally:
                self.discriminator.train(was_training)
        return th.cat(scores)

    def __call__(self, sample: TensorBatch, context: ResetContext) -> TensorBatch:
        self._total += len(sample)
        if not self.ready or not len(self.candidate_starts):
            return sample

        device = self.dataset.device
        selected = (th.rand(
            len(sample), device=device, generator=context.generator,
        ) < RESET_MINING_FRACTION).nonzero(as_tuple=True)[0]
        if not len(selected):
            return sample

        if self.category_starts is None or "skill_category" not in sample:
            starts = self.candidate_starts[th.randint(
                len(self.candidate_starts),
                (len(selected), RESET_MINING_CANDIDATES),
                device=device, generator=context.generator,
            )]
        else:
            starts = th.empty(
                (len(selected), RESET_MINING_CANDIDATES),
                dtype=th.long, device=device,
            )
            categories = sample["skill_category"][selected]
            for category, pool in enumerate(self.category_starts):
                positions = (categories == category).nonzero(as_tuple=True)[0]
                if len(positions):
                    starts[positions] = pool[th.randint(
                        len(pool), (len(positions), RESET_MINING_CANDIDATES),
                        device=device, generator=context.generator,
                    )]
        confidence = self._score(starts.flatten()).view(-1, RESET_MINING_CANDIDATES)
        best_score, best_candidate = confidence.max(dim=1)
        accepted = best_score >= RESET_MINING_MIN_CONFIDENCE
        if not accepted.any():
            return sample

        selected = selected[accepted]
        best_starts = starts[accepted, best_candidate[accepted]]
        if self.category_starts is not None:
            best_starts = best_starts + self.expert.partition_span
        self._mined += len(selected)
        return sample.replace_fields(
            frame_index=sample["frame_index"].index_copy(0, selected, best_starts),
        )

    def take_mined_fraction(self) -> float:
        fraction = self._mined / self._total if self._total else 0.0
        self._mined = self._total = 0
        return fraction


class SceneDiscriminatorLoss:
    """BCE-with-logits loss for generated-vs-expert scene windows."""

    def __init__(
        self, discriminator: SceneDiscriminator | FactorizedSceneDiscriminator,
    ) -> None:
        self.discriminator = discriminator

    def __call__(self, batch: TensorBatch) -> LossOutput:
        logit = self.discriminator(batch["window"])
        target = batch["is_agent"]
        agent = target.bool()
        metrics = {}
        if getattr(self.discriminator, "factorized", False):
            if logit.shape != (len(target), 2):
                raise ValueError("factorized discriminator must return car and ball logits")
            car_loss = F.binary_cross_entropy_with_logits(logit[:, 0], target)
            ball_errors = F.binary_cross_entropy_with_logits(
                logit[:, 1], target, reduction="none",
            )
            weights = batch.get("ball_weight")
            if weights is None:
                weights = ball_responsibility(batch["window"]).square()
                class_losses = [
                    (ball_errors[chosen] * weights[chosen]).sum()
                    / weights[chosen].sum().clamp_min(1e-6)
                    for chosen in (agent, ~agent) if chosen.any()
                ]
                ball_loss = th.stack(class_losses).mean()
            else:
                ball_loss = (ball_errors * weights).mean()
            loss = (car_loss + ball_loss) * 0.5
            metrics = {
                "car_loss": car_loss.detach(), "ball_loss": ball_loss.detach(),
                "near_ball_fraction": (nearest_ball_distance(batch["window"])
                                       <= BALL_NEAR_DISTANCE).float().mean(),
            }
        else:
            loss = F.binary_cross_entropy_with_logits(logit, target)

        with th.no_grad():
            agent_score = th.sigmoid(logit[agent]).mean()
            expert_score = th.sigmoid(logit[~agent]).mean()
            agent_accuracy = (logit[agent] > 0.0).float().mean()
            expert_accuracy = (logit[~agent] <= 0.0).float().mean()

        return LossOutput(
            loss,
            {
                "loss": loss,
                "agent_score": agent_score,
                "expert_score": expert_score,
                "agent_accuracy": agent_accuracy,
                "expert_accuracy": expert_accuracy,
                **metrics,
            },
        )


@dataclass(frozen=True)
class GeneratedSceneTimeline:
    """CPU maneuver features shared by the rollout tracker and replay sampler."""

    valid: np.ndarray
    touches: np.ndarray
    grounded: np.ndarray
    height: np.ndarray
    up_z: np.ndarray
    distance: np.ndarray
    control: np.ndarray
    goal_ends: np.ndarray


def generated_scene_timeline(
    windows: th.Tensor, indices: th.Tensor, n_envs: int,
    episode_end: th.Tensor | None = None,
    ego_ball_touch: th.Tensor | None = None,
    goal_scored: th.Tensor | None = None,
) -> GeneratedSceneTimeline:
    if n_envs < 1 or len(windows) % n_envs:
        raise ValueError("generated windows must be time-major by actor")
    steps = len(windows) // n_envs
    if episode_end is not None and episode_end.shape != (steps, n_envs):
        raise ValueError("episode ends must match the generated actor timeline")
    if ego_ball_touch is not None and ego_ball_touch.shape != (steps, n_envs):
        raise ValueError("ball touches must match the generated actor timeline")
    if goal_scored is not None and goal_scored.shape != (steps, n_envs):
        raise ValueError("scored goals must match the generated actor timeline")
    if goal_scored is not None and episode_end is None:
        raise ValueError("scored goals require episode endings")
    valid = th.zeros(len(windows), dtype=th.bool, device=windows.device)
    valid[indices] = True
    scored = (goal_scored.bool().clone() if goal_scored is not None
              else th.zeros((steps, n_envs), dtype=th.bool, device=windows.device))
    if episode_end is not None:
        valid &= ~(episode_end.bool() & ~scored).reshape(-1)
        scored &= episode_end.bool()
    else:
        scored.zero_()
    goal_ends = (scored & valid.reshape(steps, n_envs)).cpu().numpy()
    scenes = windows[:, -1]
    return GeneratedSceneTimeline(
        valid=valid.reshape(steps, n_envs).cpu().numpy(),
        touches=(ego_ball_touch.bool().cpu().numpy() if ego_ball_touch is not None
                 else np.zeros((steps, n_envs), dtype=bool)),
        grounded=recovery_surface_contact(scenes, windows[:, -2]).reshape(
            steps, n_envs,
        ).cpu().numpy(),
        height=(scenes[:, BLUE_START + 2] * POSITION_SCALE[2]).reshape(
            steps, n_envs,
        ).cpu().numpy(),
        up_z=scenes[:, BLUE_START + 14].reshape(steps, n_envs).cpu().numpy(),
        distance=nearest_ball_distance(windows).reshape(steps, n_envs).cpu().numpy(),
        control=scenes[:, list(ground_feature_indices(BLUE_START))].reshape(
            steps, n_envs, 14,
        ).cpu().numpy(),
        goal_ends=goal_ends,
    )


def generated_maneuver_pools(
    windows: th.Tensor,
    indices: th.Tensor,
    n_envs: int,
    episode_end: th.Tensor | None = None,
    *,
    ego_ball_touch: th.Tensor | None = None,
    goal_scored: th.Tensor | None = None,
    timeline: GeneratedSceneTimeline | None = None,
) -> tuple[list[SceneManeuver], ...]:
    """Find complete aerial and ground-control maneuvers per valid actor timeline."""
    if timeline is None:
        timeline = generated_scene_timeline(
            windows, indices, n_envs, episode_end, ego_ball_touch, goal_scored,
        )
    steps = len(windows) // n_envs
    timelines = timeline.valid
    scenes = windows.reshape(steps, n_envs, *windows.shape[1:])
    groups: list[list[SceneManeuver]] = [[] for _ in range(N_SITUATIONS)]
    for actor in range(n_envs):
        if timelines[:, actor].sum() < 4:
            continue
        for maneuver in (*air_maneuvers(
            timelines[:, actor], timeline.grounded[:, actor], timeline.height[:, actor],
            timeline.up_z[:, actor], timeline.distance[:, actor],
            goal_ends=timeline.goal_ends[:, actor],
        ), *ground_maneuvers(
            timelines[:, actor], timeline.control[:, actor],
            goal_ends=timeline.goal_ends[:, actor],
        )):
            if (isinstance(maneuver, AirManeuver)
                    and not maneuver.goal_terminal
                    and maneuver.recovery_stop - maneuver.action_stop < AERIAL_MIN_CONTEXT_STEPS):
                continue
            # Short recovery at the rollout edge belongs to the cross-rollout tracker.
            recovery_steps = (AERIAL_RECOVERY_STEPS if isinstance(maneuver, AirManeuver)
                              else MANEUVER_RECOVERY_STEPS)
            if (not maneuver.goal_terminal and timelines[-1, actor]
                    and maneuver.recovery_stop == steps
                    and maneuver.recovery_stop - maneuver.action_stop < recovery_steps):
                continue
            if isinstance(maneuver, AirManeuver):
                maneuver = replace(
                    maneuver, skill_category=generated_air_skill(
                        scenes[:, actor], maneuver, timeline.touches[:, actor],
                    ),
                )
            groups[maneuver.situation].append(replace(
                maneuver,
                setup_start=maneuver.setup_start * n_envs + actor,
                action_start=maneuver.action_start * n_envs + actor,
                action_stop=maneuver.action_stop * n_envs + actor,
                recovery_stop=maneuver.recovery_stop * n_envs + actor,
                stride=n_envs,
            ))
    return tuple(groups)


def aligned_maneuver_windows(
    agent: SceneManeuver,
    expert: SceneManeuver,
    budget: int,
    device: th.device,
) -> tuple[th.Tensor, th.Tensor]:
    """Pair causal windows covering setup, the entire action, and recovery.

    Long phases are spaced across their entire span. Short phases can repeat a
    window so both actors contribute the same number at each relative phase.
    """
    phases = ("setup_start", "action_start", "action_stop", "recovery_stop")
    desired = [
        min(max(
            (getattr(agent, end) - getattr(agent, start)) // agent.stride,
            (getattr(expert, end) - getattr(expert, start)) // expert.stride,
        ), maximum)
        for start, end, maximum in zip(
            phases[:-1], phases[1:], (AERIAL_SETUP_STEPS, 64, AERIAL_RECOVERY_STEPS)
        )
    ]
    if (agent.action_start == agent.setup_start
            or expert.action_start == expert.setup_start):
        desired[0] = 0
    if (agent.recovery_stop == agent.action_stop
            or expert.recovery_stop == expert.action_stop):
        desired[2] = 0
    counts = [int(desired[0] > 0), 2, int(desired[2] > 0)]
    if budget < sum(counts):
        empty = th.empty(0, dtype=th.long, device=device)
        return empty, empty.reshape(0, 1).expand(0, 2)
    remaining = budget - sum(counts)
    extra = min(remaining, desired[1] - counts[1])
    counts[1] += extra
    remaining -= extra
    # Keep approach and recovery equally visible when the budget is tight.
    while remaining and (counts[0] < desired[0] or counts[2] < desired[2]):
        for phase in (0, 2):
            if remaining and counts[phase] < desired[phase]:
                counts[phase] += 1
                remaining -= 1

    def indices(flight: SceneManeuver) -> th.Tensor:
        segments = []
        for start, end, count in zip(phases[:-1], phases[1:], counts):
            if not count:
                continue
            length = (getattr(flight, end) - getattr(flight, start)) // flight.stride
            offsets = th.linspace(0, length - 1, count, device=device).round().long()
            segments.append(getattr(flight, start) + offsets * flight.stride)
        return th.cat(segments)

    agent_indices = indices(agent)
    expert_indices = indices(expert)
    expert_pairs = th.stack((
        expert_indices, th.full_like(expert_indices, expert.actor),
    ), dim=-1)
    return agent_indices, expert_pairs


@dataclass(frozen=True)
class CompletedSceneManeuver:
    windows: th.Tensor
    span: SceneManeuver


@dataclass(frozen=True)
class PendingAirManeuver:
    windows: th.Tensor
    action_start: int
    grounded: np.ndarray
    height: np.ndarray
    up_z: np.ndarray
    distance: np.ndarray
    touches: np.ndarray


@dataclass(frozen=True)
class PendingGroundManeuver:
    windows: th.Tensor
    action_start: int
    features: np.ndarray


class GeneratedManeuverTracker:
    """Keep unfinished aerial and ball-control maneuvers across rollouts."""

    def __init__(self) -> None:
        self.n_envs: int | None = None
        self.pending: dict[int, PendingAirManeuver] = {}
        self.pending_ground: dict[int, PendingGroundManeuver] = {}
        self.ready: tuple[list[CompletedSceneManeuver], ...] = tuple(
            [] for _ in range(N_SITUATIONS)
        )

    def clear_ready(self) -> None:
        self.ready = tuple([] for _ in range(N_SITUATIONS))

    def _archive(self, windows: th.Tensor, flight: SceneManeuver) -> None:
        start = flight.setup_start
        scene_windows = windows[start:flight.recovery_stop].detach().clone()
        span = replace(
            flight, setup_start=0, action_start=flight.action_start - start,
            action_stop=flight.action_stop - start,
            recovery_stop=flight.recovery_stop - start,
        )
        group = self.ready[span.situation]
        group.append(CompletedSceneManeuver(scene_windows, span))
        if len(group) > MANEUVER_ARCHIVE_PER_SITUATION:
            group.pop(0)

    def feed(
        self, windows: th.Tensor, indices: th.Tensor, n_envs: int,
        episode_end: th.Tensor | None = None,
        *, ego_ball_touch: th.Tensor | None = None,
        goal_scored: th.Tensor | None = None,
        timeline: GeneratedSceneTimeline | None = None,
    ) -> None:
        """Accumulate train-split maneuvers; a reset or invalid window breaks one."""
        if timeline is None:
            timeline = generated_scene_timeline(
                windows, indices, n_envs, episode_end, ego_ball_touch, goal_scored,
            )
        steps = len(windows) // n_envs
        touches = timeline.touches
        if self.n_envs != n_envs:
            self.n_envs = n_envs
            self.pending.clear()
            self.pending_ground.clear()
            self.clear_ready()
        valid = timeline.valid
        goal_ends = timeline.goal_ends
        grounded = timeline.grounded
        height = timeline.height
        up_z = timeline.up_z
        distance = timeline.distance
        control = timeline.control
        scenes = windows.reshape(steps, n_envs, *windows.shape[1:])

        previous = self.pending
        self.pending = {}
        for actor, partial in previous.items():
            invalid = np.flatnonzero(~valid[:, actor])
            count = int(invalid[0]) if len(invalid) else steps
            goals = np.flatnonzero(goal_ends[:count, actor])
            count = min(count, int(goals[0]) + 1) if len(goals) else count
            if not count:
                continue
            joined = th.cat((partial.windows, scenes[:count, actor]), dim=0)
            on_ground = np.concatenate((partial.grounded, grounded[:count, actor]))
            z = np.concatenate((partial.height, height[:count, actor]))
            up = np.concatenate((partial.up_z, up_z[:count, actor]))
            near = np.concatenate((partial.distance, distance[:count, actor]))
            touch_events = np.concatenate((partial.touches, touches[:count, actor]))
            goal_at_end = bool(goal_ends[count - 1, actor])
            ending = np.zeros(len(joined), dtype=bool)
            ending[-1] = goal_at_end
            completed = next((flight for flight in air_maneuvers(
                np.ones(len(joined), dtype=bool), on_ground, z, up, near,
                goal_ends=ending,
            ) if flight.action_start == partial.action_start), None)
            if (completed is not None and
                    (completed.goal_terminal or
                     completed.recovery_stop - completed.action_stop >= AERIAL_MIN_CONTEXT_STEPS)
                    and (completed.recovery_stop - completed.action_stop >= AERIAL_RECOVERY_STEPS
                         or completed.goal_terminal or completed.recovery_stop < len(joined)
                         or count < steps)):
                self._archive(joined, replace(
                    completed, skill_category=generated_air_skill(
                        joined, completed, touch_events,
                    ),
                ))
            elif (count == steps and not goal_at_end
                  and len(joined) < MANEUVER_MAX_TRACKED_STEPS):
                self.pending[actor] = PendingAirManeuver(
                    joined, partial.action_start, on_ground, z, up, near,
                    touch_events,
                )

        previous_ground = self.pending_ground
        self.pending_ground = {}
        for actor, partial in previous_ground.items():
            invalid = np.flatnonzero(~valid[:, actor])
            count = int(invalid[0]) if len(invalid) else steps
            goals = np.flatnonzero(goal_ends[:count, actor])
            count = min(count, int(goals[0]) + 1) if len(goals) else count
            if not count:
                continue
            joined = th.cat((partial.windows, scenes[:count, actor]), dim=0)
            features = np.concatenate((partial.features, control[:count, actor]))
            goal_at_end = bool(goal_ends[count - 1, actor])
            ending = np.zeros(len(joined), dtype=bool)
            ending[-1] = goal_at_end
            completed = next((maneuver for maneuver in ground_maneuvers(
                np.ones(len(joined), dtype=bool), features,
                goal_ends=ending,
            ) if maneuver.action_start == partial.action_start), None)
            if (completed is not None and
                    (completed.recovery_stop - completed.action_stop >= MANEUVER_RECOVERY_STEPS
                     or completed.goal_terminal or completed.recovery_stop < len(joined)
                     or count < steps)):
                self._archive(joined, completed)
            elif (count == steps and not goal_at_end
                  and len(joined) < GROUND_MAX_TRACKED_STEPS
                  and dribble_control_mask(features)[-MANEUVER_RECOVERY_STEPS:].any()):
                self.pending_ground[actor] = PendingGroundManeuver(
                    joined, partial.action_start, features,
                )

        if steps < 2:
            return
        takeoffs = grounded[:-1] & ~grounded[1:] & valid[:-1] & valid[1:]
        candidates = np.flatnonzero(valid[-1] & ~goal_ends[-1] & takeoffs.any(axis=0))
        unfinished = []
        for actor in candidates:
            if actor in self.pending:
                continue
            boundaries = np.flatnonzero(~valid[:, actor] | goal_ends[:, actor])
            run_start = int(boundaries[-1] + 1) if len(boundaries) else 0
            possible = np.flatnonzero(takeoffs[run_start:, actor])
            if not len(possible):
                continue
            takeoff = run_start + int(possible[-1]) + 1
            landings = np.flatnonzero(grounded[takeoff:, actor])
            if len(landings) and takeoff + int(landings[0]) + AERIAL_RECOVERY_STEPS <= steps:
                continue
            previous_air = np.flatnonzero(~grounded[run_start:takeoff, actor])
            previous_landing = (
                run_start + int(previous_air[-1]) + 1 if len(previous_air) else run_start
            )
            setup = max(previous_landing, takeoff - AERIAL_SETUP_STEPS)
            unfinished.append((int(actor), setup, takeoff))

        slots = max(0, MANEUVER_MAX_PENDING - len(self.pending))
        if len(unfinished) > slots:
            choice = th.randperm(len(unfinished), device=windows.device)[:slots].tolist()
            unfinished = [unfinished[index] for index in choice]
        for actor, setup, takeoff in unfinished:
            self.pending[actor] = PendingAirManeuver(
                scenes[setup:, actor].detach().clone(), takeoff - setup,
                grounded[setup:, actor].copy(), height[setup:, actor].copy(),
                up_z[setup:, actor].copy(), distance[setup:, actor].copy(),
                touches[setup:, actor].copy(),
            )

        carrying = dribble_control_mask(control) & valid
        raw = carrying.copy()
        carrying[1:-1] |= raw[:-2] & raw[2:] & valid[1:-1]
        possible = np.flatnonzero(
            valid[-1] & ~goal_ends[-1]
            & carrying[-MANEUVER_RECOVERY_STEPS:].any(axis=0)
        )
        unfinished_ground = []
        for actor in possible:
            if actor in self.pending_ground:
                continue
            boundaries = np.flatnonzero(~valid[:, actor] | goal_ends[:, actor])
            run_start = int(boundaries[-1] + 1) if len(boundaries) else 0
            if not carrying[run_start:, actor].any():
                continue
            local = carrying[run_start:, actor].astype(np.int8)
            edges = np.flatnonzero(np.diff(np.pad(local, (1, 1))))
            if not len(edges):
                continue
            starts, stops = edges[::2], edges[1::2]
            begin = run_start + int(starts[-1])
            stop = run_start + int(stops[-1])
            if stop < steps and stop + MANEUVER_RECOVERY_STEPS <= steps:
                continue
            previous_end = (
                run_start + int(stops[-2]) if len(stops) > 1 else run_start
            )
            setup = max(previous_end, begin - MANEUVER_SETUP_STEPS)
            if setup == begin:
                continue
            unfinished_ground.append((int(actor), setup, begin))

        slots = max(0, min(512, MANEUVER_MAX_PENDING) - len(self.pending_ground))
        if len(unfinished_ground) > slots:
            choice = th.randperm(len(unfinished_ground), device=windows.device)[:slots].tolist()
            unfinished_ground = [unfinished_ground[index] for index in choice]
        for actor, setup, begin in unfinished_ground:
            self.pending_ground[actor] = PendingGroundManeuver(
                scenes[setup:, actor].detach().clone(), begin - setup,
                control[setup:, actor].copy(),
            )


class SceneGAIFOMinibatches:
    """Match curated training windows; retain the old sampler for legacy datasets."""

    def __init__(
        self,
        expert: ExpertSceneDataset,
        batch_size: int,
        epochs: int,
        noise_std: float,
        history: HistoricalReplayBuffer | RecencyReplayBuffer | None = None,
        mix_fraction: float = 0.5,
        factorize: bool = False,
    ) -> None:
        if batch_size < 1 or epochs < 1:
            raise ValueError("batch size and epochs must be positive")
        if not 0.0 <= mix_fraction < 1.0:
            raise ValueError("mix fraction must be in [0, 1)")
        self.expert = expert
        self.batch_size = batch_size
        self.epochs = epochs
        self.noise_std = noise_std
        self.history = history
        self.mix_fraction = mix_fraction if (history is not None) else 0.0
        self.factorize = factorize
        self._epoch_callback = None

    def set_epoch_callback(self, callback) -> None:
        self._epoch_callback = callback

    def _sample_curated_windows(
        self, windows: th.Tensor, indices: th.Tensor, n_envs: int,
        episode_end: th.Tensor | None,
        archived_flights: tuple[list[CompletedSceneManeuver], ...] | None,
        ego_ball_touch: th.Tensor | None,
        goal_scored: th.Tensor | None,
        timeline: GeneratedSceneTimeline | None,
    ):
        """Pair curated experts by exact scene bin or complete-maneuver phase."""
        if not len(indices):
            raise ValueError("curated discriminator needs generated windows")
        expert_groups = self.expert.curated_labeled_pools()
        weights = self.expert.curated_weights()
        agent_flights = (
            generated_maneuver_pools(
                windows, indices, n_envs, episode_end,
                ego_ball_touch=ego_ball_touch, goal_scored=goal_scored,
                timeline=timeline,
            )
            if self.factorize else tuple([] for _ in range(N_SITUATIONS))
        )
        archived_windows = None
        if self.factorize and archived_flights is not None:
            if len(archived_flights) != N_SITUATIONS:
                raise ValueError("archived flights must have one pool per situation")
            saved = []
            next_start = len(windows)
            for label, group in enumerate(archived_flights):
                for complete in group:
                    flight = complete.span
                    if flight.situation != label:
                        raise ValueError("archived flight has the wrong situation")
                    agent_flights[label].append(replace(
                        flight,
                        setup_start=next_start + flight.setup_start,
                        action_start=next_start + flight.action_start,
                        action_stop=next_start + flight.action_stop,
                        recovery_stop=next_start + flight.recovery_stop,
                    ))
                    saved.append(complete.windows)
                    next_start += len(complete.windows)
            if saved:
                archived_windows = th.cat(saved)
        expert_flights = self.expert._curated_maneuvers[False]
        aligned_flights = [[] for _ in range(DRIVING_SKILL)]
        for label in range(6, N_SITUATIONS):
            if label < GROUND_MANEUVER_START:
                for category in (AERIAL_TOUCH_SKILL, AERIAL_MANEUVER_SKILL):
                    agents = [clip for clip in agent_flights[label]
                              if clip.skill_category == category]
                    experts = [clip for clip in expert_flights[label]
                               if clip.skill_category == category]
                    if agents and experts:
                        aligned_flights[category].append((agents, experts))
            else:
                if agent_flights[label] and expert_flights[label]:
                    aligned_flights[DRIBBLE_SKILL + label - GROUND_MANEUVER_START].append((
                        agent_flights[label], expert_flights[label],
                    ))
        for _ in range(self.epochs):
            shuffled = indices[th.randperm(len(indices), device=indices.device)]
            for start in range(0, len(shuffled), self.batch_size):
                selected = shuffled[start:start + self.batch_size]
                count = len(selected)
                n_history = 0
                if self.history is not None and self.history.size:
                    n_history = min(int(count * self.mix_fraction), self.history.size)
                agents = windows[selected]
                if n_history:
                    agents = th.cat((
                        agents[:count - n_history],
                        self.history.sample(n_history, agents.device),
                    ))
                agent_labels = scene_situation_ids(agents)
                agent_groups = tuple(
                    (agent_labels == label).nonzero(as_tuple=True)[0]
                    for label in range(GROUND_MANEUVER_START)
                )
                shared = tuple(
                    tuple(label for label, pool in enumerate(category)
                          if len(pool) and len(agent_groups[label]))
                    for category in expert_groups
                )
                available = weights.clone()
                available *= th.tensor(
                    [bool(labels) for labels in shared], device=weights.device,
                )
                if not bool(available.sum()):
                    # Do not turn unmatched generated scenes into easy negatives.
                    continue
                categories = th.multinomial(
                    available / available.sum(), count, replacement=True,
                )
                sampled_agents = th.empty_like(agents)
                sampled_experts = th.empty((count, 2), dtype=th.long, device=agents.device)
                exactly_matched = th.zeros(count, dtype=th.bool, device=agents.device)
                phase_aligned = th.zeros_like(exactly_matched)
                for category, labels in enumerate(shared):
                    positions = (categories == category).nonzero(as_tuple=True)[0]
                    if not len(positions):
                        continue
                    candidates = th.cat([
                        agent_groups[label] for label in labels
                    ])
                    selected_agents = candidates[th.randint(
                        len(candidates), (len(positions),), device=agents.device,
                    )]
                    sampled_agents[positions] = agents[selected_agents]
                    chosen_labels = agent_labels[selected_agents]
                    for label in chosen_labels.unique().tolist():
                        spots = positions[chosen_labels == label]
                        exactly_matched[spots] = True
                        pool = expert_groups[category][label]
                        sampled_experts[spots] = pool[th.randint(
                            len(pool), (len(spots),), device=agents.device,
                        )]

                # Keep full takeoff/carry -> action -> recovery alignment when
                # the policy has completed a maneuver. The remaining examples
                # are still matched by scene bin to the same curated pools.
                for category, flight_groups in enumerate(aligned_flights):
                    if not flight_groups:
                        continue
                    positions = (categories == category).nonzero(as_tuple=True)[0]
                    budget = len(positions) // 4
                    consumed = 0
                    while budget - consumed >= 4:
                        # These are Python list indices, not GPU tensor indices.
                        # Draw on the CPU rather than synchronizing CUDA for
                        # every clip choice during discriminator training.
                        agent_group, expert_group = flight_groups[th.randint(
                            len(flight_groups), (1,), device="cpu",
                        ).item()]
                        agent_clip = agent_group[th.randint(
                            len(agent_group), (1,), device="cpu",
                        ).item()]
                        expert_clip = expert_group[th.randint(
                            len(expert_group), (1,), device="cpu",
                        ).item()]
                        agent_ids, expert_pairs = aligned_maneuver_windows(
                            agent_clip, expert_clip, budget - consumed,
                            agents.device,
                        )
                        if not len(agent_ids):
                            break
                        chosen = positions[consumed:consumed + len(agent_ids)]
                        current = agent_ids < len(windows)
                        sampled_agents[chosen[current]] = windows[agent_ids[current]]
                        if (~current).any():
                            sampled_agents[chosen[~current]] = archived_windows[
                                agent_ids[~current] - len(windows)
                            ]
                        sampled_experts[chosen] = expert_pairs
                        expert_phases = self.expert._windows_for_povs(expert_pairs)
                        exactly_matched[chosen] = (
                            scene_situation_ids(sampled_agents[chosen])
                            == scene_situation_ids(expert_phases)
                        )
                        phase_aligned[chosen] = True
                        consumed += len(agent_ids)

                agent_windows = add_scene_noise(sampled_agents, self.noise_std)
                expert_windows = add_scene_noise(
                    self.expert._windows_for_povs(sampled_experts), self.noise_std,
                )
                yield TensorBatch({
                    "window": th.cat((agent_windows, expert_windows)),
                    "is_agent": th.cat((
                        th.ones(count, device=agents.device),
                        th.zeros(count, device=agents.device),
                    )),
                    "skill_category": th.cat((categories, categories)),
                    "situation_matched": th.cat((exactly_matched, exactly_matched)),
                    "phase_aligned": th.cat((phase_aligned, phase_aligned)),
                    "grounded_random": th.cat((categories == DRIVING_SKILL,
                                                categories == DRIVING_SKILL)),
                })
            if self._epoch_callback is not None:
                self._epoch_callback()

    def sample_windows(
        self,
        windows: th.Tensor,
        indices: th.Tensor,
        *,
        n_envs: int = 1,
        episode_end: th.Tensor | None = None,
        archived_flights: tuple[list[CompletedSceneManeuver], ...] | None = None,
        ego_ball_touch: th.Tensor | None = None,
        goal_scored: th.Tensor | None = None,
        timeline: GeneratedSceneTimeline | None = None,
    ):
        if self.expert.skill_sampling:
            yield from self._sample_curated_windows(
                windows, indices, n_envs, episode_end, archived_flights,
                ego_ball_touch, goal_scored, timeline,
            )
            return
        agent_groups: list[th.Tensor] = [indices[:0] for _ in range(N_SITUATIONS)]
        grounded_agent = indices[:0]
        grounded_expert_available = False
        expert_groups: tuple[th.Tensor, ...] = ()
        agent_flights: tuple[list[SceneManeuver], ...] = tuple([] for _ in range(N_SITUATIONS))
        expert_flights: tuple[list[SceneManeuver], ...] = tuple([] for _ in range(N_SITUATIONS))
        archived_windows = None
        if self.factorize and len(indices) >= 1 / SITUATION_MATCH_FRACTION:
            grouped: list[list[th.Tensor]] = [[] for _ in range(N_SITUATIONS)]
            for chunk in indices.split(8_192):
                situations = scene_situation_ids(windows[chunk])
                on_surface = windows[chunk, -1, BLUE_START + CAR_BOOL_START] > 0.5
                for label in situations.unique().tolist():
                    if label // len(DISTANCE_BANDS) < 2:
                        grouped[label].append(chunk[(situations == label) & on_surface])
            agent_groups = [
                th.cat(group) if group else indices[:0] for group in grouped
            ]
            expert_groups = self.expert.situation_pools()
            grounded_agent = indices[
                (windows[indices, -1, BLUE_START + CAR_BOOL_START] > 0.5)
                & (windows[indices, -1, BLUE_START + 14] > 0.65)
            ]
            grounded_expert_available = len(self.expert._grounded_choices()[0]) > 0
            agent_flights = generated_maneuver_pools(
                windows, indices, n_envs, episode_end,
                goal_scored=goal_scored, timeline=timeline,
            )
            if archived_flights is not None:
                if len(archived_flights) != N_SITUATIONS:
                    raise ValueError("archived flights must have one pool per situation")
                saved = []
                next_start = len(windows)
                for label, group in enumerate(archived_flights):
                    for complete in group:
                        flight = complete.span
                        if flight.situation != label:
                            raise ValueError("archived flight has the wrong situation")
                        agent_flights[label].append(replace(
                            flight,
                            setup_start=next_start + flight.setup_start,
                            action_start=next_start + flight.action_start,
                            action_stop=next_start + flight.action_stop,
                            recovery_stop=next_start + flight.recovery_stop,
                        ))
                        saved.append(complete.windows)
                        next_start += len(complete.windows)
                if saved:
                    archived_windows = th.cat(saved)
            if any(agent_flights):
                expert_flights = self.expert.maneuver_pools()
        surface_labels = 2 * len(DISTANCE_BANDS)
        eligible = []
        for label in range(N_SITUATIONS):
            if label < surface_labels and expert_groups:
                if len(agent_groups[label]) and len(expert_groups[label]):
                    eligible.append(label)
            elif (label >= surface_labels and len(agent_flights[label])
                  and len(expert_flights[label])):
                eligible.append(label)
        for _ in range(self.epochs):
            order = indices[th.randperm(len(indices), device=indices.device)]
            agent_queues = [
                group[th.randperm(len(group), device=indices.device)]
                for group in agent_groups
            ]
            expert_queues = [
                group[th.randperm(len(group), device=indices.device)]
                for group in expert_groups
            ]
            agent_flight_queues = [
                [group[index] for index in th.randperm(len(group), device=indices.device).tolist()]
                for group in agent_flights
            ]
            expert_flight_queues = [
                [group[index] for index in th.randperm(len(group), device=indices.device).tolist()]
                for group in expert_flights
            ]
            capacities = {
                label: min(
                    len(agent_queues[label]) if label < surface_labels
                    else len(agent_flight_queues[label]),
                    len(expert_queues[label]) if label < surface_labels
                    else len(expert_flight_queues[label]),
                ) for label in eligible
            }
            used = [0] * N_SITUATIONS
            next_label = 0
            for start in range(0, len(order), self.batch_size):
                selected = order[start : start + self.batch_size]
                sample_count = len(selected)

                current_windows = windows[selected]
                n_history = 0
                if self.history is not None and self.history.size > 0:
                    n_history = min(
                        int(sample_count * self.mix_fraction),
                        self.history.size,
                    )
                n_current = sample_count - n_history

                agent_windows = current_windows[:n_current]
                agent_parts = []
                expert_parts = []
                matched = 0
                quota = min(n_current, int(sample_count * SITUATION_MATCH_FRACTION))
                grounded_count = min(
                    int(sample_count * GROUND_RANDOM_FRACTION), quota,
                ) if len(grounded_agent) and grounded_expert_available else 0
                situation_quota = quota - grounded_count
                exhausted = 0
                while matched < situation_quota and eligible and exhausted < len(eligible):
                    label = eligible[next_label]
                    next_label = (next_label + 1) % len(eligible)
                    if used[label] >= capacities[label]:
                        exhausted += 1
                        continue
                    if label < surface_labels:
                        agent_parts.append(agent_queues[label][used[label]:used[label] + 1])
                        expert_parts.append(expert_queues[label][used[label]:used[label] + 1])
                        matched += 1
                    else:
                        agent_ids, expert_pairs = aligned_maneuver_windows(
                            agent_flight_queues[label][used[label]],
                            expert_flight_queues[label][used[label]],
                            situation_quota - matched, indices.device,
                        )
                        if not len(agent_ids):
                            exhausted += 1
                            continue
                        agent_parts.append(agent_ids)
                        expert_parts.append(expert_pairs)
                        matched += len(agent_ids)
                    used[label] += 1
                    exhausted = 0
                if grounded_count:
                    grounded_indices = th.randint(
                        len(grounded_agent), (grounded_count,), device=indices.device,
                    )
                    agent_parts.append(grounded_agent[grounded_indices])
                    expert_parts.append(self.expert.random_grounded_povs(grounded_count))
                selected_count = matched + grounded_count
                if selected_count:
                    agent_indices = th.cat(agent_parts)
                    expert_pairs = th.cat(expert_parts)
                    agent_windows = agent_windows.clone()
                    if archived_windows is None:
                        agent_windows[:selected_count] = windows[agent_indices]
                    else:
                        current = agent_indices < len(windows)
                        matched_windows = agent_windows[:selected_count]
                        matched_windows[current] = windows[agent_indices[current]]
                        matched_windows[~current] = archived_windows[
                            agent_indices[~current] - len(windows)
                        ]
                if n_history > 0:
                    historical = self.history.sample(
                        n_history,
                        current_windows.device,
                    )
                    agent_windows = th.cat([agent_windows, historical], dim=0)

                agent_windows = add_scene_noise(agent_windows, self.noise_std)
                expert_windows = self.expert.sample(sample_count, agent_windows.device)
                if selected_count:
                    expert_windows[:selected_count] = self.expert._windows_for_povs(expert_pairs)
                expert_windows = add_scene_noise(expert_windows, self.noise_std)
                is_agent = th.cat(
                    [
                        th.ones(sample_count, device=agent_windows.device),
                        th.zeros(sample_count, device=agent_windows.device),
                    ]
                )
                matched_mask = th.arange(sample_count, device=agent_windows.device) < matched
                grounded_mask = ((th.arange(sample_count, device=agent_windows.device) >= matched)
                                 & (th.arange(sample_count, device=agent_windows.device)
                                    < selected_count))

                yield TensorBatch(
                    {
                        "window": th.cat([agent_windows, expert_windows]),
                        "is_agent": is_agent,
                        "situation_matched": th.cat([matched_mask, matched_mask]),
                        "grounded_random": th.cat([grounded_mask, grounded_mask]),
                    }
                )

            if self._epoch_callback is not None:
                self._epoch_callback()


def train_discriminator_minibatch(
    sample: TensorBatch,
    discriminator: SceneDiscriminator | FactorizedSceneDiscriminator,
    optimizer: th.optim.Optimizer,
    loss: SceneDiscriminatorLoss,
    microbatch_size: int,
    max_grad_norm: float,
) -> dict[str, th.Tensor]:
    """Accumulate one balanced effective batch without a full-batch GRU graph."""
    if microbatch_size < 1:
        raise ValueError("discriminator microbatch size must be positive")
    windows = sample["window"]
    labels = sample["is_agent"]
    n_agent = len(windows) // 2
    if n_agent < 1 or len(windows) != 2 * n_agent or len(labels) != len(windows):
        raise ValueError("discriminator batch must have equal agent and expert halves")

    optimizer.zero_grad(set_to_none=True)
    metrics: dict[str, th.Tensor] = {}
    if "situation_matched" in sample:
        metrics["matched_situation_fraction"] = (
            sample["situation_matched"][:n_agent].float().mean()
        )
    if "grounded_random" in sample:
        metrics["grounded_random_fraction"] = (
            sample["grounded_random"][:n_agent].float().mean()
        )
    if "phase_aligned" in sample:
        metrics["phase_aligned_fraction"] = (
            sample["phase_aligned"][:n_agent].float().mean()
        )
    if "skill_category" in sample:
        for category, name in enumerate(SKILL_CATEGORIES):
            metrics[f"{name}_fraction"] = (
                (sample["skill_category"][:n_agent] == category).float().mean()
            )
    ball_weights = None
    if getattr(discriminator, "factorized", False):
        ball_weights = ball_responsibility(windows).square()
    if ball_weights is not None:
        for chosen in (slice(None, n_agent), slice(n_agent, None)):
            ball_weights[chosen] /= ball_weights[chosen].mean().clamp_min(1e-6)
    for start in range(0, n_agent, microbatch_size):
        stop = min(start + microbatch_size, n_agent)
        chunk = TensorBatch({
            "window": th.cat((windows[start:stop], windows[n_agent + start:n_agent + stop])),
            "is_agent": th.cat((labels[start:stop], labels[n_agent + start:n_agent + stop])),
        })
        if ball_weights is not None:
            chunk = chunk.with_fields(ball_weight=th.cat((
                ball_weights[start:stop], ball_weights[n_agent + start:n_agent + stop],
            )))
        output = loss(chunk)
        fraction = (stop - start) / n_agent
        (output.loss * fraction).backward()
        for name, value in output.metrics.items():
            metrics[name] = metrics.get(name, 0.0) + value.detach() * fraction
        del output, chunk

    th.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
    optimizer.step()
    return metrics


class SceneDiscriminatorReward:
    """Combine imitation, goal, and physical touch rewards per actor.

    Short windows receive normalized expert log-odds by default, or capped
    expert-to-agent odds with the optional exponential reward. Physical bonuses
    are zero-sum in 1v1; goal and touch transitions remain learnable before
    imitation windows are valid.
    """

    def __init__(
        self,
        discriminator: SceneDiscriminator | FactorizedSceneDiscriminator,
        noise_std: float,
        trajectory_length: int,
        goal_reward_weight: float = 1.0,
        aerial_touch_reward_weight: float = 0.0,
        flip_reset_reward_weight: float = 0.0,
        batch_size: int = 16_384,
        max_magnitude: float = 10.0,
        exp_log_odds_reward: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("discriminator reward batch size must be positive")
        if not math.isfinite(max_magnitude) or max_magnitude <= 0.0:
            raise ValueError("reward max magnitude must be positive")
        if not math.isfinite(goal_reward_weight) or goal_reward_weight < 0.0:
            raise ValueError("goal reward weight must be non-negative")
        if not math.isfinite(aerial_touch_reward_weight) or aerial_touch_reward_weight < 0.0:
            raise ValueError("aerial touch reward weight must be non-negative")
        if not math.isfinite(flip_reset_reward_weight) or flip_reset_reward_weight < 0.0:
            raise ValueError("flip reset reward weight must be non-negative")
        self.discriminator = discriminator
        self.factorize = getattr(discriminator, "factorized", False)
        self.noise_std = noise_std
        self.trajectory_length = trajectory_length
        self.goal_reward_weight = goal_reward_weight
        self.aerial_touch_reward_weight = aerial_touch_reward_weight
        self.flip_reset_reward_weight = flip_reset_reward_weight
        self.batch_size = batch_size
        self.max_magnitude = max_magnitude
        self.exp_log_odds_reward = exp_log_odds_reward

    @staticmethod
    def _event_reward(
        batch: TensorBatch, name: str, weight: float, reference: th.Tensor
    ) -> th.Tensor:
        score = batch.get(name)
        if score is None:
            if weight:
                raise ValueError(f"missing {name} with nonzero reward weight")
            return th.zeros_like(reference)
        if score.shape != reference.shape:
            raise ValueError(f"{name} must match the actor rollout shape")
        return score.to(reference.dtype) * weight

    def _score_windows(
        self,
        windows: th.Tensor,
        valid: th.Tensor,
    ) -> th.Tensor:
        scores = th.zeros(
            (*valid.shape, 2) if self.factorize else valid.shape,
            dtype=windows.dtype, device=windows.device,
        )
        if not valid.any():
            return scores

        flat_windows = windows.reshape(-1, self.trajectory_length, SCENE_SIZE)
        flat_scores = scores.reshape(-1, 2) if self.factorize else scores.flatten()
        indices = th.nonzero(valid.flatten(), as_tuple=False).squeeze(-1)
        selected_scores = th.empty(
            (len(indices), 2) if self.factorize else (len(indices),),
            dtype=scores.dtype, device=scores.device,
        )
        for start in range(0, len(indices), self.batch_size):
            stop = min(start + self.batch_size, len(indices))
            noisy = add_scene_noise(
                flat_windows[indices[start:stop]], self.noise_std
            )
            logits = self.discriminator(noisy)
            if self.exp_log_odds_reward:
                # D = sigmoid(-logits) is the expert probability, so
                # exp(log D - log(1-D)) = exp(-logits).
                selected_scores[start:stop] = (-logits).clamp(
                    max=math.log(self.max_magnitude)
                ).exp()
            else:
                selected_scores[start:stop] = (-logits).clamp(
                    -self.max_magnitude, self.max_magnitude
                )

        if self.exp_log_odds_reward:
            flat_scores[indices] = selected_scores
            return scores

        if self.factorize:
            std = selected_scores.std(dim=0, unbiased=False)
            normalized = th.where(
                std > 1e-8,
                (selected_scores - selected_scores.mean(dim=0)) / std.clamp_min(1e-8),
                th.zeros_like(selected_scores),
            )
        else:
            std = selected_scores.std(unbiased=False)
            if std > 1e-8:
                normalized = (selected_scores - selected_scores.mean()) / std
            else:
                normalized = th.zeros_like(selected_scores)
        flat_scores[indices] = normalized.clamp(
            -self.max_magnitude, self.max_magnitude
        )
        return scores

    @th.no_grad()
    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        windows = batch["scene_window"]
        valid = batch["scene_window_valid"].bool()
        if windows.shape[:2] != valid.shape or windows.shape[-2:] != (
            self.trajectory_length, SCENE_SIZE
        ):
            raise ValueError("scene windows have the wrong shape")

        dtype = batch["observation"].dtype
        scores = self._score_windows(windows, valid).to(dtype)
        components = {}
        if self.factorize:
            proximity = ball_responsibility(windows).to(dtype)
            car_reward = 0.5 * scores[..., 0]
            ball_reward = 0.5 * proximity * scores[..., 1]
            imitation_reward = car_reward + ball_reward
            components = {
                "car_imitation_reward": car_reward,
                "ball_imitation_reward": ball_reward,
                "ball_proximity": proximity,
            }
        else:
            imitation_reward = scores
        goal_reward = batch["reward"].to(dtype) * self.goal_reward_weight
        if goal_reward.shape != imitation_reward.shape:
            raise ValueError("goal rewards must match the actor rollout shape")
        aerial_touch_reward = self._event_reward(
            batch, "aerial_touch_score", self.aerial_touch_reward_weight,
            imitation_reward,
        )
        flip_reset_reward = self._event_reward(
            batch, "flip_reset_event", self.flip_reset_reward_weight,
            imitation_reward,
        )

        result = batch.with_fields(
            imitation_reward=imitation_reward,
            **components,
            goal_reward=goal_reward,
            aerial_touch_reward=aerial_touch_reward,
            flip_reset_reward=flip_reset_reward,
            training_reward=(
                imitation_reward + goal_reward + aerial_touch_reward + flip_reset_reward
            ),
        )
        learner_mask = (
            valid | goal_reward.ne(0) | aerial_touch_reward.ne(0)
            | flip_reset_reward.ne(0)
        )
        if "learner_mask" in result:
            return result.replace_fields(
                learner_mask=result["learner_mask"].bool() & learner_mask
            )
        return result.with_fields(learner_mask=learner_mask)


class SelectPPOFields:
    """Drop large discriminator-only tensors before PPO sampling."""

    FIELDS = (
        "observation",
        "action",
        "old_log_prob",
        "baseline_value",
        "advantage",
        "returns",
        "learner_mask",
    )
    RECURRENT_FIELDS = FIELDS + (
        "policy_state",
        "critic_state",
        "terminated",
        "truncated",
    )

    def __init__(self, recurrent: bool = False) -> None:
        self.fields = self.RECURRENT_FIELDS if recurrent else self.FIELDS

    def __call__(
        self, batch: TensorBatch, context: PrepareContext
    ) -> TensorBatch:
        return batch.select(*self.fields)


class AdaptiveDiscriminatorUpdate:
    """Train the short-window discriminator to a held-out accuracy target."""

    def __init__(
        self,
        expert: ExpertSceneDataset,
        history: HistoricalReplayBuffer | RecencyReplayBuffer | None,
        batch_size: int,
        epochs: int,
        noise_std: float,
        heldout_size: int,
        accuracy_target: float,
        history_add_size: int,
        history_mix_fraction: float,
        max_grad_norm: float,
        discriminator: SceneDiscriminator | FactorizedSceneDiscriminator,
        optimizer: th.optim.Optimizer,
        loss: SceneDiscriminatorLoss,
        section: str = "Discriminator",
        update_interval: int = 1,
        microbatch_size: int = 1_024,
        reset_miner: ConfidentExpertResetTransform | None = None,
    ) -> None:
        if heldout_size < 0:
            raise ValueError("heldout size must be non-negative")
        if not 0.0 <= accuracy_target <= 1.0:
            raise ValueError("accuracy target must be between zero and one")
        if not 0.0 <= history_mix_fraction < 1.0:
            raise ValueError("history mix fraction must be in [0, 1)")
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
            raise ValueError("max gradient norm must be positive")
        if update_interval < 1:
            raise ValueError("update interval must be positive")
        if microbatch_size < 1:
            raise ValueError("microbatch size must be positive")
        self.expert = expert
        self.history = history
        self.batch_size = batch_size
        self.epochs = epochs
        self.noise_std = noise_std
        self.heldout_size = heldout_size
        self.accuracy_target = accuracy_target
        self.history_add_size = history_add_size
        self.history_mix_fraction = history_mix_fraction
        self.max_grad_norm = max_grad_norm
        self.discriminator = discriminator
        self.optimizer = optimizer
        self.loss = loss
        self.section = section
        self.update_interval = update_interval
        self.microbatch_size = microbatch_size
        self.reset_miner = reset_miner
        self._progress_callback = None
        self._heldout_sim: th.Tensor | None = None
        self._has_updated = False
        self._rollouts_since_update = 0

        self.maneuver_tracker = (
            GeneratedManeuverTracker() if getattr(discriminator, "factorized", False) else None
        )

    def set_progress_callback(self, callback) -> None:
        self._progress_callback = callback

    def run(self, experience: Rollout | TensorBatch):
        batch = (
            experience.steps
            if isinstance(experience, Rollout)
            else experience
        )
        windows = batch["scene_window"]
        valid = batch["scene_window_valid"].bool()
        if not valid.any():
            raise RuntimeError("no valid generated scene windows in rollout")

        flat_windows = windows.reshape(
            -1, self.expert.trajectory_length, SCENE_SIZE
        )
        train_indices, heldout_indices = self._split_generated(valid)
        terminal = batch.get("terminated")
        truncated = batch.get("truncated")
        if truncated is not None:
            terminal = truncated.bool() if terminal is None else terminal.bool() | truncated.bool()
        goal_scored = (
            batch["reward"].gt(0) & terminal.bool()
            if terminal is not None and "reward" in batch else None
        )
        touch_events = batch.get("ego_ball_touch")
        timeline = None
        if self.maneuver_tracker is not None:
            timeline = generated_scene_timeline(
                flat_windows, train_indices, valid.shape[1], terminal,
                touch_events, goal_scored,
            )
            self.maneuver_tracker.feed(
                flat_windows, train_indices, valid.shape[1], terminal,
                timeline=timeline,
            )
        heldout_generated = flat_windows[heldout_indices]
        heldout_near = None
        if (getattr(self.discriminator, "factorized", False)
                and len(heldout_generated) and self.expert.heldout_near_total):
            heldout_near = heldout_generated[
                nearest_ball_distance(heldout_generated) <= BALL_NEAR_DISTANCE
            ]

        # Use the same held-out examples to check every minibatch in this update.
        # Resampling and gathering the full validation set on every check was
        # nearly as costly as training the discriminator itself.
        validation = self._heldout_pairs(heldout_generated, heldout_near)
        evaluation = self._evaluate(heldout_generated, heldout_near, validation=validation)
        if self._has_updated:
            self._rollouts_since_update += 1
        scheduled = (
            not self._has_updated
            or self._rollouts_since_update >= self.update_interval
        )
        metrics: dict[str, float] = evaluation | {
            "updated": 0.0,
            "minibatches": 0.0,
            "scheduled": float(scheduled),
        }

        if scheduled and len(train_indices) > 0:
            sampler = SceneGAIFOMinibatches(
                self.expert,
                self.batch_size,
                self.epochs,
                self.noise_std,
                history=self.history,
                mix_fraction=self.history_mix_fraction,
                factorize=getattr(self.discriminator, "factorized", False),
            )
            sampler.set_epoch_callback(self._epoch_finished)
            metric_totals: dict[str, float | th.Tensor] = {}
            minibatch_count = 0
            callback = self._progress_callback
            if callback is not None:
                callback.start(self.epochs, self.section)
            try:
                for sample in sampler.sample_windows(
                    flat_windows, train_indices, n_envs=valid.shape[1], episode_end=terminal,
                    archived_flights=(self.maneuver_tracker.ready
                                      if self.maneuver_tracker is not None else None),
                    ego_ball_touch=touch_events, goal_scored=goal_scored,
                    timeline=timeline,
                ):
                    minibatch_metrics = train_discriminator_minibatch(
                        sample, self.discriminator, self.optimizer, self.loss,
                        self.microbatch_size, self.max_grad_norm,
                    )

                    for key, value in minibatch_metrics.items():
                        metric_totals[key] = metric_totals.get(key, 0.0) + value
                    minibatch_count += 1

                    evaluation = self._evaluate(
                        heldout_generated, heldout_near, validation=validation,
                    )
                    if evaluation["heldout_accuracy"] >= self.accuracy_target:
                        metrics["updated"] = 1.0
                        break
                else:
                    if minibatch_count > 0:
                        metrics["updated"] = 1.0
            finally:
                if callback is not None:
                    callback.finish()

            if minibatch_count > 0:
                if self.maneuver_tracker is not None:
                    self.maneuver_tracker.clear_ready()
                self._has_updated = True
                if self.reset_miner is not None:
                    self.reset_miner.ready = True
                self._rollouts_since_update = 0
                metrics["minibatches"] = float(minibatch_count)
                for key, total in metric_totals.items():
                    averaged = total / minibatch_count
                    metrics[f"train_{key}"] = (
                        float(averaged.item())
                        if isinstance(averaged, th.Tensor)
                        else float(averaged)
                    )

        metrics.update(evaluation)
        if self.reset_miner is not None:
            metrics["reset_mined_fraction"] = self.reset_miner.take_mined_fraction()

        if self.history is not None:
            add_count = min(self.history_add_size, len(train_indices))
            if add_count:
                selected = train_indices[
                    th.randperm(len(train_indices), device=train_indices.device)[:add_count]
                ]
                self.history.add(flat_windows[selected], add_count)

        return experience, {self.section: metrics}

    def _split_generated(
        self, valid: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        if valid.ndim != 2 or valid.shape[1] % N_CARS:
            raise ValueError("generated validity must be [time, 1v1 actors]")
        T, n_envs = valid.shape
        n_sim = n_envs // N_CARS
        flat_valid = valid.flatten()
        valid_indices = th.nonzero(flat_valid, as_tuple=False).squeeze(-1)
        if self.heldout_size == 0 or n_sim < 2:
            return valid_indices, valid_indices[:0]

        if self._heldout_sim is not None:
            heldout_mask = (
                self._heldout_sim.view(1, n_sim, 1)
                .expand(T, n_sim, N_CARS)
                .reshape(T, n_envs)
                & valid
            ).flatten()
            return (
                th.nonzero(flat_valid & ~heldout_mask, as_tuple=False).squeeze(-1),
                th.nonzero(heldout_mask, as_tuple=False).squeeze(-1),
            )

        per_sim = valid.view(T, n_sim, N_CARS).sum(dim=(0, 2))
        candidates = th.nonzero(per_sim > 0, as_tuple=False).squeeze(-1)
        if len(candidates) < 2:
            return valid_indices, valid_indices[:0]
        candidates = candidates[
            th.randperm(len(candidates), device=candidates.device)
        ]
        cumulative = per_sim[candidates].cumsum(0)
        count = int(
            th.searchsorted(
                cumulative,
                th.tensor(self.heldout_size, device=cumulative.device),
            ).item()
        ) + 1
        count = min(count, len(candidates) - 1)
        heldout_sim = th.zeros(n_sim, dtype=th.bool, device=valid.device)
        heldout_sim[candidates[:count]] = True
        self._heldout_sim = heldout_sim
        heldout_mask = (
            heldout_sim.view(1, n_sim, 1)
            .expand(T, n_sim, N_CARS)
            .reshape(T, n_envs)
            & valid
        ).flatten()
        return (
            th.nonzero(flat_valid & ~heldout_mask, as_tuple=False).squeeze(-1),
            th.nonzero(heldout_mask, as_tuple=False).squeeze(-1),
        )

    def _heldout_pairs(
        self, heldout_generated: th.Tensor, heldout_near: th.Tensor | None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None, th.Tensor | None] | None:
        n = min(len(heldout_generated), self.expert.heldout_total, self.heldout_size)
        if not n:
            return None
        device = heldout_generated.device
        generated = heldout_generated[th.randperm(len(heldout_generated), device=device)[:n]]
        expert = self.expert.sample_heldout(n, device)
        near_generated = near_expert = None
        if (getattr(self.discriminator, "factorized", False)
                and heldout_near is not None and len(heldout_near)):
            n_near = min(len(heldout_near), self.heldout_size)
            near_generated = heldout_near[
                th.randperm(len(heldout_near), device=device)[:n_near]
            ]
            near_expert = self.expert.sample_near(n_near, device, heldout=True)
        return generated, expert, near_generated, near_expert

    def _evaluate(
        self, heldout_generated: th.Tensor, heldout_near: th.Tensor | None = None,
        *, validation: tuple[th.Tensor, th.Tensor, th.Tensor | None, th.Tensor | None]
        | None = None,
    ) -> dict[str, float]:
        if validation is None:
            validation = self._heldout_pairs(heldout_generated, heldout_near)
        factorize = getattr(self.discriminator, "factorized", False)
        head_names = ("car", "ball") if factorize else ("unified",)
        if validation is None:
            metrics = {
                "loss": 0.0,
                "agent_score": 0.0,
                "expert_score": 0.0,
                "agent_accuracy": 0.0,
                "expert_accuracy": 0.0,
                "heldout_accuracy": 0.0,
            }
            if factorize:
                metrics.update({f"{name}_heldout_accuracy": 0.0 for name in head_names})
            return metrics

        generated, expert, near_generated, near_expert = validation
        n = len(generated)
        totals = th.zeros(5, len(head_names), device=generated.device)

        was_training = self.discriminator.training
        with th.inference_mode():
            self.discriminator.eval()
            try:
                for start in range(0, n, self.microbatch_size):
                    stop = min(start + self.microbatch_size, n)
                    logits = self.discriminator(add_scene_noise(
                        th.cat((generated[start:stop], expert[start:stop])),
                        self.noise_std,
                    ))
                    generated_logits, expert_logits = logits.split(stop - start)
                    if not factorize:
                        generated_logits = generated_logits[:, None]
                        expert_logits = expert_logits[:, None]
                    totals[0] += F.softplus(-generated_logits).sum(dim=0)
                    totals[0] += F.softplus(expert_logits).sum(dim=0)
                    totals[1] += th.sigmoid(generated_logits).sum(dim=0)
                    totals[2] += th.sigmoid(expert_logits).sum(dim=0)
                    totals[3] += (generated_logits > 0.0).sum(dim=0)
                    totals[4] += (expert_logits <= 0.0).sum(dim=0)
                near_accuracy = None
                if factorize and near_generated is not None:
                    n_near = len(near_generated)
                    near_correct = th.zeros(2, device=generated.device)
                    for start in range(0, n_near, self.microbatch_size):
                        stop = min(start + self.microbatch_size, n_near)
                        logits = self.discriminator(add_scene_noise(
                            th.cat((near_generated[start:stop], near_expert[start:stop])),
                            self.noise_std,
                        ))[:, 1]
                        generated_logit, expert_logit = logits.split(stop - start)
                        near_correct[0] += (generated_logit > 0).sum()
                        near_correct[1] += (expert_logit <= 0).sum()
                    near_accuracy = near_correct.sum().item() / (2 * n_near)
            finally:
                self.discriminator.train(was_training)
        loss, agent_score, expert_score, agent_correct, expert_correct = totals.tolist()
        head_accuracies = [
            (agent_correct[index] + expert_correct[index]) / (2 * n)
            for index in range(len(head_names))
        ]
        metrics = {
            "loss": sum(loss) / (2 * n * len(head_names)),
            "agent_score": sum(agent_score) / (n * len(head_names)),
            "expert_score": sum(expert_score) / (n * len(head_names)),
            "agent_accuracy": sum(agent_correct) / (n * len(head_names)),
            "expert_accuracy": sum(expert_correct) / (n * len(head_names)),
            "heldout_accuracy": min(head_accuracies),
        }
        if factorize:
            metrics.update({
                f"{name}_heldout_accuracy": head_accuracies[index]
                for index, name in enumerate(head_names)
            })
            if near_accuracy is not None:
                metrics["ball_near_heldout_accuracy"] = near_accuracy
                metrics["heldout_accuracy"] = min(metrics["heldout_accuracy"], near_accuracy)
        return metrics

    def _epoch_finished(self) -> None:
        if self._progress_callback is not None:
            self._progress_callback.epoch_finished()


class GAIFOCheckpoints:
    """Checkpoint the policy, discriminator and optional ASE skill encoder."""

    def __init__(
        self,
        directory: Path,
        interval: int,
        keep: int,
        policy: nn.Module,
        critic: nn.Module,
        discriminator: nn.Module,
        policy_optimizer: th.optim.Optimizer,
        critic_optimizer: th.optim.Optimizer,
        discriminator_optimizer: th.optim.Optimizer,
        buffer: RolloutBuffer,
        args: argparse.Namespace,
        *,
        skill_encoder: SkillEncoder | None = None,
        skill_optimizer: th.optim.Optimizer | None = None,
        skill_env: SkillConditionedEnv | None = None,
        skill_update: SkillEncoderUpdate | None = None,
    ) -> None:
        if getattr(args, "ase_diversity", False) and not all((
            skill_encoder, skill_optimizer, skill_env, skill_update,
        )):
            raise ValueError(
                "ASE checkpoint requires an encoder, optimizer, skill environment and update"
            )
        self.directory = Path(directory)
        self.interval = interval
        self.keep = keep
        self.policy = policy
        self.critic = critic
        self.discriminator = discriminator
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.buffer = buffer
        self.args = args
        self.skill_encoder = skill_encoder
        self.skill_optimizer = skill_optimizer
        self.skill_env = skill_env
        self.skill_update = skill_update
        self.step = 0
        self.next_step = 0
        self.clock: Clock | None = None
        self.directory.mkdir(parents=True, exist_ok=True)
        for path in self.directory.glob("gaifo_*.pt.tmp"):
            path.unlink()

    def ready(self, step: int) -> bool:
        self.step = step
        return self.buffer.position == 0 and step >= self.next_step

    def run(self) -> None:
        self.save(self.step)

    def save(self, step: int, force: bool = False) -> None:
        if not force and step < self.next_step:
            return
        payload = {
            "step": step,
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
            "config": {
                "architecture": (
                    GAIFO_ASE_ARCHITECTURE if getattr(self.args, "ase_diversity", False) else (
                        GAIFO_GRU_ARCHITECTURE if self.args.gru else GAIFO_ARCHITECTURE
                    )
                ),
                **{
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(self.args).items()
                },
            },
        }
        if self.skill_encoder is not None:
            payload.update({
                "skill_encoder": self.skill_encoder.state_dict(),
                "skill_encoder_optimizer": self.skill_optimizer.state_dict(),
                "skill_rng_state": self.skill_env.generator.get_state(),
                "skill_encoder_rng_state": self.skill_update.generator.get_state(),
            })
        if self.clock is not None:
            payload["clock"] = asdict(self.clock)
            payload["torch_rng_state"] = th.get_rng_state()
            payload["cuda_rng_state"] = th.cuda.get_rng_state_all()
        path = self.directory / f"gaifo_{step:012d}.pt"
        temporary = path.with_suffix(".pt.tmp")
        th.save(payload, temporary)
        temporary.replace(path)

        paths = sorted(self.directory.glob("gaifo_*.pt"))
        for old in paths[:-self.keep]:
            old.unlink()

        self.next_step = step + self.interval


def finite_checkpoint_state(value) -> bool:
    if isinstance(value, th.Tensor):
        return not value.is_floating_point() or bool(th.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_checkpoint_state(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_checkpoint_state(item) for item in value)
    return True


def load_resume_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"GAIFO checkpoint not found: {path}")
    payload = th.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"invalid GAIFO checkpoint: {path}")
    config = payload["config"]
    architecture = config.get("architecture")
    if architecture not in (
        GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE, GAIFO_ASE_ARCHITECTURE,
    ):
        raise ValueError(f"incompatible GAIFO architecture in {path}")
    if config.get("gru", False) != (architecture == GAIFO_GRU_ARCHITECTURE):
        raise ValueError(f"checkpoint GRU setting does not match architecture in {path}")
    if config.get("ase_diversity", False) != (architecture == GAIFO_ASE_ARCHITECTURE):
        raise ValueError(f"checkpoint ASE setting does not match architecture in {path}")
    step = payload.get("step")
    if type(step) is not int or step < 0:
        raise ValueError(f"checkpoint has an invalid training step: {path}")
    for name in ("n_sim", "rollout"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"checkpoint has an invalid {name}: {path}")
    if step % (config["n_sim"] * N_CARS):
        raise ValueError(f"checkpoint step is not a complete vector step: {path}")
    required = (
        "policy", "critic", "discriminator", "policy_optimizer",
        "critic_optimizer", "discriminator_optimizer",
    )
    if architecture == GAIFO_ASE_ARCHITECTURE:
        required += (
            "skill_encoder", "skill_encoder_optimizer",
            "skill_rng_state", "skill_encoder_rng_state",
        )
    # Older dual-timescale checkpoints may also include long-discriminator state.
    # Its short discriminator and optimizer remain compatible with this trainer.
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"checkpoint is missing {', '.join(missing)}: {path}")
    if architecture == GAIFO_ASE_ARCHITECTURE:
        for name in (
            "policy", "critic", "discriminator", "skill_encoder",
            "policy_optimizer", "critic_optimizer", "discriminator_optimizer",
            "skill_encoder_optimizer",
        ):
            if not finite_checkpoint_state(payload[name]):
                raise ValueError(
                    f"non-finite {name} in GAIFO checkpoint {path}; use an earlier checkpoint"
                )
    if "clock" in payload:
        try:
            clock = Clock(**payload["clock"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"checkpoint has an invalid training clock: {path}") from error
        if clock.env_steps != step:
            raise ValueError(f"checkpoint clock does not match step {step}: {path}")
    return payload


def validate_resume_args(args: argparse.Namespace, payload: dict | None) -> None:
    if payload is None:
        return
    step = payload["step"]
    if args.timesteps <= step:
        raise ValueError(
            f"--timesteps must exceed checkpoint step {step:,}; "
            "it is the total target, not additional steps"
        )
    config = payload["config"]
    if args.gru != config.get("gru", False):
        raise ValueError(
            "--gru must match the checkpoint architecture when resuming; "
            "start a new run to change policy/critic architecture"
        )
    if args.factorize != config.get("factorize", False):
        raise ValueError("--factorize must match the checkpoint discriminator when resuming")
    if args.ase_diversity != config.get("ase_diversity", False):
        raise ValueError("--ase-diversity must match the checkpoint architecture when resuming")
    ase_settings = (
        "ase_skill_dim", "ase_skill_steps", "ase_encoder_hidden",
        "ase_sequence_length", "ase_encoder_type",
    ) if args.ase_diversity else ()
    for name in (
        "frameskip", "trajectory_length", "policy_hidden", "critic_hidden",
        "discriminator_hidden", "frame_embedding", "temporal_hidden",
    ) + ase_settings:
        if name in ("ase_sequence_length", "ase_encoder_type") and name not in config:
            # Older ASE checkpoints can upgrade their skill predictor in place.
            continue
        if getattr(args, name) != config.get(name):
            raise ValueError(
                f"--{name.replace('_', '-')} must match the checkpoint "
                f"({config.get(name)}) when resuming"
            )


def restore_training_checkpoint(
    payload: dict,
    args: argparse.Namespace,
    modules: dict[str, nn.Module],
    optimizers: dict[str, th.optim.Optimizer],
) -> Clock:
    upgrade_encoder = False
    upgrade_discriminator = False
    for name, module in modules.items():
        if name == "discriminator":
            upgrade_discriminator = load_discriminator_state(module, payload[name])
        elif name == "skill_encoder" and isinstance(module, SkillGRUEncoder) and (
            "ase_encoder_type" not in payload["config"]
        ):
            base = {
                key: value for key, value in payload[name].items()
                if key.startswith(("car_model.", "ball_model."))
            }
            expected = {
                key for key in module.state_dict()
                if key.startswith(("car_model.", "ball_model."))
            }
            if set(base) != expected:
                raise ValueError("legacy ASE checkpoint has incompatible encoder weights")
            module.load_state_dict(base, strict=False)
            upgrade_encoder = True
        elif name == "skill_encoder" and isinstance(module, SkillSequenceEncoder) and (
            "ase_sequence_length" not in payload["config"]
        ):
            missing = module.load_state_dict(payload[name], strict=False)
            expected = {
                key for key in module.state_dict() if key.startswith("sequence_")
            }
            if set(missing.missing_keys) != expected or missing.unexpected_keys:
                raise ValueError("legacy ASE checkpoint has incompatible encoder weights")
            upgrade_encoder = True
        elif name == "skill_encoder" and type(module) is SkillEncoder and (
            "ase_encoder_type" not in payload["config"]
        ) and any(key.startswith("sequence_") for key in payload[name]):
            base = {
                key: value for key, value in payload[name].items()
                if not key.startswith("sequence_")
            }
            module.load_state_dict(base)
            upgrade_encoder = True
        else:
            module.load_state_dict(payload[name])
    for name, optimizer in optimizers.items():
        if name != "skill_encoder" or not upgrade_encoder:
            optimizer.load_state_dict(payload[f"{name}_optimizer"])
            if name == "discriminator" and upgrade_discriminator:
                expand_factorized_optimizer_state(optimizer, modules[name])
        learning_rate = (
            args.ase_encoder_lr if name == "skill_encoder" else (
                args.discriminator_lr if "discriminator" in name else args.ppo_lr
            )
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    if upgrade_encoder:
        print("Initialized ASE skill encoder from checkpoint; reset encoder optimizer")
    if upgrade_discriminator:
        print("Added initial opponent context to factorized discriminator checkpoint")

    if "torch_rng_state" in payload:
        th.set_rng_state(payload["torch_rng_state"].cpu())
    if "cuda_rng_state" in payload:
        th.cuda.set_rng_state_all(payload["cuda_rng_state"])

    if "clock" in payload:
        return Clock(**payload["clock"])
    config = payload["config"]
    vector_steps = payload["step"] // (config["n_sim"] * N_CARS)
    return Clock(
        vector_steps=vector_steps,
        env_steps=payload["step"],
        learner_updates=math.ceil(vector_steps / config["rollout"]),
    )


def parse_args() -> tuple[argparse.Namespace, dict | None]:
    resume_parser = argparse.ArgumentParser(add_help=False)
    resume_parser.add_argument("--resume-checkpoint", type=Path)
    preliminary, _ = resume_parser.parse_known_args()
    resume = (
        load_resume_checkpoint(preliminary.resume_checkpoint)
        if preliminary.resume_checkpoint is not None else None
    )

    parser = argparse.ArgumentParser(
        description="GAIfO imitation learning for 1v1 Rocket League via CARL and JARL."
    )
    parser.add_argument(
        "--resume-checkpoint", type=Path,
        help="restore a GAIFO checkpoint into a new training run",
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        required=resume is None,
        help="1v1 replay folder or its parent containing pro_1v1_fs4",
    )
    parser.add_argument("--n-sim", type=int, default=16_384)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument("--no-touch-timeout", type=float, default=30.0)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument(
        "--trajectory-length", type=int, default=8,
        help="frames in the short discriminator scene window",
    )
    parser.add_argument(
        "--factorize", action=argparse.BooleanOptionalAction, default=False,
        help="train separate car-motion and near-ball-control discriminators with proximity-gated ball reward",
    )
    parser.add_argument(
        "--hard-positive-mining", action=argparse.BooleanOptionalAction, default=False,
        help="bias replay resets toward window starts the trained discriminator confidently recognizes as expert",
    )
    parser.add_argument(
        "--exp-log-odds-reward", action=argparse.BooleanOptionalAction, default=False,
        help="reward exp(log D - log(1-D)) instead of normalized log-odds (D = expert probability)",
    )
    parser.add_argument(
        "--goal-reward-weight", type=float, default=1.0,
        help="scale the +/-1 goal reward per actor (0 disables it)",
    )
    parser.add_argument(
        "--aerial-touch-reward-weight", type=float, default=0.5,
        help="reward all aerial touches, scaling from half this weight at low height to full at the ceiling (0 disables it)",
    )
    parser.add_argument(
        "--flip-reset-reward-weight", type=float, default=1.0,
        help="reward a spent flip returning on an underside ball touch (0 disables it)",
    )
    parser.add_argument("--expert-frame-limit", type=int, default=None)
    parser.add_argument("--replay-reset-fraction", type=float, default=None)
    parser.add_argument(
        "--curated-skill-sampling", action=argparse.BooleanOptionalAction,
        default=True,
        help="restrict replay resets and discriminator positives to matched aerial, dribble, flick, driving, and kickoff clips",
    )
    parser.add_argument(
        "--general-driving-fraction", type=float, default=0.10,
        help="general driving share of curated reset and discriminator sampling",
    )
    parser.add_argument(
        "--kickoff-fraction", type=float, default=0.05,
        help="kickoff share of curated reset and discriminator sampling, separate from general driving",
    )
    parser.add_argument("--discriminator-noise", type=float, default=0.01)
    parser.add_argument(
        "--discriminator-batch", type=int, default=16_384,
        help="agent windows per optimizer step (plus equally many expert windows)",
    )
    parser.add_argument(
        "--discriminator-microbatch", type=int, default=1_024,
        help="agent windows per GRU chunk; accumulates one discriminator optimizer step per effective batch",
    )
    parser.add_argument("--discriminator-epochs", type=int, default=1)
    parser.add_argument("--discriminator-update-interval", type=int, default=4)
    parser.add_argument("--discriminator-lr", type=float, default=3e-4)
    parser.add_argument("--discriminator-hidden", type=int, default=128)
    parser.add_argument("--discriminator-heldout-size", type=int, default=16_384)
    parser.add_argument("--discriminator-accuracy-target", type=float, default=0.80)
    parser.add_argument("--frame-embedding", type=int, default=128)
    parser.add_argument("--temporal-hidden", type=int, default=128)
    parser.add_argument("--history-capacity", type=int, default=262_144)
    parser.add_argument("--history-add-size", type=int, default=16_384)
    parser.add_argument("--history-mix-fraction", type=float, default=0.5)
    parser.add_argument(
        "--recency-replay", action=argparse.BooleanOptionalAction, default=False,
        help="favor recent generated windows while keeping older ones in a reservoir",
    )
    parser.add_argument(
        "--history-reservoir-fraction", type=float, default=0.25,
        help="share of history capacity and samples reserved for old windows with --recency-replay",
    )
    parser.add_argument(
        "--reward-max-magnitude", type=float, default=10.0,
        help="clip standardized imitation rewards or cap exponential rewards",
    )
    parser.add_argument("--ppo-batch", type=int, default=16_384)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument(
        "--gru", action=argparse.BooleanOptionalAction, default=False,
        help="use GRU policy and critic with recurrent PPO (default: MLP)",
    )
    parser.add_argument(
        "--ase-diversity", action=argparse.BooleanOptionalAction, default=False,
        help="train a skill-conditioned MLP with ASE skill discovery and action diversity",
    )
    parser.add_argument("--ase-skill-dim", type=int, default=16)
    parser.add_argument("--ase-skill-steps", type=int, default=32)
    parser.add_argument("--ase-reward-weight", type=float, default=0.5)
    parser.add_argument("--ase-diversity-weight", type=float, default=0.01)
    parser.add_argument("--ase-diversity-batch", type=int, default=1_024)
    parser.add_argument("--ase-encoder-hidden", type=int, default=128)
    parser.add_argument(
        "--ase-encoder-type", choices=("gru", "transformer", "mlp"), default=None,
        help="ASE skill predictor (default: raw-motion GRU; older encoders remain available)",
    )
    parser.add_argument(
        "--ase-sequence-length", type=int, default=None,
        help="skill-aligned encoder context (default: up to 32 steps; 1 for legacy MLP)",
    )
    parser.add_argument("--ase-encoder-batch", type=int, default=4_096)
    parser.add_argument("--ase-encoder-steps", type=int, default=4)
    parser.add_argument("--ase-encoder-lr", type=float, default=3e-4)
    parser.add_argument(
        "--sequence-length", type=int, default=16,
        help="steps per recurrent PPO training sequence when --gru is enabled",
    )
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--lambda", type=float, default=0.95, dest="lambda_", metavar="LAMBDA"
    )
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument(
        "--entropy-end", type=float, default=None,
        help="linearly anneal --entropy to this value over --timesteps (default: constant)",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--policy-hidden", type=int, default=256)
    parser.add_argument("--critic-hidden", type=int, default=256)
    parser.add_argument(
        "--timesteps", type=int, default=2_000_000_000,
        help="total target environment steps, including checkpoint steps",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/gaifo"))
    parser.add_argument("--checkpoint-interval", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    if resume is not None:
        options = {action.dest for action in parser._actions}
        inherited = {
            name: Path(value) if name in {"replay_dir", "log_dir", "checkpoint_dir"}
            else value
            for name, value in resume["config"].items()
            if name in options and name not in (
                "resume_checkpoint", "replay_reset_fraction",
            )
        }
        parser.set_defaults(**inherited)
    args = parser.parse_args()
    if args.replay_reset_fraction is None:
        default = 1.0 if args.curated_skill_sampling else 0.70
        if (resume is not None and args.curated_skill_sampling ==
                resume["config"].get("curated_skill_sampling", False)):
            default = resume["config"].get("replay_reset_fraction", default)
        args.replay_reset_fraction = default
    if args.ase_diversity and args.ase_sequence_length is None:
        args.ase_sequence_length = min(32, args.ase_skill_steps)
    if args.ase_diversity and args.ase_encoder_type is None:
        args.ase_encoder_type = "gru" if args.ase_sequence_length > 1 else "mlp"
    if args.replay_dir is None:
        parser.error("--replay-dir is required when it is absent from the checkpoint")
    return args, resume


def validate_args(args: argparse.Namespace) -> None:
    if not args.replay_dir.is_dir():
        raise FileNotFoundError(args.replay_dir)

    replay_dir = args.replay_dir
    if not next(replay_dir.glob("*.npy"), None):
        for name in (f"pro_1v1_fs{args.frameskip}", "pro_1v1_fs4"):
            candidate = replay_dir / name
            if candidate.is_dir():
                replay_dir = candidate
                break

    found_1v1 = False
    for path in replay_dir.glob("*.npy"):
        source = np.load(path, mmap_mode="r")
        if source.ndim == 2 and source.shape[1] == 161:
            found_1v1 = True
            break
    if not found_1v1:
        raise FileNotFoundError(f"no 1v1 replay files in {args.replay_dir}")
    if replay_dir != args.replay_dir:
        print(f"Using 1v1 replays from {replay_dir}")
        args.replay_dir = replay_dir

    positive = (
        "n_sim",
        "frameskip",
        "max_ticks",
        "rollout",
        "trajectory_length",
        "discriminator_batch",
        "discriminator_microbatch",
        "discriminator_epochs",
        "discriminator_update_interval",
        "discriminator_hidden",
        "discriminator_heldout_size",
        "frame_embedding",
        "temporal_hidden",
        "ppo_batch",
        "ppo_epochs",
        "sequence_length",
        "policy_hidden",
        "critic_hidden",
        "timesteps",
        "checkpoint_interval",
        "checkpoint_keep",
        "history_capacity",
        "history_add_size",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    if args.no_touch_timeout is not None and (
        not math.isfinite(args.no_touch_timeout) or args.no_touch_timeout <= 0.0
    ):
        raise ValueError("--no-touch-timeout must be positive or omitted")
    if args.expert_frame_limit is not None and args.expert_frame_limit < 1:
        raise ValueError("--expert-frame-limit must be positive")
    if (
        args.expert_frame_limit is not None
        and args.expert_frame_limit < args.trajectory_length
    ):
        raise ValueError("--expert-frame-limit must fit one trajectory")
    if (
        not math.isfinite(args.replay_reset_fraction)
        or not 0.0 <= args.replay_reset_fraction <= 1.0
    ):
        raise ValueError("--replay-reset-fraction must be between zero and one")
    if (
        not math.isfinite(args.general_driving_fraction)
        or not 0.0 <= args.general_driving_fraction < 1.0
    ):
        raise ValueError("--general-driving-fraction must be in [0, 1)")
    if not math.isfinite(args.kickoff_fraction) or not 0.0 <= args.kickoff_fraction < 1.0:
        raise ValueError("--kickoff-fraction must be in [0, 1)")
    if args.general_driving_fraction + args.kickoff_fraction >= 1.0:
        raise ValueError("driving and kickoff fractions must sum to less than one")
    if not math.isfinite(args.discriminator_noise) or args.discriminator_noise < 0.0:
        raise ValueError("--discriminator-noise must be non-negative")
    if (
        not math.isfinite(args.ppo_lr)
        or not math.isfinite(args.discriminator_lr)
        or args.ppo_lr <= 0.0
        or args.discriminator_lr <= 0.0
    ):
        raise ValueError("learning rates must be positive")
    if not math.isfinite(args.gamma) or not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1]")
    if not math.isfinite(args.lambda_) or not 0.0 <= args.lambda_ <= 1.0:
        raise ValueError("--lambda must be in [0, 1]")
    if not math.isfinite(args.entropy) or args.entropy < 0.0:
        raise ValueError("--entropy must be non-negative")
    if args.entropy_end is not None and (
        not math.isfinite(args.entropy_end) or args.entropy_end < 0.0
    ):
        raise ValueError("--entropy-end must be finite and non-negative")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive")
    if not math.isfinite(args.ppo_clip) or args.ppo_clip <= 0.0:
        raise ValueError("--ppo-clip must be positive")
    if not math.isfinite(args.value_clip) or args.value_clip < 0.0:
        raise ValueError("--value-clip must be non-negative")
    if not math.isfinite(args.value_coef) or args.value_coef < 0.0:
        raise ValueError("--value-coef must be non-negative")
    if (
        not math.isfinite(args.history_mix_fraction)
        or not 0.0 <= args.history_mix_fraction < 1.0
    ):
        raise ValueError("--history-mix-fraction must be in [0, 1)")
    if (
        not math.isfinite(args.history_reservoir_fraction)
        or not 0.0 < args.history_reservoir_fraction <= 0.5
    ):
        raise ValueError("--history-reservoir-fraction must be in (0, 0.5]")
    if args.recency_replay and args.history_capacity < 2:
        raise ValueError("--history-capacity must be at least two with --recency-replay")
    if (
        not math.isfinite(args.discriminator_accuracy_target)
        or not 0.0 <= args.discriminator_accuracy_target <= 1.0
    ):
        raise ValueError("--discriminator-accuracy-target must be between zero and one")
    if not math.isfinite(args.reward_max_magnitude) or args.reward_max_magnitude <= 0.0:
        raise ValueError("--reward-max-magnitude must be positive")
    if not math.isfinite(args.goal_reward_weight) or args.goal_reward_weight < 0.0:
        raise ValueError("--goal-reward-weight must be finite and non-negative")
    if (
        not math.isfinite(args.aerial_touch_reward_weight)
        or args.aerial_touch_reward_weight < 0.0
    ):
        raise ValueError("--aerial-touch-reward-weight must be finite and non-negative")
    if (
        not math.isfinite(args.flip_reset_reward_weight)
        or args.flip_reset_reward_weight < 0.0
    ):
        raise ValueError("--flip-reset-reward-weight must be finite and non-negative")
    if args.history_add_size > args.history_capacity:
        raise ValueError(
            "--history-add-size must not exceed --history-capacity"
        )

    if args.rollout < args.trajectory_length - 1:
        raise ValueError("--rollout must be at least --trajectory-length - 1")

    generated_windows = max(
        0, args.rollout - (args.trajectory_length - 2)
    ) * args.n_sim * N_CARS
    if generated_windows < args.discriminator_batch:
        raise ValueError(
            f"rollout is too short to produce a discriminator batch: "
            f"{generated_windows} generated windows, batch size {args.discriminator_batch}"
        )

    n_envs = args.n_sim * 2
    if args.ppo_batch > args.rollout * n_envs:
        raise ValueError("--ppo-batch must fit the rollout size")
    if args.ase_diversity and args.gru:
        raise ValueError("--ase-diversity currently supports MLP policies only")
    if args.gru:
        if args.rollout % args.sequence_length:
            raise ValueError("--rollout must be divisible by --sequence-length")
        if args.ppo_batch % args.sequence_length:
            raise ValueError("--ppo-batch must be divisible by --sequence-length")
    if args.ase_diversity:
        if args.ase_skill_dim < 2:
            raise ValueError("--ase-skill-dim must be at least two")
        for name in (
            "ase_skill_steps", "ase_diversity_batch", "ase_encoder_hidden",
            "ase_encoder_batch", "ase_encoder_steps", "ase_encoder_lr",
            "ase_sequence_length",
        ):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if args.ase_sequence_length > args.ase_skill_steps:
            raise ValueError("--ase-sequence-length cannot exceed --ase-skill-steps")
        if args.ase_encoder_type != "mlp" and args.ase_sequence_length < 2:
            raise ValueError("temporal ASE encoder requires --ase-sequence-length at least two")
        for name in ("ase_reward_weight", "ase_diversity_weight"):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
                raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")


def build_env(args: argparse.Namespace) -> CARLTorchVectorEnv:
    return CARLTorchVectorEnv(
        n_sim=args.n_sim,
        n_blue=1,
        n_orange=1,
        seed=args.seed,
        frameskip=args.frameskip,
        max_ticks=args.max_ticks,
        no_touch_timeout_seconds=args.no_touch_timeout,
        normalize=True,
        discrete_actions=True,
    )


def build_policy(env, args: argparse.Namespace) -> MultiCategoricalPolicy:
    return MultiCategoricalPolicy(
        foot=LinearEncoder(args.policy_hidden, func=nn.ReLU),
        body=(
            GRU(hidden_size=args.policy_hidden) if args.gru
            else MLP(dims=[args.policy_hidden], func=nn.ReLU)
        ),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=0.01)),
        action_codec=env.action_codec,
    ).build(env).to(env.device)


def build_critic(env, args: argparse.Namespace) -> Critic:
    return Critic(
        foot=LinearEncoder(args.critic_hidden, func=nn.ReLU),
        body=(
            GRU(hidden_size=args.critic_hidden) if args.gru
            else MLP(dims=[args.critic_hidden], func=nn.ReLU)
        ),
        head=MLP(dims=[], out_init_func=orthogonal_init(std=1.0)),
    ).build(env).to(env.device)


def build_discriminator(
    args: argparse.Namespace,
) -> SceneDiscriminator | FactorizedSceneDiscriminator:
    model = FactorizedSceneDiscriminator if args.factorize else SceneDiscriminator
    return model(
        frame_embedding=args.frame_embedding,
        temporal_hidden=args.temporal_hidden,
        hidden_size=args.discriminator_hidden,
    )


def build_runner(
    env, policy, critic, buffer, args,
    gameplay: GameplayDiagnostics | None = None,
) -> Runner:
    captures = [LogProbCapture()]
    if args.gru:
        captures.extend((RecurrentStateCapture(), RecurrentCriticCapture(critic)))
    else:
        captures.append(CriticCapture(critic))
    if gameplay is not None:
        captures.append(AdvancedTouchCapture(gameplay))
        if getattr(args, "ase_diversity", False) or getattr(args, "factorize", False):
            captures.append(ASEBallTouchCapture(gameplay))
    captures.append(SceneWindowCapture(args.trajectory_length))
    return Runner(env, policy, buffer, captures=captures)


def build_ppo_sampler(args):
    if args.gru:
        return RecurrentRolloutMinibatches(
            sequence_length=args.sequence_length,
            sequences_per_batch=args.ppo_batch // args.sequence_length,
            epochs=args.ppo_epochs,
        )
    return RolloutMinibatches(args.ppo_batch, args.ppo_epochs)


def build_entropy_scheduler(
    args: argparse.Namespace, ppo_loss: PPOLoss
) -> ValueScheduler | None:
    if args.entropy_end is None:
        return None

    def set_entropy_coef(value: float) -> None:
        ppo_loss.config = replace(ppo_loss.config, entropy_coef=value)

    return ValueScheduler(ScheduledValue(
        "entropy_coef",
        LinearSchedule(args.entropy, args.entropy_end),
        set_entropy_coef,
    ))


def main() -> None:
    args, resume = parse_args()
    validate_resume_args(args, resume)
    validate_args(args)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_env = build_env(args)
    gameplay = base_env.register_reward(GameplayDiagnostics(
        base_env.n_sim,
        base_env.device,
        math.ceil(args.no_touch_timeout * 120.0 / args.frameskip),
    ))
    skill_env = (
        SkillConditionedEnv(base_env, args.ase_skill_dim, args.ase_skill_steps, args.seed)
        if args.ase_diversity else None
    )
    env = skill_env if skill_env is not None else base_env
    policy = build_policy(env, args)
    critic = build_critic(env, args)
    discriminator = build_discriminator(args).to(env.device)
    skill_stream = SkillStreamContext() if args.ase_encoder_type == "gru" else None
    skill_encoder = None
    if args.ase_diversity:
        if args.ase_encoder_type == "gru":
            skill_encoder = SkillGRUEncoder(
                args.ase_skill_dim, args.ase_encoder_hidden, args.ase_sequence_length,
            )
        elif args.ase_encoder_type == "transformer":
            skill_encoder = SkillSequenceEncoder(
                args.ase_skill_dim, args.ase_encoder_hidden, args.ase_sequence_length,
            )
        else:
            skill_encoder = SkillEncoder(args.ase_skill_dim, args.ase_encoder_hidden)
        skill_encoder = skill_encoder.to(env.device)

    expert = ExpertSceneDataset(
        args.replay_dir,
        args.trajectory_length,
        args.expert_frame_limit,
        args.seed,
        frame_skip=args.frameskip,
        device=env.device,
        heldout_size=args.discriminator_heldout_size,
        reject_discontinuities=args.curated_skill_sampling,
        skill_sampling=args.curated_skill_sampling,
        driving_fraction=args.general_driving_fraction,
        kickoff_fraction=args.kickoff_fraction,
    )
    if expert.train_total < 1:
        raise ValueError("expert dataset contains no training windows")
    if args.curated_skill_sampling and not len(expert.reset_indices):
        raise ValueError("expert dataset contains no safe curated replay reset states")

    reset_dataset = expert.reset_dataset()
    curated_resets = (
        CuratedReplayResetTransform(expert)
        if args.curated_skill_sampling else None
    )
    reset_miner = (
        ConfidentExpertResetTransform(
            expert, reset_dataset, discriminator, args.discriminator_microbatch,
        ) if args.hard_positive_mining else None
    )
    base_env.reset_state_provider = ReplayResetProvider(
        DatasetResetSampler(
            reset_dataset,
            transforms=(
                *((curated_resets,) if curated_resets is not None else ()),
                *((reset_miner,) if reset_miner is not None else ()),
            ),
            probability=args.replay_reset_fraction,
            seed=args.seed,
        ),
        expert.frames,
        expert.internal_states,
    )

    history_options = dict(
        capacity=args.history_capacity, trajectory_length=args.trajectory_length,
        device=env.device, seed=args.seed,
    )
    history = (
        RecencyReplayBuffer(
            **history_options, reservoir_fraction=args.history_reservoir_fraction,
        )
        if args.recency_replay else HistoricalReplayBuffer(**history_options)
    )

    policy_optimizer = th.optim.Adam(policy.parameters(), lr=args.ppo_lr)
    critic_optimizer = th.optim.Adam(critic.parameters(), lr=args.ppo_lr)
    discriminator_optimizer = th.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr
    )
    skill_optimizer = (
        th.optim.Adam(skill_encoder.parameters(), lr=args.ase_encoder_lr)
        if skill_encoder is not None else None
    )
    skill_update = (
        SkillEncoderUpdate(
            skill_encoder, skill_optimizer, args.ase_encoder_batch,
            args.ase_encoder_steps, args.max_grad_norm, args.seed, env.device,
            stream_context=skill_stream,
        ) if skill_encoder is not None else None
    )
    restored_clock = None
    if resume is not None:
        modules = {
            "policy": policy, "critic": critic, "discriminator": discriminator,
        }
        optimizers = {
            "policy": policy_optimizer, "critic": critic_optimizer,
            "discriminator": discriminator_optimizer,
        }
        if skill_encoder is not None:
            modules["skill_encoder"] = skill_encoder
            optimizers["skill_encoder"] = skill_optimizer
        restored_clock = restore_training_checkpoint(
            resume, args, modules, optimizers,
        )
        if skill_env is not None:
            skill_env.generator.set_state(resume["skill_rng_state"])
            skill_update.generator.set_state(resume["skill_encoder_rng_state"])
        if reset_miner is not None:
            reset_miner.ready = restored_clock.learner_updates > 0

    buffer = RolloutBuffer(
        args.rollout, env.n_envs, env.device, copy_on_finish=False
    )
    runner = build_runner(env, policy, critic, buffer, args, gameplay)

    discriminator_update = AdaptiveDiscriminatorUpdate(
        expert=expert,
        history=history,
        batch_size=args.discriminator_batch,
        epochs=args.discriminator_epochs,
        noise_std=args.discriminator_noise,
        heldout_size=args.discriminator_heldout_size,
        accuracy_target=args.discriminator_accuracy_target,
        history_add_size=args.history_add_size,
        history_mix_fraction=args.history_mix_fraction,
        max_grad_norm=args.max_grad_norm,
        discriminator=discriminator,
        optimizer=discriminator_optimizer,
        loss=SceneDiscriminatorLoss(discriminator),
        update_interval=args.discriminator_update_interval,
        microbatch_size=args.discriminator_microbatch,
        reset_miner=reset_miner,
    )

    ppo_config = PPOConfig(
        clip=args.ppo_clip,
        value_clip=args.value_clip,
        value_coef=args.value_coef,
        entropy_coef=args.entropy,
        normalize_advantage=True,
    )
    ppo_loss = (
        ASEPPOLoss(
            policy, critic, ppo_config, args.ase_skill_dim,
            args.ase_diversity_weight, args.ase_diversity_batch,
        ) if skill_encoder is not None else PPOLoss(policy, critic, ppo_config)
    )
    skill_reward = (
        SkillDiscoveryReward(
            skill_encoder, args.ase_reward_weight, args.ase_encoder_batch,
            stream_context=skill_stream,
        )
        if skill_encoder is not None else None
    )
    ppo_optimizer_steps = (
        OptimizerStep(policy, policy_optimizer, max_grad_norm=args.max_grad_norm),
        OptimizerStep(critic, critic_optimizer, max_grad_norm=args.max_grad_norm),
    )
    ppo_update = Update(
        transforms=(
            SceneDiscriminatorReward(
                discriminator=discriminator,
                noise_std=args.discriminator_noise,
                trajectory_length=args.trajectory_length,
                goal_reward_weight=args.goal_reward_weight,
                aerial_touch_reward_weight=args.aerial_touch_reward_weight,
                flip_reset_reward_weight=args.flip_reset_reward_weight,
                batch_size=args.discriminator_microbatch,
                max_magnitude=args.reward_max_magnitude,
                exp_log_odds_reward=args.exp_log_odds_reward,
            ),
            *((skill_reward,) if skill_reward is not None else ()),
            GAE(
                gamma=args.gamma,
                lambda_=args.lambda_,
                reward_field="training_reward",
            ),
            SelectPPOFields(recurrent=args.gru),
        ),
        sampler=build_ppo_sampler(args),
        loss=ppo_loss,
        optimizer_step=(
            FiniteIndependentOptimizerSteps(*ppo_optimizer_steps)
            if skill_encoder is not None
            else IndependentOptimizerSteps(*ppo_optimizer_steps)
        ),
        section="PPO",
    )
    value_scheduler = build_entropy_scheduler(args, ppo_loss)

    learner = Algorithm(discriminator_update, *(
        (skill_update,) if skill_update is not None else ()
    ), ppo_update)

    run_id = datetime.now().strftime("gaifo-%Y%m%d-%H%M%S-%f")
    logger = Logger(args.log_dir / run_id)
    for section, key, label, fmt in (
        ("Discriminator", "loss", "D loss", ".4f"),
        ("Discriminator", "agent_score", "D agent score", ".3f"),
        ("Discriminator", "expert_score", "D expert score", ".3f"),
        ("Discriminator", "heldout_accuracy", "D heldout accuracy", ".3f"),
        ("Discriminator", "updated", "D updated", ".0f"),
        ("Discriminator", "minibatches", "D batches", ".0f"),
        ("PPO", "policy_loss", "policy loss", ".4f"),
        ("PPO", "critic_loss", "critic loss", ".4f"),
        ("PPO", "entropy", "entropy", ".3f"),
        ("Gameplay", "touches_per_1000_steps", "touches/1k", ".3f"),
        ("Gameplay", "aerial_touches_per_1000_steps", "aerial/1k", ".3f"),
        ("Gameplay", "flip_resets_per_1000_steps", "flip resets/1k", ".3f"),
        ("Gameplay", "goals_for_per_1000_steps", "goals for/1k", ".3f"),
        ("Gameplay", "goals_against_per_1000_steps", "goals against/1k", ".3f"),
        ("Gameplay", "timeout_fraction", "timeout frac", ".3f"),
    ):
        logger.register_progress_metric(section, key, label, fmt)
    if args.factorize:
        for key, label in (
            ("car_heldout_accuracy", "D car accuracy"),
            ("ball_heldout_accuracy", "D ball accuracy"),
            ("ball_near_heldout_accuracy", "D ball near accuracy"),
            ("train_near_ball_fraction", "D near-ball fraction"),
        ):
            logger.register_progress_metric("Discriminator", key, label, ".3f")
    if args.hard_positive_mining:
        logger.register_progress_metric(
            "Discriminator", "reset_mined_fraction", "D mined reset frac", ".3f",
        )
    if skill_update is not None:
        for section, key, label, fmt in (
            ("Skill", "encoder_loss", "skill encoder loss", ".3f"),
            ("Skill", "alignment", "skill alignment", ".3f"),
            ("Skill", "ball_credit", "ASE ball credit", ".3f"),
            ("Skill", "reward", "ASE reward", ".3f"),
            ("PPO", "ase_diversity_loss", "ASE diversity loss", ".3f"),
            ("PPO", "ase_action_kl", "ASE action KL", ".3f"),
        ):
            logger.register_progress_metric(section, key, label, fmt)
        if isinstance(skill_encoder, (SkillSequenceEncoder, SkillGRUEncoder)):
            logger.register_progress_metric(
                "Skill", "end_alignment", "skill end alignment", ".3f",
            )
            logger.register_progress_metric(
                "Skill", "context_steps", "skill context steps", ".1f",
            )
        if isinstance(skill_encoder, SkillGRUEncoder):
            for key, label in (
                ("car_alignment", "ASE car alignment"),
                ("ball_alignment", "ASE owned-ball alignment"),
            ):
                logger.register_progress_metric("Skill", key, label, ".3f")
    if value_scheduler is not None:
        logger.register_progress_metric(
            "Schedule", "entropy_coef", "entropy coef", ".4f"
        )

    checkpoints = GAIFOCheckpoints(
        args.checkpoint_dir / run_id,
        args.checkpoint_interval,
        args.checkpoint_keep,
        policy,
        critic,
        discriminator,
        policy_optimizer,
        critic_optimizer,
        discriminator_optimizer,
        buffer,
        args,
        skill_encoder=skill_encoder,
        skill_optimizer=skill_optimizer,
        skill_env=skill_env,
        skill_update=skill_update,
    )

    def update_callback(trainer: Trainer) -> None:
        metrics = gameplay.diagnostic_metrics()
        if curated_resets is not None:
            fractions = curated_resets.take_skill_fractions()
            if fractions:
                metrics.setdefault("Replay", {}).update(fractions)
        if skill_reward is not None and skill_reward.last_mean is not None:
            metrics.setdefault("Skill", {})["reward"] = skill_reward.last_mean
        if metrics:
            trainer.logger.update(metrics, step=trainer.clock.env_steps)

    trainer = Trainer(
        runner,
        buffer,
        learner,
        OnPolicySchedule(),
        logger=logger,
        checkpoint=checkpoints,
        value_scheduler=value_scheduler,
        update_callback=update_callback,
    )
    if restored_clock is not None:
        trainer.clock = restored_clock
        print(
            f"Resuming {args.resume_checkpoint} at {trainer.clock.env_steps:,} "
            f"steps in new run {run_id}"
        )
    checkpoints.clock = trainer.clock

    try:
        checkpoints.save(trainer.clock.env_steps, force=True)
        trainer.run(args.timesteps)
        checkpoints.save(trainer.clock.env_steps, force=True)
    finally:
        logger.close()
        env.close()


if __name__ == "__main__":
    main()
