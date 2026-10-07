import argparse
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

from dodge_window import DodgeAwareCARLTorchVectorEnv
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
GLOBAL_DISCRIMINATOR_WEIGHT = 0.5
SPECIALIST_DISCRIMINATOR_WEIGHT = 0.5
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


class EgoBallTouchCapture(CaptureBase):
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
        self._context_situation_pools: dict[bool, tuple[th.Tensor, ...]] = {}
        self._curated_reset_pools: tuple[th.Tensor, ...] | None = None
        self._context_run_starts: dict[bool, th.Tensor] = {}
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

    def sample_povs(self, n: int, *, heldout: bool = False) -> th.Tensor:
        """Choose expert window starts and their stored focal viewpoint."""
        if n < 1:
            raise ValueError("expert sample count must be positive")
        generator = self._heldout_generator if heldout else self._train_generator
        if self.skill_sampling:
            categories = th.multinomial(
                self.curated_weights(heldout=heldout), n, replacement=True,
                generator=generator,
            )
            pairs = th.empty((n, 2), dtype=th.long, device=self.frames.device)
            for category, pool in enumerate(self.curated_pools(heldout=heldout)):
                selected = (categories == category).nonzero(as_tuple=True)[0]
                if len(selected):
                    pairs[selected] = pool[th.randint(
                        len(pool), (len(selected),), device=self.frames.device,
                        generator=generator,
                    )]
            return pairs
        starts = self.heldout_window_starts if heldout else self.train_window_starts
        if not len(starts):
            raise ValueError("no expert windows in the requested split")
        chosen = starts[th.randint(
            len(starts), (n,), device=self.frames.device, generator=generator,
        )]
        opponent = (self.opponent_pov_available[chosen]
                    & (th.rand(n, device=self.frames.device, generator=generator) < 0.5))
        return th.stack((chosen, opponent.long()), dim=-1)

    def context_frames(
        self, pairs: th.Tensor, length: int, *, heldout: bool = False,
        max_age: th.Tensor | None = None,
        return_age: bool = False,
    ) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        """Causal expert scenes, stopped at split, replay, or unsafe-window gaps."""
        if pairs.ndim != 2 or pairs.shape[1] != 2 or length < 1:
            raise ValueError("expert context needs window/actor pairs and a positive length")
        if not len(pairs):
            empty = self.frames.new_empty((0, length, self.frames.shape[-1]))
            return (empty, pairs.new_empty(0)) if return_age else empty
        starts = self.heldout_window_starts if heldout else self.train_window_starts
        if heldout not in self._context_run_starts:
            beginning = th.ones(len(starts), dtype=th.bool, device=starts.device)
            beginning[1:] = starts[1:] != starts[:-1] + 1
            run_starts = th.where(beginning, starts, 0).cummax(0).values
            self._context_run_starts[heldout] = run_starts
        position = th.searchsorted(starts, pairs[:, 0].contiguous())
        if (position >= len(starts)).any() or not th.equal(starts[position], pairs[:, 0]):
            raise ValueError("expert context includes an unavailable window")
        run_start = self._context_run_starts[heldout][position]
        if max_age is not None:
            if max_age.shape != (len(pairs),) or (max_age < 1).any():
                raise ValueError("expert context age must match selected windows")
            run_start = th.maximum(run_start, pairs[:, 0] - max_age + 1)
        ages = (pairs[:, 0] - run_start + 1).clamp(max=length)
        offsets = th.arange(length - 1, -1, -1, device=pairs.device)
        indices = th.maximum(pairs[:, :1] - offsets, run_start[:, None])
        scenes = self.frames[indices + self.partition_span]
        opponent = pairs[:, 1].bool()
        if opponent.any():
            scenes[opponent] = opponent_view(scenes[opponent])
        return (scenes, ages) if return_age else scenes

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

    def context_situation_pools(self, *, heldout: bool = False) -> tuple[th.Tensor, ...]:
        """Match recurrent positives to the same curated scene situations."""
        if not self.skill_sampling:
            return self.situation_pools(heldout=heldout)
        if heldout not in self._context_situation_pools:
            groups: list[list[th.Tensor]] = [[] for _ in range(N_SITUATIONS)]
            for pool in self.curated_pools(heldout=heldout):
                for chunk in pool.split(8_192):
                    labels = scene_situation_ids(self._windows_for_povs(chunk))
                    for label in labels.unique().tolist():
                        groups[label].append(chunk[labels == label])
            empty = self.train_window_starts.new_empty((0, 2))
            self._context_situation_pools[heldout] = tuple(
                th.cat(group) if group else empty for group in groups
            )
        return self._context_situation_pools[heldout]

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
    """Structured discriminator for whole-scene trajectory windows."""

    def __init__(
        self,
        frame_embedding: int,
        temporal_hidden: int,
        hidden_size: int = 128,
        *, n_cars: int = N_CARS, recurrent_global: bool = False,
    ) -> None:
        super().__init__()
        if min(frame_embedding, temporal_hidden, hidden_size) < 1:
            raise ValueError("discriminator dimensions must be positive")
        if n_cars not in (N_CARS, DOUBLES_N_CARS):
            raise ValueError("scene discriminator needs two or four cars")
        self.n_cars = n_cars
        self.scene_size = BALL_SIZE + n_cars * CAR_SIZE
        self.recurrent_global = recurrent_global
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
            frame_embedding * (n_cars + 1), temporal_hidden, batch_first=True
        )
        self.head = nn.Linear(temporal_hidden, 1)

    def _encode(self, scenes: th.Tensor) -> th.Tensor:
        B, T, _ = scenes.shape
        ball = scenes[..., :BALL_SIZE]
        cars = scenes[..., BALL_SIZE:].reshape(B, T, self.n_cars, CAR_SIZE)
        sign = scenes.new_tensor(
            [1.0] * (self.n_cars // 2) + [-1.0] * (self.n_cars // 2)
        ).view(1, 1, self.n_cars, 1)
        car_in = th.cat((cars, sign.expand(B, T, -1, -1)), dim=-1)
        ball_emb = self.ball_encoder(ball)
        car_emb = self.car_encoder(car_in).flatten(-2)
        return th.cat((ball_emb, car_emb), dim=-1)

    def forward(self, windows: th.Tensor) -> th.Tensor:
        if windows.ndim != 3 or windows.shape[-1] != self.scene_size:
            raise ValueError("scene discriminator requires two- or four-car scene windows")
        gru_out, _ = self.gru(self._encode(windows))
        return self.head(gru_out[:, -1]).squeeze(-1)

    def score_context(self, scenes: th.Tensor, ages: th.Tensor) -> th.Tensor:
        """Skip left padding so a new episode starts from a zero recurrent state."""
        if (scenes.ndim != 3 or scenes.shape[-1] != self.scene_size
                or ages.shape != (len(scenes),)):
            raise ValueError("scene context and ages must match")
        length = scenes.shape[1]
        lengths = ages.cpu()  # Packed sequences need CPU lengths; validate there too.
        if (lengths < 1).any() or (lengths > length).any():
            raise ValueError("scene context and ages must match")
        steps = th.arange(length, device=scenes.device)
        indices = (steps[None] + length - ages[:, None]).clamp(max=length - 1)
        # Move real scenes to the front; the packed GRU skips the new right padding.
        aligned = scenes.gather(1, indices[..., None].expand_as(scenes))
        packed = pack_padded_sequence(
            self._encode(aligned), lengths, batch_first=True, enforce_sorted=False,
        )
        _, state = self.gru(packed)
        return self.head(state[0]).squeeze(-1)

    def score_sequence(
        self, scenes: th.Tensor, reset: th.Tensor,
        initial_state: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Score chronological scene frames, resetting memory at episode starts."""
        if (not self.recurrent_global or scenes.ndim != 3
                or scenes.shape[-1] != self.scene_size
                or reset.shape != scenes.shape[:2]):
            raise ValueError("recurrent scene scoring needs [time, actors, scene] and reset")
        features = self._encode(scenes.transpose(0, 1))
        state = (
            features.new_zeros((1, scenes.shape[1], self.gru.hidden_size))
            if initial_state is None else initial_state
        )
        if state.shape != (1, scenes.shape[1], self.gru.hidden_size):
            raise ValueError("initial discriminator state must match actors")
        if not reset[1:].any():
            state = state.masked_fill(reset[0][None, :, None], 0)
            output, state = self.gru(features, state)
            return self.head(output).squeeze(-1).transpose(0, 1), state

        # Each actor's runs between resets are independent. Process them in
        # one packed GRU call even when different actors reset on different steps.
        starts_by_actor = reset.transpose(0, 1).clone()
        starts_by_actor[:, 0] = True
        actor, start = starts_by_actor.nonzero(as_tuple=True)
        if len(start) > max(2 * scenes.shape[1], 64):
            # Frequent resets would otherwise create an oversized padded batch.
            logits = []
            for step in range(len(scenes)):
                state = state.masked_fill(reset[step][None, :, None], 0)
                output, state = self.gru(features[:, step:step + 1], state)
                logits.append(self.head(output[:, 0]).squeeze(-1))
            return th.stack(logits), state

        steps = th.arange(len(scenes), device=scenes.device)
        times = start[:, None] + steps[None, :]
        same_actor = actor[1:] == actor[:-1]
        lengths = th.empty_like(start)
        lengths[:-1] = th.where(
            same_actor, start[1:] - start[:-1], len(scenes) - start[:-1],
        )
        lengths[-1] = len(scenes) - start[-1]
        packed = pack_padded_sequence(
            features[actor[:, None], times.clamp(max=len(scenes) - 1)],
            lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        initial = state[:, actor].masked_fill(
            ((start > 0) | reset[0, actor])[None, :, None], 0,
        )
        output, final_states = self.gru(packed, initial)
        unpacked, _ = pad_packed_sequence(output, batch_first=True, total_length=len(scenes))
        valid = steps[None, :] < lengths[:, None]
        logits = features.new_empty((len(scenes), scenes.shape[1]))
        logits[times[valid], actor[:, None].expand_as(times)[valid]] = (
            self.head(unpacked).squeeze(-1)[valid]
        )
        last_run = th.cat((~same_actor, same_actor.new_ones(1)))
        return logits, final_states[:, last_run]


class CausalSceneTransformer(nn.Module):
    """Judge bounded, causal histories of the ball and both 1v1 cars."""

    transformer_global = True
    recurrent_global = False
    n_cars = N_CARS
    scene_size = SCENE_SIZE

    def __init__(
        self, frame_embedding: int, temporal_hidden: int, hidden_size: int = 128,
        *, max_context: int = 128, layers: int = 2,
    ) -> None:
        super().__init__()
        if (min(frame_embedding, temporal_hidden, hidden_size, max_context, layers) < 1
                or temporal_hidden % 4):
            raise ValueError("Transformer dimensions must be positive and divisible by four")
        self.max_context = max_context
        self.ball_encoder = nn.Sequential(
            nn.Linear(BALL_SIZE, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding), nn.ReLU(),
        )
        self.car_encoder = nn.Sequential(
            nn.Linear(CAR_SIZE + 2, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding), nn.ReLU(),
        )
        self.frame_projection = nn.Linear(frame_embedding * 3, temporal_hidden)
        self.position = nn.Embedding(max_context, temporal_hidden)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=temporal_hidden, nhead=4,
                dim_feedforward=max(temporal_hidden * 4, hidden_size * 2),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ), num_layers=layers, enable_nested_tensor=False,
        )
        self.head = nn.Linear(temporal_hidden, 1)

    def _encode(self, scenes: th.Tensor) -> th.Tensor:
        count, steps = scenes.shape[:2]
        cars = scenes[..., BALL_SIZE:].reshape(count, steps, N_CARS, CAR_SIZE)
        roles = th.eye(N_CARS, device=scenes.device, dtype=scenes.dtype)
        roles = roles.view(1, 1, N_CARS, N_CARS).expand(count, steps, -1, -1)
        encoded = self.car_encoder(th.cat((cars, roles), dim=-1))
        combined = th.cat((
            self.ball_encoder(scenes[..., :BALL_SIZE]),
            encoded[:, :, 0], encoded[:, :, 1],
        ), dim=-1)
        return self.frame_projection(combined)

    def score_context(
        self, scenes: th.Tensor, ages: th.Tensor, *, return_previous: bool = False,
    ) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        """Score a real prefix, optionally including the score before its final frame."""
        if (scenes.ndim != 3 or scenes.shape[-1] != SCENE_SIZE
                or ages.shape != (len(scenes),) or not 1 <= scenes.shape[1] <= self.max_context
                or (ages < 1).any() or (ages > scenes.shape[1]).any()):
            raise ValueError("Transformer context needs valid 1v1 scenes and lengths")
        # Align real left-padded scenes at the start before applying the causal mask.
        scenes = scenes[:, -int(ages.max()):]
        length = scenes.shape[1]
        steps = th.arange(length, device=scenes.device)
        indices = (steps[None] + length - ages[:, None]).clamp(max=length - 1)
        aligned = scenes.gather(1, indices[..., None].expand_as(scenes))
        embedded = self._encode(aligned) + self.position(steps)[None]
        causal = th.ones(length, length, dtype=th.bool, device=scenes.device).triu(1)
        padding = steps[None] >= ages[:, None]
        output = self.temporal(embedded, mask=causal, src_key_padding_mask=padding)
        logits = self.head(output).squeeze(-1)
        current = logits.gather(1, (ages - 1)[:, None]).squeeze(1)
        if not return_previous:
            return current
        previous = logits.gather(1, (ages - 2).clamp(min=0)[:, None]).squeeze(1)
        return current, th.where(ages > 1, previous, th.zeros_like(previous))

    def forward(self, windows: th.Tensor) -> th.Tensor:
        if windows.ndim != 3:
            raise ValueError("Transformer requires [batch, frames, scene] windows")
        windows = windows[:, -self.max_context:]
        ages = th.full((len(windows),), windows.shape[1], dtype=th.long,
                       device=windows.device)
        return self.score_context(windows, ages)


class EgoBallSceneDiscriminator(nn.Module):
    """Judge the joint ego-car and ball trajectory without opponent shortcuts."""

    def __init__(self, frame_embedding: int, temporal_hidden: int, hidden_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(BALL_SIZE + CAR_SIZE + 6, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding), nn.ReLU(),
        )
        self.gru = nn.GRU(frame_embedding, temporal_hidden, batch_first=True)
        self.head = nn.Linear(temporal_hidden, 1)

    def forward(self, windows: th.Tensor) -> th.Tensor:
        if windows.ndim != 3:
            raise ValueError("near-ball discriminator needs [batch, frames, scene] windows")
        scene_car_count(windows)
        ball = windows[..., :BALL_SIZE]
        ego = windows[..., BLUE_START:BLUE_START + CAR_SIZE]
        relative_position = ball[..., :3] - ego[..., :3]
        relative_velocity = ball[..., 3:6] - ego[..., 3:6] * (CAR_MAX_SPEED / BALL_MAX_SPEED)
        inputs = th.cat((ball, ego, relative_position, relative_velocity), dim=-1)
        features, _ = self.gru(self.encoder(inputs))
        return self.head(features[:, -1]).squeeze(-1)


class FactorizedSceneDiscriminator(nn.Module):
    """Score the whole scene plus a proximity-selected ego-motion/interaction head.

    Only the global discriminator sees other cars. Legacy car/ball
    checkpoints can be inspected in their original form using the private flags.
    """

    factorized = True

    def __init__(
        self, frame_embedding: int, temporal_hidden: int, hidden_size: int = 128,
        *, n_cars: int = N_CARS, _legacy_two_heads: bool = False,
        _legacy_opponent_context: bool = False, recurrent_global: bool = False,
        transformer_global: bool = False, context_length: int = 128,
    ) -> None:
        super().__init__()
        if min(frame_embedding, temporal_hidden, hidden_size) < 1:
            raise ValueError("discriminator dimensions must be positive")
        if n_cars not in (N_CARS, DOUBLES_N_CARS):
            raise ValueError("factorized discriminator needs two or four cars")
        if _legacy_opponent_context and not _legacy_two_heads:
            raise ValueError("opponent context is only supported for legacy checkpoints")
        if transformer_global and (n_cars != N_CARS or recurrent_global or _legacy_two_heads):
            raise ValueError("1v1 Transformer requires a non-recurrent global head")
        self.n_cars = n_cars
        self.scene_size = BALL_SIZE + n_cars * CAR_SIZE
        self.recurrent_global = recurrent_global
        self.transformer_global = transformer_global
        self.other_context_size = (
            (n_cars - 1) * (CAR_SIZE + 6) if _legacy_opponent_context else 0
        )
        self.car_encoder = nn.Sequential(
            nn.Linear(CAR_SIZE + 6 + self.other_context_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, frame_embedding), nn.ReLU(),
        )
        if _legacy_two_heads:
            self.ball_encoder = nn.Sequential(
                nn.Linear(BALL_SIZE + 6 + self.other_context_size, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, frame_embedding), nn.ReLU(),
            )
        self.car_gru = nn.GRU(frame_embedding, temporal_hidden, batch_first=True)
        if _legacy_two_heads:
            self.ball_gru = nn.GRU(frame_embedding, temporal_hidden, batch_first=True)
        self.car_head = nn.Linear(temporal_hidden, 1)
        if _legacy_two_heads:
            self.ball_head = nn.Linear(temporal_hidden, 1)
        self.near_discriminator = (
            None if _legacy_two_heads else EgoBallSceneDiscriminator(
                frame_embedding, temporal_hidden, hidden_size,
            )
        )
        self.global_discriminator = (
            None if _legacy_two_heads else CausalSceneTransformer(
                frame_embedding, temporal_hidden, hidden_size, max_context=context_length,
            ) if transformer_global else SceneDiscriminator(
                frame_embedding, temporal_hidden, hidden_size, n_cars=n_cars,
                recurrent_global=recurrent_global,
            )
        )

    def specialist_logits(self, windows: th.Tensor) -> th.Tensor:
        if windows.ndim != 3 or windows.shape[-1] != self.scene_size:
            raise ValueError("factorized discriminator needs [batch, frames, scene] windows")
        ball = windows[..., :BALL_SIZE]
        ego = windows[..., BLUE_START:BLUE_START + CAR_SIZE]
        relative_position = ball[..., :3] - ego[..., :3]
        relative_velocity = (
            ball[..., 3:6] - ego[..., 3:6] * (CAR_MAX_SPEED / BALL_MAX_SPEED)
        )
        # Hold context fixed so the car head cannot classify subsequent ball motion.
        initial_context = th.cat((relative_position[:, 0], relative_velocity[:, 0]), dim=-1)
        car_input = th.cat((
            ego, initial_context[:, None].expand(-1, windows.shape[1], -1),
        ), dim=-1)
        if self.near_discriminator is None and self.other_context_size:
            others = windows[:, 0, BLUE_START + CAR_SIZE:].reshape(
                -1, self.n_cars - 1, CAR_SIZE,
            )
            relative_others = others[..., :6] - ego[:, 0, None, :6]
            initial_others = th.cat((others, relative_others), dim=-1).flatten(1)
            context = initial_others[:, None].expand(-1, windows.shape[1], -1)
            car_input = th.cat((car_input, context), dim=-1)
        car_features, _ = self.car_gru(self.car_encoder(car_input))
        far_score = self.car_head(car_features[:, -1])
        if self.near_discriminator is not None:
            near_score = self.near_discriminator(windows).unsqueeze(-1)
        else:
            ball_input = th.cat((ball, relative_position, relative_velocity), dim=-1)
            if self.other_context_size:
                ball_input = th.cat((ball_input, context), dim=-1)
            ball_features, _ = self.ball_gru(self.ball_encoder(ball_input))
            near_score = self.ball_head(ball_features[:, -1])
        return th.cat((far_score, near_score), dim=-1)

    def forward(self, windows: th.Tensor, context: th.Tensor | None = None) -> th.Tensor:
        specialist = self.specialist_logits(windows)
        if self.global_discriminator is None:
            return specialist
        if context is not None and not (self.recurrent_global or self.transformer_global):
            raise ValueError("global context requires a causal discriminator")
        global_score = self.global_discriminator(
            windows if context is None else context
        ).unsqueeze(-1)
        return th.cat((specialist, global_score), dim=-1)


def load_discriminator_state(discriminator: nn.Module, state: dict[str, th.Tensor]) -> bool:
    """Keep the old car head as far; initialize near interaction and global heads."""
    if (not isinstance(discriminator, FactorizedSceneDiscriminator)
            or discriminator.near_discriminator is None
            or any(key.startswith(("near_discriminator.", "global_discriminator."))
                   for key in state)):
        discriminator.load_state_dict(state)
        return False

    weight = state["car_encoder.0.weight"]
    expected = discriminator.car_encoder[0].weight
    width = CAR_SIZE + 6
    if weight.shape[0] != expected.shape[0] or weight.shape[1] not in (
        width, width + (discriminator.n_cars - 1) * (CAR_SIZE + 6),
    ):
        raise ValueError("incompatible legacy discriminator car inputs")
    old_car = {key: value for key, value in state.items()
               if key.startswith(("car_encoder.", "car_gru.", "car_head."))}
    old_car["car_encoder.0.weight"] = weight[:, :width].clone()
    missing = discriminator.load_state_dict(old_car, strict=False)
    new_keys = {key for key in discriminator.state_dict()
                if key.startswith(("near_discriminator.", "global_discriminator."))}
    if set(missing.missing_keys) != new_keys or missing.unexpected_keys:
        raise ValueError("legacy discriminator has incompatible parameters")
    return True


def load_legacy_factorized_optimizer_state(
    optimizer: th.optim.Optimizer, discriminator: FactorizedSceneDiscriminator,
    saved: dict,
) -> None:
    """Keep far-head moments; discard old ball-head moments and initialize near/global."""
    current = optimizer.state_dict()
    if len(saved["param_groups"]) != 1 or len(current["param_groups"]) != 1:
        raise ValueError("legacy factorized discriminator needs a single optimizer group")
    previous = saved["param_groups"][0]
    active = current["param_groups"][0]
    car_encoder = [f"car_encoder.{name}" for name, _ in discriminator.car_encoder.named_parameters()]
    car_gru = [f"car_gru.{name}" for name, _ in discriminator.car_gru.named_parameters()]
    car_head = [f"car_head.{name}" for name, _ in discriminator.car_head.named_parameters()]
    legacy_names = (
        car_encoder + [name.replace("car_encoder.", "ball_encoder.") for name in car_encoder]
        + car_gru + [name.replace("car_gru.", "ball_gru.") for name in car_gru]
        + car_head + [name.replace("car_head.", "ball_head.") for name in car_head]
    )
    named = list(discriminator.named_parameters())
    if (len(previous["params"]) != len(legacy_names)
            or any(old is not new for old, (_, new) in zip(
                optimizer.param_groups[0]["params"], named,
            ))):
        raise ValueError("legacy factorized discriminator optimizer does not match model")
    old_ids = dict(zip(legacy_names, previous["params"]))
    new_ids = active["params"]
    state = {
        param_id: saved["state"][old_ids[name]]
        for (name, _), param_id in zip(named, new_ids)
        if name in old_ids and old_ids[name] in saved["state"]
    }
    optimizer.load_state_dict({
        "state": state,
        "param_groups": [{**previous, "params": new_ids}],
    })
    layer = discriminator.car_encoder[0]
    for name, value in optimizer.state[layer.weight].items():
        if isinstance(value, th.Tensor) and value.ndim == 2:
            width = layer.weight.shape[1]
            legacy_width = width + (discriminator.n_cars - 1) * (CAR_SIZE + 6)
            if value.shape not in ((layer.weight.shape[0], width),
                                   (layer.weight.shape[0], legacy_width)):
                raise ValueError(f"incompatible factorized discriminator optimizer {name}")
            optimizer.state[layer.weight][name] = value[:, :width].clone()


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
        context_length: int = 16,
    ) -> None:
        if microbatch_size < 1 or context_length < 1:
            raise ValueError("reset scoring batch and context length must be positive")
        if len(dataset) != len(expert.reset_indices) or dataset.device != expert.frames.device:
            raise ValueError("reset dataset must match the expert training frames")
        self.expert = expert
        self.dataset = dataset
        self.discriminator = discriminator
        self.microbatch_size = (
            bounded_context_batch_size(microbatch_size, context_length, 8)
            if getattr(discriminator, "transformer_global", False) else microbatch_size
        )
        self.context_length = context_length
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

    def _confidence(
        self, windows: th.Tensor, context: th.Tensor | None = None,
        ages: th.Tensor | None = None,
    ) -> th.Tensor:
        if (getattr(self.discriminator, "recurrent_global", False)
                or getattr(self.discriminator, "transformer_global", False)):
            if context is None or ages is None:
                raise ValueError("recurrent reset scoring needs causal expert context")
            factorized = getattr(self.discriminator, "factorized", False)
            global_model = (
                self.discriminator.global_discriminator if factorized
                else self.discriminator
            )
            global_logits = global_model.score_context(context, ages)
            logits = (
                th.cat((self.discriminator.specialist_logits(windows),
                        global_logits[:, None]), dim=-1)
                if factorized else global_logits
            )
        else:
            logits = self.discriminator(windows)
        if getattr(self.discriminator, "factorized", False):
            if logits.shape != (len(windows), 3):
                raise ValueError("factorized discriminator must return far, near, and global logits")
            near = nearest_ball_distance(windows) <= BALL_NEAR_DISTANCE
            specialist = th.where(near, logits[:, 1], logits[:, 0])
            logits = (GLOBAL_DISCRIMINATOR_WEIGHT * logits[:, 2]
                      + SPECIALIST_DISCRIMINATOR_WEIGHT * specialist)
        return th.sigmoid(-logits)

    def _score(self, starts: th.Tensor) -> th.Tensor:
        scores = []
        was_training = self.discriminator.training
        with th.no_grad():
            self.discriminator.eval()
            try:
                for chunk in starts.split(self.microbatch_size):
                    windows = self.expert.frames[chunk[:, None] + self.expert.window_offsets]
                    contextual = (getattr(self.discriminator, "recurrent_global", False)
                                  or getattr(self.discriminator, "transformer_global", False))
                    if contextual:
                        context, ages = self.expert.context_frames(
                            th.stack((chunk, th.zeros_like(chunk)), dim=-1),
                            self.context_length, return_age=True,
                        )
                    else:
                        context = ages = None
                    confidence = self._confidence(windows, context, ages)
                    paired = self.expert.opponent_pov_available[chunk]
                    if paired.any():
                        if contextual:
                            opponent_context, opponent_ages = self.expert.context_frames(
                                th.stack((chunk[paired], th.ones_like(chunk[paired])), dim=-1),
                                self.context_length, return_age=True,
                            )
                        else:
                            opponent_context = opponent_ages = None
                        confidence[paired] = th.maximum(
                            confidence[paired],
                            self._confidence(
                                opponent_view(windows[paired]),
                                opponent_context, opponent_ages,
                            ),
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


def balanced_proximity_weights(near: th.Tensor, target: th.Tensor) -> th.Tensor:
    """Balance agent/expert labels within each populated near/far band."""
    if near.ndim != 1 or target.shape != near.shape:
        raise ValueError("proximity labels must match the discriminator batch")
    agent = target.bool()
    masks = [((~near) & agent, (~near) & ~agent), (near & agent, near & ~agent)]
    populated = [index for index, (generated, expert) in enumerate(masks)
                 if generated.any() and expert.any()]
    weights = target.new_zeros((len(target), 2))
    for index in populated:
        for mask in masks[index]:
            weights[mask, index] = len(target) / (2 * len(populated) * mask.sum())
    return weights


class SceneDiscriminatorLoss:
    """BCE-with-logits loss for generated-vs-expert scene windows."""

    def __init__(
        self, discriminator: SceneDiscriminator | FactorizedSceneDiscriminator,
        *, specialists_only: bool = False,
    ) -> None:
        if specialists_only and not getattr(discriminator, "factorized", False):
            raise ValueError("specialist loss requires a factorized discriminator")
        self.discriminator = discriminator
        self.specialists_only = specialists_only

    def __call__(self, batch: TensorBatch) -> LossOutput:
        logit = (
            self.discriminator.specialist_logits(batch["window"])
            if self.specialists_only else self.discriminator(batch["window"])
        )
        target = batch["is_agent"]
        agent = target.bool()
        metrics = {}
        if getattr(self.discriminator, "factorized", False):
            expected = 2 if self.specialists_only else 3
            if logit.shape != (len(target), expected):
                raise ValueError("factorized discriminator returned the wrong number of logits")
            near = batch.get("ball_near")
            if near is None:
                near = nearest_ball_distance(batch["window"]) <= BALL_NEAR_DISTANCE
            weights = batch.get("band_weight")
            if weights is None:
                weights = balanced_proximity_weights(near, target)
            errors = F.binary_cross_entropy_with_logits(
                logit[:, :2], target[:, None].expand(-1, 2), reduction="none",
            )
            head_losses = (errors * weights).mean(dim=0)
            loss = SPECIALIST_DISCRIMINATOR_WEIGHT * head_losses.sum()
            metrics = {
                "far_loss": head_losses[0].detach(),
                "near_loss": head_losses[1].detach(),
                "near_ball_fraction": near.float().mean(),
            }
            specialist = th.where(near, logit[:, 1], logit[:, 0])
            if self.specialists_only:
                logit = specialist
            else:
                global_loss = F.binary_cross_entropy_with_logits(logit[:, 2], target)
                loss += GLOBAL_DISCRIMINATOR_WEIGHT * global_loss
                metrics["global_loss"] = global_loss.detach()
                logit = (GLOBAL_DISCRIMINATOR_WEIGHT * logit[:, 2]
                         + SPECIALIST_DISCRIMINATOR_WEIGHT * specialist)
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


class GlobalContextLoss:
    """Teach a global discriminator with contiguous, matched-length histories."""

    global_only = True

    def __init__(
        self, discriminator: SceneDiscriminator | CausalSceneTransformer | FactorizedSceneDiscriminator,
    ) -> None:
        if not (discriminator.recurrent_global
                or getattr(discriminator, "transformer_global", False)):
            raise ValueError("global context loss requires a causal discriminator")
        self.global_discriminator = (
            discriminator.global_discriminator
            if getattr(discriminator, "factorized", False) else discriminator
        )
        self.transformer_global = getattr(discriminator, "transformer_global", False)

    def __call__(self, batch: TensorBatch) -> LossOutput:
        logit = self.global_discriminator.score_context(
            batch["window"], batch["age"],
        )
        target = batch["is_agent"]
        loss = F.binary_cross_entropy_with_logits(logit, target)
        agent = target.bool()
        with th.no_grad():
            agent_score = th.sigmoid(logit[agent]).mean()
            expert_score = th.sigmoid(logit[~agent]).mean()
            agent_accuracy = (logit[agent] > 0).float().mean()
            expert_accuracy = (logit[~agent] <= 0).float().mean()
        return LossOutput(loss, {
            "loss": loss,
            "global_loss": loss.detach(),
            "agent_score": agent_score,
            "expert_score": expert_score,
            "agent_accuracy": agent_accuracy,
            "expert_accuracy": expert_accuracy,
            "context_steps": batch["age"].float().mean(),
        })


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


def simulation_episode_ends(episode_end: th.Tensor) -> th.Tensor:
    """A 1v1 simulation resets both focal viewpoints when either actor is done."""
    if episode_end.ndim != 2 or episode_end.shape[1] % N_CARS:
        raise ValueError("1v1 episode ends need two actors per simulation")
    return episode_end.bool().reshape(
        len(episode_end), -1, N_CARS,
    ).any(dim=-1).repeat_interleave(N_CARS, dim=-1)


class GeneratedContextTimeline:
    """Causal 1v1 actor histories, including a bounded prefix from the last rollout."""

    def __init__(
        self, windows: th.Tensor, n_envs: int, episode_end: th.Tensor | None,
        prefix_frames: th.Tensor | None = None, prefix_ends: th.Tensor | None = None,
    ) -> None:
        if (windows.ndim != 3 or windows.shape[-1] != SCENE_SIZE or n_envs < 1
                or n_envs % N_CARS or len(windows) % n_envs):
            raise ValueError("generated context needs time-major 1v1 scene windows")
        self.n_envs = n_envs
        self.steps = len(windows) // n_envs
        if episode_end is not None and episode_end.shape != (self.steps, n_envs):
            raise ValueError("generated context episode ends must match actors")
        if (prefix_frames is None) != (prefix_ends is None):
            raise ValueError("generated context needs both previous frames and episode ends")
        current = windows.reshape(self.steps, n_envs, windows.shape[1], SCENE_SIZE)[:, :, -1]
        endings = (simulation_episode_ends(episode_end) if episode_end is not None else
                   th.zeros((self.steps, n_envs), dtype=th.bool, device=windows.device))
        self.offset = 0
        if prefix_frames is not None:
            if (prefix_frames.ndim != 3 or prefix_frames.shape[1:] != (n_envs, SCENE_SIZE)
                    or prefix_ends.shape != prefix_frames.shape[:2]
                    or prefix_frames.device != windows.device
                    or prefix_ends.device != windows.device):
                raise ValueError("previous generated contexts must match the current actors")
            self.offset = len(prefix_frames)
            current = th.cat((prefix_frames, current))
            endings = th.cat((prefix_ends.bool(), endings))
        self.frames = current
        self.ends = endings
        beginnings = th.zeros(endings.shape, dtype=th.long, device=windows.device)
        beginnings[1:] = th.where(
            endings[:-1], th.arange(1, len(endings), device=windows.device)[:, None], 0,
        )
        self.beginnings = beginnings.cummax(dim=0).values

    def contexts(self, indices: th.Tensor, length: int) -> tuple[th.Tensor, th.Tensor]:
        if length < 1:
            raise ValueError("generated context length must be positive")
        if not len(indices):
            return self.frames.new_empty((0, length, SCENE_SIZE)), indices.new_empty(0)
        if (indices < 0).any() or (indices >= self.steps * self.n_envs).any():
            raise ValueError("generated context indices must be in the rollout")
        time, actor = indices // self.n_envs + self.offset, indices % self.n_envs
        beginning = self.beginnings[time, actor]
        offsets = th.arange(length - 1, -1, -1, device=self.frames.device)
        selected = th.maximum(time[:, None] - offsets, beginning[:, None])
        return (self.frames[selected, actor[:, None]],
                (time - beginning + 1).clamp(max=length))


def generated_context_frames(
    windows: th.Tensor, indices: th.Tensor, n_envs: int,
    episode_end: th.Tensor | None, length: int,
) -> tuple[th.Tensor, th.Tensor]:
    """Gather consecutive scenes without crossing actor or episode boundaries."""
    return GeneratedContextTimeline(windows, n_envs, episode_end).contexts(indices, length)


def bounded_context_batch_size(size: int, length: int, minimum: int) -> int:
    """Limit Transformer batches to a bounded number of scene frames."""
    return min(size, max(minimum, size * 16 // length))


class GlobalContextMinibatches:
    """Train a causal global judge on adjacent actor and expert POV frames."""

    def __init__(
        self, expert: ExpertSceneDataset, batch_size: int, epochs: int,
        noise_std: float, context_length: int, stride: int,
        history: HistoricalReplayBuffer | RecencyReplayBuffer | None = None,
        mix_fraction: float = 0.0,
        variable_length: bool = False,
    ) -> None:
        if min(batch_size, epochs, context_length, stride) < 1:
            raise ValueError("global context batch, epoch, length and stride must be positive")
        if not 0.0 <= mix_fraction < 1.0:
            raise ValueError("global context history mix must be in [0, 1)")
        self.expert = expert
        self.batch_size = (
            bounded_context_batch_size(batch_size, context_length, 128)
            if variable_length else batch_size
        )
        self.epochs = epochs
        self.noise_std = noise_std
        self.context_length = context_length
        self.stride = stride
        self.history = history
        self.mix_fraction = mix_fraction if history is not None else 0.0
        self.variable_length = variable_length
        self._epoch_callback = None

    def set_epoch_callback(self, callback) -> None:
        self._epoch_callback = callback

    def sample_contexts(
        self, windows: th.Tensor, indices: th.Tensor, n_envs: int,
        episode_end: th.Tensor | None = None,
        *, timeline: GeneratedContextTimeline | None = None,
    ):
        pools = self.expert.context_situation_pools()
        available = th.tensor([bool(len(pool)) for pool in pools], device=windows.device)
        for _ in range(self.epochs):
            shuffled = indices[th.randperm(len(indices), device=indices.device)]
            quota = max(1, math.ceil(len(shuffled) / self.stride))
            if self.variable_length:
                # A short rollout may otherwise sample only unmatched situations
                # and skip Transformer training altogether.
                eligible = []
                for chunk in shuffled.split(self.batch_size):
                    matched = available[scene_situation_ids(windows[chunk])]
                    if matched.any():
                        eligible.append(chunk[matched][:quota])
                        quota -= len(eligible[-1])
                        if quota == 0:
                            break
                selected = th.cat(eligible) if eligible else shuffled[:0]
            else:
                selected = shuffled[:quota]
            for batch_indices in selected.split(self.batch_size):
                count = len(batch_indices)
                n_history = 0
                if self.history is not None and self.history.size:
                    n_history = min(int(count * self.mix_fraction), self.history.size)
                current = batch_indices[:count - n_history]
                context, ages = (
                    timeline.contexts(current, self.context_length)
                    if timeline is not None else generated_context_frames(
                        windows, current, n_envs, episode_end, self.context_length,
                    )
                )
                labels = scene_situation_ids(windows[current])
                if n_history:
                    past = self.history.sample(n_history, windows.device)
                    context = th.cat((context, past))
                    ages = th.cat((ages, ages.new_full((n_history,), self.context_length)))
                    labels = th.cat((labels, scene_situation_ids(
                        past[:, -min(self.expert.trajectory_length, self.context_length):],
                    )))
                matched = available[labels]
                if not matched.any():
                    continue
                context, ages, labels = context[matched], ages[matched], labels[matched]
                if self.variable_length:
                    # Apply the same random cap to each expert and generated pair.
                    ages = th.minimum(ages, th.randint(
                        1, self.context_length + 1, (len(ages),), device=windows.device,
                    ))
                pairs = th.empty((len(context), 2), dtype=th.long, device=windows.device)
                for label in labels.unique().tolist():
                    positions = (labels == label).nonzero(as_tuple=True)[0]
                    pool = pools[label]
                    pairs[positions] = pool[th.randint(
                        len(pool), (len(positions),), device=windows.device,
                    )]
                expert, matched_ages = self.expert.context_frames(
                    pairs, self.context_length, max_age=ages, return_age=True,
                )
                n = len(context)
                yield TensorBatch({
                    "window": th.cat((add_scene_noise(context, self.noise_std),
                                      add_scene_noise(expert, self.noise_std))),
                    "is_agent": th.cat((context.new_ones(n), context.new_zeros(n))),
                    # Match both lengths: an expert split or unsafe gap must not
                    # become a shortcut for identifying the label.
                    "age": th.cat((matched_ages, matched_ages)),
                })
            if self._epoch_callback is not None:
                self._epoch_callback()


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

                expert_windows = self.expert._windows_for_povs(sampled_experts)
                if self.factorize:
                    ball_near = th.cat((
                        nearest_ball_distance(sampled_agents) <= BALL_NEAR_DISTANCE,
                        nearest_ball_distance(expert_windows) <= BALL_NEAR_DISTANCE,
                    ))
                agent_windows = add_scene_noise(sampled_agents, self.noise_std)
                expert_windows = add_scene_noise(expert_windows, self.noise_std)
                sample = TensorBatch({
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
                yield sample.with_fields(ball_near=ball_near) if self.factorize else sample
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

                expert_windows = self.expert.sample(sample_count, agent_windows.device)
                if selected_count:
                    expert_windows[:selected_count] = self.expert._windows_for_povs(expert_pairs)
                if self.factorize:
                    ball_near = th.cat((
                        nearest_ball_distance(agent_windows) <= BALL_NEAR_DISTANCE,
                        nearest_ball_distance(expert_windows) <= BALL_NEAR_DISTANCE,
                    ))
                agent_windows = add_scene_noise(agent_windows, self.noise_std)
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

                sample = TensorBatch(
                    {
                        "window": th.cat([agent_windows, expert_windows]),
                        "is_agent": is_agent,
                        "situation_matched": th.cat([matched_mask, matched_mask]),
                        "grounded_random": th.cat([grounded_mask, grounded_mask]),
                    }
                )
                yield sample.with_fields(ball_near=ball_near) if self.factorize else sample

            if self._epoch_callback is not None:
                self._epoch_callback()


def train_discriminator_minibatch(
    sample: TensorBatch,
    discriminator: SceneDiscriminator | FactorizedSceneDiscriminator,
    optimizer: th.optim.Optimizer,
    loss: SceneDiscriminatorLoss | GlobalContextLoss,
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
    band_weights = near = None
    if getattr(discriminator, "factorized", False) and not getattr(loss, "global_only", False):
        near = sample.get("ball_near")
        if near is None:
            near = nearest_ball_distance(windows) <= BALL_NEAR_DISTANCE
        band_weights = balanced_proximity_weights(near, labels)
        metrics["near_agent_fraction"] = near[:n_agent].float().mean()
        metrics["near_expert_fraction"] = near[n_agent:].float().mean()
    for start in range(0, n_agent, microbatch_size):
        stop = min(start + microbatch_size, n_agent)
        chunk = TensorBatch({
            "window": th.cat((windows[start:stop], windows[n_agent + start:n_agent + stop])),
            "is_agent": th.cat((labels[start:stop], labels[n_agent + start:n_agent + stop])),
        })
        if "age" in sample:
            chunk = chunk.with_fields(age=th.cat((
                sample["age"][start:stop], sample["age"][n_agent + start:n_agent + stop],
            )))
        if band_weights is not None:
            chunk = chunk.with_fields(
                ball_near=th.cat((near[start:stop], near[n_agent + start:n_agent + stop])),
                band_weight=th.cat((band_weights[start:stop],
                                    band_weights[n_agent + start:n_agent + stop])),
            )
        output = loss(chunk)
        fraction = (stop - start) / n_agent
        (output.loss * fraction).backward()
        for name, value in output.metrics.items():
            metrics[name] = metrics.get(name, 0.0) + value.detach() * fraction
        del output, chunk

    th.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
    optimizer.step()
    if getattr(loss, "global_only", False):
        discriminator.context_version = getattr(discriminator, "context_version", 0) + 1
    return metrics


class SceneDiscriminatorReward:
    """Combine imitation, goal, and physical touch rewards per actor.

    Short windows receive normalized expert log-odds by default, or capped
    expert-to-agent odds with the optional exponential reward. Differential
    mode rewards the change in expert log-odds or discount-adjusted capped
    expert odds as a scene frame is added. Physical bonuses are zero-sum in
    1v1; goal and touch transitions remain learnable before imitation windows
    are valid.
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
        context_length: int = 16,
        differential: bool = False,
        gamma: float = 0.99,
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
        if context_length < 1:
            raise ValueError("recurrent context length must be positive")
        if not math.isfinite(gamma) or not 0.0 < gamma <= 1.0:
            raise ValueError("reward gamma must be in (0, 1]")
        self.discriminator = discriminator
        self.factorize = getattr(discriminator, "factorized", False)
        self.recurrent_global = getattr(discriminator, "recurrent_global", False)
        self.transformer_global = getattr(discriminator, "transformer_global", False)
        self.noise_std = noise_std
        self.trajectory_length = trajectory_length
        self.goal_reward_weight = goal_reward_weight
        self.aerial_touch_reward_weight = aerial_touch_reward_weight
        self.flip_reset_reward_weight = flip_reset_reward_weight
        self.batch_size = batch_size
        self.max_magnitude = max_magnitude
        self.exp_log_odds_reward = exp_log_odds_reward
        self.differential = differential
        self.gamma = gamma
        self.context_length = context_length
        self._recent_frames: th.Tensor | None = None
        self._recent_ends: th.Tensor | None = None
        self._last_state: th.Tensor | None = None
        self._last_global_logits: th.Tensor | None = None
        self._context_version: int | None = None

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

    def _expert_odds(self, logits: th.Tensor) -> th.Tensor:
        # D = sigmoid(-logits), so expert-to-agent odds are exp(-logits).
        return (-logits).clamp(max=math.log(self.max_magnitude)).exp()

    def _score_change(self, current: th.Tensor, previous: th.Tensor) -> th.Tensor:
        if self.exp_log_odds_reward:
            previous_odds = self._expert_odds(previous)
            return self._expert_odds(current) - (
                self.gamma if self.differential else 1.0
            ) * previous_odds
        return previous - current

    def _score_global_sequence(
        self, windows: th.Tensor, valid: th.Tensor,
        episode_end: th.Tensor | None,
    ) -> th.Tensor:
        if episode_end is None or episode_end.shape != valid.shape:
            raise ValueError("recurrent discriminator needs episode ends for each actor")
        current_frames = windows[:, :, -1]
        # A window can be unscored while its scene still belongs to this episode.
        # Only a real termination clears recurrent memory.
        current_ends = simulation_episode_ends(episode_end)
        have_history = (self._recent_frames is not None
                        and self._recent_frames.shape[1:] == current_frames.shape[1:])
        version = getattr(self.discriminator, "context_version", 0)
        carry = (have_history and self._last_state is not None
                 and self._context_version == version
                 and (not self.differential or
                      (self._last_global_logits is not None
                       and self._last_global_logits.shape == (windows.shape[1],))))
        need_prefix = have_history and (not carry or len(current_frames) < self.context_length)
        history_frames = (
            th.cat((self._recent_frames, current_frames)) if need_prefix
            else current_frames
        )
        history_ends = (
            th.cat((self._recent_ends, current_ends)) if need_prefix
            else current_ends
        )
        frames = current_frames if carry else history_frames
        endings = current_ends if carry else history_ends
        resets = th.zeros_like(endings)
        resets[0] = self._recent_ends[-1] if carry else True
        resets[1:] = endings[:-1]
        self._recent_frames = history_frames[-self.context_length:].detach().clone()
        self._recent_ends = history_ends[-self.context_length:].detach().clone()
        global_model = (
            self.discriminator.global_discriminator if self.factorize
            else self.discriminator
        )
        scores = []
        states = []
        for start in range(0, windows.shape[1], self.batch_size):
            stop = min(start + self.batch_size, windows.shape[1])
            logits, state = global_model.score_sequence(
                add_scene_noise(frames[:, start:stop], self.noise_std),
                resets[:, start:stop],
                self._last_state[:, start:stop] if carry else None,
            )
            current = logits[-len(windows):]
            if self.differential:
                if (self._last_global_logits is None
                        or self._last_global_logits.shape != (windows.shape[1],)):
                    self._last_global_logits = current.new_empty(windows.shape[1])
                if carry:
                    previous = self._last_global_logits[start:stop]
                elif need_prefix:
                    # Re-score the cached history after discriminator updates;
                    # subtracting a logit from old weights would create reward.
                    previous = logits[-len(windows) - 1]
                else:
                    previous = current.new_zeros(stop - start)
                baseline = th.cat((previous[None], current[:-1]), dim=0)
                baseline = baseline.masked_fill(
                    resets[-len(windows):, start:stop], 0,
                )
                scores.append(self._score_change(current, baseline))
                self._last_global_logits[start:stop] = current[-1].detach()
            else:
                scores.append(current)
            states.append(state)
        self._last_state = th.cat(states, dim=1).detach()
        self._context_version = version
        return th.cat(scores, dim=1)

    def _score_transformer_global(
        self, windows: th.Tensor, valid: th.Tensor,
        episode_end: th.Tensor | None,
    ) -> th.Tensor:
        if episode_end is None or episode_end.shape != valid.shape:
            raise ValueError("Transformer reward needs episode ends for each actor")
        n_envs = valid.shape[1]
        previous = self._recent_frames
        previous_ends = self._recent_ends
        if previous is not None and previous.shape[1:] != (n_envs, SCENE_SIZE):
            previous = previous_ends = None
        flat = windows.reshape(-1, self.trajectory_length, SCENE_SIZE)
        timeline = GeneratedContextTimeline(
            flat, n_envs, episode_end, previous, previous_ends,
        )
        if self.context_length > 1:
            recent = slice(-(self.context_length - 1), None)
            self._recent_frames = timeline.frames[recent].detach().clone()
            self._recent_ends = timeline.ends[recent].detach().clone()
        result = flat.new_zeros(len(flat))
        indices = valid.flatten().nonzero(as_tuple=True)[0]
        global_model = (self.discriminator.global_discriminator if self.factorize
                        else self.discriminator)
        chunk_size = bounded_context_batch_size(self.batch_size, self.context_length, 8)
        for chunk in indices.split(chunk_size):
            context, ages = timeline.contexts(chunk, self.context_length)
            current, previous = global_model.score_context(
                add_scene_noise(context, self.noise_std), ages,
                return_previous=True,
            )
            # Both logits use the identical capped window. An old frame expiring
            # from the context cannot earn reward just by disappearing.
            result[chunk] = self._score_change(current, previous).clamp(
                -self.max_magnitude, self.max_magnitude,
            )
        return result.reshape_as(valid)

    def _score_windows(
        self,
        windows: th.Tensor,
        valid: th.Tensor,
        episode_end: th.Tensor | None = None,
    ) -> th.Tensor:
        scores = th.zeros(
            (*valid.shape, 3) if self.factorize else valid.shape,
            dtype=windows.dtype, device=windows.device,
        )
        global_scores = (
            self._score_global_sequence(windows, valid, episode_end).flatten()
            if self.recurrent_global else
            self._score_transformer_global(windows, valid, episode_end).flatten()
            if self.transformer_global else None
        )
        if not valid.any():
            return scores

        if self.transformer_global and not self.factorize:
            scores.flatten()[valid.flatten()] = global_scores[valid.flatten()]
            return scores

        flat_windows = windows.reshape(-1, self.trajectory_length, SCENE_SIZE)
        flat_scores = scores.reshape(-1, 3) if self.factorize else scores.flatten()
        indices = th.nonzero(valid.flatten(), as_tuple=False).squeeze(-1)
        near = (nearest_ball_distance(flat_windows[indices]) <= BALL_NEAR_DISTANCE
                if self.factorize else None)
        selected_scores = th.empty(
            (len(indices), 3) if self.factorize else (len(indices),),
            dtype=scores.dtype, device=scores.device,
        )
        for start in range(0, len(indices), self.batch_size):
            stop = min(start + self.batch_size, len(indices))
            if self.differential:
                if self.recurrent_global and not self.factorize:
                    delta = global_scores[indices[start:stop]]
                else:
                    noisy = add_scene_noise(
                        flat_windows[indices[start:stop]], self.noise_std
                    )
                    if self.recurrent_global or self.transformer_global:
                        current = self.discriminator.specialist_logits(noisy)
                        previous = self.discriminator.specialist_logits(noisy[:, :-1])
                        global_delta = global_scores[indices[start:stop]]
                        delta = th.cat((
                            self._score_change(current, previous), global_delta[:, None],
                        ), dim=-1)
                    else:
                        previous = self.discriminator(noisy[:, :-1])
                        current = self.discriminator(noisy)
                        delta = self._score_change(current, previous)
                expected = (stop - start, 3) if self.factorize else (stop - start,)
                if delta.shape != expected:
                    raise ValueError(f"discriminator returned {tuple(delta.shape)}, expected {expected}")
                selected_scores[start:stop] = delta.clamp(
                    -self.max_magnitude, self.max_magnitude,
                )
                continue
            if self.recurrent_global and not self.factorize:
                logits = global_scores[indices[start:stop]]
            else:
                noisy = add_scene_noise(
                    flat_windows[indices[start:stop]], self.noise_std
                )
                logits = (
                    self.discriminator.specialist_logits(noisy)
                    if self.recurrent_global or self.transformer_global
                    else self.discriminator(noisy)
                )
                if self.recurrent_global:
                    logits = th.cat((
                        logits, global_scores[indices[start:stop], None],
                    ), dim=-1)
                elif self.transformer_global:
                    logits = th.cat((logits, th.zeros_like(logits[:, :1])), dim=-1)
            expected = (stop - start, 3) if self.factorize else (stop - start,)
            if logits.shape != expected:
                raise ValueError(f"discriminator returned {tuple(logits.shape)}, expected {expected}")
            if self.exp_log_odds_reward:
                selected_scores[start:stop] = self._expert_odds(logits)
            else:
                selected_scores[start:stop] = (-logits).clamp(
                    -self.max_magnitude, self.max_magnitude
                )
            if self.transformer_global:
                selected_scores[start:stop, 2] = global_scores[indices[start:stop]]

        if self.factorize:
            # Normalize the specialist scores within their own proximity bands.
            # The global score is normalized across every valid window.
            gated = th.zeros_like(selected_scores)
            for head, active in enumerate((~near, near, th.ones_like(near))):
                values = selected_scores[active, head]
                if self.differential or (head == 2 and self.transformer_global):
                    gated[active, head] = values
                elif self.exp_log_odds_reward:
                    gated[active, head] = values
                elif len(values):
                    std = values.std(unbiased=False)
                    if std > 1e-8:
                        gated[active, head] = ((values - values.mean()) / std).clamp(
                            -self.max_magnitude, self.max_magnitude,
                        )
            flat_scores[indices] = gated
        elif self.differential or self.exp_log_odds_reward:
            flat_scores[indices] = selected_scores
        else:
            std = selected_scores.std(unbiased=False)
            if std > 1e-8:
                flat_scores[indices] = ((selected_scores - selected_scores.mean()) / std).clamp(
                    -self.max_magnitude, self.max_magnitude,
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
        terminal = batch.get("terminated")
        truncated = batch.get("truncated")
        if truncated is not None:
            terminal = truncated.bool() if terminal is None else terminal.bool() | truncated.bool()
        scores = self._score_windows(windows, valid, terminal).to(dtype)
        components = {}
        if self.factorize:
            far_reward = SPECIALIST_DISCRIMINATOR_WEIGHT * scores[..., 0]
            near_reward = SPECIALIST_DISCRIMINATOR_WEIGHT * scores[..., 1]
            global_reward = GLOBAL_DISCRIMINATOR_WEIGHT * scores[..., 2]
            imitation_reward = far_reward + near_reward + global_reward
            components = {
                "far_imitation_reward": far_reward,
                "near_imitation_reward": near_reward,
                "global_imitation_reward": global_reward,
                "ball_near": (nearest_ball_distance(windows) <= BALL_NEAR_DISTANCE) & valid,
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
        context_length: int = 16,
        context_stride: int = 4,
        context_history: HistoricalReplayBuffer | RecencyReplayBuffer | None = None,
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
        if min(context_length, context_stride) < 1:
            raise ValueError("discriminator context length and stride must be positive")
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
        self.recurrent_global = getattr(discriminator, "recurrent_global", False)
        self.transformer_global = getattr(discriminator, "transformer_global", False)
        self.contextual_global = self.recurrent_global or self.transformer_global
        self.context_length = context_length
        self.context_stride = context_stride
        self.context_microbatch_size = (
            bounded_context_batch_size(microbatch_size, context_length, 8)
            if self.transformer_global else microbatch_size
        )
        self.context_heldout_size = (
            min(bounded_context_batch_size(heldout_size, context_length, 32),
                self.context_microbatch_size)
            if self.transformer_global else microbatch_size
        )
        self.context_history = context_history
        self._recent_context_frames: th.Tensor | None = None
        self._recent_context_ends: th.Tensor | None = None
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
        if self.transformer_global and terminal is None:
            raise ValueError("Transformer training needs episode ends")
        if terminal is not None and self.contextual_global:
            terminal = simulation_episode_ends(terminal)
        context_timeline = None
        if self.transformer_global:
            if (self._recent_context_frames is not None and
                    self._recent_context_frames.shape[1:] != (valid.shape[1], SCENE_SIZE)):
                self._recent_context_frames = self._recent_context_ends = None
            context_timeline = GeneratedContextTimeline(
                flat_windows, valid.shape[1], terminal,
                self._recent_context_frames, self._recent_context_ends,
            )
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
        validation = self._heldout_pairs(
            heldout_generated, heldout_near,
            flat_windows=flat_windows if self.contextual_global else None,
            heldout_indices=heldout_indices if self.contextual_global else None,
            n_envs=valid.shape[1], episode_end=terminal, timeline=context_timeline,
        )
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
            sampled_windows = (
                sampler.sample_windows(
                    flat_windows, train_indices, n_envs=valid.shape[1],
                    episode_end=terminal,
                    archived_flights=(self.maneuver_tracker.ready
                                      if self.maneuver_tracker is not None else None),
                    ego_ball_touch=touch_events, goal_scored=goal_scored,
                    timeline=timeline,
                ) if not self.contextual_global or getattr(self.discriminator, "factorized", False)
                else ()
            )
            if self.contextual_global:
                context_sampler = GlobalContextMinibatches(
                    self.expert, self.batch_size, self.epochs, self.noise_std,
                    self.context_length, self.context_stride,
                    history=self.context_history, mix_fraction=self.history_mix_fraction,
                    variable_length=self.transformer_global,
                )
                if not getattr(self.discriminator, "factorized", False):
                    context_sampler.set_epoch_callback(self._epoch_finished)
                specialist_loss = (
                    SceneDiscriminatorLoss(self.discriminator, specialists_only=True)
                    if getattr(self.discriminator, "factorized", False) else None
                )
                context_loss = GlobalContextLoss(self.discriminator)
                batches = (
                    tuple((sample, selected_loss) for sample, selected_loss in (
                        (specialist, specialist_loss), (contextual, context_loss),
                    ) if sample is not None)
                    for specialist, contextual in zip_longest(
                        sampled_windows,
                        context_sampler.sample_contexts(
                            flat_windows, train_indices, valid.shape[1], terminal,
                            timeline=context_timeline,
                        ),
                    )
                )
            else:
                batches = (((sample, self.loss),) for sample in sampled_windows)
            metric_totals: dict[str, float | th.Tensor] = {}
            metric_counts: dict[str, int] = {}
            minibatch_count = 0
            evaluated_last_batch = False
            callback = self._progress_callback
            if callback is not None:
                callback.start(self.epochs, self.section)
            try:
                for group_index, group in enumerate(batches, 1):
                    for sample, selected_loss in group:
                        minibatch_metrics = train_discriminator_minibatch(
                            sample, self.discriminator, self.optimizer, selected_loss,
                            self.context_microbatch_size
                            if getattr(selected_loss, "global_only", False)
                            else self.microbatch_size,
                            self.max_grad_norm,
                        )
                        for key, value in minibatch_metrics.items():
                            metric_totals[key] = metric_totals.get(key, 0.0) + value
                            metric_counts[key] = metric_counts.get(key, 0) + 1
                        minibatch_count += 1
                    if self.contextual_global and group_index != 1 and group_index % 4:
                        evaluated_last_batch = False
                        continue
                    evaluation = self._evaluate(
                        heldout_generated, heldout_near, validation=validation,
                    )
                    evaluated_last_batch = True
                    if evaluation["heldout_accuracy"] >= self.accuracy_target:
                        metrics["updated"] = 1.0
                        break
                else:
                    if minibatch_count > 0:
                        metrics["updated"] = 1.0
                if self.contextual_global and minibatch_count and not evaluated_last_batch:
                    evaluation = self._evaluate(
                        heldout_generated, heldout_near, validation=validation,
                    )
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
                    averaged = total / metric_counts[key]
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
        if self.context_history is not None:
            add_count = (
                min(max(1, self.history_add_size // self.context_stride), len(train_indices))
                if self.history_add_size else 0
            )
            if add_count:
                selected = train_indices[
                    th.randperm(len(train_indices), device=train_indices.device)[:add_count]
                ]
                context, ages = (
                    context_timeline.contexts(selected, self.context_length)
                    if context_timeline is not None else generated_context_frames(
                        flat_windows, selected, valid.shape[1], terminal, self.context_length,
                    )
                )
                complete = ages == self.context_length
                self.context_history.add(context[complete], add_count)

        if context_timeline is not None and self.context_length > 1:
            recent = slice(-(self.context_length - 1), None)
            self._recent_context_frames = context_timeline.frames[recent].detach().clone()
            self._recent_context_ends = context_timeline.ends[recent].detach().clone()

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
        *, flat_windows: th.Tensor | None = None,
        heldout_indices: th.Tensor | None = None, n_envs: int = 1,
        episode_end: th.Tensor | None = None,
        timeline: GeneratedContextTimeline | None = None,
    ) -> tuple | None:
        n = min(len(heldout_generated), self.expert.heldout_total, self.heldout_size)
        if not n:
            return None
        device = heldout_generated.device
        selected = th.randperm(len(heldout_generated), device=device)[:n]
        generated = heldout_generated[selected]
        if self.contextual_global:
            if flat_windows is None or heldout_indices is None:
                raise ValueError("causal heldout evaluation needs chronological windows")
            generated_context, generated_ages = (
                timeline.contexts(heldout_indices[selected], self.context_length)
                if timeline is not None else generated_context_frames(
                    flat_windows, heldout_indices[selected], n_envs,
                    episode_end, self.context_length,
                )
            )
            expert_pairs = self.expert.sample_povs(n, heldout=True)
            expert = self.expert._windows_for_povs(expert_pairs)
            expert_context, ages = self.expert.context_frames(
                expert_pairs, self.context_length, heldout=True,
                max_age=generated_ages, return_age=True,
            )
        else:
            expert = self.expert.sample_heldout(n, device)
        near_generated = near_expert = None
        if (getattr(self.discriminator, "factorized", False)
                and heldout_near is not None and len(heldout_near)):
            n_near = min(len(heldout_near), self.heldout_size)
            near_generated = heldout_near[
                th.randperm(len(heldout_near), device=device)[:n_near]
            ]
            near_expert = self.expert.sample_near(n_near, device, heldout=True)
        if self.contextual_global:
            return (
                generated, expert, near_generated, near_expert,
                generated_context, expert_context, ages,
            )
        return generated, expert, near_generated, near_expert

    def _evaluate(
        self, heldout_generated: th.Tensor, heldout_near: th.Tensor | None = None,
        *, validation: tuple | None = None,
    ) -> dict[str, float]:
        if validation is None:
            validation = self._heldout_pairs(heldout_generated, heldout_near)
        factorize = getattr(self.discriminator, "factorized", False)
        head_names = ("far", "near", "global") if factorize else ("unified",)
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

        generated, expert, near_generated, near_expert = validation[:4]
        if self.contextual_global:
            generated_context, expert_context, ages = validation[4:]
        n = len(generated)
        totals = th.zeros(5, device=generated.device)
        head_counts = th.zeros(2, len(head_names), device=generated.device)
        head_correct = th.zeros_like(head_counts)

        was_training = self.discriminator.training
        with th.inference_mode():
            self.discriminator.eval()
            try:
                for start in range(0, n, self.context_heldout_size):
                    stop = min(start + self.context_heldout_size, n)
                    raw = th.cat((generated[start:stop], expert[start:stop]))
                    if self.contextual_global:
                        context = add_scene_noise(th.cat((
                            generated_context[start:stop], expert_context[start:stop],
                        )), self.noise_std)
                        current_ages = th.cat((ages[start:stop], ages[start:stop]))
                        global_model = (
                            self.discriminator.global_discriminator if factorize
                            else self.discriminator
                        )
                        global_logits = global_model.score_context(context, current_ages)
                        logits = (
                            th.cat((self.discriminator.specialist_logits(
                                add_scene_noise(raw, self.noise_std)
                            ), global_logits[:, None]), dim=-1)
                            if factorize else global_logits
                        )
                    else:
                        logits = self.discriminator(add_scene_noise(raw, self.noise_std))
                    generated_logits, expert_logits = logits.split(stop - start)
                    if factorize:
                        near = nearest_ball_distance(raw) <= BALL_NEAR_DISTANCE
                        generated_near, expert_near = near.split(stop - start)
                        generated_masks = th.stack((
                            ~generated_near, generated_near,
                            th.ones_like(generated_near),
                        ), dim=-1)
                        expert_masks = th.stack((
                            ~expert_near, expert_near, th.ones_like(expert_near),
                        ), dim=-1)
                        generated_selected = (
                            GLOBAL_DISCRIMINATOR_WEIGHT * generated_logits[:, 2]
                            + SPECIALIST_DISCRIMINATOR_WEIGHT * th.where(
                                generated_near, generated_logits[:, 1], generated_logits[:, 0],
                            )
                        )
                        expert_selected = (
                            GLOBAL_DISCRIMINATOR_WEIGHT * expert_logits[:, 2]
                            + SPECIALIST_DISCRIMINATOR_WEIGHT * th.where(
                                expert_near, expert_logits[:, 1], expert_logits[:, 0],
                            )
                        )
                    else:
                        generated_masks = th.ones(stop - start, 1, device=generated.device)
                        expert_masks = th.ones_like(generated_masks)
                        generated_logits = generated_logits[:, None]
                        expert_logits = expert_logits[:, None]
                        generated_selected = generated_logits[:, 0]
                        expert_selected = expert_logits[:, 0]
                    head_counts[0] += generated_masks.sum(dim=0)
                    head_counts[1] += expert_masks.sum(dim=0)
                    head_correct[0] += ((generated_logits > 0) * generated_masks).sum(dim=0)
                    head_correct[1] += ((expert_logits <= 0) * expert_masks).sum(dim=0)
                    totals[0] += F.softplus(-generated_selected).sum()
                    totals[0] += F.softplus(expert_selected).sum()
                    totals[1] += th.sigmoid(generated_selected).sum()
                    totals[2] += th.sigmoid(expert_selected).sum()
                    totals[3] += (generated_selected > 0).sum()
                    totals[4] += (expert_selected <= 0).sum()
                near_accuracy = None
                if factorize and near_generated is not None:
                    n_near = len(near_generated)
                    near_correct = th.zeros(2, device=generated.device)
                    for start in range(0, n_near, self.microbatch_size):
                        stop = min(start + self.microbatch_size, n_near)
                        noisy = add_scene_noise(
                            th.cat((near_generated[start:stop], near_expert[start:stop])),
                            self.noise_std,
                        )
                        logits = (
                            self.discriminator.specialist_logits(noisy)[:, 1]
                            if self.contextual_global else self.discriminator(noisy)[:, 1]
                        )
                        generated_logit, expert_logit = logits.split(stop - start)
                        near_correct[0] += (generated_logit > 0).sum()
                        near_correct[1] += (expert_logit <= 0).sum()
                    near_accuracy = near_correct.sum().item() / (2 * n_near)
            finally:
                self.discriminator.train(was_training)
        loss, agent_score, expert_score, agent_correct, expert_correct = totals.tolist()
        head_accuracies = {}
        measured = []
        for index, name in enumerate(head_names):
            if bool((head_counts[:, index] > 0).all()):
                accuracy = (head_correct[:, index] / head_counts[:, index]).mean().item()
                measured.append(accuracy)
            else:
                accuracy = 0.0
            head_accuracies[name] = accuracy
        metrics = {
            "loss": loss / (2 * n),
            "agent_score": agent_score / n,
            "expert_score": expert_score / n,
            "agent_accuracy": agent_correct / n,
            "expert_accuracy": expert_correct / n,
            "heldout_accuracy": min(measured),
        }
        if factorize:
            metrics.update({
                f"{name}_heldout_accuracy": head_accuracies[name]
                for name in head_names
            })
            if near_accuracy is not None:
                metrics["near_heldout_accuracy"] = near_accuracy
                metrics["heldout_accuracy"] = min(metrics["heldout_accuracy"], near_accuracy)
        return metrics

    def _epoch_finished(self) -> None:
        if self._progress_callback is not None:
            self._progress_callback.epoch_finished()


class GAIFOCheckpoints:
    """Checkpoint the policy, critic, discriminator and optimizer state."""

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
    ) -> None:
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
                "architecture": GAIFO_GRU_ARCHITECTURE if self.args.gru else GAIFO_ARCHITECTURE,
                **{
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(self.args).items()
                },
            },
        }
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


def load_resume_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"GAIFO checkpoint not found: {path}")
    payload = th.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"invalid GAIFO checkpoint: {path}")
    config = payload["config"]
    architecture = config.get("architecture")
    if architecture not in (GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE):
        raise ValueError(f"incompatible GAIFO architecture in {path}")
    if config.get("gru", False) != (architecture == GAIFO_GRU_ARCHITECTURE):
        raise ValueError(f"checkpoint GRU setting does not match architecture in {path}")
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
    # Older dual-timescale checkpoints may also include long-discriminator state.
    # Its short discriminator and optimizer remain compatible with this trainer.
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"checkpoint is missing {', '.join(missing)}: {path}")
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
    if getattr(args, "expired_dodge_mask", config.get("expired_dodge_mask", False)) != (
        config.get("expired_dodge_mask", False)
    ):
        raise ValueError("--expired-dodge-mask must match the checkpoint when resuming")
    if args.transformer_global != config.get("transformer_global", False):
        raise ValueError("--transformer must match the checkpoint discriminator when resuming")
    if args.recurrent_global != config.get("recurrent_global", False):
        raise ValueError("--recurrent-global must match the checkpoint discriminator when resuming")
    for name in (
        "frameskip", "trajectory_length", "policy_hidden", "critic_hidden",
        "policy_layers", "critic_layers",
        "discriminator_hidden", "frame_embedding", "temporal_hidden",
    ) + (("discriminator_context_length", "discriminator_context_stride")
         if args.recurrent_global or args.transformer_global else ()):
        saved = (
            config.get(name, 1) if name in ("policy_layers", "critic_layers")
            else config.get(name)
        )
        if getattr(args, name) != saved:
            raise ValueError(
                f"--{name.replace('_', '-')} must match the checkpoint "
                f"({saved}) when resuming"
            )


def restore_training_checkpoint(
    payload: dict,
    args: argparse.Namespace,
    modules: dict[str, nn.Module],
    optimizers: dict[str, th.optim.Optimizer],
) -> Clock:
    upgrade_discriminator = False
    for name, module in modules.items():
        if name == "discriminator":
            upgrade_discriminator = load_discriminator_state(module, payload[name])
        else:
            module.load_state_dict(payload[name])
    for name, optimizer in optimizers.items():
        if name == "discriminator" and upgrade_discriminator:
            load_legacy_factorized_optimizer_state(
                optimizer, modules[name], payload[f"{name}_optimizer"],
            )
        else:
            optimizer.load_state_dict(payload[f"{name}_optimizer"])
        learning_rate = args.discriminator_lr if "discriminator" in name else args.ppo_lr
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    if upgrade_discriminator:
        print("Restored far-car discriminator; initialized near-ball and global discriminators")

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
    parser.add_argument(
        "--n-sim", type=int, default=None,
        help="parallel simulations (default: 16384 for GRU, 256 for Transformer)",
    )
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument(
        "--expired-dodge-mask", action=argparse.BooleanOptionalAction, default=True,
        help="track CARL's dodge window in policy observations and mask expired airborne jumps",
    )
    parser.add_argument("--max-ticks", type=int, default=1_000_000)
    parser.add_argument("--no-touch-timeout", type=float, default=30.0)
    parser.add_argument("--rollout", type=int, default=32)
    parser.add_argument(
        "--trajectory-length", type=int, default=8,
        help="frames in the short discriminator scene window",
    )
    parser.add_argument(
        "--recurrent-global", action=argparse.BooleanOptionalAction, default=False,
        help="carry the always-on discriminator's GRU memory across scene windows (default: off)",
    )
    parser.add_argument(
        "--transformer", "--transformer-global", dest="transformer_global",
        action=argparse.BooleanOptionalAction, default=False,
        help="use a capped causal Transformer for the 1v1 global discriminator (default: GRU)",
    )
    parser.add_argument(
        "--discriminator-context-length", type=int, default=None,
        help="consecutive global context frames (default: 16 for GRU, 128 for Transformer)",
    )
    parser.add_argument(
        "--discriminator-context-stride", type=int, default=None,
        help="sample one global context endpoint per this many steps (default: 4 for GRU, 16 for Transformer)",
    )
    parser.add_argument(
        "--factorize", action=argparse.BooleanOptionalAction, default=False,
        help="train an always-on global scene discriminator and proximity-gated far-car and near-car/ball specialists",
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
        "--differential", action=argparse.BooleanOptionalAction, default=False,
        help="reward expert log-odds changes, or current capped expert odds minus --gamma times previous capped odds with --exp-log-odds-reward (Transformer global reward is already differential)",
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
        "--discriminator-batch", type=int, default=None,
        help="agent windows per optimizer step (default: 16384 for GRU, 2048 for Transformer)",
    )
    parser.add_argument(
        "--discriminator-microbatch", type=int, default=1_024,
        help="agent windows per GRU chunk; accumulates one discriminator optimizer step per effective batch",
    )
    parser.add_argument("--discriminator-epochs", type=int, default=1)
    parser.add_argument("--discriminator-update-interval", type=int, default=4)
    parser.add_argument("--discriminator-lr", type=float, default=3e-4)
    parser.add_argument(
        "--discriminator-lr-end", type=float, default=None,
        help="linearly anneal the discriminator learning rate over --timesteps (default: constant)",
    )
    parser.add_argument("--discriminator-hidden", type=int, default=128)
    parser.add_argument(
        "--discriminator-heldout-size", type=int, default=None,
        help="held-out windows (default: 16384 for GRU, 512 for Transformer)",
    )
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
        "--sequence-length", type=int, default=16,
        help="steps per recurrent PPO training sequence when --gru is enabled",
    )
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument(
        "--ppo-lr-end", type=float, default=None,
        help="linearly anneal policy and critic learning rates over --timesteps (default: constant)",
    )
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
    parser.add_argument("--policy-hidden", type=int, default=320)
    parser.add_argument("--critic-hidden", type=int, default=320)
    parser.add_argument(
        "--policy-layers", type=int, default=2,
        help="policy hidden layers after the encoder (extra layers follow the GRU)",
    )
    parser.add_argument(
        "--critic-layers", type=int, default=2,
        help="critic hidden layers after the encoder (extra layers follow the GRU)",
    )
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
        # Checkpoints predating configurable depth used one hidden layer.
        inherited.setdefault("policy_layers", 1)
        inherited.setdefault("critic_layers", 1)
        inherited.setdefault("expired_dodge_mask", False)
        inherited.setdefault("recurrent_global", False)
        inherited.setdefault("transformer_global", False)
        parser.set_defaults(**inherited)
    args = parser.parse_args()
    if args.n_sim is None:
        args.n_sim = 256 if args.transformer_global else 16_384
    if args.discriminator_batch is None:
        args.discriminator_batch = 2_048 if args.transformer_global else 16_384
    if args.discriminator_heldout_size is None:
        args.discriminator_heldout_size = 512 if args.transformer_global else 16_384
    if args.discriminator_context_length is None:
        args.discriminator_context_length = 128 if args.transformer_global else 16
    if args.discriminator_context_stride is None:
        args.discriminator_context_stride = 16 if args.transformer_global else 4
    if args.replay_reset_fraction is None:
        default = 1.0 if args.curated_skill_sampling else 0.70
        if (resume is not None and args.curated_skill_sampling ==
                resume["config"].get("curated_skill_sampling", False)):
            default = resume["config"].get("replay_reset_fraction", default)
        args.replay_reset_fraction = default
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

    if args.transformer_global and args.recurrent_global:
        raise ValueError("--transformer and --recurrent-global are mutually exclusive")
    if args.transformer_global and args.discriminator_context_length < 2:
        raise ValueError("--discriminator-context-length must be at least two with --transformer")
    if args.transformer_global and args.temporal_hidden % 4:
        raise ValueError("--temporal-hidden must be divisible by four with --transformer")

    positive = (
        "n_sim",
        "frameskip",
        "max_ticks",
        "rollout",
        "trajectory_length",
        "discriminator_context_length",
        "discriminator_context_stride",
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
        "policy_layers",
        "critic_layers",
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
    for name in ("ppo_lr", "discriminator_lr"):
        end = getattr(args, f"{name}_end")
        if end is not None and (not math.isfinite(end) or not 0.0 <= end <= getattr(args, name)):
            raise ValueError(
                f"--{name.replace('_', '-')}-end must be finite and between zero and --{name.replace('_', '-')}"
            )
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
    if args.gru:
        if args.rollout % args.sequence_length:
            raise ValueError("--rollout must be divisible by --sequence-length")
        if args.ppo_batch % args.sequence_length:
            raise ValueError("--ppo-batch must be divisible by --sequence-length")


def build_env(args: argparse.Namespace) -> CARLTorchVectorEnv:
    env_type = (DodgeAwareCARLTorchVectorEnv if getattr(args, "expired_dodge_mask", True)
                else CARLTorchVectorEnv)
    return env_type(
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
    layers = getattr(args, "policy_layers", 1)
    return MultiCategoricalPolicy(
        foot=LinearEncoder(args.policy_hidden, func=nn.ReLU),
        body=(
            GRU(hidden_size=args.policy_hidden) if args.gru
            else MLP(dims=[args.policy_hidden] * layers, func=nn.ReLU)
        ),
        head=MLP(
            dims=[args.policy_hidden] * (layers - 1) if args.gru else [],
            out_init_func=orthogonal_init(std=0.01),
        ),
        action_codec=env.action_codec,
    ).build(env).to(env.device)


def build_critic(env, args: argparse.Namespace) -> Critic:
    layers = getattr(args, "critic_layers", 1)
    return Critic(
        foot=LinearEncoder(args.critic_hidden, func=nn.ReLU),
        body=(
            GRU(hidden_size=args.critic_hidden) if args.gru
            else MLP(dims=[args.critic_hidden] * layers, func=nn.ReLU)
        ),
        head=MLP(
            dims=[args.critic_hidden] * (layers - 1) if args.gru else [],
            out_init_func=orthogonal_init(std=1.0),
        ),
    ).build(env).to(env.device)


def build_discriminator(
    args: argparse.Namespace,
) -> SceneDiscriminator | CausalSceneTransformer | FactorizedSceneDiscriminator:
    if getattr(args, "transformer_global", False) and not args.factorize:
        return CausalSceneTransformer(
            args.frame_embedding, args.temporal_hidden, args.discriminator_hidden,
            max_context=args.discriminator_context_length,
        )
    model = FactorizedSceneDiscriminator if args.factorize else SceneDiscriminator
    options = dict(
        frame_embedding=args.frame_embedding,
        temporal_hidden=args.temporal_hidden,
        hidden_size=args.discriminator_hidden,
        recurrent_global=getattr(args, "recurrent_global", False),
    )
    if args.factorize:
        options.update(
            transformer_global=getattr(args, "transformer_global", False),
            context_length=getattr(args, "discriminator_context_length", 16),
        )
    return model(**options)


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
        if getattr(args, "factorize", False):
            captures.append(EgoBallTouchCapture(gameplay))
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


def build_training_scheduler(
    args: argparse.Namespace, ppo_loss: PPOLoss,
    policy_optimizer: th.optim.Optimizer, critic_optimizer: th.optim.Optimizer,
    discriminator_optimizer: th.optim.Optimizer,
) -> ValueScheduler | None:
    entropy_scheduler = build_entropy_scheduler(args, ppo_loss)
    values = list(entropy_scheduler.values) if entropy_scheduler is not None else []

    def set_learning_rate(value: float, *optimizers: th.optim.Optimizer) -> None:
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = value

    if args.ppo_lr_end is not None:
        values.append(ScheduledValue(
            "ppo_lr", LinearSchedule(args.ppo_lr, args.ppo_lr_end),
            lambda value: set_learning_rate(value, policy_optimizer, critic_optimizer),
        ))
    if args.discriminator_lr_end is not None:
        values.append(ScheduledValue(
            "discriminator_lr",
            LinearSchedule(args.discriminator_lr, args.discriminator_lr_end),
            lambda value: set_learning_rate(value, discriminator_optimizer),
        ))
    return ValueScheduler(*values) if values else None


def main() -> None:
    args, resume = parse_args()
    validate_resume_args(args, resume)
    validate_args(args)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = build_env(args)
    gameplay = env.register_reward(GameplayDiagnostics(
        env.n_sim,
        env.device,
        math.ceil(args.no_touch_timeout * 120.0 / args.frameskip),
    ))
    policy = build_policy(env, args)
    critic = build_critic(env, args)
    discriminator = build_discriminator(args).to(env.device)

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
            context_length=args.discriminator_context_length,
        ) if args.hard_positive_mining else None
    )
    env.reset_state_provider = ReplayResetProvider(
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
    ) if not (args.recurrent_global or args.transformer_global) or args.factorize else None
    context_history = None
    if args.recurrent_global or args.transformer_global:
        context_options = dict(
            capacity=max(
                2 if args.recency_replay else 1,
                round(args.history_capacity * args.trajectory_length
                      / args.discriminator_context_length),
            ),
            trajectory_length=args.discriminator_context_length,
            device=env.device, seed=args.seed,
        )
        context_history = (
            RecencyReplayBuffer(
                **context_options, reservoir_fraction=args.history_reservoir_fraction,
            ) if args.recency_replay else HistoricalReplayBuffer(**context_options)
        )

    policy_optimizer = th.optim.Adam(policy.parameters(), lr=args.ppo_lr)
    critic_optimizer = th.optim.Adam(critic.parameters(), lr=args.ppo_lr)
    discriminator_optimizer = th.optim.Adam(
        discriminator.parameters(), lr=args.discriminator_lr
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
        restored_clock = restore_training_checkpoint(
            resume, args, modules, optimizers,
        )
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
        context_length=args.discriminator_context_length,
        context_stride=args.discriminator_context_stride,
        context_history=context_history,
    )

    ppo_config = PPOConfig(
        clip=args.ppo_clip,
        value_clip=args.value_clip,
        value_coef=args.value_coef,
        entropy_coef=args.entropy,
        normalize_advantage=True,
    )
    ppo_loss = PPOLoss(policy, critic, ppo_config)
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
                context_length=args.discriminator_context_length,
                differential=args.differential,
                gamma=args.gamma,
            ),
            GAE(
                gamma=args.gamma,
                lambda_=args.lambda_,
                reward_field="training_reward",
            ),
            SelectPPOFields(recurrent=args.gru),
        ),
        sampler=build_ppo_sampler(args),
        loss=ppo_loss,
        optimizer_step=IndependentOptimizerSteps(*ppo_optimizer_steps),
        section="PPO",
    )
    value_scheduler = build_training_scheduler(
        args, ppo_loss, policy_optimizer, critic_optimizer, discriminator_optimizer,
    )

    learner = Algorithm(discriminator_update, ppo_update)

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
            ("far_heldout_accuracy", "D far accuracy"),
            ("near_heldout_accuracy", "D near accuracy"),
            ("global_heldout_accuracy", "D global accuracy"),
            ("train_near_ball_fraction", "D near-ball fraction"),
        ):
            logger.register_progress_metric("Discriminator", key, label, ".3f")
    if args.hard_positive_mining:
        logger.register_progress_metric(
            "Discriminator", "reset_mined_fraction", "D mined reset frac", ".3f",
        )
    for enabled, key, label, fmt in (
        (args.entropy_end is not None, "entropy_coef", "entropy coef", ".4f"),
        (args.ppo_lr_end is not None, "ppo_lr", "PPO LR", ".2e"),
        (args.discriminator_lr_end is not None, "discriminator_lr", "D LR", ".2e"),
    ):
        if enabled:
            logger.register_progress_metric("Schedule", key, label, fmt)

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
    )

    def update_callback(trainer: Trainer) -> None:
        metrics = gameplay.diagnostic_metrics()
        if curated_resets is not None:
            fractions = curated_resets.take_skill_fractions()
            if fractions:
                metrics.setdefault("Replay", {}).update(fractions)
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
