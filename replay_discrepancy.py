import argparse
import csv
import itertools
import json

from pathlib import Path

import numpy as np
import torch as th

from carl.gymnasium import CARLObservation, CARLTorchVectorEnv
from carl.gymnasium.action import ACTION_NVECS

from physics_utils import forward_up_to_quat


POSITION_SCALE = th.tensor((4108.0, 6000.0, 2076.0))
BALL_SPEED_SCALE = 6000.0
BALL_ANGULAR_SPEED_SCALE = 6.0
CAR_SPEED_SCALE = 2300.0
CAR_ANGULAR_SPEED_SCALE = 5.5
BOOST_SCALE = 100.0
NEUTRAL_ACTION = np.zeros(7, dtype=np.int32)
ACTION_NAMES = (
    "horizontal",
    "vertical",
    "throttle",
    "powerslide",
    "boost",
    "air_roll",
    "jump",
)


def infer_car_count(width: int) -> int:
    remainder = width - 107
    if remainder < 0 or remainder % 27:
        raise ValueError(f"invalid parsed replay width: {width}")
    n_cars = remainder // 27
    if n_cars not in (2, 4, 6):
        raise ValueError(f"unsupported parsed replay car count: {n_cars}")
    return n_cars


def resolve_replay(path: Path, pattern: str, replay_index: int) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no replay files matching {pattern!r} in {path}")
    if not 0 <= replay_index < len(matches):
        raise IndexError(f"replay index {replay_index} is outside [0, {len(matches)})")
    return matches[replay_index]


def observation(rows: th.Tensor) -> CARLObservation:
    return CARLObservation.from_tensor(rows[..., :30], 1)


def physical_values(rows: th.Tensor) -> dict[str, th.Tensor]:
    obs = observation(rows)
    ball = obs.ball
    ego = obs.cars.ego
    position_scale = POSITION_SCALE.to(rows.device)
    ball_position = ball.position * position_scale
    ego_position = ego.position * position_scale
    ball_velocity = ball.velocity * BALL_SPEED_SCALE
    ego_velocity = ego.velocity * CAR_SPEED_SCALE
    return {
        "ball_position": ball_position,
        "ball_velocity": ball_velocity,
        "ball_angular_velocity": ball.angular_velocity * BALL_ANGULAR_SPEED_SCALE,
        "ego_position": ego_position,
        "ego_velocity": ego_velocity,
        "ego_angular_velocity": ego.angular_velocity * CAR_ANGULAR_SPEED_SCALE,
        "ego_forward": ego.forward,
        "ego_up": ego.up,
        "relative_ball_position": ball_position - ego_position,
        "relative_ball_velocity": ball_velocity - ego_velocity,
    }


def errors(actual: th.Tensor, target: th.Tensor) -> dict[str, th.Tensor]:
    actual_values = physical_values(actual)
    target_values = physical_values(target)
    return {
        name: th.linalg.vector_norm(actual_values[name] - target_values[name], dim=-1)
        for name in (
            "ego_position",
            "ego_velocity",
            "ego_angular_velocity",
            "ego_forward",
            "ego_up",
            "relative_ball_position",
            "relative_ball_velocity",
            "ball_angular_velocity",
        )
    }


def tracking_cost(actual: th.Tensor, target: th.Tensor) -> th.Tensor:
    error = errors(actual, target)
    return (
        (error["relative_ball_position"] / 100.0).square()
        + 0.50 * (error["relative_ball_velocity"] / 100.0).square()
        + 0.50 * (error["ego_position"] / 100.0).square()
        + 0.25 * (error["ego_velocity"] / 100.0).square()
        + 0.10 * error["ego_angular_velocity"].square()
        + 2.00 * error["ego_forward"].square()
        + 2.00 * error["ego_up"].square()
    )


class ReplayReset:
    def __init__(self, rows: th.Tensor, internal_start: int) -> None:
        self.rows = rows
        self.internal_start = internal_start

    def __call__(self, mask: th.Tensor) -> dict[str, th.Tensor]:
        rows = self.rows[mask]
        obs = observation(rows)
        ball = obs.ball
        ego = obs.cars.ego
        position_scale = POSITION_SCALE.to(rows.device)
        return {
            "simulation_indices": mask.nonzero(as_tuple=True)[0],
            "ball_position": ball.position * position_scale,
            "ball_velocity": ball.velocity * BALL_SPEED_SCALE,
            "ball_angular_velocity": ball.angular_velocity * BALL_ANGULAR_SPEED_SCALE,
            "car_position": ego.position[:, None, :] * position_scale,
            "car_rotation": forward_up_to_quat(ego.forward, ego.up)[:, None, :],
            "car_velocity": ego.velocity[:, None, :] * CAR_SPEED_SCALE,
            "car_angular_velocity": ego.angular_velocity[:, None, :] * CAR_ANGULAR_SPEED_SCALE,
            "car_demoed": ego.demoed[:, None],
            "car_boost": ego.boost[:, None] * BOOST_SCALE,
            "car_internal_state": rows[
                :, self.internal_start:self.internal_start + 19
            ][:, None, :],
            "blue_score": th.zeros(len(rows), dtype=th.int32, device=rows.device),
            "orange_score": th.zeros(len(rows), dtype=th.int32, device=rows.device),
            "episode_ticks": th.zeros(len(rows), dtype=th.int32, device=rows.device),
        }


def transition_observation(
    env: CARLTorchVectorEnv,
    obs: th.Tensor,
    terminated: th.Tensor,
    truncated: th.Tensor,
    info: dict,
) -> th.Tensor:
    done = terminated | truncated
    if done.any():
        obs = obs.clone()
        obs[done] = info["final_obs"][done]
    return obs[..., :30]


def all_actions(device: th.device) -> th.Tensor:
    return th.tensor(
        list(itertools.product(*(range(n) for n in ACTION_NVECS))),
        dtype=th.int32,
        device=device,
    )


def infer_actions(
    rows: th.Tensor,
    internal_start: int,
    frameskip: int,
    simulation_batch_size: int,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    candidates = all_actions(rows.device)
    action_count = len(candidates)
    transitions_per_batch = simulation_batch_size // action_count
    if transitions_per_batch < 1:
        raise ValueError(f"simulation batch size must be at least {action_count}")
    n_sim = transitions_per_batch * action_count
    reset = ReplayReset(rows[:1].expand(n_sim, -1), internal_start)
    env = CARLTorchVectorEnv(
        n_sim=n_sim,
        n_blue=1,
        n_orange=0,
        frameskip=frameskip,
        max_ticks=1_000_000,
        normalize=True,
        reset_state_provider=reset,
    )
    selected_actions = []
    selected_states = []
    selected_costs = []

    try:
        for start in range(0, len(rows) - 1, transitions_per_batch):
            count = min(transitions_per_batch, len(rows) - 1 - start)
            source_indices = th.arange(start, start + count, device=rows.device)
            source_indices = source_indices.repeat_interleave(action_count)
            if len(source_indices) < n_sim:
                source_indices = th.cat((
                    source_indices,
                    source_indices[:1].expand(n_sim - len(source_indices)),
                ))
            reset.rows = rows[source_indices]
            env.reset()
            actions = candidates.repeat(transitions_per_batch, 1)
            obs, _, terminated, truncated, info = env.step(actions)
            obs = transition_observation(env, obs, terminated, truncated, info)
            obs = obs[:count * action_count].view(count, action_count, 30)
            target = rows[start + 1:start + count + 1, :30, None].transpose(1, 2)
            costs = tracking_cost(obs, target)
            best = costs.argmin(dim=1)
            batch = th.arange(count, device=rows.device)
            selected_actions.append(candidates[best])
            selected_states.append(obs[batch, best])
            selected_costs.append(costs[batch, best])
    finally:
        env.close()

    return (
        th.cat(selected_actions),
        th.cat(selected_states),
        th.cat(selected_costs),
    )


def simulate_actions(
    rows: th.Tensor,
    actions: th.Tensor,
    internal_start: int,
    frameskip: int,
    simulation_batch_size: int,
) -> th.Tensor:
    count = len(rows) - 1
    n_sim = min(count, simulation_batch_size)
    reset = ReplayReset(rows[:1].expand(n_sim, -1), internal_start)
    env = CARLTorchVectorEnv(
        n_sim=n_sim,
        n_blue=1,
        n_orange=0,
        frameskip=frameskip,
        max_ticks=1_000_000,
        normalize=True,
        reset_state_provider=reset,
    )
    states = []

    try:
        for start in range(0, count, n_sim):
            batch_count = min(n_sim, count - start)
            source = rows[start:start + batch_count]
            batch_actions = actions[start:start + batch_count]
            if batch_count < n_sim:
                source = th.cat((source, source[:1].expand(n_sim - batch_count, -1)))
                batch_actions = th.cat((
                    batch_actions,
                    batch_actions[:1].expand(n_sim - batch_count, -1),
                ))

            reset.rows = source
            env.reset()
            obs, _, terminated, truncated, info = env.step(batch_actions)
            states.append(
                transition_observation(env, obs, terminated, truncated, info)[:batch_count]
            )
    finally:
        env.close()

    return th.cat(states)


def rollout(
    rows: th.Tensor,
    actions: th.Tensor,
    internal_start: int,
    frameskip: int,
) -> tuple[th.Tensor, int | None]:
    reset = ReplayReset(rows[:1], internal_start)
    env = CARLTorchVectorEnv(
        n_sim=1,
        n_blue=1,
        n_orange=0,
        frameskip=frameskip,
        max_ticks=1_000_000,
        normalize=True,
        reset_state_provider=reset,
    )
    states = []
    ended_at = None
    try:
        states.append(env.reset()[0, :30])
        for index, action in enumerate(actions):
            obs, _, terminated, truncated, info = env.step(action[None, :])
            states.append(
                transition_observation(env, obs, terminated, truncated, info)[0]
            )
            if (terminated | truncated).item():
                ended_at = index + 1
                break
    finally:
        env.close()
    return th.stack(states), ended_at


def ball_impulse(rows: th.Tensor, frameskip: int) -> th.Tensor:
    values = physical_values(rows)
    gravity_delta = th.tensor(
        (0.0, 0.0, -650.0 * frameskip / 120.0), device=rows.device
    )
    return values["ball_velocity"][1:] - values["ball_velocity"][:-1] - gravity_delta


def classify_impulse(
    position: np.ndarray,
    impulse: np.ndarray,
    ego_touch: bool,
    other_touch: bool,
) -> str:
    if ego_touch:
        return "ego_car"
    if other_touch:
        return "other_car"
    if np.linalg.norm(impulse) < 100.0:
        return "none"
    x, y, z = np.abs(position)
    if x > 3000.0 and y > 4000.0:
        return "corner"
    if z < 160.0:
        return "floor"
    if z > 1900.0:
        return "ceiling"
    if x > 3900.0:
        return "side_wall"
    if y > 5000.0:
        return "back_wall"
    return "unclassified"


def load_actions(
    path: Path,
    count: int,
    device: th.device,
    start: int = 0,
) -> th.Tensor:
    loaded = np.load(path)
    actions = loaded["carl"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"actions must have shape [steps, 7], got {actions.shape}")
    if len(actions) < start + count:
        raise ValueError(
            f"actions has {len(actions)} rows but {start + count} are required"
        )
    actions = actions[start:start + count]
    if not np.isfinite(actions).all() or not np.equal(actions, np.floor(actions)).all():
        raise ValueError("actions must contain finite integer values")
    nvec = np.asarray(ACTION_NVECS)
    if np.any(actions < 0) or np.any(actions >= nvec):
        raise ValueError("actions contain a value outside CARL's MultiDiscrete ranges")
    return th.as_tensor(actions, dtype=th.int32, device=device)


def metric_columns(prefix: str, values: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {f"{prefix}_{name}_error": float(value[index]) for name, value in values.items()}


def valid_mean(values: np.ndarray, valid: np.ndarray) -> float | None:
    return float(values[valid].mean()) if valid.any() else None


def write_results(
    prefix: Path,
    replay_path: Path,
    source_rows: np.ndarray,
    replay: th.Tensor,
    actions: th.Tensor,
    oracle_states: th.Tensor | None,
    oracle_costs: th.Tensor | None,
    rollout_states: th.Tensor,
    start: int,
    frameskip: int,
    ended_at: int | None,
    action_source: str,
) -> dict:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    count = len(rollout_states) - 1
    replay = replay[:count + 1]
    actions = actions[:count]
    rollout_states = rollout_states[:count + 1]
    rollout_errors = {
        key: value.detach().cpu().numpy()
        for key, value in errors(rollout_states[1:], replay[1:]).items()
    }
    oracle_errors = None
    if oracle_states is not None:
        oracle_errors = {
            key: value.detach().cpu().numpy()
            for key, value in errors(oracle_states[:count], replay[1:]).items()
        }

    replay_values = physical_values(replay)
    replay_positions = replay_values["ball_position"][1:].detach().cpu().numpy()
    replay_impulses = ball_impulse(replay, frameskip).detach().cpu().numpy()
    rollout_impulses = ball_impulse(rollout_states, frameskip).detach().cpu().numpy()
    oracle_impulses = None
    if oracle_states is not None:
        replay_ball_velocity = replay_values["ball_velocity"][:-1]
        oracle_ball_velocity = physical_values(oracle_states[:count])["ball_velocity"]
        gravity_delta = th.tensor(
            (0.0, 0.0, -650.0 * frameskip / 120.0), device=replay.device
        )
        oracle_impulses = (
            oracle_ball_velocity - replay_ball_velocity - gravity_delta
        ).detach().cpu().numpy()
    action_values = actions.detach().cpu().numpy()
    event_rows = source_rows[1:count + 1, -5:]

    csv_path = prefix.with_suffix(".csv")
    records = []
    for index in range(count):
        event = classify_impulse(
            replay_positions[index],
            replay_impulses[index],
            bool(event_rows[index, 0]),
            bool(event_rows[index, 1]),
        )
        record = {
            "frame": start + index,
            "target_frame": start + index + 1,
            "event": event,
            "ego_touch": bool(event_rows[index, 0]),
            "other_touch": bool(event_rows[index, 1]),
            "bump": bool(event_rows[index, 2]),
            "large_replay_correction": bool(event_rows[index, 3]),
            "physics_discontinuity": bool(event_rows[index, 4]),
            "replay_ball_impulse": float(np.linalg.norm(replay_impulses[index])),
            "rollout_ball_impulse": float(np.linalg.norm(rollout_impulses[index])),
            "ball_impulse_vector_error": float(
                np.linalg.norm(rollout_impulses[index] - replay_impulses[index])
            ),
            **{
                name: int(action_values[index, action_index])
                for action_index, name in enumerate(ACTION_NAMES)
            },
            **metric_columns("rollout", rollout_errors, index),
        }
        if oracle_errors is not None and oracle_costs is not None:
            record.update(metric_columns("oracle", oracle_errors, index))
            record["oracle_cost"] = float(oracle_costs[index].item())
            record["oracle_ball_impulse_vector_error"] = float(
                np.linalg.norm(oracle_impulses[index] - replay_impulses[index])
            )
        records.append(record)

    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    npz_path = prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        replay_path=str(replay_path),
        start=start,
        frameskip=frameskip,
        actions=action_values,
        replay_states=replay.detach().cpu().numpy(),
        oracle_states=(
            oracle_states[:count].detach().cpu().numpy()
            if oracle_states is not None else np.empty((0, 30), dtype=np.float32)
        ),
        rollout_states=rollout_states.detach().cpu().numpy(),
    )

    # Other cars are intentionally absent from the CARL run, and parser-marked
    # corrections/discontinuities are not meaningful simulator comparisons.
    invalid_rows = source_rows[:count + 1, -4:].astype(bool).any(axis=1)
    valid = ~invalid_rows[:-1] & ~invalid_rows[1:]
    event_labels = np.asarray([record["event"] for record in records])
    events, event_counts = np.unique(
        event_labels, return_counts=True
    )
    impulse_errors = np.linalg.norm(rollout_impulses - replay_impulses, axis=1)
    oracle_impulse_errors = (
        np.linalg.norm(oracle_impulses - replay_impulses, axis=1)
        if oracle_impulses is not None else None
    )
    event_metrics = {}
    for event in events:
        selected = valid & (event_labels == event)
        event_metrics[event] = {
            "valid_transitions": int(selected.sum()),
            "mean_rollout_relative_ball_position_error": valid_mean(
                rollout_errors["relative_ball_position"], selected
            ),
            "mean_ball_impulse_vector_error": valid_mean(impulse_errors, selected),
        }
        if oracle_errors is not None:
            event_metrics[event]["mean_oracle_relative_ball_position_error"] = valid_mean(
                oracle_errors["relative_ball_position"], selected
            )
            event_metrics[event]["mean_oracle_ball_impulse_vector_error"] = valid_mean(
                oracle_impulse_errors, selected
            )

    worst = np.argsort(impulse_errors[valid])[-10:][::-1] if valid.any() else []
    valid_indices = np.flatnonzero(valid)
    summary = {
        "replay": str(replay_path),
        "start": start,
        "transitions": count,
        "frameskip": frameskip,
        "rollout_ended_at_transition": ended_at,
        "valid_transitions": int(valid.sum()),
        "events": dict(zip(events.tolist(), event_counts.tolist())),
        "event_metrics": event_metrics,
        "worst_ball_impulse_frames": [
            start + int(valid_indices[index]) + 1 for index in worst
        ],
        "mean_rollout_relative_ball_position_error": valid_mean(
            rollout_errors["relative_ball_position"], valid
        ),
        "mean_rollout_ego_position_error": valid_mean(
            rollout_errors["ego_position"], valid
        ),
        "mean_ball_impulse_vector_error": valid_mean(
            impulse_errors, valid
        ),
        "action_source": action_source,
        "limitations": [
            "CARL has no boost-pad state setter, so replay pad availability is not restored",
            "contact caches are cleared whenever a replay frame is injected",
        ],
        "csv": str(csv_path),
        "npz": str(npz_path),
    }
    if oracle_errors is not None:
        summary["mean_oracle_relative_ball_position_error"] = valid_mean(
            oracle_errors["relative_ball_position"], valid
        )
        summary["mean_oracle_ego_position_error"] = valid_mean(
            oracle_errors["ego_position"], valid
        )
        summary["mean_oracle_ball_impulse_vector_error"] = valid_mean(
            oracle_impulse_errors, valid
        )
    return summary


def parse_args() -> argparse.Namespace:
    default_replays = Path(__file__).parent / "ballchasing_replays" / "parsed_replays"
    parser = argparse.ArgumentParser(
        description=(
            "Compare CARL ego/relative-ball physics against a parsed Rocket League replay. "
            "Without --actions, all 648 discrete actions are tested at each frame and the "
            "best one-step action sequence is rolled out without resets."
        )
    )
    parser.add_argument("replay", type=Path, nargs="?", default=default_replays)
    parser.add_argument("--pattern", default="*.npy")
    parser.add_argument("--replay-index", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--frameskip", type=int, default=4)
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--neutral", action="store_true")
    parser.add_argument("--simulation-batch-size", type=int, default=3888)
    parser.add_argument("--output-prefix", type=Path, default=Path("replay_discrepancy"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.actions is not None and args.neutral:
        raise ValueError("--actions and --neutral are mutually exclusive")
    if args.start < 0 or args.steps < 1 or args.frameskip < 1:
        raise ValueError("start must be nonnegative; steps and frameskip must be positive")

    replay_path = resolve_replay(args.replay, args.pattern, args.replay_index)
    source = np.load(replay_path, mmap_mode="r")
    n_cars = infer_car_count(source.shape[1])
    end = min(args.start + args.steps + 1, len(source))
    if end - args.start < 2:
        raise ValueError("the selected replay range has no transitions")
    source_rows = np.array(source[args.start:end], dtype=np.float32, copy=True)
    rows = th.as_tensor(source_rows, device="cuda:0")
    internal_start = 83 + 27 * n_cars
    count = len(rows) - 1

    oracle_states = None
    oracle_costs = None
    companion_actions = replay_path.with_suffix(".actions.npz")
    if args.actions is not None:
        actions = load_actions(args.actions, count, rows.device)
        action_source = str(args.actions)
    elif args.neutral:
        actions = th.as_tensor(NEUTRAL_ACTION, device=rows.device).expand(count, -1)
        action_source = "neutral"
    elif companion_actions.exists():
        actions = load_actions(
            companion_actions, count, rows.device, start=args.start
        )
        action_source = str(companion_actions)
    else:
        actions, oracle_states, oracle_costs = infer_actions(
            rows, internal_start, args.frameskip, args.simulation_batch_size
        )
        action_source = "inferred"

    if oracle_states is None:
        oracle_states = simulate_actions(
            rows,
            actions,
            internal_start,
            args.frameskip,
            args.simulation_batch_size,
        )
        oracle_costs = tracking_cost(oracle_states, rows[1:, :30])

    rollout_states, ended_at = rollout(rows, actions, internal_start, args.frameskip)
    summary = write_results(
        args.output_prefix,
        replay_path,
        source_rows,
        rows[:, :30],
        actions,
        oracle_states,
        oracle_costs,
        rollout_states,
        args.start,
        args.frameskip,
        ended_at,
        action_source,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
