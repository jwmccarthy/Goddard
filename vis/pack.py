import importlib.util
import json
import queue
import re
import threading
import time
from dataclasses import asdict
from pathlib import Path

import torch

from vis.playback import snapshot
from vis.scene import Car, Physics


def condition_label(condition):
    kind = type(condition).__name__

    if kind == "StateCondition":
        metric = condition.metric.replace("_", " ").capitalize()
        lower, upper = condition.minimum, condition.maximum

        if lower is not None and upper is not None:
            return f"{metric} {lower:g}–{upper:g}"
        if lower is not None:
            return f"{metric} ≥ {lower:g}"
        if upper is not None:
            return f"{metric} ≤ {upper:g}"

        return metric

    if kind in ("LeaveSurface", "SurfaceContact"):
        surface = getattr(condition.surface, "name", None)
        surface = surface.replace("_", " ").replace("|", " / ").lower() if surface else "surface"
        entity = "Ball" if condition.entity == "ball" else "Car"
        verb = "leaves" if kind == "LeaveSurface" else "touches"

        return f"{entity} {verb} {surface}"

    if kind == "BallTouch":
        surface = (" / ".join(part.name for part in condition.surface)
                   if isinstance(condition.surface, tuple) else getattr(condition.surface, "name", None))
        label = f"Ball touch · {surface.lower()}" if surface else "Ball touch"
        minimum = getattr(condition, "min_speed_gain", None)
        maximum = getattr(condition, "max_speed_gain", None)
        if minimum is not None:
            label += f" · speed change ≥ {minimum:g}"
        if maximum is not None:
            label += f" · speed change ≤ {maximum:g}"
        return label

    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", kind).capitalize()


def load_pack(path):
    from carl.training import TrainingPack

    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("goddard_viewer_pack", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pack = getattr(module, "PACK", None)

    if not isinstance(pack, TrainingPack):
        raise ValueError(f"{path} must export a CARL TrainingPack named PACK")

    return pack


class PreviewProvider:

    def __init__(self, pack, seed):
        from carl.training import ExerciseResetProvider

        self.pack = pack
        self.selected = 0
        self.exercise_index = torch.zeros(1, device="cuda:0", dtype=torch.int64)
        self.providers = [
            ExerciseResetProvider(exercise, n_sim=1, seed=seed + index)
            for index, exercise in enumerate(pack.exercises)
        ]

    def __call__(self, mask):
        self.exercise_index = torch.where(mask, self.selected, self.exercise_index)
        return self.providers[self.selected](mask)

    def select(self, offset):
        self.selected = (self.selected + offset) % len(self.pack.exercises)


def terminal_snapshot(observation, pads, normalized=False):
    import carl

    position_scale = carl.OBS_POSITION_SCALE if normalized else (1, 1, 1)
    ball_velocity_scale = carl.BALL_MAX_SPEED if normalized else 1
    ball_angular_scale = carl.BALL_MAX_ANG_SPEED if normalized else 1
    car_velocity_scale = carl.CAR_MAX_SPEED if normalized else 1
    car_angular_scale = carl.CAR_MAX_ANG_SPEED if normalized else 1
    boost_scale = carl.BOOST_MAX if normalized else 1

    def vector(value, scale):
        values = value.cpu().tolist()
        return tuple(component * factor for component, factor in zip(values, scale))

    ball = Physics(
        vector(observation.ball_position[0], position_scale),
        velocity=tuple(value * ball_velocity_scale for value in observation.ball_velocity[0].cpu().tolist()),
        angular_velocity=tuple(value * ball_angular_scale for value in observation.ball_angular_velocity[0].cpu().tolist()),
    )
    car = observation.cars.ego
    cars = (Car(
        id=1,
        team=0,
        physics=Physics(
            vector(car.position[0], position_scale),
            tuple(car.forward[0].cpu().tolist()),
            tuple(car.up[0].cpu().tolist()),
            tuple(value * car_velocity_scale for value in car.velocity[0].cpu().tolist()),
            tuple(value * car_angular_scale for value in car.angular_velocity[0].cpu().tolist()),
        ),
        boost=float(car.boost[0]) * boost_scale,
        on_ground=bool(car.on_ground[0]),
        demoed=bool(car.demoed[0]),
    ),)

    return ball, cars, pads


class PackMatch:

    def __init__(self, path=None, seed=0, checkpoint=None, sample=False):
        from carl.gymnasium import CARLTorchVectorEnv
        from carl.training import (
            CompiledTrainingPack,
            TrainingPackRuntime,
            TrainingPackTerminationManager,
        )

        payload = None
        config = {}
        attention_checkpoint = False

        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            config = payload.get("config", {})
            attention_checkpoint = payload.get("format") == "carl-training-pack-ppo-v1"

            if attention_checkpoint:
                from examples.training_packs.playback import checkpoint_pack

                self.pack = checkpoint_pack(payload)
            else:
                import carl

                from training.wall_flip_reset import wall_flip_reset_pack

                objective = config.get("pack_objective")

                if (payload.get("format") != "goddard-leanrl-v1" or payload.get("algorithm") != "pack-ppo"
                        or objective not in ("wall-flip-reset-v1", "wall-flip-reset-v2",
                                             "wall-flip-reset-v3", "wall-flip-reset-v4", "wall-flip-reset-v5",
                                             "wall-flip-reset-v6", "wall-flip-reset-v7", "wall-flip-reset-v8",
                                             "wall-flip-reset-v9", "wall-flip-reset-v10", "wall-flip-reset-v11", "wall-flip-reset-v12", "wall-flip-reset-v13")):
                    raise ValueError("expected a compatible wall-flip-reset pack-ppo checkpoint")

                observation_size = (carl.OBS_BALL + carl.OBS_PER_CAR + carl.OBS_BOOST_PADS
                                    + carl.OBS_RELATIVE_EGO_BALL + carl.OBS_RELATIVE_GOALS + carl.OBS_EGO_DODGE)
                stages = payload["agent"]["actor.0.weight"].shape[1] - observation_size - 1

                from training.legacy_wall_flip_reset import wall_flip_reset_pack as legacy_pack

                self.pack = wall_flip_reset_pack() if objective in ("wall-flip-reset-v11", "wall-flip-reset-v12", "wall-flip-reset-v13") else legacy_pack(
                    include_ground_setup=objective != "wall-flip-reset-v1",
                    include_ceiling_failure=objective in ("wall-flip-reset-v3", "wall-flip-reset-v4", "wall-flip-reset-v5", "wall-flip-reset-v6", "wall-flip-reset-v7", "wall-flip-reset-v8", "wall-flip-reset-v9", "wall-flip-reset-v10", "wall-flip-reset-v11"),
                    airborne_setup=objective in ("wall-flip-reset-v4", "wall-flip-reset-v5", "wall-flip-reset-v6", "wall-flip-reset-v7", "wall-flip-reset-v8", "wall-flip-reset-v9", "wall-flip-reset-v10", "wall-flip-reset-v11"),
                    extra_air_touches=(objective in ("wall-flip-reset-v4", "wall-flip-reset-v5", "wall-flip-reset-v6", "wall-flip-reset-v7", "wall-flip-reset-v8", "wall-flip-reset-v9", "wall-flip-reset-v10", "wall-flip-reset-v11") or
                                       (objective == "wall-flip-reset-v3" and stages == 10)),
                    require_takeoff=objective in ("wall-flip-reset-v5", "wall-flip-reset-v6", "wall-flip-reset-v7", "wall-flip-reset-v8", "wall-flip-reset-v9", "wall-flip-reset-v10", "wall-flip-reset-v11"),
                    front_air_touches=objective in ("wall-flip-reset-v6", "wall-flip-reset-v7", "wall-flip-reset-v8", "wall-flip-reset-v9", "wall-flip-reset-v10", "wall-flip-reset-v11"),
                    minimum_air_touch_gain=(200 if objective in ("wall-flip-reset-v8", "wall-flip-reset-v9") else
                                            100 if objective in ("wall-flip-reset-v7", "wall-flip-reset-v10", "wall-flip-reset-v11") else None),
                    repeat_air_touches=objective in ("wall-flip-reset-v8", "wall-flip-reset-v9", "wall-flip-reset-v10"),
                    max_extra_air_touches=3 if objective == "wall-flip-reset-v10" else None,
                )
                if objective == "wall-flip-reset-v11" and stages == 8:
                    del self.pack.exercises[0].stages[6:8]
        else:
            self.pack = load_pack(path)

        self.provider = PreviewProvider(self.pack, seed)
        compiled = CompiledTrainingPack.compile(self.pack)
        self.runtime = TrainingPackRuntime(compiled, n_sim=1)
        manager = TrainingPackTerminationManager(self.runtime, self.provider)
        self.frameskip = config.get("frameskip", 4)
        self.env = CARLTorchVectorEnv(
            n_sim=1, n_blue=1, n_orange=0, seed=seed, frameskip=self.frameskip,
            max_ticks=config.get("max_ticks", 36_000),
            no_touch_timeout_seconds=config.get("no_touch_timeout"),
            normalize=payload is not None, discrete_actions=payload is not None,
            capture_events=True, capture_ball_touches=compiled.needs_touch,
            reset_state_provider=self.provider, termination_manager=manager,
        )

        self.agent = None
        self.observations = None

        if payload is not None:
            try:
                if attention_checkpoint:
                    from examples.training_packs.playback import checkpoint_policy

                    self.agent = checkpoint_policy(payload, self.runtime, self.env.single_observation_space.shape[0])
                else:
                    from training.models import Agent
                    from training.pack_ppo import PackActionCodec, PackObservationEnv

                    base_size = self.env.single_observation_space.shape[0]
                    if objective == "wall-flip-reset-v12":
                        observation_size = base_size
                    else:
                        self.observations = PackObservationEnv(self.env, self.runtime, include_repeat_count=objective == "wall-flip-reset-v10")
                        observation_size = base_size + compiled.kind.shape[1] + 1 + (objective == "wall-flip-reset-v10")
                    action_codec = (PackActionCodec(self.env.action_codec, base_size)
                                    if objective in ("wall-flip-reset-v4", "wall-flip-reset-v5", "wall-flip-reset-v6", "wall-flip-reset-v7", "wall-flip-reset-v8", "wall-flip-reset-v9", "wall-flip-reset-v10", "wall-flip-reset-v11", "wall-flip-reset-v13") else self.env.action_codec)
                    self.agent = Agent(
                        observation_size, self.env.single_action_space.nvec, action_codec,
                        config["hidden_size"], config["layers"],
                    ).to(self.env.device)
                    actor = {
                        key.removeprefix("actor."): value
                        for key, value in payload["agent"].items() if key.startswith("actor.")
                    }
                    self.agent.actor.load_state_dict(actor, strict=True)
                    self.agent.eval()
            except BaseException:
                self.env.close()
                raise

            print(f"Loaded pack checkpoint at step {payload.get('steps', payload.get('step', 0)):,}.", flush=True)

        self.condition = threading.Condition()
        self.seed = seed
        self.sample = sample
        self.stopped = threading.Event()
        self.revision = 0
        self.data = None
        self.reset_requests = queue.SimpleQueue()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join()

    def request_reset(self, mode):
        if mode not in ("pack", "next", "previous"):
            raise ValueError(f"unknown reset mode: {mode}")

        self.reset_requests.put(mode)

    def metadata(self, outcome=None):
        index = int(self.runtime.exercise_index[0])
        stage = int(self.runtime.stage_index[0])
        remaining = int(self.runtime.ticks_remaining[0])
        exercise = self.pack.exercises[index]
        current = exercise.stages[stage]
        elapsed = int((self.runtime.elapsed_ticks if current.from_exercise_start
                       else self.runtime.stage_elapsed_ticks)[0])

        return {
            "pack": self.pack.name,
            "exercise": exercise.name,
            "exercise_index": index,
            "exercise_count": len(self.pack.exercises),
            "stage": stage + 1,
            "stage_count": len(exercise.stages),
            "stages": [
                {
                    "label": (condition_label(item.event) +
                              (f" (repeat: {condition_label(item.repeat_on)})" if item.repeat_on else "")),
                    "deadline": item.within_ticks,
                    "min_ticks": item.min_ticks,
                    "from_exercise_start": item.from_exercise_start,
                    "failures": [condition_label(condition) for condition in item.fail_on],
                }
                for item in exercise.stages
            ],
            "failures": [condition_label(condition) for condition in exercise.fail_on],
            "remaining": remaining if remaining >= 0 else None,
            "elapsed": elapsed,
            "outcome": outcome,
            "terminal": False,
        }

    def publish(self, scene, episode, metadata, controls=None):
        ball, cars, pads = scene
        data = json.dumps({
            "episode": episode,
            "ball": asdict(ball),
            "cars": [asdict(car) for car in cars],
            "pads": pads,
            "training": metadata,
            "controls": controls,
        }, allow_nan=False, separators=(",", ":"))

        with self.condition:
            self.data = data
            self.revision += 1
            self.condition.notify_all()

    @torch.inference_mode()
    def run(self):
        import logging

        try:
            torch.manual_seed(self.seed)
            observation = self.env.reset()
            actions = torch.zeros((1, 7), device=self.env.device)
            episode = 0
            outcome = None
            deadline = time.monotonic()

            while not self.stopped.is_set():
                try:
                    request = self.reset_requests.get_nowait()
                except queue.Empty:
                    request = None

                if request is not None:
                    if request == "next":
                        self.provider.select(1)
                    elif request == "previous":
                        self.provider.select(-1)

                    observation = self.env.reset()
                    outcome = None
                    episode += 1
                    deadline = time.monotonic()

                if self.agent is not None:
                    policy_observation = (self.observations.observation(
                        observation, self.runtime.stage_index, self.runtime.ticks_remaining,
                    ) if self.observations is not None else observation)
                    actions = self.agent.get_action(policy_observation, sample=self.sample)

                from vis.controls import control_values
                controls = control_values(actions, self.env.discrete_actions) if self.agent is not None else None
                before = self.metadata(outcome)
                self.publish(snapshot(self.env, blue_cars=1), episode, before, controls)

                observation, _, terminated, truncated, info = self.env.step(actions)

                if bool((terminated | truncated)[0]):
                    outcome = ("success" if bool(info["exercise_success"][0]) else
                               "failure" if bool(info["exercise_failure"][0]) else
                               "timeout" if bool(info["exercise_timeout"][0]) else "episode ended")
                    before["outcome"] = outcome
                    before["terminal"] = True
                    _, _, pads = snapshot(self.env, blue_cars=1)
                    self.publish(terminal_snapshot(info["final_obs"], pads, normalized=self.agent is not None),
                                 episode, before, controls)
                    self.stopped.wait(0.75)
                    episode += 1
                    deadline = time.monotonic()

                deadline = max(deadline + self.frameskip / 120.0, time.monotonic())
                self.stopped.wait(max(0.0, deadline - time.monotonic()))

        except Exception:
            logging.exception("Training pack playback stopped")

        finally:
            self.env.close()
            self.stopped.set()
            with self.condition:
                self.condition.notify_all()
