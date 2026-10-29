import json
import logging
import threading
import time
from dataclasses import asdict
from types import SimpleNamespace

import torch
from carl.gymnasium.state import CarlState

from training.models import Agent
from training.setup import build_env
from training.tensors import plain_tensor
from vis.scene import Car, Physics


def load_policy(checkpoint, seed):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if payload.get("format") != "goddard-leanrl-v1":
        raise ValueError("expected a goddard-leanrl-v1 checkpoint")

    config = dict(payload["config"], n_sim=1, seed=seed)
    env = build_env(SimpleNamespace(**config))

    try:
        agent = Agent(
            env.single_observation_space.shape[0],
            env.single_action_space.nvec,
            env.action_codec,
            config["hidden_size"],
            config["layers"],
        ).to(env.device)
        agent.load_state_dict(payload["agent"], strict=True)
        agent.eval()
    except BaseException:
        env.close()
        raise

    return env, agent, config, payload["step"]


def snapshot(env):
    # CARL's torch wrapper has no public physical-state accessor yet.
    raw = torch.from_dlpack(env._env.get_state()).cpu().clone()
    state = CarlState.from_raw(raw, env.n_cars, env._boost_pad_positions, env._team_sign)
    ball = Physics(
        tuple(state.ball_position[0].tolist()),
        velocity=tuple(state.ball_velocity[0].tolist()),
        angular_velocity=tuple(state.ball_angular_velocity[0].tolist()),
    )
    cars = tuple(
        Car(
            id=index + 1,
            team=int(index >= env.n_cars // 2),
            physics=Physics(
                tuple(state.car_position[0, index].tolist()),
                tuple(state.car_forward[0, index].tolist()),
                tuple(state.car_up[0, index].tolist()),
                tuple(state.car_velocity[0, index].tolist()),
                tuple(state.car_angular_velocity[0, index].tolist()),
            ),
            boost=float(state.car_boost[0, index]),
            on_ground=bool(state.car_on_ground[0, index]),
            demoed=bool(state.car_demoed[0, index]),
        )
        for index in range(env.n_cars)
    )
    pads = tuple(
        (tuple(position), active)
        for position, active in zip(
            env._boost_pad_positions.cpu().tolist(), state.boost_pad_active[0].tolist()
        )
    )

    return ball, cars, pads


class Match:

    def __init__(self, checkpoint, seed=0):
        torch.manual_seed(seed)

        self.env, self.agent, self.config, step = load_policy(checkpoint, seed)
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.revision = 0
        self.data = None
        self.thread = threading.Thread(target=self.run, daemon=True)

        print(f"Loaded checkpoint at step {step:,}.", flush=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join()

    @torch.inference_mode()
    def run(self):
        try:
            observation = plain_tensor(self.env.reset())
            timestep = self.config["frameskip"] / 120.0
            deadline = time.monotonic()
            episode = 0

            while not self.stopped.is_set():
                ball, cars, pads = snapshot(self.env)
                data = json.dumps({
                    "episode": episode,
                    "ball":    asdict(ball),
                    "cars":    [asdict(car) for car in cars],
                    "pads":    pads,
                }, allow_nan=False, separators=(",", ":"))

                with self.condition:
                    self.data = data
                    self.revision += 1
                    self.condition.notify_all()

                actions, _, _, _ = self.agent.get_action_and_value(observation)
                observation, _, terminated, truncated, _ = self.env.step(actions)
                observation = plain_tensor(observation)
                episode += int((terminated | truncated).any().item())
                deadline = max(deadline + timestep, time.monotonic())

                self.stopped.wait(max(0.0, deadline - time.monotonic()))

        except Exception:
            logging.exception("Checkpoint playback stopped")
            
        finally:
            self.env.close()
            self.stopped.set()
            with self.condition:
                self.condition.notify_all()
