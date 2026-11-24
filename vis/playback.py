from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from carl.gymnasium.state import CarlState

from training.checkpoints import load_payload
from training.models import Agent
from training.replay_resets import ReplayResetSampler
from training.residual import attach_residual_actor
from training.setup import build_env
from training.tensors import plain_tensor
from vis.controls import control_values
from vis.scene import Car, Physics

if TYPE_CHECKING:
    from carl.gymnasium import CARLResetState, CARLTorchVectorEnv


def build_agent(
    env:     CARLTorchVectorEnv,
    config:  dict[str, Any],
    payload: dict[str, Any]
) -> Agent:
    """Rebuild a checkpoint's agent on env's device with its actor weights."""
    agent = Agent(
        env.single_observation_space.shape[0],
        env.single_action_space.nvec,
        env.action_codec,
        config["hidden_size"],
        config["layers"],
        policy_hidden=config["policy_hidden_size"],
        policy_layers=config["policy_layers"]
    )

    if config.get("residual_policy", False):
        attach_residual_actor(
            agent, env.single_observation_space.shape[0], config
        )

    agent = agent.to(env.device)

    actor = {
        key.removeprefix("actor."): value
        for key, value in payload["agent"].items()
        if key.startswith("actor.")
    }

    agent.actor.load_state_dict(actor, strict=True)
    agent.eval()

    return agent


def load_policy(
    checkpoint: Path,
    seed:       int
) -> tuple[CARLTorchVectorEnv, Agent, dict[str, Any], int]:
    payload = load_payload(checkpoint)
    config = dict(payload["config"], n_sim=1, seed=seed, resume=None)
    env = build_env(SimpleNamespace(**config))

    try:
        agent = build_agent(env, config, payload)
    except BaseException:
        env.close()

        raise

    return (env, agent, config, payload["step"])


def snapshot(
    env:       CARLTorchVectorEnv,
    blue_cars: int | None = None
) -> tuple[Physics, tuple[Car, ...], tuple[tuple[tuple[float, ...], bool], ...]]:
    # CARL's torch wrapper has no public physical-state accessor yet.
    raw = torch.from_dlpack(env._env.get_state()).cpu().clone()

    state = CarlState.from_raw(
        raw,
        env.n_cars,
        env._boost_pad_positions,
        env._team_sign
    )

    ball = Physics(
        tuple(state.ball_position[0].tolist()),
        velocity=tuple(state.ball_velocity[0].tolist()),
        angular_velocity=tuple(state.ball_angular_velocity[0].tolist())
    )

    blue_cars = env.n_cars // 2 if blue_cars is None else blue_cars

    cars = tuple(
        Car(
            id=index + 1,
            team=int(index >= blue_cars),
            physics=Physics(
                tuple(state.car_position[0, index].tolist()),
                tuple(state.car_forward[0, index].tolist()),
                tuple(state.car_up[0, index].tolist()),
                tuple(state.car_velocity[0, index].tolist()),
                tuple(state.car_angular_velocity[0, index].tolist())
            ),
            boost=float(state.car_boost[0, index]),
            on_ground=bool(state.car_on_ground[0, index]),
            demoed=bool(state.car_demoed[0, index]),
            boosting=bool(state.car_is_boosting[0, index])
        )
        for index in range(env.n_cars)
    )

    pads = tuple(
        (tuple(position), active)
        for position, active in zip(
            env._boost_pad_positions.cpu().tolist(),
            state.boost_pad_active[0].tolist()
        )
    )

    return (ball, cars, pads)


class Match:

    def __init__(
        self,
        checkpoint:  Path,
        seed:        int = 0,
        replay_dir:  Path | None = None,
        replay_seed: int | None = None,
        sample:      bool = False
    ) -> None:
        torch.manual_seed(seed)

        # Keep latest.pt unresolved so future reloads follow its new target.
        self.checkpoint = checkpoint.expanduser().absolute()
        self.replay_dir = replay_dir
        self.env, self.agent, self.config, step = load_policy(self.checkpoint, seed)
        self.checkpoint_step = step

        if replay_dir is not None:
            self.config["replay_dir"] = replay_dir

        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.revision = 0
        self.data = None
        self.requests = queue.SimpleQueue()
        self.replay_resets = None
        self.seed = seed
        self.sample = sample
        self.replay_seed = replay_seed
        self.thread = threading.Thread(target=self.run, daemon=True)

        logging.info("Checkpoint: %s steps.", f"{step:,}")

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stopped.set()

        with self.condition:
            self.condition.notify_all()

        if self.thread.ident is not None:
            self.thread.join()
        else:
            self.env.close()

    def request_reset(self, mode: str) -> None:
        if mode not in ("replay", "kickoff"):
            raise ValueError(f"unknown reset mode: {mode}")

        self.requests.put(mode)

    def request_reload(self) -> None:
        self.requests.put("checkpoint")

    def reload_checkpoint(self) -> torch.Tensor:
        """Load the latest policy before swapping it in."""
        checkpoint = self.checkpoint.parent / "latest.pt"
        env, agent, config, step = load_policy(checkpoint, self.seed)

        try:
            if self.replay_dir is not None:
                config["replay_dir"] = self.replay_dir

            observation = plain_tensor(env.reset())
            replay_resets = self.replay_resets

            if (
                config["team_size"] != self.config["team_size"]
                or Path(config["replay_dir"]).resolve()
                != Path(self.config["replay_dir"]).resolve()
            ):
                replay_resets = None
        except BaseException:
            env.close()

            raise

        previous_env = self.env

        self.env, self.agent, self.config = env, agent, config
        self.checkpoint_step = step
        self.replay_resets = replay_resets

        previous_env.close()

        logging.info("Reloaded checkpoint: %s steps (%s).", f"{step:,}", checkpoint)

        return observation

    def reset(self, mode: str) -> torch.Tensor:
        state = None

        if mode == "replay":
            if self.replay_resets is None:
                self.replay_resets = ReplayResetSampler(
                    self.config["replay_dir"],
                    self.config["team_size"],
                    seed=self.replay_seed
                )

            state = self.replay_resets.sample(self.env.device)

        provider = self.env.reset_state_provider

        def replay_state(mask: torch.Tensor) -> CARLResetState:
            if self.env.masked_resets:
                # The viewer has one simulation, so the sampled row is also
                # the full batch required by CARL's masked-reset API.
                return replace(state, simulation_indices=None, reset_mask=mask)

            return state

        try:
            self.env.reset_state_provider = replay_state if state is not None else None

            return plain_tensor(self.env.reset())
        finally:
            self.env.reset_state_provider = provider

    @torch.inference_mode()
    def run(self) -> None:
        try:
            observation = plain_tensor(self.env.reset())
            timestep = self.config["frameskip"] / 120.0
            deadline = time.monotonic()
            episode = 0

            while not self.stopped.is_set():
                try:
                    mode = self.requests.get_nowait()
                except queue.Empty:
                    mode = None

                if mode is not None:
                    try:
                        observation = (
                            self.reload_checkpoint() if mode == "checkpoint"
                            else self.reset(mode)
                        )
                    except Exception:
                        logging.exception("Could not apply viewer command: %s", mode)
                    else:
                        episode += 1
                        timestep = self.config["frameskip"] / 120.0
                        deadline = time.monotonic()

                actions = self.agent.get_action(observation, sample=self.sample)

                ball, cars, pads = snapshot(self.env)

                data = json.dumps(
                    {
                        "episode":         episode,
                        "checkpoint_step": self.checkpoint_step,
                        "ball":            asdict(ball),
                        "cars":            [asdict(car) for car in cars],
                        "pads":            pads,
                        "controls":        control_values(
                            actions,
                            self.env.discrete_actions
                        )
                    },
                    allow_nan=False,
                    separators=(",", ":")
                )

                with self.condition:
                    self.data = data
                    self.revision += 1
                    self.condition.notify_all()

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
