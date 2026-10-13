"""Check the RLBot adapter against the GAIFO training policy and native CARL."""

import argparse
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as Obj
from unittest.mock import patch

import numpy as np
import torch as th
from carl.gymnasium import CARLTorchVectorEnv
from carl.gymnasium.action import ACTION_NVECS
from gymnasium.spaces import Box, MultiDiscrete

from action_codec import GroundAerialActionCodec
from gaifo import (
    GAIFO_ARCHITECTURE, GAIFO_GRU_ARCHITECTURE,
    GAIFO_TEAM_ARCHITECTURE, GAIFO_TEAM_GRU_ARCHITECTURE,
    build_policy,
)
from replay_layout import team_live_observation_size
from rlbot_bot.export_policy import export_policy
from rlbot_bot.observation import BOOST_PAD_POSITIONS, CARLObservationBuilder
from rlbot_bot.policy import decode_action, load_policy_checkpoint, observation_size


class PolicyEnv:
    device = th.device("cpu")
    action_codec = GroundAerialActionCodec()
    single_action_space = MultiDiscrete(ACTION_NVECS)

    def __init__(self, team_size: int):
        self.single_observation_space = Box(
            -np.inf, np.inf, shape=(team_live_observation_size(team_size),), dtype=np.float32,
        )


def rotation_from_basis(forward: np.ndarray, up: np.ndarray):
    yaw = math.atan2(forward[1], forward[0])
    pitch = math.asin(float(np.clip(forward[2], -1.0, 1.0)))
    base_right = np.asarray((-math.sin(yaw), math.cos(yaw), 0.0))
    base_up = np.asarray((
        -math.cos(yaw) * math.sin(pitch), -math.sin(yaw) * math.sin(pitch),
        math.cos(pitch),
    ))
    roll = math.atan2(np.dot(up, base_right), np.dot(up, base_up))
    return Obj(pitch=pitch, yaw=yaw, roll=roll)


def packet_from_carl(raw: np.ndarray, team_size: int):
    def vec(values):
        return Obj(x=values[0], y=values[1], z=values[2])

    ball = Obj(physics=Obj(
        location=vec(raw[:3]), velocity=vec(raw[3:6]),
        angular_velocity=vec(raw[6:9]),
    ))
    cars = raw[9:9 + 22 * 2 * team_size].reshape(2 * team_size, 22)
    players = []
    for index, car in enumerate(cars):
        players.append(Obj(
            team=index // team_size,
            physics=Obj(
                location=vec(car[:3]), velocity=vec(car[3:6]),
                angular_velocity=vec(car[6:9]),
                rotation=rotation_from_basis(car[9:12], car[12:15]),
            ),
            boost=car[15], air_state="ground" if car[16] else "air",
            demolished_timeout=1 if car[17] else -1,
            has_dodged=bool(car[18]), has_double_jumped=bool(car[19]),
            has_jumped=not bool(car[16]), dodge_timeout=1.25,
            last_input=Obj(boost=bool(car[20])),
        ))
    # RLBot field/packet pad order is independent of CARL's storage order.
    shuffled = list(reversed(range(34)))
    pads = raw[9 + 22 * 2 * team_size:]
    field = Obj(boost_pads=[Obj(location=vec(BOOST_PAD_POSITIONS[i])) for i in shuffled])
    packet = Obj(
        balls=[ball], players=players,
        boost_pads=[Obj(is_active=bool(pads[i])) for i in shuffled],
    )
    return field, packet


class RLBotGAIFOTests(unittest.TestCase):
    def test_policy_checkpoint_logits_actions_and_state_match_gaifo(self):
        for team_size, gru, layers in (
            (1, False, 1), (1, False, 3), (1, True, 2),
            (2, False, 2), (3, True, 3),
        ):
            with self.subTest(team_size=team_size, gru=gru, layers=layers):
                th.manual_seed(13)
                env = PolicyEnv(team_size)
                reference = build_policy(env, argparse.Namespace(
                    policy_hidden=24, policy_layers=layers, gru=gru,
                ))
                architecture = (
                    GAIFO_GRU_ARCHITECTURE if gru else GAIFO_ARCHITECTURE
                ) if team_size == 1 else (
                    GAIFO_TEAM_GRU_ARCHITECTURE if gru else GAIFO_TEAM_ARCHITECTURE
                )
                with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
                    path = Path(directory) / "gaifo_000000000001.pt"
                    th.save({
                        "config": {
                            "architecture": architecture, "team_size": team_size,
                            "gru": gru, "frameskip": 4, "policy_hidden": 24,
                            "policy_layers": layers,
                        },
                        "policy": reference.state_dict(),
                        "critic": {"unused": th.ones(1)},
                    }, path)
                    policy = load_policy_checkpoint(path)
                self.assertEqual(policy.config.observation_size, team_live_observation_size(team_size))
                self.assertEqual(policy.config.observation_size, observation_size(team_size))
                observation = th.randn(2, policy.config.observation_size)
                observation[:, 24] = th.tensor([100, 0])
                observation[:, 25] = th.tensor([1, 0])
                observation[:, -2:] = th.tensor([[1, 1.25], [1, 0]])
                state = reference.initial_state(2)
                native_state = policy.initial_state(2)
                for _ in range(2):
                    with th.inference_mode():
                        features, next_state = reference.body_features(observation, state)
                        expected_logits = reference.head(features)
                        expected_action = reference.act(
                            observation, state, deterministic=True,
                        ).action
                        actual_logits, native_next = policy(observation, native_state)
                        action, _ = policy.act(observation, native_state)
                    th.testing.assert_close(actual_logits, expected_logits)
                    th.testing.assert_close(action, expected_action)
                    if gru:
                        th.testing.assert_close(native_next.transpose(0, 1), next_state)
                    else:
                        self.assertIsNone(native_next)
                    state, native_state = next_state, native_next

    def test_checkpoint_rejects_incompatible_observations(self):
        env = PolicyEnv(1)
        reference = build_policy(env, argparse.Namespace(
            policy_hidden=16, policy_layers=2, gru=False,
        ))
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "gaifo.pt"
            config = {
                "architecture": GAIFO_ARCHITECTURE, "team_size": 1,
                "gru": False, "policy_hidden": 16, "policy_layers": 2,
                "frameskip": 4,
            }
            state = reference.state_dict()
            state["foot.model.0.weight"] = th.zeros(16, 137)
            th.save({"config": config, "policy": state}, path)
            with self.assertRaisesRegex(ValueError, "needs 137 features"):
                load_policy_checkpoint(path)
            config["architecture"] = "unknown"
            th.save({"config": config, "policy": state}, path)
            with self.assertRaisesRegex(ValueError, "unsupported GAIFO architecture"):
                load_policy_checkpoint(path)

    def test_exported_checkpoint_runs_the_same_policy_without_training_state(self):
        reference = build_policy(PolicyEnv(1), argparse.Namespace(
            policy_hidden=24, policy_layers=3, gru=False,
        ))
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            source = Path(directory) / "gaifo_000000000100.pt"
            exported = Path(directory) / "policy_latest.pt"
            th.save({
                "config": {
                    "architecture": GAIFO_ARCHITECTURE, "gru": False,
                    "team_size": 1, "frameskip": 4,
                    "policy_hidden": 24, "policy_layers": 3,
                },
                "policy": reference.state_dict(),
                "discriminator": {"unused": th.ones(512)},
            }, source)
            export_policy(source, exported)
            compact = th.load(exported, map_location="cpu", weights_only=True)
            self.assertEqual(set(compact), {"config", "policy"})
            policy = load_policy_checkpoint(exported)
            observation = th.randn(1, 139)
            observation[:, 25] = 1
            with th.inference_mode():
                th.testing.assert_close(
                    policy(observation)[0], reference.head(reference.body(reference.foot(observation))),
                )

    def test_dodge_window_and_action_decoding(self):
        field = Obj(boost_pads=[Obj(location=Obj(x=x, y=y, z=z))
                                for x, y, z in BOOST_PAD_POSITIONS])
        builder = CARLObservationBuilder(1, field, "ground")
        car = Obj(air_state="air", has_jumped=True, has_dodged=False,
                  has_double_jumped=False, dodge_timeout=0.7)
        self.assertEqual(builder._dodge(car), (1.0, 0.7))
        car.dodge_timeout = -1
        self.assertEqual(builder._dodge(car), (0.0, 0.0))
        car.has_jumped = False
        self.assertEqual(builder._dodge(car), (1.0, 1.25))
        car.has_dodged = True
        self.assertEqual(builder._dodge(car), (0.0, 0.0))
        self.assertEqual(decode_action([2, 1, 2, 1, 1, 1, 1]), {
            "throttle": 1.0, "steer": 1.0, "yaw": 1.0,
            "pitch": -1.0, "roll": -1.0,
            "jump": True, "boost": True, "handbrake": True,
        })

    @unittest.skipUnless(th.cuda.is_available(), "native CARL requires CUDA")
    def test_packet_observations_match_native_carl_for_both_teams(self):
        for team_size in (1, 2, 3):
            with self.subTest(team_size=team_size):
                env = CARLTorchVectorEnv(
                    n_sim=1, n_blue=team_size, n_orange=team_size,
                    seed=3, frameskip=4, normalize=True, discrete_actions=True,
                )
                try:
                    reference = env.reset()
                    action = th.tensor(
                        [[2, 0, 2, 0, 1, 0, 0]] * (2 * team_size),
                        dtype=th.int64, device=env.device,
                    )
                    for moving in (False, True):
                        if moving:
                            for _ in range(4):
                                reference, _, _, _, _ = env.step(action)
                        raw = th.from_dlpack(env._env.get_state()).cpu().numpy()[0]
                        field, packet = packet_from_carl(raw, team_size)
                        builder = CARLObservationBuilder(team_size, field, "ground")
                        for index in range(2 * team_size):
                            with self.subTest(index=index, moving=moving):
                                actual = builder.build(packet, index)
                                np.testing.assert_allclose(
                                    actual, reference[index].cpu().numpy(), atol=1e-5, rtol=1e-5,
                                )
                finally:
                    env.close()

    @unittest.skipUnless(th.cuda.is_available(), "native CARL requires CUDA")
    def test_packet_observations_match_native_carl_during_jumps_and_flips(self):
        env = CARLTorchVectorEnv(
            n_sim=1, n_blue=1, n_orange=1, seed=7, frameskip=4,
            normalize=True, discrete_actions=True,
        )
        try:
            env.reset()
            for action in (
                (0, 0, 0, 0, 0, 0, 1),
                (0, 0, 2, 0, 1, 0, 0),
                (2, 1, 2, 0, 1, 2, 0),
                (2, 1, 2, 0, 1, 2, 1),
                (0, 0, 2, 0, 0, 2, 0),
            ):
                actions = th.tensor([action] * 2, dtype=th.int32, device=env.device)
                reference, _, _, _, _ = env.step(actions)
                raw = th.from_dlpack(env._env.get_state()).cpu().numpy()[0]
                field, packet = packet_from_carl(raw, 1)
                # Provide the equivalent native timer: Rocket League's timer
                # can include up to 0.2 seconds of additional held-jump time.
                for i, player in enumerate(packet.players):
                    player.dodge_timeout = float(reference[i, -1])
                builder = CARLObservationBuilder(1, field, "ground")
                for i in range(2):
                    np.testing.assert_allclose(
                        builder.build(packet, i), reference[i].cpu().numpy(),
                        atol=1e-5, rtol=1e-5,
                    )
        finally:
            env.close()

    def test_rlbot_v5_runner_loads_checkpoint_and_uses_saved_cadence(self):
        try:
            from rlbot import flat
            from rlbot_bot.bot import GoddardBot
        except ImportError:
            self.skipTest("RLBot v5 is installed only in the bot environment")

        reference = build_policy(PolicyEnv(1), argparse.Namespace(
            policy_hidden=16, policy_layers=3, gru=False,
        ))
        with tempfile.TemporaryDirectory(dir="/tmp/opencode") as directory:
            path = Path(directory) / "gaifo_000000000001.pt"
            th.save({
                "config": {
                    "architecture": GAIFO_ARCHITECTURE, "team_size": 1,
                    "gru": False, "policy_hidden": 16, "policy_layers": 3,
                    "frameskip": 4,
                },
                "policy": reference.state_dict(),
            }, path)
            field = flat.FieldInfo(boost_pads=[
                flat.BoostPad(location=flat.Vector3(*(float(v) for v in position)))
                for position in BOOST_PAD_POSITIONS
            ])
            players = [flat.PlayerInfo(
                team=index,
                physics=flat.Physics(
                    location=flat.Vector3(0, -4608 if index == 0 else 4608, 17),
                    rotation=flat.Rotator(yaw=math.pi / 2 if index == 0 else -math.pi / 2),
                ),
                boost=33, air_state=flat.AirState.OnGround, demolished_timeout=-1,
            ) for index in range(2)]
            ball = flat.BallInfo(physics=flat.Physics(location=flat.Vector3(0, 0, 93)))
            pads = [flat.BoostPadState(is_active=True) for _ in BOOST_PAD_POSITIONS]

            def packet(frame, phase):
                return flat.GamePacket(
                    players=players, balls=[ball], boost_pads=pads,
                    match_info=flat.MatchInfo(
                        seconds_elapsed=frame / 120, frame_num=frame, match_phase=phase,
                    ),
                )

            bot = GoddardBot.__new__(GoddardBot)
            bot.field_info = field
            bot.team, bot.index = 0, 0
            with (patch.dict("os.environ", {"GODDARD_CHECKPOINT": str(path)}),
                  patch("builtins.print") as output):
                bot.initialize()
                np.testing.assert_array_equal(
                    bot.builder.build(packet(0, flat.MatchPhase.Kickoff), 0)[-2:],
                    (1.0, 1.25),
                )
                first = bot.get_output(packet(0, flat.MatchPhase.Kickoff))
                self.assertIsInstance(first, flat.ControllerState)
                self.assertEqual(bot.last_inference_frame, 0)
                self.assertIs(bot.get_output(packet(3, flat.MatchPhase.Active)), first)
                self.assertIsNot(bot.get_output(packet(4, flat.MatchPhase.Active)), first)
                self.assertEqual(bot.last_inference_frame, 4)
                self.assertIsInstance(
                    bot.get_output(packet(5, flat.MatchPhase.Countdown)), flat.ControllerState,
                )
                self.assertLess(bot.last_inference_frame, 0)
            entries = [json.loads(call.args[0].removeprefix("GODDARD_DIAG "))
                       for call in output.call_args_list]
            self.assertEqual(entries[0]["type"], "startup")
            self.assertEqual(entries[0]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(entries[0]["frameskip"], 4)
            decisions = [entry for entry in entries if entry["type"] == "decision"]
            self.assertEqual([entry["frame"] for entry in decisions], [0, 4])
            self.assertEqual(decisions[0]["obs_ground"], 1.0)
            self.assertEqual(len(decisions[0]["observation"]), 139)
            self.assertEqual(len(decisions[0]["boost_pads_active"]), 34)
            self.assertEqual(decisions[0]["controls"]["throttle"], first.throttle)
            self.assertEqual(decisions[1]["frame_gap"], 4)

            quiet_bot = GoddardBot.__new__(GoddardBot)
            quiet_bot.field_info = field
            quiet_bot.team, quiet_bot.index = 0, 0
            with (patch.dict("os.environ", {
                      "GODDARD_CHECKPOINT": str(path), "GODDARD_DIAGNOSTICS": "0",
                  }), patch("builtins.print") as quiet_output):
                quiet_bot.initialize()
                quiet_bot.get_output(packet(0, flat.MatchPhase.Kickoff))
            quiet_output.assert_not_called()


if __name__ == "__main__":
    unittest.main()
