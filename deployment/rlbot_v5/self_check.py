"""Standalone packaging/inference/adapter check; does not launch Rocket League."""

import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import runtime
from bot import GoddardBot
from observations import ControllerTracker, ObservationBuilder, floor_contacts
from policy import Policy, decode_actions
from rlbot import flat
from rlbot.config import load_match_config, load_player_config
from schema import DEFAULT_SCHEMA, LEGACY_SCHEMA


def replace(
    value,
    **changes,
):
    fields = {name: getattr(value, name) for name in value.__match_args__}
    return type(value)(**(fields | changes))


def make_packet(
    mirror=False,
):
    sign = -1 if mirror else 1
    players = []
    for i in range(2):
        controls = flat.ControllerState(throttle=1, yaw=.5, steer=.5)
        players.append(flat.PlayerInfo(
            physics=flat.Physics(
                location=flat.Vector3(sign * (200 + i * 400), sign * (-1200 + i * 2400), 17),
                rotation=flat.Rotator(pitch=.1, yaw=.4 + (math.pi if mirror else 0), roll=.2),
                velocity=flat.Vector3(sign * 100, sign * 300, 0),
                angular_velocity=flat.Vector3(sign * .1, sign * .2, .3),
            ),
            team=(1 - i if mirror else i), boost=50, air_state=flat.AirState.OnGround,
            demolished_timeout=-1, dodge_timeout=-1,
            last_input=controls, dodge_dir=flat.Vector2(),
        ))
    return flat.GamePacket(
        players=players,
        boost_pads=[flat.BoostPadState(is_active=True) for _ in range(34)],
        balls=[flat.BallInfo(physics=flat.Physics(
            location=flat.Vector3(sign * 500, sign * 600, 500),
            rotation=flat.Rotator(), velocity=flat.Vector3(sign * 200, sign * 50, 100),
            angular_velocity=flat.Vector3(sign * .1, sign * .2, .3),
        ))],
        match_info=flat.MatchInfo(
            frame_num=100, world_gravity_z=-650, game_speed=1, match_phase=flat.MatchPhase.Active,
        ), teams=[], tiles=[],
    )


def main():
    runtime.ENABLED = False  # Never mistake synthetic self-check packets for live telemetry.
    root = Path(__file__).resolve().parent
    metadata = json.loads((root / "export.json").read_text())
    for relative, expected in metadata["files"].items():
        if relative == "settings.json":
            continue  # Sampling settings are intentionally user-editable.
        with (root / relative).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"export file changed or damaged: {relative}")
    policy = Policy(root / "actor.npz", sample=False)
    with np.load(root / "reference.npz", allow_pickle=False) as reference:
        actual = policy.logits(reference["observations"])
        np.testing.assert_allclose(actual, reference["logits"], atol=2e-4, rtol=2e-5)
    # Deliberately reverse RLBot pad ordering: the bridge must match by location.
    field = flat.FieldInfo(boost_pads=[
        flat.BoostPad(location=flat.Vector3(*map(float, p))) for p in policy.pad_positions[::-1]
    ], goals=[], tiles=[])
    def make_builder():
        return ObservationBuilder(field, policy.pad_positions,
                                  policy.observation_schema, policy.previous_controls)

    observations = []
    for mirror in (False, True):
        packet = make_packet(mirror)
        builder = make_builder()
        builder.update(packet)
        observation = builder.build(packet, 0)
        assert observation.shape == (policy.observation_size,) and np.isfinite(observation).all()
        if policy.observation_schema == LEGACY_SCHEMA:
            assert np.array_equal(observation[158:162], np.zeros(4))
        following = replace(packet, match_info=replace(packet.match_info, frame_num=101))
        builder.update(following)
        if policy.observation_schema == LEGACY_SCHEMA:
            np.testing.assert_array_equal(builder.build(following, 0)[158:162], floor_contacts(following, 0))
        observations.append(observation)
        controls = decode_actions(policy.actions(observation))
        assert all(-1 <= float(value) <= 1 for value in controls.values())
    np.testing.assert_allclose(observations[0], observations[1], atol=1e-6)
    builder = make_builder()
    packet = make_packet()
    packet = replace(packet, players=[
        replace(packet.players[0], air_state=flat.AirState.InAir, boost=0, has_double_jumped=True),
        packet.players[1],
    ])
    builder.update(packet)
    observation = builder.build(packet, 0)
    original_logits = policy.logits
    forced = np.zeros(18, dtype=np.float32)
    forced[[10, 12, 17]] = 100
    policy.logits = lambda value: forced
    actions = policy.actions(observation)
    assert actions[3] == actions[4] == actions[6] == 0
    policy.logits = original_logits
    # Real V5 initial-jump sentinel: holding jump must remain legal. Also
    # preserve expired, spent and demo masks; fix the actor inputs, not just
    # the post-network mask.
    base = make_packet()
    jumping = replace(base.players[0], air_state=flat.AirState.Jumping,
                      has_jumped=True, dodge_timeout=-1,
                      last_input=flat.ControllerState(jump=True))
    forced = np.zeros(18, dtype=np.float32)
    forced[17] = 100
    policy.logits = lambda value: forced
    cases = [
        (jumping, True, 1.25, True),
        (replace(jumping, air_state=flat.AirState.InAir, dodge_timeout=.7), True, .7, True),
        (replace(jumping, air_state=flat.AirState.InAir), False, 0, False),
        (replace(jumping, air_state=flat.AirState.InAir, dodge_timeout=0), False, 0, False),
        (replace(jumping, has_dodged=True), False, 0, False),
        (replace(jumping, has_double_jumped=True), False, 0, False),
        (replace(jumping, demolished_timeout=2), False, 0, False),
        (replace(jumping, air_state=flat.AirState.InAir, has_jumped=False), True, 1.25, True),
        (replace(jumping, air_state=flat.AirState.InAir, dodge_timeout=1/240), True, 1/240, False),
    ]
    try:
        for car, available, remaining, jump_allowed in cases:
            packet = replace(base, players=[car, base.players[1]])
            builder = make_builder()
            builder.update(packet)
            observation = builder.build(packet, 0)
            np.testing.assert_allclose(observation[137:139], [available, remaining])
            assert bool(policy.actions(observation)[6]) == jump_allowed
    finally:
        policy.logits = original_logits
    # V3 state and optional controls come directly from the packet. Toggling
    # controls must preserve all 144 default values, including the dodge mask.
    for mirror in (False, True):
        base = make_packet(mirror)
        for air, jumped, dodged, elapsed, demo in (
            (flat.AirState.OnGround, False, False, -1., -1.),
            (flat.AirState.Jumping, True, False, -1., -1.),
            (flat.AirState.Dodging, True, True, .15, -1.),
            (flat.AirState.InAir, True, True, .6, -1.),
            (flat.AirState.InAir, False, False, -1., 2.),
        ):
            car = replace(base.players[0], air_state=air, has_jumped=jumped,
                          has_dodged=dodged, dodge_elapsed=elapsed, demolished_timeout=demo)
            packet = replace(base, players=[car, base.players[1]])
            values = []
            for controls in (False, True):
                builder = ObservationBuilder(field, policy.pad_positions, DEFAULT_SCHEMA, controls)
                builder.update(packet)
                value = builder.build(packet, 0)
                assert value.shape == (144 + 8 * controls,)
                np.testing.assert_allclose(value[139:144], [jumped, air == flat.AirState.Jumping,
                    air == flat.AirState.Dodging, max(0, elapsed) if dodged else 0, max(0, demo)])
                values.append(value)
            np.testing.assert_array_equal(values[0], values[1][:144])
            np.testing.assert_array_equal(values[1][144:], [1, .5, .5, 0, 0, 0, 0, 0])
    # Open flat floor: recover front/rear/side-only contacts without changing
    # V5's grounded flag. Wheel order must be identical on both teams.
    for mirror in (False, True):
        base = make_packet(mirror)
        for pitch, roll, height, expected in [
            (0, 0, 17, [1, 1, 1, 1]),
            (.35, 0, 35, [0, 0, 1, 1]),
            (-.35, 0, 35, [1, 1, 0, 0]),
            (0, .35, 35, [1, 0, 1, 0]),
            (0, 0, 100, [0, 0, 0, 0]),
        ]:
            car = base.players[0]
            car = replace(car, air_state=flat.AirState.InAir, physics=replace(
                car.physics, location=replace(car.physics.location, z=height),
                rotation=flat.Rotator(pitch=pitch, roll=roll, yaw=math.pi if mirror else 0),
                velocity=flat.Vector3(), angular_velocity=flat.Vector3()))
            packet = replace(base, players=[car, base.players[1]])
            np.testing.assert_array_equal(floor_contacts(packet, 0), expected)
            builder = ObservationBuilder(field, policy.pad_positions, LEGACY_SCHEMA)
            builder.update(packet)
            assert not builder.build(packet, 0)[158:162].any()
            builder.update(packet)  # Duplicate frame is not a second physics step.
            assert not builder.build(packet, 0)[158:162].any()
            packet = replace(packet, match_info=replace(packet.match_info, frame_num=101))
            builder.update(packet)
            observation = builder.build(packet, 0)
            np.testing.assert_array_equal(observation[158:162], expected)
            assert observation[25] == 0
            # Nearby balls/cars, ramps, and inverted poses must use the fallback.
            for unsafe in [
                replace(car, physics=replace(car.physics, location=flat.Vector3(3700, 0, 17))),
                replace(car, physics=replace(car.physics, location=flat.Vector3(0, 4700, 17))),
                replace(car, physics=replace(car.physics, rotation=flat.Rotator(roll=math.pi))),
                replace(car, physics=replace(car.physics, location=base.balls[0].physics.location)),
                replace(car, physics=replace(car.physics, location=base.players[1].physics.location)),
            ]:
                assert floor_contacts(replace(packet, players=[unsafe, base.players[1]]), 0) is None
            demo = replace(packet, players=[replace(car, demolished_timeout=2), base.players[1]])
            builder.update(demo)
            assert not builder.build(demo, 0)[158:162].any()
    # Contacts refer to the pre-integration tick, not the rendered pose.
    base = make_packet()
    car = replace(base.players[0], physics=replace(base.players[0].physics,
                  location=flat.Vector3(200, -1200, 30), rotation=flat.Rotator(),
                  velocity=flat.Vector3(0, 0, -1200), angular_velocity=flat.Vector3()))
    np.testing.assert_array_equal(floor_contacts(replace(base, players=[car, base.players[1]]), 0), np.zeros(4))
    match = load_match_config(root / "match.toml")
    assert len(match.player_configurations) == 2
    load_player_config(root / "bot.toml", 0)
    load_player_config(root / "bot.toml", 1)
    # Test the real Bot.get_output path without opening any network connection.
    bot = GoddardBot.__new__(GoddardBot)
    bot.index, bot.team, bot.field_info = 0, 0, field
    bot.logger = logging.getLogger("goddard.self_check")
    bot.initialize()
    hold = bot.settings["frameskip"]
    count = [0]
    def counted(value):
        count[0] += 1
        return [0] * 7
    bot.policy.actions = counted
    packet = make_packet()
    first = bot.get_output(packet)
    for frame in range(101, 100 + hold):
        packet = replace(packet, match_info=replace(packet.match_info, frame_num=frame))
        assert bot.get_output(packet) is first
    assert count[0] == 1
    packet = replace(packet, match_info=replace(packet.match_info, frame_num=100 + hold))
    bot.get_output(packet)
    assert count[0] == 2
    packet = replace(packet, match_info=replace(packet.match_info, match_phase=flat.MatchPhase.GoalScored))
    assert not bot.get_output(packet).jump
    assert bot.next_decision is None
    packet = replace(packet, match_info=replace(packet.match_info, match_phase=flat.MatchPhase.Active))
    packet = replace(packet, match_info=replace(packet.match_info, frame_num=200))
    bot.get_output(packet)
    assert count[0] == 3
    # Landings keep CARL's last flip torque; reset/demo must forget it, even
    # when V5 still reports the old dodge direction.
    tracker = ControllerTracker()
    car = replace(make_packet().players[0], air_state=flat.AirState.Dodging,
                  has_dodged=True, dodge_dir=flat.Vector2(.6, -.8))
    tracker.update(car, 100)
    np.testing.assert_allclose(tracker.controller_features(car)[15:18], [.8, .6, 0])
    landed = replace(car, air_state=flat.AirState.OnGround, has_dodged=False)
    tracker.update(landed, 108)
    np.testing.assert_allclose(tracker.controller_features(landed)[15:18], [.8, .6, 0])
    tracker.reset()
    tracker.update(landed, 116)
    assert not tracker.controller_features(landed)[15:18].any()
    tracker.update(car, 124)
    demoed = replace(car, demolished_timeout=2)
    tracker.update(demoed, 132)
    assert not tracker.controller_features(demoed)[15:18].any()
    tracker.update(landed, 140)
    assert not tracker.controller_features(landed)[15:18].any()
    # Exercise real scheduling with jitter, duplicate packets and missed slots.
    bot.reset_state()
    count[0] = 0
    expected_deadline = None
    for frame in [100, 103, 107, 109, 109, 112, 116, 125, 125, 126, 149, 150, 157]:
        before = count[0]
        due = expected_deadline is None or frame >= expected_deadline
        packet = replace(make_packet(), match_info=replace(make_packet().match_info, frame_num=frame))
        bot.get_output(packet)
        assert count[0] - before == int(due)
        if due:
            expected_deadline = (frame + hold if expected_deadline is None else
                                 expected_deadline + ((frame - expected_deadline) // hold + 1) * hold)
        assert bot.next_decision == expected_deadline
        assert bot.next_decision > frame and (bot.next_decision - 100) % hold == 0
    for frame in [10, 200]:  # Rewind and long discontinuity re-anchor safely.
        packet = replace(packet, match_info=replace(packet.match_info, frame_num=frame))
        before = count[0]
        bot.get_output(packet)
        assert count[0] == before + 1 and bot.next_decision == frame + hold
    print(f"PASS: actor step {metadata['checkpoint_step']:,}; weight parity, V5 packets, "
          f"{policy.observation_schema}, optional controls, blue/orange observations, "
          f"pad mapping, action masks, {hold}-tick hold, "
          "initial contact reset, phase reset, persistent flip torque, jittered phase-anchored clock, "
          "held-jump sentinel/masks, partial floor contacts and V5 configs. "
          "This self-check does not launch a game.")


if __name__ == "__main__":
    main()
