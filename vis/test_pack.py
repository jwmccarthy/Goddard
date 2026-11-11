import argparse
import importlib.util
import json
import queue
import time
from pathlib import Path

import pytest
import torch

from vis.pack import PackMatch, PreviewProvider, condition_label, load_pack, terminal_snapshot
from vis.playback import snapshot


def test_pack_reset_requests_validate_modes():
    match = PackMatch.__new__(PackMatch)
    match.reset_requests = queue.SimpleQueue()

    for mode in ("pack", "next", "previous"):
        match.request_reset(mode)
        assert match.reset_requests.get_nowait() == mode

    with pytest.raises(ValueError, match="unknown reset"):
        match.request_reset("replay")


def test_condition_labels_are_concise():
    from carl.events import BallTouch, ContactSurface

    state = type("StateCondition", (), {"metric": "ball_height", "minimum": None, "maximum": 300})()
    flip = type("FlipReset", (), {})()

    assert condition_label(state) == "Ball height ≤ 300"
    assert condition_label(flip) == "Flip reset"
    assert condition_label(BallTouch((ContactSurface.FRONT, ContactSurface.ROOF))) == "Ball touch · front / roof"


@pytest.mark.parametrize("payload", [
    {"algorithm": "pack-ppo"},
    {"format": "carl-training-pack-ppo-v1"},
    {"format": "carl-training-pack-ppo-v1", "algorithm": "attention-pack-ppo"},
])
def test_cli_routes_pack_ppo_checkpoint(monkeypatch, tmp_path, payload):
    import sys

    from vis import __main__ as viewer

    checkpoint = tmp_path / "latest.pt"
    checkpoint.touch()
    mesh = tmp_path / "arena.obj"
    mesh.touch()
    calls = []

    class Match:
        def __init__(self, *, seed, checkpoint, sample):
            calls.append(("load", seed, checkpoint, sample))

        def start(self):
            calls.append("start")

        def close(self):
            calls.append("close")

    class Server:
        server_port = 1234

        def __init__(self, address, handler):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def serve_forever(self):
            pass

    monkeypatch.setattr(sys, "argv", ["vis", str(checkpoint), "--mesh", str(mesh), "--seed", "5", "--sample"])
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr("vis.pack.PackMatch", Match)
    monkeypatch.setattr(viewer, "ThreadingHTTPServer", Server)
    viewer.main()

    assert calls == [("load", 5, checkpoint, True), "start", "close"]


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("carl.training") is None,
    reason="requires CUDA and CARL training-pack API",
)
def test_pack_viewer_streams_terminal_and_switches_exercise():
    match = PackMatch(Path(__file__).with_name("example_pack.py"), seed=3)
    match.start()

    try:
        deadline = time.monotonic() + 15
        revision = 0
        outcome = None
        stages = set()

        while time.monotonic() < deadline:
            with match.condition:
                match.condition.wait_for(lambda: match.revision != revision or match.stopped.is_set(), timeout=2)
                revision = match.revision
                data = json.loads(match.data) if match.data else None

            assert not match.stopped.is_set()

            if data:
                stages.add(data["training"]["stage"])

            if data and data["training"]["terminal"]:
                outcome = data
                break

        assert outcome is not None
        assert outcome["training"]["exercise"] == "Falling ball"
        assert outcome["training"]["outcome"] == "success"
        assert [stage["label"] for stage in outcome["training"]["stages"]] == [
            "Ball height ≤ 700", "Ball height ≤ 500", "Ball height ≤ 300",
        ]
        assert {2, 3} <= stages
        assert outcome["cars"][0]["team"] == 0
        assert outcome["ball"]["position"][2] <= 300

        match.request_reset("next")
        deadline = time.monotonic() + 10

        while time.monotonic() < deadline:
            with match.condition:
                match.condition.wait_for(lambda: match.revision != revision or match.stopped.is_set(), timeout=2)
                revision = match.revision
                data = json.loads(match.data) if match.data else None

            assert not match.stopped.is_set()

            if data and data["training"]["exercise"] == "Side wall":
                break
        else:
            pytest.fail("viewer did not switch exercises")

    finally:
        match.close()


@pytest.mark.skipif(
    importlib.util.find_spec("carl.training") is None,
    reason="requires CARL training-pack API",
)
def test_load_pack_requires_export(tmp_path):
    path = tmp_path / "pack.py"
    path.write_text("PACK = None\n")

    with pytest.raises(ValueError, match="TrainingPack named PACK"):
        load_pack(path)


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("carl.training") is None,
    reason="requires CUDA and CARL training-pack API",
)
def test_wall_flip_reset_setup_samples_rising_ball_on_defensive_half():
    pack = load_pack(Path(__file__).with_name("example_pack.py"))
    exercise = pack.exercises[2]
    provider = PreviewProvider(pack, seed=17)
    provider.select(2)
    mask = torch.ones(1, device="cuda", dtype=torch.bool)
    positions = []

    assert exercise.name == "Wall flip reset"
    assert [condition_label(condition) for condition in exercise.fail_on] == ["Car touches ceiling"]
    assert [condition_label(stage.event) for stage in exercise.stages] == [
        "Car leaves surface", "Ball touch · front · speed change ≥ 100", "Ball touch · front · speed change ≥ 100",
        "Ball touch · front · speed change ≥ 100",
        "Flip reset", "Dodge", "Flip reset", "Dodge", "Ball touch", "Goal",
    ]

    for _ in range(3):
        state = provider(mask)
        car = state.cars.ego
        positions.append(car.position[0, 0].item())

        assert -150 <= car.position[0, 0] <= 150
        assert car.boost[0] == 100
        assert -3150 <= car.position[0, 1] <= -3050
        assert -150 <= state.ball.position[0, 0] <= 150
        assert -2450 <= state.ball.position[0, 1] <= -2250
        assert 300 <= state.ball.position[0, 2] <= 500
        assert state.ball.position[0, 1] > car.position[0, 1]
        assert car.forward[0, 1] > 0.99
        assert torch.all(car.velocity == 0)
        assert state.ball.velocity[0, 0] == 0
        assert 600 <= state.ball.velocity[0, 2] <= 900

    assert len(set(positions)) == 3


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("carl.training") is None,
    reason="requires CUDA and CARL training-pack API",
)
def test_pack_checkpoint_restores_actor_and_physical_terminal_frame(tmp_path):
    from training.arguments import add_ppo_arguments
    from training.checkpoints import save_checkpoint
    from training.pack_ppo import prepare_pack_ppo

    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser)
    args = parser.parse_args(["--n-sim", "2", "--hidden-size", "8", "--layers", "1", "--no-bf16"])
    args.stage_reward_base = 1.0
    args.stage_reward_multiplier = 2.0
    args.pack_objective = "wall-flip-reset-v13"
    env, agent, optimizer, _, _ = prepare_pack_ppo(args)
    checkpoint = tmp_path / "latest.pt"

    try:
        expected = agent.actor[0].weight.detach().cpu().clone()
        assert expected.shape[1] == env.single_observation_space.shape[0] + env.runtime.pack.kind.shape[1] + 1
        save_checkpoint(checkpoint, "pack-ppo", args, agent, optimizer, step=100)
    finally:
        env.close()

    match = PackMatch(checkpoint=checkpoint, seed=5)
    match.start()

    try:
        with match.condition:
            match.condition.wait_for(lambda: match.data is not None or match.stopped.is_set(), timeout=15)

        assert not match.stopped.is_set()
        data = json.loads(match.data)
        assert data["training"]["exercise"] == "Wall flip reset"
        assert set(data["controls"][0]) == {"horizontal", "vertical", "throttle", "roll", "slide", "boost", "jump"}
        stages = data["training"]["stages"]
        assert [stage["min_ticks"] for stage in stages] == [stage.min_ticks for stage in match.pack.exercises[0].stages]
        assert all(stages[index]["min_ticks"] > 0 for index in (1, 2, 3))
        assert stages[1]["from_exercise_start"]
        assert not stages[2]["from_exercise_start"]
        assert match.agent.actor[0].weight.shape == expected.shape
        torch.testing.assert_close(match.agent.actor[0].weight.cpu(), expected)
    finally:
        match.close()

    match = PackMatch(checkpoint=checkpoint, seed=5)

    try:
        observation = match.env.reset()
        physical_ball, physical_cars, _ = snapshot(match.env, blue_cars=1)
        terminal_ball, terminal_cars, _ = terminal_snapshot(observation, (), normalized=True)
        torch.testing.assert_close(torch.tensor(terminal_ball.position), torch.tensor(physical_ball.position))
        torch.testing.assert_close(torch.tensor(terminal_cars[0].physics.position),
                                   torch.tensor(physical_cars[0].physics.position))
        assert terminal_cars[0].boost == pytest.approx(physical_cars[0].boost, abs=1e-4)
    finally:
        match.env.close()


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("carl.training") is None,
    reason="requires CUDA and CARL training-pack API",
)
def test_carl_attention_checkpoint_restores_encoder_and_switches_mirrored_exercise(tmp_path):
    from dataclasses import asdict

    import carl

    ppo = pytest.importorskip("examples.training_packs.ppo")
    from examples.training_packs.pack import aerial_pack

    config = ppo.PPOConfig(n_sim=16_384, embedding_size=16, attention_heads=2,
                           attention_layers=1, hidden_size=32, bf16=False, compile=False)
    observation_size = (carl.OBS_BALL + carl.OBS_PER_CAR + carl.OBS_BOOST_PADS
                        + carl.OBS_RELATIVE_EGO_BALL + carl.OBS_RELATIVE_GOALS + carl.OBS_EGO_DODGE)
    agent = ppo.Agent(observation_size, config).eval()
    checkpoint = tmp_path / "attention.pt"
    # No algorithm/observation_size fields: checkpoints saved before the viewer
    # adapter must work without conversion or overwriting a live training run.
    torch.save({
        "format": ppo.CHECKPOINT_FORMAT,
        "feature_dim": ppo.POLICY_FEATURE_DIM,
        "pack_signature": ppo._pack_signature(aerial_pack()),
        "config": asdict(config),
        "agent": agent.state_dict(),
        "steps": 200,
    }, checkpoint)
    match = PackMatch(checkpoint=checkpoint, seed=5)

    try:
        assert match.observations is None  # No legacy stage-one-hot wrapper.
        assert len(match.pack.exercises) == 12
        for name, value in agent.state_dict().items():
            torch.testing.assert_close(match.agent.state_dict()[name].cpu(), value)
        observation = match.env.reset()
        physical_ball, physical_cars, _ = snapshot(match.env, blue_cars=1)
        terminal_ball, terminal_cars, _ = terminal_snapshot(observation, (), normalized=True)
        torch.testing.assert_close(torch.tensor(terminal_ball.position), torch.tensor(physical_ball.position))
        torch.testing.assert_close(torch.tensor(terminal_cars[0].physics.position),
                                   torch.tensor(physical_cars[0].physics.position))
        match.start()
        with match.condition:
            match.condition.wait_for(lambda: match.data is not None or match.stopped.is_set(), timeout=15)
        assert not match.stopped.is_set()
        data = json.loads(match.data)
        assert data["training"]["exercise"] == "Aerial touch"
        assert set(data["controls"][0]) == {"horizontal", "vertical", "throttle", "roll", "slide", "boost", "jump"}
        match.request_reset("next")
        deadline = time.monotonic() + 10
        revision = match.revision
        while time.monotonic() < deadline:
            with match.condition:
                match.condition.wait_for(lambda: match.revision != revision or match.stopped.is_set(), timeout=2)
                revision = match.revision
                data = json.loads(match.data)
            assert not match.stopped.is_set()
            if data["training"]["exercise"] == "Aerial touch (mirrored)":
                break
        else:
            pytest.fail("viewer did not switch to the mirrored exercise")
    finally:
        if match.thread.ident is not None:
            match.close()
        else:
            match.env.close()
