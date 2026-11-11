from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from torch import nn

from imitation.models import build_discriminator
from training.checkpoints import load_checkpoint, save_checkpoint
from training.models import Agent, critic_hidden_size, critic_layers, policy_window_length
from training.ppo import ppo_loss
from training.test_policy_history import Codec, arguments


def build_agent(args):
    return Agent(6, [2, 3], Codec(), args.hidden_size, args.layers,
                 policy_window_length=policy_window_length(args),
                 critic_hidden_size=critic_hidden_size(args), critic_layers=critic_layers(args),
                 gru_policy=args.gru_policy, gru_critic=args.gru_critic,
                 mlp_policy=args.mlp_policy, mlp_critic=args.mlp_critic)


def linear_layers(module):
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


@pytest.mark.parametrize("flags", [[], ["--mlp-policy", "--mlp-critic"], ["--gru-policy", "--gru-critic"]])
def test_critic_controls_inherit_shared_defaults_and_overrides(flags):
    for width, depth in ((384, 3), (12, 2)):
        args = arguments(*flags, "--hidden-size", str(width), "--layers", str(depth))
        assert args.critic_hidden_size == critic_hidden_size(args) == width
        assert args.critic_layers == critic_layers(args) == depth


def test_critic_hidden_alias_preserves_the_saved_configuration_field():
    short = arguments("--critic-hidden", "12", "--critic-layers", "4")
    long = arguments("--critic-hidden-size", "12", "--critic-layers", "4")
    assert vars(short) == vars(long)
    assert short.critic_hidden_size == 12 and short.critic_layers == 4
    assert not hasattr(short, "critic_hidden")


@pytest.mark.parametrize("flags,message", [(["--critic-hidden", "0"], "hidden sizes"),
                                           (["--critic-hidden", "-1"], "hidden sizes"),
                                           (["--critic-layers", "-1"], "critic layers")])
def test_invalid_critic_dimensions_are_rejected(flags, message):
    with pytest.raises(ValueError, match=message):
        arguments(*flags)


@pytest.mark.parametrize("depth", [-1, -3])
def test_agent_rejects_negative_critic_depth(depth):
    with pytest.raises(ValueError, match="critic layers"):
        Agent(6, [2, 3], Codec(), 8, 1, critic_layers=depth)


@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("actor_kind", ["state", "window", "gru"])
@pytest.mark.parametrize("depth", [0, 1, 4])
def test_critic_depth_and_width_leave_actor_parameters_and_history_unchanged(windowed, actor_kind, depth):
    flags = (["--mlp-critic"] if windowed else [])
    if actor_kind != "state":
        flags += ["--mlp-policy" if actor_kind == "window" else "--gru-policy"]
    args = arguments(*flags, "--layers", "2", "--policy-window-length", "4")
    torch.manual_seed(45)
    baseline = build_agent(args)
    overridden = arguments(*flags, "--layers", "2", "--policy-window-length", "4",
                           "--critic-hidden", "12", "--critic-layers", str(depth))
    torch.manual_seed(45)
    agent = build_agent(overridden)
    for name, value in baseline.actor.state_dict().items():
        torch.testing.assert_close(value, agent.actor.state_dict()[name], rtol=0, atol=0)
    assert agent.policy_window_length == baseline.policy_window_length == 4
    assert agent.critic_hidden_size == 12 and agent.critic_layers == depth
    layers = linear_layers(agent.critic)
    assert len(layers) == depth + 1
    assert [layer.out_features for layer in layers] == [12] * depth + [1]
    assert layers[0].in_features == (28 if windowed else 6)
    observations = torch.randn(4, 4, 6)
    observations[0, :2] = float("nan")
    states = observations[:, -1]
    actions = torch.zeros(4, 2, dtype=torch.long)
    loss, _ = ppo_loss(agent, states, actions, torch.zeros(4), torch.randn(4), torch.randn(4),
                       torch.zeros(4), .2, .01, .5, False, observations)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in agent.parameters())


def test_critic_controls_do_not_change_discriminator_configuration_or_parameters():
    args = arguments("--critic-hidden", "12", "--critic-layers", "4")
    states = []
    for config in (arguments(), args):
        torch.manual_seed(42)
        states.append(build_discriminator(config, 2).state_dict())
    for name, value in states[0].items():
        torch.testing.assert_close(value, states[1][name], rtol=0, atol=0)
    baseline = arguments()
    for name in vars(args):
        if name.startswith("discriminator_"):
            assert getattr(args, name) == getattr(baseline, name)


def test_gru_critic_width_is_independent_but_depth_remains_single_layer():
    states = []
    for depth in (1, 4):
        args = arguments("--gru-policy", "--gru-critic", "--critic-hidden", "12", "--critic-layers", str(depth))
        torch.manual_seed(21)
        agent = build_agent(args)
        assert agent.actor.gru.hidden_size == 8 and agent.critic.gru.hidden_size == 12
        assert agent.actor.gru.num_layers == agent.critic.gru.num_layers == 1
        states.append(agent.state_dict())
    for name, value in states[0].items():
        torch.testing.assert_close(value, states[1][name], rtol=0, atol=0)


@pytest.mark.parametrize("config,expected", [({"layers": 3}, 3), ({"layers": 2, "critic_layers": None}, 2),
                                            ({"layers": 3, "critic_layers": 0}, 0),
                                            ({"layers": 3, "critic_layers": 4}, 4)])
def test_legacy_and_explicit_critic_depths_are_resolved_from_dicts_and_namespaces(config, expected):
    assert critic_layers(config) == critic_layers(SimpleNamespace(**config)) == expected


@pytest.mark.parametrize("flags", [[], ["--mlp-critic"]])
@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_rejects_changed_critic_depth_with_a_useful_error(tmp_path, flags, legacy):
    old = arguments(*flags, "--layers", "2")
    if legacy:
        del old.critic_layers
    new = arguments(*flags, "--layers", "2", "--critic-layers", "4")
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(old)}, path)
    with pytest.raises(ValueError, match="critic layers differ.*--critic-layers 2"):
        load_checkpoint(path, "gaifo", new, None, None)


@pytest.mark.parametrize("algorithm", ["gaifo", "dac"])
@pytest.mark.parametrize("windowed", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_and_viewer_restore_independent_or_legacy_critic_depth(tmp_path, monkeypatch,
                                                                        algorithm, windowed, legacy):
    from vis import playback

    flags = ["--mlp-policy", "--mlp-critic"] if windowed else []
    args = arguments(*flags, "--layers", "2", "--policy-window-length", "4", "--critic-hidden", "12",
                     "--critic-layers", "2" if legacy else "4")
    agent = build_agent(args)
    optimizer = torch.optim.Adam(agent.parameters())
    observations = torch.randn(4, 4, 6) if windowed else torch.randn(4, 6)
    agent.get_value(observations).square().mean().backward()
    optimizer.step()
    if legacy:
        del args.critic_layers
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, algorithm, args, agent, optimizer, 10)
    resumed_args = arguments(*flags, "--layers", "2", "--policy-window-length", "4", "--critic-hidden-size", "12",
                             "--critic-layers", "2" if legacy else "4")
    restored = build_agent(resumed_args)
    restored_optimizer = torch.optim.Adam(restored.parameters())
    load_checkpoint(path, algorithm, resumed_args, restored, restored_optimizer)
    for name, value in agent.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, restored_optimizer.state_dict()["state"][key][name])
    env = SimpleNamespace(single_observation_space=Box(-1., 1., shape=(6,)),
                          single_action_space=SimpleNamespace(nvec=np.array([2, 3])), action_codec=Codec(),
                          device=torch.device("cpu"), close=lambda: None)
    monkeypatch.setattr(playback, "build_env", lambda args: env)
    _, viewer_model, _, step = playback.load_policy(path, 0)
    assert step == 10
    assert viewer_model.critic_hidden_size == 12 and viewer_model.critic_layers == (2 if legacy else 4)
    assert len(linear_layers(viewer_model.actor)) == 3
    assert len(linear_layers(viewer_model.critic)) == (3 if legacy else 5)
    torch.testing.assert_close(viewer_model.get_action(observations, sample=False),
                               agent.get_action(observations, sample=False))


def test_gru_checkpoint_can_resume_when_only_ignored_mlp_depth_changes(tmp_path):
    old = arguments("--gru-critic", "--critic-layers", "1")
    new = arguments("--gru-critic", "--critic-layers", "4")
    agent = build_agent(old)
    optimizer = torch.optim.Adam(agent.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, "gaifo", old, agent, optimizer, 10)
    load_checkpoint(path, "gaifo", new, agent, optimizer)


@pytest.mark.parametrize("flags", [[], ["--mlp-policy", "--mlp-critic"], ["--gru-policy", "--gru-critic"]])
def test_omitted_critic_depth_preserves_existing_model_parameters(flags):
    args = arguments(*flags)
    torch.manual_seed(9)
    explicit = build_agent(args)
    options = dict(gru_policy=args.gru_policy, gru_critic=args.gru_critic,
                   mlp_policy=args.mlp_policy, mlp_critic=args.mlp_critic)
    torch.manual_seed(9)
    inherited = Agent(6, [2, 3], Codec(), 8, 1, policy_window_length=16, **options)
    for name, value in explicit.state_dict().items():
        torch.testing.assert_close(value, inherited.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("windowed", [False, True])
def test_training_setup_passes_critic_only_overrides_to_agent(monkeypatch, windowed):
    from training import setup

    flags = ["--mlp-policy", "--mlp-critic"] if windowed else []
    args = arguments(*flags, "--critic-hidden", "12", "--critic-layers", "4")
    env = SimpleNamespace(single_observation_space=Box(-1., 1., shape=(6,)),
                          single_action_space=SimpleNamespace(nvec=np.array([2, 3])), action_codec=Codec(),
                          device=torch.device("cpu"), n_envs=2, close=lambda: None)
    monkeypatch.setattr(setup, "build_env", lambda *args: env)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _, agent, _, _ = setup.prepare(args)
    assert agent.critic_hidden_size == 12 and agent.critic_layers == 4
    assert len(linear_layers(agent.actor)) == 2 and len(linear_layers(agent.critic)) == 5
