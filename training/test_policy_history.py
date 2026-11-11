import argparse
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from imitation.models import build_discriminator
from imitation.rollout import TransitionRollout
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint, save_checkpoint
from training.models import Agent, critic_hidden_size, policy_window_length
from training.ppo import ppo_loss
from training.rollout import Rollout, collect


def arguments(*flags):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=True)
    add_gaifo_arguments(parser)
    args = parser.parse_args(["--hidden-size", "8", "--layers", "1", "--frame-embedding", "8",
                              "--discriminator-hidden", "8", "--discriminator-window-length", "2",
                              "--discriminator-decision-stride", "3", *flags])
    resolve_gaifo_defaults(args)
    return args


@pytest.mark.parametrize("flags", [[], ["--gru-policy"], ["--gru-critic"], ["--gru-policy", "--gru-critic"]])
def test_shared_policy_history_is_independent_of_discriminator_length_and_stride(flags):
    for length, stride in ((2, 1), (2, 3), (4, 3)):
        args = arguments(*flags, "--discriminator-window-length", str(length),
                         "--discriminator-decision-stride", str(stride))
        assert args.policy_window_length == policy_window_length(args) == 16
        assert args.policy_observation_hz == 30
        assert args.policy_window_span_ticks == (60 if flags else 0)
        assert args.discriminator_observation_hz == 30 / stride
        assert args.discriminator_window_span_ticks == (length - 1) * stride * 4


@pytest.mark.parametrize("value", ["0", "-1"])
def test_policy_window_must_be_positive(value):
    with pytest.raises(ValueError, match="policy window length"):
        arguments("--policy-window-length", value)


def test_long_actor_critic_history_does_not_require_more_expert_context(tmp_path):
    (tmp_path / "selection.json").write_text(json.dumps({"frameskip": 4, "team_size": 1, "history": 4}))
    args = arguments("--gru-policy", "--gru-critic", "--policy-window-length", "64", "--replay-dir", str(tmp_path))
    assert args.policy_window_length == 64
    assert args.discriminator_window_length == 2 and args.discriminator_decision_stride == 3
    assert args.policy_window_span_ticks == 252


def test_legacy_history_and_critic_width_are_reconstructed_without_changing_old_models():
    legacy = {"hidden_size": 384, "discriminator_window_length": 4, "gru_critic": True}
    assert policy_window_length(legacy) == 4
    assert critic_hidden_size(legacy) == 64
    assert critic_hidden_size({**legacy, "gru_critic": False}) == 384
    explicit = {**legacy, "policy_window_length": 16, "critic_hidden_size": 256}
    assert policy_window_length(explicit) == 16
    assert critic_hidden_size(explicit) == 256


def test_default_gru_widths_match_and_critic_override_does_not_change_actor():
    args = arguments("--hidden-size", "384", "--gru-policy", "--gru-critic")
    agent = Agent(6, [2, 3], None, args.hidden_size, args.layers, gru_policy=True, gru_critic=True,
                  policy_window_length=policy_window_length(args), critic_hidden_size=critic_hidden_size(args))
    assert agent.actor.gru.hidden_size == agent.critic.gru.hidden_size == 384
    assert agent.actor.window_length == agent.critic.window_length == agent.policy_window_length == 16
    assert agent.critic_window_length == 16  # Old API name remains an alias.
    args = arguments("--gru-policy", "--gru-critic", "--critic-hidden-size", "32")
    agent = Agent(6, [2, 3], None, args.hidden_size, args.layers, gru_policy=True, gru_critic=True,
                  policy_window_length=policy_window_length(args), critic_hidden_size=critic_hidden_size(args))
    assert agent.actor.gru.hidden_size == 8 and agent.critic.gru.hidden_size == 32


@pytest.mark.parametrize("flag", ["--critic-hidden-size", "--hidden-size"])
def test_hidden_sizes_must_be_positive(flag):
    with pytest.raises(ValueError, match="hidden sizes"):
        arguments(flag, "0")


def test_policy_settings_do_not_change_discriminator_parameters():
    states = []
    for length, width in ((4, 8), (32, 64)):
        args = arguments("--policy-window-length", str(length), "--critic-hidden-size", str(width))
        torch.manual_seed(41)
        states.append(build_discriminator(args, 2).state_dict())
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


class Codec:
    def mask(self, observations):
        return torch.ones(len(observations), 5, dtype=torch.bool, device=observations.device)


class CounterEnv:
    device = torch.device("cpu")

    def __init__(self):
        self.time = 0
        self.finals = []

    def observation(self):
        result = torch.zeros(2, 6)
        result[:, 0] = self.time
        if self.time >= 2:
            result[0, 0] += 100
        return result

    def step(self, actions):
        self.time += 1
        observation = self.observation()
        final = observation.clone()
        truncated = torch.tensor([self.time == 2, False])
        if self.time == 2:
            final[0, 0] = 2
        self.finals.append(final)
        return observation, torch.zeros(2), torch.zeros(2, dtype=torch.bool), truncated, {"final_obs": final}


class RecordedAgent(Agent):
    def __init__(self, length):
        super().__init__(6, [2, 3], Codec(), hidden_size=8, layers=1, gru_policy=True,
                         gru_critic=True, policy_window_length=length)
        self.actor_inputs = []
        self.critic_inputs = []

    def get_action_and_log_prob(self, observations, *args, **kwargs):
        self.actor_inputs.append(observations.detach().clone())
        return super().get_action_and_log_prob(observations, *args, **kwargs)

    def get_value_encoded(self, encoded, padding):
        self.critic_inputs.append((encoded.detach().clone(), padding.clone()))
        return super().get_value_encoded(encoded, padding)


@pytest.mark.parametrize("length", [1, 4, 8])
@pytest.mark.parametrize("stride", [1, 3])
def test_acting_ppo_and_timeouts_share_dense_history_while_discriminator_is_strided(length, stride):
    torch.manual_seed(16)
    env = CounterEnv()
    agent = RecordedAgent(length)
    args = SimpleNamespace(rollout=3, bf16=False)
    rollout = Rollout.allocate(3, 2, 6, 2, "cpu")
    observation = env.observation()
    scenes = TransitionRollout(observation, 3, 1, 2, decision_stride=stride)
    for block in range(2):
        actor_start, critic_start = len(agent.actor_inputs), len(agent.critic_inputs)
        observation, _ = collect(env, agent, agent.get_action_and_value, rollout, observation, args, scenes)
        actor_windows = agent.actor_inputs[actor_start:actor_start + 3]
        critic_windows = agent.critic_inputs[critic_start:critic_start + 6]
        for step, window in enumerate(actor_windows):
            time = block * 3 + step
            assert window.shape == (2, length, 6)
            for actor in range(2):
                start = max(2 if actor == 0 and time >= 2 else 0, time - length + 1)
                expected = torch.arange(start, time + 1).float()
                if actor == 0 and time >= 2:
                    expected += 100
                torch.testing.assert_close(window[actor, -len(expected):, 0], expected)
                assert torch.isnan(window[actor, :length - len(expected)]).all()
            encoded, padding = critic_windows[step * 2]
            torch.testing.assert_close(padding, torch.isnan(window[..., 0]))
            projected = agent.critic.projection(torch.nan_to_num(window))
            torch.testing.assert_close(encoded[~padding], projected[~padding])
            final_window = torch.cat((window[:, 1:], env.finals[time][:, None]), dim=1)
            final_encoded, final_padding = critic_windows[step * 2 + 1]
            torch.testing.assert_close(final_padding, torch.isnan(final_window[..., 0]))
            projected_final = agent.critic.projection(torch.nan_to_num(final_window))
            torch.testing.assert_close(final_encoded[~final_padding], projected_final[~final_padding])
            timeout = torch.tensor([time == 1, False])
            torch.testing.assert_close(rollout.timeout_values[step], agent.get_value(final_window) * timeout)
            d_window = scenes.transitions(torch.tensor([step * 2 + 1]))[0, :, 0]
            assert d_window[-1] == time + 1
            if time + 1 >= stride:
                assert d_window[0] == time + 1 - stride
            else:
                assert torch.isnan(d_window[0])
        rollout.prepare_critic_windows()
        reconstructed = rollout.critic_windows(torch.arange(6))
        torch.testing.assert_close(reconstructed, torch.stack(actor_windows).flatten(0, 1), equal_nan=True)
        torch.testing.assert_close(agent.get_value(reconstructed), rollout.values.flatten())
        agent.zero_grad(set_to_none=True)
        loss, metrics = ppo_loss(agent, rollout.observations.flatten(0, 1), rollout.actions.flatten(0, 1),
                                 rollout.log_probs.flatten(), torch.randn(6), torch.randn(6), rollout.values.flatten(),
                                 .2, .01, .5, True, reconstructed)
        assert abs(metrics[3].item()) < 1e-6
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in agent.parameters())


@pytest.mark.parametrize("flags", [["--gru-policy"], ["--gru-critic"], ["--gru-policy", "--gru-critic"]])
def test_checkpoint_rejects_changed_shared_history(tmp_path, flags):
    old, new = arguments(*flags, "--policy-window-length", "4"), arguments(*flags, "--policy-window-length", "8")
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(old)}, path)
    with pytest.raises(ValueError, match="actor/critic history length differs"):
        load_checkpoint(path, "gaifo", new, None, None)


@pytest.mark.parametrize("flags", [[], ["--gru-critic"]])
def test_checkpoint_rejects_changed_critic_width(tmp_path, flags):
    old, new = arguments(*flags), arguments(*flags, "--critic-hidden-size", "32")
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(old)}, path)
    with pytest.raises(ValueError, match="critic hidden size differs"):
        load_checkpoint(path, "gaifo", new, None, None)


def test_legacy_gru_checkpoint_can_resume_with_explicit_old_history_and_width(tmp_path):
    args = arguments("--gru-policy", "--gru-critic", "--policy-window-length", "4", "--critic-hidden-size", "64",
                     "--discriminator-window-length", "4")
    agent = Agent(6, [2, 3], Codec(), 8, 1, gru_policy=True, gru_critic=True,
                  policy_window_length=4, critic_hidden_size=64)
    optimizer = torch.optim.Adam(agent.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, "gaifo", args, agent, optimizer, 10)
    payload = torch.load(path, weights_only=False)
    for key in ("policy_window_length", "critic_hidden_size", "policy_observation_hz", "policy_window_span_ticks"):
        payload["config"].pop(key)
    torch.save(payload, path)
    before = copy.deepcopy(agent.state_dict())
    load_checkpoint(path, "gaifo", args, agent, optimizer)
    for name, value in before.items():
        torch.testing.assert_close(value, agent.state_dict()[name])


@pytest.mark.parametrize("legacy", [False, True])
def test_viewer_reconstructs_independent_or_legacy_policy_history(tmp_path, monkeypatch, legacy):
    from vis import playback

    length, d_length, width = (4, 4, 64) if legacy else (7, 2, 8)
    args = arguments("--gru-policy", "--gru-critic", "--policy-window-length", str(length),
                     "--discriminator-window-length", str(d_length), "--critic-hidden-size", str(width))
    agent = Agent(6, [2, 3], Codec(), 8, 1, gru_policy=True, gru_critic=True,
                  policy_window_length=length, critic_hidden_size=width)
    config = vars(args).copy()
    if legacy:
        config.pop("policy_window_length")
        config.pop("critic_hidden_size")
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "config": config, "agent": agent.state_dict(), "step": 10}, path)
    env = SimpleNamespace(single_observation_space=Box(-1., 1., shape=(6,)),
                          single_action_space=SimpleNamespace(nvec=np.array([2, 3])), action_codec=Codec(),
                          device=torch.device("cpu"), close=lambda: None)
    monkeypatch.setattr(playback, "build_env", lambda args: env)
    _, restored, _, _ = playback.load_policy(path, 0)
    assert restored.policy_window_length == length
    assert restored.critic.gru.hidden_size == width
    observations = torch.randn(2, length, 6)
    torch.testing.assert_close(restored.get_action(observations, sample=False), agent.get_action(observations, sample=False))
