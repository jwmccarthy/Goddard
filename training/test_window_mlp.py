import copy

import pytest
import torch
from torch import nn

from imitation.rollout import TransitionRollout
from training.checkpoints import load_checkpoint, save_checkpoint
from training.models import Agent, ObservationWindowMLP
from training.ppo import ppo_loss
from training.rollout import Rollout, collect
from training.test_policy_history import Codec, CounterEnv, arguments


def test_window_mlp_flags_are_explicit_and_mutually_exclusive_with_grus():
    args = arguments("--mlp-policy", "--mlp-critic", "--policy-window-length", "8")
    assert args.policy_architecture == args.critic_architecture == "mlp-window-v1"
    assert args.policy_window_span_ticks == 28
    assert args.discriminator_observation_hz == 10
    assert not args.gru_policy and not args.gru_critic
    with pytest.raises(SystemExit):
        arguments("--mlp-policy", "--gru-policy")
    with pytest.raises(SystemExit):
        arguments("--mlp-critic", "--gru-critic")


def test_raw_window_mlp_masks_padding_marks_valid_frames_and_preserves_order():
    head = ObservationWindowMLP(3, 2, 8, 1, 4, .01)
    captured = []
    head.network.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    observations = torch.tensor([[[float("nan"), 99., 88.], [1., 2., 3.]]])
    short = head(observations)
    features = captured[-1].reshape(1, 4, 4)
    torch.testing.assert_close(features[0, :3], torch.zeros(3, 4))
    torch.testing.assert_close(features[0, -1], torch.tensor([1., 2., 3., 1.]))
    full = torch.cat((torch.full((1, 2, 3), float("nan")), observations), dim=1)
    torch.testing.assert_close(short, head(full))


@pytest.mark.parametrize("shape", [(2, 3), (2, 0, 3), (2, 5, 3)])
def test_window_mlp_rejects_missing_empty_or_oversized_history(shape):
    with pytest.raises(ValueError, match="observation histories"):
        ObservationWindowMLP(3, 2, 8, 1, 4, .01)(torch.zeros(shape))


def test_window_mlp_actor_and_critic_use_the_past_but_action_masks_only_see_current_physical_state():
    observations = torch.randn(2, 4, 9)
    observations[..., -3:] = torch.eye(3)[:2, None]
    observations[0, :2] = float("nan")
    current_physical = observations[:, -1, :6].clone()

    class PhysicalCodec:
        def mask(self, value):
            torch.testing.assert_close(value, current_physical)
            return torch.ones(len(value), 5, dtype=torch.bool)

    torch.manual_seed(7)
    model = Agent(9, [2, 3], PhysicalCodec(), 8, 1, mlp_policy=True, mlp_critic=True,
                  policy_window_length=4, info_codes=3)
    assert model.actor.network[0].in_features == model.critic.network[0].in_features == 40
    actions = torch.tensor([[0, 1], [1, 2]])
    baseline = model.get_action_and_value(observations, actions)
    changed = observations.clone()
    changed[:, -2, :6] += 3
    modified = model.get_action_and_value(changed, actions)
    assert not torch.allclose(baseline[1], modified[1], atol=1e-7, rtol=1e-7)
    assert not torch.allclose(baseline[-1], modified[-1])
    changed = observations.clone()
    changed[0, :2, 1:] = 1000
    ignored_padding = model.get_action_and_value(changed, actions)
    torch.testing.assert_close(baseline[1], ignored_padding[1])
    torch.testing.assert_close(baseline[-1], ignored_padding[-1])


class RecordedAgent(Agent):
    def __init__(self, actor_kind, critic_kind, length):
        super().__init__(6, [2, 3], Codec(), 8, 1, policy_window_length=length,
                         gru_policy=actor_kind == "gru", mlp_policy=actor_kind == "window",
                         gru_critic=critic_kind == "gru", mlp_critic=critic_kind == "window")
        self.actor_inputs = []

    def get_action_and_log_prob(self, observations, *args, **kwargs):
        self.actor_inputs.append(observations.detach().clone())
        return super().get_action_and_log_prob(observations, *args, **kwargs)


@pytest.mark.parametrize("actor_kind", ["state", "window", "gru"])
@pytest.mark.parametrize("critic_kind", ["state", "window", "gru"])
@pytest.mark.parametrize("length", [1, 4])
def test_window_mlp_gru_and_state_heads_share_correct_acting_ppo_and_timeout_inputs(actor_kind, critic_kind, length):
    torch.manual_seed(9)
    model = RecordedAgent(actor_kind, critic_kind, length)
    env = CounterEnv()
    args = arguments("--rollout", "3")
    rollout = Rollout.allocate(3, 2, 6, 2, "cpu")
    observation = env.observation()
    scenes = TransitionRollout(observation, 3, 1, 2, decision_stride=3)
    for block in range(2):
        start = len(model.actor_inputs)
        observation, _ = collect(env, model, model.get_action_and_value, rollout, observation, args, scenes)
        actor_inputs = torch.stack(model.actor_inputs[start:start + 3]).flatten(0, 1)
        indices = torch.arange(6)
        if rollout.critic_prefix is not None:
            rollout.prepare_critic_windows()
            windows = rollout.critic_windows(indices)
        else:
            windows = None
        states = rollout.observations.flatten(0, 1)
        expected_actor = windows if model.actor_uses_history else states
        torch.testing.assert_close(actor_inputs, expected_actor, equal_nan=True)
        expected_critic = windows if model.critic_uses_history else states
        torch.testing.assert_close(model.get_value(expected_critic), rollout.values.flatten())
        for step in range(3):
            time = block * 3 + step
            if model.critic_uses_history:
                current_window = windows.reshape(3, 2, length, 6)[step]
                final = torch.cat((current_window[:, 1:], env.finals[time][:, None]), dim=1)
            else:
                final = env.finals[time]
            torch.testing.assert_close(rollout.timeout_values[step],
                                       model.get_value(final) * torch.tensor([time == 1, False]))
        model.zero_grad(set_to_none=True)
        loss, metrics = ppo_loss(model, states, rollout.actions.flatten(0, 1), rollout.log_probs.flatten(),
                                 torch.randn(6), torch.randn(6), rollout.values.flatten(), .2, .01, .5, False, windows)
        assert abs(metrics[3].item()) < 1e-6
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if block == 1:
            d_windows = scenes.transitions(torch.tensor([1, 3, 5]))[..., 0]
            torch.testing.assert_close(d_windows[:, -1] - d_windows[:, 0], torch.full((3,), 3.))


@pytest.mark.parametrize("old_flags,new_flags,message", [
    ([], ["--mlp-policy"], "actor architecture"),
    ([], ["--mlp-critic"], "critic architecture"),
    (["--mlp-policy", "--mlp-critic"], ["--gru-policy", "--gru-critic"], "actor architecture"),
    (["--mlp-policy", "--policy-window-length", "4"], ["--mlp-policy", "--policy-window-length", "8"], "history length"),
])
def test_window_mlp_checkpoints_reject_incompatible_architectures_and_history(tmp_path, old_flags, new_flags, message):
    old, new = arguments(*old_flags), arguments(*new_flags)
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(old)}, path)
    with pytest.raises(ValueError, match=message):
        load_checkpoint(path, "gaifo", new, None, None)


def test_window_mlp_checkpoint_and_viewer_restore_the_full_window(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    from gymnasium.spaces import Box

    from vis import playback

    args = arguments("--mlp-policy", "--mlp-critic", "--policy-window-length", "4")
    model = Agent(6, [2, 3], Codec(), 8, 1, policy_window_length=4, mlp_policy=True, mlp_critic=True)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, "gaifo", args, model, optimizer, 10)
    restored = copy.deepcopy(model)
    load_checkpoint(path, "gaifo", args, restored, optimizer)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])
    env = SimpleNamespace(single_observation_space=Box(-1., 1., shape=(6,)),
                          single_action_space=SimpleNamespace(nvec=np.array([2, 3])),
                          action_codec=Codec(), device=torch.device("cpu"), close=lambda: None)
    monkeypatch.setattr(playback, "build_env", lambda args: env)
    _, viewer_model, _, _ = playback.load_policy(path, 0)
    assert isinstance(viewer_model.actor, ObservationWindowMLP)
    assert viewer_model.actor_uses_history and viewer_model.policy_window_length == 4
    windows = torch.randn(2, 4, 6)
    torch.testing.assert_close(viewer_model.get_action(windows, sample=False), model.get_action(windows, sample=False))


def test_default_single_state_mlp_parameters_are_unchanged():
    states = []
    for flags in ({}, {"mlp_policy": False, "mlp_critic": False}):
        torch.manual_seed(18)
        model = Agent(6, [2, 3], Codec(), 8, 1, **flags)
        assert isinstance(model.actor, nn.Sequential) and isinstance(model.critic, nn.Sequential)
        assert not model.actor_uses_history and not model.critic_uses_history
        states.append(model.state_dict())
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)
