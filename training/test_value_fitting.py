import argparse
import copy
import warnings
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint, save_checkpoint
from training.models import Agent
from training.ppo import clip_actor_critic_gradients, ppo_loss, update_ppo, value_statistics
from training.rollout import Rollout


def arguments(*flags, gaifo=True):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=gaifo)
    if gaifo:
        add_gaifo_arguments(parser)
    args = parser.parse_args(["--hidden-size", "8", "--layers", "1", "--minibatch-size", "4",
                              "--update-epochs", "2", "--no-bf16", *flags])
    if gaifo:
        resolve_gaifo_defaults(args)
    return args


@pytest.mark.parametrize("gaifo", [False, True])
def test_value_defaults_preserve_ppo_and_condition_adversarial_critic_fitting(gaifo):
    args = arguments(gaifo=gaifo)
    assert args.clip_value_loss is not gaifo
    assert args.normalize_value_loss is gaifo
    assert args.value_clip_coef == .2
    assert args.critic_max_grad_norm == (1.0 if gaifo else None)
    assert args.max_grad_norm == .5
    assert args.gamma == .995 and args.gae_lambda == .98
    assert args.update_epochs == 2  # Explicit override; the actual default remains 4.


@pytest.mark.parametrize("flag", ["--value-clip-coef", "--critic-max-grad-norm"])
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_critic_fitting_limits_must_be_finite_and_positive(flag, value):
    with pytest.raises(ValueError, match="finite and positive"):
        arguments(f"{flag}={value}")


@pytest.mark.parametrize("kind,explained,bias", [("perfect", 1., 0.), ("mean", 0., 0.),
                                                 ("reversed", -3., 0.), ("biased", 1., 5.)])
def test_explained_variance_and_bias_are_separate(kind, explained, bias):
    target = torch.tensor([-2., -1., 1., 2.])
    values = {"perfect": target, "mean": torch.zeros_like(target), "reversed": -target,
              "biased": target + 5}[kind]
    statistics = value_statistics(values, target)
    assert statistics["explained_variance"].item() == pytest.approx(explained)
    assert statistics["value/explained_variance_valid"] == 1
    assert statistics["value/bias"].item() == pytest.approx(bias)
    assert statistics["value/rmse"].item() == pytest.approx((values - target).square().mean().sqrt().item())


@pytest.mark.parametrize("size", [1, 8])
def test_constant_return_targets_have_finite_metrics_and_mark_variance_undefined(size):
    statistics = value_statistics(torch.arange(size).float(), torch.full((size,), 3.))
    assert statistics["explained_variance"] == 0
    assert statistics["value/explained_variance_valid"] == 0
    assert statistics["value/return_std"] == 0
    assert all(torch.isfinite(value) for value in statistics.values())


def test_value_diagnostics_are_detached_and_do_not_consume_random_numbers():
    target = torch.randn(16, requires_grad=True)
    values = torch.randn(16, requires_grad=True)
    state = torch.get_rng_state()
    statistics = value_statistics(values, target)
    assert torch.equal(state, torch.get_rng_state())
    assert all(not value.requires_grad for value in statistics.values())


class Codec:
    def mask(self, observations):
        return torch.ones(len(observations), 2, dtype=torch.bool, device=observations.device)


@pytest.mark.parametrize("clipping", [False, True])
def test_standardized_value_objective_is_scale_invariant_and_preserves_actor_gradients(clipping):
    torch.manual_seed(17)
    original = Agent(3, [2], Codec(), hidden_size=8, layers=1)
    scaled = copy.deepcopy(original)
    with torch.no_grad():
        scaled.critic[-1].weight.mul_(7)
        scaled.critic[-1].bias.mul_(7)
    observations = torch.randn(8, 3)
    actions = torch.zeros(8, 1, dtype=torch.long)
    returns, old_values, advantages = torch.randn(8), torch.randn(8), torch.randn(8)
    losses, metrics = [], []
    for model, multiplier in ((original, 1), (scaled, 7)):
        loss, measured = ppo_loss(model, observations, actions, torch.zeros(8), advantages,
                                  returns * multiplier, old_values * multiplier, .1, .01, .5, clipping,
                                  value_scale=torch.tensor(2.5 * multiplier), value_clip_coef=.3)
        loss.backward()
        losses.append(loss)
        metrics.append(measured)
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(metrics[0], metrics[1])
    for left, right in zip(original.actor.parameters(), scaled.actor.parameters()):
        torch.testing.assert_close(left.grad, right.grad)


class ConstantValueAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.))

    def get_action_and_value(self, observations, actions):
        zeros = observations.new_zeros(len(observations))
        return actions, zeros, zeros, self.value.expand(len(observations))


@pytest.mark.parametrize("clipping,scale,expected_gradient", [(False, 1., -9.), (True, 1., 0.), (True, 10., -.09)])
def test_unclipped_default_avoids_old_raw_value_clip_plateau(clipping, scale, expected_gradient):
    model = ConstantValueAgent()
    loss, _ = ppo_loss(model, torch.zeros(4, 2), torch.zeros(4, 1, dtype=torch.long), torch.zeros(4),
                       torch.ones(4), torch.full((4,), 10.), torch.zeros(4), .2, 0., 1., clipping,
                       value_scale=torch.tensor(scale), value_clip_coef=.2)
    loss.backward()
    assert model.value.grad.item() == pytest.approx(expected_gradient)


def test_critic_clip_coefficient_is_independent_of_actor_clip():
    model = ConstantValueAgent()
    loss, _ = ppo_loss(model, torch.zeros(4, 2), torch.zeros(4, 1, dtype=torch.long), torch.zeros(4),
                       torch.ones(4), torch.full((4,), 10.), torch.zeros(4), .01, 0., 1., True,
                       value_clip_coef=2.)
    loss.backward()
    assert model.value.grad.item() == pytest.approx(-9.)


def test_separate_critic_gradient_limit_preserves_actor_limit():
    agent = SimpleNamespace(actor=nn.Linear(1, 1, bias=False), critic=nn.Linear(1, 1, bias=False))
    agent.actor.weight.grad = torch.full_like(agent.actor.weight, 3.)
    agent.critic.weight.grad = torch.full_like(agent.critic.weight, 4.)
    actor_norm, critic_norm = clip_actor_critic_gradients(agent, .5, 1.)
    assert actor_norm.item() == pytest.approx(3.) and critic_norm.item() == pytest.approx(4.)
    assert agent.actor.weight.grad.norm().item() == pytest.approx(.5)
    assert agent.critic.weight.grad.norm().item() == pytest.approx(1.)


def test_shared_parameters_use_the_stricter_gradient_limit():
    agent = nn.Module()
    agent.actor = nn.Linear(1, 1, bias=False)
    agent.critic = agent.actor
    agent.actor.weight.grad = torch.full_like(agent.actor.weight, 4.)
    clip_actor_critic_gradients(agent, .5, .1)
    assert agent.actor.weight.grad.norm().item() == pytest.approx(.1)


@pytest.mark.parametrize("flags", [[], ["--clip-value-loss"], ["--no-normalize-value-loss", "--clip-value-loss"]])
@pytest.mark.parametrize("constant_target", [False, True])
def test_update_logs_raw_pre_fit_diagnostics_and_uses_one_scale_for_every_minibatch(flags, constant_target):
    torch.manual_seed(23)
    args = arguments(*flags)
    model = Agent(3, [2], Codec(), hidden_size=8, layers=1)
    rollout = Rollout.allocate(2, 4, 3, 1, "cpu")
    rollout.observations.normal_()
    with torch.no_grad():
        actions, log_probs, _, values = model.get_action_and_value(rollout.observations.flatten(0, 1))
        rollout.actions.copy_(actions.reshape(2, 4, 1))
        rollout.log_probs.copy_(log_probs.reshape(2, 4))
        rollout.values.copy_(values.reshape(2, 4))
        rollout.returns.copy_(rollout.values + torch.linspace(-3, 5, 8).reshape(2, 4))
        if constant_target:
            rollout.returns.fill_(3.)
        rollout.advantages.copy_(rollout.returns - rollout.values)
    original_values, original_returns = rollout.values.clone(), rollout.returns.clone()
    expected = value_statistics(rollout.values, rollout.returns)
    expected_scale = max(expected["value/return_std"].item(), 1.) if args.normalize_value_loss else 1.
    observed_scales = []
    forward_calls = {"actor": 0, "critic": 0}

    def record_forward(name):
        def record(*unused):
            forward_calls[name] += 1
        return record

    model.actor.register_forward_hook(record_forward("actor"))
    model.critic.register_forward_hook(record_forward("critic"))

    def recorded_loss(*inputs):
        observed_scales.append(inputs[-2].item())
        return ppo_loss(*inputs)

    metrics = update_ppo(model, torch.optim.Adam(model.parameters(), lr=args.learning_rate),
                         recorded_loss, rollout, args)
    assert len(observed_scales) == 4
    assert forward_calls == {"actor": 4, "critic": 4}  # Diagnostics add no inference passes.
    assert all(scale == pytest.approx(expected_scale) for scale in observed_scales)
    for name, value in expected.items():
        assert metrics[name] == pytest.approx(value.item())
    assert metrics["value/loss_scale"] == pytest.approx(expected_scale)
    assert metrics["value/clip_range"] == pytest.approx(.2 * expected_scale if args.clip_value_loss else 0.)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    torch.testing.assert_close(rollout.values, original_values)
    torch.testing.assert_close(rollout.returns, original_returns)


def test_legacy_checkpoint_warns_but_preserves_weights_when_new_fitting_defaults_are_used(tmp_path):
    args = arguments()
    model = Agent(3, [2], Codec(), hidden_size=8, layers=1)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, "gaifo", args, model, optimizer, 1)
    payload = torch.load(path, weights_only=False)
    payload["config"]["clip_value_loss"] = True
    for key in ("normalize_value_loss", "value_clip_coef", "critic_max_grad_norm"):
        payload["config"].pop(key)
    torch.save(payload, path)
    before = copy.deepcopy(model.state_dict())
    with pytest.warns(UserWarning, match="Critic fitting configuration changed"):
        load_checkpoint(path, "gaifo", args, model, optimizer)
    for name, value in before.items():
        torch.testing.assert_close(value, model.state_dict()[name])
    legacy = arguments("--clip-value-loss", "--no-normalize-value-loss", "--critic-max-grad-norm", ".5")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        load_checkpoint(path, "gaifo", legacy, model, optimizer)
    assert not any("Critic fitting" in str(warning.message) for warning in recorded)
