import argparse
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from imitation.absorbing import (
    AbsorbingDiscriminator,
    AbsorbingRollout,
    AbsorbingValue,
    absorbing_reward,
    update_absorbing_value,
)
from imitation.contrastive import evaluate_consistency
from imitation.features import GLOBAL_WEIGHT, SPECIALIST_WEIGHT
from imitation.models import build_discriminator
from imitation.rewards import discriminator_rewards, imitation_rewards
from imitation.rollout import TransitionRollout
from imitation.update import discriminator_loss, train_discriminator_batch
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint, save_checkpoint


def arguments(*flags):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=True)
    add_gaifo_arguments(parser)
    args = parser.parse_args([
        "--hidden-size", "8", "--layers", "1", "--frame-embedding", "8",
        "--discriminator-hidden", "8", "--discriminator-window-length", "2", *flags
    ])
    resolve_gaifo_defaults(args)
    return args


class ScalarScore(nn.Module):
    factorized = False

    def forward(self, windows):
        return windows[:, -1, 0]


class FactorizedScore(nn.Module):
    factorized = True

    def head_score(self, windows, head):
        return windows[:, -1, head]


def test_amp_is_opt_in_and_not_mixed_precision():
    legacy = arguments()
    assert not legacy.amp and not legacy.bf16
    del legacy.amp
    resolve_gaifo_defaults(legacy)
    assert vars(legacy) == vars(arguments())
    assert arguments("--amp").amp and not arguments("--amp").bf16
    assert arguments("--amp", "--bf16").bf16


@pytest.mark.parametrize("flag", ["--discriminator-log-reward", "--exp-log-odds-reward"])
def test_amp_rejects_other_reward_mappings(flag):
    with pytest.raises(ValueError, match="AMP.*mutually exclusive"):
        arguments("--amp", flag)


@pytest.mark.parametrize("flags,legacy,amp", [
    ([], "gaifo-bce-no-goal-v2", "gaifo-amp-global-no-goal-v1"),
    (["--ego-only"], "gaifo-ego-only-no-goal-v1", "gaifo-amp-ego-only-no-goal-v1"),
    (["--factorize"], "gaifo-ego-global-no-goal-v3", "gaifo-amp-ego-global-no-goal-v1"),
    (["--factorize", "--ego-only"], "gaifo-ego-only-no-goal-v1", "gaifo-amp-ego-only-no-goal-v1"),
])
def test_amp_objective_identity_preserves_legacy_head_modes(flags, legacy, amp):
    assert arguments(*flags).discriminator_objective == legacy
    assert arguments("--amp", *flags).discriminator_objective == amp


def test_amp_loss_regresses_unclipped_scores_to_signed_targets():
    scores = torch.tensor([1., -1., 0., 0., 3., -3.], requires_grad=True)
    labels = torch.tensor([1., 0., 1., 0., 1., 0.])
    windows = scores[:, None, None].expand(-1, 2, 1)
    loss = discriminator_loss(ScalarScore(), windows, labels, amp=True)
    torch.testing.assert_close(loss, torch.tensor(10 / 6))
    loss.backward()
    targets = torch.tensor([1., -1., 1., -1., 1., -1.])
    torch.testing.assert_close(scores.grad, 2 * (scores.detach() - targets) / 6)
    optimal = targets[:, None, None].expand(-1, 2, 1)
    assert discriminator_loss(ScalarScore(), optimal, labels, amp=True) == 0
    assert discriminator_loss(ScalarScore(), windows, labels, (), amp=True) == 0


@pytest.mark.parametrize("active", [None, (0,), (1,), ()])
def test_amp_factorized_loss_respects_weights_and_inactive_heads(active):
    scores = torch.tensor([[1., 0.], [-2., -3.]], requires_grad=True)
    windows = scores[:, None].expand(-1, 2, -1)
    loss = discriminator_loss(FactorizedScore(), windows, torch.tensor([1., 0.]), active, amp=True)
    heads = (0, 1) if active is None else active
    weights = torch.tensor([SPECIALIST_WEIGHT, GLOBAL_WEIGHT])
    errors = scores.detach() - torch.tensor([[1.], [-1.]])
    expected = sum(weights[head] * errors[:, head].square().mean() for head in heads)
    assert loss.item() == pytest.approx(float(expected))
    if heads:
        loss.backward()
        gradient = errors * weights
        for head in set((0, 1)) - set(heads):
            gradient[:, head] = 0
        torch.testing.assert_close(scores.grad, gradient)
    else:
        assert not loss.requires_grad


def test_amp_batch_uses_balanced_mse_and_accumulates_partial_microbatches():
    class LinearScore(nn.Module):
        factorized = False

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

        def forward(self, windows):
            return self.weight * windows[:, -1, 0]

    args = arguments("--amp", "--discriminator-motion-noise-scale", "0",
                     "--discriminator-update-microbatch-size", "2", "--max-grad-norm", "100")
    model = LinearScore()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    generated = torch.ones(3, 2, 51)
    loss = train_discriminator_batch(model, optimizer, generated, -generated, args)
    torch.testing.assert_close(loss, torch.tensor(1.))
    torch.testing.assert_close(model.weight, torch.tensor(.2))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_amp_reward_is_bounded_and_peaks_at_expert_target(dtype):
    limit = torch.finfo(dtype).max
    scores = torch.tensor([-limit, -4., -3., -2., -1., 0., 1., 2., limit], dtype=dtype)
    rewards = discriminator_rewards(scores, amp=True)
    assert rewards.dtype == torch.float32
    torch.testing.assert_close(rewards, torch.tensor([0., 0., 0., .75, 1., .75, 0., 0., 0.]))
    assert torch.isfinite(rewards).all()


@pytest.mark.parametrize("multiply", [False, True])
def test_amp_maps_each_head_before_reward_reduction(multiply):
    scores = torch.tensor([[-1., -1.], [-1., 0.], [0., 0.], [-3., -1.], [1., 0.]])
    head_rewards = torch.tensor([[1., 1.], [1., .75], [.75, .75], [0., 1.], [0., .75]])
    expected = head_rewards.prod(-1) if multiply else head_rewards.mean(-1)
    torch.testing.assert_close(discriminator_rewards(scores, multiplicative_reward=multiply, amp=True), expected)


@pytest.mark.parametrize("shaping", [False, True])
@pytest.mark.parametrize("include_terminal", [False, True])
@pytest.mark.parametrize("gameplay", [[], ["--goal-reward"], ["--reward-plus"]])
def test_amp_rollout_reward_potential_and_gameplay_terms(shaping, include_terminal, gameplay):
    args = arguments("--amp", *gameplay, "--reward-plus-weight", ".25",
                     "--discriminator-inference-microbatch-size", "2",
                     *(["--discriminator-potential-shaping"] if shaping else []),
                     "--discriminator-potential-weight", ".5")
    args.reward_gamma = .9
    current = torch.tensor([[-1.], [0.], [-2.]])
    final = torch.tensor([[0.], [-2.], [-1.]])
    scenes = TransitionRollout(current, 1, 1, 2, potential_shaping=shaping)
    scenes.record(0, current, final, final, torch.tensor([False, True, True]),
                  torch.tensor([False, True, False]))
    rollout = SimpleNamespace(rewards=torch.tensor([[10., 20., 30.]]))
    metrics = imitation_rewards(ScalarScore(), scenes, rollout, args, include_terminal=include_terminal)
    base = torch.tensor([.75, .75 if include_terminal else 0., 1.])
    expected = base.clone()
    if gameplay:
        expected += torch.tensor([2.5, 5., 7.5])
    if shaping:
        potential = .5 * (.9 * torch.tensor([.75, 0., 1.]) - torch.tensor([1., .75, .75]))
        expected += potential
        assert metrics["discriminator_potential/mean"] == pytest.approx(potential.mean().item())
    torch.testing.assert_close(rollout.rewards[0], expected)
    assert metrics["imitation_reward"] == pytest.approx(base.mean().item())


@pytest.mark.parametrize("flags,expected", [([], .75), (["--factorize"], .375),
                                           (["--factorize", "--multiplicative-reward"], 0.)])
def test_amp_dac_terminal_and_absorbing_rewards_share_the_mapping(flags, expected):
    class IndicatorScore(nn.Module):
        def forward(self, windows):
            scores = windows[:, -1, -1] - 1
            return torch.stack((scores, scores + 1), -1) if args.factorize else scores

    args = arguments("--amp", *flags)
    model = IndicatorScore()
    scenes = AbsorbingRollout(torch.zeros(2, 51), 1, 51, 2)
    goals = torch.ones(2, dtype=torch.bool)
    scenes.record(0, torch.zeros(2, 51), torch.ones(2, 51), torch.ones(2, 51), goals, goals)
    rollout = SimpleNamespace(rewards=torch.zeros(1, 2))
    imitation_rewards(model, scenes, rollout, args, include_terminal=True)
    torch.testing.assert_close(rollout.rewards, torch.full((1, 2), expected))
    reward = absorbing_reward(model, args, "cpu", 51)
    torch.testing.assert_close(reward, torch.tensor(expected))
    value = AbsorbingValue()
    update_absorbing_value(value, torch.optim.SGD(value.parameters(), lr=.2), reward, gamma=.5, updates=250)
    torch.testing.assert_close(value.value, torch.tensor(2 * expected), rtol=1e-5, atol=1e-5)


def test_amp_contrastive_diagnostics_use_the_selected_reward_mapping():
    clean = torch.zeros(4, 2, 1)
    perturbed = clean.clone()
    perturbed[:, -1, 0] = -1
    metrics = evaluate_consistency(ScalarScore(), clean, perturbed, torch.ones(4, dtype=torch.bool),
                                   arguments("--amp"))
    for source in ("generated", "expert"):
        assert metrics[f"discriminator/contrastive_{source}_heldout_reward_abs_change"] == .25
        assert metrics[f"discriminator/global_contrastive_{source}_heldout_mse"] == 1.


@pytest.mark.parametrize("algorithm", ["gaifo", "dac"])
@pytest.mark.parametrize("mode", ["amp", "bce", "legacy"])
def test_amp_and_legacy_checkpoint_roundtrips(tmp_path, algorithm, mode):
    flags = ["--factorize", "--mlp-discriminator"] + (["--amp"] if mode == "amp" else [])
    args = arguments(*flags)
    if algorithm == "dac":
        args.discriminator_objective += "-absorbing-v1"
    model = AbsorbingDiscriminator(args, 2) if algorithm == "dac" else build_discriminator(args, 2)
    discriminator_optimizer = torch.optim.Adam(model.parameters())
    generated = torch.randn(4, 2, 52 if algorithm == "dac" else 51) * .1
    if algorithm == "dac":
        generated[..., -1] = 0
        generated[1, -1] = 0
        generated[1, -1, -1] = 1
        generated[2] = 0
        generated[2, :, -1] = 1
    train_discriminator_batch(model, discriminator_optimizer, generated, generated.clone(), args)
    agent = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(agent.parameters())
    agent(torch.ones(2, 2)).square().mean().backward()
    optimizer.step()
    if mode == "legacy":
        del args.amp
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, algorithm, args, agent, optimizer, 10, discriminator=model.state_dict(),
                    discriminator_optimizer=discriminator_optimizer.state_dict())
    requested = arguments(*flags)
    requested.discriminator_objective = args.discriminator_objective
    restored = nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored.parameters())
    payload = load_checkpoint(path, algorithm, requested, restored, restored_optimizer)
    assert payload["step"] == 10
    assert payload["config"].get("amp", False) == (mode == "amp")
    torch.testing.assert_close(restored.state_dict(), agent.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(restored_optimizer.state_dict()["state"], optimizer.state_dict()["state"])
    restored_discriminator = AbsorbingDiscriminator(requested, 2) if algorithm == "dac" else build_discriminator(requested, 2)
    restored_discriminator.load_state_dict(payload["discriminator"])
    restored_discriminator_optimizer = torch.optim.Adam(restored_discriminator.parameters())
    restored_discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
    torch.testing.assert_close(restored_discriminator.state_dict(), model.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(restored_discriminator_optimizer.state_dict()["state"],
                               discriminator_optimizer.state_dict()["state"])


@pytest.mark.parametrize("algorithm", ["gaifo", "dac"])
@pytest.mark.parametrize("original_amp", [None, False, True])
def test_checkpoint_cannot_silently_switch_amp_mode(tmp_path, algorithm, original_amp):
    original = arguments(*(["--amp"] if original_amp else []))
    if original_amp is None:
        del original.amp
    requested = arguments(*([] if original_amp else ["--amp"]))
    if algorithm == "dac":
        original.discriminator_objective += "-absorbing-v1"
        requested.discriminator_objective += "-absorbing-v1"
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": algorithm, "config": vars(original)}, path)
    with pytest.raises(ValueError, match="AMP mode differs"):
        load_checkpoint(path, algorithm, requested, None, None)
