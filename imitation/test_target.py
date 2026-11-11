import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from imitation import update
from imitation.absorbing import AbsorbingDiscriminator
from imitation.models import build_discriminator
from imitation.policy_history import PolicyWindowHistory
from imitation.rewards import discriminator_rewards, imitation_rewards
from imitation.rollout import TransitionRollout
from imitation.target import create_target_discriminator, polyak_update
from imitation.test_adaptive_update import Experts, Scenes
from imitation.test_amp import arguments
from training.checkpoints import load_checkpoint, save_checkpoint


def test_target_is_opt_in_and_defaults_to_eight_update_span():
    args = arguments()
    assert not args.target_discriminator and args.target_discriminator_objective is None
    assert args.target_discriminator_span == 8 and args.target_discriminator_tau == pytest.approx(2 / 9)
    assert arguments("--target-discriminator").target_discriminator
    assert arguments("--target-discriminator-span", "4").target_discriminator_tau == .4
    assert arguments("--target-discriminator-tau", ".5").target_discriminator_span == 3
    assert arguments("--target-discriminator-tau", "1").target_discriminator_span == 1
    with pytest.raises(SystemExit):
        arguments("--target-discriminator-span", "8", "--target-discriminator-tau", ".2")
    for config in (args, SimpleNamespace()):
        rng = torch.get_rng_state()
        assert create_target_discriminator(nn.Identity(), config) is None
        assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("flag,value", [
    ("span", "0"), ("span", ".5"), ("span", "nan"), ("span", "inf"),
    ("tau", "0"), ("tau", "-1"), ("tau", "1.1"), ("tau", "nan"), ("tau", "inf"),
])
def test_invalid_target_smoothing_is_rejected(flag, value):
    with pytest.raises(ValueError, match="target discriminator"):
        arguments("--target-discriminator", f"--target-discriminator-{flag}", value)


@pytest.mark.parametrize("architecture", [[], ["--gru"], ["--mlp-discriminator"]])
@pytest.mark.parametrize("head", [[], ["--ego-only"], ["--factorize"]])
@pytest.mark.parametrize("absorbing", [False, True])
@pytest.mark.parametrize("relative", [False, True])
def test_target_is_an_independent_frozen_copy_of_all_discriminator_variants(architecture, head, absorbing, relative):
    args = arguments("--target-discriminator", *architecture, *head,
                     *(["--discriminator-relative"] if relative else []))
    online = AbsorbingDiscriminator(args, 2) if absorbing else build_discriminator(args, 2)
    rng = torch.get_rng_state()
    target = create_target_discriminator(online, args)
    assert torch.equal(rng, torch.get_rng_state())
    assert not target.training and all(not parameter.requires_grad for parameter in target.parameters())
    original = {name: value.detach().clone() for name, value in online.state_dict().items()}
    windows = torch.randn(4, 2, 52 if absorbing else 51)
    if absorbing:
        windows[..., -1] = 0
        windows[1, -1] = 0
        windows[1, -1, -1] = 1
    windows[0, 0] = float("nan")
    torch.testing.assert_close(target(windows), online(windows))
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(1)
    torch.testing.assert_close(target.state_dict(), original, rtol=0, atol=0)
    polyak_update(target, online, args.target_discriminator_tau)
    for name, value in target.state_dict().items():
        expected = original[name] + args.target_discriminator_tau * (online.state_dict()[name] - original[name])
        torch.testing.assert_close(value, expected)
    assert all(parameter.grad is None for parameter in target.parameters())


def test_polyak_averages_floating_buffers_and_copies_discrete_buffers():
    online = nn.Linear(1, 1, bias=False)
    online.weight.data.fill_(1)
    online.register_buffer("running", torch.tensor([2., 4.], dtype=torch.float64))
    online.register_buffer("count", torch.tensor(3))
    online.register_buffer("mask", torch.tensor(True))
    online.register_buffer("complex_value", torch.tensor([1 + 2j]))
    target = create_target_discriminator(online, arguments("--target-discriminator"))
    with torch.no_grad():
        online.weight.fill_(5)
        online.running.fill_(9)
        online.count.fill_(9)
        online.mask.fill_(False)
        online.complex_value.fill_(5 + 8j)
    polyak_update(target, online, .25)
    torch.testing.assert_close(target.weight, torch.tensor([[2.]]))
    torch.testing.assert_close(target.running, torch.tensor([3.75, 5.25], dtype=torch.float64))
    assert target.count == 9 and not target.mask
    torch.testing.assert_close(target.complex_value, torch.tensor([2 + 3.5j]))
    polyak_update(target, online, 1.)
    torch.testing.assert_close(target.state_dict(), online.state_dict(), rtol=0, atol=0)


def test_polyak_rejects_aliases_trainable_targets_and_mismatched_models():
    online = nn.Linear(2, 1)
    with pytest.raises(ValueError, match="independent frozen"):
        polyak_update(online, online, .2)
    with pytest.raises(ValueError, match="independent frozen"):
        polyak_update(copy.deepcopy(online), online, .2)
    target = copy.deepcopy(online).requires_grad_(False)
    with pytest.raises(ValueError, match="tensor differs"):
        polyak_update(target, nn.Linear(3, 1), .2)
    with pytest.raises(ValueError, match="structures differ"):
        polyak_update(target, nn.Sequential(nn.Linear(2, 1)), .2)
    for tau in (0, -1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="tau"):
            polyak_update(target, online, tau)


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("head", [[], ["--ego-only"], ["--factorize"]])
@pytest.mark.parametrize("validation", [False, True])
def test_target_mode_updates_every_head_even_at_perfect_or_missing_accuracy(monkeypatch, amp, head, validation):
    args = arguments("--target-discriminator", "--mlp-discriminator", *head, *(["--amp"] if amp else []),
                     "--discriminator-updates", "2", "--discriminator-batch-size", "2",
                     "--discriminator-update-microbatch-size", "1", "--discriminator-motion-noise-scale", "0")
    online = build_discriminator(args, 2)
    target = create_target_discriminator(online, args)
    optimizer = torch.optim.Adam(online.parameters(), lr=.01)
    monkeypatch.setattr(update, "sample_discriminator_validation", lambda *a: torch.zeros(4, 2, 51) if validation else None)
    monkeypatch.setattr(update, "evaluate_discriminator", lambda *a: {
        "discriminator/heldout_accuracy": 1., "discriminator/ego_heldout_accuracy": 1.,
        "discriminator/global_heldout_accuracy": 1.
    } if validation else {})
    expected = {name: value.detach().clone() for name, value in target.named_parameters()}
    step = optimizer.step
    online_steps = []

    def recorded_step(*a, **kw):
        result = step(*a, **kw)
        online_steps.append({name: value.detach().clone() for name, value in online.named_parameters()})
        return result

    monkeypatch.setattr(optimizer, "step", recorded_step)
    metrics = update.update_discriminator(online, optimizer, Experts(), Scenes(), torch.arange(5), torch.tensor([8]),
                                          args, PolicyWindowHistory(16, 3), target_discriminator=target)
    assert len(online_steps) == metrics["discriminator/target_updates"] == 6
    assert metrics["discriminator/accuracy_gating_enabled"] == 0
    for values in online_steps:
        expected = {name: (1 - args.target_discriminator_tau) * old + args.target_discriminator_tau * values[name]
                    for name, old in expected.items()}
    for name, parameter in target.named_parameters():
        torch.testing.assert_close(parameter, expected[name])
        assert parameter.grad is None
    for name in ("ego", "global") if online.factorized else (("ego",) if head else ()):
        assert metrics[f"discriminator/{name}_update_enabled"] == 1
    assert set(map(id, target.parameters())).isdisjoint(map(id, optimizer.state))


def test_target_requires_its_copy_and_does_not_advance_without_optimizer_steps(monkeypatch):
    args = arguments("--target-discriminator")
    online = build_discriminator(args, 2)
    optimizer = torch.optim.Adam(online.parameters())
    with pytest.raises(ValueError, match="independent reward copy"):
        update.update_discriminator(online, optimizer, Experts(), Scenes(), torch.arange(4), torch.tensor([8]),
                                    args, PolicyWindowHistory(0, 0))
    target = create_target_discriminator(online, args)
    before = copy.deepcopy(target.state_dict())
    monkeypatch.setattr(update, "sample_discriminator_validation", lambda *a: None)
    metrics = update.update_discriminator(online, optimizer, Experts(), Scenes(), torch.empty(0, dtype=torch.long),
                                          torch.tensor([8]), args, PolicyWindowHistory(0, 0), target_discriminator=target)
    assert metrics["discriminator/target_updates"] == 0 and not optimizer.state
    torch.testing.assert_close(target.state_dict(), before, rtol=0, atol=0)


def test_polyak_cadence_is_independent_of_backward_microbatching(monkeypatch):
    args = arguments("--target-discriminator", "--mlp-discriminator", "--discriminator-batch-size", "5")
    original = build_discriminator(args, 2)
    monkeypatch.setattr(update, "sample_discriminator_validation", lambda *a: None)
    results = []
    for microbatch in (5, 2):
        online = copy.deepcopy(original)
        target = create_target_discriminator(online, args)
        args.discriminator_update_microbatch_size = microbatch
        torch.manual_seed(79)
        metrics = update.update_discriminator(online, torch.optim.SGD(online.parameters(), lr=.01), Experts(), Scenes(),
                                              torch.arange(5), torch.tensor([8]), args, PolicyWindowHistory(0, 0),
                                              target_discriminator=target)
        assert metrics["discriminator/target_updates"] == 1
        results.append((online.state_dict(), target.state_dict()))
    torch.testing.assert_close(results[0], results[1], rtol=1e-5, atol=1e-7)


class Score(nn.Module):
    def __init__(self, factorized=False):
        super().__init__()
        self.factorized = factorized
        self.bias = nn.Parameter(torch.tensor([0., -.5] if factorized else [0.]))

    def forward(self, windows):
        values = windows[:, -1, :1] + self.bias
        return values if self.factorized else values[:, 0]


@pytest.mark.parametrize("reward_flags", [[], ["--amp"], ["--discriminator-log-reward"], ["--exp-log-odds-reward"]])
@pytest.mark.parametrize("factorized", [False, True])
@pytest.mark.parametrize("shaping", [False, True])
def test_exploitation_metrics_compare_fresh_live_rewards_without_changing_them(reward_flags, factorized, shaping):
    args = arguments("--target-discriminator", *reward_flags, *(["--factorize"] if factorized else []),
                     *(["--discriminator-potential-shaping"] if shaping else []),
                     "--discriminator-inference-microbatch-size", "3")
    args.reward_gamma = .9
    online = Score(factorized)
    target = create_target_discriminator(online, args)
    with torch.no_grad():
        online.bias.add_(1)
    current, final = torch.tensor([[-1.], [-2.], [2.], [3.]]), torch.tensor([[0.], [-1.], [.5], [-.5]])
    scenes = TransitionRollout(current, 1, 1, 2, potential_shaping=shaping)
    terminated = torch.tensor([False, True, False, False])
    scenes.record(0, current, final, final, torch.tensor([True, True, False, False]), terminated)
    baseline, logged = SimpleNamespace(rewards=torch.zeros(1, 4)), SimpleNamespace(rewards=torch.zeros(1, 4))
    imitation_rewards(target, scenes, baseline, args)
    rng = torch.get_rng_state()
    state = copy.deepcopy(online.state_dict())
    metrics = imitation_rewards(target, scenes, logged, args, online_discriminator=online)
    torch.testing.assert_close(logged.rewards, baseline.rewards, rtol=0, atol=0)
    assert torch.equal(rng, torch.get_rng_state()) and online.training and not target.training
    torch.testing.assert_close(online.state_dict(), state, rtol=0, atol=0)
    windows = scenes.transitions(torch.arange(4))
    live = ~terminated
    target_scores, online_scores = target(windows).detach(), online(windows).detach()
    mapping = dict(exp_log_odds_reward=args.exp_log_odds_reward, discriminator_log_reward=args.discriminator_log_reward,
                   amp=args.amp, multiplicative_reward=args.multiplicative_reward)
    gap = discriminator_rewards(target_scores, **mapping)[live] - discriminator_rewards(online_scores, **mapping)[live]
    assert metrics["discriminator/exploitation_samples"] == 3  # Includes the real timeout, not the goal.
    assert metrics["discriminator/exploitation_valid"] == 1
    assert metrics["discriminator/exploitation_gap"] == pytest.approx(gap.mean().item())
    assert metrics["discriminator/exploitation_gap_abs_mean"] == pytest.approx(gap.abs().mean().item())
    assert metrics["discriminator/exploitation_gap_rms"] == pytest.approx(gap.square().mean().sqrt().item())
    assert metrics["discriminator/exploitation_gap_positive_fraction"] == pytest.approx((gap > 0).float().mean().item())
    assert metrics["discriminator/target_rollout_score"] == pytest.approx(target_scores[live].mean().item())
    assert metrics["discriminator/online_rollout_score"] == pytest.approx(online_scores[live].mean().item())
    assert all(parameter.grad is None for parameter in online.parameters())


def test_amp_exploitation_reward_gap_does_not_confuse_score_overshoot_with_a_better_reward():
    args = arguments("--target-discriminator", "--amp")
    online = Score()
    online.bias.data.fill_(-3)
    target = create_target_discriminator(online, args)
    online.bias.data.fill_(-1)
    scenes = TransitionRollout(torch.zeros(1, 1), 1, 1, 2)
    scenes.record(0, torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1), torch.tensor([False]))
    metrics = imitation_rewards(target, scenes, SimpleNamespace(rewards=torch.zeros(1, 1)), args,
                                 online_discriminator=online)
    assert metrics["discriminator/exploitation_logit_gap"] == 2
    assert metrics["discriminator/exploitation_gap"] == -1  # AMP peaks at -1, not increasingly negative logits.


def test_exploitation_logging_is_invalid_not_nan_without_live_windows():
    args = arguments("--target-discriminator")
    online = Score()
    target = create_target_discriminator(online, args)
    scenes = TransitionRollout(torch.zeros(1, 1), 1, 1, 2)
    done = torch.tensor([True])
    scenes.record(0, torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1), done, done)
    metrics = imitation_rewards(target, scenes, SimpleNamespace(rewards=torch.zeros(1, 1)), args,
                                 online_discriminator=online)
    assert metrics["discriminator/exploitation_samples"] == 0
    assert metrics["discriminator/exploitation_valid"] == 0
    assert metrics["discriminator/exploitation_gap"] == 0


def test_exploitation_diagnostics_do_not_update_online_buffers_modes_or_rng():
    class StatefulScore(nn.Module):
        factorized = False

        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm1d(1)
            self.dropout = nn.Dropout(.5)

        def forward(self, windows):
            return self.dropout(self.bn(windows[:, -1, :1]))[:, 0]

    args = arguments("--target-discriminator", "--discriminator-inference-microbatch-size", "2")
    online = StatefulScore()
    online.dropout.eval()  # Preserve mixed modes, not just the root's training flag.
    target = create_target_discriminator(online, args)
    current = torch.arange(5.)[:, None]
    scenes = TransitionRollout(current, 1, 1, 2)
    scenes.record(0, current, current + 1, current + 1, torch.zeros(5, dtype=torch.bool))
    before = copy.deepcopy(online.state_dict())
    rng = torch.get_rng_state()
    imitation_rewards(target, scenes, SimpleNamespace(rewards=torch.zeros(1, 5)), args, online_discriminator=online)
    torch.testing.assert_close(online.state_dict(), before, rtol=0, atol=0)
    assert online.training and online.bn.training and not online.dropout.training
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("architecture", [[], ["--gru"], ["--mlp-discriminator"]])
@pytest.mark.parametrize("head", [[], ["--ego-only"], ["--factorize"]])
@pytest.mark.parametrize("relative", [False, True])
def test_target_and_online_frame_caches_match_full_window_reward_comparisons(architecture, head, relative):
    args = arguments("--target-discriminator", "--amp", "--discriminator-potential-shaping", *architecture, *head,
                     *(["--discriminator-relative"] if relative else []), "--discriminator-inference-microbatch-size", "2")
    args.reward_gamma = .99
    online = build_discriminator(args, 2)
    target = create_target_discriminator(online, args)
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(.01)
    scenes = TransitionRollout(torch.zeros(2, 51), 3, 51, 2, potential_shaping=True)
    for step in range(3):
        current = torch.randn(2, 51)
        done = torch.tensor([step == 1, False])
        scenes.record(step, current, current + 1, current + 2, done, done)

    class Direct(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.factorized = model.factorized
            self.head_name = getattr(model, "head_name", "global")

        def forward(self, windows):
            return self.model(windows)

    cached, direct = SimpleNamespace(rewards=torch.zeros(3, 2)), SimpleNamespace(rewards=torch.zeros(3, 2))
    left = imitation_rewards(target, scenes, cached, args, online_discriminator=online)
    right = imitation_rewards(Direct(target), scenes, direct, args, online_discriminator=Direct(online))
    torch.testing.assert_close(cached.rewards, direct.rewards)
    for name, value in left.items():
        assert right[name] == pytest.approx(value, rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("algorithm", ["gaifo", "dac"])
@pytest.mark.parametrize("amp", [False, True])
def test_target_checkpoint_roundtrip_preserves_averaging_history(tmp_path, algorithm, amp):
    args = arguments("--target-discriminator", "--mlp-discriminator", "--factorize", *(["--amp"] if amp else []))
    if algorithm == "dac":
        args.discriminator_objective += "-absorbing-v1"
    online = AbsorbingDiscriminator(args, 2) if algorithm == "dac" else build_discriminator(args, 2)
    target = create_target_discriminator(online, args)
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(2)
    polyak_update(target, online, args.target_discriminator_tau)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, algorithm, args, model, optimizer, 10, discriminator=online.state_dict(),
                    target_discriminator=target.state_dict())
    payload = load_checkpoint(path, algorithm, args, model, optimizer)
    restored = create_target_discriminator(online, args)
    restored.load_state_dict(payload["target_discriminator"])
    torch.testing.assert_close(restored.state_dict(), target.state_dict(), rtol=0, atol=0)
    assert any(not torch.equal(parameter, dict(online.named_parameters())[name])
               for name, parameter in restored.named_parameters())
    polyak_update(target, online, args.target_discriminator_tau)
    polyak_update(restored, online, args.target_discriminator_tau)
    torch.testing.assert_close(restored.state_dict(), target.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("old_target", [None, False, True])
def test_resume_cannot_silently_switch_target_mode(tmp_path, old_target):
    original = arguments(*(["--target-discriminator"] if old_target else []))
    requested = arguments(*([] if old_target else ["--target-discriminator"]))
    if old_target is None:
        for key in ("target_discriminator", "target_discriminator_tau", "target_discriminator_span", "target_discriminator_objective"):
            delattr(original, key)
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(original)}, path)
    with pytest.raises(ValueError, match="target discriminator mode differs"):
        load_checkpoint(path, "gaifo", requested, None, None)


def test_missing_and_nonfinite_target_checkpoint_states_are_rejected(tmp_path):
    args = arguments("--target-discriminator")
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "checkpoint.pt"
    with pytest.raises(ValueError, match="missing the Polyak"):
        save_checkpoint(path, "gaifo", args, model, optimizer, 0)
    payload = {"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(args)}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="missing the Polyak"):
        load_checkpoint(path, "gaifo", args, None, None)
    payload["target_discriminator"] = {}
    payload["target_discriminator"] = {"bad": torch.tensor(float("nan"))}
    torch.save(payload, path)
    with pytest.raises(FloatingPointError, match="target_discriminator"):
        load_checkpoint(path, "gaifo", args, None, None)


def test_resume_can_raise_tau_without_resetting_the_averaged_weights(tmp_path):
    original = arguments("--target-discriminator")
    requested = arguments("--target-discriminator", "--target-discriminator-span", "4")
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    target = copy.deepcopy(model).requires_grad_(False)
    target.weight.data.add_(3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, "gaifo", original, model, optimizer, 10, target_discriminator=target.state_dict())
    with pytest.warns(UserWarning, match="tau changed.*without resetting"):
        payload = load_checkpoint(path, "gaifo", requested, model, optimizer)
    restored = create_target_discriminator(model, requested)
    restored.load_state_dict(payload["target_discriminator"])
    torch.testing.assert_close(restored.state_dict(), target.state_dict(), rtol=0, atol=0)


def test_legacy_checkpoints_remain_compatible_when_target_mode_is_disabled(tmp_path):
    original = arguments()
    for name in ("target_discriminator", "target_discriminator_tau", "target_discriminator_span", "target_discriminator_objective"):
        delattr(original, name)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, "gaifo", original, model, optimizer, 10)
    payload = load_checkpoint(path, "gaifo", arguments("--target-discriminator-span", "4"), model, optimizer)
    assert "target_discriminator" not in payload
