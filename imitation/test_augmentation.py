import argparse
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from imitation import update
from imitation.absorbing import AbsorbingDiscriminator, AbsorbingRollout
from imitation.augmentation import add_motion_noise
from imitation.models import relative_window
from imitation.policy_history import PolicyWindowHistory
from imitation.rewards import imitation_rewards
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint


def arguments(*flags):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=True)
    add_gaifo_arguments(parser)
    args = parser.parse_args(["--frame-embedding", "8", "--discriminator-hidden", "8",
                              "--discriminator-window-length", "2", "--mlp-discriminator", *flags])
    resolve_gaifo_defaults(args)
    return args


def motion_columns(n_cars):
    return [*range(9), *(9 + car * 21 + field for car in range(n_cars) for field in range(9))]


def test_noise_default_is_small_and_zero_disables_without_drawing_randomness():
    assert arguments().discriminator_motion_noise_scale == .001
    assert arguments("--discriminator-motion-noise-scale", "0").discriminator_motion_noise_scale == 0
    windows = torch.randn(2, 2, 51)
    rng = torch.get_rng_state()
    assert add_motion_noise(windows, 0) is windows
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize("scale", ["-1", "nan", "inf", "-inf"])
def test_noise_scale_must_be_finite_and_nonnegative(scale):
    with pytest.raises(ValueError, match="noise scale must be finite and nonnegative"):
        arguments(f"--discriminator-motion-noise-scale={scale}")
    with pytest.raises(ValueError, match="noise scale must be finite and nonnegative"):
        add_motion_noise(torch.zeros(2, 2, 51), float(scale))


@pytest.mark.parametrize("n_cars", [2, 4, 6])
@pytest.mark.parametrize("absorbing", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_noise_only_touches_live_motion_and_never_mutates_inputs(n_cars, absorbing, dtype, monkeypatch):
    width = 9 + n_cars * 21 + int(absorbing)
    windows = torch.zeros(4, 3, width, dtype=dtype)
    windows[0, 0] = float("nan")
    if absorbing:
        windows[1, -1, -1] = 1
        windows[2, :, -1] = 1
    original = windows.clone()
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.ones_like(tensor))
    noisy = add_motion_noise(windows, .01)
    torch.testing.assert_close(windows, original, equal_nan=True)
    assert noisy.data_ptr() != windows.data_ptr()
    assert noisy.dtype == windows.dtype and noisy.device == windows.device
    selected = motion_columns(n_cars)
    untouched = [i for i in range(width) if i not in selected]
    torch.testing.assert_close(noisy[..., untouched], original[..., untouched], equal_nan=True)
    live = ~torch.isnan(original[..., 0])
    if absorbing:
        live &= original[..., -1] < .5
    torch.testing.assert_close(noisy[..., selected][live], original[..., selected][live] + .01)
    torch.testing.assert_close(noisy[~live], original[~live], equal_nan=True)


def test_noise_distribution_is_equal_for_both_sources_and_independent_across_frames_and_bodies():
    torch.manual_seed(96)
    windows = torch.zeros(8192, 2, 51)
    noisy = add_motion_noise(windows, .003)
    for source in noisy.chunk(2):
        motion = source[..., motion_columns(2)]
        assert abs(motion.mean().item()) < .00003
        assert motion.std().item() == pytest.approx(.003, rel=.02)
    difference = noisy[:, 1, 0] - noisy[:, 0, 0]
    assert difference.std().item() == pytest.approx(2 ** .5 * .003, rel=.03)
    assert not torch.equal(noisy[..., :9], noisy[..., 9:18])
    assert not torch.equal(noisy, add_motion_noise(windows, .003))


def test_motion_noise_is_applied_before_relative_encoding(monkeypatch):
    windows = torch.zeros(2, 2, 51)
    values = torch.tensor([1., 3.]).view(1, 2, 1)
    monkeypatch.setattr(torch, "randn_like", lambda tensor: values.expand_as(tensor))
    transformed = relative_window(add_motion_noise(windows, .01))
    torch.testing.assert_close(transformed[:, 0, 0], torch.full((2,), .01))
    # Independent endpoint noise becomes a difference, not another absolute draw.
    torch.testing.assert_close(transformed[:, 1, 0], torch.full((2,), .02))


def test_training_noises_both_sources_and_shares_the_perturbation_between_heads(monkeypatch):
    class Capture(torch.nn.Module):
        factorized = True

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.1))
            self.calls = []

        def head_score(self, windows, head):
            self.calls.append((head, windows.detach().clone()))
            return windows[..., 0].sum(-1) * self.weight

    args = arguments("--factorize", "--discriminator-update-microbatch-size", "2")
    model = Capture()
    generated, expert = torch.zeros(4, 2, 51), torch.zeros(4, 2, 51)
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.ones_like(tensor))
    loss = update.train_discriminator_batch(model, torch.optim.SGD(model.parameters(), lr=.01),
                                            generated, expert, args)
    assert torch.isfinite(loss)
    assert len(model.calls) == 4
    for i in (0, 2):
        left, right = model.calls[i:i + 2]
        assert left[0] == 0 and right[0] == 1
        torch.testing.assert_close(left[1], right[1])
        # Each microbatch contains two policy followed by two expert windows.
        torch.testing.assert_close(left[1][:, :, 0], torch.full((4, 2), .001))
    assert generated.eq(0).all() and expert.eq(0).all()


def test_validation_reuses_fixed_noise_and_reports_clean_accuracy_without_noising_history(monkeypatch):
    validation = torch.zeros(4, 2, 51)
    seen = []
    args = arguments("--factorize", "--discriminator-batch-size", "2")
    monkeypatch.setattr(update, "sample_discriminator_validation", lambda *args: validation)
    monkeypatch.setattr(torch, "randn_like", lambda tensor: torch.ones_like(tensor))

    def evaluate(*args):
        seen.append(args[-1])
        return {"discriminator/heldout_accuracy": .5,
                "discriminator/ego_heldout_accuracy": .5,
                "discriminator/global_heldout_accuracy": .5}

    monkeypatch.setattr(update, "evaluate_discriminator", evaluate)
    monkeypatch.setattr(update, "train_discriminator_batch", lambda *args: torch.tensor(0.))
    scenes = SimpleNamespace(transitions=lambda indices: torch.zeros(len(indices), 2, 51))
    experts = SimpleNamespace(sample=lambda count: torch.zeros(count, 2, 51))
    history = PolicyWindowHistory(8, 2)
    metrics = update.update_discriminator(SimpleNamespace(factorized=True), None, experts, scenes, torch.arange(2),
                                          torch.arange(2), args, history)
    assert len(seen) == 3 and seen[0] is seen[1]
    torch.testing.assert_close(seen[0][..., 0], torch.full((4, 2), .001))
    assert seen[2] is validation and validation.eq(0).all()
    assert metrics["discriminator/clean_heldout_accuracy"] == .5
    assert metrics["discriminator/ego_clean_heldout_accuracy"] == .5
    assert metrics["discriminator/global_clean_heldout_accuracy"] == .5
    assert metrics["discriminator/motion_noise_scale"] == .001
    assert history.windows[:history.size].eq(0).all()


@pytest.mark.parametrize("relative", [False, True])
def test_policy_reward_scoring_stays_clean_deterministic_and_cache_compatible(relative):
    args = arguments("--factorize", "--discriminator-log-reward",
                     *(["--discriminator-relative"] if relative else []))
    model = AbsorbingDiscriminator(args, 2)
    scenes = AbsorbingRollout(torch.zeros(2, 51), 3, 51, 2)
    for step in range(3):
        current = torch.randn(2, 51)
        done = torch.zeros(2, dtype=torch.bool)
        scenes.record(step, current, current + .01, current + .01, done, done)
    before = scenes.all_frames().clone()
    positive = SimpleNamespace(rewards=torch.zeros(3, 2))
    zero = SimpleNamespace(rewards=torch.zeros(3, 2))
    rng = torch.get_rng_state()
    imitation_rewards(model, scenes, positive, args, include_terminal=True)
    assert torch.equal(rng, torch.get_rng_state())
    args.discriminator_motion_noise_scale = 0
    imitation_rewards(model, scenes, zero, args, include_terminal=True)
    torch.testing.assert_close(positive.rewards, zero.rewards)
    torch.testing.assert_close(scenes.all_frames(), before, equal_nan=True)


@pytest.mark.parametrize("old,new", [(None, .001), (.001, .002), (.001, .001), (None, 0)])
def test_checkpoint_records_noise_and_warns_when_augmentation_changes(tmp_path, old, new):
    args = arguments("--discriminator-motion-noise-scale", str(new))
    args.discriminator_heldout_replays = ["heldout-match"]
    config = copy.deepcopy(vars(args))
    if old is None:
        config.pop("discriminator_motion_noise_scale")
    else:
        config["discriminator_motion_noise_scale"] = old
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    payload = {"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": config,
               "agent": model.state_dict(), "optimizer": optimizer.state_dict(),
               "torch_rng": torch.get_rng_state(), "cuda_rng": [],
               "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    if (old or 0) != new:
        with pytest.warns(UserWarning, match="motion-noise scale changed"):
            load_checkpoint(path, "gaifo", args, model, optimizer)
    else:
        load_checkpoint(path, "gaifo", args, model, optimizer)
