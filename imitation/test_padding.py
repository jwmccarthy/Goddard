from types import SimpleNamespace

import pytest
import torch
from torch import nn

from imitation.absorbing import AbsorbingDiscriminator
from imitation.evaluation import evaluate_discriminator, sample_discriminator_validation
from imitation.models import build_discriminator
from imitation.rollout import TransitionRollout
from imitation.test_amp import arguments
from imitation.update import train_discriminator_batch
from imitation.windows import balanced_window_batch


@pytest.mark.parametrize("length,stride,partial", [(2, 1, 0), (2, 3, 2), (8, 3, 20)])
def test_reset_shortcut_is_removed_and_adjacent_pairs_never_have_it(length, stride, partial):
    steps = (length - 1) * stride + 1
    scenes = TransitionRollout(torch.zeros(1, 51), steps, 51, length, decision_stride=stride)
    for step in range(steps):
        scenes.record(step, torch.zeros(1, 51), torch.zeros(1, 51), torch.zeros(1, 51), torch.tensor([False]))
    generated = scenes.transitions(torch.arange(steps))
    expert = torch.zeros_like(generated)
    assert torch.isnan(generated[..., 0]).any(dim=1).sum().item() == partial
    matched = balanced_window_batch(generated, expert)
    # Identical physical observations now yield identical model inputs, even
    # when only the generated side started with incomplete reset history.
    torch.testing.assert_close(matched[:steps], matched[steps:], equal_nan=True)


def test_padding_union_matches_both_sources_without_fabricating_history_or_mutating_inputs():
    generated = torch.randn(4, 4, 51)
    expert = torch.randn_like(generated)
    generated[0, :2] = float("nan")
    expert[1, :3] = float("nan")
    generated[2, :1] = float("nan")
    expert[2, :2] = float("nan")
    before = generated.clone(), expert.clone()
    rng = torch.get_rng_state()
    batch = balanced_window_batch(generated, expert)
    mask = torch.isnan(generated[..., 0]) | torch.isnan(expert[..., 0])
    assert torch.equal(torch.isnan(batch[:4, :, 0]), mask)
    assert torch.equal(torch.isnan(batch[4:, :, 0]), mask)
    torch.testing.assert_close(batch[:4][~mask], generated[~mask])
    torch.testing.assert_close(batch[4:][~mask], expert[~mask])
    torch.testing.assert_close((generated, expert), before, equal_nan=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.isnan(batch[:, -1]).any()


def test_full_and_empty_batches_are_unchanged():
    for count in (0, 3):
        generated, expert = torch.randn(count, 2, 51), torch.randn(count, 2, 51)
        torch.testing.assert_close(balanced_window_batch(generated, expert), torch.cat((generated, expert)))
    with pytest.raises(ValueError, match="equal source window shapes"):
        balanced_window_batch(torch.zeros(2, 2, 51), torch.zeros(3, 2, 51))


@pytest.mark.parametrize("architecture", [[], ["--gru"], ["--mlp-discriminator"]])
@pytest.mark.parametrize("head", [[], ["--ego-only"], ["--factorize"]])
@pytest.mark.parametrize("relative", [False, True])
def test_absorbing_indicator_bookkeeping_cannot_leak_through_padded_tokens(architecture, head, relative):
    args = arguments(*architecture, *head, *(["--discriminator-relative"] if relative else []))
    model = AbsorbingDiscriminator(args, 2)
    generated, expert = torch.randn(3, 2, 52), torch.randn(3, 2, 52)
    generated[..., -1] = expert[..., -1] = 0
    generated[0, 0] = float("nan")
    # A self-loop paired with a partial live window may lose physical history,
    # but its original indicator still identifies it as a loop in diagnostics.
    expert[0] = 0
    expert[0, :, -1] = 1
    expert[1, 0] = float("nan")
    generated[1] = 0
    generated[1, :, -1] = 1
    matched = balanced_window_batch(generated, expert, absorbing=True)
    torch.testing.assert_close(matched[..., -1], torch.cat((generated, expert))[..., -1], equal_nan=True)
    all_nan_padding = matched.masked_fill(torch.isnan(matched[..., :1]), float("nan"))
    with torch.no_grad():
        torch.testing.assert_close(model(matched), model(all_nan_padding))


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("microbatch", [1, 3, 4])
def test_training_balances_masks_before_microbatching_and_motion_noise(amp, microbatch):
    class Recorder(nn.Module):
        factorized = False

        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(()))
            self.inputs = []

        def forward(self, windows):
            self.inputs.append(windows.detach().clone())
            return self.bias.expand(len(windows))

    args = arguments(*(["--amp"] if amp else []), "--discriminator-update-microbatch-size", str(microbatch))
    generated, expert = torch.randn(4, 2, 51), torch.randn(4, 2, 51)
    generated[:2, 0] = float("nan")
    expert[2, 0] = float("nan")
    before = generated.clone(), expert.clone()
    model = Recorder()
    train_discriminator_batch(model, torch.optim.SGD(model.parameters(), lr=.01), generated, expert, args)
    for windows in model.inputs:
        half = len(windows) // 2
        assert torch.equal(torch.isnan(windows[:half, :, 0]), torch.isnan(windows[half:, :, 0]))
    torch.testing.assert_close((generated, expert), before, equal_nan=True)


def test_validation_uses_the_same_mask_policy_and_reports_equal_padding_fractions():
    generated, expert = torch.randn(4, 2, 51), torch.randn(4, 2, 51)
    generated[:2, 0] = float("nan")
    args = arguments("--mlp-discriminator", "--discriminator-heldout-size", "4")
    model = build_discriminator(args, 2)
    experts = SimpleNamespace(heldout_starts=torch.arange(4), sample_heldout=lambda count: expert[:count])
    scenes = SimpleNamespace(transitions=lambda indices: generated[indices])
    validation = sample_discriminator_validation(model, experts, scenes, torch.arange(4), args)
    assert torch.equal(torch.isnan(validation[:4, :, 0]), torch.isnan(validation[4:, :, 0]))
    metrics = evaluate_discriminator(model, experts, scenes, torch.arange(4), args, validation)
    assert metrics["discriminator/generated_padded_fraction"] == .5
    assert metrics["discriminator/expert_padded_fraction"] == .5


def test_matching_keeps_absorbing_validation_category_counts():
    args = arguments("--mlp-discriminator", "--discriminator-heldout-size", "2")
    model = AbsorbingDiscriminator(args, 2)
    generated, expert = torch.zeros(2, 2, 52), torch.zeros(2, 2, 52)
    generated[0, 0] = float("nan")
    expert[0, :, -1] = 1  # Loop paired with a short live window.
    generated[1, -1, -1] = 1  # Terminal.
    validation = balanced_window_batch(generated, expert, absorbing=True)
    metrics = evaluate_discriminator(model, None, None, None, args, validation)
    assert metrics["discriminator/absorbing_expert_count"] == 1
    assert metrics["discriminator/terminal_generated_count"] == 1
    assert metrics["discriminator/live_generated_count"] == metrics["discriminator/live_expert_count"] == 1
