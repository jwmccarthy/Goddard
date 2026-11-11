import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from imitation.absorbing import AbsorbingDiscriminator, AbsorbingRollout
from imitation.dataset import ExpertDataset
from imitation.models import build_discriminator
from imitation.policy_history import PolicyWindowHistory
from imitation.rewards import imitation_rewards
from imitation.rollout import TransitionRollout
from replays.layout import team_replay_row_size
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint


def arguments(*flags):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=True)
    add_gaifo_arguments(parser)
    args = parser.parse_args(["--frame-embedding", "8", "--discriminator-hidden", "8",
                              "--discriminator-window-length", "2", *flags])
    resolve_gaifo_defaults(args)
    return args


def test_dense_stride_is_opt_in_and_requires_matching_frameskip():
    assert arguments().discriminator_decision_stride == 1
    args = arguments("--discriminator-decision-stride", "3")
    assert args.discriminator_observation_hz == 10
    assert args.discriminator_window_span_ticks == 12
    for stride in ("0", "-1"):
        with pytest.raises(ValueError, match="stride must be positive"):
            arguments("--discriminator-decision-stride", stride)
    with pytest.raises(ValueError, match="requires --frameskip 4"):
        arguments("--discriminator-decision-stride", "3", "--frameskip", "12")


@pytest.mark.parametrize("flags", [
    ["--sparse-replay-dir", "unused"], ["--sparse-period-dir", "unused"], ["--sparse-hold"],
    ["--sparse-hold-decision-stride", "3"], ["--sparse-hold-velocity-quantum", ".01"],
    ["--sparse-hold-position-quantum", ".01"], ["--sparse-hold-pool-size", "32"],
    ["--sparse-hold-pool-ticks", "64"], ["--sparse-horizon-ticks", "8"],
    ["--sparse-max-updates", "8"], ["--sparse-position-noise", ".1"],
])
def test_removed_measurement_flags_are_rejected(flags):
    with pytest.raises(SystemExit):
        arguments(*flags)


def test_dense_expert_sampling_uses_bookends_and_right_aligned_short_prefixes(tmp_path):
    rows = np.zeros((8, team_replay_row_size(1)), dtype=np.float32)
    rows[:, 0] = np.arange(8)
    np.save(tmp_path / "match.npy", rows)
    dataset = ExpertDataset(tmp_path, 1, None, "cpu", window_length=2, decision_stride=3)
    assert dataset.history_length == 4
    assert dataset.train_lengths.tolist() == [2, 3, 4, 4, 4, 4, 4]
    windows = dataset.sample(200)
    end = windows[:, -1, 0]
    full = end >= 3
    assert full.any() and (~full).any()
    assert torch.isnan(windows[~full, 0]).all()
    torch.testing.assert_close(windows[full, 1, 0] - windows[full, 0, 0], torch.full_like(end[full], 3))


def test_dense_experts_reject_invalid_frames_between_bookends(tmp_path):
    rows = np.zeros((12, team_replay_row_size(1)), dtype=np.float32)
    rows[:, 0] = np.arange(12)
    rows[4, -1] = 1
    np.save(tmp_path / "match.npy", rows)
    dataset = ExpertDataset(tmp_path, 1, None, "cpu", window_length=2, decision_stride=3)
    windows = dataset.sample(500)
    ends = windows[:, -1, 0]
    assert not ((ends >= 4) & (ends <= 7)).any()
    assert (ends == 8).any()


def test_strided_dac_experts_preserve_goals_loops_and_replay_boundaries(tmp_path):
    for replay in range(2):
        rows = np.zeros((8, team_replay_row_size(1)), dtype=np.float32)
        rows[:, 0] = np.arange(8) + replay * 100
        rows[-1, 0] = float("nan")
        rows[-1, -1] = 1
        path = tmp_path / f"match{replay}.npy"
        np.save(path, rows)
        path.with_suffix(".json").write_text(json.dumps({"goal_terminal": True}))
    dataset = ExpertDataset(tmp_path, 1, None, "cpu", window_length=2, decision_stride=3,
                            terminal_aware=True, heldout_size=1)
    for windows in (dataset.sample(500), dataset.sample_heldout(500)):
        terminal = (windows[:, -1, -1] == 1) & (windows[:, 0, -1] == 0)
        loops = (windows[..., -1] == 1).all(-1)
        live = (windows[..., -1] == 0).all(-1)
        assert terminal.any() and loops.any() and live.any()
        assert windows[terminal, -1, :-1].eq(0).all()
        assert (windows[terminal, 0, 0] % 100).eq(4).all()
        assert windows[loops, :, :-1].eq(0).all()
        torch.testing.assert_close(windows[live, 1, 0] - windows[live, 0, 0],
                                   torch.full((int(live.sum()),), 3.))


@pytest.mark.parametrize("potential", [False, True])
def test_strided_rollout_slides_every_decision_across_short_rollouts(potential):
    scene = TransitionRollout(torch.zeros(2, 1), 2, 1, 2, decision_stride=3, potential_shaping=potential)
    done = torch.zeros(2, dtype=torch.bool)
    for rollout in range(4):
        for step in range(2):
            decision = rollout * 2 + step
            current = torch.tensor([[float(decision)], [100. + decision]])
            scene.record(step, current, current + 1, current + 1, done)
        # Frame caches are populated after collection, as in the training loop.
        for step in range(2):
            decision = rollout * 2 + step
            current = torch.tensor([[float(decision)], [100. + decision]])
            windows = scene.transitions(torch.tensor([step * 2, step * 2 + 1]))[..., 0]
            torch.testing.assert_close(windows[:, -1], current[:, 0] + 1)
            if decision >= 2:
                torch.testing.assert_close(windows[:, 0], current[:, 0] - 2)
            else:
                assert torch.isnan(windows[:, 0]).all()
            if potential:
                previous = scene.previous_windows(torch.tensor([step * 2, step * 2 + 1]))[..., 0]
                torch.testing.assert_close(previous[:, -1], current[:, 0])
                if decision >= 3:
                    torch.testing.assert_close(previous[:, 0], current[:, 0] - 3)


@pytest.mark.parametrize("absorbing", [False, True])
@pytest.mark.parametrize("potential", [False, True])
@pytest.mark.parametrize("length,stride", [(2, 3), (4, 3), (2, 1)])
def test_strided_windows_equal_subsampled_full_interval(absorbing, potential, length, stride):
    cls = AbsorbingRollout if absorbing else TransitionRollout
    span = (length - 1) * stride + 1
    scene = cls(torch.zeros(2, 1), 2, 1, length, decision_stride=stride, potential_shaping=potential)
    reference = cls(torch.zeros(2, 1), 2, 1, span, potential_shaping=potential)
    for rollout in range(6):
        for step in range(2):
            decision = rollout * 2 + step
            current = torch.tensor([[float(decision)], [100. + decision]])
            done = torch.tensor([decision in (3, 9), decision == 7])
            terminated = done & torch.tensor([True, False])
            for target in (scene, reference):
                target.record(step, current, current + 1, current + 10, done, terminated)
        indices = torch.arange(4)
        torch.testing.assert_close(scene.transitions(indices), reference.transitions(indices)[:, ::stride],
                                   equal_nan=True)
        if potential:
            torch.testing.assert_close(scene.previous_windows(indices),
                                       reference.previous_windows(indices)[:, ::stride], equal_nan=True)


@pytest.mark.parametrize("absorbing", [False, True])
def test_strided_rollout_masks_intermediate_reset_and_keeps_terminal_timeout(absorbing):
    cls = AbsorbingRollout if absorbing else TransitionRollout
    scene = cls(torch.zeros(2, 1), 8, 1, 2, decision_stride=3)
    for step in range(8):
        current = torch.full((2, 1), float(step))
        done = torch.tensor([step == 3, step == 3])
        terminated = torch.tensor([step == 3, False])
        scene.record(step, current, current + 1, current + 20, done, terminated)
    terminal, timeout = scene.transitions(torch.tensor([6, 7]))
    assert terminal[0, 0] == timeout[0, 0] == 1
    if absorbing:
        assert terminal[-1].tolist() == [0., 1.]
        assert timeout[-1].tolist() == [23., 0.]
        loop = scene.transitions(torch.tensor([6 + scene.dones.numel()]))
        assert loop[..., -1].eq(1).all() and loop[..., :-1].eq(0).all()
    else:
        assert terminal[-1, 0] == timeout[-1, 0] == 23
    # Reset at decision 3 lies strictly between these bookends (2 and 5).
    assert torch.isnan(scene.transitions(torch.tensor([8, 9]))[:, 0]).all()
    assert torch.isnan(scene.transitions(torch.tensor([10, 11]))[:, 0]).all()
    torch.testing.assert_close(scene.transitions(torch.tensor([12, 13]))[:, :, 0],
                               torch.tensor([[4., 7.], [4., 7.]]))


@pytest.mark.parametrize("absorbing", [False, True])
@pytest.mark.parametrize("architecture", ["mlp", "gru", "transformer"])
@pytest.mark.parametrize("relative", [False, True])
def test_strided_cached_rewards_and_potentials_match_direct(absorbing, architecture, relative):
    flags = {"mlp": ["--mlp-discriminator"], "gru": ["--gru"], "transformer": []}[architecture]
    args = arguments(*flags, *(["--discriminator-relative"] if relative else []), "--factorize",
                     "--discriminator-decision-stride", "3", "--discriminator-log-reward",
                     "--discriminator-potential-shaping", "--discriminator-inference-microbatch-size", "3")
    args.reward_gamma = .99
    model = AbsorbingDiscriminator(args, 2) if absorbing else build_discriminator(args, 2)
    cls = AbsorbingRollout if absorbing else TransitionRollout
    scene = cls(torch.zeros(2, 51), 5, 51, 2, decision_stride=3, potential_shaping=True)
    for step in range(5):
        current = torch.randn(2, 51)
        done = torch.tensor([step == 1, False])
        scene.record(step, current, current + 1, current + 2, done, done)

    class Direct(torch.nn.Module):
        def forward(self, windows):
            return model(windows)

    cached, direct = (SimpleNamespace(rewards=torch.zeros(5, 2)) for _ in range(2))
    imitation_rewards(model, scene, cached, args, include_terminal=absorbing)
    imitation_rewards(Direct(), scene, direct, args, include_terminal=absorbing)
    torch.testing.assert_close(cached.rewards, direct.rewards)
    assert torch.isfinite(cached.rewards).all()


def test_policy_history_stores_only_selected_dense_observations():
    scene = TransitionRollout(torch.zeros(1, 1), 4, 1, 2, decision_stride=3)
    for step in range(4):
        current = torch.tensor([[float(step)]])
        scene.record(step, current, current + 1, current + 1, torch.tensor([False]))
    history = PolicyWindowHistory(2, 2)
    assert history.add(scene, torch.tensor([2, 3])) == 2
    assert history.windows.shape == (2, 2, 1)
    torch.testing.assert_close(history.windows[:, 1, 0] - history.windows[:, 0, 0], torch.full((2,), 3.))


@pytest.mark.parametrize("old_stride,new_stride", [(1, 3), (3, 1)])
def test_checkpoint_rejects_changed_discriminator_spacing(tmp_path, old_stride, new_stride):
    original = arguments("--discriminator-decision-stride", str(old_stride))
    config = vars(original).copy()
    if old_stride == 1:
        config.pop("discriminator_decision_stride")  # Legacy checkpoints mean stride 1.
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": config}, path)
    with pytest.raises(ValueError, match="decision stride differs"):
        load_checkpoint(path, "gaifo", arguments("--discriminator-decision-stride", str(new_stride)), None, None)
