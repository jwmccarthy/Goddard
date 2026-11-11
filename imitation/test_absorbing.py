import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from imitation.absorbing import (
    AbsorbingDiscriminator,
    AbsorbingRollout,
    AbsorbingValue,
    absorbing_diagnostics,
    split_absorbing_generated,
    update_absorbing_value,
)
from imitation.dataset import ExpertDataset
from imitation.evaluation import evaluate_discriminator
from imitation.models import relative_window
from imitation.rewards import imitation_rewards
from imitation.update import train_discriminator_batch
from replays.layout import team_replay_row_size
from training.rollout import Rollout, compute_gae


def arguments(factorize=True, gru=True):
    return SimpleNamespace(
        factorize=factorize, gru=gru, frame_embedding=8, discriminator_hidden=16,
        discriminator_window_length=4, discriminator_update_microbatch_size=2,
        discriminator_inference_microbatch_size=16, bf16=False, max_grad_norm=.5
    )


@pytest.mark.parametrize("factorize", [False, True])
@pytest.mark.parametrize("architecture", ["transformer", "gru", "mlp"])
@pytest.mark.parametrize("amp", [False, True])
def test_absorbing_cached_scores_and_training_are_finite(factorize, architecture, amp):
    args = arguments(factorize, architecture == "gru")
    args.amp = amp
    args.mlp_discriminator = architecture == "mlp"
    model = AbsorbingDiscriminator(args, 2)
    scenes = AbsorbingRollout(torch.zeros(2, 51), 3, 51, 4)
    for step in range(3):
        current = torch.randn(2, 51)
        done = torch.tensor([step == 1, False])
        scenes.record(step, current, current + 1, current + 2, done, done)
    indices = torch.arange(6)
    windows = scenes.transitions(indices)
    window_indices, padding = scenes.window_indices(indices)
    padding |= torch.isnan(windows[..., 0])
    encoded = model.encode_frames(scenes.all_frames()[:, None])
    cached = model.score_encoded([part[:, 0][window_indices] for part in encoded], padding)
    torch.testing.assert_close(cached, model(windows))
    assert windows[2, -1, -1] == 1
    assert (windows[2, -1, :-1] == 0).all()
    assert windows[3, -1, -1] == 0
    loops = scenes.transitions(torch.tensor([8]))
    assert (loops[..., :-1] == 0).all() and (loops[..., -1] == 1).all()
    optimizer = torch.optim.Adam(model.parameters())
    loss = train_discriminator_batch(model, optimizer, windows[:4], windows[2:6], args)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("architecture", ["transformer", "gru", "mlp"])
@pytest.mark.parametrize("amp", [False, True])
def test_absorbing_potential_rewards_cached_match_direct(architecture, amp):
    args = arguments(True, architecture == "gru")
    args.amp = amp
    args.mlp_discriminator = architecture == "mlp"
    args.discriminator_potential_shaping = True
    args.discriminator_potential_weight = 0.5
    args.discriminator_log_reward = not amp
    args.exp_log_odds_reward = False
    args.multiplicative_reward = False
    args.reward_plus = False
    args.goal_reward = False
    args.reward_gamma = 0.99
    model = AbsorbingDiscriminator(args, 2)
    scenes = AbsorbingRollout(torch.zeros(2, 51), 3, 51, 4, potential_shaping=True)
    for step in range(3):
        current = torch.randn(2, 51)
        done = torch.tensor([step == 1, False])
        scenes.record(step, current, current + 1, current + 2, done, done)

    class DirectScore(torch.nn.Module):

        def forward(self, windows):
            return model(windows)

    cached = SimpleNamespace(rewards=torch.zeros(3, 2))
    direct = SimpleNamespace(rewards=torch.zeros(3, 2))
    imitation_rewards(model, scenes, cached, args, include_terminal=True)
    imitation_rewards(DirectScore(), scenes, direct, args, include_terminal=True)
    torch.testing.assert_close(cached.rewards, direct.rewards)


@pytest.mark.parametrize("architecture", ["transformer", "gru", "mlp"])
@pytest.mark.parametrize("factorize", [False, True])
def test_relative_absorbing_model_preserves_absorbing_indicator(architecture, factorize):
    args = arguments(factorize, architecture == "gru")
    args.mlp_discriminator = architecture == "mlp"
    absolute = AbsorbingDiscriminator(args, 2)
    args.discriminator_relative = True
    relative = AbsorbingDiscriminator(args, 2)
    relative.load_state_dict(absolute.state_dict())
    windows = torch.randn(4, 4, 52)
    windows[..., -1] = 0
    windows[0, :2] = float("nan")
    windows[1, -1] = 0
    windows[1, -1, -1] = 1
    windows[2] = 0
    windows[2, :, -1] = 1
    transformed = torch.cat((relative_window(windows[..., :-1]), windows[..., -1:]), dim=-1)

    assert not relative.supports_frame_cache
    torch.testing.assert_close(relative(windows), absolute(transformed))
    if factorize:
        for head in range(2):
            torch.testing.assert_close(relative.head_score(windows, head), absolute.head_score(transformed, head))


def test_generated_goal_self_loops_are_simulation_disjoint_and_timeouts_are_not_absorbing():
    scenes = AbsorbingRollout(torch.zeros(6, 51), 2, 51, 2)
    for step in range(2):
        done = torch.tensor([True, True, False, False, True, True]) if step == 0 else torch.zeros(6, dtype=torch.bool)
        terminated = torch.tensor([True, True, False, False, False, False]) if step == 0 else done
        scenes.record(step, torch.zeros(6, 51), torch.ones(6, 51), torch.full((6, 51), 2.), done, terminated)
    train, heldout, simulations = split_absorbing_generated(scenes, None, 1, 2)
    total = scenes.dones.numel()
    for indices in (train, heldout):
        physical = set(indices[indices < total].tolist())
        loops = set((indices[indices >= total] - total).tolist())
        assert loops == physical & {0, 1}
    assert not set(((train[train < total] % 6) // 2).tolist()) & set(
        ((heldout[heldout < total] % 6) // 2).tolist()
    )
    assert simulations.sum() == 1
    timeout = scenes.transitions(torch.tensor([4]))
    assert timeout[0, -1, -1] == 0
    assert timeout[0, -1, 0] == 2


@pytest.mark.parametrize("amp", [False, True])
def test_inactive_absorbing_head_is_not_updated_by_adam_momentum(amp):
    args = arguments()
    args.amp = amp
    model = AbsorbingDiscriminator(args, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    windows = torch.randn(4, 4, 52) * .1
    windows[..., -1] = 0
    expert = windows.clone()
    expert[..., :-1] += 0.01
    train_discriminator_batch(model, optimizer, windows, expert, args)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    train_discriminator_batch(model, optimizer, windows, expert, args, active_heads=(0,))
    for name, parameter in model.named_parameters():
        if name.startswith("base.global_discriminator.") or name == "absorbing.1":
            torch.testing.assert_close(before[name], parameter, rtol=0, atol=0)


def test_expert_goal_metadata_adds_terminal_and_self_loop_without_crossing_files(tmp_path):
    for i, goal in enumerate((True, True, False)):
        path = tmp_path / f"{i}-0-00000000-0000-0000-0000-{i:012d}.npy"
        rows = np.zeros((6, team_replay_row_size(1)), np.float32)
        rows[:, 0] = np.arange(6) + 10 * i
        if goal:
            rows[-1, -1] = 1
        np.save(path, rows)
        path.with_suffix(".json").write_text(json.dumps({"goal_terminal": goal}))
    dataset = ExpertDataset(tmp_path, 1, None, "cpu", seed=0, heldout_size=1, window_length=4,
                            terminal_aware=True)
    assert int((dataset.train_kinds == 1).sum() + (dataset.heldout_kinds == 1).sum()) == 2
    assert int((dataset.train_kinds == 2).sum() + (dataset.heldout_kinds == 2).sum()) == 2
    for windows in (dataset.sample(2000), dataset.sample_heldout(2000)):
        absorbing = windows[:, -1, -1] == 1
        terminal = absorbing & (windows[:, -2, -1] == 0)
        loops = absorbing & (windows[:, -2, -1] == 1)
        if terminal.any():
            assert (windows[terminal, -1, :-1] == 0).all()
        if loops.any():
            assert (windows[loops, :, :-1] == 0).all()
            assert (windows[loops, :, -1] == 1).all()


def test_missing_terminal_metadata_fails_closed(tmp_path):
    np.save(tmp_path / "1-0-match.npy", np.zeros((3, team_replay_row_size(1)), np.float32))
    with pytest.raises(ValueError, match="missing goal terminal metadata"):
        ExpertDataset(tmp_path, 1, None, "cpu", terminal_aware=True)


def test_absorbing_value_bellman_learning_and_terminal_gae():
    value = AbsorbingValue()
    optimizer = torch.optim.SGD(value.parameters(), lr=.2)
    update_absorbing_value(value, optimizer, torch.tensor(-.7), gamma=.5, updates=250)
    torch.testing.assert_close(value.value, torch.tensor(-1.4), rtol=1e-5, atol=1e-5)
    rollout = Rollout.allocate(2, 1, 1, 1, "cpu")
    rollout.rewards[:, 0] = torch.tensor([-.1, -.2])
    rollout.values.zero_()
    rollout.timeout_values[:, 0] = torch.tensor([0., value.value.item()])
    rollout.dones[:, 0] = torch.tensor([False, True])
    compute_gae(rollout, torch.zeros(1), gamma=.5, gae_lambda=1.)
    torch.testing.assert_close(rollout.returns[:, 0], torch.tensor([-.55, -.9]), rtol=1e-5, atol=1e-5)
    terminal_return = -.7 + .5 * value.value.item()
    continuing_return = -.7 + .5 * (-.7 + .5 * value.value.item())
    assert terminal_return == pytest.approx(continuing_return, abs=1e-5)


@pytest.mark.parametrize("factorize", [False, True])
def test_absorbing_diagnostics(factorize):
    args = arguments(factorize)
    args.reward_gamma = .5
    model = AbsorbingDiscriminator(args, 2)
    value = AbsorbingValue()
    scenes = SimpleNamespace(dones=torch.zeros(2, 2, dtype=torch.bool))
    experts = SimpleNamespace(train_kinds=torch.tensor([0, 1, 2, 2]),
                              heldout_kinds=torch.tensor([0, 2]))
    metrics = absorbing_diagnostics(model, args, "cpu", 51, torch.tensor(-.7), value,
                                    scenes, torch.tensor([0, 1, 4]), torch.tensor([2, 6]), experts)
    assert metrics["absorbing/value_fixed_point"] == pytest.approx(-1.4)
    assert metrics["absorbing/value_error"] == pytest.approx(1.4)
    assert metrics["absorbing/train_generated_pool_loop_fraction"] == pytest.approx(1 / 3)
    assert metrics["absorbing/train_expert_pool_loop_fraction"] == pytest.approx(.5)
    assert metrics["absorbing/heldout_generated_pool_loop_fraction"] == pytest.approx(.5)
    assert metrics["absorbing/heldout_expert_pool_loop_fraction"] == pytest.approx(.5)
    assert "absorbing/global_logit" in metrics
    assert ("absorbing/ego_logit" in metrics) == factorize


@pytest.mark.parametrize("gru", [False, True])
def test_ego_only_absorbing_discriminator_and_metrics(gru):
    args = arguments(False, gru)
    args.ego_only = True
    args.reward_gamma = .5
    model = AbsorbingDiscriminator(args, 2)
    assert not model.factorized and len(model.absorbing) == 1
    windows = torch.zeros(6, 4, 52)
    windows[[1, 4], -1, -1] = 1
    windows[[2, 5], :, -1] = 1
    logits = model(windows)
    assert logits.shape == (6,)
    torch.testing.assert_close(logits, model.score_encoded(model.encode_frames(windows),
                                                         torch.isnan(windows[..., 0])))
    metrics = evaluate_discriminator(model, None, None, None, args, windows)
    assert "discriminator/ego_heldout_accuracy" in metrics
    assert "discriminator/ego_absorbing_expert_accuracy" in metrics
    assert "discriminator/global_absorbing_expert_accuracy" not in metrics
    value = AbsorbingValue()
    scenes = SimpleNamespace(dones=torch.zeros(2, 2, dtype=torch.bool))
    experts = SimpleNamespace(train_kinds=torch.tensor([0, 2]), heldout_kinds=torch.tensor([0, 2]))
    metrics = absorbing_diagnostics(model, args, "cpu", 51, torch.tensor(-.7), value,
                                    scenes, torch.tensor([0, 4]), torch.tensor([1, 5]), experts)
    assert "absorbing/ego_logit" in metrics and "absorbing/global_logit" not in metrics
    optimizer = torch.optim.Adam(model.parameters())
    assert torch.isfinite(train_discriminator_batch(model, optimizer, windows[:2], windows[2:4], args))


def test_goal_reward_is_preserved_and_dac_scores_terminal_transition():
    class Score(torch.nn.Module):
        def forward(self, windows):
            return windows[:, -1, -1] * 2

    args = SimpleNamespace(
        reward_plus=False, goal_reward=True, reward_plus_weight=1.,
        discriminator_inference_microbatch_size=8, bf16=False,
        exp_log_odds_reward=False, multiplicative_reward=False, discriminator_log_reward=True
    )
    scenes = AbsorbingRollout(torch.zeros(2, 51), 1, 51, 2)
    goal = torch.tensor([True, True])
    scenes.record(0, torch.zeros(2, 51), torch.ones(2, 51), torch.ones(2, 51), goal, goal)
    rollout = SimpleNamespace(rewards=torch.tensor([[50., -50.]]))
    imitation_rewards(Score(), scenes, rollout, args, include_terminal=True)
    expected = torch.tensor([[50., -50.]]) + torch.nn.functional.logsigmoid(torch.tensor(-2.))
    torch.testing.assert_close(rollout.rewards, expected)


def test_validation_reports_terminal_and_absorbing_classes():
    model = AbsorbingDiscriminator(arguments(), 2)
    windows = torch.zeros(6, 4, 52)
    windows[[1, 4], -1, -1] = 1
    windows[[2, 5], :, -1] = 1
    metrics = evaluate_discriminator(model, None, None, None, arguments(), windows)
    for category in ("live", "terminal", "absorbing"):
        for source in ("generated", "expert"):
            assert metrics[f"discriminator/{category}_{source}_count"] == 1
            assert f"discriminator/ego_{category}_{source}_accuracy" in metrics
