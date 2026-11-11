import argparse
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from imitation.dataset import ExpertDataset
from imitation.models import WindowHead, build_discriminator, relative_window
from imitation.rewards import discriminator_rewards, imitation_rewards
from imitation.rollout import TransitionRollout, split_generated
from imitation.update import discriminator_loss, train_discriminator_batch
from replays.layout import team_replay_row_size, team_scene_size
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint
from training.models import Agent
from training.rollout import Rollout


def arguments(*flags):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=True)
    add_gaifo_arguments(parser)
    args = parser.parse_args(["--frame-embedding", "8", "--discriminator-hidden", "8", *flags])
    resolve_gaifo_defaults(args)
    return args


class ScalarScore(nn.Module):

    factorized = False

    def forward(self, transitions):
        return transitions[:, 1, 0]


@pytest.mark.parametrize("factorize", [False, True])
@pytest.mark.parametrize("team_size", [1, 2, 3])
def test_feed_forward_transition_models(factorize, team_size):
    args = arguments(*(["--factorize"] if factorize else []))
    model = build_discriminator(args, 2 * team_size)
    transitions = torch.randn(4, 2, team_scene_size(team_size))
    assert model(transitions).shape == ((4, 2) if factorize else (4,))
    assert not any(isinstance(module, nn.RNNBase) for module in model.modules())


@pytest.mark.parametrize("team_size", [1, 2, 3])
def test_ego_head_receives_only_ego_and_relative_ball_and_opponent_positions(team_size):
    model = build_discriminator(arguments("--factorize"), 2 * team_size)
    specialist = model.ego_discriminator
    specialist.encoder = nn.Identity()
    transitions = torch.randn(4, 2, team_scene_size(team_size))
    cars = transitions[..., 9:].unflatten(-1, (2 * team_size, 21))
    expected = torch.cat((cars[..., 0, :], transitions[..., :3] - cars[..., 0, :3],
                          (cars[..., team_size:, :3] - cars[..., :1, :3]).flatten(-2)), dim=-1)
    torch.testing.assert_close(specialist._encode(transitions), expected)
    changed = transitions.clone()
    changed[..., 3:9] += 100
    changed_cars = changed[..., 9:].unflatten(-1, (2 * team_size, 21))
    changed_cars[..., 1:team_size, :] += 100
    changed_cars[..., team_size:, 3:] += 100
    torch.testing.assert_close(specialist._encode(changed), expected)
    changed_cars[..., team_size:, :3] += 1
    assert not torch.allclose(specialist._encode(changed), expected)


@pytest.mark.parametrize("architecture", ["transformer", "gru", "mlp"])
@pytest.mark.parametrize("factorize", [False, True])
def test_ego_only_uses_ego_head_and_cached_scores(architecture, factorize):
    architecture_flags = {"transformer": [], "gru": ["--gru"], "mlp": ["--mlp-discriminator"]}[architecture]
    flags = ["--ego-only"] + (["--factorize"] if factorize else []) + architecture_flags
    args = arguments(*flags)
    model = build_discriminator(args, 2)
    assert not model.factorized
    assert not hasattr(model, "global_discriminator")
    factorized = build_discriminator(arguments("--factorize", *architecture_flags), 2)
    model.load_state_dict(factorized.ego_discriminator.state_dict())
    windows = torch.randn(4, 2, 51)
    windows[0, 0] = float("nan")
    logits = model(windows)
    torch.testing.assert_close(logits, factorized(windows)[:, 0])
    torch.testing.assert_close(logits, model.score_encoded(model.encode_frames(windows),
                                                         torch.isnan(windows[..., 0])))
    loss = discriminator_loss(model, windows, torch.tensor([0., 0., 1., 1.]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_ego_only_rejects_multiplicative_reward_and_old_objectives(tmp_path):
    with pytest.raises(ValueError, match="ego-only"):
        arguments("--ego-only", "--multiplicative-reward")
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(arguments())}, path)
    with pytest.raises(ValueError, match="objective differs"):
        load_checkpoint(path, "gaifo", arguments("--ego-only"), None, None)


def test_paper_classification_and_reward_signs():
    model = ScalarScore()
    transitions = torch.zeros(2, 2, 1)
    transitions[:, 1, 0] = torch.tensor([3., -3.])
    labels = torch.tensor([1., 0.])
    loss = discriminator_loss(model, transitions, labels)
    torch.testing.assert_close(loss, 2 * F.softplus(torch.tensor(-3.)))
    rewards = discriminator_rewards(torch.tensor([-3., 3.]))
    torch.testing.assert_close(rewards, F.softplus(torch.tensor([3., -3.])))
    assert rewards[0] > rewards[1]


def test_factorized_reward_averages_or_multiplies_active_heads():
    logits = torch.tensor([[1., 0.], [1., 2.]])
    expected = F.softplus(-torch.tensor([1., 1.])) * F.softplus(-torch.tensor([0., 2.]))
    torch.testing.assert_close(discriminator_rewards(logits, multiplicative_reward=True), expected)
    expected = (F.softplus(-torch.tensor([1., 1.])) + F.softplus(-torch.tensor([0., 2.]))) / 2
    torch.testing.assert_close(discriminator_rewards(logits), expected)


def test_factorized_reward_has_no_proximity_switch():
    torch.manual_seed(9)
    model = build_discriminator(arguments("--factorize"), 2)
    windows = torch.zeros(2, 2, 51)
    windows[0, :, 0] = 999.99 / 4108
    windows[1, :, 0] = 1000.01 / 4108
    rewards = discriminator_rewards(model(windows))
    torch.testing.assert_close(rewards[0], rewards[1], rtol=1e-4, atol=1e-5)


def test_exponentiated_expert_odds_are_capped_before_factorization():
    logits = torch.tensor([[-100., -100.], [-2., 1.]])
    expected = torch.tensor([100., torch.exp(torch.tensor(1.)).item()])
    torch.testing.assert_close(discriminator_rewards(logits, True, True), expected)
    single = discriminator_rewards(torch.tensor([-100., 0., 2.]), True)
    torch.testing.assert_close(single, torch.tensor([10., 1., torch.exp(torch.tensor(-2.)).item()]))


def test_expert_odds_flag_is_opt_in():
    assert not arguments().exp_log_odds_reward
    assert arguments("--exp-log-odds-reward").exp_log_odds_reward


def test_log_expert_reward_is_stable_negative_and_increases_with_expert_probability():
    logits = torch.tensor([-1000., -3., 0., 3., 1000.])
    rewards = discriminator_rewards(logits, discriminator_log_reward=True)
    torch.testing.assert_close(rewards, F.logsigmoid(-logits))
    assert torch.isfinite(rewards).all()
    assert (rewards <= 0).all()
    assert (rewards[1:] < rewards[:-1]).all()
    factorized = torch.stack((logits, logits + 1), dim=-1)
    torch.testing.assert_close(discriminator_rewards(factorized, discriminator_log_reward=True),
                               F.logsigmoid(-factorized).mean(-1))


def test_log_reward_rejects_incompatible_reward_modes():
    assert not arguments().discriminator_log_reward
    assert arguments("--discriminator-log-reward").discriminator_log_reward
    with pytest.raises(ValueError, match="mutually exclusive"):
        arguments("--discriminator-log-reward", "--exp-log-odds-reward")
    with pytest.raises(ValueError, match="negative head rewards"):
        arguments("--factorize", "--discriminator-log-reward", "--multiplicative-reward")


@pytest.mark.parametrize("original_log", [False, True])
def test_log_reward_checkpoint_mode_must_match(tmp_path, original_log):
    path = tmp_path / "checkpoint.pt"
    original = arguments(*(["--discriminator-log-reward"] if original_log else []))
    requested = arguments(*([] if original_log else ["--discriminator-log-reward"]))
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(original)}, path)
    with pytest.raises(ValueError, match="reward mode differs"):
        load_checkpoint(path, "gaifo", requested, None, None)


def test_log_reward_goal_transition_still_gets_zero_imitation_reward():
    args = arguments("--discriminator-log-reward")
    scenes = TransitionRollout(torch.zeros(1, 51), 1, 51)
    scenes.record(0, torch.zeros(1, 51), torch.ones(1, 51), torch.ones(1, 51),
                  torch.tensor([True]), torch.tensor([True]))
    rollout = SimpleNamespace(rewards=torch.zeros(1, 1))
    imitation_rewards(ScalarScore(), scenes, rollout, args)
    torch.testing.assert_close(rollout.rewards, torch.zeros(1, 1))


@pytest.mark.parametrize("enabled", [False, True])
def test_reward_plus_is_added_only_when_enabled(enabled):
    args = arguments(*(["--reward-plus"] if enabled else []), "--reward-plus-weight", "1")
    scenes = TransitionRollout(torch.zeros(1, 51), 1, 51)
    scenes.record(0, torch.zeros(1, 51), torch.ones(1, 51), torch.ones(1, 51), torch.tensor([False]))
    rollout = SimpleNamespace(rewards=torch.full((1, 1), 3.0))
    metrics = imitation_rewards(ScalarScore(), scenes, rollout, args)
    imitation = F.softplus(torch.tensor(-1.)).item()
    torch.testing.assert_close(rollout.rewards, torch.tensor([[imitation + (3.0 if enabled else 0.0)]]))
    assert metrics["imitation_reward"] == pytest.approx(imitation)


def test_reward_plus_weight_scales_only_gameplay_reward():
    args = arguments("--reward-plus", "--reward-plus-weight", "0.25")
    scenes = TransitionRollout(torch.zeros(1, 51), 1, 51)
    scenes.record(0, torch.zeros(1, 51), torch.ones(1, 51), torch.ones(1, 51), torch.tensor([False]))
    rollout = SimpleNamespace(rewards=torch.full((1, 1), -4.0))
    metrics = imitation_rewards(ScalarScore(), scenes, rollout, args)
    imitation = F.softplus(torch.tensor(-1.)).item()
    torch.testing.assert_close(rollout.rewards, torch.tensor([[imitation - 1.0]]))
    assert metrics["reward_plus/total_rms"] == pytest.approx(4.0)
    assert metrics["reward_plus/weighted_rms"] == pytest.approx(1.0)
    assert metrics["rms/imitation"] == pytest.approx(imitation)


def test_goal_reward_adds_scaled_goal_to_imitation():
    args = arguments("--goal-reward", "--reward-plus-weight", "0.25")
    scenes = TransitionRollout(torch.zeros(1, 51), 1, 51)
    scenes.record(0, torch.zeros(1, 51), torch.ones(1, 51), torch.ones(1, 51), torch.tensor([False]))
    rollout = SimpleNamespace(rewards=torch.tensor([[200.0]]))
    imitation_rewards(ScalarScore(), scenes, rollout, args)
    torch.testing.assert_close(rollout.rewards, torch.tensor([[50.0 + F.softplus(torch.tensor(-1.)).item()]]))


def test_goal_reward_options_and_checkpoint(tmp_path):
    assert not arguments().goal_reward
    with pytest.raises(ValueError, match="mutually exclusive"):
        arguments("--goal-reward", "--reward-plus")
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(arguments())}, path)
    with pytest.raises(ValueError, match="goal reward mode differs"):
        load_checkpoint(path, "gaifo", arguments("--goal-reward"), None, None)


def test_reward_plus_checkpoint_mode_must_match(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(arguments())}, path)
    with pytest.raises(ValueError, match="reward-plus mode differs"):
        load_checkpoint(path, "gaifo", arguments("--reward-plus"), None, None)


def test_reward_plus_default_weight_depends_on_imitation_reward_mode():
    assert arguments().reward_plus_weight == .25
    assert arguments("--reward-plus").reward_plus_weight == 1.0
    assert arguments("--reward-plus", "--exp-log-odds-reward").reward_plus_weight == 1.0
    assert arguments("--goal-reward").reward_plus_weight == .25
    assert arguments("--exp-log-odds-reward").reward_plus_weight == 2.4
    assert arguments("--reward-plus-weight", "0.7").reward_plus_weight == .7


def test_reward_plus_checkpoint_weight_must_match(tmp_path):
    path = tmp_path / "checkpoint.pt"
    original = arguments("--reward-plus", "--reward-plus-weight", "0.5")
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(original)}, path)
    with pytest.raises(ValueError, match="reward-plus weight differs"):
        load_checkpoint(path, "gaifo", arguments("--reward-plus"), None, None)


def test_discriminator_default_batch_sizes():
    args = arguments()
    assert args.discriminator_batch_size == 65_536
    assert args.discriminator_update_microbatch_size == 16_384
    assert args.discriminator_inference_microbatch_size == 65_536
    assert args.discriminator_min_heldout_replays == 20
    assert arguments("--discriminator-heldout-replays", "7").discriminator_min_heldout_replays == 7
    assert arguments("--team-size", "2").discriminator_batch_size == 16_384
    assert arguments("--team-size", "3").discriminator_batch_size == 8_192
    assert arguments("--discriminator-batch-size", "123").discriminator_batch_size == 123
    assert args.discriminator_history_interval == 4
    assert arguments("--discriminator-history-interval", "1").discriminator_history_interval == 1
    with pytest.raises(ValueError, match="history interval"):
        arguments("--discriminator-history-interval", "0")
    assert args.discriminator_accuracy_target == .85
    assert args.discriminator_accuracy_target_end == 1.
    assert arguments("--discriminator-accuracy-target-end", ".9").discriminator_accuracy_target_end == .9
    with pytest.raises(ValueError, match="final accuracy target"):
        arguments("--discriminator-accuracy-target-end", "1.1")


def test_terminal_transition_uses_final_scene_and_reward_is_discriminator_only():
    args = arguments()
    initial = torch.zeros(1, 51)
    scenes = TransitionRollout(initial, 2, 51)
    reset, final = initial + 9, initial + 1
    scenes.record(0, initial, reset, final, torch.tensor([True]))
    scenes.record(1, reset, final, final, torch.tensor([False]))
    torch.testing.assert_close(scenes.transitions(torch.tensor([0, 1]))[:, :, 0],
                               torch.tensor([[0., 1.], [9., 1.]]))
    rollout = SimpleNamespace(rewards=torch.full((2, 1), 100.))
    imitation_rewards(ScalarScore(), scenes, rollout, args)
    torch.testing.assert_close(rollout.rewards, torch.full((2, 1), F.softplus(torch.tensor(-1.)).item()))


def test_goal_transitions_are_excluded_from_discriminator_and_imitation_reward():
    args = arguments("--reward-plus", "--reward-plus-weight", "1")
    scenes = TransitionRollout(torch.zeros(1, 51), 2, 51)
    for step in range(2):
        terminated = torch.tensor([step == 0])
        scenes.record(step, torch.zeros(1, 51), torch.ones(1, 51), torch.ones(1, 51), terminated, terminated)
    train, heldout, _ = split_generated(torch.zeros(2, 1), None, 0, 1, scenes.terminated)
    assert train.tolist() == [1]
    assert not len(heldout)
    rollout = SimpleNamespace(rewards=torch.tensor([[200.], [0.]]))
    imitation_rewards(ScalarScore(), scenes, rollout, args)
    torch.testing.assert_close(rollout.rewards[:, 0], torch.tensor([200., F.softplus(torch.tensor(-1.)).item()]))


def test_window_masks_resets_and_carries_between_rollouts():
    initial = torch.zeros(1, 51)
    scenes = TransitionRollout(initial, 3, 51, 4)
    for step in range(3):
        current = initial + step
        scenes.record(step, current, current + 1, current + 1, torch.tensor([step == 1]))
    windows = scenes.transitions(torch.arange(3))[:, :, 0]
    assert torch.isnan(windows[0, :2]).all()
    torch.testing.assert_close(windows[1, 1:], torch.tensor([0., 1., 2.]))
    assert torch.isnan(windows[2, :2]).all()
    scenes.record(0, initial + 3, initial + 4, initial + 4, torch.tensor([False]))
    window = scenes.transitions(torch.tensor([0]))[0, :, 0]
    assert torch.isnan(window[0])
    torch.testing.assert_close(window[1:], torch.tensor([2., 3., 4.]))


def test_potential_windows_carry_history_and_mask_episode_resets():
    scenes = TransitionRollout(torch.zeros(1, 1), 3, 1, 4, potential_shaping=True)
    for step in range(3):
        current = torch.tensor([[float(step)]])
        scenes.record(step, current, current + 1, current + 1, torch.tensor([step == 1]))

    before = scenes.previous_windows(torch.arange(3))[:, :, 0]
    assert torch.isnan(before[0, :3]).all() and before[0, -1] == 0
    torch.testing.assert_close(before[1, -2:], torch.tensor([0., 1.]))
    assert torch.isnan(before[2, :3]).all() and before[2, -1] == 2

    scenes.record(0, torch.tensor([[3.]]), torch.tensor([[4.]]), torch.tensor([[4.]]), torch.tensor([False]))
    before = scenes.previous_windows(torch.tensor([0]))[0, :, 0]
    assert torch.isnan(before[:2]).all()
    torch.testing.assert_close(before[-2:], torch.tensor([2., 3.]))


@pytest.mark.parametrize("include_terminal", [False, True])
def test_discriminator_potential_shaping_respects_terminal_and_timeout(include_terminal):
    args = arguments("--discriminator-log-reward", "--discriminator-potential-shaping",
                     "--discriminator-potential-weight", "2")
    args.reward_gamma = 0.9
    scenes = TransitionRollout(torch.zeros(2, 1), 1, 1, 2, potential_shaping=True)
    current = torch.tensor([[1.], [2.]])
    final = torch.tensor([[3.], [4.]])
    scenes.record(0, current, final, final, torch.tensor([True, True]), torch.tensor([True, False]))
    rollout = SimpleNamespace(rewards=torch.zeros(1, 2))
    metrics = imitation_rewards(ScalarScore(), scenes, rollout, args, include_terminal=include_terminal)
    before = F.logsigmoid(-current[:, 0])
    after = F.logsigmoid(-final[:, 0])
    base = after.clone()
    if not include_terminal:
        base[0] = 0
    potential_after = after.clone()
    potential_after[0] = 0
    shaping = 2 * (0.9 * potential_after - before)

    torch.testing.assert_close(rollout.rewards[0], base + shaping)
    assert metrics["discriminator_potential/mean"] == pytest.approx(shaping.mean().item())
    assert metrics["discriminator_potential/rms"] == pytest.approx(shaping.square().mean().sqrt().item())


def test_discriminator_potential_config_is_validated_for_resume(tmp_path):
    path = tmp_path / "checkpoint.pt"
    original = arguments("--discriminator-potential-shaping", "--discriminator-potential-weight", "0.3")
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(original)}, path)
    with pytest.raises(ValueError, match="potential shaping differs"):
        load_checkpoint(path, "gaifo", arguments(), None, None)
    with pytest.raises(ValueError, match="potential weight differs"):
        load_checkpoint(path, "gaifo", arguments("--discriminator-potential-shaping"), None, None)
    with pytest.raises(ValueError, match="potential weight must be finite"):
        arguments("--discriminator-potential-weight", "nan")


def test_discriminator_potential_telescopes_over_goal_episode():
    args = arguments("--discriminator-log-reward", "--discriminator-potential-shaping")
    args.reward_gamma = 0.9
    scenes = TransitionRollout(torch.zeros(1, 1), 3, 1, 2, potential_shaping=True)
    for step in range(3):
        current = torch.tensor([[float(step + 1)]])
        final = current + 1
        terminal = torch.tensor([step == 2])
        scenes.record(step, current, final, final, terminal, terminal)

    baseline_args = arguments("--discriminator-log-reward")
    baseline = SimpleNamespace(rewards=torch.zeros(3, 1))
    shaped = SimpleNamespace(rewards=torch.zeros(3, 1))
    imitation_rewards(ScalarScore(), scenes, baseline, baseline_args)
    imitation_rewards(ScalarScore(), scenes, shaped, args)
    differences = (shaped.rewards - baseline.rewards)[:, 0]
    discounts = torch.tensor([1., 0.9, 0.9 ** 2])
    torch.testing.assert_close((differences * discounts).sum(), -F.logsigmoid(torch.tensor(-1.)))


@pytest.mark.parametrize("architecture", ["transformer", "gru", "mlp"])
@pytest.mark.parametrize("amp", [False, True])
def test_cached_potential_rewards_match_direct(architecture, amp):
    flags = {"transformer": [], "gru": ["--gru"], "mlp": ["--mlp-discriminator"]}[architecture]
    args = arguments(*flags, *(["--amp"] if amp else []), "--discriminator-window-length", "4", "--discriminator-potential-shaping",
                     "--discriminator-inference-microbatch-size", "3")
    args.reward_gamma = 0.99
    model = build_discriminator(args, 2)
    scenes = TransitionRollout(torch.zeros(2, 51), 4, 51, 4, potential_shaping=True)
    for step in range(4):
        current = torch.randn(2, 51)
        scenes.record(step, current, current + 1, current + 2, torch.tensor([step == 1, False]))

    class DirectScore(nn.Module):

        def forward(self, windows):
            return model(windows)

    cached = SimpleNamespace(rewards=torch.zeros(4, 2))
    direct = SimpleNamespace(rewards=torch.zeros(4, 2))
    imitation_rewards(model, scenes, cached, args)
    imitation_rewards(DirectScore(), scenes, direct, args)
    torch.testing.assert_close(cached.rewards, direct.rewards)


@pytest.mark.parametrize("architecture", ["transformer", "gru", "mlp"])
@pytest.mark.parametrize("factorize", [False, True])
def test_cached_window_scores_match_direct(factorize, architecture):
    flags = {"transformer": [], "gru": ["--gru"], "mlp": ["--mlp-discriminator"]}[architecture]
    args = arguments(*(["--factorize"] if factorize else []), *flags, "--discriminator-window-length", "4")
    model = build_discriminator(args, 2)
    scenes = TransitionRollout(torch.zeros(2, 51), 4, 51, 4)
    for step in range(4):
        current = torch.randn(2, 51)
        scenes.record(step, current, current + 1, current + 2, torch.tensor([step == 1, False]))
    indices = torch.arange(8)
    windows = scenes.transitions(indices)
    window_indices, padding = scenes.window_indices(indices)
    padding |= torch.isnan(windows[..., 0])
    encoded = model.encode_frames(scenes.all_frames()[:, None])
    cached = model.score_encoded([part[:, 0][window_indices] for part in encoded], padding)
    torch.testing.assert_close(cached, model(windows))


@pytest.mark.parametrize("architecture", ["transformer", "gru", "mlp"])
@pytest.mark.parametrize("factorize", [False, True])
@pytest.mark.parametrize("reward_flags", [[], ["--discriminator-log-reward"], ["--amp"]])
def test_cached_reward_matches_direct_reward(factorize, architecture, reward_flags):
    flags = {"transformer": [], "gru": ["--gru"], "mlp": ["--mlp-discriminator"]}[architecture]
    args = arguments(*(["--factorize"] if factorize else []), *flags, "--discriminator-window-length", "4",
                     *reward_flags,
                     "--discriminator-inference-microbatch-size", "3")
    model = build_discriminator(args, 2)
    scenes = TransitionRollout(torch.zeros(2, 51), 4, 51, 4)
    for step in range(4):
        current = torch.randn(2, 51)
        scenes.record(step, current, current + 1, current + 2, torch.tensor([step == 1, False]))

    class DirectScore(nn.Module):

        def forward(self, windows):
            return model(windows)

    cached = SimpleNamespace(rewards=torch.zeros(4, 2))
    direct = SimpleNamespace(rewards=torch.zeros(4, 2))
    imitation_rewards(model, scenes, cached, args)
    imitation_rewards(DirectScore(), scenes, direct, args)
    torch.testing.assert_close(cached.rewards, direct.rewards)


def test_critic_windows_mask_resets_and_keep_prefix():
    rollout = Rollout.allocate(3, 1, 1, 1, "cpu")
    rollout.critic_prefix = torch.tensor([[[float("nan")]], [[-1.]]])
    rollout.critic_prefix_dones = torch.zeros(2, 1, dtype=torch.bool)
    rollout.observations[:, 0, 0] = torch.tensor([0., 1., 2.])
    rollout.dones[:, 0] = torch.tensor([False, True, False])
    rollout.prepare_critic_windows()
    windows = rollout.critic_windows(torch.arange(3))[:, :, 0]
    assert torch.isnan(windows[0, 0])
    torch.testing.assert_close(windows[0, 1:], torch.tensor([-1., 0.]))
    torch.testing.assert_close(windows[1], torch.tensor([-1., 0., 1.]))
    assert torch.isnan(windows[2, :2]).all()
    assert windows[2, 2] == 2


def test_critic_cached_projection_matches_direct_scoring():
    agent = Agent(6, [2], None, hidden_size=8, layers=1, critic_window_length=4, gru_critic=True)
    windows = torch.randn(3, 4, 6)
    windows[0, :2] = float("nan")
    padding = torch.isnan(windows[..., 0])
    encoded = agent.critic.projection(torch.nan_to_num(windows))
    torch.testing.assert_close(agent.get_value_encoded(encoded, padding), agent.get_value(windows))


def test_default_mlp_critic_uses_current_observation():
    agent = Agent(6, [2], None, hidden_size=8, layers=1, critic_window_length=4)
    windows = torch.randn(3, 4, 6)
    windows[:, :2] = float("nan")
    assert isinstance(agent.critic, nn.Sequential)
    torch.testing.assert_close(agent.get_value(windows), agent.get_value(windows[:, -1]))


def test_gru_padding_matches_individually_scored_windows():
    torch.manual_seed(4)
    head = WindowHead(5, 8, 8, 5, gru=True)
    frames = torch.randn(4, 5, 5)
    lengths = [1, 2, 4, 5]
    padding = torch.arange(5)[None] < (5 - torch.tensor(lengths))[:, None]
    actual = head(frames, padding)
    expected = torch.cat([head(frames[i:i + 1, -length:]) for i, length in enumerate(lengths)])
    torch.testing.assert_close(actual, expected)


def test_mlp_window_padding_is_masked_and_short_windows_are_right_aligned():
    torch.manual_seed(4)
    head = WindowHead(5, 8, 8, 5, mlp=True)
    frames = torch.randn(4, 5, 5)
    lengths = [1, 2, 4, 5]
    padding = torch.arange(5)[None] < (5 - torch.tensor(lengths))[:, None]
    actual = head(frames, padding)
    altered = torch.where(padding[..., None], torch.randn_like(frames) * 1_000, frames)
    torch.testing.assert_close(actual, head(altered, padding))
    expected = torch.cat([head(frames[i:i + 1, -length:]) for i, length in enumerate(lengths)])
    torch.testing.assert_close(actual, expected)


def test_mlp_discriminator_cli_and_checkpoint_architecture(tmp_path):
    args = arguments("--mlp-discriminator")
    assert args.discriminator_architecture == "mlp-window-v1"
    with pytest.raises(SystemExit):
        arguments("--gru", "--mlp-discriminator")

    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "dac", "config": vars(arguments())}, path)
    with pytest.raises(ValueError, match="discriminator architecture differs"):
        load_checkpoint(path, "dac", args, None, None)


def test_relative_window_uses_first_valid_frame_and_preserves_padding():
    frames = torch.tensor([
        [[float("nan"), float("nan")], [2., 10.], [3., 7.], [5., 11.]],
        [[1., 4.], [2., 6.], [4., 7.], [5., 8.]],
    ])
    actual = relative_window(frames)
    expected = torch.tensor([
        [[float("nan"), float("nan")], [2., 10.], [1., -3.], [3., 1.]],
        [[1., 4.], [1., 2.], [3., 3.], [4., 4.]],
    ])
    torch.testing.assert_close(actual, expected, equal_nan=True)


@pytest.mark.parametrize("architecture", ["transformer", "gru", "mlp"])
@pytest.mark.parametrize("head_mode", ["global", "ego", "factorized"])
def test_relative_discriminator_matches_manually_transformed_windows(architecture, head_mode):
    arch = {"transformer": [], "gru": ["--gru"], "mlp": ["--mlp-discriminator"]}[architecture]
    head = {"global": [], "ego": ["--ego-only"], "factorized": ["--factorize"]}[head_mode]
    absolute = build_discriminator(arguments(*arch, *head, "--discriminator-window-length", "4"), 2)
    relative = build_discriminator(arguments(*arch, *head, "--discriminator-window-length", "4",
                                             "--discriminator-relative"), 2)
    relative.load_state_dict(absolute.state_dict())
    windows = torch.randn(4, 4, 51)
    windows[0, :2] = float("nan")
    windows[1, :1] = float("nan")

    assert not relative.supports_frame_cache
    torch.testing.assert_close(relative(windows), absolute(relative_window(windows)))

    if head_mode == "factorized":
        for index in range(2):
            torch.testing.assert_close(relative.head_score(windows, index),
                                       absolute.head_score(relative_window(windows), index))


def test_relative_discriminator_checkpoint_input_mode_is_validated(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(arguments())}, path)
    with pytest.raises(ValueError, match="discriminator input mode differs"):
        load_checkpoint(path, "gaifo", arguments("--discriminator-relative"), None, None)


def test_relative_discriminator_reward_and_potential_use_full_windows():
    args = arguments("--mlp-discriminator", "--discriminator-relative", "--discriminator-potential-shaping",
                     "--discriminator-window-length", "4", "--discriminator-inference-microbatch-size", "3")
    args.reward_gamma = 0.99
    model = build_discriminator(args, 2)
    scenes = TransitionRollout(torch.zeros(2, 51), 4, 51, 4, potential_shaping=True)
    for step in range(4):
        current = torch.randn(2, 51)
        scenes.record(step, current, current + 1, current + 2, torch.tensor([step == 1, False]))

    windows = scenes.transitions(torch.arange(8))
    optimizer = torch.optim.Adam(model.parameters())
    assert torch.isfinite(train_discriminator_batch(model, optimizer, windows, windows, args))
    rollout = SimpleNamespace(rewards=torch.zeros(4, 2))
    metrics = imitation_rewards(model, scenes, rollout, args)
    assert torch.isfinite(rollout.rewards).all()
    assert np.isfinite(metrics["discriminator_potential/rms"])


@pytest.mark.parametrize("discriminator_gru", [False, True])
@pytest.mark.parametrize("critic_gru", [False, True])
def test_gru_discriminator_and_critic_are_independent(discriminator_gru, critic_gru):
    args = arguments("--factorize", *(["--gru"] if discriminator_gru else []),
                     *(["--gru-critic"] if critic_gru else []))
    model = build_discriminator(args, 2)
    agent = Agent(6, [2], None, hidden_size=8, layers=1, critic_window_length=4, gru_critic=args.gru_critic)
    assert all((head.gru is not None) == discriminator_gru for head in (
        model.ego_discriminator.head, model.global_discriminator.head
    ))
    assert isinstance(agent.critic, WindowHead) == critic_gru
    assert args.discriminator_architecture == ("gru-window-v1" if discriminator_gru else "transformer-window-v1")


def test_gru_policy_uses_history_and_masks_padding():
    class Codec:

        def mask(self, observations):
            return torch.ones(len(observations), 5, dtype=torch.bool)

    torch.manual_seed(8)
    agent = Agent(6, [2, 3], Codec(), hidden_size=8, layers=1, critic_window_length=4, gru_policy=True)
    windows = torch.randn(2, 4, 6)
    windows[:, :2] = float("nan")
    actions = torch.tensor([[0, 1], [1, 2]])
    _, log_prob, _ = agent.get_action_and_log_prob(windows, actions)
    changed = windows.clone()
    changed[:, -2] += 3
    _, changed_log_prob, _ = agent.get_action_and_log_prob(changed, actions)
    assert not torch.allclose(log_prob, changed_log_prob)
    one = agent.get_action_and_log_prob(windows[:1], actions[:1])[1]
    torch.testing.assert_close(log_prob[:1], one)


def test_factorized_reward_checkpoint_mode_must_match(tmp_path):
    path = tmp_path / "checkpoint.pt"
    original = arguments("--factorize", "--multiplicative-reward")
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(original)}, path)
    with pytest.raises(ValueError, match="factorized reward mode differs"):
        load_checkpoint(path, "gaifo", arguments("--factorize"), None, None)


def test_old_factorized_discriminator_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "checkpoint.pt"
    args = arguments("--factorize")
    config = vars(args).copy()
    config["discriminator_objective"] = "gaifo-bce-no-goal-v2"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": config}, path)
    with pytest.raises(ValueError, match="discriminator objective differs"):
        load_checkpoint(path, "gaifo", args, None, None)


def test_gru_policy_checkpoint_mode_must_match(tmp_path):
    path = tmp_path / "checkpoint.pt"
    original = arguments("--gru-policy")
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(original)}, path)
    with pytest.raises(ValueError, match="actor architecture differs"):
        load_checkpoint(path, "gaifo", arguments(), None, None)


def test_gru_critic_checkpoint_mode_must_match(tmp_path):
    path = tmp_path / "checkpoint.pt"
    original = arguments("--gru-critic")
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(original)}, path)
    with pytest.raises(ValueError, match="critic architecture differs"):
        load_checkpoint(path, "gaifo", arguments(), None, None)


def test_legacy_gru_checkpoint_requires_gru_critic(tmp_path):
    path = tmp_path / "checkpoint.pt"
    original = arguments("--gru")
    config = vars(original).copy()
    config.pop("gru_critic")
    config.pop("critic_architecture")
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": config}, path)
    with pytest.raises(ValueError, match="critic architecture differs"):
        load_checkpoint(path, "gaifo", original, None, None)


def test_legacy_transformer_critic_is_not_silently_loaded_as_mlp(tmp_path):
    path = tmp_path / "checkpoint.pt"
    args = arguments()
    config = vars(args).copy()
    config.pop("critic_architecture")
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": config}, path)
    with pytest.raises(ValueError, match="critic architecture differs"):
        load_checkpoint(path, "gaifo", args, None, None)


def test_expert_windows_allow_short_prefixes(tmp_path):
    rows = np.zeros((6, team_replay_row_size(1)), np.float32)
    rows[:, 0] = np.arange(6)
    np.save(tmp_path / "match.npy", rows)
    dataset = ExpertDataset(tmp_path, 1, None, "cpu", window_length=4)
    assert dataset.train_lengths.tolist() == [2, 3, 4, 4, 4]
    windows = dataset.sample(100)
    assert torch.isnan(windows[:, :-2, 0]).any()
    torch.testing.assert_close(windows[:, -1, 0] - windows[:, -2, 0], torch.ones(100))


@pytest.mark.parametrize("factorize", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("gru", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_optimizer_updates_with_finite_loss(factorize, bf16, gru, amp):
    args = arguments(*(["--factorize"] if factorize else []), *(["--bf16"] if bf16 else []),
                     *(["--amp"] if amp else []),
                     *(["--gru"] if gru else []),
                     "--discriminator-update-microbatch-size", "2")
    model = build_discriminator(args, 2)
    optimizer = torch.optim.Adam(model.parameters())
    generated = torch.randn(4, 2, 51) * .1
    expert = torch.randn(4, 2, 51) * .1
    loss = train_discriminator_batch(model, optimizer, generated, expert, args)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("factorize", [False, True])
@pytest.mark.parametrize("noise_scale", [0.0, 0.001])
@pytest.mark.parametrize("amp", [False, True])
def test_microbatching_preserves_discriminator_update(factorize, noise_scale, amp):
    torch.manual_seed(7)
    args = arguments(*(["--factorize"] if factorize else []), *(["--amp"] if amp else []),
                     "--discriminator-motion-noise-scale", str(noise_scale))
    model = build_discriminator(args, 2)
    other = copy.deepcopy(model)
    generated = torch.randn(6, 2, 51) * .01
    expert = torch.randn(6, 2, 51) * .01
    generated[:2, :, 0] = .8
    expert[2:4, :, 0] = .8
    args.discriminator_update_microbatch_size = 6
    rng = torch.get_rng_state()
    full = train_discriminator_batch(model, torch.optim.SGD(model.parameters(), lr=.01), generated, expert, args)
    args.discriminator_update_microbatch_size = 2
    torch.set_rng_state(rng)
    micro = train_discriminator_batch(other, torch.optim.SGD(other.parameters(), lr=.01), generated, expert, args)
    torch.testing.assert_close(full, micro)
    for expected, actual in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(expected, actual)


def test_expert_classifier_gradient_moves_logits_toward_correct_labels():
    class TwoLogits(nn.Module):

        factorized = False

        def __init__(self):
            super().__init__()
            self.values = nn.Parameter(torch.zeros(2))

        def forward(self, transitions):
            return self.values

    model = TwoLogits()
    loss = discriminator_loss(model, torch.zeros(2, 2, 1), torch.tensor([1., 0.]))
    loss.backward()
    assert model.values.grad[0] < 0
    assert model.values.grad[1] > 0


def test_expert_pairs_exclude_invalid_frames(tmp_path):
    rows = np.zeros((8, team_replay_row_size(1)), np.float32)
    rows[:, 0] = np.arange(8)
    rows[3, -1] = 1
    rows[6, -2] = 1
    np.save(tmp_path / "pro.npy", rows)
    experts = ExpertDataset(tmp_path, 1, None, "cpu")
    assert experts.starts.tolist() == [0, 1, 4]
    transitions = experts.sample(100)
    torch.testing.assert_close(transitions[:, 1, 0] - transitions[:, 0, 0], torch.ones(100))


def test_generated_holdout_is_simulation_disjoint():
    train, heldout, simulations = split_generated(torch.zeros(4, 8), None, 1, 2)
    assert len(heldout) == 8
    assert not set(((train % 8) // 2).tolist()) & set(((heldout % 8) // 2).tolist())
    assert sorted(torch.cat((train, heldout)).tolist()) == list(range(32))
    assert simulations.sum() == 1


def test_legacy_discriminator_cannot_resume(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo",
                "config": {"discriminator_architecture": "legacy-sequence"}}, path)
    with pytest.raises(ValueError, match="discriminator architecture differs"):
        load_checkpoint(path, "gaifo", arguments(), None, None)
