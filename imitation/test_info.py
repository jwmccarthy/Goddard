import argparse
import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from imitation.absorbing import AbsorbingRollout
from imitation.info import InfoGAIL, TransitionPosterior
from imitation.rollout import TransitionRollout
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint, save_checkpoint


def arguments(*flags):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=True)
    add_gaifo_arguments(parser)
    args = parser.parse_args(["--frame-embedding", "8", "--discriminator-hidden", "8",
                              "--discriminator-window-length", "2", *flags])
    resolve_gaifo_defaults(args)
    return args


def test_info_is_opt_in_with_categorical_defaults():
    assert not arguments().info
    args = arguments("--info")
    assert args.info_codes == 8 and args.info_weight == .1
    assert args.info_objective == "categorical-state-window-v1"


@pytest.mark.parametrize("flag,value,message", [
    ("codes", "1", "two latent"), ("codes", "0", "two latent"),
    ("weight", "-1", "weight"), ("weight", "nan", "weight"), ("weight", "inf", "weight"),
    ("learning-rate", "0", "learning rate"), ("learning-rate", "nan", "learning rate"),
    ("learning-rate", "inf", "learning rate"), ("epochs", "0", "epoch"), ("batch-size", "0", "batch"),
])
def test_info_arguments_are_validated(flag, value, message):
    with pytest.raises(ValueError, match=message):
        arguments("--info", f"--info-{flag}={value}")


@pytest.mark.parametrize("absorbing", [False, True])
def test_uniform_frozen_posterior_gives_exact_log_q_reward_and_no_terminal_bonus(absorbing):
    args = arguments("--info", "--info-codes", "3", "--info-batch-size", "2")
    info = InfoGAIL(args, 2, "cpu")
    for parameter in info.posterior.parameters():
        nn.init.zeros_(parameter)
    obs = torch.zeros(2, 5)
    scenes = (AbsorbingRollout if absorbing else TransitionRollout)(obs, 3, 2, 2)
    rollout = SimpleNamespace(observations=torch.zeros(3, 2, 5), rewards=torch.full((3, 2), 7.))
    rollout.observations[..., -3:] = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    for step in range(3):
        terminated = torch.tensor([step == 1, False])
        scenes.record(step, obs, obs + 1, obs + 2, terminated, terminated)
    train = torch.tensor([0, 2, 4, 8]) if absorbing else torch.tensor([0, 2, 4])
    metrics = info.update(scenes, rollout, train, torch.tensor([1, 3, 5]))
    expected = torch.full((3, 2), 7. - args.info_weight * math.log(3))
    expected[1, 0] = 7.
    torch.testing.assert_close(rollout.rewards, expected)
    assert metrics["info/mi_lower_bound"] == pytest.approx(0., abs=1e-6)
    assert metrics["info/train_windows"] == 2
    assert metrics["info/heldout_windows"] == 3
    assert metrics["info/valid_fraction"] == pytest.approx(5 / 6)
    assert metrics["info/reward"] == pytest.approx(-args.info_weight * math.log(3) * 5 / 6)
    assert not rollout.rewards.requires_grad


def test_posterior_only_trains_on_live_training_simulations_without_codes_actions_or_absorbing_flags():
    args = arguments("--info", "--info-codes", "3")
    info = InfoGAIL(args, 2, "cpu")

    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.output = nn.Linear(2, 3)
            self.trained = []

        def forward(self, frames):
            assert frames.shape[-1] == 2
            if torch.is_grad_enabled():
                self.trained.extend(frames[:, -1, 0].tolist())
            return self.output(frames[:, -1])

    info.posterior = Recorder()
    info.optimizer = torch.optim.Adam(info.posterior.parameters())
    obs = torch.zeros(2, 5)
    scenes = AbsorbingRollout(obs, 3, 2, 2)
    rollout = SimpleNamespace(observations=torch.zeros(3, 2, 5), rewards=torch.zeros(3, 2))
    rollout.observations[..., -3] = 1
    for step in range(3):
        next_obs = obs.clone()
        next_obs[:, 0] = torch.tensor([step * 2 + 1., step * 2 + 2.])
        terminated = torch.tensor([step == 1, False])
        scenes.record(step, obs, next_obs, next_obs, terminated, terminated)
    info.update(scenes, rollout, torch.tensor([0, 2, 4, 8]), torch.tensor([1, 3, 5]))
    assert set(info.posterior.trained) == {1., 5.}


def test_insufficient_strided_history_never_trains_or_rewards_a_single_state():
    args = arguments("--info", "--discriminator-window-length", "4", "--discriminator-decision-stride", "3")
    info = InfoGAIL(args, 2, "cpu")
    before = copy.deepcopy(info.posterior.state_dict())
    obs = torch.zeros(2, 2 + args.info_codes)
    rollout = SimpleNamespace(observations=obs[None], rewards=torch.ones(1, 2))
    scenes = TransitionRollout(obs, 1, 2, 4, decision_stride=3)
    scenes.record(0, obs, obs + 1, obs + 1, torch.zeros(2, dtype=torch.bool))
    metrics = info.update(scenes, rollout, torch.tensor([0]), torch.tensor([1]))
    torch.testing.assert_close(rollout.rewards, torch.ones(1, 2))
    assert metrics["info/train_windows"] == metrics["info/heldout_windows"] == metrics["info/valid_fraction"] == 0
    assert not info.optimizer.state
    for name, value in info.posterior.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_state_only_posterior_can_learn_decodable_modes_on_unseen_windows():
    torch.manual_seed(2)
    args = arguments("--info", "--info-codes", "2", "--info-learning-rate", ".03")
    info = InfoGAIL(args, 2, "cpu")
    obs = torch.zeros(64, 4)
    labels = torch.arange(64) % 2
    obs[:, 2:] = torch.nn.functional.one_hot(labels, 2)
    obs[:, 0] = labels.float() * 2 - 1
    obs[:, 1] = torch.randn(64) * .01
    scenes = TransitionRollout(obs, 1, 2, 2)
    scenes.record(0, obs, obs, obs, torch.zeros(64, dtype=torch.bool))
    rollout = SimpleNamespace(observations=obs[None], rewards=torch.zeros(1, 64))
    train, heldout = torch.arange(32), torch.arange(32, 64)
    initial = info.evaluate(scenes, heldout, labels)
    for _ in range(24):
        metrics = info.update(scenes, rollout, train, heldout)
    assert metrics["info/heldout_accuracy"] > .95
    assert metrics["info/heldout_log_q"] > initial["info/heldout_log_q"] + .4


@pytest.mark.parametrize("relative", [False, True])
def test_posterior_masks_padding_and_supports_relative_causal_windows(relative):
    args = arguments("--info", "--discriminator-window-length", "4",
                     *(["--discriminator-relative"] if relative else []))
    posterior = TransitionPosterior(args, 5)
    frames = torch.randn(3, 4, 5)
    frames[0, :2] = float("nan")
    frames[1, :1] = float("nan")
    result = posterior(frames)
    assert result.shape == (3, 8) and torch.isfinite(result).all()
    for i in range(3):
        torch.testing.assert_close(result[i:i + 1], posterior(frames[i:i + 1]))
    result.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in posterior.parameters())


@pytest.mark.parametrize("old_flags,new_flags,message", [
    ([], ["--info"], "info mode"), (["--info"], [], "info mode"),
    (["--info"], ["--info", "--info-codes", "4"], "info_codes"),
    (["--info"], ["--info", "--info-weight", ".2"], "info_weight"),
])
def test_incompatible_info_checkpoints_fail_before_loading_weights(tmp_path, old_flags, new_flags, message):
    old, new = arguments(*old_flags), arguments(*new_flags)
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "goddard-leanrl-v1", "algorithm": "gaifo", "config": vars(old)}, path)
    with pytest.raises(ValueError, match=message):
        load_checkpoint(path, "gaifo", new, None, None)


def test_info_checkpoint_round_trip_restores_posterior_and_optimizer(tmp_path):
    args = arguments("--info")
    args.discriminator_heldout_replays = []
    model = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters())
    info = InfoGAIL(args, 2, "cpu")
    loss = info.posterior(torch.randn(4, 2, 2)).square().mean()
    loss.backward()
    info.optimizer.step()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, "gaifo", args, model, optimizer, 10, **info.checkpoint())
    payload = load_checkpoint(path, "gaifo", args, model, optimizer)
    restored = InfoGAIL(args, 2, "cpu")
    restored.load_checkpoint(payload)
    assert len(restored.optimizer.state) == len(info.optimizer.state) > 0
    for left, right in zip(restored.posterior.parameters(), info.posterior.parameters()):
        torch.testing.assert_close(left, right)
    with torch.no_grad():
        next(info.posterior.parameters()).fill_(float("nan"))
    previous = path.read_bytes()
    with pytest.raises(FloatingPointError, match="info_posterior"):
        save_checkpoint(path, "gaifo", args, model, optimizer, 11, **info.checkpoint())
    assert path.read_bytes() == previous
