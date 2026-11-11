from types import SimpleNamespace

import pytest
import torch

from imitation import update
from imitation.evaluation import evaluate_discriminator
from imitation.models import EgoSceneDiscriminator, FactorizedSceneDiscriminator
from imitation.policy_history import PolicyWindowHistory


class Scenes:

    def __init__(self):
        self.requested = []

    def transitions(self, indices):
        self.requested.extend(indices.tolist())
        windows = torch.zeros(len(indices), 2, 51)
        windows[indices % 2 == 0, :, 0] = 1
        windows[:, :, 5] = indices[:, None]
        return windows


class Experts:

    def sample(self, count):
        windows = torch.zeros(count, 2, 51)
        windows[::2, :, 0] = 1
        return windows


def arguments():
    return SimpleNamespace(
        discriminator_accuracy_target=.8,
        discriminator_accuracy_target_end=.8,
        discriminator_updates=1,
        discriminator_batch_size=4,
        discriminator_update_microbatch_size=2,
        discriminator_history_mix_fraction=.5,
        bf16=False,
        max_grad_norm=.5
    )


@pytest.mark.parametrize("amp", [False, True])
def test_only_below_threshold_head_updates_even_with_adam_momentum(monkeypatch, amp):
    torch.manual_seed(7)
    model = FactorizedSceneDiscriminator(8, 8, 2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    scenes, experts, args = Scenes(), Experts(), arguments()
    args.amp = amp
    indices = torch.arange(4)
    update.train_discriminator_batch(model, optimizer, scenes.transitions(indices), experts.sample(4), args)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    steps = {id(p): optimizer.state[p]["step"].clone() for p in model.parameters()}
    monkeypatch.setattr(update, "sample_discriminator_validation", lambda *args: torch.zeros(2, 2, 51))
    monkeypatch.setattr(update, "evaluate_discriminator", lambda *args: {
        "discriminator/ego_heldout_accuracy": .7,
        "discriminator/global_heldout_accuracy": .9
    })
    history = PolicyWindowHistory(8, 2)
    result = update.update_discriminator(model, optimizer, experts, scenes, indices, torch.tensor([4, 5]), args, history)
    assert result["discriminator/ego_update_enabled"] == 1
    assert result["discriminator/global_update_enabled"] == 0
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items()
               if name.startswith("ego_discriminator."))
    for name, parameter in model.named_parameters():
        if not name.startswith("ego_discriminator."):
            torch.testing.assert_close(before[name], parameter, rtol=0, atol=0)
            torch.testing.assert_close(steps[id(parameter)], optimizer.state[parameter]["step"])
    assert history.size == 2
    assert not set(scenes.requested) & {4, 5}


def test_no_update_when_all_heads_meet_target(monkeypatch):
    model = FactorizedSceneDiscriminator(8, 8, 2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    monkeypatch.setattr(update, "sample_discriminator_validation", lambda *args: torch.zeros(2, 2, 51))
    monkeypatch.setattr(update, "evaluate_discriminator", lambda *args: {
        f"discriminator/{name}_heldout_accuracy": .8 for name in ("ego", "global")
    })
    history = PolicyWindowHistory(8, 2)
    metrics = update.update_discriminator(model, optimizer, Experts(), Scenes(), torch.arange(4),
                                          torch.tensor([4]), arguments(), history)
    assert metrics["discriminator/updated"] == 0
    assert not optimizer.state
    assert history.size == 2


def test_ego_only_accuracy_gates_training(monkeypatch):
    model = EgoSceneDiscriminator(8, 8, 2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    monkeypatch.setattr(update, "sample_discriminator_validation", lambda *args: torch.zeros(2, 2, 51))
    monkeypatch.setattr(update, "evaluate_discriminator", lambda *args: {
        "discriminator/ego_heldout_accuracy": .7
    })
    metrics = update.update_discriminator(model, optimizer, Experts(), Scenes(), torch.arange(4),
                                          torch.tensor([4]), arguments(), PolicyWindowHistory(8, 2))
    assert metrics["discriminator/updated"] == 1
    assert metrics["discriminator/ego_update_enabled"] == 1
    assert optimizer.state


def test_accuracy_target_anneals_and_gates_each_head(monkeypatch):
    model = FactorizedSceneDiscriminator(8, 8, 2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    args = arguments()
    args.discriminator_accuracy_target = .85
    args.discriminator_accuracy_target_end = 1.0
    monkeypatch.setattr(update, "sample_discriminator_validation", lambda *args: torch.zeros(2, 2, 51))
    monkeypatch.setattr(update, "evaluate_discriminator", lambda *args: {
        "discriminator/ego_heldout_accuracy": .90,
        "discriminator/global_heldout_accuracy": .96
    })
    history = PolicyWindowHistory(8, 2)
    for progress, target, ego, global_head in [
        (0, .85, 0, 0), (.5, .925, 1, 0), (1, 1., 1, 1), (2, 1., 1, 1)
    ]:
        metrics = update.update_discriminator(model, optimizer, Experts(), Scenes(), torch.arange(4),
                                              torch.tensor([4]), args, history, progress=progress)
        assert abs(metrics["discriminator/accuracy_target"] - target) < 1e-7
        assert metrics["discriminator/ego_update_enabled"] == ego
        assert metrics["discriminator/global_update_enabled"] == global_head


def test_validation_scores_both_heads_without_proximity_bands():
    model = FactorizedSceneDiscriminator(8, 8, 2, 2)
    validation = torch.zeros(4, 2, 51)
    validation[:, :, 0] = 1
    args = SimpleNamespace(discriminator_inference_microbatch_size=8)
    metrics = evaluate_discriminator(model, None, None, None, args, validation)
    assert "discriminator/ego_heldout_accuracy" in metrics
    assert "discriminator/global_heldout_accuracy" in metrics


def test_history_ring_contains_only_previously_added_windows():
    torch.manual_seed(0)
    history = PolicyWindowHistory(3, 2)
    scenes = Scenes()
    history.add(scenes, torch.tensor([0, 1]))
    first = history.sample(20)
    assert set(first[:, 0, 0].tolist()) == {0, 1}
    history.add(scenes, torch.tensor([2, 3]))
    assert history.size == 3
    assert history.windows.shape == (3, 2, 51)


def test_history_adds_every_four_rollouts_and_keeps_spaced_snapshots():
    history = PolicyWindowHistory(16, 1, interval=4)
    scenes = Scenes()
    for rollout in range(65):
        added = history.add(scenes, torch.tensor([rollout]))
        assert added == int(rollout % 4 == 0)
    assert history.size == 16
    assert set(history.windows[:, 0, 5].tolist()) == set(range(4, 65, 4))
    assert scenes.requested == list(range(0, 65, 4))


def test_history_cadence_continues_when_no_windows_are_eligible():
    history = PolicyWindowHistory(4, 1, interval=4)
    scenes = Scenes()
    assert history.add(scenes, torch.tensor([], dtype=torch.long)) == 0
    for _ in range(3):
        assert history.add(scenes, torch.tensor([1])) == 0
    assert history.add(scenes, torch.tensor([2])) == 1
    assert scenes.requested == [2]


def test_update_mixes_only_prior_training_windows(monkeypatch):
    model = FactorizedSceneDiscriminator(8, 8, 2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    scenes, args = Scenes(), arguments()
    history = PolicyWindowHistory(8, 4)
    history.add(scenes, torch.arange(4))
    monkeypatch.setattr(update, "sample_discriminator_validation", lambda *args: torch.zeros(2, 2, 51))
    monkeypatch.setattr(update, "evaluate_discriminator", lambda *args: {
        "discriminator/global_heldout_accuracy": .7
    })
    batches = []

    def train(model, optimizer, generated, expert, args, active):
        batches.append(generated.clone())
        assert active == (1,)
        return torch.tensor(1.0)

    monkeypatch.setattr(update, "train_discriminator_batch", train)
    update.update_discriminator(model, optimizer, Experts(), scenes, torch.arange(10, 14),
                                torch.tensor([20]), args, history)
    assert len(batches) == 1
    ids = batches[0][:, 0, 5]
    assert int((ids < 4).sum()) == 2
    assert int((ids >= 10).sum()) == 2
    assert 20 not in scenes.requested
