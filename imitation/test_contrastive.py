import argparse
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from imitation import update
from imitation.absorbing import AbsorbingDiscriminator, AbsorbingRollout, absorbing_reward
from imitation.contrastive import consistency_loss, source_balanced_weights, temporal_motion_view
from imitation.policy_history import PolicyWindowHistory
from imitation.rewards import imitation_rewards
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults
from training.checkpoints import load_checkpoint


def arguments(*flags):
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=True)
    add_gaifo_arguments(parser)
    args = parser.parse_args(['--frame-embedding', '8', '--discriminator-hidden', '8',
                              '--discriminator-window-length', '4', '--mlp-discriminator',
                              '--discriminator-decision-stride', '3', *flags])
    resolve_gaifo_defaults(args)
    return args


def windows(count=8, length=4, cars=2, absorbing=True, dtype=torch.float32):
    result = torch.zeros(count, length, 9 + 21 * cars + int(absorbing), dtype=dtype)
    result[..., 2] = 1000 / 2076
    for car in range(cars):
        start = 9 + car * 21
        direction = 1 if car % 2 == 0 else -1
        result[..., start] = (-direction * 1500 + direction * 600 * .1 * torch.arange(length)) / 4108
        result[..., start + 1] = (-2000 + 800 * car) / 6000
        result[..., start + 2] = 17 / 2076
        result[..., start + 3] = direction * 600 / 2300
        result[..., start + 9] = direction
        result[..., start + 14] = 1
        result[..., start + 15] = .5
        result[..., start + 16] = 1
    return result


class Toy(torch.nn.Module):
    factorized = True

    def __init__(self):
        super().__init__()
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(.2)), torch.nn.Parameter(torch.tensor(.3))])
        self.calls = []

    def head_score(self, inputs, head):
        self.calls.append((head, inputs.detach().clone()))
        midpoint = (inputs[:, 0, 9] * 2 + inputs[:, -1, 9]) / 3
        return self.weights[head] * (inputs[:, 1, 9] - midpoint) * 500

    def forward(self, inputs):
        return torch.stack([self.head_score(inputs, head) for head in (0, 1)], dim=-1)


def test_contrastive_is_opt_in_with_bounded_defaults():
    args = arguments()
    assert not args.discriminator_contrastive
    enabled = arguments('--discriminator-contrastive')
    assert enabled.discriminator_contrastive
    assert enabled.discriminator_contrastive_weight == 1
    assert enabled.discriminator_contrastive_radius == 40
    with pytest.raises(ValueError, match='at least three'):
        arguments('--discriminator-contrastive', '--discriminator-window-length', '2')


@pytest.mark.parametrize('flag,value,message', [
    ('weight', '-1', 'weight'), ('weight', 'nan', 'weight'), ('weight', 'inf', 'weight'),
    ('radius', '0', 'radius'), ('radius', '-1', 'radius'), ('radius', 'nan', 'radius'), ('radius', 'inf', 'radius'),
])
def test_contrastive_hyperparameters_are_validated(flag, value, message):
    with pytest.raises(ValueError, match=message):
        arguments('--discriminator-contrastive', f'--discriminator-contrastive-{flag}={value}')


@pytest.mark.parametrize('cars', [2, 4, 6])
@pytest.mark.parametrize('absorbing', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16])
def test_bridge_is_bounded_preserves_endpoints_and_only_changes_car_xy(cars, absorbing, dtype):
    torch.manual_seed(42)
    original = windows(count=32, cars=cars, absorbing=absorbing, dtype=dtype)
    before = original.clone()
    transformed, eligible = temporal_motion_view(original, .1, 40)
    assert transformed.dtype == original.dtype and transformed.device == original.device
    assert eligible.all()
    torch.testing.assert_close(original, before)
    torch.testing.assert_close(transformed[:, [0, -1]], original[:, [0, -1]], rtol=0, atol=0)
    changed_columns = [9 + 21 * car + field for car in range(cars) for field in (0, 1)]
    unchanged = [i for i in range(original.shape[-1]) if i not in changed_columns]
    torch.testing.assert_close(transformed[..., unchanged], original[..., unchanged], rtol=0, atol=0)
    for car in range(cars):
        start = 9 + 21 * car
        delta = (transformed[..., start:start + 2].double() - original[..., start:start + 2].double()) * torch.tensor([4108, 6000])
        assert delta.norm(dim=-1).max() <= 40.001


def test_short_and_ineligible_windows_are_identity_without_random_draws():
    for original in (windows(length=2), torch.zeros(4, 4, 52)):
        state = torch.get_rng_state()
        transformed, eligible = temporal_motion_view(original, .1)
        assert transformed is original and not eligible.any()
        assert torch.equal(state, torch.get_rng_state())
    with pytest.raises(ValueError, match='frame interval'):
        temporal_motion_view(windows(), 0)
    with pytest.raises(ValueError, match='scene windows'):
        temporal_motion_view(torch.zeros(2, 4, 42), .1)


def test_padding_absorbing_and_terminal_windows_are_untouched():
    original = windows(4)
    original[0, 0] = float('nan')
    original[1, -1] = 0
    original[1, -1, -1] = 1
    original[2] = 0
    original[2, :, -1] = 1
    transformed, eligible = temporal_motion_view(original, .1)
    assert eligible.tolist() == [False, False, False, True]
    torch.testing.assert_close(transformed[:3], original[:3], equal_nan=True, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['ball_close', 'cars_close', 'wall', 'demo', 'jump', 'boost_change',
                                  'rotate', 'accelerate', 'ground_change', 'ball_impulse'])
def test_contact_and_event_gates_preserve_affected_cars(kind):
    original = windows(4)
    if kind == 'ball_close':
        original[..., :3] = original[..., 9:12]
        original[..., 2] += 100 / 2076
    elif kind == 'cars_close':
        original[..., 30:33] = original[..., 9:12]
    elif kind == 'wall':
        original[..., 9] = 3800 / 4108
    elif kind == 'demo':
        original[:, 1, 26] = 1
    elif kind == 'jump':
        original[:, 1, 27] = 1
    elif kind == 'boost_change':
        original[:, 1, 29] = 1
    elif kind == 'rotate':
        original[:, 1, 18:21] = torch.tensor([0., 1., 0.])
    elif kind == 'accelerate':
        original[:, 1, 12] += 1000 / 2300
    elif kind == 'ground_change':
        original[:, 1, 25] = 0
    elif kind == 'ball_impulse':
        original[:, 1, 3] = .5
    transformed, _ = temporal_motion_view(original, .1)
    torch.testing.assert_close(transformed[..., 9:30], original[..., 9:30], rtol=0, atol=0)


def test_long_window_keeps_event_neighborhood_but_augments_other_smooth_regions():
    original = windows(16, length=9)
    original[:, 4, 27] = 1
    transformed, eligible = temporal_motion_view(original, .1)
    assert eligible.all()
    torch.testing.assert_close(transformed[:, 3:6, 9:30], original[:, 3:6, 9:30], rtol=0, atol=0)
    assert (transformed[:, 1:3, 9:11] != original[:, 1:3, 9:11]).any()
    assert (transformed[:, 6:8, 9:11] != original[:, 6:8, 9:11]).any()


def test_temporal_draws_are_correlated_and_symmetric_between_sources():
    torch.manual_seed(781)
    original = windows(4096, length=9)
    transformed, eligible = temporal_motion_view(original, .1)
    assert eligible.all()
    delta = (transformed[..., 9:11] - original[..., 9:11]) * torch.tensor([4108, 6000])
    adjacent = torch.corrcoef(torch.stack((delta[:, 3].flatten(), delta[:, 4].flatten())))[0, 1]
    distant = torch.corrcoef(torch.stack((delta[:, 1].flatten(), delta[:, 7].flatten())))[0, 1]
    assert adjacent > .5 and adjacent > distant + .3
    generated, expert = delta.chunk(2)
    assert abs(generated.mean().item()) < .5 and abs(expert.mean().item()) < .5
    assert generated.std().item() == pytest.approx(expert.std().item(), rel=.06)
    assert not torch.equal(generated, expert)


def test_consistency_balances_sources_not_eligible_counts():
    class Single(torch.nn.Module):
        factorized = False

        def forward(self, inputs):
            return inputs[:, 1, 9]

    clean = torch.zeros(8, 4, 51)
    perturbed = clean.clone()
    perturbed[:3, 1, 9] = 1
    perturbed[4, 1, 9] = 3
    eligible = torch.tensor([True, True, True, False, True, False, False, False])
    weight = source_balanced_weights(eligible)
    loss = consistency_loss(Single(), clean, perturbed, weight)
    assert loss.item() == pytest.approx(10.)
    assert weight[:4].sum() == pytest.approx(.5) and weight[4:].sum() == pytest.approx(.5)
    weight = source_balanced_weights(torch.tensor([False] * 4 + [True] * 4))
    assert weight[:4].sum() == 0 and weight[4:].sum() == pytest.approx(.5)
    for bad in (torch.zeros(0, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)):
        with pytest.raises(ValueError, match='equal nonempty'):
            source_balanced_weights(bad)


def test_clean_and_augmented_views_are_shared_between_heads_and_never_stored():
    args = arguments('--factorize', '--discriminator-contrastive', '--discriminator-update-microbatch-size', '8',
                     '--discriminator-motion-noise-scale', '0')
    model = Toy()
    generated, expert = windows(4, absorbing=False), windows(4, absorbing=False)
    before = generated.clone()
    stats = {}
    loss = update.train_discriminator_batch(model, torch.optim.SGD(model.parameters(), lr=.001),
                                             generated, expert, args, metrics=stats)
    assert torch.isfinite(loss) and stats['contrastive_loss'] > 0
    assert len(model.calls) == 6
    for a, b in ((2, 4), (3, 5)):
        torch.testing.assert_close(model.calls[a][1], model.calls[b][1])
    torch.testing.assert_close(model.calls[0][1], model.calls[1][1])
    torch.testing.assert_close(model.calls[0][1], model.calls[3][1])
    torch.testing.assert_close(model.calls[2][1], torch.cat((before, before)))
    assert not torch.equal(model.calls[3][1][:4], model.calls[3][1][4:])
    torch.testing.assert_close(generated, before)
    torch.testing.assert_close(expert, before)


@pytest.mark.parametrize('architecture', ['mlp', 'gru', 'transformer'])
@pytest.mark.parametrize('mode', ['global', 'ego', 'factorized'])
@pytest.mark.parametrize('relative', [False, True])
def test_consistency_supports_every_discriminator_architecture(architecture, mode, relative):
    flags = (['--factorize'] if mode == 'factorized' else ['--ego-only'] if mode == 'ego' else [])
    args = arguments('--discriminator-contrastive', *flags,
                     *(['--discriminator-relative'] if relative else []))
    args.mlp_discriminator = architecture == 'mlp'
    args.gru = architecture == 'gru'
    model = AbsorbingDiscriminator(args, 2)
    stats = {}
    loss = update.train_discriminator_batch(model, torch.optim.Adam(model.parameters()),
                                             windows(4), windows(4), args, metrics=stats)
    assert torch.isfinite(loss) and torch.isfinite(stats['contrastive_loss'])
    assert stats['contrastive_generated_fraction'] == 1
    assert stats['contrastive_expert_fraction'] == 1


@pytest.mark.parametrize('factorized', [False, True])
def test_accumulated_update_is_independent_of_microbatch_size(factorized):
    flags = ['--discriminator-contrastive', *(['--factorize'] if factorized else [])]
    small = arguments(*flags, '--discriminator-update-microbatch-size', '2')
    big = arguments(*flags, '--discriminator-update-microbatch-size', '8')
    torch.manual_seed(419)
    left = AbsorbingDiscriminator(small, 2)
    right = copy.deepcopy(left)
    generated, expert = windows(), windows()
    losses, states = [], []
    for model, args in ((left, small), (right, big)):
        torch.manual_seed(317)
        losses.append(update.train_discriminator_batch(model, torch.optim.SGD(model.parameters(), lr=.01),
                                                       generated, expert, args))
        states.append(torch.get_rng_state())
    torch.testing.assert_close(losses[0], losses[1], rtol=1e-6, atol=1e-6)
    assert torch.equal(*states)
    for a, b in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


def test_disabled_feature_preserves_legacy_update_and_random_stream():
    args = arguments('--factorize')
    legacy = copy.deepcopy(args)
    for key in ('discriminator_contrastive', 'discriminator_contrastive_weight', 'discriminator_contrastive_radius'):
        delattr(legacy, key)
    torch.manual_seed(317)
    left, right = Toy(), Toy()
    losses, states = [], []
    for model, config in ((left, args), (right, legacy)):
        torch.manual_seed(192)
        losses.append(update.train_discriminator_batch(model, torch.optim.SGD(model.parameters(), lr=.001),
                                                       windows(absorbing=False), windows(absorbing=False), config))
        states.append(torch.get_rng_state())
    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    assert torch.equal(*states)
    for a, b in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_zero_weight_matches_disabled_feature_without_drawing_bridge_noise():
    enabled = arguments('--factorize', '--discriminator-contrastive', '--discriminator-contrastive-weight', '0')
    disabled = arguments('--factorize')
    left, right = Toy(), Toy()
    for model, args in ((left, enabled), (right, disabled)):
        torch.manual_seed(743)
        update.train_discriminator_batch(model, torch.optim.SGD(model.parameters(), lr=.001),
                                         windows(absorbing=False), windows(absorbing=False), args)
    for a, b in zip(left.parameters(), right.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_consistency_can_update_above_accuracy_gate_without_classification(monkeypatch):
    args = arguments('--factorize', '--discriminator-contrastive', '--discriminator-batch-size', '8')
    args.discriminator_accuracy_target = args.discriminator_accuracy_target_end = .8
    model = Toy()
    before = [p.clone() for p in model.parameters()]
    validation = windows(8, absorbing=False)
    monkeypatch.setattr(update, 'sample_discriminator_validation', lambda *unused: validation)
    monkeypatch.setattr(update, 'evaluate_discriminator', lambda *unused: {
        'discriminator/ego_heldout_accuracy': .9, 'discriminator/global_heldout_accuracy': .9})
    scenes = SimpleNamespace(transitions=lambda indices: windows(len(indices), absorbing=False))
    experts = SimpleNamespace(sample=lambda count: windows(count, absorbing=False))
    history = PolicyWindowHistory(0, 0)
    result = update.update_discriminator(model, torch.optim.SGD(model.parameters(), lr=.001), experts, scenes,
                                         torch.arange(8), torch.arange(8), args, history)
    assert result['discriminator/updated'] == 1
    assert result['discriminator/classification_loss'] == 0
    assert result['discriminator/contrastive_loss'] > 0
    assert result['discriminator/ego_classification_update_enabled'] == 0
    assert result['discriminator/global_contrastive_update_enabled'] == 1
    assert all(abs(p.item()) < abs(old.item()) for p, old in zip(model.parameters(), before))
    for name in ('ego', 'global'):
        assert result[f'discriminator/{name}_contrastive_expert_heldout_mse'] < result[f'discriminator/pre_{name}_contrastive_expert_heldout_mse']
    torch.testing.assert_close(validation, windows(8, absorbing=False))


def test_no_eligible_consistency_only_batch_does_not_step_optimizer(monkeypatch):
    args = arguments('--factorize', '--discriminator-contrastive')
    model = Toy()
    optimizer = torch.optim.Adam(model.parameters())
    monkeypatch.setattr(update, 'sample_discriminator_validation', lambda *unused: torch.zeros(4, 4, 51))
    monkeypatch.setattr(update, 'evaluate_discriminator', lambda *unused: {
        'discriminator/ego_heldout_accuracy': 1., 'discriminator/global_heldout_accuracy': 1.})
    scenes = SimpleNamespace(transitions=lambda indices: torch.zeros(len(indices), 4, 51))
    experts = SimpleNamespace(sample=lambda count: torch.zeros(count, 4, 51))
    result = update.update_discriminator(model, optimizer, experts, scenes, torch.arange(4), torch.arange(4),
                                         args, PolicyWindowHistory(0, 0))
    assert not optimizer.state and result['discriminator/updated'] == 0
    assert result['discriminator/contrastive_generated_fraction'] == 0
    assert result['discriminator/contrastive_expert_fraction'] == 0


def test_validation_reuses_the_same_clean_and_temporal_pairs_before_after(monkeypatch):
    args = arguments('--factorize', '--discriminator-contrastive')
    validation = windows(8, absorbing=False)
    pairs, classification_views = [], []
    monkeypatch.setattr(update, 'sample_discriminator_validation', lambda *unused: validation)

    def evaluate_classification(*values):
        classification_views.append(values[-1])
        return {'discriminator/ego_heldout_accuracy': .5, 'discriminator/global_heldout_accuracy': .5}

    def evaluate_pairs(model, clean, transformed, eligible, config):
        pairs.append((clean, transformed, eligible))
        return {'discriminator/contrastive_expert_heldout_fraction': eligible.float().mean().item()}

    monkeypatch.setattr(update, 'evaluate_discriminator', evaluate_classification)
    monkeypatch.setattr(update, 'evaluate_consistency', evaluate_pairs)
    model = Toy()
    scenes = SimpleNamespace(transitions=lambda indices: windows(len(indices), absorbing=False))
    experts = SimpleNamespace(sample=lambda count: windows(count, absorbing=False))
    history = PolicyWindowHistory(8, 4)
    update.update_discriminator(model, torch.optim.SGD(model.parameters(), lr=.001), experts, scenes,
                                torch.arange(4), torch.arange(4), args, history)
    assert len(pairs) == 2
    assert all(a is b for a, b in zip(pairs[0], pairs[1]))
    assert pairs[0][0] is validation
    assert classification_views[0] is classification_views[1]
    assert classification_views[2] is validation
    torch.testing.assert_close(history.windows[:history.size], windows(4, absorbing=False))
    torch.testing.assert_close(validation, windows(8, absorbing=False))


def test_inactive_consistency_head_keeps_adam_state_frozen():
    args = arguments('--factorize', '--discriminator-contrastive')
    model = Toy()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    update.train_discriminator_batch(model, optimizer, windows(absorbing=False), windows(absorbing=False), args)
    before = model.weights[1].clone()
    step = optimizer.state[model.weights[1]]['step'].clone()
    update.train_discriminator_batch(model, optimizer, windows(absorbing=False), windows(absorbing=False), args,
                                     active_heads=(), contrastive_heads=(0,))
    torch.testing.assert_close(model.weights[1], before, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state[model.weights[1]]['step'], step, rtol=0, atol=0)


def test_rewards_potential_and_absorbing_value_inference_stay_clean_and_deterministic():
    args = arguments('--factorize', '--discriminator-contrastive', '--discriminator-relative',
                     '--discriminator-log-reward', '--discriminator-potential-shaping')
    args.reward_gamma = .99
    model = AbsorbingDiscriminator(args, 2)
    obs = windows(2, absorbing=False)[:, 0]
    scenes = AbsorbingRollout(obs, 4, 51, 4, potential_shaping=True, decision_stride=3)
    for step in range(4):
        done = torch.zeros(2, dtype=torch.bool)
        scenes.record(step, obs, obs, obs, done, done)
    stored = scenes.all_frames().clone()
    enabled, disabled = SimpleNamespace(rewards=torch.zeros(4, 2)), SimpleNamespace(rewards=torch.zeros(4, 2))
    rng = torch.get_rng_state()
    imitation_rewards(model, scenes, enabled, args, include_terminal=True)
    absorbing_enabled = absorbing_reward(model, args, 'cpu', 51)
    args.discriminator_contrastive = False
    imitation_rewards(model, scenes, disabled, args, include_terminal=True)
    absorbing_disabled = absorbing_reward(model, args, 'cpu', 51)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(enabled.rewards, disabled.rewards, rtol=0, atol=0)
    torch.testing.assert_close(absorbing_enabled, absorbing_disabled, rtol=0, atol=0)
    torch.testing.assert_close(scenes.all_frames(), stored, equal_nan=True, rtol=0, atol=0)


@pytest.mark.parametrize('old,new,weight,radius,warn', [
    (None, False, 1., 40., False), (None, True, 1., 40., True),
    (True, False, 1., 40., True), (True, True, 1., 40., False),
    (True, True, .5, 40., True), (True, True, 1., 20., True),
])
def test_checkpoint_contrastive_compatibility_warns_without_rejecting(tmp_path, old, new, weight, radius, warn):
    flags = ['--discriminator-contrastive'] if new else []
    args = arguments(*flags, '--discriminator-contrastive-weight', str(weight),
                     '--discriminator-contrastive-radius', str(radius))
    args.discriminator_heldout_replays = ['heldout-match']
    config = copy.deepcopy(vars(args))
    if old is None:
        for key in ('discriminator_contrastive', 'discriminator_contrastive_weight', 'discriminator_contrastive_radius'):
            config.pop(key)
    else:
        config.update(discriminator_contrastive=old, discriminator_contrastive_weight=1., discriminator_contrastive_radius=40.)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    payload = {'format': 'goddard-leanrl-v1', 'algorithm': 'gaifo', 'config': config,
               'agent': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'torch_rng': torch.get_rng_state(), 'cuda_rng': [], 'numpy_rng': np.random.get_state(),
               'python_rng': random.getstate()}
    path = tmp_path / 'checkpoint.pt'
    torch.save(payload, path)
    if warn:
        with pytest.warns(UserWarning, match='contrastive configuration changed'):
            load_checkpoint(path, 'gaifo', args, model, optimizer)
    else:
        load_checkpoint(path, 'gaifo', args, model, optimizer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
@pytest.mark.parametrize('bf16', [False, True])
def test_cuda_contrastive_training_is_finite(bf16):
    if bf16 and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 is unavailable')
    args = arguments('--factorize', '--discriminator-contrastive', '--discriminator-relative',
                     '--discriminator-update-microbatch-size', '2')
    args.bf16 = bf16
    model = AbsorbingDiscriminator(args, 2).cuda()
    stats = {}
    loss = update.train_discriminator_batch(model, torch.optim.Adam(model.parameters()),
                                             windows(4).cuda(), windows(4).cuda(), args, metrics=stats)
    assert torch.isfinite(loss) and torch.isfinite(stats['contrastive_loss'])
    assert stats['contrastive_generated_fraction'] == 1
    assert stats['contrastive_expert_fraction'] == 1
