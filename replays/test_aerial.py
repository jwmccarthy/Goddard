import argparse
import hashlib
import json
from collections import Counter

import numpy as np
import pytest
import torch

from imitation.dataset import ExpertDataset
from replays.aerial import (
    FORMAT,
    HISTORY,
    annotate,
    diverse_choices,
    quality_mask,
    runs,
    select_match,
    valid_endings,
)
from replays.build_aerial import balanced_select, pack_dataset, quotas, save_arrays
from replays.corpus import write_json
from replays.layout import team_observation_size, team_replay_row_size
from replays.resets import ReplayResetDataset
from training.arguments import add_gaifo_arguments, add_ppo_arguments, resolve_gaifo_defaults


def scene_rows(n=240):
    rows = np.zeros((n, team_replay_row_size(1)), np.float32)
    rows[:, 2] = 600 / 2076
    for start, direction in ((9, 1), (30, -1)):
        rows[:, start + 1] = direction * 400 / 6000
        rows[:, start + 2] = 17 / 2076
        rows[:, start + 9] = 1
        rows[:, start + 14] = 1
        rows[:, start + 15] = .5
        rows[:, start + 16] = 1
    rows[:, 9 + 3] = 500 / 2300
    return rows


def test_phase_annotations_retain_leadin_takeoff_execution_recovery_and_declines():
    rows = scene_rows()
    rows[60:120, 11] = (17 + 600 * np.sin(np.linspace(0, np.pi, 60))) / 2076
    rows[60:120, 25] = 0
    rows[90, -5] = 1
    labels, events = annotate(rows, np.zeros(len(rows), bool))
    assert len(events) == 1
    assert labels['phase'][59] == 1
    assert labels['phase'][60] == 2
    assert labels['phase'][85] == 3
    assert labels['phase'][119] == 4
    assert labels['decision'][200] == 2  # Same elevated ball, expert remains grounded.
    assert events[0]['touch_observed']
    assert events[0]['origin'] == 'floor'


def test_invalid_history_never_crosses_a_quality_break_or_invents_takeoff():
    rows = scene_rows(120)
    rows[:, 11] = 500 / 2076
    rows[:, 25] = 0
    bad = np.zeros(len(rows), bool)
    bad[55] = True
    labels, events = annotate(rows, bad)
    assert not (labels['phase'] == 2).any()
    ends = valid_endings(rows, bad, HISTORY)
    assert not np.isin(np.arange(55, 55 + HISTORY), ends).any()
    assert events[0]['origin'] == 'segment_airborne'
    assert 1 in ends  # Padding exists only at the actual segment start.


def test_terminal_row_is_not_live_and_does_not_make_reset_data():
    rows = scene_rows(120)
    bad = np.zeros(len(rows), bool)
    assert len(rows) - 1 not in valid_endings(rows, bad, goal=True)
    assert len(rows) - 2 in valid_endings(rows, bad, goal=True)


def test_quality_filter_excludes_prolonged_stationary_away_not_brief_positioning():
    rows = scene_rows(400)
    rows[:, 0] = 3000 / 4108
    rows[:, 12:15] = 0
    bad, count = quality_mask(rows)
    assert bad.all() and count == len(rows)
    bad, _ = quality_mask(rows[:120])
    assert not bad.any()
    assert runs(np.array([False, True, True, False, True])) == [(1, 3), (4, 5)]


def test_match_selection_has_exact_counterpart_internal_states_and_no_phase_padding():
    first = scene_rows(180)
    first[:, 0] = np.arange(len(first)) / 10000
    second = first.copy()
    second[:, 9:30], second[:, 30:51] = first[:, 30:51], first[:, 9:30]
    for start in (0, 9, 30):
        for field in ((0, 3, 6) if start == 0 else (0, 3, 6, 9, 12)):
            second[:, start + field:start + field + 2] *= -1
    internal_start = team_observation_size(1)
    first[:, internal_start + 1], second[:, internal_start + 1] = .25, .75
    source = {'match_key': 'physical-match', 'players': [{'id': {'platform': 'steam', 'id': 'one'}},
                                                       {'id': {'platform': 'steam', 'id': 'two'}}]}
    segments = [(first, {'period': 0, 'goal_terminal': True, 'goal_outcome': 'scored', 'online_id': 'one', 'ego_id': '1'}),
                (second, {'period': 0, 'goal_terminal': True, 'goal_outcome': 'conceded', 'online_id': 'two', 'ego_id': '2'})]
    payload, _ = select_match(segments, source, 42)
    again, _ = select_match(segments, source, 42)
    for key in payload:
        np.testing.assert_equal(payload[key], again[key])
    assert set(payload['component']) >= {0, 1, 5}
    assert len(payload['reset_scenes']) > 0
    for index, player in enumerate(payload['reset_players']):
        np.testing.assert_allclose(payload['reset_internal'][index, :, 1], [.25, .75] if player.endswith('one') else [.75, .25])
        assert payload['reset_end'][index] < 179
    for index, end in enumerate(payload['end']):
        length = int(payload['lengths'][index])
        original = first if payload['players'][index].endswith('one') else second
        np.testing.assert_array_equal(payload['windows'][index, -length:], original[end - length + 1:end + 1, :51])


def test_event_round_robin_and_global_caps_do_not_silently_redistribute():
    items = [{'player': 'a', 'period': 0, 'role': 0, 'event': event, 'end': end}
             for event, ends in ((1, range(100)), (2, range(200, 203))) for end in ends]
    selected = diverse_choices(items, 2, np.random.default_rng(1))
    assert {item['event'] for item in selected} == {1, 2}
    with pytest.raises(ValueError, match='underfilled'):
        balanced_select({0: [{'player': 'a', 'pair': ('a', 'b'), 'period': 0, 'end': i} for i in range(20)]},
                        5, 1, Counter(), Counter(), set(), 2, 100, 100)
    assert quotas(100) == [40, 20, 15, 15, 10]
    with pytest.raises(ValueError, match='exact mixture'):
        quotas(99)


def make_candidates(root, history=22, n_sources=20, reset_profile="setup", reset_selection="events"):
    (root / 'candidates').mkdir(parents=True)
    sources = []
    for source_index in range(n_sources):
        source = {'id': f'id-{source_index}', 'match_key': f'match-{source_index}',
                  'sha256': f'{source_index:064x}', 'completion_verified': True,
                  'players': [{'id': {'platform': 'steam', 'id': f'{source_index}-a'}},
                              {'id': {'platform': 'steam', 'id': f'{source_index}-b'}}],
                  'source': 'pro_playoffs' if source_index % 2 else 'ssl_duels',
                  'split': 'heldout' if source_index % 5 == 0 else 'train'}
        sources.append(source)
        components = np.repeat(np.arange(5), 12).tolist() + [5, 5]
        n = len(components)
        windows = np.repeat(scene_rows(history)[None, :, :51], n, axis=0)
        windows[:, :, 0] = np.asarray(components)[:, None]
        windows[:, :, 3] = np.arange(history)
        reset_components = np.repeat(range(5) if reset_profile in ('midair', 'balanced') else [0, 1, 2, 4],
                                     32 if reset_selection == 'diverse' else 12)
        payload = {'windows': windows, 'lengths': np.full(n, history, np.int16), 'component': np.array(components),
                   'players': np.array([f'steam:{source_index}-a' if i % 2 else f'steam:{source_index}-b' for i in range(n)]),
                   'end': np.arange(n) * history, 'period': np.zeros(n, np.int32), 'event': np.arange(n),
                   'role': np.zeros(n, np.int8), 'origin': np.ones(n, np.int8), 'decision': np.ones(n, np.int8),
                   'touch': np.zeros(n, np.int8), 'ego_ids': np.repeat('1', n),
                   'reset_scenes': scene_rows(len(reset_components))[:, :51],
                   'reset_internal': np.zeros((len(reset_components), 2, 19), np.float32),
                   'reset_component': reset_components, 'reset_end': np.arange(len(reset_components)) * history,
                   'reset_period': np.zeros(len(reset_components), np.int32),
                   'reset_players': np.repeat(f'steam:{source_index}-a', len(reset_components))}
        if reset_profile in ('midair', 'balanced'):
            airborne = np.isin(reset_components, (2, 3, 4))
            payload['reset_scenes'][airborne, 11] = 500 / 2076
            payload['reset_scenes'][airborne, 25] = 0
        if reset_selection == 'diverse':
            from replays.reset_diversity import reset_bins, reset_signature

            scene = payload['reset_scenes']
            scene[:, 9] = ((np.arange(len(scene)) % 8) * 256 - 1024 + source_index * 2) / 4108
            scene[:, 10] = (source_index * 256 - 2500) / 6000
            scene[:, 0] = ((source_index + 1) * 11 + np.arange(len(scene)) * 10) / 4108
            spatial, context = reset_bins(scene)
            payload.update(reset_spatial=spatial, reset_context=context,
                reset_event=np.arange(len(scene), dtype=np.int32) // 6,
                reset_role=np.zeros(len(scene), np.int32), reset_origin=np.ones(len(scene), np.int32),
                reset_signature=np.asarray([reset_signature(s, t) for s, t in zip(scene, payload['reset_internal'])], dtype='S32'))
        save_arrays(root / 'candidates' / f"{source['id']}.npz", **payload)
        write_json(root / 'candidates' / f"{source['id']}.json", {'valid_live_endings': 10000, 'valid_goals': 10, 'afk_rows_excluded': 0})
    write_json(root / 'parsed_sources.json', {'records': sources, 'reset_profile': reset_profile, 'reset_selection': reset_selection})


def test_packed_reader_honors_selected_endings_mixtures_splits_context_and_memory(tmp_path):
    make_candidates(tmp_path)
    report = pack_dataset(tmp_path, row_cap=10000, reset_count=200)
    assert report['retained_rows'] <= 10000
    assert report['scene_tensor_bytes'] == report['scene_rows'] * 204
    directory = tmp_path / 'dataset'
    expert = ExpertDataset(directory, 1, None, 'cpu', window_length=4, decision_stride=3, terminal_aware=True)
    assert expert.train_replay_keys.isdisjoint(expert.heldout_replay_keys)
    samples = expert.sample(20000)
    live = samples[..., -1].sum(-1) == 0
    frequencies = torch.bincount(samples[live, -1, 0].long(), minlength=5).float() / live.sum()
    torch.testing.assert_close(frequencies, torch.tensor([.4, .2, .15, .15, .1]), atol=.015, rtol=0)
    torch.testing.assert_close(samples[live][:, :, 3], torch.tensor([12., 15., 18., 21.]).expand(live.sum(), -1))
    assert len(expert.starts) < len(expert.frames) / 5  # Context is not itself sampleable.
    assert (expert.train_kinds == 1).any() and (expert.train_kinds == 2).any()
    gaifo = ExpertDataset(directory, 1, None, 'cpu', window_length=8, decision_stride=3)
    assert not gaifo.train_kinds.any()
    resets = ReplayResetDataset(directory, 1, 'cpu')
    assert len(resets.frames) == 200
    with pytest.raises(ValueError, match='exceeds retained'):
        ExpertDataset(directory, 1, None, 'cpu', window_length=9, decision_stride=3)
    with pytest.raises(ValueError, match='truncate stratified'):
        ExpertDataset(directory, 1, 100, 'cpu')
    with pytest.raises(ValueError, match='fewer preassigned'):
        ExpertDataset(directory, 1, None, 'cpu', heldout_size=1, heldout_replays=10)


def test_selected_terminal_and_loop_geometry_and_sampling_rng_are_preserved(tmp_path):
    make_candidates(tmp_path)
    pack_dataset(tmp_path, row_cap=10000, reset_count=200)
    directory = tmp_path / 'dataset'
    a = ExpertDataset(directory, 1, None, 'cpu', window_length=4, terminal_aware=True, seed=123)
    b = ExpertDataset(directory, 1, None, 'cpu', window_length=4, terminal_aware=True, seed=123)
    a.sample_heldout(100)
    torch.testing.assert_close(a.sample(100), b.sample(100), equal_nan=True)
    for kind in (1, 2):
        pool = torch.nonzero(a.train_kinds == kind).flatten()
        windows = a._sample(a.train_starts[pool], a.train_lengths[pool], a.train_kinds[pool], 20, a.train_generator)
        if kind == 1:
            assert (windows[:, -1, -1] == 1).all() and (windows[:, :-1, -1] == 0).all()
            assert (windows[:, -1, :-1] == 0).all()
        else:
            assert (windows[..., -1] == 1).all() and (windows[..., :-1] == 0).all()


def test_pack_is_reproducible_and_readers_fail_closed_on_bad_metadata(tmp_path):
    for name in ('a', 'b'):
        root = tmp_path / name
        make_candidates(root)
        pack_dataset(root, row_cap=10000, reset_count=200)
    for path in (tmp_path / 'a/dataset').glob('*.npz'):
        with np.load(path) as left, np.load(tmp_path / 'b/dataset' / path.name) as right:
            for key in left:
                np.testing.assert_equal(left[key], right[key])
    directory = tmp_path / 'a/dataset'
    metadata = json.loads((directory / 'selection.json').read_text())
    assert metadata['format'] == FORMAT
    metadata['retained_rows'] = metadata['row_cap'] + 1
    write_json(directory / 'selection.json', metadata)
    with pytest.raises(ValueError, match='budget exceeded'):
        ExpertDataset(directory, 1, None, 'cpu')


@pytest.mark.parametrize('flags,message', [(['--frameskip', '8'], 'matching frameskip'),
                                         (['--team-size', '2'], 'matching frameskip'),
                                         (['--discriminator-window-length', '24'], 'enough original')])
def test_training_rejects_selected_timebase_or_context_mismatch_before_building_env(tmp_path, flags, message):
    write_json(tmp_path / 'selection.json', {'frameskip': 4, 'team_size': 1, 'history': 22})
    parser = argparse.ArgumentParser()
    add_ppo_arguments(parser, gaifo=True)
    add_gaifo_arguments(parser)
    args = parser.parse_args(['--replay-dir', str(tmp_path), '--discriminator-window-length', '4', *flags])
    with pytest.raises(ValueError, match=message):
        resolve_gaifo_defaults(args)


def paired_aerial_segments():
    first = scene_rows(240)
    first[60:150, 11] = (17 + 600 * np.sin(np.linspace(0, np.pi, 90))) / 2076
    first[60:150, 25] = 0
    first[60:150, 27] = 1
    first[100, -5] = 1
    internal_start = team_observation_size(1)
    first[:, internal_start] = first[:, 25]
    first[:, internal_start + 8] = first[:, 27]
    first[:, internal_start + 1] = .25
    second = first.copy()
    second[:, 9:30], second[:, 30:51] = first[:, 30:51], first[:, 9:30]
    for start in (0, 9, 30):
        for field in ((0, 3, 6) if start == 0 else (0, 3, 6, 9, 12)):
            second[:, start + field:start + field + 2] *= -1
    second[:, internal_start] = second[:, 25]
    second[:, internal_start + 8] = second[:, 27]
    second[:, internal_start + 1] = .75
    source = {'match_key': 'aerial-match', 'players': [{'id': {'platform': 'steam', 'id': 'one'}},
                                                    {'id': {'platform': 'steam', 'id': 'two'}}]}
    segments = [(first, {'period': 0, 'goal_terminal': True, 'goal_outcome': 'scored', 'online_id': 'one', 'ego_id': '1'}),
                (second, {'period': 0, 'goal_terminal': True, 'goal_outcome': 'conceded', 'online_id': 'two', 'ego_id': '2'})]
    return segments, source


def test_midair_candidates_add_execution_without_changing_expert_windows():
    segments, source = paired_aerial_segments()
    original, _ = select_match(segments, source, 42)
    updated, _ = select_match(segments, source, 42, reset_profile='midair')
    for key in original:
        if not key.startswith('reset_'):
            np.testing.assert_equal(updated[key], original[key])
    components = updated['reset_component']
    assert (components == 3).any()
    scenes = updated['reset_scenes']
    targeted = np.isin(components, (2, 3, 4))
    assert (scenes[targeted, 11] * 2076 > 182.5).all()
    assert (scenes[targeted, 25] == 0).all()
    assert (scenes[components == 1, 25] == 1).all()
    assert (updated['reset_internal'][components == 3, 0, 8] == 1).all()  # Do not discard active flips.
    for index, player in enumerate(updated['reset_players']):
        end = updated['reset_end'][index]
        assert end != 100 and end != 239  # No active contact or goal terminal.
        raw = segments[0 if player.endswith('one') else 1][0]
        np.testing.assert_array_equal(scenes[index], raw[end, :51])
        np.testing.assert_allclose(updated['reset_internal'][index, :, 1], [.25, .75] if player.endswith('one') else [.75, .25])


def test_midair_reset_candidates_do_not_spawn_inside_the_ball():
    segments, source = paired_aerial_segments()
    first, second = segments[0][0], segments[1][0]
    first[70:107, 9:12] = first[70:107, :3]
    # Keep the counterpart aligned so this tests penetration, not missing state.
    second[70:107, 30:33] = first[70:107, 9:12] * (-1, -1, 1)
    payload, _ = select_match(segments, source, 42, reset_profile='midair')
    assert not (payload['reset_component'] == 3).any()


def test_midair_bank_has_exact_separate_mixture_and_preserves_budget_and_splits(tmp_path):
    make_candidates(tmp_path, reset_profile='midair')
    report = pack_dataset(tmp_path, row_cap=10000, reset_count=200, reset_profile='midair')
    assert report['reset_mixture'] == {'representative': 40, 'setup': 40, 'takeoff': 20, 'execution': 80, 'recovery': 20}
    assert report['retained_rows'] <= 10000
    directory = tmp_path / 'dataset'
    with np.load(directory / 'resets.npz') as data:
        airborne = (data['frames'][:, 11] * 2076 > 182.5) & (data['frames'][:, 25] < .5)
        assert airborne.mean() >= .6
        metadata = json.loads((directory / 'selection.json').read_text())
        assert all(metadata['match_splits'][str(key)] == 'train' for key in data['matches'])
    assert metadata['reset_profile'] == 'midair'
    assert pack_dataset(tmp_path, row_cap=10000, reset_count=200, reset_profile='midair') == json.loads(json.dumps(report))
    with pytest.raises(ValueError, match='configuration changed'):
        pack_dataset(tmp_path, row_cap=10000, reset_count=200)


def test_midair_profile_cannot_silently_reuse_setup_only_candidates(tmp_path):
    make_candidates(tmp_path)
    with pytest.raises(ValueError, match='candidate reset profile differs'):
        pack_dataset(tmp_path, row_cap=10000, reset_count=200, reset_profile='midair')


def test_reset_variant_preserves_expert_shards_sampling_and_original_reset_bank(tmp_path):
    original, variant = tmp_path / 'original', tmp_path / 'variant'
    make_candidates(original)
    before = pack_dataset(original, row_cap=10000, reset_count=200)
    base = original / 'dataset'
    old_resets = (base / 'resets.npz').read_bytes()
    make_candidates(variant, reset_profile='midair')
    # Different reconstructed expert candidates must not change frozen experts.
    candidate = variant / 'candidates/id-1.npz'
    with np.load(candidate) as data:
        arrays = {key: data[key] for key in data}
    arrays['windows'][:, :, 0] += 99
    save_arrays(candidate, **arrays)
    report = pack_dataset(variant, row_cap=10000, reset_count=200, reset_profile='midair', expert_dataset=base)
    target = variant / 'dataset'
    assert report['expert_shards_unchanged']
    assert report['reset_mixture']['execution'] == 80
    assert report['live_endings'] == before['live_endings']
    assert report['retained_rows'] == before['retained_rows']
    metadata = json.loads((base / 'selection.json').read_text())
    updated = json.loads((target / 'selection.json').read_text())
    for key in ('shards', 'scene_rows', 'mixture', 'match_splits', 'native_terminal_probability'):
        assert metadata[key] == updated[key]
    for shard in metadata['shards']:
        assert (base / shard['file']).read_bytes() == (target / shard['file']).read_bytes()
        assert (base / shard['file']).stat().st_ino != (target / shard['file']).stat().st_ino
    assert (base / 'resets.npz').read_bytes() == old_resets
    assert (target / 'resets.npz').read_bytes() != old_resets
    left = ExpertDataset(base, 1, None, 'cpu', seed=11, window_length=4, decision_stride=3, terminal_aware=True)
    right = ExpertDataset(target, 1, None, 'cpu', seed=11, window_length=4, decision_stride=3, terminal_aware=True)
    torch.testing.assert_close(left.sample(1000), right.sample(1000), equal_nan=True, atol=0, rtol=0)
    assert left.heldout_replay_keys == right.heldout_replay_keys
    assert updated['expert_selection_sha256'] == hashlib.sha256((base / 'selection.json').read_bytes()).hexdigest()


@pytest.mark.parametrize('case,message', [('split', 'identity or split'), ('sha', 'identity or split'),
                                        ('budget', 'exceeds'), ('in_place', 'separate output')])
def test_reset_variant_rejects_source_split_budget_and_in_place_changes(tmp_path, case, message):
    original, variant = tmp_path / 'original', tmp_path / 'variant'
    make_candidates(original)
    pack_dataset(original, row_cap=10000, reset_count=200)
    make_candidates(variant, reset_profile='midair')
    parsed_path = variant / 'parsed_sources.json'
    parsed = json.loads(parsed_path.read_text())
    if case == 'split':
        parsed['records'][0]['split'] = 'train'
    elif case == 'sha':
        parsed['records'][0]['sha256'] = 'changed'
    write_json(parsed_path, parsed)
    with pytest.raises(ValueError, match=message):
        pack_dataset(original if case == 'in_place' else variant, row_cap=9800 if case == 'budget' else 10000,
                     reset_count=200, reset_profile='midair', expert_dataset=original / 'dataset')


def test_diverse_balanced_variant_preserves_experts_and_has_only_one_third_airborne(tmp_path):
    original, variant = tmp_path / 'original', tmp_path / 'variant'
    make_candidates(original)
    pack_dataset(original, row_cap=10000, reset_count=200)
    make_candidates(variant, reset_profile='balanced', reset_selection='diverse')
    report = pack_dataset(variant, row_cap=10200, reset_count=500, reset_profile='balanced',
                          reset_selection='diverse', expert_dataset=original / 'dataset')
    assert report['expert_shards_unchanged']
    with np.load(variant / 'dataset/resets.npz') as data:
        high = (data['frames'][:, 25] < .5) & (data['frames'][:, 11] * 2076 > 182.5)
        assert high.sum() == 167
        assert (data['frames'][~high, 25] > .5).all()
        assert len(set(data['signatures'].tolist())) == 500
    assert report['reset_diversity']['unique_exact_states'] == 500
    assert report['retained_rows'] <= 10200
