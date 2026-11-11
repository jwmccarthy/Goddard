from collections import Counter

import numpy as np
import pytest

from replays.aerial import RESET_PROFILES
from replays.build_aerial import reset_quotas
from replays.reset_diversity import (
    balanced_spatial_select,
    different_scene,
    diverse_reset_choices,
    reset_bins,
    reset_signature,
)
from replays.test_aerial import paired_aerial_segments, scene_rows


def test_balanced_reset_quotas_round_only_the_airborne_third():
    counts = reset_quotas(500000, RESET_PROFILES['balanced'])
    assert counts == [133333, 200000, 33334, 100000, 33333]
    assert sum(counts) == 500000 and sum(counts[2:]) == 166667
    for size in (20, 100, 200, 100000):
        counts = reset_quotas(size, RESET_PROFILES['balanced'])
        assert sum(counts) == size
        assert abs(sum(counts[2:]) - size / 3) <= .5


def test_scene_novelty_rejects_tiny_edits_and_accepts_material_motion_changes():
    scene = scene_rows(1)[0, :51]
    close = scene.copy()
    close[9] += 20 / 4108
    assert not different_scene(scene, close)
    for kind in ('position', 'velocity', 'orientation', 'boost'):
        changed = scene.copy()
        if kind == 'position':
            changed[9] += 150 / 4108
        elif kind == 'velocity':
            changed[12] += 300 / 2300
        elif kind == 'orientation':
            changed[18:21] = [0, 1, 0]
        else:
            changed[24] += .2
        assert different_scene(scene, changed)


def items_from_rows(rows, events):
    spatial, context = reset_bins(rows[:, :51])
    return [{'rows': rows, 'end': index * 2, 'player': 'one', 'period': 0, 'role': 0,
             'event': event, 'spatial': tuple(spatial[index]), 'context': tuple(context[index])}
            for index, event in enumerate(events)]


def test_spatial_candidate_round_robin_does_not_favor_a_large_common_cell():
    rows = scene_rows(21)
    rows[:20, 0] = np.arange(20) * 150 / 4108
    rows[20, 9] = -3000 / 4108
    items = items_from_rows(rows, [1] * 20 + [2])
    for index, item in enumerate(items):
        item['rows'] = np.repeat(rows[index:index + 1], 41, axis=0)
    choices = diverse_reset_choices(items, 2, np.random.default_rng(4))
    assert {item['spatial'] for item in choices} == {item['spatial'] for item in items}
    assert {item['event'] for item in choices} == {1, 2}


def test_per_event_near_duplicates_and_long_flight_dominance_are_bounded():
    rows = scene_rows(100)
    rows[:, 9] = np.arange(100) * 10 / 4108
    spatial, context = reset_bins(rows[:, :51])
    items = [{'rows': rows, 'end': i, 'player': 'one', 'period': 0, 'role': 0, 'event': 1,
              'spatial': tuple(spatial[i]), 'context': tuple(context[i])} for i in range(100)]
    chosen = diverse_reset_choices(items, 100, np.random.default_rng(1))
    assert 1 < len(chosen) <= 8
    assert np.diff(sorted(item['end'] for item in chosen)).min() >= 2
    for i, left in enumerate(chosen):
        for right in chosen[i + 1:]:
            assert different_scene(rows[left['end'], :51], rows[right['end'], :51])


def test_exact_state_signatures_treat_signed_zero_as_identical_and_keep_internal_state():
    scene = np.zeros(51, np.float32)
    internal = np.zeros((2, 19), np.float32)
    other = -scene
    assert reset_signature(scene, internal) == reset_signature(other, internal)
    internal[1, 8] = 1
    assert reset_signature(scene, internal) != reset_signature(scene, np.zeros_like(internal))


def test_global_selection_balances_cells_matches_and_rejects_duplicate_states():
    pools = {}
    for match in range(5):
        pool = []
        for cell, count in enumerate((90, 4, 4, 4)):
            for sample in range(count):
                index = len(pool)
                pool.append({'player': f'p-{match}', 'pair': (f'p-{match}', 'opponent'), 'period': 0,
                             'end': index, 'spatial': (cell, 0, 0),
                             'signature': f'{match}-{index}'.encode()})
        pools[match] = pool
    pools[0][0]['signature'] = pools[1][0]['signature']
    players, pairs, used, signatures = Counter(), Counter(), set(), set()
    first = balanced_spatial_select(pools, 60, 7, players, pairs, used, 1000, 1000, 1000, signatures)
    assert len({item['signature'] for item in first}) == 60
    counts = Counter(item['spatial'] for item in first)
    assert max(counts.values()) - min(counts.values()) <= 1
    again = balanced_spatial_select(pools, 20, 8, players, pairs, used, 1000, 1000, 1000, signatures)
    assert len(signatures) == 80
    assert {item['signature'] for item in first}.isdisjoint(item['signature'] for item in again)
    with pytest.raises(ValueError, match='underfilled'):
        balanced_spatial_select(pools, 1000, 1, Counter(), Counter(), set(), 1, 1000, 1000, set())


def test_balanced_profile_keeps_ground_buckets_grounded_and_experts_unchanged():
    from replays.aerial import select_match

    segments, source = paired_aerial_segments()
    original, _ = select_match(segments, source, 42)
    diverse, _ = select_match(segments, source, 42, reset_profile='balanced', reset_selection='diverse')
    for key in original:
        if not key.startswith('reset_'):
            np.testing.assert_equal(original[key], diverse[key])
    components, scenes = diverse['reset_component'], diverse['reset_scenes']
    assert (scenes[components < 2, 25] > .5).all()
    assert (scenes[components >= 2, 25] < .5).all()
    assert (scenes[components >= 2, 11] * 2076 > 182.5).all()
    assert all(name in diverse for name in ('reset_spatial', 'reset_context', 'reset_event', 'reset_signature'))
    reset_only, _ = select_match(segments, source, 42, reset_profile='balanced', reset_selection='diverse', reset_only=True)
    assert len(reset_only['windows']) == 0
    for key in diverse:
        if key.startswith('reset_'):
            np.testing.assert_equal(reset_only[key], diverse[key])
