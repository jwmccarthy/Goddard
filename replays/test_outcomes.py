from dataclasses import asdict

import numpy as np
import pytest

from replays.layout import team_replay_row_size
from replays.outcomes import OutcomeConfig, annotate_outcomes, contact_clusters, goal_threat


def world_rows(n=210):
    rows = np.zeros((n, team_replay_row_size(1)), np.float32)
    rows[:, 2] = 150 / 2076
    for start in (9, 30):
        rows[:, start + 2] = 17 / 2076
        rows[:, start + 9] = 1
        rows[:, start + 14] = 1
        rows[:, start + 15] = .5
        rows[:, start + 16] = 1
    rows[:, 30] = 3000 / 4108
    return rows


def goal_rows(n=150, contact=110):
    rows = world_rows(n)
    rows[:, 1] = 2500 / 6000
    rows[contact:, 1] = np.linspace(2500, 5230, n - contact) / 6000
    rows[:, 10] = 2380 / 6000
    rows[:, 4] = 2000 / 6000
    rows[contact, -5] = 1
    return rows, {"goal_terminal": True, "goal_outcome": "scored"}


def carry_rows(n=210, lo=50, hi=130):
    rows = world_rows(n)
    rows[:, 10] = np.arange(n) * (600 / 30) / 6000
    rows[:, 12 + 1] = 600 / 2300
    rows[lo:hi, 11] = 500 / 2076
    rows[lo:hi, 25] = 0
    rows[:, :3] = rows[:, 9:12]
    rows[:, 2] += 120 / 2076
    rows[:, 4] = 600 / 6000
    for index in (lo + 10, lo + 40, lo + 65):
        if index < min(hi, n):
            rows[index, -5] = 1
    return rows, {"goal_terminal": False, "goal_outcome": None}


def save_rows(n=150, contact=30):
    rows = world_rows(n)
    rows[:, 1] = (-2000 - 2000 * np.arange(n) / 30) / 6000
    rows[:, 2] = 350 / 2076
    rows[:, 4] = -2000 / 6000
    rows[contact + 1:, 1] = (-4000 + 1200 * np.arange(1, n - contact) / 30) / 6000
    rows[contact + 1:, 4] = 1200 / 6000
    rows[:, 10] = -4120 / 6000
    rows[20:45, 11] = 200 / 2076
    rows[20:45, 25] = 0
    rows[contact, -5] = 1
    return rows, {"goal_terminal": False, "goal_outcome": None}


def successes(rows, metadata, config=None):
    return [event for event in annotate_outcomes(rows, metadata, config) if event["status"] == "success"]


def test_contact_clustering_does_not_count_each_flagged_frame_as_a_touch():
    mask = np.zeros(30, bool)
    mask[2:5] = True
    mask[6] = True
    mask[20:23] = True
    assert contact_clusters(mask) == [(2, 7), (20, 23)]
    assert contact_clusters(np.zeros(0, bool)) == []


@pytest.mark.parametrize("kwargs", [{"goal_horizon": 0}, {"save_horizon": float("nan")},
                                     {"carry_seconds": -1}, {"setup_seconds": -1},
                                     {"fps": 10}, {"prediction_margin": 300}])
def test_outcome_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        OutcomeConfig(**kwargs)
    assert asdict(OutcomeConfig())["fps"] == 30


def test_direct_goal_prediction_requires_a_clear_goal_aperture_without_bounces():
    assert 0 < goal_threat([0, -4000, 350], [0, -2000, 0]) < 1
    assert goal_threat([0, -4000, 350], [0, 2000, 0]) is None
    assert goal_threat([1300, -4000, 350], [0, -2000, 0]) is None
    assert goal_threat([0, -4000, 1000], [0, -2000, 0]) is None
    assert goal_threat([0, -1000, 100], [0, -1000, -400]) is None
    assert goal_threat([0, -5000, 150], [0, -500, 0]) is None  # Entry alone is not a whole-ball goal.
    assert goal_threat([0, -5300, 350], [0, -2000, 0]) is None
    with pytest.raises(ValueError, match="inputs"):
        goal_threat([0, -4000, np.nan], [0, -2000, 0])


def test_goal_credit_is_local_and_preserves_setup_execution_and_followthrough():
    rows, metadata = goal_rows()
    before = rows.copy()
    events = successes(rows, metadata)
    assert len(events) == 1 and events[0]["kind"] == "goal_touch"
    event = events[0]
    assert event["start"] < event["contact_start"] < event["end"] == len(rows) - 1
    assert event["phases"]["setup"] == [65, 110]
    assert event["phases"]["execution"] == [110, 117]
    assert event["terminal"] and event["verification_end"] == len(rows) - 1
    np.testing.assert_array_equal(rows, before)
    assert annotate_outcomes(rows, metadata) == annotate_outcomes(rows, metadata)
    # A period ending in our goal does NOT qualify its much earlier contact.
    stale, metadata = goal_rows(contact=10)
    assert not successes(stale, metadata)
    assert annotate_outcomes(stale, metadata)[0]["reason"] == "goal_outside_credit_horizon"


@pytest.mark.parametrize("case", ["opponent_touch", "conceded", "quality_gap", "bad_contact", "unknown_outcome"])
def test_goal_filter_excludes_ambiguous_failed_and_unreliable_events(case):
    rows, metadata = goal_rows()
    if case == "opponent_touch":
        rows[130, -4] = 1
    elif case == "conceded":
        metadata["goal_outcome"] = "conceded"
    elif case == "quality_gap":
        rows[130, -1] = 1
    elif case == "bad_contact":
        rows[110, 9] = 2000 / 4108
    else:
        metadata = {"goal_terminal": False, "goal_outcome": None}
    assert not successes(rows, metadata)


def test_replaced_goal_physics_is_never_used_as_motion_or_quality_evidence():
    rows, metadata = goal_rows()
    rows[-1] = np.nan
    events = successes(rows, metadata)
    assert events[0]["end"] == len(rows) - 1
    # Missing the metadata itself is not repaired using guessed goal geometry.
    with pytest.raises(ValueError, match="known"):
        annotate_outcomes(rows, {"goal_terminal": True, "goal_outcome": None})


def test_contact_cluster_spanning_takeoff_keeps_chronological_phase_bounds():
    rows, metadata = goal_rows()
    rows[110:113, -5] = 1
    rows[111:124, 25] = 0
    rows[111:124, 11] = 300 / 2076
    event = successes(rows, metadata)[0]
    assert event["kind"] == "goal_touch"
    assert event["contact_start"] == 110
    assert event["phases"]["takeoff"] == [111, 111]
    cursor = event["start"]
    for lo, hi in event["phases"].values():
        assert lo == cursor and hi >= lo
        cursor = hi
    assert cursor == event["end"]


def test_air_dribble_requires_sustained_control_and_a_verified_complete_finish():
    rows, metadata = carry_rows()
    events = successes(rows, metadata)
    assert len(events) == 1 and events[0]["kind"] == "air_dribble"
    event = events[0]
    assert event["evidence"]["finish"] == "landed_with_retained_possession"
    assert event["evidence"]["contact_episodes"] == 3
    assert event["start"] == 5 and event["end"] == 154
    assert all(end > start for start, end in event["phases"].values())


@pytest.mark.parametrize("case", ["lost_control", "one_contact", "censored", "takeoff_missing", "quality_gap", "contested"])
def test_air_dribble_filter_does_not_promote_long_flights_or_failed_finishes(case):
    rows, metadata = carry_rows()
    if case == "lost_control":
        rows[130:, 0] = 2000 / 4108
    elif case == "one_contact":
        rows[:, -5] = 0
        rows[60:63, -5] = 1
    elif case == "censored":
        rows = rows[:135]
    elif case == "takeoff_missing":
        rows = rows[50:]
    elif case == "quality_gap":
        rows[110, -2] = 1
    else:
        rows[115, -4] = 1
    assert not successes(rows, metadata)


def test_goal_finishing_air_dribble_is_one_skill_event_not_duplicate_goal_credit():
    rows, _ = carry_rows(n=150, hi=149)
    metadata = {"goal_terminal": True, "goal_outcome": "scored"}
    rows[110, -5] = 1
    events = annotate_outcomes(rows, metadata)
    accepted = [event for event in events if event["status"] == "success"]
    assert len(accepted) == 1 and accepted[0]["kind"] == "air_dribble"
    assert accepted[0]["terminal"] and accepted[0]["evidence"]["finish"] == "goal_touch"
    assert any(event["status"] == "superseded" for event in events)


def test_save_requires_a_pretouch_goal_threat_and_observed_clearance():
    rows, metadata = save_rows()
    accepted = successes(rows, metadata)
    assert len(accepted) == 1 and accepted[0]["kind"] == "save"
    assert accepted[0]["evidence"]["prediction"] == "no-bounce-drag-gravity-v1"
    assert accepted[0]["outcome_frame"] > accepted[0]["end"]  # Offline-only future assessment.
    assert not accepted[0]["terminal"]


def test_goal_finishing_save_keeps_its_terminal_without_duplicate_goal_events():
    rows, _ = save_rows(n=75)
    accepted = successes(rows, {"goal_terminal": True, "goal_outcome": "scored"})
    assert len(accepted) == 1 and accepted[0]["kind"] == "save"
    assert accepted[0]["terminal"] and accepted[0]["evidence"]["also_goal_touch"]


@pytest.mark.parametrize("case", ["wide", "not_cleared", "conceded", "truncated", "opponent", "quality_gap"])
def test_save_filter_rejects_nonthreats_failures_and_unresolved_outcomes(case):
    rows, metadata = save_rows()
    if case == "wide":
        rows[:, 0] = rows[:, 9] = 1500 / 4108
    elif case == "not_cleared":
        rows[31:, 4] = -2000 / 6000
    elif case == "conceded":
        rows = rows[:75]
        metadata = {"goal_terminal": True, "goal_outcome": "conceded"}
    elif case == "truncated":
        rows = rows[:75]
    elif case == "opponent":
        rows[70, -4] = 1
    else:
        rows[70, -1] = 1
    assert not successes(rows, metadata)


def test_compact_scenes_and_inconsistent_goal_metadata_are_rejected():
    with pytest.raises(ValueError, match="complete dense"):
        annotate_outcomes(np.zeros((20, 51)), {})
    with pytest.raises(ValueError, match="nonterminal"):
        annotate_outcomes(world_rows(), {"goal_terminal": False, "goal_outcome": "scored"})
