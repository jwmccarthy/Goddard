import copy
from datetime import datetime, timezone

import pytest

from replays.duel_audit import CLOCK, FORFEIT, OVERTIME, clock_audit, header_audit, header_identity
from replays.test_collect_duels import CONFIG, listing


def decoded_replay():
    pair = [{"PlayerID": {"fields": {"Uid": str(i), "EpicAccountId": ""}},
             "Platform": {"value": "OnlinePlatform_Steam"}, "Team": i - 1,
             "Score": 500, "Shots": 5, "bBot": False} for i in (1, 2)]
    return {"properties": {"TeamSize": 1, "MatchType": "Online", "TotalSecondsPlayed": 303,
                           "Team0Score": 5, "Team1Score": 4, "PlayerStats": pair,
                           "MatchGUID": "test-match", "Goals": [{"frame": 3}]},
            "objects": [CLOCK, FORFEIT, OVERTIME],
            "network_frames": {"frames": [
                {"time": 0, "updated_actors": [{"object_id": 0, "attribute": {"Int": 300}}]},
                {"time": 300, "updated_actors": [{"object_id": 0, "attribute": {"Int": 0}}]},
                {"time": 301, "updated_actors": []},
                {"time": 310, "updated_actors": []}]}}


def test_missing_forfeit_flag_is_not_completion_without_clock_evidence():
    value = decoded_replay()
    report = header_audit(value, listing(), CONFIG)
    assert report["rejection"] is None and not report["completion_verified"]
    empty = copy.deepcopy(value)
    empty["network_frames"] = None
    assert clock_audit(empty, report)["rejection"] == "end_clock_unverified"
    assert clock_audit(value, report)["completion_verified"]


@pytest.mark.parametrize("case,reason", [("forfeit", "forfeit"), ("short", "short_play"),
                                       ("score", "score_conflict"), ("bot", "identity_unverified"),
                                       ("identity", "identity_unverified"), ("shots", "activity")])
def test_header_rejects_early_forfeits_and_mismatches(case, reason):
    value = decoded_replay()
    props = value["properties"]
    if case == "forfeit":
        props["bForfeit"] = True
    elif case == "short":
        props["TotalSecondsPlayed"] = 262.7
    elif case == "score":
        props["Team1Score"] = 3
    elif case == "bot":
        props["PlayerStats"][0]["bBot"] = True
    elif case == "identity":
        props["PlayerStats"][0]["PlayerID"]["fields"]["Uid"] = "another-account"
    elif case == "shots":
        props["PlayerStats"][0]["Shots"] = 0
    assert header_audit(value, listing(), CONFIG)["rejection"] == reason


def test_epic_identity_uses_account_id_not_zero_online_id_or_name():
    p = {"Name": "pro", "OnlineID": "0", "Platform": {"value": "OnlinePlatform_Epic"},
         "PlayerID": {"fields": {"Uid": "0", "EpicAccountId": "exact-account"}}}
    assert header_identity(p) == "epic:exact-account"
    p["PlayerID"]["fields"]["EpicAccountId"] = ""
    assert header_identity(p) is None


def test_positive_final_clock_and_network_forfeit_are_rejected():
    value = decoded_replay()
    report = header_audit(value, listing(), CONFIG)
    value["network_frames"]["frames"][1]["updated_actors"][0]["attribute"]["Int"] = 38
    assert clock_audit(value, report)["rejection"] == "end_clock_unverified"
    value["network_frames"]["frames"][-1]["updated_actors"] = [
        {"object_id": 1, "attribute": {"Boolean": True}}]
    assert clock_audit(value, report)["rejection"] == "forfeit"


def test_overtime_requires_raw_overtime_and_subsequent_winning_goal():
    value = decoded_replay()
    report = header_audit(value, listing(), CONFIG)
    assert clock_audit(value, report, overtime=True)["rejection"] == "overtime_finish_unverified"
    value["network_frames"]["frames"][2]["updated_actors"] = [
        {"object_id": 2, "attribute": {"Boolean": True}}]
    assert clock_audit(value, report, overtime=True)["completion_verified"]
    value["properties"]["Goals"] = [{"frame": 1}]
    assert clock_audit(value, report, overtime=True)["rejection"] == "overtime_finish_unverified"


def test_raw_epoch_must_independently_satisfy_date_bounds():
    value = decoded_replay()
    config = dict(CONFIG, replay_date_after="2026-01-01T00:00:00Z", replay_date_before="2026-11-09T00:00:00Z")
    props = value["properties"]
    assert header_audit(value, listing(), config)["rejection"] == "date_unverified"
    props["MatchStartEpoch"] = str(datetime(2025, 12, 31, tzinfo=timezone.utc).timestamp())
    assert header_audit(value, listing(), config)["rejection"] == "date_outside_range"
    props["MatchStartEpoch"] = str(datetime(2026, 10, 18, tzinfo=timezone.utc).timestamp())
    assert header_audit(value, listing(), config)["rejection"] is None


def test_a_clock_jump_without_regulation_time_is_not_a_complete_game():
    value = decoded_replay()
    report = header_audit(value, listing(), CONFIG)
    value["network_frames"]["frames"][1]["time"] = 1
    assert clock_audit(value, report)["rejection"] == "end_clock_unverified"


def test_overtime_clock_counts_up_and_repeated_true_flags_do_not_move_the_start():
    value = decoded_replay()
    report = header_audit(value, listing(), CONFIG)
    frames = value["network_frames"]["frames"]
    frames[2]["updated_actors"] = [{"object_id": 2, "attribute": {"Boolean": True}}]
    frames.append({"time": 315, "updated_actors": [
        {"object_id": 2, "attribute": {"Boolean": True}},
        {"object_id": 0, "attribute": {"Int": 6}}]})
    result = clock_audit(value, report, overtime=True)
    assert result["completion_verified"]
    assert result["final_clock"] == 6 and result["overtime_start"] == 301
