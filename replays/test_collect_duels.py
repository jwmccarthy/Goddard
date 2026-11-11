import copy
import json
from types import SimpleNamespace
from uuid import UUID

import pytest

from replays.api import API_URL, BallchasingClient
from replays.collect_duels import SSL, collect, detail_rejection, listing_rejection

CONFIG = {"max_goal_margin": 2, "min_play_seconds": 300, "min_player_score": 100, "min_player_shots": 1}


def listing(index=1, pro=True):
    def player(identity):
        return {"id": {"platform": "steam", "id": str(identity)}, "name": f"player-{identity}",
                "rank": {"id": SSL, "tier": 22}, "score": 500, "start_time": 0, "end_time": 400}
    blue, orange = player(1), player(2)
    blue["pro"] = pro
    return {"id": str(UUID(int=index)), "rocket_league_id": f"rocket-{index}",
            "playlist_id": "ranked-duels", "duration": 400,
            "blue": {"goals": 5, "players": [blue]}, "orange": {"goals": 4, "players": [orange]}}


def detail(replay, match_key=None):
    result = copy.deepcopy(replay)
    result.update(status="ok", team_size=1, play_duration=330,
                  match_guid=match_key or f"match-{replay['id']}")
    for color in ("blue", "orange"):
        player = result[color]["players"][0]
        player["stats"] = {"core": {"goals": result[color]["goals"], "score": 500, "shots": 5}}
    return result


def test_both_players_must_have_explicit_ssl_ranks_not_just_aggregate_rank():
    replay = listing()
    assert listing_rejection(replay, CONFIG) is None
    replay["min_rank"] = replay["max_rank"] = {"id": SSL}
    replay["orange"]["players"][0].pop("rank")
    assert listing_rejection(replay, CONFIG) == "unverified_ssl"
    replay["orange"]["players"][0]["rank"] = {"id": "grand-champion-3"}
    assert listing_rejection(replay, CONFIG) == "unverified_ssl"


def test_unknown_pro_rank_is_not_assumed_to_be_ssl():
    replay = listing(pro=True)
    replay["blue"]["players"][0]["rank"] = None
    assert listing_rejection(replay, CONFIG) == "unverified_ssl"


@pytest.mark.parametrize("goals,reason", [(5, "score_margin"), (1, "score_margin"), (3, None)])
def test_close_score_filter_rejects_ties_and_blowouts(goals, reason):
    replay = listing()
    replay["orange"]["goals"] = goals
    assert listing_rejection(replay, CONFIG) == reason


def test_listing_requires_duels_and_both_players_active():
    replay = listing()
    replay["playlist_id"] = "private"
    assert listing_rejection(replay, CONFIG) == "playlist"
    replay = listing()
    replay["orange"]["players"][0]["score"] = 0
    assert listing_rejection(replay, CONFIG) == "player_score"


@pytest.mark.parametrize("case,reason", [("duration", "play_duration"), ("missing_duration", "play_duration"),
                                      ("late_join", "participation"), ("early_leave", "participation"),
                                      ("no_shots", "player_activity"), ("rank", "rank_conflict"),
                                      ("forfeit", "forfeit"), ("score", "score_conflict")])
def test_detail_quality_checks(case, reason):
    replay = listing()
    value = detail(replay)
    player = value["orange"]["players"][0]
    if case == "duration":
        value["play_duration"] = 100
    elif case == "missing_duration":
        value.pop("play_duration")
    elif case == "late_join":
        player["start_time"] = 50
    elif case == "early_leave":
        player["end_time"] = 200
    elif case == "no_shots":
        player["stats"]["core"]["shots"] = 0
    elif case == "rank":
        player["rank"] = {"id": "grand-champion-3"}
    elif case == "forfeit":
        value["forfeit"] = True
    elif case == "score":
        player["stats"]["core"]["goals"] = 1
    assert detail_rejection(replay, value, CONFIG) == reason


def test_detail_can_omit_rank_but_cannot_substitute_another_player():
    replay = listing()
    value = detail(replay)
    value["orange"]["players"][0].pop("rank")
    assert detail_rejection(replay, value, CONFIG) is None
    value["orange"]["players"][0]["id"]["id"] = "different-player"
    assert detail_rejection(replay, value, CONFIG) == "player_mismatch"


def test_collection_prefers_pros_deduplicates_matches_and_resumes(tmp_path):
    pro, duplicate, ordinary, unknown = listing(1), listing(2), listing(3, pro=False), listing(4)
    unknown["orange"]["players"][0].pop("rank")

    class Client:
        def __init__(self):
            self.downloaded, self.calls = [], []

        def get_json(self, url, **params):
            self.calls.append((url, params))
            if url == f"{API_URL}/replays":
                return {"list": [unknown, pro, duplicate] if params.get("pro") else [pro, ordinary]}
            replay_id = url.rsplit("/", 1)[-1]
            replay = {r["id"]: r for r in (pro, duplicate, ordinary)}[replay_id]
            return detail(replay, match_key="same-match" if replay_id != ordinary["id"] else "other-match")

        def download_replay(self, replay_id, directory):
            self.downloaded.append(replay_id)
            (directory / f"{replay_id}.replay").write_bytes(b"test replay")
            return True

    args = SimpleNamespace(output=tmp_path, limit=2, max_pages=1, **CONFIG)
    client = Client()
    report = collect(args, client)
    assert report["downloaded"] == 2 and report["pro_replays"] == 1
    assert client.downloaded == [pro["id"], ordinary["id"]]
    assert report["counts_this_run"]["duplicate_match"] == 1
    records = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert all(not record["completion_verified"] for record in records)
    assert all(record["sha256"] and record["bytes"] > 0 for record in records)
    calls = len(client.calls)
    assert collect(args, client)["downloaded"] == 2
    assert len(client.calls) == calls
    args.max_goal_margin = 3
    with pytest.raises(ValueError, match="different filters"):
        collect(args, client)


def test_api_never_sends_credentials_to_another_origin():
    client = BallchasingClient("not-a-real-token")
    with pytest.raises(ValueError, match="another endpoint"):
        client.get_json("https://example.com/api/replays")
    with pytest.raises(ValueError):
        client.download_replay("../../other-path", None)


def test_replay_download_is_atomic(tmp_path, monkeypatch):
    client = BallchasingClient("not-a-real-token")
    response = SimpleNamespace(status_code=200, content=b"complete replay", raise_for_status=lambda: None)
    monkeypatch.setattr(client, "_wait_for_download_slot", lambda: None)
    monkeypatch.setattr(client, "_download_session", lambda: SimpleNamespace(get=lambda *args, **kwargs: response))
    replay_id = str(UUID(int=1))
    assert client.download_replay(replay_id, tmp_path)
    assert (tmp_path / f"{replay_id}.replay").read_bytes() == b"complete replay"
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("date,reason", [("2026-10-18T12:00:00Z", None),
                                      ("2026-01-01T00:00:00Z", "date_outside_range"),
                                      ("2026-11-09T00:00:00Z", "date_outside_range"),
                                      (None, "date_unverified"),
                                      ("2026-10-18T12:00:00", "date_unverified")])
def test_date_bounds_are_enforced_locally_and_exclusive(date, reason):
    replay = listing()
    replay["date"] = date
    config = dict(CONFIG, replay_date_after="2026-01-01T00:00:00Z", replay_date_before="2026-11-09T00:00:00Z")
    assert listing_rejection(replay, config) == reason


def test_broad_discovery_does_not_assume_missing_ranks_and_can_use_explicit_details(tmp_path):
    missing, verified = listing(1), listing(2)
    for r in (missing, verified):
        r["date"] = "2026-10-18T12:00:00Z"
        for p in (r["blue"]["players"][0], r["orange"]["players"][0]):
            p.pop("rank")

    class Client:
        def __init__(self):
            self.calls = []

        def get_json(self, url, **params):
            self.calls.append((url, params))
            if url == f"{API_URL}/replays":
                assert "min_rank" not in params and "max_rank" not in params
                assert params["replay_date_after"] == "2026-01-01T00:00:00Z"
                return {"list": [missing, verified]}
            value = detail(missing if url.endswith(missing["id"]) else verified)
            if url.endswith(verified["id"]):
                for color in ("blue", "orange"):
                    value[color]["players"][0]["rank"] = {"id": SSL, "tier": 22}
            return value

        def download_replay(self, replay_id, directory):
            assert replay_id == verified["id"]
            (directory / f"{replay_id}.replay").write_bytes(b"complete replay")
            return True

    args = SimpleNamespace(output=tmp_path, limit=1, max_pages=1, rank_discovery="broad",
                           replay_date_after="2026-01-01T00:00:00Z", **CONFIG)
    report = collect(args, Client())
    assert report["downloaded"] == 1 and report["counts_this_run"]["unverified_ssl"] == 1
    record = json.loads((tmp_path / "manifest.jsonl").read_text())
    assert {e["source"] for e in record["rank_evidence"]} == {"ballchasing-detail"}


def test_raw_audit_failure_cannot_enter_manifest_and_rejected_files_are_preserved(tmp_path, monkeypatch):
    replay = listing()

    class Client:
        def get_json(self, url, **params):
            return {"list": [replay]} if url == f"{API_URL}/replays" else detail(replay)

        def download_replay(self, replay_id, directory):
            (directory / f"{replay_id}.replay").write_bytes(b"raw forfeit")
            return True

    monkeypatch.setattr("replays.collect_duels.audit_replay", lambda *a: {
        "rejection": "forfeit", "completion_verified": False})
    args = SimpleNamespace(output=tmp_path, limit=1, max_pages=1, rrrocket=tmp_path / "decoder", **CONFIG)
    report = collect(args, Client())
    assert report["downloaded"] == 0 and report["counts_this_run"]["raw_forfeit"] == 2
    assert (tmp_path / "raw" / f"{replay['id']}.replay").exists()
    assert not (tmp_path / "manifest.jsonl").exists()


def test_raw_audit_cache_is_bound_to_file_hash_and_success_records_provenance(tmp_path, monkeypatch):
    from replays.duel_audit import FORMAT
    replay = listing()
    calls = []

    class Client:
        def get_json(self, url, **params):
            return {"list": [replay]} if url == f"{API_URL}/replays" else detail(replay)

        def download_replay(self, replay_id, directory):
            (directory / f"{replay_id}.replay").write_bytes(b"initial raw file")
            return True

    def audit(path, *args):
        calls.append(path.read_bytes())
        success = len(calls) == 2
        return {"format": FORMAT, "rejection": None if success else "forfeit",
                "completion_verified": success, "match_guid": f"MATCH-{replay['id']}"}

    monkeypatch.setattr("replays.collect_duels.audit_replay", audit)
    args = SimpleNamespace(output=tmp_path, limit=1, max_pages=1, rrrocket=tmp_path / "decoder", **CONFIG)
    assert collect(args, Client())["downloaded"] == 0
    assert len(calls) == 1  # Same rejection cache is reused in the second phase.
    (tmp_path / "raw" / f"{replay['id']}.replay").write_bytes(b"different raw file")
    assert collect(args, Client())["downloaded"] == 1
    assert len(calls) == 2
    record = json.loads((tmp_path / "manifest.jsonl").read_text())
    assert record["completion_verified"]
    assert record["raw_audit"]["format"] == FORMAT
    assert record["raw_audit"]["sha256"] == record["sha256"]
    assert len(record["rank_evidence"]) == 2
