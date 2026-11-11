import copy
import json
from datetime import timedelta

import pytest

from replays.duel_ranks import (
    PLAYLIST_ROUTE,
    PROFILE_ROUTE,
    BlastRankClient,
    decode_router,
    history_evidence,
    profile_document,
    season_at,
    timestamp,
)
from replays.research_duels import investigate
from replays.test_collect_duels import CONFIG, listing

START = timestamp("2026-10-18T12:00:00Z")
END = START + timedelta(minutes=8)


def document():
    return {"source_url": "https://blast.tv/rl/user/steam/1/playlists?playlist=10",
            "fetched_at": "2026-11-08T00:00:00Z", "platform": "steam", "account_id": "1",
            "selected_playlist_id": 10, "selected_season_id": "season-24",
            "seasons": [{"id": "season-24", "startedAt": "2026-09-23T17:00:00Z", "endedAt": None},
                        {"id": "season-23", "startedAt": "2026-06-10T17:00:00Z",
                         "endedAt": "2026-09-23T17:00:00Z"}],
            "snapshots": [{"id": "before", "playlistId": 10, "tier": 22,
                           "placementMatchesPlayed": 10, "scrapedAt": "2026-10-18T11:00:00Z"},
                          {"id": "after", "playlistId": 10, "tier": 22,
                           "placementMatchesPlayed": 10, "scrapedAt": "2026-10-18T13:00:00Z"}]}


def router_html(value):
    flat = []

    def encode(value):
        index = len(flat)
        flat.append(None)
        if isinstance(value, dict):
            flat[index] = {f"_{encode(k)}": encode(v) for k, v in value.items()}
        elif isinstance(value, list):
            flat[index] = [encode(v) for v in value]
        else:
            flat[index] = value
        return index

    encode(value)
    return "<script>window.__reactRouterContext.streamController.enqueue(" + json.dumps(json.dumps(flat)) + ")</script>"


def page_html():
    doc = document()
    return router_html({"loaderData": {
        PROFILE_ROUTE: {"platform": "steam", "userId": "1", "lastScrapedAt": "2099-01-01T00:00:00Z"},
        PLAYLIST_ROUTE: {"selectedPlaylistId": 10, "selectedSeasonId": "season-24",
                         "seasons": doc["seasons"], "snapshots": doc["snapshots"],
                         "playlists": [{"playlistId": 11, "tier": 22}]},
        "root": {"env": {"irrelevant": "must-not-be-retained"}}}})


def test_decoder_uses_only_json_and_keeps_tagged_values_inert():
    html = "<script>window.__reactRouterContext.streamController.enqueue(" + json.dumps(json.dumps([["P", 1], "inert"])) + ")</script>"
    assert decode_router(router_html({"x": [1, False, None]})) == {"x": [1, False, None]}
    assert decode_router(html) == ["P", 1]
    with pytest.raises(ValueError):
        decode_router("<script>throw new Error('do not run scripts')</script>")


def test_profile_identity_playlist_and_season_are_bound_and_env_is_not_cached():
    html = page_html()
    value = profile_document(html, "steam", "1", "public-url", season_id="season-24")
    assert value["snapshots"][0]["scrapedAt"] == "2026-10-18T11:00:00Z"
    assert not value["playlists"]  # Doubles SSL is not duel evidence.
    assert "env" not in json.dumps(value) and "lastScrapedAt" not in json.dumps(value)
    with pytest.raises(ValueError, match="identity mismatch"):
        profile_document(html, "steam", "another-account", "public-url")
    with pytest.raises(ValueError, match="requested season"):
        profile_document(html, "steam", "1", "public-url", season_id="season-23")
    decoded = decode_router(html)
    decoded["loaderData"][PLAYLIST_ROUTE]["selectedPlaylistId"] = 11
    with pytest.raises(ValueError, match="ranked duels"):
        profile_document(router_html(decoded), "steam", "1", "public-url")


def test_nearby_placed_ssl_bracket_is_support_not_verified_match_rank():
    evidence = history_evidence(document(), START, END)
    assert evidence["status"] == "ssl_bracketed_not_match_verified"
    assert evidence["rank_verified_at_match"] is False
    assert evidence["before"]["id"] == "before" and evidence["after"]["id"] == "after"


@pytest.mark.parametrize("case,status", [("missing", "missing_bracket"), ("stale", "stale_bracket"),
                                       ("rank", "not_ssl_through_bracket"),
                                       ("peak", "not_ssl_through_bracket"),
                                       ("placements", "placements_unfinished"),
                                       ("wrong_playlist", "season_or_playlist_mismatch"),
                                       ("wrong_season", "season_or_playlist_mismatch")])
def test_rank_evidence_fails_closed(case, status):
    value = document()
    p = value["snapshots"][0]
    if case == "missing":
        value["snapshots"].pop()
    elif case == "stale":
        p["scrapedAt"] = "2026-10-01T11:00:00Z"
    elif case == "rank":
        p["tier"] = 21
    elif case == "peak":
        p.pop("tier")
        p["tierMax"] = p["bestTier"] = 22
    elif case == "placements":
        p["placementMatchesPlayed"] = 0
    elif case == "wrong_playlist":
        p["playlistId"] = 11
    elif case == "wrong_season":
        value["selected_season_id"] = "season-23"
    assert history_evidence(value, START, END)["status"] == status


def test_nearest_non_ssl_or_intervening_demotion_cannot_be_skipped():
    value = document()
    intervening = copy.deepcopy(value["snapshots"][0])
    intervening.update(tier=21, scrapedAt="2026-10-18T12:05:00Z")
    value["snapshots"].append(intervening)
    assert history_evidence(value, START, END)["status"] == "not_ssl_through_bracket"


def test_season_boundaries_and_timezones():
    seasons = document()["seasons"]
    assert season_at(seasons, timestamp("2026-09-23T16:59:59Z"))["id"] == "season-23"
    assert season_at(seasons, timestamp("2026-09-23T17:00:00Z"))["id"] == "season-24"
    assert timestamp("2026-10-18T14:00:00+02:00") == START
    with pytest.raises(ValueError, match="timezone-aware"):
        timestamp("2026-10-18T12:00:00")


def test_public_client_uses_separate_unauthenticated_session_and_validates_cache(tmp_path, monkeypatch):
    client = BlastRankClient(tmp_path, interval=0)
    calls = []

    def get(url, **kwargs):
        from types import SimpleNamespace
        calls.append((url, kwargs))
        return SimpleNamespace(text=page_html(), raise_for_status=lambda: None)

    monkeypatch.setattr(client.session, "get", get)
    assert "Authorization" not in client.session.headers
    assert client.profile("steam", "1")["account_id"] == "1"
    assert client.profile("steam", "1")["account_id"] == "1"
    assert len(calls) == 1 and calls[0][0].startswith("https://blast.tv/rl/user/steam/1/")
    path = next(tmp_path.glob("*.json"))
    cached = json.loads(path.read_text())
    cached["account_id"] = "another-account"
    path.write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="provenance mismatch"):
        client.profile("steam", "1")


def test_research_does_not_approve_even_two_bracketed_ssl_accounts(tmp_path):
    replay = listing()
    replay["date"] = "2026-10-18T12:00:00Z"
    page, report = tmp_path / "page.json", tmp_path / "report.json"
    page.write_text(json.dumps({"list": [replay, replay]}))

    class Client:
        def evidence(self, *args, **kwargs):
            return {"status": "ssl_bracketed_not_match_verified", "rank_verified_at_match": False}

    result = investigate([page], Client(), dict(CONFIG), report)
    assert result["examined"] == result["both_ssl_bracketed"] == 1
    assert result["approved_matches"] == 0
    assert result["matches"][0]["approved_for_training"] is False
    assert not (tmp_path / "manifest.jsonl").exists()
