"""Public BLAST duel-rank history for investigation, never match-rank substitution.

SSL snapshots on either side of a match are supporting evidence, not proof of
the rank inside that interval. This module never labels a match rank verified.
"""

import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests

PROFILE_ROUTE = "sites/blast/routes/rl.user.$platform.$userId.($username)"
PLAYLIST_ROUTE = PROFILE_ROUTE + ".playlists"
FORMAT = "blast-duel-history-v1"
SNAPSHOT_FIELDS = ("id", "playlistId", "scrapedAt", "tier", "division", "skill",
                   "matchesPlayed", "lifetimeMatchesPlayed", "placementMatchesPlayed")


def timestamp(value):
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("expected a timezone-aware ISO timestamp") from None
    if result.tzinfo is None:
        raise ValueError("expected a timezone-aware ISO timestamp")
    return result.astimezone(timezone.utc)


def decode_router(html):
    """Decode the inert JSON reference table; never evaluate page JavaScript."""
    chunks = re.findall(r'streamController\.enqueue\(("(?:\\.|[^"\\])*")\)', html)
    if not chunks:
        raise ValueError("BLAST page has no structured router data")
    flat = json.loads(json.loads(chunks[0]).splitlines()[0])
    if not isinstance(flat, list):
        raise ValueError("unexpected router reference table")
    cache = {}

    def decode(index):
        if type(index) is not int or index >= len(flat):
            raise ValueError("invalid router reference")
        if index < 0:
            return None
        if index in cache:
            return cache[index]
        value = flat[index]
        if isinstance(value, dict):
            result = {}
            cache[index] = result
            for key, reference in value.items():
                if not re.fullmatch(r"_\d+", key):
                    raise ValueError("unexpected router object key")
                name = decode(int(key[1:]))
                if not isinstance(name, str):
                    raise ValueError("unexpected router key reference")
                result[name] = decode(reference)
            return result
        if isinstance(value, list):
            # Tagged router values (promises etc.) are deliberately left inert.
            if value and isinstance(value[0], str):
                return value
            result = []
            cache[index] = result
            result.extend(decode(item) for item in value)
            return result
        return value

    return decode(0)


def profile_document(html, platform, account_id, url, *, season_id=None):
    root = decode_router(html)
    if not isinstance(root, dict) or not isinstance(root.get("loaderData"), dict):
        raise ValueError("unexpected BLAST loader data")
    loader = root["loaderData"]
    profile, page = loader.get(PROFILE_ROUTE, {}), loader.get(PLAYLIST_ROUTE, {})
    if not isinstance(profile, dict) or not isinstance(page, dict):
        raise ValueError("unexpected BLAST profile data")
    if profile.get("platform") != platform or profile.get("userId") != account_id:
        raise ValueError("BLAST account identity mismatch")
    if page.get("selectedPlaylistId") != 10:
        raise ValueError("BLAST did not select ranked duels")
    if season_id and page.get("selectedSeasonId") != season_id:
        raise ValueError("BLAST did not select the requested season")
    snapshots = page.get("snapshots", [])
    if not isinstance(snapshots, list) or any(not isinstance(p, dict) or p.get("playlistId") != 10 for p in snapshots):
        raise ValueError("BLAST returned another playlist's history")
    if (not isinstance(page.get("seasons"), list) or not isinstance(page.get("playlists"), list)
            or any(not isinstance(p, dict) for p in page["seasons"] + page["playlists"])):
        raise ValueError("unexpected BLAST season or playlist data")
    # Retain only rank provenance, not unrelated site configuration or scripts.
    return {"format": FORMAT, "source_url": url, "fetched_at": datetime.now(timezone.utc).isoformat(),
            "platform": platform, "account_id": account_id, "selected_playlist_id": 10,
            "selected_season_id": page.get("selectedSeasonId"),
            "seasons": [{k: s.get(k) for k in ("id", "name", "season", "startedAt", "endedAt")}
                        for s in page.get("seasons", [])],
            "playlists": [{k: p.get(k) for k in SNAPSHOT_FIELDS}
                          for p in page.get("playlists", []) if p.get("playlistId") == 10],
            "snapshots": [{k: p.get(k) for k in SNAPSHOT_FIELDS} for p in snapshots]}


def season_at(seasons, when):
    for season in seasons:
        if not isinstance(season, dict):
            raise ValueError("unexpected BLAST season data")
        start = timestamp(season.get("startedAt"))
        end = timestamp(season["endedAt"]) if season.get("endedAt") else None
        if start <= when and (end is None or when < end):
            return season
    return None


def history_evidence(document, start, end, *, max_gap_days=3):
    """Require nearby, placed SSL duel snapshots bracketing the entire match."""
    result = {"source": "blast-public-history", "source_url": document["source_url"],
              "fetched_at": document["fetched_at"],
              "player_id": f"{document['platform']}:{document['account_id']}",
              "playlist_id": 10, "season_id": document.get("selected_season_id"),
              "rank_verified_at_match": False, "max_gap_days": max_gap_days,
              "match_start": start.isoformat(), "match_end": end.isoformat(),
              "status": "unverified", "before": None, "after": None}
    season = season_at(document["seasons"], start)
    if (end < start or not season or season["id"] != document.get("selected_season_id")
            or season_at(document["seasons"], end) != season or document.get("selected_playlist_id") != 10):
        result["status"] = "season_or_playlist_mismatch"
        return result
    observations = []
    for snapshot in document["snapshots"]:
        if snapshot.get("playlistId") != 10:
            result["status"] = "season_or_playlist_mismatch"
            return result
        observed_at = timestamp(snapshot.get("scrapedAt"))
        if season_at(document["seasons"], observed_at) == season:
            observations.append((observed_at, snapshot))
    before = [(t, p) for t, p in observations if t <= start]
    after = [(t, p) for t, p in observations if t >= end]
    left = max(before, key=lambda row: row[0]) if before else None
    right = min(after, key=lambda row: row[0]) if after else None
    result.update(before=left[1] if left else None, after=right[1] if right else None)
    if not left or not right:
        result["status"] = "missing_bracket"
    elif start - left[0] > timedelta(days=max_gap_days) or right[0] - end > timedelta(days=max_gap_days):
        result["status"] = "stale_bracket"
    else:
        interval = [p for t, p in observations if left[0] <= t <= right[0]]
        if any(type(p.get("tier")) is not int or p["tier"] != 22 for p in interval):
            result["status"] = "not_ssl_through_bracket"
        elif any(type(p.get("placementMatchesPlayed")) is not int or p["placementMatchesPlayed"] < 10
                 for p in interval):
            result["status"] = "placements_unfinished"
        else:
            result["status"] = "ssl_bracketed_not_match_verified"
    return result


class BlastRankClient:
    """Separate, unauthenticated, conservatively rate-limited public-page client."""

    def __init__(self, cache, *, interval=1.0):
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.interval, self.next_request = interval, 0.0

    def profile(self, platform, account_id, *, season_id=None):
        if platform not in ("steam", "epic"):
            raise ValueError("unsupported BLAST account platform")
        url = f"https://blast.tv/rl/user/{platform}/{quote(account_id, safe='')}/playlists?playlist=10"
        if season_id:
            url += f"&season={quote(season_id, safe='')}"
        path = self.cache / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if path.exists():
            document = json.loads(path.read_text())
            if (document.get("format") != FORMAT or document.get("source_url") != url
                    or document.get("platform") != platform or document.get("account_id") != account_id
                    or document.get("selected_playlist_id") != 10
                    or (season_id and document.get("selected_season_id") != season_id)):
                raise ValueError("BLAST cache provenance mismatch")
            return document
        time.sleep(max(0, self.next_request - time.monotonic()))
        self.next_request = time.monotonic() + self.interval
        response = self.session.get(url, timeout=45)
        response.raise_for_status()
        document = profile_document(response.text, platform, account_id, url, season_id=season_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, indent=2) + "\n")
        temporary.replace(path)
        return document

    def evidence(self, player, start, end, *, max_gap_days=3):
        identity = player["id"]
        document = self.profile(identity["platform"], identity["id"])
        season = season_at(document["seasons"], start)
        if season and season["id"] != document["selected_season_id"]:
            document = self.profile(identity["platform"], identity["id"], season_id=season["id"])
        return history_evidence(document, start, end, max_gap_days=max_gap_days)
