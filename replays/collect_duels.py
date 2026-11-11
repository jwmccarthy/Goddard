"""Collect close SSL-vs-SSL ranked duels, preferring Ballchasing-identified pros.

API activity is a proxy, not proof of effort. Use --rrrocket to reject raw
forfeits and require end-clock evidence before accepting a replay.
Player ranks must be explicitly recorded; aggregate min/max rank is insufficient.
"""

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from uuid import UUID

from replays.api import API_URL, BallchasingClient
from replays.duel_audit import FORMAT as AUDIT_FORMAT
from replays.duel_audit import audit_replay
from replays.duel_ranks import timestamp

SSL = "supersonic-legend"


def player_id(player):
    identity = player.get("id", {})
    platform, value = identity.get("platform"), identity.get("id")
    return f"{platform}:{value}" if platform and value else None


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def players(replay):
    teams = [replay.get(color, {}).get("players", []) for color in ("blue", "orange")]
    return [team[0] for team in teams] if all(isinstance(team, list) and len(team) == 1 for team in teams) else []


def listing_rejection(replay, config, *, require_rank=True):
    if replay.get("playlist_id") != "ranked-duels":
        return "playlist"
    pair = players(replay)
    if len(pair) != 2 or any(not player_id(player) for player in pair):
        return "players"
    if player_id(pair[0]) == player_id(pair[1]):
        return "players"
    if require_rank and any((player.get("rank") or {}).get("id") != SSL for player in pair):
        return "unverified_ssl"
    if any(p.get("rank") is not None and p["rank"].get("id") != SSL for p in pair):
        return "rank_conflict"
    if config.get("replay_date_after") or config.get("replay_date_before"):
        try:
            date = timestamp(replay.get("date"))
        except ValueError:
            return "date_unverified"
        if config.get("replay_date_after") and date <= timestamp(config["replay_date_after"]):
            return "date_outside_range"
        if config.get("replay_date_before") and date >= timestamp(config["replay_date_before"]):
            return "date_outside_range"
    goals = [replay.get(color, {}).get("goals") for color in ("blue", "orange")]
    if any(not number(goal) or goal < 0 or int(goal) != goal for goal in goals):
        return "score_missing"
    if not 1 <= abs(goals[0] - goals[1]) <= config["max_goal_margin"]:
        return "score_margin"
    duration = replay.get("duration")
    if not number(duration) or duration < config["min_play_seconds"]:
        return "duration"
    if any(not number(player.get("score")) or player["score"] < config["min_player_score"] for player in pair):
        return "player_score"
    return None


def detail_rejection(listing, detail, config):
    if detail.get("status") != "ok" or detail.get("team_size") != 1 or detail.get("playlist_id") != "ranked-duels":
        return "detail_format"
    pair = players(detail)
    if len(pair) != 2 or [player_id(p) for p in pair] != [player_id(p) for p in players(listing)]:
        return "player_mismatch"
    # Details can omit ranks which the listing explicitly supplied; a conflicting
    # recorded rank is different and must fail closed.
    if any(p.get("rank") is not None and p["rank"].get("id") != SSL for p in pair):
        return "rank_conflict"
    duration, played = detail.get("duration"), detail.get("play_duration")
    if not number(played) or played < config["min_play_seconds"] or not number(duration):
        return "play_duration"
    for player in pair:
        start, end = player.get("start_time"), player.get("end_time")
        if not number(start) or not number(end) or start > 5 or end < duration - 5:
            return "participation"
        core = player.get("stats", {}).get("core", {})
        if not number(core.get("score")) or core["score"] < config["min_player_score"]:
            return "detail_score"
        if not number(core.get("shots")) or core["shots"] < config["min_player_shots"]:
            return "player_activity"
    goals = [p["stats"]["core"].get("goals") for p in pair]
    if goals != [listing[color]["goals"] for color in ("blue", "orange")]:
        return "score_conflict"
    if detail.get("forfeit") is True or detail.get("abandoned") is True:
        return "forfeit"
    return None


def recorded_rank_evidence(listing, detail):
    evidence = []
    for listed, actual in zip(players(listing), players(detail)):
        if player_id(listed) != player_id(actual):
            return None
        source = "listing" if (listed.get("rank") or {}).get("id") == SSL else "detail"
        rank = (listed if source == "listing" else actual).get("rank") or {}
        if rank.get("id") != SSL:
            return None
        evidence.append({"player_id": player_id(actual), "playlist": "ranked-duels",
                         "source": f"ballchasing-{source}", "rank": rank})
    return evidence if len(evidence) == 2 else None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def cached_json(path, client, url, **params):
    if not path.exists():
        write_json(path, client.get_json(url, **params))
    return json.loads(path.read_text())


def collect(args, client):
    output = args.output
    for child in ("raw", "pages", "metadata"):
        (output / child).mkdir(parents=True, exist_ok=True)
    config = {"format": "ssl-duels-v1", "playlist": "ranked-duels", "rank": SSL,
              "max_goal_margin": args.max_goal_margin, "min_play_seconds": args.min_play_seconds,
              "min_player_score": args.min_player_score, "min_player_shots": args.min_player_shots,
              "preference": "pro-first", "completion_check": "api-proxy-not-end-clock-audited"}
    broad = getattr(args, "rank_discovery", "filtered") == "broad"
    decoder = getattr(args, "rrrocket", None)
    after, before = getattr(args, "replay_date_after", None), getattr(args, "replay_date_before", None)
    if after or before or broad or decoder:
        config.update(format="ssl-duels-v2", replay_date_after=after, replay_date_before=before,
                      rank_discovery="broad" if broad else "filtered", rank_policy="both-recorded-ssl")
    if decoder:
        config["completion_check"] = "raw-clock-and-forfeit-v2"
        (output / "audits").mkdir(exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("existing collection has different filters; choose another output directory")
    write_json(config_path, config)
    manifest = output / "manifest.jsonl"
    accepted, match_ids, rocket_ids = {}, set(), set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            record = json.loads(line)
            path = output / "raw" / f"{record['id']}.replay"
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise ValueError(f"missing or truncated accepted replay: {path}")
            accepted[record["id"]] = record
            match_ids.add(record["match_key"])
            if record.get("rocket_league_id"):
                rocket_ids.add(record["rocket_league_id"])
    counts = Counter()

    def progress(phase, page):
        report = {"target": args.limit, "downloaded": len(accepted),
                  "pro_replays": sum(bool(r["pro_players"]) for r in accepted.values()),
                  "bytes": sum(r["bytes"] for r in accepted.values()),
                  "phase": phase, "page": page, "counts_this_run": dict(counts),
                  "updated": datetime.now(timezone.utc).isoformat()}
        write_json(output / "progress.json", report)
        print(json.dumps(report), flush=True)
        return report

    if len(accepted) >= args.limit:
        return progress("complete", 0)
    for phase in ("pro", "all"):
        url = f"{API_URL}/replays"
        params = {"playlist": "ranked-duels",
                  "count": 200, "sort_by": "replay-date", "sort_dir": "desc"}
        if not broad:
            params.update(min_rank=SSL, max_rank=SSL)
        if after:
            params["replay_date_after"] = after
        if before:
            params["replay_date_before"] = before
        if phase == "pro":
            params["pro"] = "true"
        source_params = params.copy()
        for page_number in range(args.max_pages):
            page = cached_json(output / "pages" / f"{phase}-{page_number:05d}.json", client, url, **params)
            for listing in page.get("list", []):
                counts["listed"] += 1
                try:
                    replay_id = str(UUID(listing["id"]))
                except (KeyError, ValueError, TypeError, AttributeError):
                    counts["invalid_id"] += 1
                    continue
                if replay_id in accepted:
                    counts["already_downloaded"] += 1
                    continue
                reason = listing_rejection(listing, config, require_rank=not broad)
                if reason:
                    counts[reason] += 1
                    continue
                pro_players = [player_id(p) for p in players(listing) if p.get("pro") is True]
                if phase == "pro" and not pro_players:
                    counts["unverified_pro"] += 1
                    continue
                rocket_id = listing.get("rocket_league_id")
                if rocket_id and rocket_id in rocket_ids:
                    counts["duplicate_match"] += 1
                    continue
                detail = cached_json(output / "metadata" / f"{replay_id}.json", client,
                                     f"{API_URL}/replays/{replay_id}")
                reason = detail_rejection(listing, detail, config)
                if reason:
                    counts[reason] += 1
                    continue
                rank_evidence = recorded_rank_evidence(listing, detail)
                if not rank_evidence:
                    counts["unverified_ssl"] += 1
                    continue
                match_key = detail.get("match_guid") or detail.get("rocket_league_id")
                if not match_key:
                    counts["missing_match_id"] += 1
                    continue
                if match_key in match_ids:
                    counts["duplicate_match"] += 1
                    continue
                path = output / "raw" / f"{replay_id}.replay"
                if not path.is_file() and not client.download_replay(replay_id, output / "raw"):
                    counts["file_unavailable"] += 1
                    continue
                content = path.read_bytes()
                if not content:
                    raise ValueError(f"empty replay download: {replay_id}")
                digest = hashlib.sha256(content).hexdigest()
                audit = None
                if decoder:
                    audit_path = output / "audits" / f"{replay_id}.json"
                    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
                    if audit is None or audit.get("sha256") != digest or audit.get("format") != AUDIT_FORMAT:
                        audit_listing = dict(listing, overtime=detail.get("overtime", listing.get("overtime")))
                        audit = audit_replay(path, decoder, audit_listing, config)
                        audit["sha256"] = digest
                        write_json(audit_path, audit)
                    if audit.get("rejection") or not audit.get("completion_verified"):
                        counts[f"raw_{audit.get('rejection') or 'completion_unverified'}"] += 1
                        continue
                    if (audit.get("match_guid") or "").casefold() != match_key.casefold():
                        counts["raw_match_mismatch"] += 1
                        continue
                record = {"id": replay_id, "match_key": match_key,
                          "rocket_league_id": rocket_id, "date": listing.get("date"),
                          "goals": [listing[color]["goals"] for color in ("blue", "orange")],
                          "players": players(listing), "pro_players": pro_players,
                          "duration": detail["duration"], "play_duration": detail["play_duration"],
                          "overtime": detail.get("overtime", False),
                          "completion_verified": bool(audit and audit["completion_verified"]),
                          "bytes": len(content), "sha256": digest}
                if config["format"] == "ssl-duels-v2":
                    record.update(rank_evidence=rank_evidence, raw_audit=audit)
                with manifest.open("a") as file:
                    file.write(json.dumps(record) + "\n")
                    file.flush()
                    os.fsync(file.fileno())
                accepted[replay_id] = record
                match_ids.add(match_key)
                if rocket_id:
                    rocket_ids.add(rocket_id)
                if len(accepted) % 10 == 0 or len(accepted) >= args.limit:
                    progress(phase, page_number)
                if len(accepted) >= args.limit:
                    return progress("complete", page_number)
            progress(phase, page_number)
            next_url = page.get("next")
            if not next_url:
                break
            parsed = urlparse(next_url)
            if parsed.scheme != "https" or parsed.netloc != "ballchasing.com" or parsed.path != "/api/replays":
                raise ValueError("unexpected Ballchasing pagination URL")
            next_params = parse_qs(parsed.query)
            url = next_url
            params = {key: value for key, value in source_params.items()
                      if key.replace("_", "-") not in next_params}
    return progress("exhausted_or_page_limit", args.max_pages)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=Path("replays/ssl_duels"))
    result.add_argument("--limit", type=int, default=10000)
    result.add_argument("--max-pages", type=int, default=1000, help="maximum pages per pro/all phase")
    result.add_argument("--max-goal-margin", type=int, default=2)
    result.add_argument("--min-play-seconds", type=float, default=300)
    result.add_argument("--min-player-score", type=int, default=100)
    result.add_argument("--min-player-shots", type=int, default=1)
    result.add_argument("--token-file", type=Path, help="optional private token file outside the repository")
    result.add_argument("--replay-date-after", help="exclusive match-date lower bound (timezone-aware ISO timestamp)")
    result.add_argument("--replay-date-before", help="exclusive match-date upper bound (timezone-aware ISO timestamp)")
    result.add_argument("--rank-discovery", choices=("filtered", "broad"), default="filtered",
                        help="broad omits server rank filters, but STILL requires explicit ranks for both players")
    result.add_argument("--rrrocket", type=Path, help="decoder executable; require raw forfeit/end-clock audit")
    return result


def main():
    args = parser().parse_args()
    if args.limit < 1 or args.max_pages < 1 or args.max_goal_margin < 1:
        raise ValueError("limit, page count and score margin must be positive")
    if not math.isfinite(args.min_play_seconds) or args.min_play_seconds <= 0:
        raise ValueError("minimum play duration must be finite and positive")
    if args.min_player_score < 0 or args.min_player_shots < 0:
        raise ValueError("activity thresholds must be nonnegative")
    for key in ("replay_date_after", "replay_date_before"):
        if getattr(args, key):
            setattr(args, key, timestamp(getattr(args, key)).isoformat())
    if args.replay_date_after and args.replay_date_before:
        if timestamp(args.replay_date_after) >= timestamp(args.replay_date_before):
            raise ValueError("match-date lower bound must precede upper bound")
    if args.rank_discovery == "broad" and not args.replay_date_after:
        raise ValueError("broad discovery requires an explicit --replay-date-after")
    if args.rrrocket and (not args.rrrocket.is_file() or not os.access(args.rrrocket, os.X_OK)):
        raise ValueError("--rrrocket must name an executable decoder")
    if args.token_file and args.token_file.stat().st_mode & 0o077:
        raise ValueError("token file must not be accessible to other users; chmod 600 first")
    token = args.token_file.read_text().strip() if args.token_file else os.environ.get("BALLCHASING_TOKEN", "")
    if not token:
        raise ValueError("set BALLCHASING_TOKEN or supply a private --token-file")
    client = BallchasingClient(token, list_rate=2, download_rate=1 / 18)
    tier = client.get_json(f"{API_URL}/").get("type", "regular")
    # Respect hourly limits too, not just burst requests per second.
    list_rate, download_rate = {"gc": (8, 2), "champion": (4, 2000 / 3600),
                                "diamond": (2000 / 3600, 1000 / 3600),
                                "gold": (1000 / 3600, 400 / 3600)}.get(tier, (500 / 3600, 200 / 3600))
    client.list_interval, client.download_interval = 1 / list_rate, 1 / download_rate
    print(json.dumps({"account_tier": tier, "download_rate": download_rate,
                      "rank_policy": "both players explicitly SSL", "token": "not logged"}), flush=True)
    collect(args, client)


if __name__ == "__main__":
    main()
