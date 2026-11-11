"""Investigate missing duel ranks in cached Ballchasing listing pages.

Writes a research report, NOT an approved corpus or training manifest. Both
nearby historical SSL observations are still not exact match-rank evidence.
"""

import argparse
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

import requests

from replays.collect_duels import listing_rejection, player_id, players, write_json
from replays.duel_ranks import BlastRankClient, timestamp


def investigate(pages, rank_client, config, output, *, max_matches=200, max_gap_days=3):
    seen, reports, counts = set(), [], Counter()
    for path in pages:
        page = json.loads(path.read_text())
        for listing in page.get("list", []):
            replay_id = listing["id"]
            if replay_id in seen:
                continue
            seen.add(replay_id)
            reason = listing_rejection(listing, config, require_rank=False)
            if reason:
                counts[reason] += 1
                continue
            end = timestamp(listing.get("date"))
            # Listing dates can be local recording dates, not reliable epochs.
            # Use a generous interval for preliminary screening. Raw epochs must
            # be checked independently before considering an otherwise good pair.
            start = end - timedelta(seconds=listing["duration"], hours=2)
            end += timedelta(hours=2)
            report = {"id": replay_id, "date": listing["date"],
                      "players": [{"id": player_id(p), "name": p.get("name"),
                                   "pro": p.get("pro") is True} for p in players(listing)],
                      "goals": [listing[c]["goals"] for c in ("blue", "orange")],
                      "rank_verified_at_match": False, "approved_for_training": False,
                      "time_basis": "listing-date-plus-minus-two-hours; raw epoch not checked",
                      "evidence": []}
            for player in players(listing):
                try:
                    evidence = rank_client.evidence(player, start, end, max_gap_days=max_gap_days)
                except (ValueError, requests.RequestException) as error:
                    # Do not log request headers, page bodies or credentials.
                    evidence = {"player_id": player_id(player), "status": "lookup_failed",
                                "error_type": type(error).__name__, "rank_verified_at_match": False}
                report["evidence"].append(evidence)
                counts[evidence["status"]] += 1
                if evidence["status"] != "ssl_bracketed_not_match_verified":
                    break  # No pair can qualify; avoid unnecessary public requests.
            report["both_ssl_bracketed"] = (len(report["evidence"]) == 2 and all(
                e["status"] == "ssl_bracketed_not_match_verified" for e in report["evidence"]))
            reports.append(report)
            result = {"format": "duel-rank-research-v1", "approved_matches": 0,
                      "examined": len(reports), "both_ssl_bracketed": sum(r["both_ssl_bracketed"] for r in reports),
                      "counts": dict(counts), "config": config, "matches": reports}
            write_json(output, result)
            print(json.dumps({k: v for k, v in result.items() if k not in ("matches", "config")}), flush=True)
            if len(reports) >= max_matches:
                return result
    result = {"format": "duel-rank-research-v1", "approved_matches": 0,
              "examined": len(reports), "both_ssl_bracketed": sum(r["both_ssl_bracketed"] for r in reports),
              "counts": dict(counts), "config": config, "matches": reports}
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pages", nargs="+", type=Path, help="cached API listing page JSON files")
    parser.add_argument("--output", type=Path, required=True, help="research JSON report, not a training manifest")
    parser.add_argument("--cache", type=Path, required=True, help="public, sanitized rank-history cache")
    parser.add_argument("--replay-date-after", required=True)
    parser.add_argument("--replay-date-before")
    parser.add_argument("--max-matches", type=int, default=200)
    parser.add_argument("--max-gap-days", type=float, default=3)
    args = parser.parse_args()
    after = timestamp(args.replay_date_after).isoformat()
    before = timestamp(args.replay_date_before).isoformat() if args.replay_date_before else None
    if args.max_matches < 1 or not 0 < args.max_gap_days <= 7:
        raise ValueError("max matches must be positive; maximum snapshot gap must be in (0, 7] days")
    if before and timestamp(after) >= timestamp(before):
        raise ValueError("match-date lower bound must precede upper bound")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {"max_goal_margin": 2, "min_play_seconds": 300, "min_player_score": 100,
              "min_player_shots": 1, "replay_date_after": after, "replay_date_before": before}
    investigate(args.pages, BlastRankClient(args.cache), config, args.output,
                max_matches=args.max_matches, max_gap_days=args.max_gap_days)


if __name__ == "__main__":
    main()
