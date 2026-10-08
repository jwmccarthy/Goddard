"""Build a recent, two-verified-pro 1v1 corpus directly from Ballchasing."""

import argparse
import hashlib
import json
import os

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from ballchasing_replays.ballchasing_api import API_URL, BallchasingClient
from ballchasing_replays.download_mechanical_duels import (
    UUID, choose_focal_povs, existing_replay_ids, fair_download_order,
    month_offset, replay_datetime, replay_player_ids, verified_players,
)
from ballchasing_replays.parse_replays import parse


DEFAULT_ROSTER = Path(__file__).resolve().parent.parent / "mechanical_duels_roster.json"
DEFAULT_REPLAYS = Path(__file__).resolve().parent.parent / "parsed_replays" / "duels"
DEFAULT_PARSED = DEFAULT_REPLAYS
MIN_REPLAY_BYTES = 8192
MAX_REPLAY_BYTES = 20 * 1024 * 1024


def periods_between(after: datetime, before: datetime) -> list[tuple[datetime, datetime]]:
    starts = [after]
    while True:
        next_start = month_offset(after, 3 * len(starts))
        if before - next_start < timedelta(days=30):
            break
        starts.append(next_start)
    return list(zip(starts, starts[1:] + [before]))


def select_replays(client: BallchasingClient, roster: dict, after: datetime,
                   before: datetime, *, max_pages: int = 4) -> tuple[dict[str, dict], Counter]:
    """Keep actual 1v1s when both IDs are pro-tagged or independently verified."""
    if max_pages < 1 or after >= before:
        raise ValueError("page limit and date range must be valid")
    identities = verified_players(roster)
    selected = {}
    by_match = {}
    counts = Counter()
    for playlist in ("private", "ranked-duels"):
        for player in roster["players"]:
            for platform_id in player["platform_ids"]:
                for period, (start, end) in enumerate(periods_between(after, before)):
                    pages = client.iter_replay_pages(
                        playlist=playlist, player_id=platform_id,
                        replay_date_after=start.isoformat(),
                        replay_date_before=end.isoformat(),
                        count=200, sort_by="replay-date", sort_dir="desc",
                    )
                    for page_no, page in enumerate(pages, 1):
                        counts["pages"] += 1
                        for replay in page:
                            replay_id = str(replay.get("id", "")).lower()
                            if (not UUID.fullmatch(replay_id)
                                    or replay.get("playlist_id") != playlist
                                    or replay.get("duration", 0) < 60
                                    or not replay.get("date")):
                                continue
                            date = replay_datetime(replay["date"])
                            players = replay_player_ids(replay)
                            if (not start <= date < end or players is None
                                    or platform_id.lower() not in players):
                                continue
                            info = [p for side in ("blue", "orange")
                                    for p in replay[side]["players"]]
                            verified = [p.get("pro") is True or pid in identities
                                        for p, pid in zip(info, players)]
                            if not all(verified):
                                counts["not_two_pros"] += 1
                                continue
                            counts[playlist] += 1
                            match_id = replay.get("rocket_league_id")
                            match_key = (str(match_id or replay_id).lower(),
                                         tuple(sorted(players)))
                            canonical = by_match.setdefault(match_key, replay_id)
                            targets = {identities[pid]: pid.split(":", 1)[1]
                                       for pid in players if pid in identities}
                            selected.setdefault(canonical, {
                                "date": date.isoformat(), "duration": replay["duration"],
                                "playlist_id": playlist, "period": period,
                                "match_id": match_id,
                                "players": [
                                    {"id": pid, "name": p.get("name", ""),
                                     "ballchasing_pro": p.get("pro") is True,
                                     "roster_pro": pid in identities}
                                    for p, pid in zip(info, players)
                                ],
                                "target_online_ids": {},
                            })["target_online_ids"].update(targets)
                        if page_no >= max_pages:
                            counts["page_limit_reached"] += 1
                            break
    return selected, counts


def existing_corpus_ids(replay_dir: Path, parsed_dir: Path) -> set[str]:
    """Check all local raw/parsed corpora, excluding this downloader's raw files."""
    root = Path(__file__).resolve().parent.parent
    directories = [parsed_dir, root / "parsed_replays" / "ranked"]
    directories.extend((root / "parsed_replays").glob("pro_*_fs4"))
    result = existing_replay_ids(directories)
    for path in (root / "ballchasing_replays").rglob("*"):
        if (path.suffix not in (".npy", ".replay")
                or replay_dir.resolve() in path.parents):
            continue
        if match := UUID.search(path.stem):
            result.add(match.group().lower())
    return result


def download_order(candidates: dict[str, dict], names: list[str]) -> list[str]:
    """Include ranked matches and balance each mode over players and quarters."""
    ranked = {key: item for key, item in candidates.items()
              if item["playlist_id"] == "ranked-duels"}
    private = {key: item for key, item in candidates.items()
               if item["playlist_id"] == "private"}
    order = []
    for group in (ranked, private):
        if group:
            order.extend(fair_download_order(group, names, len(group)))
    return order


def file_signature(path: Path) -> tuple[int, str] | None:
    if not path.is_file():
        return None
    size = path.stat().st_size
    if not MIN_REPLAY_BYTES <= size <= MAX_REPLAY_BYTES:
        return None
    with path.open("rb") as file:
        header = file.read(12)
    if not 8 <= int.from_bytes(header[:4], "little") < size:
        return None
    if int.from_bytes(header[8:12], "little") <= 0:
        return None
    return size, hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifests(directory: Path, manifest: dict, names: list[str]) -> None:
    choose_focal_povs(manifest["selected"], names)
    files = {
        "selection.json": manifest,
        "pov_players.json": {replay_id: [entry["focal_online_id"]]
                             for replay_id, entry in manifest["selected"].items()},
    }
    for filename, content in files.items():
        temp = directory / f".{filename}.tmp"
        temp.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
        temp.replace(directory / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED)
    parser.add_argument("--since", help="UTC YYYY-MM-DD (default: 24 months ago)")
    parser.add_argument("--until", help="exclusive UTC YYYY-MM-DD (default: tomorrow)")
    parser.add_argument("--max-pages-per-period", type=int, default=4)
    parser.add_argument("--max-downloads", type=int, default=1000)
    parser.add_argument("--target-parsed", type=int, default=1000)
    parser.add_argument("--list-rate", type=float, default=2)
    parser.add_argument("--download-rate", type=float, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--exclude-ids-manifest", type=Path,
                        help="ignore IDs from an interim local corpus when deduplicating")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--parse", action="store_true")
    args = parser.parse_args()
    if min(args.max_pages_per_period, args.max_downloads, args.target_parsed,
           args.list_rate, args.download_rate, args.workers) <= 0:
        parser.error("page, replay, frame and rate limits must be positive")
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    after = replay_datetime(args.since) if args.since else month_offset(today, -24)
    before = replay_datetime(args.until) if args.until else today + timedelta(days=1)
    if after >= before:
        parser.error("since must precede until")
    roster = json.loads(args.roster.read_text())
    names = [player["name"] for player in roster["players"]]
    roster_ids = verified_players(roster)
    roster_hash = hashlib.sha256(json.dumps(roster_ids, sort_keys=True).encode()).hexdigest()
    selection_path = args.replay_dir / "selection.json"
    previous = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
    if previous and (previous["source"] != API_URL or previous["after"] != after.isoformat()
                     or previous["before"] != before.isoformat()
                     or previous["roster_hash"] != roster_hash):
        parser.error("existing selection uses a different source, date range, or roster")
    if not os.environ.get("BALLCHASING_TOKEN"):
        parser.error("set BALLCHASING_TOKEN in the environment or gitignored .env")
    client = BallchasingClient(os.environ["BALLCHASING_TOKEN"],
                              list_rate=args.list_rate, download_rate=args.download_rate)
    if previous:
        candidates = previous["candidates"]
    else:
        candidates, counts = select_replays(client, roster, after, before,
                                            max_pages=args.max_pages_per_period)
        print(f"Queried {counts['pages']} Ballchasing pages; "
              f"{counts['page_limit_reached']} searches reached their page limit",
              flush=True)
    ignored = set()
    if args.exclude_ids_manifest:
        ignored = set(json.loads(args.exclude_ids_manifest.read_text())["selected"])
    existing = (existing_corpus_ids(args.replay_dir, args.parsed_dir)
                - set(previous.get("selected", {})) - ignored)
    candidates = {replay_id: entry for replay_id, entry in candidates.items()
                  if replay_id not in existing}
    order = download_order(candidates, names)
    modes = Counter(item["playlist_id"] for item in candidates.values())
    print(f"{after.date()} to {before.date()} (exclusive): {len(candidates)} "
          f"new, two-pro 1v1 matches from Ballchasing ({dict(modes)})", flush=True)
    if args.metadata_only:
        return
    args.replay_dir.mkdir(parents=True, exist_ok=True)
    manifest = previous or {
        "source": API_URL, "after": after.isoformat(), "before": before.isoformat(),
        "roster_hash": roster_hash, "candidates": candidates, "selected": {},
        "unavailable": [],
    }
    selected = manifest["selected"]
    for replay_id in list(selected):
        entry = selected[replay_id]
        actual = file_signature(args.replay_dir / f"{replay_id}.replay")
        if not actual or actual != (entry.get("size_bytes"), entry.get("sha256")):
            del selected[replay_id]
    unavailable = set(manifest["unavailable"])
    rejected_after_parse = set(manifest.get("rejected_after_parse", []))
    remaining = (replay_id for replay_id in order
                 if (replay_id not in selected and replay_id not in unavailable
                     and replay_id not in rejected_after_parse))
    while len(selected) < args.max_downloads:
        batch = []
        for _ in range(min(25, args.max_downloads - len(selected))):
            if (replay_id := next(remaining, None)) is None:
                break
            batch.append(replay_id)
        if not batch:
            break
        with TemporaryDirectory(dir=args.replay_dir) as temporary:
            client.download_replays(batch, temporary)
            for replay_id in batch:
                raw = Path(temporary) / f"{replay_id}.replay"
                signature = file_signature(raw)
                if signature is None:
                    unavailable.add(replay_id)
                    continue
                raw.replace(args.replay_dir / raw.name)
                selected[replay_id] = {
                    **candidates[replay_id],
                    "size_bytes": signature[0], "sha256": signature[1],
                }
        manifest["unavailable"] = sorted(unavailable)
        write_manifests(args.replay_dir, manifest, names)
        print(f"Downloaded {len(selected)}/{args.max_downloads} direct Ballchasing "
              f"replays ({len(unavailable)} unavailable)", flush=True)
    if len(selected) < args.max_downloads:
        raise RuntimeError(f"only {len(selected)} of {args.max_downloads} replays available")
    if args.parse:
        parse(str(args.replay_dir), str(args.parsed_dir), frame_skip=4,
              workers=args.workers, pov_manifest=str(args.replay_dir / "pov_players.json"),
              replay_ids=set(selected), require_pov_manifest=True, team_size=1)
        parsed = len(existing_replay_ids([args.parsed_dir]) & selected.keys())
        print(f"Parsed {parsed}/{len(selected)} selected replays", flush=True)
        if parsed < args.target_parsed:
            raise RuntimeError(f"only {parsed}/{args.target_parsed} training-ready games; "
                               "raise --max-downloads to draw replacement matches")


if __name__ == "__main__":
    main()
