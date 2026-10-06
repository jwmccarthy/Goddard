"""Select verified players' ranked 1v1 or 2v2 replays and parse one focal POV."""

import argparse
import hashlib
import json
import os
import re

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from ballchasing_replays.ballchasing_api import BallchasingClient
    from ballchasing_replays.parse_replays import parse
except ModuleNotFoundError:
    from ballchasing_api import BallchasingClient
    from parse_replays import parse


UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", re.I)
DEFAULT_ROSTER = Path(__file__).resolve().parent.parent / "mechanical_duels_roster.json"
DEFAULT_REPLAYS = Path(__file__).resolve().parent / "mechanical_duels" / "replays"
DEFAULT_PARSED = Path(__file__).resolve().parent.parent / "parsed_replays" / "pro_1v1_fs4"
DOUBLES_REPLAYS = Path(__file__).resolve().parent / "mechanical_doubles" / "replays"
DOUBLES_PARSED = Path(__file__).resolve().parent.parent / "parsed_replays" / "pro_2v2_fs4"


def month_offset(day: datetime, months: int) -> datetime:
    """Advance on the same calendar day, clamping short months."""
    month = day.year * 12 + day.month - 1 + months
    year, remainder = divmod(month, 12)
    from calendar import monthrange

    return day.replace(year=year, month=remainder + 1,
                       day=min(day.day, monthrange(year, remainder + 1)[1]))


def replay_datetime(value: str) -> datetime:
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return date.replace(tzinfo=timezone.utc) if date.tzinfo is None else date.astimezone(timezone.utc)


def replay_player_ids(replay: dict, team_size: int = 1) -> tuple[str, ...] | None:
    blue = replay.get("blue", {}).get("players", ())
    orange = replay.get("orange", {}).get("players", ())
    if len(blue) != team_size or len(orange) != team_size:
        return None
    result = []
    for player in (*blue, *orange):
        identity = player.get("id") or {}
        if not identity.get("platform") or not identity.get("id"):
            return None
        result.append(f"{identity['platform'].lower()}:{str(identity['id']).lower()}")
    return tuple(result)


def accepted_replay(
    replay: dict, player_id: str, after: datetime, before: datetime,
    *, playlist: str = "ranked-duels",
) -> bool:
    players = replay_player_ids(replay, 1 if playlist == "ranked-duels" else 2)
    return bool(
        replay.get("playlist_id") == playlist
        and players is not None and player_id in players
        and replay.get("duration", 0) >= 60
        and after <= replay_datetime(replay["date"]) < before
        and UUID.fullmatch(replay.get("id", ""))
    )


def verified_players(roster: dict) -> dict[str, str]:
    identities = {}
    for player in roster["players"]:
        for platform_id in player["platform_ids"]:
            identity = platform_id.lower()
            if identity in identities and identities[identity] != player["name"]:
                raise ValueError(f"duplicate player ID {identity} in roster")
            identities[identity] = player["name"]
    return identities


def select_replays(
    client: BallchasingClient, roster: dict, after: datetime, before: datetime,
    *, max_per_player: int = 200, max_pages: int = 4,
    playlist: str = "ranked-duels",
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Spread each player's capped selection across roughly three-month intervals."""
    if max_per_player < 1 or max_pages < 1 or after >= before:
        raise ValueError("per-player limit, page limit and date range must be valid")
    if playlist not in ("ranked-duels", "ranked-doubles"):
        raise ValueError("playlist must be ranked duels or doubles")
    identities = verified_players(roster)
    periods = [after]
    while True:
        next_start = month_offset(after, 3 * len(periods))
        # Fold a short final remainder into the previous interval.
        if before - next_start < timedelta(days=30):
            break
        periods.append(next_start)
    periods.append(before)
    n_periods = len(periods) - 1
    # If the quota is smaller than the number of intervals, prefer recent ones.
    quotas = [max_per_player // n_periods + (i >= n_periods - max_per_player % n_periods)
              for i in range(n_periods)]
    selected: dict[str, dict] = {}
    by_match: dict[tuple[str, tuple[str, ...]], str] = {}
    counts: dict[str, dict] = {}
    for player in roster["players"]:
        name = player["name"]
        total = 0
        counts[name] = {"by_period": [], "total": 0}
        seen_for_player: set[tuple[str, tuple[str, ...]]] = set()
        for period, (start, end, quota) in enumerate(zip(periods, periods[1:], quotas)):
            found = 0
            if quota:
                for platform_id in player["platform_ids"]:
                    pages = client.iter_replay_pages(
                        playlist=playlist, player_id=platform_id,
                        replay_date_after=start.isoformat(),
                        replay_date_before=end.isoformat(),
                        count=200, sort_by="replay-date", sort_dir="desc",
                    )
                    for page_no, page in enumerate(pages, 1):
                        for replay in page:
                            replay_id = replay.get("id", "").lower()
                            if (
                                found >= quota
                                or not accepted_replay(
                                    replay, platform_id.lower(), start, end,
                                    playlist=playlist,
                                )
                            ):
                                continue
                            present = replay_player_ids(
                                replay, 1 if playlist == "ranked-duels" else 2,
                            )
                            match_key = (
                                str(replay.get("rocket_league_id") or replay_id).lower(),
                                tuple(sorted(present)),
                            )
                            if match_key in seen_for_player:
                                continue
                            targets = {identities[pid]: pid.split(":", 1)[1]
                                       for pid in present if pid in identities}
                            canonical_id = by_match.setdefault(match_key, replay_id)
                            selected.setdefault(canonical_id, {
                                "date": replay["date"], "duration": replay["duration"],
                                "players": list(present), "period": period,
                                "target_online_ids": {},
                            })["target_online_ids"].update(targets)
                            seen_for_player.add(match_key)
                            found += 1
                            total += 1
                        if found >= quota or page_no >= max_pages:
                            break
                    if found >= quota:
                        break
            counts[name]["by_period"].append(found)
        counts[name]["total"] = total
    return selected, counts


def existing_replay_ids(directories: list[Path]) -> set[str]:
    """Avoid downloading the same UUID already parsed under another POV or filename."""
    existing = set()
    for directory in directories:
        if directory.is_dir():
            for path in directory.glob("*.npy"):
                match = UUID.search(path.stem)
                if match:
                    existing.add(match.group().lower())
    return existing


def downloaded_replay_ids(directory: Path) -> set[str]:
    """Track raw downloads separately so interrupted batches can resume."""
    return {
        path.stem.lower() for path in directory.glob("*.replay")
        if UUID.fullmatch(path.stem)
    }


def fair_download_order(replays: dict[str, dict], names: list[str], limit: int) -> list[str]:
    """Round-robin players AND time periods within a limited download budget."""
    if limit < 1:
        raise ValueError("download limit must be positive")
    periods = sorted({replay.get("period", 0) for replay in replays.values()})
    queues = {
        (period, name): [replay_id for replay_id, replay in sorted(
            replays.items(), key=lambda entry: (entry[1]["date"], entry[0]),
        ) if replay.get("period", 0) == period and name in replay["target_online_ids"]]
        for period in periods for name in names
    }
    selected = []
    seen = set()
    while len(selected) < limit and any(queues.values()):
        for period in periods:
            for name in names:
                queue = queues[period, name]
                while queue and queue[-1] in seen:
                    queue.pop()
                if queue and len(selected) < limit:
                    replay_id = queue.pop()
                    selected.append(replay_id)
                    seen.add(replay_id)
    return selected


def choose_focal_povs(replays: dict[str, dict], roster_names: list[str]) -> None:
    """Retain exactly one roster player's POV per replay, balanced within each period."""
    totals: Counter[str] = Counter()
    by_period: Counter[tuple[int, str]] = Counter()
    for replay_id, replay in sorted(
        replays.items(), key=lambda entry: (entry[1].get("period", 0), entry[1]["date"], entry[0]),
    ):
        candidates = [name for name in roster_names if name in replay["target_online_ids"]]
        if not candidates:
            raise ValueError(f"no roster player POV for selected replay {replay_id}")
        focal = replay.get("focal_player")
        if focal is not None and focal not in candidates:
            raise ValueError(f"focal player {focal} is not in selected replay {replay_id}")
        period = replay.get("period", 0)
        if focal is None:
            focal = min(candidates, key=lambda name: (
                by_period[period, name], totals[name],
                hashlib.sha256(f"{replay_id}:{name}".encode()).digest(),
            ))
        replay["focal_player"] = focal
        replay["focal_online_id"] = replay["target_online_ids"][focal]
        totals[focal] += 1
        by_period[period, focal] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", type=Path, default=DEFAULT_ROSTER)
    parser.add_argument("--playlist", choices=("ranked-duels", "ranked-doubles"),
                        default="ranked-duels")
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--parsed-dir", type=Path)
    parser.add_argument("--max-per-player", type=int, default=200)
    parser.add_argument("--max-downloads", type=int, default=600)
    parser.add_argument("--max-pages-per-period", type=int, default=4)
    parser.add_argument("--since", help="UTC YYYY-MM-DD (default: 24 months ago)")
    parser.add_argument("--until", help="exclusive UTC YYYY-MM-DD (default: tomorrow)")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--parse", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    doubles = args.playlist == "ranked-doubles"
    args.replay_dir = args.replay_dir or (DOUBLES_REPLAYS if doubles else DEFAULT_REPLAYS)
    args.parsed_dir = args.parsed_dir or (DOUBLES_PARSED if doubles else DEFAULT_PARSED)
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    after = replay_datetime(args.since) if args.since else month_offset(today, -24)
    before = replay_datetime(args.until) if args.until else today + timedelta(days=1)
    if args.max_downloads < 1 or args.max_per_player < 1 or args.workers < 1:
        parser.error("download limit, per-player limit and workers must be positive")
    roster = json.loads(args.roster.read_text())
    token = os.environ.get("BALLCHASING_TOKEN")
    if not token:
        parser.error("set BALLCHASING_TOKEN in the environment or gitignored .env")
    # The unaffiliated/free Ballchasing API tier permits two list calls/second.
    client = BallchasingClient(token, list_rate=2, download_rate=2)
    selected, counts = select_replays(
        client, roster, after, before,
        max_per_player=args.max_per_player, max_pages=args.max_pages_per_period,
        playlist=args.playlist,
    )
    skip = existing_replay_ids(
        [args.parsed_dir, args.parsed_dir.parent / "pro_2v2_fs4",
         args.parsed_dir.parent / "mechanical_2v2_fs4"] if doubles else
        [args.parsed_dir, args.parsed_dir.parent / "pro_1v1_fs4",
         args.parsed_dir.parent / "mechanical_1v1_fs4",
         args.parsed_dir.parent / "ranked"]
    )
    raw = downloaded_replay_ids(args.replay_dir)
    new = {replay_id: replay for replay_id, replay in selected.items()
           if replay_id not in skip and replay_id not in raw}
    fresh = fair_download_order(
        new, [player["name"] for player in roster["players"]], args.max_downloads,
    )
    print(f"{after.date()} to {before.date()} (exclusive): "
          f"{len(selected)} {args.playlist} selected under per-player quotas, "
          f"{len(set(selected) & skip)} already parsed, "
          f"{len((set(selected) & raw) - skip)} already downloaded, "
          f"{len(fresh)} selected for download")
    for name, info in counts.items():
        print(f"{name}: {info['total']} selected "
              f"({', '.join(map(str, info['by_period']))} per period; capped)")
    args.replay_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.replay_dir / "ranked_selection.json"
    pov_path = args.replay_dir / "pov_players.json"
    previous = json.loads(selection_path.read_text()) if selection_path.exists() else {}
    if previous and previous.get("playlist") != args.playlist:
        parser.error("replay directory already contains a different playlist's selection")
    history = previous.get("selected", {})
    history.update({replay_id: replay for replay_id, replay in selected.items()
                    if replay_id in raw and replay_id not in skip})
    history.update({replay_id: selected[replay_id] for replay_id in fresh})
    choose_focal_povs(history, [player["name"] for player in roster["players"]])
    selection = {
        "playlist": args.playlist, "after": after.isoformat(),
        "before": before.isoformat(), "per_player": counts, "selected": history,
    }
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
    )
    pov_path.write_text(
        json.dumps({replay_id: [replay["focal_online_id"]]
                    for replay_id, replay in selection["selected"].items()},
                   indent=2, sort_keys=True) + "\n",
    )
    if not args.metadata_only:
        client.download_replays(fresh, args.replay_dir)
        if args.parse:
            parse(str(args.replay_dir), str(args.parsed_dir), frame_skip=4,
                   workers=args.workers,
                    pov_manifest=str(pov_path), replay_ids=set(history),
                    require_pov_manifest=True, fail_on_errors=True,
                    team_size=2 if doubles else 1)


if __name__ == "__main__":
    main()
