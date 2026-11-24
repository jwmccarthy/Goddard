"""Download Ballchasing group trees described by a JSON config."""

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from replays import REPLAYS_DIR
from replays.api import API_URL, CachedClient
from replays.utils import file_sha256, write_json
from training.logging import configure_logging


def group_children(
    client:   CachedClient,
    group_id: str
) -> list[dict[str, Any]]:
    page = client.get(f"{API_URL}/groups", group=group_id, count=200)

    children = list(page.replays)

    while page.next_url:
        page = client.get(page.next_url)
        children.extend(page.replays)

    return sorted(
        children,
        key=lambda group: (group.get("name", ""), group["id"])
    )


def discover(
    client:         CachedClient,
    root:           dict[str, Any],
    include_groups: str | None = None
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    visited: set[str] = set()
    pattern = re.compile(include_groups, re.IGNORECASE) if include_groups else None

    def walk(group: dict[str, Any], path: list[str], selected: bool) -> None:
        if group["id"] in visited:
            return

        visited.add(group["id"])
        name = group.get("name", group["id"])
        path = path + [name]
        selected = selected or pattern is None or bool(pattern.search(name))
        direct = group.get("direct_replays")

        if direct != 0 and selected:
            for item in client.find_replay_entries(
                group=group["id"],
                count=200,
                sort_by="replay-date",
                sort_dir="asc"
            ):
                uid = item["id"].lower()
                record = entries.setdefault(
                    uid,
                    {"id": uid, "metadata": item, "occurrences": []}
                )
                record["occurrences"].append(
                    {"group_id": group["id"], "path": path}
                )

        if (
            direct
            and group.get("total_replays", group.get("indirect_replays"))
            == direct
        ):
            return

        for child in group_children(client, group["id"]):
            walk(child, path, selected)

    group = client.get_json(f"{API_URL}/groups/{root['group_id']}")
    walk(group, [], False)

    return entries


def fetch_pending(
    client:  CachedClient,
    pending: list[str],
    raw:     Path,
    workers: int,
    errors:  dict[str, str]
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(client.download_replay, uid, raw): uid
            for uid in pending
        }

        for future in as_completed(futures):
            uid = futures[future]

            try:
                if not future.result():
                    errors[uid] = "not_found"
            except Exception as error:
                errors[uid] = str(error)


def update_record(
    previous:  dict[str, Any],
    uid:       str,
    item:      dict[str, Any],
    path:      Path,
    root:      dict[str, Any],
    team_size: int
) -> dict[str, Any]:
    sha = file_sha256(path)

    if previous.get("raw_sha256", sha) != sha:
        raise ValueError(f"raw replay checksum changed: {uid}")

    occurrences = list(previous.get("provenance", {}).get("occurrences", []))

    for occurrence in item["occurrences"]:
        if occurrence not in occurrences:
            occurrences.append(occurrence)

    return {
        **previous,
        **root.get("metadata", {}),
        "id":         uid,
        "raw":        str(path),
        "raw_sha256": sha,
        "team_size":  team_size,
        "provenance": {
            "id":          uid,
            "metadata":    item["metadata"],
            "occurrences": occurrences
        }
    }


def match_key(item: dict[str, Any]) -> str | None:
    """In-game match GUID, which identifies a match uploaded more than once."""
    guid = item["metadata"].get("rocket_league_id")

    if guid and set(guid.replace("-", "")) != {"0"}:
        return guid.lower()

    return None


def collect(
    client:  CachedClient,
    config:  dict[str, Any],
    output:  Path,
    records: dict[str, dict[str, Any]]
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Discover every root, deduplicating by replay id and by match."""
    selected: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    matches = {
        key: uid for uid, record in records.items()
        if (key := match_key(record["provenance"]))
    }

    for root in config["roots"]:
        entries = discover(client, root, config.get("include_groups"))
        write_json(output / "provenance" / f"{root['group_id']}.json", entries)
        logging.info("%s: %s discovered.", root["group_id"], len(entries))

        for uid, item in entries.items():
            key = match_key(item)
            owner = matches.setdefault(key, uid) if key else uid

            if owner != uid:
                logging.info("%s duplicates match of %s.", uid, owner)
                continue

            if uid in selected:
                selected[uid][1]["occurrences"].extend(
                    occurrence for occurrence in item["occurrences"]
                    if occurrence not in selected[uid][1]["occurrences"]
                )
            else:
                selected[uid] = (root, item)

    return selected


def download(
    config:  dict[str, Any],
    output:  Path,
    workers: int,
    refresh: bool = False
) -> None:
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    client = CachedClient(
        os.environ["BALLCHASING_TOKEN"], output / "provenance/api_cache", refresh
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = {record["id"]: record for record in manifest.get("records", [])}
    selected = collect(client, config, output, records)
    pending = [uid for uid in selected if not (raw / f"{uid}.replay").exists()]
    errors: dict[str, str] = {}

    logging.info("%s downloads.", f"{len(pending):,}")
    fetch_pending(client, pending, raw, workers, errors)

    for uid, (root, item) in selected.items():
        path = raw / f"{uid}.replay"

        if path.is_file():
            records[uid] = update_record(
                records.get(uid, {}), uid, item, path, root, config["team_size"]
            )

    write_json(
        manifest_path,
        {
            "config":  config,
            "records": sorted(records.values(), key=lambda r: r["id"])
        }
    )
    write_json(output / "provenance/download-errors.json", errors)

    if errors:
        raise RuntimeError(
            f"{len(errors)} downloads failed; see provenance/download-errors.json"
        )


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--refresh", action="store_true",
        help="Refresh cached group inventories."
    )
    args = parser.parse_args()

    download(
        json.loads(args.config.read_text()),
        REPLAYS_DIR / args.config.stem,
        args.workers,
        args.refresh
    )


if __name__ == "__main__":
    main()
