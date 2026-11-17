"""Download Ballchasing group trees described by a JSON config."""

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

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
    pattern = (
        re.compile(include_groups, re.IGNORECASE)
        if include_groups
        else None
    )

    def walk(group: dict[str, Any], path: list[str], selected: bool) -> None:
        if group["id"] in visited:
            return

        visited.add(group["id"])
        name = group.get("name", group["id"])
        path = path + [name]
        selected = selected or pattern is None or bool(pattern.search(name))
        direct = group.get("direct_replays", 0)

        if direct and selected:
            for item in client.find_replay_entries(
                group=group["id"],
                count=200,
                sort_by="replay-date",
                sort_dir="asc"
            ):
                uid = item["id"].lower()
                record = entries.setdefault(
                    uid,
                    {
                        "id":          uid,
                        "metadata":    item,
                        "occurrences": [],
                    }
                )

                record["occurrences"].append(
                    {
                        "group_id": group["id"],
                        "path":     path,
                    }
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


def download(config: dict[str, Any], output: Path, workers: int) -> None:
    output = output.resolve()
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    client = CachedClient(os.environ["BALLCHASING_TOKEN"], output / "provenance/api_cache")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = {
        record["id"]: record
        for record in manifest.get("records", [])
    }

    errors: dict[str, str] = {}

    for root in config["roots"]:
        entries = discover(client, root, config.get("include_groups"))

        write_json(output / "provenance" / f"{root['group_id']}.json", entries)
        pending = [uid for uid in entries if not (raw / f"{uid}.replay").exists()]
        logging.info(
            "%s: %s downloads.",
            root["group_id"],
            f"{len(pending):,}"
        )

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

        for uid, item in entries.items():
            path = raw / f"{uid}.replay"

            if not path.is_file():
                continue

            sha = file_sha256(path)
            previous = records.get(uid, {})

            if (
                previous.get("raw_sha256", sha)
                != sha
            ):
                raise ValueError(f"raw replay checksum changed: {uid}")

            old_provenance = previous.get("provenance", {})
            occurrences = list(old_provenance.get("occurrences", []))

            for occurrence in item["occurrences"]:
                if occurrence not in occurrences:
                    occurrences.append(occurrence)

            provenance = {
                "id":          uid,
                "metadata":    item["metadata"],
                "occurrences": occurrences,
            }

            records[uid] = {
                **previous,
                **root.get("metadata", {}),
                "id":         uid,
                "raw":        str(path),
                "raw_sha256": sha,
                "team_size":  config["team_size"],
                "provenance": provenance,
            }

        write_json(
            manifest_path,
            {
                "config":  config,
                "records": sorted(
                    records.values(),
                    key=lambda record: record["id"]
                ),
            }
        )

    write_json(output / "provenance/download-errors.json", errors)

    if errors:
        raise RuntimeError(f"{len(errors)} downloads failed; see provenance/download-errors.json")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)

    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()
    download(json.loads(args.config.read_text()), args.output_dir, args.workers)


if __name__ == "__main__":
    main()
