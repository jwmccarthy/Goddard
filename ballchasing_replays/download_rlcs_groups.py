import os
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

try:
    from ballchasing_replays.ballchasing_api import API_URL, BallchasingClient
except ModuleNotFoundError:
    from ballchasing_api import API_URL, BallchasingClient


UUID_PATTERN = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)
UNSAFE_COMPONENT = re.compile(r"[^a-z0-9]+")
REGIONS = (
    (re.compile(r"\b(?:eu|europe)\b", re.IGNORECASE), "EU"),
    (re.compile(r"\b(?:mena|middle east(?: and north africa)?)\b", re.IGNORECASE), "MENA"),
    (re.compile(r"\b(?:na|north america)\b", re.IGNORECASE), "NA"),
    (re.compile(r"\b(?:sam|south america)\b", re.IGNORECASE), "SAM"),
    (re.compile(r"\b(?:oce|oceania)\b", re.IGNORECASE), "OCE"),
    (re.compile(r"\b(?:apac|asia-pacific)\b", re.IGNORECASE), "APAC"),
    (re.compile(r"\b(?:ssa|sub-saharan africa)\b", re.IGNORECASE), "SSA"),
)


@dataclass(frozen=True)
class Root:
    group_id: str
    mode: str
    event: str
    region: str | None = None


@dataclass(frozen=True)
class ReplaySelection:
    replay: dict[str, Any]
    mode: str
    region: str
    event_path: tuple[str, ...]
    series: str
    leaf_id: str
    game_number: int


ROOTS = (
    Root("split-1-boston-major-5brncjgft3", "3v3", "Split 1 Boston Major"),
    Root("split-2-paris-major-otlj36qgqw", "3v3", "Split 2 Paris Major"),
    Root("2-playoffs-day-1-j9b7jsu3l8", "2v2", "EU Playoffs Day 1", "EU"),
    Root("3-playoffs-day-2-e0fx30bxur", "2v2", "EU Playoffs Day 2", "EU"),
    Root("2-playoffs-day-1-kb7fgd9l0o", "2v2", "MENA Playoffs Day 1", "MENA"),
    Root("3-playoffs-day-2-g57x9qfz42", "2v2", "MENA Playoffs Day 2", "MENA"),
    Root("2-playoffs-day-1-30tu03rgdn", "2v2", "NA Playoffs Day 1", "NA"),
    Root("3-playoffs-day-2-ligywgpjcy", "2v2", "NA Playoffs Day 2", "NA"),
)


def sanitize(value: Any, fallback: str = "unknown", limit: int = 48) -> str:
    component = UNSAFE_COMPONENT.sub(
        "-", "" if value is None else str(value).lower()
    ).strip("-")
    component = component[:limit].rstrip("-")
    return component or fallback


def replay_uuid(path: Path) -> str | None:
    match = UUID_PATTERN.search(path.stem)
    return match.group(1).lower() if match else None


def infer_region(path: tuple[str, ...]) -> str:
    text = " ".join(path)
    for pattern, region in REGIONS:
        if pattern.search(text):
            return region
    return "INTL"


def list_group_children(
    client: BallchasingClient, group_id: str
) -> list[dict[str, Any]]:
    page = client._get(f"{API_URL}/groups", group=group_id, count=200)
    children = list(page.replays)
    while page.next_url:
        page = client._get(page.next_url)
        children.extend(page.replays)
    return sorted(children, key=lambda group: (group.get("name", ""), group["id"]))


def descendant_leaves(
    client: BallchasingClient,
    root: Root,
) -> Iterator[tuple[dict[str, Any], tuple[str, ...]]]:
    visited = {root.group_id}

    def walk(group: dict[str, Any], path: tuple[str, ...]) -> Iterator[
        tuple[dict[str, Any], tuple[str, ...]]
    ]:
        group_id = group["id"]
        if group_id in visited:
            return
        visited.add(group_id)
        next_path = path + (group.get("name") or group_id,)
        children = list_group_children(client, group_id)
        if not children:
            if group.get("direct_replays", 0):
                yield group, next_path
            return
        for child in children:
            yield from walk(child, next_path)

    for child in list_group_children(client, root.group_id):
        yield from walk(child, (root.event,))


def select_replays(
    client: BallchasingClient,
) -> tuple[dict[str, ReplaySelection], dict[str, int], dict[str, set[str]]]:
    occurrences: list[ReplaySelection] = []
    discovered = {"3v3": 0, "2v2": 0}
    unique_by_mode = {"3v3": set(), "2v2": set()}

    for root in ROOTS:
        for leaf, path in descendant_leaves(client, root):
            entries = client.find_replay_entries(
                group=leaf["id"], count=200, sort_by="replay-date", sort_dir="asc"
            )
            entries.sort(key=lambda replay: (replay.get("date", ""), replay["id"].lower()))
            discovered[root.mode] += len(entries)
            for game_number, replay in enumerate(entries, 1):
                replay_id = replay["id"].lower()
                unique_by_mode[root.mode].add(replay_id)
                occurrences.append(
                    ReplaySelection(
                        replay=replay,
                        mode=root.mode,
                        region=root.region or infer_region(path),
                        event_path=path[:-1] or path,
                        series=path[-1],
                        leaf_id=leaf["id"],
                        game_number=game_number,
                    )
                )

    # A replay can appear below overlapping or accidentally duplicated groups.
    occurrences.sort(
        key=lambda item: (
            item.replay["id"].lower(),
            item.mode,
            item.region,
            item.event_path,
            item.series,
            item.leaf_id,
            item.game_number,
        )
    )
    selected: dict[str, ReplaySelection] = {}
    for occurrence in occurrences:
        selected.setdefault(occurrence.replay["id"].lower(), occurrence)
    return selected, discovered, unique_by_mode


def replay_filename(selection: ReplaySelection) -> str:
    replay = selection.replay
    replay_id = replay["id"].lower()
    date = str(replay.get("date", ""))[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        date = "unknown-date"
    event_path = "-".join(sanitize(part, limit=24) for part in selection.event_path)
    event_path = sanitize(event_path, "unknown-event", 50)
    series = sanitize(selection.series, "unknown-series", 36)
    blue = sanitize(replay.get("blue", {}).get("name"), "blue", 24)
    orange = sanitize(replay.get("orange", {}).get("name"), "orange", 24)
    matchup = f"{blue}-vs-{orange}"
    prefix = "__".join(
        (
            date,
            selection.mode,
            selection.region.lower(),
            event_path,
            series,
            matchup,
            f"g{selection.game_number:02d}",
        )
    )
    return f"{prefix}__{replay_id}.replay"


def rename_downloaded(
    replay_dir: Path, selections: dict[str, ReplaySelection]
) -> None:
    reserved = {path.name for path in replay_dir.glob("*.replay")}
    for replay_id in sorted(selections):
        source = replay_dir / f"{replay_id}.replay"
        if not source.exists():
            continue
        filename = replay_filename(selections[replay_id])
        target = replay_dir / filename
        collision = 2
        while target.name in reserved and target != source:
            stem = filename[: -len(f"__{replay_id}.replay")]
            target = replay_dir / f"{stem}__c{collision}__{replay_id}.replay"
            collision += 1
        source.rename(target)
        reserved.discard(source.name)
        reserved.add(target.name)


def main() -> None:
    replay_dir = Path(__file__).parent / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        replay_id
        for path in replay_dir.glob("*.replay")
        if (replay_id := replay_uuid(path)) is not None
    }

    client = BallchasingClient(os.environ["BALLCHASING_TOKEN"])
    selected, discovered, unique_by_mode = select_replays(client)
    new_ids = sorted(set(selected) - existing)
    new_by_mode = {mode: 0 for mode in discovered}
    for replay_id in new_ids:
        new_by_mode[selected[replay_id].mode] += 1

    for mode in ("3v3", "2v2"):
        print(
            f"{mode}: discovered={discovered[mode]} "
            f"deduped={len(unique_by_mode[mode])} new={new_by_mode[mode]}"
        )
    print(f"total: deduped={len(selected)} new={len(new_ids)}")

    try:
        client.download_replays(new_ids, replay_dir)
    finally:
        rename_downloaded(
            replay_dir,
            {replay_id: selected[replay_id] for replay_id in new_ids},
        )


if __name__ == "__main__":
    main()
