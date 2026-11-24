"""Match winners and player teams for discriminator expert selection."""

import json
import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from rlgym_tools.rocket_league.replays import parsed_replay

from replays.utils import file_sha256


def winning_team(metadata: dict) -> int | None:
    game = metadata.get("game", {})
    scores = (game.get("team_0_score"), game.get("team_1_score"))

    if any(type(score) is not int or score < 0 for score in scores):
        return None

    if scores[0] == scores[1]:
        return None

    return int(scores[1] > scores[0])


def player_results(metadata: dict) -> dict[str, bool]:
    winner = winning_team(metadata)

    if winner is None:
        return {}

    results = {}

    for player in metadata.get("players", []):
        identity = player.get("unique_id")
        orange = player.get("is_orange")

        if identity is None or type(orange) is not bool:
            continue

        results[str(identity)] = int(orange) == winner

    return results


def replay_result(root: Path, identity: str) -> dict[str, bool]:
    raw = root / "raw" / f"{identity}.replay"

    if not raw.is_file():
        return {}

    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = cache_dir / "goddard" / "replay-results" / f"{file_sha256(raw)}.json"

    cached = read_cache(cache)

    if cached is not None:
        return cached

    results = player_results(decode_metadata(raw))
    write_cache(cache, results)

    return results


def read_cache(cache: Path) -> dict[str, bool] | None:
    if not cache.is_file():
        return None

    try:
        data = json.loads(cache.read_text())
    except (OSError, ValueError):
        return None

    if isinstance(data, dict) and all(
        isinstance(key, str) and type(value) is bool
        for key, value in data.items()
    ):
        return data

    return None


def decode_metadata(raw: Path) -> dict:
    with TemporaryDirectory(prefix="goddard-replay-") as temporary:
        subprocess.run(
            [
                parsed_replay._default_executable,
                "--skip-data-frames", "--skip-analysis", "parquet",
                "-i", str(raw), "-o", temporary
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=parsed_replay.ENV,
            timeout=60,
            check=True
        )

        return json.loads((Path(temporary) / "metadata.json").read_text())


def write_cache(cache: Path, results: dict[str, bool]) -> None:
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)

        with TemporaryDirectory(dir=cache.parent) as temporary:
            path = Path(temporary) / "result.json"
            path.write_text(json.dumps(results))
            path.replace(cache)
    except OSError as error:
        logging.warning("Could not cache replay result: %s", error)


def winning_perspectives(paths: list[Path], directory: Path) -> list[Path]:
    """Select only known match-winning ego perspectives, never reset data."""

    root = directory.parent if directory.name == "parsed" else directory
    selected = []
    pending: dict[str, list[tuple[Path, str]]] = {}
    losing = 0

    for path in paths:
        metadata = path.with_suffix(".json")
        values = json.loads(metadata.read_text()) if metadata.is_file() else {}

        if "ego_won" in values:
            if values["ego_won"] is True:
                selected.append(path)
            elif values["ego_won"] is False:
                losing += 1

            continue

        parts = path.stem.split("-", 2)

        if len(parts) != 3:
            continue

        pending.setdefault(parts[2], []).append((path, parts[0]))

    def load(identity: str) -> dict[str, bool]:
        try:
            return replay_result(root, identity)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            logging.warning(
                "Cannot determine replay result %s: %s", identity, error
            )

            return {}

    if pending:
        logging.info("Resolving winners for %s replays.", f"{len(pending):,}")

        with ThreadPoolExecutor(max_workers=4) as executor:
            for identity, results in zip(pending, executor.map(load, pending)):
                for path, ego in pending[identity]:
                    if results.get(ego) is True:
                        selected.append(path)
                    elif results.get(ego) is False:
                        losing += 1

    selected.sort()
    unknown = len(paths) - len(selected) - losing

    logging.info(
        "Expert perspectives: %s winning, %s losing, %s unknown.",
        f"{len(selected):,}", f"{losing:,}", f"{unknown:,}"
    )

    if unknown:
        logging.warning(
            "Excluded %s expert perspectives with tied or unknown results.",
            f"{unknown:,}"
        )

    return selected
