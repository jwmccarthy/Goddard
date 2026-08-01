import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import TextIO
from urllib.parse import urljoin, urlparse

import numpy as np
import requests


BASE_URL = "https://ballchasing.com"
USER_AGENT = "babytowniv-rl-dataset/1.0"
MAX_REQUEST_RATE = 5.0
SHARD_SCHEMA_VERSION = 3
GOAL_EXCLUSION_SECONDS = 5.0
RESET_EVENT_OFFSET_SECONDS = 3.0
RESET_EVENT_WINDOW_SECONDS = 0.5
RESET_SAMPLES_PER_REPLAY = 480
RESET_HISTORY_LENGTH = 16
REPLAY_PATH = re.compile(
    r"^/dl/replay/"
    r"(?P<id>[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})$",
    re.IGNORECASE,
)
GLOBAL_FEATURES = [
    "BallRigidBodyQuaternionVelocities",
    "FrameTime",
    "ReplicatedStateName",
]
PLAYER_FEATURES = [
    "PlayerRigidBodyQuaternionVelocities",
    "PlayerBoost",
    "PlayerJump",
    "PlayerDemolishedBy",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "ballchasing.com":
        raise ValueError(f"Unexpected ballchasing URL: {url}")
    return url


def search_url(min_rank: int, max_rank: int) -> str:
    return (
        f"{BASE_URL}/?playlist=10&min-rank={min_rank}&max-rank={max_rank}"
        "&sort-by=replay-date&sort-dir=desc"
    )


@dataclass(frozen=True)
class ReplayLink:
    replay_id:     str
    download_path: str


@dataclass(frozen=True)
class SearchPage:
    replays:  tuple[ReplayLink, ...]
    next_url: str | None


class SearchPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.replays: list[ReplayLink] = []
        self.next_href: str | None = None

    def handle_starttag(
        self,
        tag:   str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        if attributes.get("data-trev") == "replay-download":
            path = attributes.get("data-post-url")
            match = REPLAY_PATH.fullmatch(path or "")
            if match:
                self.replays.append(ReplayLink(match["id"].lower(), path or ""))

        classes = (attributes.get("class") or "").split()
        if "pagination-next" in classes and attributes.get("href"):
            self.next_href = attributes["href"]


def parse_search_page(
    html:        str,
    current_url: str,
) -> SearchPage:
    parser = SearchPageParser()
    parser.feed(html)

    next_url = None
    if parser.next_href is not None:
        next_url = validate_url(urljoin(current_url, parser.next_href))
    return SearchPage(tuple(parser.replays), next_url)


class RateLimiter:
    def __init__(
        self,
        requests_per_second: float,
        path:                Path,
    ) -> None:
        if not 0 < requests_per_second <= MAX_REQUEST_RATE:
            raise ValueError(
                f"requests-per-second must be between zero and {MAX_REQUEST_RATE:g}"
            )
        self._interval = 1.0 / requests_per_second
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            with self._path.open("a+", encoding="ascii") as state:
                fcntl.flock(state, fcntl.LOCK_EX)
                next_request = self._read_deadline(state)
                delay = max(0.0, next_request - time.time())
                if delay:
                    time.sleep(delay)

                self._write_deadline(state, time.time() + self._interval)

    def defer(self, delay: float) -> None:
        deadline = time.time() + max(0.0, delay)
        with self._lock:
            with self._path.open("a+", encoding="ascii") as state:
                fcntl.flock(state, fcntl.LOCK_EX)
                current = self._read_deadline(state)
                self._write_deadline(state, max(current, deadline))

    @staticmethod
    def _read_deadline(state: TextIO) -> float:
        state.seek(0)
        try:
            return float(state.read() or 0.0)
        except ValueError:
            return 0.0

    @staticmethod
    def _write_deadline(
        state:    TextIO,
        deadline: float,
    ) -> None:
        state.seek(0)
        state.truncate()
        state.write(str(deadline))
        state.flush()


class BallchasingClient:
    def __init__(
        self,
        requests_per_second: float,
        retries:             int,
        timeout:             float,
        rate_limit_path:     Path,
    ) -> None:
        if retries < 0:
            raise ValueError("retries cannot be negative")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.rate_limiter = RateLimiter(requests_per_second, rate_limit_path)
        self.retries = retries
        self.timeout = timeout
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
            self._local.session = session
        return session

    def request(
        self,
        method: str,
        url:    str,
        **kwargs,
    ) -> requests.Response:
        validate_url(url)
        retry_statuses = {429, 500, 502, 503, 504}
        for attempt in range(self.retries + 1):
            delay = min(30.0, 2.0**attempt) + random.uniform(0.0, 0.25)
            self.rate_limiter.wait()
            try:
                response = self._session().request(
                    method,
                    url,
                    timeout=self.timeout,
                    allow_redirects=False,
                    **kwargs,
                )
            except requests.RequestException:
                if attempt == self.retries:
                    raise
            else:
                if response.status_code not in retry_statuses:
                    return response
                if attempt == self.retries:
                    return response
                retry_after = response.headers.get("Retry-After")
                response.close()
                if retry_after is not None:
                    try:
                        delay = parse_retry_after(retry_after)
                    except (TypeError, ValueError, OverflowError):
                        pass
                if response.status_code == 429 or retry_after is not None:
                    self.rate_limiter.defer(delay)
                    continue
            time.sleep(delay)
        raise RuntimeError("Request retry loop exited unexpectedly")


def parse_retry_after(value: str) -> float:
    try:
        return max(0.0, float(value))
    except ValueError:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, retry_at.timestamp() - time.time())


def rate_limit_path() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache / "goddard" / "ballchasing-rate-limit"


class ReplayManifest:
    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS replays (
                replay_id TEXT PRIMARY KEY,
                download_path TEXT NOT NULL,
                source_url TEXT NOT NULL,
                status TEXT NOT NULL,
                size INTEGER,
                sha256 TEXT,
                frame_count INTEGER,
                error TEXT,
                discovered_at TEXT NOT NULL,
                downloaded_at TEXT,
                parsed_at TEXT
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def add(
        self,
        replay:     ReplayLink,
        source_url: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO replays (
                replay_id, download_path, source_url, status, discovered_at
            ) VALUES (?, ?, ?, 'discovered', ?)
            ON CONFLICT(replay_id) DO UPDATE SET
                download_path = excluded.download_path,
                source_url = excluded.source_url
            """,
            (replay.replay_id, replay.download_path, source_url, utc_now()),
        )

    def commit(self) -> None:
        self.connection.commit()

    def mark_downloaded(self, result: "DownloadResult") -> None:
        self.connection.execute(
            """
            UPDATE replays SET
                status = 'downloaded', size = ?, sha256 = ?, error = NULL,
                downloaded_at = ?
            WHERE replay_id = ?
            """,
            (result.size, result.sha256, utc_now(), result.replay_id),
        )
        self.connection.commit()

    def mark_failed(
        self,
        replay_id: str,
        error:     BaseException,
    ) -> None:
        self.connection.execute(
            "UPDATE replays SET status = 'failed', error = ? WHERE replay_id = ?",
            (str(error), replay_id),
        )
        self.connection.commit()

    def mark_parsed(self, result: "ParseResult") -> None:
        cursor = self.connection.execute(
            """
            UPDATE replays SET
                status = 'parsed', frame_count = ?, error = NULL, parsed_at = ?
            WHERE replay_id = ?
            """,
            (result.frame_count, utc_now(), result.replay_id),
        )
        if cursor.rowcount != 1:
            raise ValueError(
                f"Replay is not tracked in the manifest: {result.replay_id}"
            )
        self.connection.commit()

    def mark_parse_failed(
        self,
        replay_id: str,
        error:     BaseException,
    ) -> None:
        self.connection.execute(
            """
            UPDATE replays SET status = 'parse_failed', error = ?
            WHERE replay_id = ?
            """,
            (str(error), replay_id),
        )
        self.connection.commit()

    def parsed_replay_ids(self) -> list[str]:
        rows = self.connection.execute(
            """
            SELECT replay_id FROM replays
            WHERE status = 'parsed'
            ORDER BY replay_id
            """
        )
        return [row[0] for row in rows]

    def expected_download(self, replay_id: str) -> tuple[int, str] | None:
        row = self.connection.execute(
            "SELECT size, sha256 FROM replays WHERE replay_id = ?",
            (replay_id,),
        ).fetchone()
        if row is None or row[0] is None or row[1] is None:
            return None
        return int(row[0]), str(row[1])


@dataclass(frozen=True)
class DownloadResult:
    replay_id: str
    size:      int
    sha256:    str


def discover_replays(
    client:      BallchasingClient,
    manifest:    ReplayManifest,
    count:       int,
    initial_url: str,
) -> list[ReplayLink]:
    if count <= 0:
        raise ValueError("count must be positive")

    current_url: str | None = initial_url
    visited_pages: set[str] = set()
    discovered: dict[str, ReplayLink] = {}
    while current_url is not None and len(discovered) < count:
        if current_url in visited_pages:
            raise RuntimeError("Ballchasing pagination loop detected")
        visited_pages.add(current_url)
        response = client.request("GET", current_url)
        response.raise_for_status()
        page = parse_search_page(response.text, current_url)
        if not page.replays:
            raise RuntimeError(f"No replay links found on {current_url}")
        for replay in page.replays:
            if replay.replay_id in discovered:
                continue
            discovered[replay.replay_id] = replay
            manifest.add(replay, current_url)
            if len(discovered) == count:
                break
        manifest.commit()
        current_url = page.next_url
    if len(discovered) < count:
        raise RuntimeError(
            f"Requested {count} replays but discovered only {len(discovered)}"
        )
    return list(discovered.values())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_is_valid(
    manifest:  ReplayManifest,
    replay_id: str,
    path:      Path,
) -> bool:
    expected = manifest.expected_download(replay_id)
    if expected is None or not path.is_file():
        return False
    expected_size, expected_hash = expected
    return _file_matches(path, expected_size, expected_hash)


def _file_matches(
    path:          Path,
    expected_size: int,
    expected_hash: str,
) -> bool:
    return path.stat().st_size == expected_size and file_sha256(path) == expected_hash


def download_replay(
    client:       BallchasingClient,
    replay:       ReplayLink,
    directory:    Path,
    maximum_size: int,
) -> DownloadResult:
    if maximum_size <= 0:
        raise ValueError("maximum-size must be positive")

    output = directory / f"{replay.replay_id}.replay"
    response = client.request(
        "POST",
        urljoin(BASE_URL, replay.download_path),
        data=b"",
        stream=True,
    )
    try:
        response.raise_for_status()
        media_type = response.headers.get("Content-Type", "").split(";", 1)[0]
        if media_type != "application/octet-stream":
            raise RuntimeError(f"Unexpected replay content type: {media_type}")
        declared_size = int(response.headers.get("Content-Length", "0") or 0)
        if declared_size > maximum_size:
            raise RuntimeError(f"Replay exceeds {maximum_size} bytes")

        digest = hashlib.sha256()
        size = 0
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=directory,
                suffix=".part",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                for chunk in response.iter_content(64 * 1024):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > maximum_size:
                        raise RuntimeError(f"Replay exceeds {maximum_size} bytes")
                    digest.update(chunk)
                    temporary.write(chunk)
            if declared_size and size != declared_size:
                raise RuntimeError(
                    f"Expected {declared_size} replay bytes but received {size}"
                )
            if size == 0:
                raise RuntimeError("Replay response was empty")
            os.replace(temporary_path, output)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
    finally:
        response.close()
    return DownloadResult(replay.replay_id, size, digest.hexdigest())


def acquire_replays(arguments: argparse.Namespace) -> None:
    if arguments.min_rank > arguments.max_rank:
        raise ValueError("min-rank cannot exceed max-rank")
    output = arguments.output
    replay_directory = output / "replays"
    output.mkdir(parents=True, exist_ok=True)
    replay_directory.mkdir(exist_ok=True)

    client = BallchasingClient(
        arguments.requests_per_second,
        arguments.retries,
        arguments.timeout,
        rate_limit_path(),
    )
    manifest = ReplayManifest(output / "manifest.sqlite3")
    try:
        replays = discover_replays(
            client,
            manifest,
            arguments.count,
            search_url(arguments.min_rank, arguments.max_rank),
        )
        collection = {
            "playlist": 10,
            "min_rank": arguments.min_rank,
            "max_rank": arguments.max_rank,
        }
        (output / "collection.json").write_text(
            json.dumps(collection, indent=2) + "\n"
        )
        pending = [
            replay
            for replay in replays
            if not download_is_valid(
                manifest,
                replay.replay_id,
                replay_directory / f"{replay.replay_id}.replay",
            )
        ]
        _download_pending(
            client=client,
            manifest=manifest,
            replays=replays,
            pending=pending,
            directory=replay_directory,
            workers=arguments.workers,
            maximum_size=arguments.maximum_size,
        )
    finally:
        manifest.close()


def _download_pending(
    client:       BallchasingClient,
    manifest:     ReplayManifest,
    replays:      list[ReplayLink],
    pending:      list[ReplayLink],
    directory:    Path,
    workers:      int,
    maximum_size: int,
) -> None:
    print(f"Discovered {len(replays)} replays; downloading {len(pending)}")
    executor = ThreadPoolExecutor(max_workers=workers)
    failures = 0

    try:
        futures = {
            executor.submit(
                download_replay,
                client,
                replay,
                directory,
                maximum_size,
            ): replay.replay_id
            for replay in pending
        }
        completed = len(replays) - len(pending)

        for future in as_completed(futures):
            replay_id = futures[future]
            try:
                result = future.result()
            except BaseException as error:
                manifest.mark_failed(replay_id, error)
                failures += 1
                print(f"Failed {replay_id}: {error}")
            else:
                manifest.mark_downloaded(result)
                completed += 1
                print(f"Downloaded {completed}/{len(replays)}: {replay_id}")
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown()

    if failures:
        raise RuntimeError(f"Failed to download {failures} replays")


@dataclass(frozen=True)
class ParseResult:
    replay_id:   str
    frame_count: int


def replay_columns() -> list[str]:
    import subtr_actor

    headers = subtr_actor.get_column_headers(
        global_feature_adders=GLOBAL_FEATURES,
        player_feature_adders=PLAYER_FEATURES,
    )
    global_headers = list(headers["global_headers"])
    player_headers = list(headers["player_headers"])
    return global_headers + [
        f"player {index} - {header}" for index in range(2) for header in player_headers
    ]


def parse_replay_file(
    replay_path:     Path,
    frame_directory: Path,
    fps:             float,
    source_sha256:   str,
) -> ParseResult:
    import subtr_actor

    replay_id = replay_path.stem
    metadata, frames = subtr_actor.get_ndarray_with_info_from_replay_filepath(
        str(replay_path),
        global_feature_adders=GLOBAL_FEATURES,
        player_feature_adders=PLAYER_FEATURES,
        fps=fps,
        dtype="float32",
    )
    replay_metadata = metadata["replay_meta"]
    blue_players = replay_metadata["team_zero"]
    orange_players = replay_metadata["team_one"]
    if len(blue_players) != 1 or len(orange_players) != 1:
        raise ValueError(
            "Expected one player on each team, found "
            f"{len(blue_players)} blue and {len(orange_players)} orange"
        )

    columns = replay_columns()
    _validate_frame_matrix(frames, columns)
    replay_data = subtr_actor.get_replay_frames_data(str(replay_path))
    goal_times = [event["time"] for event in replay_data["goal_events"]]
    frames = _select_live_gameplay(frames, columns, goal_times)
    frames = _exclude_pre_goal_frames(frames, columns, goal_times)
    _write_parsed_shard(
        frame_directory / f"{replay_id}.npz",
        frames=frames,
        columns=columns,
        replay_id=replay_id,
        fps=fps,
        source_sha256=source_sha256,
    )
    return ParseResult(replay_id, len(frames))


def _exclude_pre_goal_frames(
    frames:     np.ndarray,
    columns:    list[str],
    goal_times: list[float],
) -> np.ndarray:
    return frames[_pre_goal_mask(frames, columns, goal_times)]


def _pre_goal_mask(
    frames:     np.ndarray,
    columns:    list[str],
    goal_times: list[float],
) -> np.ndarray:
    frame_times = frames[:, columns.index("frame time")]
    keep = np.ones(len(frames), dtype=np.bool_)
    for goal_time in goal_times:
        near_goal = frame_times > goal_time - GOAL_EXCLUSION_SECONDS
        near_goal &= frame_times <= goal_time
        keep &= ~near_goal
    return keep


def _select_live_gameplay(
    frames:     np.ndarray,
    columns:    list[str],
    goal_times: list[float],
) -> np.ndarray:
    return frames[_live_gameplay_mask(frames, columns, goal_times)]


def _reset_candidate_mask(
    frames:  np.ndarray,
    columns: list[str],
    sample_count: int = RESET_SAMPLES_PER_REPLAY,
) -> np.ndarray:
    """Select live-play frames near kickoff and goal transitions."""
    if not len(frames):
        return np.zeros(0, dtype=np.bool_)
    live = _live_gameplay_mask(frames, columns, [])
    candidates = np.flatnonzero(live)
    if len(candidates) <= sample_count:
        return live

    # Stratify by ball height, car height, and field half so ordinary grounded
    # kickoff frames cannot crowd out aerial, wall, and recovery states.
    ball_z = frames[:, columns.index("Ball - position z")]
    car_z = frames[:, columns.index("player 0 - position z")]
    ball_y = frames[:, columns.index("Ball - position y")]
    ball_x = frames[:, columns.index("Ball - position x")]
    car0_x = frames[:, columns.index("player 0 - position x")]
    car0_y = frames[:, columns.index("player 0 - position y")]
    car0_z = frames[:, columns.index("player 0 - position z")]
    car1_x = frames[:, columns.index("player 1 - position x")]
    car1_y = frames[:, columns.index("player 1 - position y")]
    car1_z = frames[:, columns.index("player 1 - position z")]
    kickoff_like = (
        np.hypot(ball_x[candidates], ball_y[candidates]) < 10.0
    ) & (np.hypot(car0_x[candidates], car0_y[candidates]) > 2000.0) & (
        np.hypot(car1_x[candidates], car1_y[candidates]) > 2000.0
    )
    kickoff_candidates = candidates[kickoff_like]
    candidates = candidates[~kickoff_like]
    ball_airborne = ball_z[candidates] > 2.0 * 91.25
    car_airborne = (car0_z[candidates] > 1.5 * 91.25) | (
        car1_z[candidates] > 1.5 * 91.25
    )
    ball_near_car = (
        np.minimum(
            np.hypot(ball_x[candidates] - car0_x[candidates], ball_y[candidates] - car0_y[candidates]),
            np.hypot(ball_x[candidates] - car1_x[candidates], ball_y[candidates] - car1_y[candidates]),
        ) < 1400.0
    )
    car_on_wall = (
        (np.abs(car0_x[candidates]) > 3800.0)
        | (np.abs(car1_x[candidates]) > 3800.0)
        | (np.abs(car0_y[candidates]) > 4800.0)
        | (np.abs(car1_y[candidates]) > 4800.0)
    )
    aerial_like = ball_airborne & ball_near_car & (car_airborne | car_on_wall)
    aerial_candidates = candidates[aerial_like]
    candidates = candidates[~aerial_like]
    buckets = (
        (ball_z[candidates] > 180.0).astype(np.int32)
        + 2 * (car_z[candidates] > 100.0).astype(np.int32)
        + 4 * (np.abs(ball_y[candidates]) > 2560.0).astype(np.int32)
    )
    selected = []
    aerial_quota = max(1, sample_count // 4)
    if len(aerial_candidates):
        selected.append(
            aerial_candidates[
                np.linspace(
                    0,
                    len(aerial_candidates) - 1,
                    min(aerial_quota, len(aerial_candidates)),
                    dtype=int,
                )
            ]
        )
    quota = max(1, (sample_count - aerial_quota) // 8)
    for bucket in range(8):
        members = candidates[buckets == bucket]
        if len(members):
            selected.append(members[np.linspace(0, len(members) - 1, min(quota, len(members)), dtype=int)])
    kickoff_quota = max(1, sample_count // 50)
    if len(kickoff_candidates):
        selected.append(
            kickoff_candidates[
                np.linspace(
                    0,
                    len(kickoff_candidates) - 1,
                    min(kickoff_quota, len(kickoff_candidates)),
                    dtype=int,
                )
            ]
        )
    selected = np.concatenate(selected) if selected else candidates
    if len(selected) < sample_count:
        remaining = np.setdiff1d(candidates, selected, assume_unique=False)
        count = min(sample_count - len(selected), len(remaining))
        selected = np.concatenate(
            (selected, remaining[np.linspace(0, len(remaining) - 1, count, dtype=int)])
        )
    mask = np.zeros(len(frames), dtype=np.bool_)
    mask[selected[:sample_count]] = True
    return mask


def _live_gameplay_mask(
    frames:     np.ndarray,
    columns:    list[str],
    goal_times: list[float],
) -> np.ndarray:
    frame_times = frames[:, columns.index("frame time")]
    game_state = frames[:, columns.index("game state")]
    candidates = []

    for goal_time in goal_times:
        index = np.searchsorted(frame_times, goal_time, side="left") - 1
        if index >= 0:
            candidates.append(game_state[index])

    if candidates:
        states, counts = np.unique(candidates, return_counts=True)
    else:
        states, counts = np.unique(game_state, return_counts=True)
    live_state = states[counts.argmax()]
    return game_state == live_state


def _validate_frame_matrix(
    frames:  np.ndarray,
    columns: list[str],
) -> None:
    if frames.dtype != np.float32 or frames.ndim != 2:
        raise ValueError("Replay frames must be a two-dimensional float32 array")
    if frames.shape[1] != len(columns):
        raise ValueError(f"Expected {len(columns)} columns, found {frames.shape[1]}")
    if not np.isfinite(frames).all():
        raise ValueError("Replay frames contain non-finite values")


def _write_parsed_shard(
    output:        Path,
    frames:        np.ndarray,
    columns:       list[str],
    replay_id:     str,
    fps:           float,
    source_sha256: str,
) -> None:
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                frames=frames,
                columns=np.asarray(columns, dtype=np.str_),
                metadata=np.asarray(
                    json.dumps(
                        {
                            "replay_id":  replay_id,
                            "team_order": ["blue", "orange"],
                        }
                    ),
                    dtype=np.str_,
                ),
                fps=np.asarray(fps, dtype=np.float64),
                schema_version=np.asarray(SHARD_SCHEMA_VERSION, dtype=np.int64),
                source_sha256=np.asarray(source_sha256, dtype=np.str_),
            )
        os.replace(temporary_path, output)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def valid_parsed_shard(
    path:          Path,
    fps:           float,
    source_sha256: str,
) -> int | None:
    try:
        with np.load(path, allow_pickle=False) as shard:
            frames = shard["frames"]
            columns = shard["columns"].tolist()
            shard_fps = float(shard["fps"])
            if (
                columns != replay_columns()
                or shard_fps != fps
                or int(shard["schema_version"]) != SHARD_SCHEMA_VERSION
                or str(shard["source_sha256"]) != source_sha256
            ):
                return None
            _validate_frame_matrix(frames, columns)
            return len(frames)
    except (OSError, KeyError, ValueError):
        return None


def build_dataset(
    output:       Path,
    replay_ids:   list[str],
    dataset_name: str = "dataset",
    reset_samples_per_replay: int = RESET_SAMPLES_PER_REPLAY,
) -> None:
    if not replay_ids:
        raise ValueError("No parsed replays are available to build")

    frame_directory = output / "frames"
    dataset_root = output / dataset_name
    dataset_root.mkdir(exist_ok=True)

    reset_only = dataset_name == "reset_dataset"
    shards, columns, fps = _inspect_shards(
        frame_directory, replay_ids, reset_only=reset_only,
        reset_samples_per_replay=reset_samples_per_replay,
    )
    source = _dataset_source(output)
    generation_directory = _write_dataset_generation(
        dataset_root,
        frame_directory,
        shards,
        columns,
        fps,
        source,
        reset_only=reset_only,
        reset_samples_per_replay=reset_samples_per_replay,
    )
    _publish_generation(dataset_root, generation_directory.name)
    print(
        f"Built {sum(count for _, count in shards)} frames from "
        f"{len(shards)} replays in {generation_directory}"
    )


def _inspect_shards(
    frame_directory: Path,
    replay_ids:      list[str],
    *,
    reset_only:      bool = False,
    reset_samples_per_replay: int = RESET_SAMPLES_PER_REPLAY,
) -> tuple[list[tuple[str, int]], list[str], float]:
    expected_columns = replay_columns()
    shards = []
    fps: float | None = None

    for replay_id in replay_ids:
        with np.load(frame_directory / f"{replay_id}.npz", allow_pickle=False) as shard:
            shard_columns = shard["columns"].tolist()
            shard_fps = float(shard["fps"])
            if not math.isfinite(shard_fps) or shard_fps <= 0:
                raise ValueError(f"Incompatible parsed shard: {replay_id}")
            if shard_columns != expected_columns:
                raise ValueError(f"Incompatible parsed shard: {replay_id}")
            if int(shard["schema_version"]) != SHARD_SCHEMA_VERSION:
                raise ValueError(f"Incompatible parsed shard: {replay_id}")

            frames = shard["frames"]
            _validate_frame_matrix(frames, shard_columns)
            if fps is None:
                fps = shard_fps
            elif fps != shard_fps:
                raise ValueError(f"Incompatible parsed shard: {replay_id}")
            count = (
                int(_reset_candidate_mask(
                    frames, shard_columns, reset_samples_per_replay
                ).sum())
                if reset_only
                else len(frames)
            )
            if count:
                shards.append((replay_id, count))

    total_frames = sum(frame_count for _, frame_count in shards)
    if total_frames == 0 or fps is None:
        raise ValueError("Parsed replay dataset is empty")
    return shards, expected_columns, fps


def _write_dataset_generation(
    dataset_root:    Path,
    frame_directory: Path,
    shards:          list[tuple[str, int]],
    columns:         list[str],
    fps:             float,
    source:          str,
    reset_only:      bool = False,
    reset_samples_per_replay: int = RESET_SAMPLES_PER_REPLAY,
) -> Path:
    total_frames = sum(frame_count for _, frame_count in shards)
    generation = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    generation_directory = dataset_root / generation
    generation_directory.mkdir()

    try:
        frames = np.lib.format.open_memmap(
            generation_directory / "frames.npy",
            mode="w+",
            dtype=np.float32,
            shape=(total_frames, len(columns)),
        )
        replay_index = np.lib.format.open_memmap(
            generation_directory / "replay_index.npy",
            mode="w+",
            dtype=np.int32,
            shape=(total_frames,),
        )
        histories = None
        history_valid = None
        if reset_only:
            histories = np.lib.format.open_memmap(
                generation_directory / "histories.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total_frames, RESET_HISTORY_LENGTH, len(columns)),
            )
            history_valid = np.lib.format.open_memmap(
                generation_directory / "history_valid.npy",
                mode="w+",
                dtype=np.bool_,
                shape=(total_frames, RESET_HISTORY_LENGTH),
            )
        replay_records = []
        offset = 0

        for index, (replay_id, frame_count) in enumerate(shards):
            with np.load(
                frame_directory / f"{replay_id}.npz", allow_pickle=False
            ) as shard:
                source_frames = shard["frames"]
                selected = source_frames
                if reset_only:
                    selected_indices = np.flatnonzero(
                        _reset_candidate_mask(
                            source_frames, columns, reset_samples_per_replay
                        )
                    )
                    selected = source_frames[selected_indices]
                if len(selected) != frame_count:
                    raise ValueError(
                        f"Reset candidate count changed for replay {replay_id}"
                    )
                end = offset + frame_count
                frames[offset:end] = selected
                replay_index[offset:end] = index
                if histories is not None:
                    history_offsets = np.arange(-RESET_HISTORY_LENGTH, 0)
                    history_indices = np.maximum(
                        selected_indices[:, None] + history_offsets[None, :], 0
                    )
                    histories[offset:end] = source_frames[history_indices]
                    history_valid[offset:end] = (
                        selected_indices[:, None] + history_offsets[None, :]
                    ) >= 0
            replay_records.append(
                {
                    "replay_id": replay_id,
                    "start":     offset,
                    "stop":      end,
                }
            )
            offset = end

        frames.flush()
        replay_index.flush()
        if histories is not None:
            histories.flush()
            history_valid.flush()
        del frames
        del replay_index
        del histories
        del history_valid

        metadata = {
            "schema_version": SHARD_SCHEMA_VERSION,
            "created_at":     utc_now(),
            "source":         source,
            "fps":            fps,
            "frame_count":    total_frames,
            "history_length": RESET_HISTORY_LENGTH if reset_only else 0,
            "columns":        columns,
            "replays":        replay_records,
        }
        with (generation_directory / "metadata.json").open(
            "w", encoding="utf-8"
        ) as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
            metadata_file.write("\n")
    except BaseException:
        shutil.rmtree(generation_directory)
        raise
    return generation_directory


def _dataset_source(output: Path) -> str:
    collection_path = output / "collection.json"
    if not collection_path.is_file():
        return "ballchasing.com SSL game average ranked duels"
    collection = json.loads(collection_path.read_text())
    return (
        "ballchasing.com ranked duels with game average rank "
        f"{collection['min_rank']} through {collection['max_rank']}"
    )


def _publish_generation(
    dataset_root: Path,
    generation:   str,
) -> None:
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            dir=dataset_root,
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(f"{generation}\n")
        os.replace(temporary_path, dataset_root / "CURRENT")
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def parse_replays(arguments: argparse.Namespace) -> None:
    if arguments.fps <= 0:
        raise ValueError("fps must be positive")

    output = arguments.output
    replay_directory = output / "replays"
    frame_directory = output / "frames"
    if not replay_directory.is_dir():
        raise ValueError(f"Replay directory does not exist: {replay_directory}")
    frame_directory.mkdir(exist_ok=True)

    manifest = ReplayManifest(output / "manifest.sqlite3")
    try:
        replay_paths = sorted(replay_directory.glob("*.replay"))
        if arguments.limit is not None:
            replay_paths = replay_paths[: arguments.limit]

        pending, validated_replay_ids = _classify_replays(
            replay_paths,
            frame_directory,
            manifest,
            arguments.fps,
        )
        _parse_pending(
            pending=pending,
            frame_directory=frame_directory,
            manifest=manifest,
            validated_replay_ids=validated_replay_ids,
            total_replays=len(replay_paths),
            workers=arguments.workers,
            fps=arguments.fps,
        )

        if not arguments.skip_build:
            build_dataset(output, sorted(validated_replay_ids))
    finally:
        manifest.close()


def build_pro_reset_dataset(arguments: argparse.Namespace) -> None:
    replay_directory = arguments.source / "replays"
    frame_directory = arguments.output / "frames"
    frame_directory.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        replay_directory.glob("*.replay"),
        key=lambda path: hashlib.sha256(path.stem.encode()).digest(),
    )[: arguments.replays + max(64, arguments.replays // 4)]
    if len(candidates) < arguments.replays:
        raise ValueError("Not enough pro replay files are available")

    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                _try_parse_replay_file,
                path,
                frame_directory,
                arguments.fps,
                file_sha256(path),
            ): path.stem
            for path in candidates
        }
        replay_ids = []
        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, str):
                print(f"Skipped pro replay {futures[future]}: {result}")
                continue
            replay_ids.append(result.replay_id)
            print(f"Parsed pro replay {len(replay_ids)}/{arguments.replays}")

    if len(replay_ids) < arguments.replays:
        raise RuntimeError(
            f"Only {len(replay_ids)} of {arguments.replays} pro replays parsed"
        )

    build_dataset(
        arguments.output,
        sorted(replay_ids[: arguments.replays]),
        dataset_name="reset_dataset",
        reset_samples_per_replay=math.ceil(
            arguments.states / arguments.replays
        ),
    )


def _try_parse_replay_file(
    replay_path: Path,
    frame_directory: Path,
    fps: float,
    source_sha256: str,
) -> ParseResult | str:
    try:
        existing = valid_parsed_shard(
            frame_directory / f"{replay_path.stem}.npz",
            fps,
            source_sha256,
        )
        if existing is not None:
            return ParseResult(replay_path.stem, existing)
        return parse_replay_file(
            replay_path, frame_directory, fps, source_sha256
        )
    except BaseException as error:
        return f"{type(error).__name__}: {error}"


def _classify_replays(
    replay_paths:    list[Path],
    frame_directory: Path,
    manifest:        ReplayManifest,
    fps:             float,
) -> tuple[list[tuple[Path, str]], set[str]]:
    pending = []
    validated_replay_ids = set()

    for path in replay_paths:
        expected = manifest.expected_download(path.stem)
        if expected is None:
            raise ValueError(f"Replay is not tracked in the manifest: {path.stem}")

        expected_size, source_sha256 = expected
        if not _file_matches(path, expected_size, source_sha256):
            raise ValueError(f"Replay does not match its manifest hash: {path.stem}")

        shard = frame_directory / f"{path.stem}.npz"
        frame_count = valid_parsed_shard(shard, fps, source_sha256)
        if frame_count is None:
            pending.append((path, source_sha256))
        else:
            manifest.mark_parsed(ParseResult(path.stem, frame_count))
            validated_replay_ids.add(path.stem)

    print(f"Found {len(replay_paths)} replays; parsing {len(pending)}")
    return pending, validated_replay_ids


def _parse_pending(
    pending:              list[tuple[Path, str]],
    frame_directory:      Path,
    manifest:             ReplayManifest,
    validated_replay_ids: set[str],
    total_replays:        int,
    workers:              int,
    fps:                  float,
) -> None:
    executor = ProcessPoolExecutor(max_workers=workers)
    failures = 0

    try:
        futures = {
            executor.submit(
                parse_replay_file,
                path,
                frame_directory,
                fps,
                source_sha256,
            ): path.stem
            for path, source_sha256 in pending
        }
        completed = total_replays - len(pending)

        for future in as_completed(futures):
            replay_id = futures[future]
            try:
                result = future.result()
            except BaseException as error:
                manifest.mark_parse_failed(replay_id, error)
                failures += 1
                print(f"Failed to parse {replay_id}: {error}")
            else:
                manifest.mark_parsed(result)
                validated_replay_ids.add(result.replay_id)
                completed += 1
                print(f"Parsed {completed}/{total_replays}: {replay_id}")
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown()

    if failures:
        raise RuntimeError(f"Failed to parse {failures} replays")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def request_rate(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= MAX_REQUEST_RATE:
        raise argparse.ArgumentTypeError(
            f"must be between zero and {MAX_REQUEST_RATE:g}"
        )
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an SSL ranked-duels replay dataset from ballchasing.com"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    acquire = subparsers.add_parser("acquire", help="discover and download replays")
    acquire.add_argument(
        "--output", type=Path, default=Path("data/ballchasing-ssl-1v1")
    )
    acquire.add_argument("--count", type=positive_int, default=1_000)
    acquire.add_argument("--min-rank", type=nonnegative_int, default=22)
    acquire.add_argument("--max-rank", type=nonnegative_int, default=22)
    acquire.add_argument("--workers", type=positive_int, default=4)
    acquire.add_argument(
        "--requests-per-second",
        type=request_rate,
        default=MAX_REQUEST_RATE,
    )
    acquire.add_argument("--retries", type=nonnegative_int, default=5)
    acquire.add_argument("--timeout", type=positive_float, default=60.0)
    acquire.add_argument("--maximum-size", type=positive_int, default=20 * 1024 * 1024)
    acquire.set_defaults(func=acquire_replays)

    parse = subparsers.add_parser("parse", help="parse downloaded replay frames")
    parse.add_argument("--output", type=Path, default=Path("data/ballchasing-ssl-1v1"))
    parse.add_argument("--fps", type=positive_float, default=10.0)
    parse.add_argument("--workers", type=positive_int, default=4)
    parse.add_argument("--limit", type=positive_int)
    parse.add_argument("--skip-build", action="store_true")
    parse.set_defaults(func=parse_replays)
    pro_reset = subparsers.add_parser(
        "pro-reset", help="build a stratified reset dataset from pro replays"
    )
    pro_reset.add_argument("--source", type=Path, default=Path("data/pro-1v1"))
    pro_reset.add_argument(
        "--output", type=Path, default=Path("data/pro-1v1-reset")
    )
    pro_reset.add_argument("--replays", type=positive_int, default=4096)
    pro_reset.add_argument("--states", type=positive_int, default=256_000)
    pro_reset.add_argument("--workers", type=positive_int, default=8)
    pro_reset.add_argument("--fps", type=positive_float, default=10.0)
    pro_reset.set_defaults(func=build_pro_reset_dataset)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    arguments.func(arguments)


if __name__ == "__main__":
    main()
