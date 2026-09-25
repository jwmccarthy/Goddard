"""Fetch the exact archived July 2026 replay IDs through Ballchasing's current API.

The July 31 ``replay_dataset.py acquire`` uses a web download endpoint that now
requires login. This helper changes only the download transport: it uses the
archived manifest's IDs and SHA-256 hashes, then leaves parsing and dataset
construction entirely to the unmodified July 31 ``replay_dataset.py``.

Pass the API token as one line on standard input; it is never stored or logged.
"""

import argparse
import hashlib
import os
import sqlite3
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


API_URL = "https://ballchasing.com/api/replays/{replay_id}/file"


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / requests_per_second
        self.lock = threading.Lock()
        self.next_request = time.monotonic()

    def wait(self) -> None:
        with self.lock:
            scheduled = max(time.monotonic(), self.next_request)
            self.next_request = scheduled + self.interval
        time.sleep(max(0.0, scheduled - time.monotonic()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(
    replay_id: str,
    expected_size: int,
    expected_hash: str,
    output: Path,
    token: str,
    limiter: RateLimiter,
) -> str:
    path = output / f"{replay_id}.replay"
    if path.is_file() and path.stat().st_size == expected_size:
        if sha256(path) == expected_hash:
            return "already verified"

    for attempt in range(6):
        limiter.wait()
        temporary_path = None
        try:
            with requests.get(
                API_URL.format(replay_id=replay_id),
                headers={"Authorization": token},
                timeout=60,
                stream=True,
            ) as response:
                if response.status_code in (429, 500, 502, 503, 504):
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        pause = float(retry_after)
                    except ValueError:
                        pause = min(30.0, 2.0**attempt)
                    time.sleep(max(0.0, pause))
                    continue
                response.raise_for_status()
                digest = hashlib.sha256()
                size = 0
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=output, suffix=".part", delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            size += len(chunk)
                            digest.update(chunk)
                            temporary.write(chunk)
                if size != expected_size or digest.hexdigest() != expected_hash:
                    raise ValueError(f"archived manifest hash/size mismatch for {replay_id}")
                os.replace(temporary_path, path)
                return "downloaded"
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    raise RuntimeError(f"Ballchasing retries exhausted for {replay_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    args = parser.parse_args()
    if args.workers < 1 or not 0 < args.requests_per_second <= 2:
        parser.error("workers must be positive and API download rate must be in (0, 2]")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if not args.output.is_dir():
        parser.error(f"replay output directory does not exist: {args.output}")

    token = sys.stdin.readline().strip()
    if not token:
        parser.error("provide the Ballchasing API token on standard input")

    with sqlite3.connect(f"file:{args.manifest}?mode=ro", uri=True) as manifest:
        entries = manifest.execute(
            "SELECT replay_id, size, sha256 FROM replays ORDER BY replay_id"
        ).fetchall()
    if args.limit is not None:
        entries = entries[:args.limit]
    if not entries or any(size is None or digest is None for _, size, digest in entries):
        raise ValueError("archived manifest has missing replay IDs or checksums")

    print(f"Downloading and verifying {len(entries)} original replay IDs", flush=True)
    limiter = RateLimiter(args.requests_per_second)
    failures = []
    counts = {"downloaded": 0, "already verified": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, replay_id, size, digest, args.output, token, limiter):
            replay_id
            for replay_id, size, digest in entries
        }
        for future in as_completed(futures):
            try:
                counts[future.result()] += 1
            except Exception as error:
                failures.append((futures[future], str(error)))
                print(f"FAILED {futures[future]}: {error}", flush=True)
            completed = sum(counts.values()) + len(failures)
            if completed % 25 == 0 or completed == len(entries):
                print(f"Checked {completed}/{len(entries)}: {counts}, failures={len(failures)}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} original replays could not be verified")


if __name__ == "__main__":
    main()
