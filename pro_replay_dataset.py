import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path

import requests
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn


API = "https://ballchasing.com/api"


def player_id(player: dict) -> str:
    identity = player.get("id") or {}
    return f"{identity.get('platform')}:{identity.get('id')}"


def pro_duel(replay: dict) -> bool:
    blue = replay.get("blue", {}).get("players", [])
    orange = replay.get("orange", {}).get("players", [])
    return len(blue) == len(orange) == 1


class BallchasingClient:
    def __init__(
        self,
        token: str,
        requests_per_second: float,
        name: str,
    ) -> None:
        self.headers = {"Authorization": token}
        self.interval = 1.0 / requests_per_second
        self.name = name
        self.next_request = 0.0
        self.lock = threading.Lock()
        self.local = threading.local()

    def get(self, url: str, **kwargs) -> requests.Response:
        for attempt in range(6):
            with self.lock:
                delay = max(0.0, self.next_request - time.monotonic())
                if delay:
                    time.sleep(delay)
                self.next_request = time.monotonic() + self.interval

            try:
                response = self._session().get(
                    url, headers=self.headers, timeout=120, **kwargs
                )
            except requests.RequestException:
                if attempt == 5:
                    raise
                time.sleep(min(30.0, 2.0**attempt))
                continue

            if response.status_code not in {429, 500, 502, 503, 504}:
                return response
            status = response.status_code
            retry_after = float(response.headers.get("Retry-After", 2.0**attempt))
            response.close()
            print(
                f"{self.name} request returned {status}; "
                f"waiting {retry_after:g}s",
                flush=True,
            )
            with self.lock:
                self.next_request = max(
                    self.next_request, time.monotonic() + retry_after
                )
        raise RuntimeError(f"Ballchasing request failed after retries: {url}")

    def _session(self) -> requests.Session:
        session = getattr(self.local, "session", None)
        if session is None:
            session = requests.Session()
            self.local.session = session
        return session


def uses_octane_hitbox(path: Path) -> bool:
    import subtr_actor

    metadata = subtr_actor.get_replay_meta(str(path))["replay_meta"]
    players = (*metadata["team_zero"], *metadata["team_one"])
    return len(players) == 2 and all(
        player.get("car_hitbox_family") == "Octane" for player in players
    )


def download_replay(
    client: BallchasingClient,
    replay_id: str,
    output: Path,
) -> tuple[str, bool]:
    temporary_path = None
    try:
        with client.get(f"{API}/replays/{replay_id}/file", stream=True) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=output.parent, suffix=".part", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        temporary.write(chunk)
        eligible = uses_octane_hitbox(temporary_path)
        if eligible:
            os.replace(temporary_path, output)
        else:
            temporary_path.unlink()
        return replay_id, eligible
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/pro-1v1"))
    parser.add_argument("--target", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    parser.add_argument("--list-requests-per-second", type=float, default=16.0)
    arguments = parser.parse_args()
    token = os.environ.get("BALLCHASING_TOKEN")
    if not token:
        raise RuntimeError("BALLCHASING_TOKEN is required")
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    if not 0 < arguments.requests_per_second <= 2:
        parser.error("--requests-per-second must be in (0, 2]")
    if not 0 < arguments.list_requests_per_second <= 16:
        parser.error("--list-requests-per-second must be in (0, 16]")

    arguments.output.mkdir(parents=True, exist_ok=True)
    replay_dir = arguments.output / "replays"
    replay_dir.mkdir(exist_ok=True)
    database = sqlite3.connect(arguments.output / "manifest.sqlite3")
    database.execute(
        "CREATE TABLE IF NOT EXISTS replays ("
        "id TEXT PRIMARY KEY, date TEXT, blue TEXT, orange TEXT, downloaded INTEGER)"
    )
    columns = {
        row[1] for row in database.execute("PRAGMA table_info(replays)").fetchall()
    }
    if "octane" not in columns:
        database.execute("ALTER TABLE replays ADD COLUMN octane INTEGER")
    database.execute(
        "CREATE TABLE IF NOT EXISTS collector_state ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    database.commit()
    listing_client = BallchasingClient(
        token, arguments.list_requests_per_second, "metadata"
    )
    download_client = BallchasingClient(
        token, arguments.requests_per_second, "download"
    )
    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed:,.0f}/{task.total:,.0f}"),
    )
    progress.start()

    unknown = database.execute(
        "SELECT id FROM replays WHERE downloaded = 1 AND octane IS NULL"
    ).fetchall()
    validate_task = progress.add_task("validate", total=len(unknown))
    for (replay_id,) in unknown:
        output = replay_dir / f"{replay_id}.replay"
        eligible = output.is_file() and uses_octane_hitbox(output)
        if not eligible:
            output.unlink(missing_ok=True)
        database.execute(
            "UPDATE replays SET downloaded = ?, octane = ? WHERE id = ?",
            (int(eligible), int(eligible), replay_id),
        )
        progress.advance(validate_task)
    database.commit()

    downloaded = database.execute(
        "SELECT count(*) FROM replays WHERE downloaded = 1"
    ).fetchone()[0]
    download_task = progress.add_task(
        "download", total=arguments.target, completed=downloaded
    )
    discovered = database.execute("SELECT count(*) FROM replays").fetchone()[0]
    scan_task = progress.add_task(
        "pro 1v1 found", total=arguments.target, completed=discovered
    )

    pending: dict[Future[tuple[str, bool]], str] = {}
    scheduled = {
        row[0] for row in database.execute("SELECT id FROM replays WHERE octane = 0")
    }

    def drain(block: bool) -> None:
        nonlocal downloaded
        if not pending:
            return
        done = (
            {future for future in pending if future.done()}
            if not block
            else wait(pending, return_when=FIRST_COMPLETED).done
        )
        for future in done:
            replay_id = pending.pop(future)
            try:
                _, eligible = future.result()
            except Exception as error:
                scheduled.discard(replay_id)
                progress.console.print(f"[red]{replay_id}: {error}")
                continue
            database.execute(
                "UPDATE replays SET downloaded = ?, octane = ? WHERE id = ?",
                (int(eligible), int(eligible), replay_id),
            )
            if eligible:
                downloaded += 1
                progress.update(download_task, completed=downloaded)
        database.commit()

    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        # Resume files discovered before an interruption without rescanning first.
        for (replay_id,) in database.execute(
            "SELECT id FROM replays WHERE downloaded = 0 AND octane IS NOT 0"
        ):
            output = replay_dir / f"{replay_id}.replay"
            if output.is_file():
                eligible = uses_octane_hitbox(output)
                if not eligible:
                    output.unlink()
                database.execute(
                    "UPDATE replays SET downloaded = ?, octane = ? WHERE id = ?",
                    (int(eligible), int(eligible), replay_id),
                )
                if eligible:
                    downloaded += 1
                    progress.update(download_task, completed=downloaded)
                else:
                    scheduled.add(replay_id)
                continue
            scheduled.add(replay_id)
            future = executor.submit(
                download_replay, download_client, replay_id, output
            )
            pending[future] = replay_id
            if len(pending) >= arguments.workers * 2:
                drain(block=True)
        database.commit()
        while pending:
            drain(block=True)

        cursor = database.execute(
            "SELECT value FROM collector_state WHERE key = 'pro_scan_cursor'"
        ).fetchone()
        url = cursor[0] if cursor else f"{API}/replays"
        params = None if cursor else [
            ("pro", "true"),
            ("count", "200"),
            ("sort-by", "replay-date"),
            ("sort-dir", "desc"),
        ]
        while url and downloaded + len(pending) < arguments.target:
            response = listing_client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            for replay in payload.get("list", []):
                if not pro_duel(replay):
                    continue
                replay_id = replay["id"]
                blue = player_id(replay["blue"]["players"][0])
                orange = player_id(replay["orange"]["players"][0])
                inserted = database.execute(
                    "INSERT OR IGNORE INTO replays "
                    "(id, date, blue, orange, downloaded, octane) "
                    "VALUES (?, ?, ?, ?, 0, NULL)",
                    (replay_id, replay.get("date"), blue, orange),
                )
                if inserted.rowcount:
                    progress.advance(scan_task)
                output = replay_dir / f"{replay_id}.replay"
                if output.is_file():
                    continue
                if replay_id in scheduled:
                    continue
                scheduled.add(replay_id)
                future = executor.submit(
                    download_replay, download_client, replay_id, output
                )
                pending[future] = replay_id
                if len(pending) >= arguments.workers * 2:
                    drain(block=True)
            drain(block=False)
            url = payload.get("next")
            params = None
            if url:
                database.execute(
                    "INSERT OR REPLACE INTO collector_state VALUES "
                    "('pro_scan_cursor', ?)",
                    (url,),
                )
            else:
                database.execute(
                    "DELETE FROM collector_state WHERE key = 'pro_scan_cursor'"
                )
            database.commit()
        while pending:
            drain(block=True)
    database.close()
    progress.stop()


if __name__ == "__main__":
    main()
