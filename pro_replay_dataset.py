import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
import sqlite3
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import requests
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn


API = "https://ballchasing.com/api"
PLAYERS = {
    "zen":      "steam:76561198144145654",
    "vatira":   "steam:76561198960239428",
    "exotiik":  "steam:76561198258466111",
    "atow":     "steam:76561198289610054",
    "juicy":    "steam:76561199013030566",
    "stizzy":   "steam:76561199047701758",
    "seikoo":   "steam:76561199013057612",
    "oski":     "steam:76561198381037239",
    "joreuz":   "steam:76561198974429177",
    "crr":      "steam:76561198845948731",
    "rezears":  "steam:76561198994499073",
    "mawkzy":   "epic:6ea3d3d4f992494dacd7f757ff4e2b1a",
    "diaz":     "steam:76561198880724484",
}


def player_id(player: dict) -> str:
    identity = player.get("id") or {}
    return f"{identity.get('platform')}:{identity.get('id')}"


def elite_duel(replay: dict, verified: set[str]) -> bool:
    blue = replay.get("blue", {}).get("players", [])
    orange = replay.get("orange", {}).get("players", [])
    return (
        len(blue) == len(orange) == 1
        and player_id(blue[0]) in verified
        and player_id(orange[0]) in verified
    )


class BallchasingClient:
    def __init__(self, token: str, requests_per_second: float) -> None:
        self.headers = {"Authorization": token}
        self.interval = 1.0 / requests_per_second
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
            retry_after = float(response.headers.get("Retry-After", 2.0**attempt))
            response.close()
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


def discover_pool(
    client: BallchasingClient,
    pool_size: int,
    replay_limit: int,
    progress: Progress,
    task: int,
) -> dict[str, str]:
    counts = Counter()
    names = {}
    url = f"{API}/replays"
    params = [
        ("pro", "true"),
        ("count", "200"),
        ("sort-by", "replay-date"),
        ("sort-dir", "desc"),
    ]
    inspected = 0
    while url and inspected < replay_limit:
        response = client.get(url, params=params)
        response.raise_for_status()
        payload = response.json()
        for replay in payload.get("list", []):
            blue = replay.get("blue", {}).get("players", [])
            orange = replay.get("orange", {}).get("players", [])
            if len(blue) != 1 or len(orange) != 1:
                continue
            inspected += 1
            progress.update(task, completed=inspected)
            for player in (*blue, *orange):
                identity = player_id(player)
                counts[identity] += 1
                names[identity] = player.get("name") or identity
        url = payload.get("next")
        params = None

    selected = set(PLAYERS.values())
    selected.update(identity for identity, _ in counts.most_common(pool_size))
    ranked = sorted(selected, key=lambda identity: counts[identity], reverse=True)
    preferred = list(dict.fromkeys(PLAYERS.values()))
    ranked = preferred + [identity for identity in ranked if identity not in preferred]
    ranked = ranked[:pool_size]
    return {identity: names.get(identity, identity) for identity in ranked}


def download_replay(
    client: BallchasingClient,
    replay_id: str,
    output: Path,
) -> str:
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
        os.replace(temporary_path, output)
        return replay_id
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/pro-1v1"))
    parser.add_argument("--target", type=int, default=100_000)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--discovery-limit", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    arguments = parser.parse_args()
    token = os.environ.get("BALLCHASING_TOKEN")
    if not token:
        raise RuntimeError("BALLCHASING_TOKEN is required")
    if arguments.workers < 1:
        parser.error("--workers must be positive")
    if not 0 < arguments.requests_per_second <= 2:
        parser.error("--requests-per-second must be in (0, 2]")

    arguments.output.mkdir(parents=True, exist_ok=True)
    replay_dir = arguments.output / "replays"
    replay_dir.mkdir(exist_ok=True)
    database = sqlite3.connect(arguments.output / "manifest.sqlite3")
    database.execute(
        "CREATE TABLE IF NOT EXISTS replays ("
        "id TEXT PRIMARY KEY, date TEXT, blue TEXT, orange TEXT, downloaded INTEGER)"
    )
    client = BallchasingClient(token, arguments.requests_per_second)
    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed:,.0f}/{task.total:,.0f}"),
    )
    progress.start()
    discovery_task = progress.add_task("discover", total=arguments.discovery_limit)
    pool_path = arguments.output / "player_pool.json"
    if pool_path.is_file():
        pool = json.loads(pool_path.read_text())
        progress.update(discovery_task, completed=arguments.discovery_limit)
    else:
        pool = discover_pool(
            client,
            arguments.pool_size,
            arguments.discovery_limit,
            progress,
            discovery_task,
        )
        pool_path.write_text(json.dumps(pool, indent=2) + "\n")
    verified = set(pool)
    downloaded = database.execute(
        "SELECT count(*) FROM replays WHERE downloaded = 1"
    ).fetchone()[0]
    download_task = progress.add_task(
        "download", total=arguments.target, completed=downloaded
    )

    pending: dict[Future[str], str] = {}
    scheduled: set[str] = set()

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
                future.result()
            except Exception as error:
                scheduled.discard(replay_id)
                progress.console.print(f"[red]{replay_id}: {error}")
                continue
            database.execute(
                "UPDATE replays SET downloaded = 1 WHERE id = ?", (replay_id,)
            )
            downloaded += 1
            progress.update(download_task, completed=downloaded)
        database.commit()

    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        for source_id in verified:
            url = f"{API}/replays"
            params = [
                ("player-id", source_id),
                ("pro", "true"),
                ("count", "200"),
                ("sort-by", "replay-date"),
                ("sort-dir", "desc"),
            ]
            while url and downloaded + len(pending) < arguments.target:
                response = client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
                for replay in payload.get("list", []):
                    if not elite_duel(replay, verified):
                        continue
                    replay_id = replay["id"]
                    blue = player_id(replay["blue"]["players"][0])
                    orange = player_id(replay["orange"]["players"][0])
                    database.execute(
                        "INSERT OR IGNORE INTO replays VALUES (?, ?, ?, ?, 0)",
                        (replay_id, replay.get("date"), blue, orange),
                    )
                    output = replay_dir / f"{replay_id}.replay"
                    if output.is_file():
                        database.execute(
                            "UPDATE replays SET downloaded = 1 WHERE id = ?",
                            (replay_id,),
                        )
                        continue
                    if replay_id in scheduled:
                        continue
                    scheduled.add(replay_id)
                    future = executor.submit(download_replay, client, replay_id, output)
                    pending[future] = replay_id
                    if len(pending) >= arguments.workers * 2:
                        drain(block=True)
                    if downloaded + len(pending) >= arguments.target:
                        break
                drain(block=False)
                url = payload.get("next")
                params = None
        while pending:
            drain(block=True)
    database.close()
    progress.stop()


if __name__ == "__main__":
    main()
