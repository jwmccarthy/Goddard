import os
import time
import requests

from typing import Any
from dataclasses import dataclass

from dotenv import load_dotenv
from pathlib import Path

from threading import Lock, local
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from rich.progress import BarColumn, Progress, TextColumn


load_dotenv()

API_URL = "https://ballchasing.com/api"
API_RETRIES = 5
API_LIST_RATE = 16.0
API_DOWNLOAD_RATE = 2.0
DOWNLOAD_WORKERS = 4


@dataclass(slots=True)
class ReplayPage:
    replays:  list[dict[str, Any]]
    next_url: str | None


class BallchasingClient:

    def __init__(
        self,
        token:         str,
        list_rate:     float = API_LIST_RATE,
        download_rate: float = API_DOWNLOAD_RATE,
    ) -> None:
        self.headers = {"Authorization": token}

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self.list_interval = 1.0 / list_rate
        self.download_interval = 1.0 / download_rate

        self._download_lock = Lock()
        self._next_download_at = time.monotonic()
        self._download_sessions = local()

    @retry(
        retry=retry_if_exception_type(requests.ConnectionError),
        stop=stop_after_attempt(API_RETRIES),
        reraise=True,
    )
    def _get(self, url: str, **params: Any) -> ReplayPage:
        time.sleep(self.list_interval)

        params = {
            key.replace("_", "-"): value
            for key, value in params.items()
        }

        for attempt in range(API_RETRIES):
            response = self.session.get(url, params=params)
            if response.status_code != 429:
                break
            retry_after = response.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else 2 ** attempt)
        response.raise_for_status()

        content = response.json()

        return ReplayPage(
            replays=content["list"],
            next_url=content.get("next"),
        )

    def _download_session(self) -> requests.Session:
        session = getattr(self._download_sessions, "session", None)

        if session is None:
            session = requests.Session()
            session.headers.update(self.headers)
            self._download_sessions.session = session

        return session

    def _wait_for_download_slot(self) -> None:
        with self._download_lock:
            now = time.monotonic()
            download_at = max(now, self._next_download_at)
            self._next_download_at = download_at + self.download_interval

        time.sleep(max(0.0, download_at - now))

    def _download(self, replay_id: str, output_dir: Path) -> bool:
        for attempt in range(API_RETRIES):
            self._wait_for_download_slot()
            response = self._download_session().get(
                f"{API_URL}/replays/{replay_id}/file"
            )
            if response.status_code != 429:
                break
            retry_after = response.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else 2 ** attempt)

        if response.status_code == 404:
            return False
        response.raise_for_status()

        path = output_dir / f"{replay_id}.replay"
        path.write_bytes(response.content)
        return True

    def find_replay_entries(self, **params: Any) -> list[dict[str, Any]]:
        page = self._get(f"{API_URL}/replays", **params)
        replays = list(page.replays)

        while page.next_url:
            page = self._get(page.next_url)
            replays.extend(page.replays)

        return replays

    def find_replays(self, **params: Any) -> list[str]:
        return [
            replay["id"]
            for replay in self.find_replay_entries(**params)
        ]

    def download_replays(
        self,
        replay_ids: list[str],
        output_dir: str | Path
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Skip existing replay downloads
        replay_ids = [
            replay_id
            for replay_id in replay_ids
            if not (output_dir / f"{replay_id}.replay").exists()
        ]

        self._next_download_at = time.monotonic()
        unavailable = 0

        with (
            Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.completed:,.0f}/{task.total:,.0f}"),
            ) as progress,
            ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor,
        ):
            task = progress.add_task(
                "Downloading replays",
                total=len(replay_ids),
            )

            futures = [
                executor.submit(self._download, replay_id, output_dir)
                for replay_id in replay_ids
            ]

            for future in as_completed(futures):
                if not future.result():
                    unavailable += 1
                progress.advance(task)

        if unavailable:
            print(f"Unavailable replay files: {unavailable}")


if __name__ == "__main__":
    client = BallchasingClient(
        os.environ["BALLCHASING_TOKEN"]
    )

    print("Finding replays...")

    replay_ids = client.find_replays(
        playlist="ranked-duels",
        min_rank="supersonic-legend",
        pro="true",
    )

    client.download_replays(replay_ids, "./ballchasing_replays/replays")
