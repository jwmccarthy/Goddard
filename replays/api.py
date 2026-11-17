import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, local
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse
from uuid import UUID

import requests
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from replays.utils import write_json

load_dotenv()


API_URL = "https://ballchasing.com/api"
API_RETRIES = 5
LIST_RATE = 16.0
DOWNLOAD_RATE = 2.0
REQUEST_TIMEOUT = 30


@dataclass(slots=True)
class ReplayPage:

    replays:  list[dict[str, Any]]
    next_url: str | None


def _retry_delay(response: requests.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")

    if retry_after:
        return float(retry_after)

    return float(2**attempt)


class BallchasingClient:

    def __init__(
        self,
        token:         str,
        list_rate:     float = LIST_RATE,
        download_rate: float = DOWNLOAD_RATE
    ) -> None:
        self.headers = {
            "Authorization": token,
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self.list_interval = 1.0 / list_rate
        self.download_interval = 1.0 / download_rate

        self._download_lock = Lock()
        self._next_download_at = time.monotonic()
        self._download_sessions = local()

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(API_RETRIES),
        reraise=True
    )
    def get_json(self, url: str, **params: Any) -> dict[str, Any]:
        if url != API_URL and not url.startswith(f"{API_URL}/"):
            raise ValueError("refusing to send Ballchasing credentials to another endpoint")

        time.sleep(self.list_interval)

        params = {
            key.replace("_", "-"): value
            for key, value in params.items()
        }

        for attempt in range(API_RETRIES):
            response = self.session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code not in (429, 500, 502, 503, 504):
                break

            time.sleep(_retry_delay(response, attempt))

        response.raise_for_status()

        return response.json()

    def get(self, url: str, **params: Any) -> ReplayPage:
        content = self.get_json(url, **params)

        return ReplayPage(
            replays=content["list"],
            next_url=content.get("next")
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

    @retry(
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        stop=stop_after_attempt(API_RETRIES),
        reraise=True
    )
    def download_replay(self, replay_id: str, output_dir: Path) -> bool:
        replay_id = str(UUID(replay_id))

        for attempt in range(API_RETRIES):
            self._wait_for_download_slot()

            response = self._download_session().get(
                f"{API_URL}/replays/{replay_id}/file",
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code not in (429, 500, 502, 503, 504):
                break

            time.sleep(_retry_delay(response, attempt))

        if response.status_code == 404:
            return False

        response.raise_for_status()

        path = output_dir / f"{replay_id}.replay"
        temporary = path.with_suffix(".replay.part")
        temporary.write_bytes(response.content)
        temporary.replace(path)

        return True

    def iter_replay_pages(
        self,
        **params: Any
    ) -> Iterator[list[dict[str, Any]]]:
        page = self.get(f"{API_URL}/replays", **params)

        while True:
            yield page.replays

            if not page.next_url:
                break

            next_params = parse_qs(urlparse(page.next_url).query)

            missing = {
                key: value
                for key, value in params.items()
                if key.replace("_", "-")
                not in next_params
            }

            page = self.get(page.next_url, **missing)

    def find_replay_entries(self, **params: Any) -> list[dict[str, Any]]:
        replays: list[dict[str, Any]] = []

        for page in self.iter_replay_pages(**params):
            replays.extend(page)

        return replays


class CachedClient(BallchasingClient):
    """Credential-free response cache for resumable downloads."""

    def __init__(self, token: str, directory: str | Path) -> None:
        super().__init__(token, list_rate=2.0, download_rate=2.0)

        self.cache = Path(directory)
        self.cache.mkdir(parents=True, exist_ok=True)

    def get_json(self, url: str, **params: Any) -> dict[str, Any]:
        if url != API_URL and not url.startswith(API_URL + "/"):
            raise ValueError("refusing foreign API URL")

        key = json.dumps([url, params], sort_keys=True)

        path = self.cache / (hashlib.sha256(key.encode()).hexdigest() + ".json")

        if path.exists():
            return json.loads(path.read_text())

        result = super().get_json(url, **params)
        write_json(path, result)

        return result
