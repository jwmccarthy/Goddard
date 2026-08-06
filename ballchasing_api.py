import os
import json
import requests

from pathlib import Path
from dotenv import load_dotenv
from types import SimpleNamespace
from tenacity import (
    retry, 
    retry_if_exception_type, 
    stop_after_attempt
)

from dataclasses import dataclass
from typing import List, Dict, Any


load_dotenv()

with open("config.json") as f:
    CONFIG = json.load(
        f, object_hook=lambda values: SimpleNamespace(**values)
    )


@dataclass
class BallchasingResponse:
    list: List[Dict[str, Any]]
    next: str


class BallchasingClient:

    def __init__(
        self,
        token: str,
        rate:  float = 2.0  # Requests/sec
    ) -> None:
        self.headers = {"Authorization": token}
        self.session = requests.Session()
        self.interval = 1.0 / rate

    @retry(
        retry=retry_if_exception_type(ConnectionError),
        stop=stop_after_attempt(CONFIG.api.retries),
        reraise=True,
    )
    def _get(self, url: str, **params: Dict[str, Any]) -> BallchasingResponse:
        params = {
            k.replace("_", "-"): v 
            for k, v in params.items()
        }
        response = self.session.get(
            url, headers=self.headers, params=params
        ).json()

        return BallchasingResponse(
            response["list"], response.get("next")
        )

    def find_replays(self, **params: Dict[str, Any]) -> List[Dict[str, Any]]:
        url = f"{CONFIG.api.url}/replays"

        response = self._get(url, **params)
        replays = response.list

        while response.next:
            response = self._get(response.next)
            replays.extend(response.list)
            break

        return replays

if __name__ == "__main__":
    token = os.getenv("BALLCHASING_TOKEN")
    client = BallchasingClient(token)
    response = client.find_replays(
        playlist="ranked-duels",
        min_rank="supersonic-legend"
    )

    print(response[0].keys())