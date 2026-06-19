from __future__ import annotations

import json
from datetime import date
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen


class ProviderError(RuntimeError):
    pass


def get_json(url: str, params: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    query = f"?{urlencode(params)}" if params else ""
    with urlopen(f"{url}{query}", timeout=timeout) as response:  # noqa: S310 - user-configured data APIs.
        payload = response.read().decode("utf-8")
    return json.loads(payload)


class PolygonClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ProviderError("Polygon API key is missing.")
        self.api_key = api_key

    def daily_aggregate(self, ticker: str, day: date) -> dict[str, Any]:
        url = f"https://api.polygon.io/v1/open-close/{ticker}/{day.isoformat()}"
        return get_json(url, {"adjusted": "true", "apiKey": self.api_key})


class FredClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ProviderError("FRED API key is missing.")
        self.api_key = api_key

    def series_observation(self, series_id: str, day: date) -> dict[str, Any]:
        return get_json(
            "https://api.stlouisfed.org/fred/series/observations",
            {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": day.isoformat(),
                "observation_end": day.isoformat(),
            },
        )


class SecEdgarClient:
    def company_submissions_url(self, cik: str) -> str:
        return f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
