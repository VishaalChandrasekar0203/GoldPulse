from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests

from tenacity import retry, stop_after_attempt, wait_exponential

from gold.config import AppConfig
from gold.exceptions import SourceUnavailable
from gold.ingestion.base import DataSource
from gold.logging import get_logger

log = get_logger("ingestion.worldbank")

WB_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"


class WorldBankSource(DataSource):
    name = "worldbank"

    def __init__(self, cfg: AppConfig):
        super().__init__(cfg)
        self.timeout = min(20, int(cfg.get("ingestion.request_timeout_seconds", 30)))
        self.ua = str(cfg.get("ingestion.user_agent", "gold-forecasting/0.1"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _get(self, country: str, indicator: str):
        resp = requests.get(
            WB_URL.format(country=country, indicator=indicator),
            params={"format": "json", "per_page": 20000},
            timeout=self.timeout,
            headers={"User-Agent": self.ua},
        )
        resp.raise_for_status()
        return resp.json()

    def fetch(self, series_id: str, meta: dict[str, Any], start: date, end: date) -> pd.DataFrame:
        indicator = meta.get("ticker")
        country = meta.get("country", "IND")
        if not indicator:
            raise SourceUnavailable(f"{series_id} missing World Bank indicator")
        try:
            payload = self._get(country, indicator)
        except Exception as exc:  # noqa: BLE001
            raise SourceUnavailable(f"World Bank failed for {indicator}: {exc}") from exc
        if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
            raise SourceUnavailable(f"World Bank returned no observations for {indicator}")
        rows = []
        for item in payload[1]:
            if item.get("value") is None or not item.get("date"):
                continue
            year = str(item["date"])
            try:
                obs = date(int(year[:4]), 12, 31) if len(year) == 4 else date.fromisoformat(year)
            except ValueError:
                continue
            rows.append({"obs_date": obs, "value": float(item["value"])})
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise SourceUnavailable(f"World Bank {indicator} parsed empty")
        frame = frame[(frame["obs_date"] >= start) & (frame["obs_date"] <= end)]
        if frame.empty:
            raise SourceUnavailable(f"World Bank {indicator} has no rows in {start}–{end}")
        log.info("World Bank %s (%s): %s rows", series_id, indicator, len(frame))
        return frame
