from __future__ import annotations

import io
from datetime import date
from typing import Any

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from gold.config import AppConfig
from gold.exceptions import SourceUnavailable
from gold.ingestion.base import DataSource
from gold.logging import get_logger

log = get_logger("ingestion.fred")

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class FredSource(DataSource):
    name = "fred"

    def __init__(self, cfg: AppConfig):
        super().__init__(cfg)
        # FRED graph CSV is often slow/unreachable; fail fast and leave the series empty.
        self.timeout = min(8, int(cfg.get("ingestion.request_timeout_seconds", 30)))
        self.ua = str(cfg.get("ingestion.user_agent", "gold-forecasting/0.1"))

    @retry(stop=stop_after_attempt(1), wait=wait_exponential(multiplier=1, min=1, max=2), reraise=True)
    def _get(self, series: str) -> str:
        resp = requests.get(
            FRED_CSV,
            params={"id": series},
            timeout=self.timeout,
            headers={"User-Agent": self.ua},
        )
        resp.raise_for_status()
        return resp.text

    def fetch(self, series_id: str, meta: dict[str, Any], start: date, end: date) -> pd.DataFrame:
        ticker = meta.get("ticker")
        if not ticker:
            raise SourceUnavailable(f"{series_id} missing FRED ticker")
        try:
            text = self._get(ticker)
        except Exception as exc:  # noqa: BLE001
            raise SourceUnavailable(f"FRED failed for {ticker}: {exc}") from exc
        df = pd.read_csv(io.StringIO(text))
        if df.empty or len(df.columns) < 2:
            raise SourceUnavailable(f"FRED returned an empty table for {ticker}")
        date_col, value_col = df.columns[0], df.columns[1]
        frame = pd.DataFrame(
            {
                "obs_date": pd.to_datetime(df[date_col], errors="coerce"),
                "value": pd.to_numeric(df[value_col].replace(".", pd.NA), errors="coerce"),
            }
        )
        frame = frame.dropna(subset=["obs_date", "value"])
        frame = frame[(frame["obs_date"].dt.date >= start) & (frame["obs_date"].dt.date <= end)]
        if frame.empty:
            raise SourceUnavailable(f"FRED {ticker} has no observations in {start}–{end}")
        log.info("FRED %s (%s): %s rows", series_id, ticker, len(frame))
        return frame
