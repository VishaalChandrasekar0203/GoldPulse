from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from gold.config import AppConfig
from gold.exceptions import SourceUnavailable
from gold.ingestion.base import DataSource
from gold.logging import get_logger

log = get_logger("ingestion.yfinance")


class YahooFinanceSource(DataSource):
    name = "yfinance"

    def __init__(self, cfg: AppConfig):
        super().__init__(cfg)
        self.timeout = int(cfg.get("ingestion.request_timeout_seconds", 30))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _download(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        # yfinance end is exclusive for some intervals; add one day.
        data = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if data is None or data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
        return data

    def fetch(self, series_id: str, meta: dict[str, Any], start: date, end: date) -> pd.DataFrame:
        ticker = meta.get("ticker")
        if not ticker:
            raise SourceUnavailable(f"{series_id} missing yfinance ticker")
        try:
            raw = self._download(ticker, start, end)
        except Exception as exc:  # noqa: BLE001 — network/source failures must not fabricate
            raise SourceUnavailable(f"yfinance failed for {ticker}: {exc}") from exc
        if raw.empty:
            raise SourceUnavailable(f"yfinance returned no rows for {ticker} ({start} to {end})")
        close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        frame = pd.DataFrame({"obs_date": pd.to_datetime(raw.index).tz_localize(None), "value": close.astype(float).values})
        frame = frame.dropna(subset=["value"])
        log.info("yfinance %s (%s): %s rows", series_id, ticker, len(frame))
        return frame
