"""US Treasury Daily Treasury Par Real Yield Curve (public domain)."""

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

log = get_logger("ingestion.treasury")

ARCHIVE_2003_2023 = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rate-archives/par-real-yield-curve-rates-2003-2023.csv"
)
YEARLY_CSV = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
)


def yearly_real_yield_url(year: int) -> str:
    return f"{YEARLY_CSV.format(year=year)}?_format=csv&type=daily_treasury_real_yield_curve"


def _tenor_column(columns: list[str]) -> str | None:
    for orig in columns:
        compact = orig.lower().replace(" ", "").replace("_", "").replace("-", "")
        if compact in {"10yr", "10year", "tc10year", "10y"}:
            return orig
    return None


def parse_real_yield_csv(text: str) -> pd.DataFrame:
    """Parse a Treasury real-yield CSV into obs_date / 10y value columns."""
    text = (text or "").strip()
    if not text:
        return pd.DataFrame(columns=["obs_date", "value"])
    first = text.splitlines()[0].lower()
    if "date" in first or "yr" in first or "year" in first:
        df = pd.read_csv(io.StringIO(text))
    else:
        df = pd.read_csv(
            io.StringIO(text),
            header=None,
            names=["date", "5 yr", "7 yr", "10 yr", "20 yr", "30 yr"],
        )
    if df.empty:
        return pd.DataFrame(columns=["obs_date", "value"])
    date_col = next((c for c in df.columns if "date" in str(c).lower()), df.columns[0])
    ten = _tenor_column([str(c) for c in df.columns])
    if ten is None:
        raise SourceUnavailable(f"Treasury CSV missing 10y column: {list(df.columns)}")
    frame = pd.DataFrame(
        {
            "obs_date": pd.to_datetime(df[date_col], errors="coerce"),
            "value": pd.to_numeric(df[ten].replace("N/A", pd.NA), errors="coerce"),
        }
    )
    return frame.dropna(subset=["obs_date", "value"])


class TreasurySource(DataSource):
    """10-year TIPS real yield from Treasury.gov (not FRED)."""

    name = "treasury"

    def __init__(self, cfg: AppConfig):
        super().__init__(cfg)
        self.timeout = int(cfg.get("ingestion.request_timeout_seconds", 30))
        self.ua = str(cfg.get("ingestion.user_agent", "gold-forecasting/0.1"))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def _get(self, url: str) -> str:
        resp = requests.get(url, timeout=self.timeout, headers={"User-Agent": self.ua})
        resp.raise_for_status()
        return resp.text

    def _frames(self, start: date, end: date) -> list[pd.DataFrame]:
        frames: list[pd.DataFrame] = []
        try:
            archive = parse_real_yield_csv(self._get(ARCHIVE_2003_2023))
            frames.append(archive)
        except Exception as exc:  # noqa: BLE001
            log.warning("Treasury archive 2003-2023 failed: %s", exc)
        years = list(range(max(2024, start.year), end.year + 1))
        if not frames:
            years = list(range(start.year, end.year + 1))
        for year in years:
            try:
                frames.append(parse_real_yield_csv(self._get(yearly_real_yield_url(year))))
            except Exception as exc:  # noqa: BLE001
                log.warning("Treasury real yield %s failed: %s", year, exc)
        return frames

    def fetch(self, series_id: str, meta: dict[str, Any], start: date, end: date) -> pd.DataFrame:
        frames = self._frames(start, end)
        if not frames:
            raise SourceUnavailable("Treasury real-yield endpoints returned no tables")
        frame = pd.concat(frames, ignore_index=True)
        frame = frame.drop_duplicates(subset=["obs_date"]).sort_values("obs_date")
        frame = frame[(frame["obs_date"].dt.date >= start) & (frame["obs_date"].dt.date <= end)]
        if frame.empty:
            raise SourceUnavailable(f"Treasury real yield has no observations in {start}–{end}")
        log.info("Treasury %s: %s rows", series_id, len(frame))
        return frame.reset_index(drop=True)
