from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from gold.config import AppConfig
from gold.exceptions import SourceUnavailable
from gold.ingestion.base import DataSource
from gold.logging import get_logger

log = get_logger("ingestion.static")


class StaticTableSource(DataSource):
    """Curated public policy/calendar facts — not market prints."""

    name = "static"

    def __init__(self, cfg: AppConfig):
        super().__init__(cfg)

    def _load_step_series(self, csv_name: str, value_col: str, start: date, end: date) -> pd.DataFrame:
        path = self.cfg.project_root / "config" / csv_name
        if not path.exists():
            raise SourceUnavailable(f"Missing static table {path}")
        raw = pd.read_csv(path, parse_dates=["effective_date"])
        raw = raw.sort_values("effective_date")
        idx = pd.date_range(start, end, freq="D")
        stepped = pd.DataFrame({"obs_date": pd.to_datetime(idx).astype("datetime64[ns]")})
        events = raw.rename(columns={"effective_date": "obs_date", value_col: "value"})
        events["obs_date"] = pd.to_datetime(events["obs_date"]).astype("datetime64[ns]")
        merged = pd.merge_asof(
            stepped.sort_values("obs_date"),
            events[["obs_date", "value"]].sort_values("obs_date"),
            on="obs_date",
            direction="backward",
        )
        merged = merged.dropna(subset=["value"])
        return merged

    def fetch(self, series_id: str, meta: dict[str, Any], start: date, end: date) -> pd.DataFrame:
        if series_id == "gold_import_duty":
            return self._load_step_series("import_duties.csv", "duty_pct", start, end)
        if series_id == "rbi_repo_rate":
            return self._load_step_series("rbi_repo_rate.csv", "rate_pct", start, end)
        if series_id == "india_festivals":
            path = self.cfg.project_root / "config" / "festivals.csv"
            if not path.exists():
                raise SourceUnavailable(f"Missing festival table {path}")
            raw = pd.read_csv(path, parse_dates=["event_date"])
            idx = pd.date_range(start, end, freq="D")
            flags = pd.DataFrame({"obs_date": idx, "value": 0.0})
            event_days = set(pd.to_datetime(raw["event_date"]).dt.date)
            flags["value"] = [1.0 if d.date() in event_days else 0.0 for d in flags["obs_date"]]
            return flags
        raise SourceUnavailable(f"No static loader for {series_id}")
