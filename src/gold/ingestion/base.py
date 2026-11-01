from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from gold.config import AppConfig
from gold.exceptions import SourceUnavailable


def utcnow() -> datetime:
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def apply_availability(df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    out = df.copy()
    out["obs_date"] = pd.to_datetime(out["obs_date"]).dt.date
    out["available_at"] = [
        d + timedelta(days=int(lag_days)) if d is not None else None for d in out["obs_date"]
    ]
    return out


class DataSource(ABC):
    name: str

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    @abstractmethod
    def fetch(self, series_id: str, meta: dict[str, Any], start: date, end: date) -> pd.DataFrame:
        """Return observations or raise SourceUnavailable. Never fabricate rows."""

    def empty_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "series_id",
                "obs_date",
                "value",
                "available_at",
                "frequency",
                "source",
                "ingested_at",
                "run_id",
                "is_derived",
                "quality_flag",
                "provenance",
            ]
        )

    def finalize(
        self,
        raw: pd.DataFrame,
        series_id: str,
        meta: dict[str, Any],
        run_id: str,
        is_derived: bool = False,
    ) -> pd.DataFrame:
        if raw.empty:
            return self.empty_frame()
        df = raw.copy()
        df["series_id"] = series_id
        df["frequency"] = meta.get("frequency", "daily")
        df["source"] = meta.get("source", self.name)
        df["ingested_at"] = utcnow()
        df["run_id"] = run_id
        df["is_derived"] = is_derived
        df["quality_flag"] = df.get("quality_flag", "ok")
        lag = int(meta.get("publication_lag_days", 0) or 0)
        df = apply_availability(df, lag)
        df["provenance"] = [
            {
                "source": meta.get("source"),
                "source_url": meta.get("source_url"),
                "ticker": meta.get("ticker"),
                "license": meta.get("license"),
                "is_derived": is_derived,
                "fetched_at": utcnow().isoformat(),
            }
        ] * len(df)
        return df[
            [
                "series_id",
                "obs_date",
                "value",
                "available_at",
                "frequency",
                "source",
                "ingested_at",
                "run_id",
                "is_derived",
                "quality_flag",
                "provenance",
            ]
        ]
