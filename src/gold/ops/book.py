"""Canonical forecast book shared by CLI brief and the Streamlit UI."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

FINGERPRINT_COLS = (
    "forecast_id",
    "generated_at",
    "model_name",
    "model_version",
    "cutoff_date",
    "horizon_calendar_days",
    "target_date",
    "point",
    "lower",
    "upper",
)


def _iso(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.replace(microsecond=0).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime().replace(microsecond=0).isoformat()


def _day(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return str(value)[:10]
    return ts.date().isoformat()


def _num(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def forecast_fingerprint(forecasts: pd.DataFrame) -> list[dict[str, Any]]:
    """Stable records: values, timestamps, and model versions for CLI/UI equality."""
    if forecasts is None or forecasts.empty:
        return []
    df = forecasts.copy()
    if "horizon_calendar_days" in df.columns:
        df = df.sort_values("horizon_calendar_days")
    rows: list[dict[str, Any]] = []
    for rec in df.itertuples():
        rows.append(
            {
                "forecast_id": str(getattr(rec, "forecast_id", "")),
                "generated_at": _iso(getattr(rec, "generated_at", None)),
                "model_name": str(getattr(rec, "model_name", "")),
                "model_version": str(getattr(rec, "model_version", "")),
                "cutoff_date": _day(getattr(rec, "cutoff_date", None)),
                "horizon": int(getattr(rec, "horizon_calendar_days")),
                "target_date": _day(getattr(rec, "target_date", None)),
                "point": _num(getattr(rec, "point", None)),
                "lower": _num(getattr(rec, "lower", None)),
                "upper": _num(getattr(rec, "upper", None)),
            }
        )
    return rows


def brief_forecast_fingerprint(brief: dict[str, Any]) -> list[dict[str, Any]]:
    return forecast_fingerprint(brief.get("forecasts"))
