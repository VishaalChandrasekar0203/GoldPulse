from __future__ import annotations

from datetime import date

import pandas as pd


def asof_join(dates: pd.DatetimeIndex, series: pd.DataFrame, value_col: str, name: str) -> pd.Series:
    """For each feature date t, take the last observation with available_at <= t.

    This is the anti-lookahead join. Observations are never used before their
    publication/availability date.
    """
    if series.empty:
        return pd.Series(index=dates, dtype=float, name=name)
    src = series.copy()
    src["available_at"] = pd.to_datetime(src["available_at"]).astype("datetime64[ns]")
    src = src.dropna(subset=["available_at", value_col]).sort_values("available_at")
    left = pd.DataFrame({"available_at": pd.to_datetime(dates).astype("datetime64[ns]")}).sort_values(
        "available_at"
    )
    merged = pd.merge_asof(
        left,
        src[["available_at", value_col]],
        on="available_at",
        direction="backward",
    )
    merged = merged.set_index("available_at")[value_col]
    merged.name = name
    return merged.reindex(dates)


def assert_no_lookahead(features: pd.DataFrame, as_of_col: str = "as_of") -> None:
    """Sanity: feature timestamp must be on or after as_of (availability) for that row."""
    if as_of_col not in features.columns:
        return
    as_of = pd.to_datetime(features[as_of_col])
    idx = pd.to_datetime(features["obs_date"] if "obs_date" in features.columns else features.index)
    if (as_of > idx).any():
        raise ValueError("Look-ahead detected: availability date after feature date")
