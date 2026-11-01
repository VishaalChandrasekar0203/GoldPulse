from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from gold.ingestion.base import utcnow


def make_obs(series_id: str, values: list[tuple[date, float]], lag: int = 0) -> pd.DataFrame:
    rows = []
    for d, v in values:
        rows.append(
            {
                "series_id": series_id,
                "obs_date": d,
                "value": v,
                "available_at": d + timedelta(days=lag),
                "frequency": "daily",
                "source": "test",
                "ingested_at": utcnow(),
                "run_id": "test",
                "is_derived": False,
                "quality_flag": "ok",
                "provenance": {"source": "unit-test-fixture", "not_market_data": True},
            }
        )
    return pd.DataFrame(rows)


def business_days(n: int, start: date = date(2020, 1, 2)) -> list[date]:
    out = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out
