from __future__ import annotations

from datetime import date

import pandas as pd

from gold.features.alignment import asof_join


def test_asof_join_does_not_use_future_prints():
    # CPI for January published mid-March (lag ~45 days).
    series = pd.DataFrame(
        {
            "available_at": [date(2024, 3, 15), date(2024, 4, 15)],
            "value": [100.0, 110.0],
        }
    )
    dates = pd.to_datetime(["2024-03-01", "2024-03-15", "2024-03-16", "2024-04-15"])
    joined = asof_join(dates, series, "value", "india_cpi")
    assert pd.isna(joined.iloc[0])  # before first publication
    assert joined.iloc[1] == 100.0
    assert joined.iloc[2] == 100.0
    assert joined.iloc[3] == 110.0
    # The April CPI value must not appear on 16 Mar.
    assert joined.loc[pd.Timestamp("2024-03-16")] != 110.0
