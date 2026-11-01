from __future__ import annotations

from datetime import date

import pandas as pd

from gold.data.quality import validate_observations
from tests.helpers import make_obs


def test_range_violations_dropped_not_imputed(cfg):
    df = make_obs("gold_inr_10g", [(date(2024, 1, 2), 65000.0), (date(2024, 1, 3), -5.0)])
    cleaned, anomalies = validate_observations(df, "gold_inr_10g", cfg)
    assert len(cleaned) == 1
    assert float(cleaned["value"].iloc[0]) == 65000.0
    assert (anomalies["reason"] == "range").any()


def test_non_finite_dropped(cfg):
    df = make_obs("usd_inr", [(date(2024, 1, 2), 83.0)])
    extra = df.iloc[[0]].copy()
    extra["obs_date"] = date(2024, 1, 3)
    extra["value"] = float("nan")
    cleaned, _ = validate_observations(pd.concat([df, extra], ignore_index=True), "usd_inr", cfg)
    assert len(cleaned) == 1
