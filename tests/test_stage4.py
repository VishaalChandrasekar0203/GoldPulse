from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from gold.data.quality import capture_quality_snapshot, is_stale, timestamp_issues
from gold.evaluation.monitor import rolling_metrics, score_live_forecasts
from gold.forecasting.volatility import ewma_variance
from gold.ingestion.derived import derive_gold_inr_10g
from gold.scenarios.engine import NAMED_PACK_DEFAULTS, parity_inr_10g, run_named_packs, snapshot
from tests.helpers import business_days, make_obs


def test_named_packs_combined_identity_not_forecast():
    from gold.scenarios.engine import MarketSnapshot

    snap = MarketSnapshot(
        as_of=pd.Timestamp("2024-01-02"),
        gold_usd_oz=2000.0,
        usd_inr=80.0,
        parity_inr_10g=parity_inr_10g(2000.0, 80.0),
        dxy=100.0,
        us_10y_yield=4.0,
        us_real_yield_10y=1.5,
        gold_import_duty=6.0,
        etf_parity_gap_pct=-0.02,
        etf_inr_10g=None,
    )

    class Cfg:
        troy_ounce_grams = 31.1034768

        def get(self, dotted, default=None):
            return default

    table = run_named_packs(snap, Cfg(), {})
    bull = table[(table["pack"] == "bull") & (table["kind"] == "identity_combined")].iloc[0]
    spec = NAMED_PACK_DEFAULTS["bull"]
    expected = snap.parity_inr_10g * (1 + spec["xau_pct"] / 100) * (1 + spec["usdinr_pct"] / 100)
    np.testing.assert_allclose(bull["scenario_parity_inr_10g"], expected)
    assert "NOT A FORECAST" in str(bull["notes"]).upper() or "not a forecast" in str(bull["methodology"]).lower()
    assert (table["kind"] == "identity_combined").sum() == 3


def test_timestamp_issues_and_stale():
    df = pd.DataFrame(
        {
            "obs_date": [date(2024, 1, 10), date(2024, 1, 11)],
            "available_at": [date(2024, 1, 9), date(2024, 1, 11)],
            "provenance": [None, {"ok": True}],
        }
    )
    issues = timestamp_issues(df)
    assert (issues["issue"] == "available_at_before_obs_date").sum() == 1
    assert is_stale(date(2024, 1, 1), date(2024, 1, 10), 4) is True
    assert is_stale(date(2024, 1, 8), date(2024, 1, 10), 4) is False


def test_quality_snapshot_does_not_fabricate_restricted(store, cfg):
    store.upsert_catalog(
        [
            {
                "series_id": "mcx_gold_futures",
                "name": "MCX",
                "description": "",
                "unit": "INR",
                "frequency": "daily",
                "source": "restricted",
                "source_url": "",
                "license": "",
                "publication_lag_days": 0,
                "category": "price",
                "status": "unavailable",
                "notes": "not fabricated",
            }
        ]
    )
    snap = capture_quality_snapshot(cfg, store)
    mcx = snap[snap["series_id"] == "mcx_gold_futures"].iloc[0]
    assert mcx["n_obs"] == 0
    assert mcx["stale"] is False or mcx["stale"] == False  # noqa: E712
    assert "Not fabricated" in str(mcx["notes"])
    assert store.load_series("mcx_gold_futures").empty


def test_score_live_forecast_when_realized_arrives(store, cfg):
    days = business_days(30)
    gold = [(d, 1800 + i) for i, d in enumerate(days)]
    fx = [(d, 80.0) for d in days]
    store.upsert_observations(make_obs("gold_usd_oz", gold), "gold_usd_oz")
    store.upsert_observations(make_obs("usd_inr", fx), "usd_inr")
    derived = derive_gold_inr_10g(store, cfg, run_id="fixture")
    store.upsert_observations(derived, "gold_inr_10g")
    target = store.load_series("gold_inr_10g").sort_values("obs_date")
    origin = float(target.iloc[-5]["value"])
    realized = float(target.iloc[-4]["value"])
    cutoff = target.iloc[-5]["obs_date"]
    tdate = target.iloc[-4]["obs_date"]
    fc = pd.DataFrame(
        [
            {
                "forecast_id": "fc_test",
                "generated_at": pd.Timestamp("2024-01-01"),
                "model_name": "naive",
                "model_version": "test",
                "target_series": "gold_inr_10g",
                "cutoff_date": cutoff,
                "horizon_calendar_days": 1,
                "horizon_trading_days": 1,
                "target_date": tdate,
                "target_is_trading_day": True,
                "point": origin,
                "lower": origin * 0.99,
                "upper": origin * 1.01,
                "interval_low": 0.1,
                "interval_high": 0.9,
                "data_cutoff": pd.Timestamp("2024-01-01"),
                "notes": "Interval method=ewma_sqrt_h. test",
            }
        ]
    )
    store.save_forecasts(fc)
    scored = score_live_forecasts(cfg, store)
    assert len(scored) == 1
    np.testing.assert_allclose(scored.iloc[0]["realized"], realized)
    np.testing.assert_allclose(scored.iloc[0]["abs_error"], abs(realized - origin))
    roll = rolling_metrics(scored, window=21)
    assert not roll.empty
    assert roll.iloc[0]["horizon"] == 1


def test_ewma_monitor_prefix_matches_full_path():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.01, 300)
    full = ewma_variance(r)
    pref = ewma_variance(r[:120])
    np.testing.assert_allclose(full[:120], pref)
