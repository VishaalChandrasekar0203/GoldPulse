from __future__ import annotations

import numpy as np
import pandas as pd

from gold.forecasting.volatility import (
    completed_h_day_return,
    empirical_h_exponent,
    ewma_variance,
    interval_coverage,
    price_interval,
    winkler_score,
)
from gold.ingestion.derived import derive_gold_inr_10g
from gold.scenarios.engine import SCENARIO_DISCLAIMER, build, parity_inr_10g, run_scenarios, snapshot
from tests.helpers import business_days, make_obs


def test_ewma_is_causal():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 400)
    full = ewma_variance(r, lam=0.94)
    prefix = ewma_variance(r[:180], lam=0.94)
    np.testing.assert_allclose(full[:180], prefix)
    later = ewma_variance(np.concatenate([r[:180], rng.normal(0, 0.05, 50)]), lam=0.94)
    np.testing.assert_allclose(later[:180], prefix)


def test_winkler_and_coverage():
    y = np.array([0.0, 0.0, 2.0])
    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    assert interval_coverage(y, lo, hi) == 2 / 3
    # Miss of 1.0 at alpha=0.2 costs 2/0.2 * 1 = 10 on that obs; mean width=2.
    assert winkler_score(y, lo, hi, 0.2) == (2 + 2 + 12) / 3


def test_empirical_lambda_near_half_for_iid():
    rng = np.random.default_rng(1)
    daily = rng.normal(0, 0.01, 2000)
    close = 100 * np.exp(np.cumsum(daily))
    h = 21
    y = np.log(close[h:] / close[:-h])
    lam = empirical_h_exponent(daily[:-h], h, y)
    assert 0.40 < lam < 0.60


def test_completed_h_day_return_known_at_t():
    close = np.array([100.0, 110.0, 121.0])
    r = completed_h_day_return(close, 1)
    assert np.isnan(r[0])
    np.testing.assert_allclose(r[1:], [np.log(1.1), np.log(1.1)])


def test_price_interval_symmetric_in_log_space():
    lo, hi = price_interval([100.0], [0.0], [0.1], 1.0)
    np.testing.assert_allclose(lo * hi, [100.0**2])


def test_snapshot_does_not_use_future_fx(store, cfg):
    days = business_days(10)
    gold = [(d, 2000.0) for d in days[:-1]]
    fx = [(d, 80.0) for d in days[:-1]] + [(days[-1], 90.0)]
    store.upsert_observations(make_obs("gold_usd_oz", gold), "gold_usd_oz")
    store.upsert_observations(make_obs("usd_inr", fx), "usd_inr")
    snap = snapshot(store, cfg)
    assert snap.usd_inr == 80.0
    np.testing.assert_allclose(snap.parity_inr_10g, parity_inr_10g(2000.0, 80.0, cfg.troy_ounce_grams))


def test_scenario_identity_and_onshore_do_not_relabel_parity(store, cfg):
    days = business_days(120)
    gold = [(d, 2000.0) for d in days]
    fx = [(d, 80.0) for d in days]
    store.upsert_observations(make_obs("gold_usd_oz", gold), "gold_usd_oz")
    store.upsert_observations(make_obs("usd_inr", fx), "usd_inr")
    store.upsert_observations(derive_gold_inr_10g(store, cfg, run_id="fixture"), "gold_inr_10g")
    snap, elas, table = build(cfg, store, xau_pct=1.0, usdinr_pct=1.0, duty_pp=1.0, premium_pp=1.0)
    p0 = parity_inr_10g(2000.0, 80.0, cfg.troy_ounce_grams)
    np.testing.assert_allclose(snap.parity_inr_10g, p0)
    xau_row = table[table["scenario"].str.startswith("XAU/USD")].iloc[0]
    fx_row = table[table["scenario"].str.startswith("USD/INR")].iloc[0]
    both = table[table["scenario"].str.contains("combined")].iloc[0]
    np.testing.assert_allclose(xau_row["scenario_parity_inr_10g"], p0 * 1.01)
    np.testing.assert_allclose(fx_row["scenario_parity_inr_10g"], p0 * 1.01)
    np.testing.assert_allclose(both["scenario_parity_inr_10g"], p0 * 1.01 * 1.01)
    duty = table[table["kind"] == "onshore_addon"].iloc[0]
    assert duty["scenario_parity_inr_10g"] == p0
    assert "SCENARIO, NOT A FORECAST" in SCENARIO_DISCLAIMER
    assert "MCX" in SCENARIO_DISCLAIMER
    assert not table.empty


def test_run_scenarios_duty_delta_is_onshore_only():
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
            if dotted == "scenario":
                return {"gst_rate": 0.03}
            return default

    table = run_scenarios(snap, Cfg(), {}, duty_pp=1.0)
    duty = table[table["scenario"].str.contains("Import duty")].iloc[0]
    gst = 1.03
    expected = snap.parity_inr_10g * (0.01) * gst
    np.testing.assert_allclose(duty["delta_inr_10g"], expected)
    np.testing.assert_allclose(duty["scenario_parity_inr_10g"], snap.parity_inr_10g)
