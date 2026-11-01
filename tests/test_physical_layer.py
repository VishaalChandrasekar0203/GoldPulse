from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from gold.ingestion.derived import derive_gold_inr_10g
from gold.intelligence.physical import (
    apply_quotes,
    estimated_physical,
    hallmarked_metal,
    physical_base,
    physical_gold_premium,
    tax_wedge_pct,
)
from tests.helpers import business_days, make_obs


def _seed_market(
    store,
    cfg,
    n: int = 120,
    xau0: float = 2000.0,
    fx0: float = 80.0,
    xau_step: float = 0.0,
    fx_step: float = 0.0,
    start: date | None = None,
):
    start = start or (date.today() - timedelta(days=int(n * 1.7) + 10))
    days = business_days(n, start=start)
    xau = [(d, xau0 + i * xau_step) for i, d in enumerate(days)]
    fx = [(d, fx0 + i * fx_step) for i, d in enumerate(days)]
    store.upsert_observations(make_obs("gold_usd_oz", xau), "gold_usd_oz")
    store.upsert_observations(make_obs("usd_inr", fx), "usd_inr")
    store.upsert_observations(make_obs("gold_import_duty", [(days[0], 6.0), (days[-1], 6.0)]), "gold_import_duty")
    derived = derive_gold_inr_10g(store, cfg, run_id="t")
    store.upsert_observations(derived, "gold_inr_10g")


def test_premium_is_dealer_minus_parity():
    assert physical_gold_premium(130_000.0, 119_276.0) == 130_000.0 - 119_276.0
    assert physical_gold_premium(None, 119_276.0) is None
    assert physical_gold_premium(0.0, 119_276.0) is None


def test_estimated_physical_applies_duty_and_gst():
    parity = 100_000.0
    got = estimated_physical(parity, duty_pct=6.0, gst_pct=3.0, making_pct=0.0)
    assert abs(got - 100_000.0 * 1.06 * 1.03) < 1e-6
    assert abs(tax_wedge_pct(6.0, 3.0, 0.0) - 100.0 * (1.06 * 1.03 - 1.0)) < 1e-9


def test_22k_hallmark_scales_fine_gold():
    parity = 100_000.0
    metal = hallmarked_metal(parity, 916)
    assert abs(metal - 100_000.0 * 916 / 999) < 1e-6
    got = estimated_physical(parity, duty_pct=6.0, gst_pct=3.0, making_pct=0.0, hallmark_fineness=916)
    assert abs(got - metal * 1.06 * 1.03) < 1e-6
    layer = apply_quotes({"parity_inr_10g": parity, "duty_pct": 6.0, "data_ok": True}, hallmark_fineness=916)
    assert layer["quotes"]["hallmark_fineness"] == 916
    np.testing.assert_allclose(layer["quotes"]["hallmarked_metal_inr_10g"], metal)


def test_history_uses_full_sample_not_one_year(store, cfg):
    _seed_market(store, cfg, n=300, xau_step=0.5, start=date(2010, 1, 4))
    base = physical_base(store, cfg)
    assert base["n_obs"] == 300
    assert base["history_start"] == "2010-01-04"
    assert base["history_years"] is not None and base["history_years"] > 1.0
    assert "2010-01-01" in base["history_note"]
    assert "1-year sample" in base["history_note"] or "not a 1-year" in base["history_note"].lower()


def test_wait_when_near_highs_and_above_averages(store, cfg):
    _seed_market(store, cfg, n=120, xau_step=2.0)
    base = physical_base(store, cfg)
    layer = apply_quotes(base, dealer_price=None)
    assert layer["drawdown"]["90"] > -0.015
    assert layer["stance"]["label"] in {"Wait", "Review"}
    fired = {s["id"]: s["fires"] for s in layer["stance"]["signals"]}
    assert fired["near_90d_high"] is True
    assert fired["above_all_mas"] is True


def test_buy_when_drawn_down_below_longer_averages(store, cfg):
    _seed_market(store, cfg, n=120, xau0=2600.0, xau_step=-2.0)
    base = physical_base(store, cfg)
    layer = apply_quotes(base)
    fired = {s["id"]: s["fires"] for s in layer["stance"]["signals"]}
    assert fired["off_90d_high"] is True
    assert fired["below_longer_mas"] is True
    assert layer["stance"]["label"] in {"Buy", "Review"}
    assert layer["stance"]["buy_hits"] >= 2


def test_fat_premium_is_wait_and_formula_is_dealer_minus_parity(store, cfg):
    _seed_market(store, cfg, n=120, xau_step=0.2)
    base = physical_base(store, cfg)
    parity = base["parity_inr_10g"]
    dealer = parity * 1.15
    layer = apply_quotes(base, dealer_price=dealer, gst_pct=3.0, making_pct=0.0)
    quotes = layer["quotes"]
    np.testing.assert_allclose(quotes["physical_premium_inr"], dealer - parity)
    fired = {s["id"]: s["fires"] for s in layer["stance"]["signals"]}
    assert fired["fat_premium"] is True
    assert "Dealer Price" in quotes["premium_formula"]


def test_missing_series_is_review(store, cfg):
    base = physical_base(store, cfg)
    layer = apply_quotes(base)
    assert base["data_ok"] is False
    assert layer["stance"]["label"] == "Review"
    fired = {s["id"]: s["fires"] for s in layer["stance"]["signals"]}
    assert fired["data_gap"] is True


def test_ui_payload_includes_physical(store, cfg):
    from gold.web.load import load_ui_payload

    _seed_market(store, cfg, n=120, xau_step=0.4)
    payload = load_ui_payload(cfg=cfg, store=store)
    assert payload["physical"]["n_obs"] == 120
    assert payload["physical"]["parity_inr_10g"] > 0
    assert "parity" in payload
    assert payload["forecast_book"] == payload["brief"]["forecast_book"]
