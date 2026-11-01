from __future__ import annotations

import numpy as np

from gold.features.builder import FeatureBuilder
from gold.forecasting.engine import ForecastEngine
from gold.ingestion.derived import derive_gold_inr_10g
from gold.training.orchestrator import TrainingOrchestrator
from tests.helpers import business_days, make_obs


def test_features_train_forecast_on_fixture_series(store, cfg):
    """End-to-end on labelled test fixtures — not real market data."""
    cfg.raw.setdefault("training", {})
    cfg.raw["training"]["min_train_trading_days"] = 50
    cfg.raw["training"]["holdout_trading_days"] = 20
    cfg.raw["training"]["walk_forward_step"] = 15
    cfg.raw["training"]["walk_forward_test_size"] = 5
    cfg.raw["forecast"] = {"horizons_calendar_days": [1, 7], "interval_low": 0.1, "interval_high": 0.9}

    days = business_days(180)
    gold = [(d, 1800 + i * 0.4 + 5 * np.sin(i / 8)) for i, d in enumerate(days)]
    fx = [(d, 74 + 0.01 * i) for i, d in enumerate(days)]
    store.upsert_observations(make_obs("gold_usd_oz", gold), "gold_usd_oz")
    store.upsert_observations(make_obs("usd_inr", fx), "usd_inr")
    derived = derive_gold_inr_10g(store, cfg, run_id="fixture")
    store.upsert_observations(derived, "gold_inr_10g")

    builder = FeatureBuilder(cfg, store)
    feats = builder.build()
    assert len(feats) > 80
    assert "close_lag_1" not in feats.columns
    assert "roll_mean_5" not in feats.columns
    assert "ret_lag_1" in feats.columns
    assert "roll_vol_5" in feats.columns
    assert feats["ret_lag_1"].notna().all()

    X, y, meta = builder.supervised(1)
    assert float(y.abs().max()) < 1.0
    np.testing.assert_allclose(meta["target_price"], meta["close"] * np.exp(y), rtol=1e-8)

    results = TrainingOrchestrator(cfg, store).train_all(models=["naive", "moving_average", "xgboost"])
    assert not results.empty
    assert set(results["horizon"]) <= {1, 7}
    champs = store.champions()
    assert not champs.empty

    fc = ForecastEngine(cfg, store).generate(horizons=[1, 7])
    assert len(fc) == 2
    assert fc["point"].notna().all()
    assert (fc["upper"] >= fc["lower"]).all()
    assert fc["model_version"].notna().all()
