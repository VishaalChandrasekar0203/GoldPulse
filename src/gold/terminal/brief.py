"""Assemble the daily intelligence brief from stored series. No new models."""

from __future__ import annotations

from datetime import datetime, timezone

from gold.config import AppConfig
from gold.database.store import GoldStore
from gold.evaluation.monitor import rolling_metrics
from gold.forecasting.volatility import ewma_variance
from gold.intelligence.dashboard import market_dashboard
from gold.logging import get_logger
from gold.ops.book import forecast_fingerprint
from gold.scenarios.engine import SCENARIO_DISCLAIMER, build_packs, snapshot

log = get_logger("brief")


def _vol_1d(store: GoldStore, series_id: str) -> float | None:
    df = store.load_series(series_id)
    if df.empty or len(df) < 40:
        return None
    import numpy as np

    v = df.sort_values("obs_date")["value"].astype(float)
    r = np.log(v / v.shift(1)).fillna(0.0).to_numpy()
    return float(np.sqrt(ewma_variance(r)[-1]))


def build_brief(cfg: AppConfig, store: GoldStore, run_monitor: bool = False) -> dict:
    if run_monitor:
        from gold.evaluation.monitor import run_monitor as _mon

        _mon(cfg, store, backfill=True)
    try:
        snap = snapshot(store, cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("snapshot failed: %s", exc)
        snap = None
    forecasts = store.latest_forecasts()
    champs = store.champions()
    outcomes = store.forecast_outcomes()
    window = int((cfg.get("monitor", {}) or {}).get("rolling_window", 21))
    roll = rolling_metrics(outcomes, window=window)
    alerts = store.alerts(limit=20)
    intel = market_dashboard(cfg, store)
    packs = None
    elas = {}
    if snap is not None:
        try:
            _, elas, packs = build_packs(cfg, store)
        except Exception as exc:  # noqa: BLE001
            log.warning("scenario packs failed: %s", exc)
    vol = _vol_1d(store, cfg.primary_series)
    fc_1d = None
    if not forecasts.empty:
        hit = forecasts[forecasts["horizon_calendar_days"] == 1]
        if not hit.empty:
            fc_1d = hit.iloc[0].to_dict()
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "snapshot": snap,
        "parity_vol_ewma": vol,
        "forecasts": forecasts,
        "forecast_book": forecast_fingerprint(forecasts),
        "forecast_1d": fc_1d,
        "champions": champs,
        "rolling": roll,
        "alerts": alerts,
        "intelligence": intel,
        "scenario_packs": packs,
        "elasticities": elas,
        "disclaimer_scenarios": SCENARIO_DISCLAIMER,
        "target_note": (
            "Primary target is international parity gold_inr_10g, not MCX or IBJA. "
            "GOLDBEES is an onshore ETF proxy. Production champions: naive 1d/30d/90d, ARIMA 7d. "
            "1d interval is causal EWMA × √h."
        ),
    }
