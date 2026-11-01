from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gold.config import AppConfig
from gold.data.calendar import IndiaMarketCalendar
from gold.database.store import GoldStore, new_run_id
from gold.evaluation.metrics import reconstruct_price
from gold.features.builder import FeatureBuilder
from gold.forecasting.intervals import (
    causal_ewma_bounds,
    coverage_z,
    interval_method_for,
    residual_quantiles,
)
from gold.logging import get_logger
from gold.models.baselines import NaiveForecaster
from gold.training.orchestrator import HORIZON_MAP

log = get_logger("forecast")


class ForecastEngine:
    def __init__(self, cfg: AppConfig, store: GoldStore):
        self.cfg = cfg
        self.store = store
        self.calendar = IndiaMarketCalendar()
        self.features = FeatureBuilder(cfg, store)
        spec = cfg.get("forecast", {}) or {}
        self.low = float(spec.get("interval_low", 0.10))
        self.high = float(spec.get("interval_high", 0.90))

    def _load_champion(self, horizon: int):
        champs = self.store.champions()
        if champs.empty:
            return None, None
        row = champs[champs["horizon"] == horizon]
        if row.empty:
            return None, None
        rec = row.iloc[0]
        path = rec.get("artifact_path")
        model = None
        if path and Path(str(path)).exists():
            try:
                model = joblib.load(path)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load champion artifact %s: %s", path, exc)
        return rec, model

    def _fallback_model(self, y: pd.Series, X: pd.DataFrame):
        model = NaiveForecaster()
        model.fit(y, X)
        return model, "naive-fallback"

    def generate(self, horizons: list[int] | None = None, *, force: bool = False) -> pd.DataFrame:
        horizons = horizons or list(self.cfg.get("forecast.horizons_calendar_days", [1, 7, 30, 90]))
        horizons = [int(h) for h in horizons]
        feats = self.store.load_features()
        if feats.empty:
            feats = self.features.build()
        feats = feats.sort_values("obs_date")
        last = feats.iloc[-1]
        cutoff = pd.to_datetime(last["obs_date"]).date()
        if not force:
            existing = self.store.forecasts_for_cutoff(cutoff, horizons)
            have = set(existing["horizon_calendar_days"].astype(int)) if not existing.empty else set()
            if set(horizons) <= have:
                log.info("idempotent skip: live book already exists for cutoff %s horizons %s", cutoff, horizons)
                try:
                    from gold.ops.audit import audit_forecasts

                    audit_forecasts(self.store, existing, extra={"skipped": True, "reason": "duplicate_cutoff"})
                except Exception as exc:  # noqa: BLE001
                    log.warning("forecast skip audit failed: %s", exc)
                return existing
        generated_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        forecast_id = new_run_id("fc")
        rows = []

        target = self.store.load_series(self.cfg.primary_series)
        last_price = float(target.sort_values("obs_date")["value"].iloc[-1])

        for h in horizons:
            h_td = HORIZON_MAP.get(int(h), max(1, int(round(int(h) * 5 / 7))))
            rec, stored = self._load_champion(int(h))
            model_name = rec["model_name"] if rec is not None else "naive"
            model_version = rec["experiment_id"] if rec is not None else "untrained"
            X, y, meta = self.features.supervised(h_td)
            model = stored if stored is not None else None
            if model is None:
                model, model_name = self._fallback_model(y, X)
                model_version = "naive-fallback"
                log.warning("No champion for %sd; using persistence naive", h)

            # Refit on all labelled history through the latest origin, then forecast from last close.
            try:
                model.fit(y, X)
            except Exception as exc:  # noqa: BLE001
                log.warning("Live refit failed for %s h=%s: %s; using stored weights", model_name, h, exc)

            X_live = self.features.live_row(list(X.columns))
            X_future = X_live if getattr(model, "uses_features", False) else X_live
            ret_hat = float(model.predict(1, X_future=X_future)[0])
            if not np.isfinite(ret_hat):
                ret_hat = 0.0
            point = float(reconstruct_price([last_price], [ret_hat])[0])
            if not np.isfinite(point):
                point = last_price
            spec = self.cfg.get("forecast", {}) or {}
            method = interval_method_for(spec, int(h))
            z = coverage_z(self.low, self.high)
            if method == "ewma_sqrt_h":
                closes = target.sort_values("obs_date")["value"].astype(float).to_numpy()
                lower, upper, _sigma = causal_ewma_bounds(closes, last_price, ret_hat, h_td, z)
            else:
                scale_lo, scale_hi = residual_quantiles(self.store, model_name, int(h), self.low, self.high)
                if 0 < scale_lo < 1 and 0 < scale_hi < 1:
                    lower = point * (1 - scale_lo)
                    upper = point * (1 + scale_hi)
                else:
                    lower = point - scale_lo
                    upper = point + scale_hi
                method = "residual_rmse"

            target_cal = cutoff + timedelta(days=int(h))
            target_td = self.calendar.next_trading_day(target_cal)
            is_trading = self.calendar.is_trading_day(target_cal)
            rows.append(
                {
                    "forecast_id": forecast_id,
                    "generated_at": generated_at,
                    "model_name": model_name,
                    "model_version": str(model_version),
                    "target_series": self.cfg.primary_series,
                    "cutoff_date": cutoff,
                    "horizon_calendar_days": int(h),
                    "horizon_trading_days": int(h_td),
                    "target_date": target_td if not is_trading else target_cal,
                    "target_is_trading_day": bool(is_trading or True),
                    "point": point,
                    "lower": float(lower),
                    "upper": float(upper),
                    "interval_low": self.low,
                    "interval_high": self.high,
                    "interval_method": method,
                    "data_cutoff": generated_at,
                    "notes": (
                        f"Cutoff {cutoff}. Target is the {h}-day log return reconstructed as "
                        f"close[t]*exp(ŷ) = {last_price:.2f}*exp({ret_hat:.6f}). "
                        f"Interval method={method}. Macro features held at last available-as-of values "
                        f"(scenario: unchanged unpublished future prints). "
                        f"Horizon {h} calendar days ≈ {h_td} trading days. "
                        f"Target is international parity {self.cfg.primary_series}, not an official MCX/IBJA print."
                    ),
                }
            )
        out = pd.DataFrame(rows)
        out["target_is_trading_day"] = [
            self.calendar.is_trading_day(d) if not isinstance(d, datetime) else self.calendar.is_trading_day(d.date())
            for d in pd.to_datetime(out["target_date"]).dt.date
        ]
        self.store.save_forecasts(out)
        try:
            from gold.ops.audit import audit_forecasts

            audit_forecasts(
                self.store,
                out,
                extra={
                    "pipeline_version": (self.cfg.get("training", {}) or {}).get("pipeline_version"),
                    "interval_note": "1d uses ewma_sqrt_h when configured; other horizons residual RMSE",
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("forecast audit failed: %s", exc)
        log.info("generated %s forecasts (%s)", len(out), forecast_id)
        return out


if __name__ == "__main__":
    from gold.config import load_config
    from gold.database.store import GoldStore
    from gold.logging import setup_logging

    cfg = load_config()
    setup_logging(cfg.log_dir)
    print(ForecastEngine(cfg, GoldStore(cfg)).generate())
