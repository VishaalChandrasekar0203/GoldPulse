"""Score stored and production-policy forecasts against realized parity. No new models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd

from gold.config import AppConfig
from gold.database.store import GoldStore, new_run_id
from gold.forecasting.intervals import causal_ewma_bounds, coverage_z
from gold.forecasting.volatility import completed_h_day_return, price_interval, winkler_score
from gold.logging import get_logger
from gold.training.orchestrator import HORIZON_MAP

log = get_logger("monitor")


def _alpha(low: float, high: float) -> float:
    return max(1e-6, 1.0 - (float(high) - float(low)))


def _direction(origin, point, realized) -> bool | None:
    o, p, r = float(origin), float(point), float(realized)
    actual = np.sign(r - o)
    pred = np.sign(p - o)
    if actual == 0:
        return None
    return bool(actual == pred)


def _winkler_one(realized, lo, hi, alpha: float) -> float:
    return winkler_score([realized], [lo], [hi], alpha)


def _realized_on(target: pd.DataFrame, day) -> float | None:
    if target.empty:
        return None
    d = pd.to_datetime(day).date()
    hit = target[target["obs_date"] == d]
    if hit.empty:
        later = target[target["obs_date"] >= d]
        if later.empty:
            return None
        return float(later.sort_values("obs_date").iloc[0]["value"])
    return float(hit.iloc[0]["value"])


def _origin_close(target: pd.DataFrame, cutoff) -> float | None:
    if target.empty:
        return None
    d = pd.to_datetime(cutoff).date()
    hit = target[target["obs_date"] == d]
    if hit.empty:
        prior = target[target["obs_date"] <= d]
        if prior.empty:
            return None
        return float(prior.sort_values("obs_date").iloc[-1]["value"])
    return float(hit.iloc[0]["value"])


def score_live_forecasts(cfg: AppConfig, store: GoldStore) -> pd.DataFrame:
    """Match production forecasts to realized gold_inr_10g. Does not invent missing prints."""
    fc = store.all_forecasts()
    target = store.load_series(cfg.primary_series).sort_values("obs_date")
    if fc.empty or target.empty:
        return pd.DataFrame()
    existing = store.forecast_outcomes()
    seen = set()
    seen_ids: set[str] = set()
    if not existing.empty:
        seen = set(zip(existing["forecast_id"].astype(str), existing["horizon_calendar_days"].astype(int)))
        seen_ids = set(existing["outcome_id"].astype(str))
    last_obs = max(target["obs_date"])
    spec = cfg.get("forecast", {}) or {}
    low = float(spec.get("interval_low", 0.10))
    high = float(spec.get("interval_high", 0.90))
    alpha = _alpha(low, high)
    rows = []
    scored_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    for rec in fc.itertuples():
        h = int(rec.horizon_calendar_days)
        key = (str(rec.forecast_id), h)
        oid = f"live_{rec.forecast_id}_{h}"
        if key in seen or oid in seen_ids:
            continue
        target_date = pd.to_datetime(rec.target_date).date()
        if target_date > last_obs:
            continue
        realized = _realized_on(target, target_date)
        origin = _origin_close(target, rec.cutoff_date)
        if realized is None or origin is None:
            continue
        lo, hi, point = float(rec.lower), float(rec.upper), float(rec.point)
        method = "ewma_sqrt_h" if h == 1 else "residual_rmse"
        notes = str(getattr(rec, "notes", "") or "")
        if "ewma_sqrt_h" in notes:
            method = "ewma_sqrt_h"
        elif "residual_rmse" in notes:
            method = "residual_rmse"
        direction = _direction(origin, point, realized)
        rows.append(
            {
                "outcome_id": f"live_{rec.forecast_id}_{h}",
                "forecast_id": str(rec.forecast_id),
                "source": "live",
                "model_name": rec.model_name,
                "model_version": str(rec.model_version),
                "target_series": rec.target_series,
                "cutoff_date": pd.to_datetime(rec.cutoff_date).date(),
                "target_date": target_date,
                "horizon_calendar_days": h,
                "interval_method": method,
                "origin_close": origin,
                "point": point,
                "lower": lo,
                "upper": hi,
                "realized": realized,
                "abs_error": abs(realized - point),
                "signed_error": realized - point,
                "covered": bool(lo <= realized <= hi),
                "winkler": _winkler_one(realized, lo, hi, alpha),
                "direction_correct": direction,
                "scored_at": scored_at,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        store.save_forecast_outcomes(out)
        log.info("scored %s live forecasts", len(out))
    return out


def backfill_production_policy(cfg: AppConfig, store: GoldStore) -> pd.DataFrame:
    """Leak-free history of the production policy: naive point, EWMA 1d, train σ otherwise.

    7d live champion is ARIMA (0.04% near-tie). Monitor backfill uses persistence so we
    do not refit ARIMA at every origin. Live 7d rows are scored separately when realized.
    """
    spec = cfg.get("monitor", {}) or {}
    n_orig = int(spec.get("backfill_origins", 80))
    step = int(spec.get("backfill_step", 5))
    target = store.load_series(cfg.primary_series).sort_values("obs_date")
    if len(target) < 600:
        return pd.DataFrame()
    existing = store.forecast_outcomes()
    have = set()
    if not existing.empty:
        back = existing[existing["source"] == "production_policy_backfill"]
        if not back.empty:
            have = set(zip(back["cutoff_date"].astype(str), back["horizon_calendar_days"].astype(int)))
    close = target["value"].astype(float).to_numpy()
    dates = pd.to_datetime(target["obs_date"]).to_numpy()
    fspec = cfg.get("forecast", {}) or {}
    low = float(fspec.get("interval_low", 0.10))
    high = float(fspec.get("interval_high", 0.90))
    z = coverage_z(low, high)
    alpha = _alpha(low, high)
    min_i = 504
    last_ok = len(close) - max(HORIZON_MAP.values()) - 1
    origins = list(range(min_i, last_ok, step))[-n_orig:]
    scored_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    rows = []
    for i in origins:
        cutoff = pd.Timestamp(dates[i]).date()
        origin_px = float(close[i])
        hist = close[: i + 1]
        for h, h_td in HORIZON_MAP.items():
            if i + h_td >= len(close):
                continue
            key = (str(cutoff), int(h))
            if key in have:
                continue
            realized = float(close[i + h_td])
            point = origin_px
            if h == 1:
                lo, hi, _ = causal_ewma_bounds(hist, origin_px, 0.0, h_td, z)
                method = "ewma_sqrt_h"
            else:
                y = completed_h_day_return(hist, h_td)
                sigma = float(np.nanstd(y, ddof=1)) if np.isfinite(y).sum() > 10 else 0.02
                lo_a, hi_a = price_interval([origin_px], [0.0], [sigma], z)
                lo, hi = float(lo_a[0]), float(hi_a[0])
                method = "const_train"
            rows.append(
                {
                    "outcome_id": f"mon_{cutoff}_{h}",
                    "forecast_id": f"mon_{cutoff}",
                    "source": "production_policy_backfill",
                    "model_name": "naive",
                    "model_version": "production-policy-monitor",
                    "target_series": cfg.primary_series,
                    "cutoff_date": cutoff,
                    "target_date": pd.Timestamp(dates[i + h_td]).date(),
                    "horizon_calendar_days": int(h),
                    "interval_method": method,
                    "origin_close": origin_px,
                    "point": point,
                    "lower": float(lo),
                    "upper": float(hi),
                    "realized": realized,
                    "abs_error": abs(realized - point),
                    "signed_error": realized - point,
                    "covered": bool(lo <= realized <= hi),
                    "winkler": _winkler_one(realized, lo, hi, alpha),
                    "direction_correct": _direction(origin_px, point, realized),
                    "scored_at": scored_at,
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        store.save_forecast_outcomes(out)
        log.info("backfilled %s production-policy outcomes", len(out))
    return out


def rolling_metrics(outcomes: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    if outcomes.empty:
        return pd.DataFrame()
    rows = []
    for h, g in outcomes.groupby("horizon_calendar_days"):
        g = g.sort_values("cutoff_date")
        tail = g.tail(int(window))
        da = tail["direction_correct"].dropna()
        rows.append(
            {
                "horizon": int(h),
                "n": int(len(tail)),
                "mae": float(tail["abs_error"].mean()),
                "directional_accuracy": float(da.mean()) if len(da) else float("nan"),
                "coverage": float(tail["covered"].mean()),
                "winkler": float(tail["winkler"].mean()),
                "interval_method": str(tail["interval_method"].iloc[-1]),
                "source": ",".join(sorted(tail["source"].unique())),
            }
        )
    return pd.DataFrame(rows)


def raise_alerts(cfg: AppConfig, store: GoldStore, roll: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    spec = cfg.get("monitor", {}) or {}
    floors = spec.get("coverage_alert_floor") or {1: 0.70, 7: 0.65, 30: 0.60, 90: 0.55}
    holdout = spec.get("holdout_mae") or {1: 1782, 7: 3942, 30: 7827, 90: 14522}
    multiple = float(spec.get("mae_alert_multiple", 1.5))
    created = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    rows = []

    def add(severity, kind, message, horizon=None, series_id=None, payload=None):
        rows.append(
            {
                "alert_id": f"al_{created.strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}",
                "created_at": created,
                "severity": severity,
                "kind": kind,
                "horizon": horizon,
                "series_id": series_id,
                "message": message,
                "payload": json.dumps(payload or {}),
            }
        )

    for rec in roll.itertuples():
        h = int(rec.horizon)
        floor = float(floors.get(h, floors.get(str(h), 0.60)))
        base = float(holdout.get(h, holdout.get(str(h), rec.mae)))
        if rec.mae > multiple * base:
            add(
                "warning",
                "mae_deterioration",
                f"{h}d rolling MAE {rec.mae:.0f} exceeds {multiple:.1f}× holdout {base:.0f}",
                horizon=h,
                payload={"mae": rec.mae, "holdout": base},
            )
        if rec.coverage < floor:
            add(
                "warning",
                "coverage_deterioration",
                f"{h}d rolling coverage {rec.coverage:.1%} below floor {floor:.0%}",
                horizon=h,
                payload={"coverage": rec.coverage, "floor": floor},
            )
    if not quality.empty:
        stale = quality[quality["stale"] == True]  # noqa: E712
        for rec in stale.itertuples():
            add(
                "warning",
                "stale_series",
                f"{rec.series_id} last print {rec.end_date} looks stale",
                series_id=rec.series_id,
            )
        restricted = quality[quality["series_id"].isin(["mcx_gold_futures", "ibja_gold"])]
        if restricted.empty:
            add(
                "info",
                "restricted_absent",
                "MCX and IBJA remain unavailable and were not fabricated.",
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        store.save_alerts(out)
    return out


def run_monitor(cfg: AppConfig, store: GoldStore, backfill: bool = True) -> dict:
    live = score_live_forecasts(cfg, store)
    hist = backfill_production_policy(cfg, store) if backfill else pd.DataFrame()
    outcomes = store.forecast_outcomes()
    window = int((cfg.get("monitor", {}) or {}).get("rolling_window", 21))
    roll = rolling_metrics(outcomes, window=window)
    from gold.data.quality import capture_quality_snapshot

    quality = capture_quality_snapshot(cfg, store)
    alerts = raise_alerts(cfg, store, roll, quality)
    charts = _write_charts(cfg, outcomes)
    return {
        "n_live_scored": int(len(live)),
        "n_backfill": int(len(hist)),
        "n_outcomes": int(len(outcomes)),
        "rolling": roll,
        "alerts": alerts,
        "quality": quality,
        "charts": charts,
        "run_id": new_run_id("mon"),
    }


def _write_charts(cfg: AppConfig, outcomes: pd.DataFrame) -> list[str]:
    if outcomes.empty:
        return []
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", str(cfg.reports_dir / ".mpl"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        log.warning("monitor charts skipped: %s", exc)
        return []
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for h, g in outcomes.groupby("horizon_calendar_days"):
        g = g.sort_values("cutoff_date")
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(pd.to_datetime(g["cutoff_date"]), g["abs_error"], lw=1.0, label="Absolute error (INR/10g)")
        roll = g["abs_error"].rolling(21, min_periods=5).mean()
        ax.plot(pd.to_datetime(g["cutoff_date"]), roll, lw=1.4, label="21-origin rolling MAE")
        ax.set_title(f"{int(h)}d production-policy absolute error vs realized parity")
        ax.set_ylabel("INR per 10g")
        ax.set_xlabel("Forecast origin")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = cfg.reports_dir / f"monitor_mae_{int(h)}d.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(str(path))
        fig, ax = plt.subplots(figsize=(9, 3.2))
        cov = g["covered"].astype(float).rolling(21, min_periods=5).mean()
        ax.plot(pd.to_datetime(g["cutoff_date"]), cov, lw=1.4, label="21-origin rolling coverage")
        ax.axhline(0.8, color="black", ls="--", lw=0.8, label="80% nominal")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{int(h)}d interval coverage vs realized parity")
        ax.set_ylabel("Coverage")
        ax.set_xlabel("Forecast origin")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = cfg.reports_dir / f"monitor_coverage_{int(h)}d.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(str(path))
    return paths
