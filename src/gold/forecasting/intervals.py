from __future__ import annotations

import json

import numpy as np

from gold.database.store import GoldStore
from gold.forecasting.volatility import Z_SCORES, ewma_variance, price_interval


def residual_quantiles(store: GoldStore, model_name: str, horizon: int, low: float, high: float) -> tuple[float, float]:
    """Empirical interval from holdout residuals stored in experiment metrics when available.

    Fallback: 1% / 2% of last price for 10/90 if residuals are missing.
    Scale is absolute INR (holdout RMSE of reconstructed prices).
    """
    exps = store.experiments()
    if exps.empty:
        return 0.01, 0.02
    subset = exps[(exps["model_name"] == model_name) & (exps["horizon"] == horizon)]
    if subset.empty:
        subset = exps[exps["horizon"] == horizon]
    if subset.empty:
        return 0.01, 0.02
    metrics = subset.iloc[0]["metrics"]
    if isinstance(metrics, str):
        metrics = json.loads(metrics)
    holdout = metrics.get("holdout") or {}
    rmse = holdout.get("rmse")
    if rmse is None or not np.isfinite(rmse):
        return 0.01, 0.02
    z_low, z_high = 1.2816, 1.2816
    return float(z_low * rmse), float(z_high * rmse)


def coverage_z(low: float, high: float) -> float:
    cov = float(high) - float(low)
    key = min(Z_SCORES, key=lambda k: abs(k - cov))
    if abs(key - cov) > 0.06:
        return 1.2816
    return float(Z_SCORES[key])


def causal_ewma_bounds(
    close: np.ndarray | pd.Series,
    last_price: float,
    ret_hat: float,
    h_td: int,
    z: float,
    lam: float = 0.94,
) -> tuple[float, float, float]:
    """Log-return EWMA σ_t × √h, mapped to INR via exp. Uses closes through t only."""
    c = np.asarray(close, dtype=float)
    c = c[np.isfinite(c)]
    if len(c) < 10:
        lo, hi = last_price * 0.99, last_price * 1.01
        return float(lo), float(hi), float("nan")
    daily = np.diff(np.log(np.clip(c, 1e-12, None)))
    sigma1 = float(np.sqrt(ewma_variance(daily, lam=lam)[-1]))
    sigma = sigma1 * np.sqrt(max(int(h_td), 1))
    lo, hi = price_interval([last_price], [ret_hat], [sigma], z)
    return float(lo[0]), float(hi[0]), sigma


def interval_method_for(cfg_forecast: dict, horizon: int) -> str:
    by_h = cfg_forecast.get("interval_method_by_horizon") or {}
    if str(horizon) in by_h:
        return str(by_h[str(horizon)])
    if horizon in by_h:
        return str(by_h[horizon])
    return str(cfg_forecast.get("interval_method") or "residual_rmse")
