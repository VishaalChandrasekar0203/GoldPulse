"""Causal volatility scales for prediction intervals. No look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd

Z_SCORES = {
    0.80: 1.2815515655446004,
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
}


def ewma_variance(returns: np.ndarray, lam: float = 0.94, warmup: int = 60) -> np.ndarray:
    """RiskMetrics EWMA variance. var[t] uses r[0..t] only."""
    r = np.asarray(returns, dtype=float)
    n = len(r)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    w = min(max(warmup, 5), n)
    finite = r[:w][np.isfinite(r[:w])]
    var = float(np.var(finite)) if len(finite) else 1e-8
    var = max(var, 1e-12)
    for i, x in enumerate(r):
        if np.isfinite(x):
            var = lam * var + (1.0 - lam) * x * x
        out[i] = var
    return out


def empirical_h_exponent(daily: np.ndarray, h_td: int, y_h: np.ndarray) -> float:
    """Estimate λ in σ_h ≈ σ_1 * h^λ from train RMS, using known-at-t series only."""
    d = np.asarray(daily, dtype=float)
    y = np.asarray(y_h, dtype=float)
    n = min(len(d), len(y))
    d, y = d[:n], y[:n]
    mask = np.isfinite(d) & np.isfinite(y)
    if mask.sum() < 40 or h_td <= 1:
        return 0.5
    rms1 = float(np.sqrt(np.mean(d[mask] ** 2)))
    rmsh = float(np.sqrt(np.mean(y[mask] ** 2)))
    if rms1 <= 0 or rmsh <= 0:
        return 0.5
    return float(np.log(rmsh / rms1) / np.log(h_td))


def winkler_score(y, lo, hi, alpha: float) -> float:
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def interval_coverage(y, lo, hi) -> float:
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def price_interval(close, ret_hat, sigma, z: float) -> tuple[np.ndarray, np.ndarray]:
    close = np.asarray(close, dtype=float)
    ret_hat = np.asarray(ret_hat, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    lo = close * np.exp(ret_hat - z * sigma)
    hi = close * np.exp(ret_hat + z * sigma)
    return lo, hi


def completed_h_day_return(close: np.ndarray, h_td: int) -> np.ndarray:
    """r[t] = log(close[t]/close[t-h]), known at t. Leading h values are NaN."""
    c = np.asarray(close, dtype=float)
    out = np.full(len(c), np.nan)
    if h_td < 1 or len(c) <= h_td:
        return out
    out[h_td:] = np.log(c[h_td:] / np.clip(c[:-h_td], 1e-12, None))
    return out
