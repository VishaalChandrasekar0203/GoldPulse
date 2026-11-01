"""Expanding-window evaluation across documented historical gold regimes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gold.evaluation.metrics import reconstruct_price, regression_metrics
from gold.models.baselines import NaiveForecaster
from gold.models.base import BaseForecaster

DEFAULT_REGIMES: list[dict[str, str]] = [
    {
        "id": "taper_2014",
        "label": "2013-2014 taper",
        "test_start": "2014-07-01",
        "test_end": "2014-12-31",
    },
    {
        "id": "range_2018",
        "label": "2016-2018 range",
        "test_start": "2018-07-01",
        "test_end": "2018-12-31",
    },
    {
        "id": "covid_2020",
        "label": "2020 covid",
        "test_start": "2020-07-01",
        "test_end": "2020-12-31",
    },
    {
        "id": "hikes_2022",
        "label": "2022 rate hikes",
        "test_start": "2022-07-01",
        "test_end": "2022-12-31",
    },
    {
        "id": "bull_2026",
        "label": "2024-2026 bull",
        "test_start": "2026-05-01",
        "test_end": "2026-10-31",
    },
]


def _predict_aligned(model: BaseForecaster, n: int, X_test: pd.DataFrame | None) -> np.ndarray:
    pred = np.asarray(model.predict(n, X_future=X_test), dtype=float)[:n]
    if len(pred) < n:
        fill = pred[-1] if len(pred) else 0.0
        pred = np.pad(pred, (0, n - len(pred)), constant_values=fill)
    return pred


def _price_mae(close: np.ndarray, y_ret: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    price_true = reconstruct_price(close, y_ret)
    price_pred = reconstruct_price(close, y_hat)
    return regression_metrics(price_true, price_pred, close)


def expanding_regime_evaluate(
    model_factory,
    y: pd.Series,
    X: pd.DataFrame | None,
    meta: pd.DataFrame,
    regimes: list[dict[str, str]] | None = None,
    min_train: int = 252,
) -> dict[str, object]:
    """Train on all history before each regime test window; score reconstructed prices."""
    regimes = regimes or DEFAULT_REGIMES
    obs = pd.to_datetime(meta["obs_date"])
    close_all = pd.to_numeric(meta["close"], errors="coerce").to_numpy(dtype=float)
    y_all = np.asarray(y, dtype=float)
    details: list[dict[str, object]] = []
    n_beat = 0
    n_eval = 0

    for spec in regimes:
        ts = pd.Timestamp(spec["test_start"])
        te = pd.Timestamp(spec["test_end"])
        train_idx = np.flatnonzero(obs < ts)
        test_idx = np.flatnonzero((obs >= ts) & (obs <= te))
        if len(train_idx) < min_train or len(test_idx) < 10:
            details.append(
                {
                    "id": spec["id"],
                    "label": spec.get("label", spec["id"]),
                    "skipped": True,
                    "reason": f"train={len(train_idx)} test={len(test_idx)} (need train>={min_train}, test>=10)",
                }
            )
            continue
        X_train = X.iloc[train_idx] if X is not None else None
        X_test = X.iloc[test_idx] if X is not None else None
        y_train = y.iloc[train_idx]
        y_test = y_all[test_idx]
        close_test = close_all[test_idx]
        model: BaseForecaster = model_factory()
        model.fit(y_train, X_train)
        pred = _predict_aligned(model, len(test_idx), X_test)
        naive = NaiveForecaster()
        naive.fit(y_train, X_train)
        naive_pred = _predict_aligned(naive, len(test_idx), X_test)
        model_m = _price_mae(close_test, y_test, pred)
        naive_m = _price_mae(close_test, y_test, naive_pred)
        beat = bool(model_m["mae"] < naive_m["mae"] - 1e-12)
        n_eval += 1
        n_beat += int(beat)
        details.append(
            {
                "id": spec["id"],
                "label": spec.get("label", spec["id"]),
                "skipped": False,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "test_start": str(pd.to_datetime(meta["obs_date"].iloc[test_idx[0]]).date()),
                "test_end": str(pd.to_datetime(meta["obs_date"].iloc[test_idx[-1]]).date()),
                "mae": float(model_m["mae"]),
                "rmse": float(model_m["rmse"]),
                "mape": float(model_m["mape"]),
                "naive_mae": float(naive_m["mae"]),
                "mae_vs_naive": float(model_m["mae"] / naive_m["mae"]) if naive_m["mae"] else None,
                "beats_naive": beat,
            }
        )

    return {
        "regimes_beat": int(n_beat),
        "regimes_evaluated": int(n_eval),
        "details": details,
    }
