from __future__ import annotations

import numpy as np
import pandas as pd

from gold.evaluation.metrics import score_log_return_forecast
from gold.models.base import BaseForecaster


def chronological_split(
    n: int, holdout: int
) -> tuple[np.ndarray, np.ndarray]:
    if holdout >= n:
        raise ValueError("Holdout is larger than the sample")
    idx = np.arange(n)
    return idx[:-holdout], idx[-holdout:]


def walk_forward_evaluate(
    model_factory,
    y: pd.Series,
    X: pd.DataFrame | None,
    meta: pd.DataFrame,
    min_train: int,
    test_size: int,
    step: int,
    max_folds: int = 8,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Expanding-origin folds. `y` is the h-day log return; metrics are reconstructed prices."""
    n = len(y)
    preds = []
    actuals = []
    prevs = []
    dates = []
    start = min_train
    folds = 0
    while start + test_size <= n and folds < max_folds:
        train_slice = slice(0, start)
        test_slice = slice(start, start + test_size)
        model: BaseForecaster = model_factory()
        y_train = y.iloc[train_slice]
        X_train = X.iloc[train_slice] if X is not None else None
        model.fit(y_train, X_train)
        X_test = X.iloc[test_slice] if X is not None else None
        y_test = y.iloc[test_slice]
        pred = model.predict(len(y_test), X_future=X_test)
        pred = np.asarray(pred, dtype=float)[: len(y_test)]
        if len(pred) < len(y_test):
            fill = pred[-1] if len(pred) else 0.0
            pred = np.pad(pred, (0, len(y_test) - len(pred)), constant_values=fill)
        preds.append(pred)
        actuals.append(np.asarray(y_test, dtype=float))
        prevs.append(np.asarray(meta["close"].iloc[test_slice], dtype=float))
        dates.append(pd.to_datetime(meta["target_date"].iloc[test_slice]))
        start += step
        folds += 1
    if not preds:
        return {"n": 0}, pd.DataFrame()
    y_hat = np.concatenate(preds)
    y_true = np.concatenate(actuals)
    y_prev = np.concatenate(prevs)
    target_dates = pd.concat(dates, ignore_index=True)
    scored = score_log_return_forecast(y_prev, y_true, y_hat)
    metrics = dict(scored["price"])
    metrics["return_mae"] = scored["return"]["mae"]
    metrics["return_rmse"] = scored["return"]["rmse"]
    detail = pd.DataFrame(
        {
            "target_date": target_dates,
            "y_true": y_true,
            "y_pred": y_hat,
            "y_prev": y_prev,
            "price_true": scored["price_true"],
            "price_pred": scored["price_pred"],
        }
    )
    return metrics, detail
