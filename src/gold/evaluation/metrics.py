from __future__ import annotations

import numpy as np
import pandas as pd


def _arr(x: pd.Series | np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(y_true, y_pred) -> float:
    yt, yp = _arr(y_true), _arr(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _arr(y_true), _arr(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true, y_pred) -> float:
    yt, yp = _arr(y_true), _arr(y_pred)
    denom = np.clip(np.abs(yt), 1e-8, None)
    return float(np.mean(np.abs((yt - yp) / denom)) * 100)


def smape(y_true, y_pred) -> float:
    yt, yp = _arr(y_true), _arr(y_pred)
    denom = np.clip(np.abs(yt) + np.abs(yp), 1e-8, None)
    return float(np.mean(2.0 * np.abs(yp - yt) / denom) * 100)


def directional_accuracy(y_true, y_pred, y_prev) -> float:
    yt, yp, y0 = _arr(y_true), _arr(y_pred), _arr(y_prev)
    actual_dir = np.sign(yt - y0)
    pred_dir = np.sign(yp - y0)
    mask = actual_dir != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(actual_dir[mask] == pred_dir[mask]))


def reconstruct_price(close, log_return) -> np.ndarray:
    """price[t+h] = close[t] * exp(h-day log return)."""
    return _arr(close) * np.exp(_arr(log_return))


def return_target(close, future_close) -> np.ndarray:
    """h-day log return from current close to future close."""
    c0 = _arr(close)
    c1 = _arr(future_close)
    return np.log(c1 / np.clip(c0, 1e-12, None))


def score_log_return_forecast(close, y_ret_true, y_ret_pred) -> dict[str, object]:
    """Score h-day log-return forecasts on reconstructed INR prices and in return space."""
    close_a = _arr(close)
    y_true = _arr(y_ret_true)
    y_pred = _arr(y_ret_pred)
    price_true = reconstruct_price(close_a, y_true)
    price_pred = reconstruct_price(close_a, y_pred)
    price = regression_metrics(price_true, price_pred, close_a)
    ret = regression_metrics(y_true, y_pred, np.zeros_like(y_true))
    return {
        "price": price,
        "return": ret,
        "price_true": price_true,
        "price_pred": price_pred,
    }


def regression_metrics(y_true, y_pred, y_prev=None) -> dict[str, float]:
    out = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "n": int(len(_arr(y_true))),
    }
    if y_prev is not None:
        out["directional_accuracy"] = directional_accuracy(y_true, y_pred, y_prev)
    return out


def classification_metrics(*args, **kwargs):  # pragma: no cover - alias for CLI naming
    return directional_accuracy(*args, **kwargs)
