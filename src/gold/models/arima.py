from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from gold.models.base import BaseForecaster


def daily_log_returns(y: pd.Series, X: pd.DataFrame | None) -> pd.Series:
    """Unshifted daily log returns of close[t], never of the already-shifted target."""
    if X is not None and "close" in getattr(X, "columns", []):
        close = pd.to_numeric(X["close"], errors="coerce").dropna()
        if len(close) >= 3:
            ret = np.log(close).diff().dropna()
            if len(ret) >= 2:
                return pd.Series(np.asarray(ret, dtype=float))
    return pd.Series(np.asarray(y, dtype=float)).dropna()


class ARIMAForecaster(BaseForecaster):
    """ARIMA on daily log returns of the original close; ŷ is the sum of h daily steps."""

    name = "arima"
    uses_features = False
    CANDIDATES = [(1, 0, 1), (1, 0, 0), (0, 0, 1), (2, 0, 1)]

    def __init__(self, order: tuple[int, int, int] | None = None, horizon_steps: int = 1, **params):
        super().__init__(order=order, horizon_steps=horizon_steps, **params)
        self.order = order
        self.horizon_steps = max(1, int(horizon_steps))
        self.result = None
        self.last: float = 0.0
        self.h_day_return: float = 0.0

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "ARIMAForecaster":
        series = daily_log_returns(y, X)
        self.last = 0.0
        self.h_day_return = 0.0
        orders = [self.order] if self.order else list(self.CANDIDATES)
        best_aic = np.inf
        best = None
        best_order = orders[0]
        for order in orders:
            if order is None:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        series,
                        order=order,
                        seasonal_order=(0, 0, 0, 0),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = model.fit(disp=False, maxiter=80)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best = res
                    best_order = order
            except Exception:  # noqa: BLE001
                continue
        self.result = best
        self.params["order"] = best_order
        self.params["aic"] = None if best is None else float(best_aic)
        self.params["horizon_steps"] = self.horizon_steps
        self.h_day_return = self._h_step_return()
        self.last = self.h_day_return
        return self

    def _h_step_return(self) -> float:
        if self.result is None:
            return 0.0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = np.asarray(self.result.forecast(steps=self.horizon_steps), dtype=float)
            return float(np.nansum(fc))
        except Exception:  # noqa: BLE001
            return 0.0

    def predict(self, steps: int, X_future=None, last_row=None) -> np.ndarray:
        return np.full(int(steps), self.h_day_return, dtype=float)


class SARIMAForecaster(ARIMAForecaster):
    name = "sarima"

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "SARIMAForecaster":
        series = daily_log_returns(y, X)
        self.last = 0.0
        self.h_day_return = 0.0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    series,
                    order=(1, 0, 1),
                    seasonal_order=(1, 0, 1, 5),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self.result = model.fit(disp=False, maxiter=80)
                self.params["aic"] = float(self.result.aic)
                self.params["order"] = (1, 0, 1)
                self.params["seasonal_order"] = (1, 0, 1, 5)
        except Exception:  # noqa: BLE001
            super().fit(y, X)
            return self
        self.params["horizon_steps"] = self.horizon_steps
        self.h_day_return = self._h_step_return()
        self.last = self.h_day_return
        return self
