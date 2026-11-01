from __future__ import annotations

import numpy as np
import pandas as pd

from gold.models.base import BaseForecaster


class NaiveForecaster(BaseForecaster):
    """Zero h-day log return: reconstructed price equals close[t] (persistence)."""

    name = "naive"
    uses_features = False

    def __init__(self, **params):
        super().__init__(**params)
        self.last = 0.0

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "NaiveForecaster":
        self.last = 0.0
        return self

    def predict(self, steps: int, X_future=None, last_row=None) -> np.ndarray:
        return np.zeros(int(steps), dtype=float)


class MovingAverageForecaster(BaseForecaster):
    """Mean of the last `window` observed h-day log returns."""

    name = "moving_average"
    uses_features = False

    def __init__(self, window: int = 21, **params):
        super().__init__(window=window, **params)
        self.window = window
        self.value: float = 0.0

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "MovingAverageForecaster":
        series = pd.Series(np.asarray(y, dtype=float)).dropna()
        if series.empty:
            self.value = 0.0
        else:
            w = min(self.window, len(series))
            self.value = float(series.iloc[-w:].mean())
        return self

    def predict(self, steps: int, X_future=None, last_row=None) -> np.ndarray:
        return np.full(int(steps), self.value, dtype=float)
