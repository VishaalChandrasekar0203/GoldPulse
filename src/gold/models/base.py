from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    name: str
    uses_features: bool = True

    def __init__(self, **params: Any):
        self.params = params

    @abstractmethod
    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, steps: int, X_future: pd.DataFrame | None = None, last_row: pd.Series | None = None) -> np.ndarray:
        ...

    def predict_one(self, X_row: pd.DataFrame) -> float:
        preds = self.predict(1, X_future=X_row)
        return float(preds[0])
