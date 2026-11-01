from __future__ import annotations

import numpy as np
import pandas as pd

from gold.models.base import BaseForecaster


class EnsembleForecaster(BaseForecaster):
    """Simple equal-weight blend of already-fit member predictions (set externally)."""

    name = "ensemble"
    uses_features = True

    def __init__(self, members: list[BaseForecaster] | None = None, **params):
        super().__init__(**params)
        self.members = members or []
        self.last: float | None = None

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "EnsembleForecaster":
        self.last = 0.0
        for m in self.members:
            m.fit(y, X)
        return self

    def predict(self, steps: int, X_future=None, last_row=None) -> np.ndarray:
        preds = []
        for m in self.members:
            try:
                preds.append(np.asarray(m.predict(steps, X_future=X_future, last_row=last_row), dtype=float))
            except Exception:  # noqa: BLE001
                continue
        if not preds:
            return np.full(steps, 0.0)
        stacked = np.vstack(preds)
        return np.nanmean(stacked, axis=0)
