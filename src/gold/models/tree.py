from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from gold.features.columns import tree_feature_frame
from gold.models.base import BaseForecaster


def _clean_X(X: pd.DataFrame | None) -> pd.DataFrame:
    return tree_feature_frame(X)


class XGBoostForecaster(BaseForecaster):
    name = "xgboost"
    uses_features = True

    def __init__(self, **params):
        defaults = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=42,
            verbosity=0,
        )
        defaults.update(params)
        super().__init__(**defaults)
        self.model = XGBRegressor(**{k: v for k, v in defaults.items() if k != "name"})
        self.last: float | None = None
        self.feature_names: list[str] = []

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "XGBoostForecaster":
        Xc = _clean_X(X)
        self.feature_names = list(Xc.columns)
        self.last = 0.0
        self.model.fit(Xc, np.asarray(y, dtype=float))
        return self

    def predict(self, steps: int, X_future=None, last_row=None) -> np.ndarray:
        if X_future is not None and len(X_future):
            Xc = _clean_X(pd.DataFrame(X_future))
            Xc = Xc.reindex(columns=self.feature_names)
            preds = self.model.predict(Xc)
            if len(preds) >= steps:
                return np.asarray(preds[:steps], dtype=float)
            return np.pad(preds.astype(float), (0, steps - len(preds)), constant_values=preds[-1])
        return np.full(steps, 0.0)


class LightGBMForecaster(BaseForecaster):
    name = "lightgbm"
    uses_features = True

    def __init__(self, **params):
        defaults = dict(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
            n_jobs=1,
        )
        defaults.update(params)
        super().__init__(**defaults)
        self.model = LGBMRegressor(**defaults)
        self.last: float | None = None
        self.feature_names: list[str] = []

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "LightGBMForecaster":
        Xc = _clean_X(X)
        self.feature_names = list(Xc.columns)
        self.last = 0.0
        self.model.fit(Xc, np.asarray(y, dtype=float))
        return self

    def predict(self, steps: int, X_future=None, last_row=None) -> np.ndarray:
        if X_future is not None and len(X_future):
            Xc = _clean_X(pd.DataFrame(X_future))
            Xc = Xc.reindex(columns=self.feature_names)
            preds = self.model.predict(Xc)
            if len(preds) >= steps:
                return np.asarray(preds[:steps], dtype=float)
            return np.pad(np.asarray(preds, dtype=float), (0, steps - len(preds)), constant_values=preds[-1])
        return np.full(steps, self.last if self.last is not None else 0.0)
