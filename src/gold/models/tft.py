from __future__ import annotations

import numpy as np
import pandas as pd

from gold.models.base import BaseForecaster
from gold.logging import get_logger

log = get_logger("models.tft")

MIN_ROWS_FOR_TFT = 1500


class TFTForecaster(BaseForecaster):
    """Temporal Fusion Transformer is only trained when justified and the extra is installed.

    Justification rule: multiple related series + long daily history. For the MVP single-target
    gold series, gradient boosting typically dominates a small TFT. We therefore skip unless
    pytorch-forecasting is present AND history exceeds MIN_ROWS_FOR_TFT.
    """

    name = "tft"
    uses_features = True

    def __init__(self, **params):
        super().__init__(**params)
        self.last: float | None = None
        self.available = False
        self.skip_reason: str | None = None

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "TFTForecaster":
        if X is not None and "close" in X.columns:
            self.last = float(pd.to_numeric(X["close"], errors="coerce").iloc[-1])
        else:
            self.last = float(pd.Series(y).iloc[-1])
        try:
            import pytorch_forecasting  # noqa: F401
        except ImportError:
            self.skip_reason = (
                "TFT skipped: pytorch-forecasting is not installed. "
                "Not justified for MVP single-target gold unless extras are added."
            )
            log.info(self.skip_reason)
            return self
        if len(y) < MIN_ROWS_FOR_TFT:
            self.skip_reason = (
                f"TFT skipped: {len(y)} rows < {MIN_ROWS_FOR_TFT}. "
                "A small TFT is not justified versus gradient boosting on this sample."
            )
            log.info(self.skip_reason)
            return self
        # A full TFT training loop is deferred until the dataset/extras justify the cost.
        self.skip_reason = "TFT extras present but training is gated pending multi-series panel support."
        log.info(self.skip_reason)
        return self

    def predict(self, steps: int, X_future=None, last_row=None) -> np.ndarray:
        return np.full(steps, self.last if self.last is not None else np.nan)
