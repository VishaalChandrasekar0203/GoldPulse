from __future__ import annotations

import numpy as np
import pandas as pd

from gold.models.base import BaseForecaster
from gold.logging import get_logger

log = get_logger("models.lstm")


class LSTMForecaster(BaseForecaster):
    """Optional univariate LSTM. Requires torch extra (`pip install gold-forecasting[deep]`)."""

    name = "lstm"
    uses_features = False

    def __init__(self, lookback: int = 30, hidden: int = 32, epochs: int = 25, lr: float = 1e-3, **params):
        super().__init__(lookback=lookback, hidden=hidden, epochs=epochs, lr=lr, **params)
        self.lookback = lookback
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.mean = 0.0
        self.std = 1.0
        self.last: float | None = None
        self._model = None
        self.available = False
        self.skip_reason: str | None = None

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "LSTMForecaster":
        series = np.asarray(y, dtype=float)
        if X is not None and "close" in X.columns:
            self.last = float(pd.to_numeric(X["close"], errors="coerce").iloc[-1])
        else:
            self.last = float(series[-1])
        try:
            import torch
            from torch import nn
        except ImportError:
            self.skip_reason = "PyTorch is not installed. LSTM skipped (optional extra: gold-forecasting[deep])."
            log.warning(self.skip_reason)
            return self

        if len(series) < self.lookback + 10:
            self.skip_reason = "Not enough history for LSTM lookback."
            return self

        self.mean = float(series.mean())
        self.std = float(series.std() + 1e-8)
        z = (series - self.mean) / self.std
        xs, ys = [], []
        for i in range(self.lookback, len(z)):
            xs.append(z[i - self.lookback : i])
            ys.append(z[i])
        xt = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
        yt = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)

        class _LSTM(nn.Module):
            def __init__(self, hidden: int):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden, batch_first=True)
                self.fc = nn.Linear(hidden, 1)

            def forward(self, inp):
                out, _ = self.lstm(inp)
                return self.fc(out[:, -1, :])

        torch.manual_seed(42)
        model = _LSTM(self.hidden)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = model(xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()
        self._model = model
        self._history = z[-self.lookback :].copy()
        self.available = True
        return self

    def predict(self, steps: int, X_future=None, last_row=None) -> np.ndarray:
        if not self.available or self._model is None:
            return np.full(steps, self.last if self.last is not None else np.nan)
        import torch

        hist = list(self._history)
        self._model.eval()
        out = []
        with torch.no_grad():
            for _ in range(steps):
                inp = torch.tensor(hist[-self.lookback :], dtype=torch.float32).view(1, self.lookback, 1)
                z = float(self._model(inp).item())
                hist.append(z)
                out.append(z * self.std + self.mean)
        return np.asarray(out, dtype=float)
