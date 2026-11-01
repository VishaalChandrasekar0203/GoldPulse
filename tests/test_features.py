from __future__ import annotations

import numpy as np
import pandas as pd

from gold.features.technical import macd, rsi


def test_rsi_bounds_and_warmup():
    close = pd.Series(np.linspace(100, 200, 40))
    values = rsi(close, 14)
    assert values.iloc[:13].isna().all()
    assert values.dropna().between(0, 100).all()


def test_realized_vol_uses_known_returns():
    log_return = pd.Series([0.01, 0.02, -0.01, 0.00, 0.03])
    vol = log_return.rolling(2).std() * np.sqrt(252)
    assert pd.isna(vol.iloc[0])
    assert vol.iloc[1] > 0


def test_macd_columns():
    close = pd.Series(np.arange(40, dtype=float) + 100)
    out = macd(close)
    assert {"macd", "macd_signal", "macd_hist"} <= set(out.columns)
    np.testing.assert_allclose(out["macd_hist"], out["macd"] - out["macd_signal"])
