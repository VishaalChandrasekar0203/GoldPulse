from __future__ import annotations

import numpy as np
import pandas as pd

from gold.evaluation.metrics import reconstruct_price
from gold.models.arima import ARIMAForecaster, daily_log_returns


def test_daily_log_returns_use_unshifted_close():
    close = pd.Series([100.0, 110.0, 121.0, 133.1])
    X = pd.DataFrame({"close": close})
    y = pd.Series([9.0, 9.0, 9.0, 9.0])  # would be wrong if ARIMA fit the shifted target
    ret = daily_log_returns(y, X)
    np.testing.assert_allclose(ret.to_numpy(), np.log(close).diff().dropna().to_numpy(), rtol=1e-6)


def test_arima_forecast_is_h_day_log_return_not_price_path():
    rng = np.random.default_rng(0)
    daily = rng.normal(0.0004, 0.01, 180)
    close = 1000.0 * np.exp(np.cumsum(daily))
    X = pd.DataFrame({"close": close})
    y = pd.Series(np.zeros(len(close)))
    model = ARIMAForecaster(order=(1, 0, 0), horizon_steps=5).fit(y, X)
    pred = model.predict(8, X_future=X.iloc[-8:])
    assert pred.shape == (8,)
    assert np.allclose(pred, pred[0])
    assert np.max(np.abs(pred)) < 1.0
    reconstructed = reconstruct_price(close[-8:], pred)
    assert reconstructed.min() > 0
