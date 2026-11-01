from __future__ import annotations

import numpy as np
import pandas as pd

from gold.evaluation.metrics import reconstruct_price
from gold.models.baselines import MovingAverageForecaster, NaiveForecaster
from gold.models.ensemble import EnsembleForecaster


def test_naive_predicts_zero_log_return():
    y = pd.Series([0.01, -0.02, 0.03])
    model = NaiveForecaster().fit(y)
    np.testing.assert_allclose(model.predict(3), [0.0, 0.0, 0.0])


def test_naive_persistence_reconstructs_to_current_close():
    close = np.array([100.0, 105.0])
    pred_ret = NaiveForecaster().fit(pd.Series([0.1, 0.2])).predict(2)
    np.testing.assert_allclose(reconstruct_price(close, pred_ret), close)


def test_moving_average_window_of_returns():
    y = pd.Series([0.10, 0.20, 0.30, 0.40])
    model = MovingAverageForecaster(window=2).fit(y)
    np.testing.assert_allclose(model.predict(1), [0.35])


def test_ensemble_averages_member_returns():
    y = pd.Series([0.10, 0.20, 0.30])
    ens = EnsembleForecaster(members=[NaiveForecaster(), MovingAverageForecaster(window=2)])
    ens.fit(y)
    pred = ens.predict(1)
    assert pred.shape == (1,)
    # naive 0 and MA(0.20, 0.30)=0.25 → 0.125
    np.testing.assert_allclose(pred, [0.125])
