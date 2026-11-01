from __future__ import annotations

import numpy as np

from gold.evaluation.metrics import mape, reconstruct_price, regression_metrics, smape


def test_perfect_forecast_metrics():
    y = np.array([100.0, 110.0, 120.0])
    m = regression_metrics(y, y, y_prev=np.array([90.0, 100.0, 110.0]))
    assert m["mae"] == 0
    assert m["rmse"] == 0
    assert m["mape"] == 0
    assert m["smape"] == 0
    assert m["directional_accuracy"] == 1.0


def test_smape_symmetric():
    assert abs(smape([100], [80]) - smape([80], [100])) < 1e-9
    assert mape([100], [80]) == 20.0


def test_reconstruct_price_exp():
    np.testing.assert_allclose(reconstruct_price([100.0], [np.log(1.1)]), [110.0])
