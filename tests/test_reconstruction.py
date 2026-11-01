from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.evaluation.metrics import reconstruct_price, return_target, score_log_return_forecast
from gold.training.promotion import beats_promotion_bar


def test_reconstruct_price_matches_future_close():
    close = np.array([100.0, 200.0])
    future = np.array([110.0, 180.0])
    ret = return_target(close, future)
    np.testing.assert_allclose(reconstruct_price(close, ret), future)


def test_zero_return_is_persistence():
    close = np.array([134000.0])
    np.testing.assert_allclose(reconstruct_price(close, [0.0]), close)


def test_score_log_return_reports_price_and_return_mae():
    close = np.array([100.0, 100.0])
    y_true = np.log(np.array([110.0, 90.0]) / close)
    y_pred = np.zeros(2)
    scored = score_log_return_forecast(close, y_true, y_pred)
    assert scored["price"]["mae"] == pytest.approx(10.0, rel=1e-9)
    assert scored["return"]["n"] == 2


def test_promotion_requires_holdout_and_three_regimes():
    assert not beats_promotion_bar(10, 20, regimes_beat=2, regimes_evaluated=5)
    assert not beats_promotion_bar(30, 20, regimes_beat=5, regimes_evaluated=5)
    assert not beats_promotion_bar(10, 20, regimes_beat=3, regimes_evaluated=2)
    assert beats_promotion_bar(10, 20, regimes_beat=3, regimes_evaluated=5)
