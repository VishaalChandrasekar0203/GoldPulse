from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gold.models.baselines import NaiveForecaster
from gold.training.walkforward import chronological_split, walk_forward_evaluate


def test_chronological_split_is_contiguous():
    train, test = chronological_split(10, 3)
    assert list(train) == list(range(7))
    assert list(test) == [7, 8, 9]


def test_random_split_not_used():
    train, test = chronological_split(20, 5)
    assert train.max() < test.min()


def test_walk_forward_naive():
    n = 80
    close = np.linspace(100, 180, n)
    y = pd.Series(np.zeros(n))  # true h-day log return is 0 ⇒ persistence is exact
    meta = pd.DataFrame(
        {
            "close": close,
            "target_date": pd.date_range("2022-01-01", periods=n, freq="B"),
        }
    )
    metrics, detail = walk_forward_evaluate(
        NaiveForecaster, y, X=None, meta=meta, min_train=40, test_size=5, step=10, max_folds=4
    )
    assert metrics["n"] > 0
    assert not detail.empty
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)
