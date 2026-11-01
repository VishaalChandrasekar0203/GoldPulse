from __future__ import annotations

from collections.abc import Callable

from gold.models.arima import ARIMAForecaster, SARIMAForecaster
from gold.models.baselines import MovingAverageForecaster, NaiveForecaster
from gold.models.ensemble import EnsembleForecaster
from gold.models.lstm import LSTMForecaster
from gold.models.tft import TFTForecaster
from gold.models.tree import LightGBMForecaster, XGBoostForecaster

MODEL_FACTORIES: dict[str, Callable] = {
    "naive": NaiveForecaster,
    "moving_average": MovingAverageForecaster,
    "arima": ARIMAForecaster,
    "sarima": SARIMAForecaster,
    "xgboost": XGBoostForecaster,
    "lightgbm": LightGBMForecaster,
    "lstm": LSTMForecaster,
    "tft": TFTForecaster,
    "ensemble": EnsembleForecaster,
}


def available_models(include_optional: bool = False) -> list[str]:
    names = ["naive", "moving_average", "arima", "sarima", "xgboost", "lightgbm", "ensemble"]
    if include_optional:
        names.extend(["lstm", "tft"])
    return names


SKIP_UNTIL_BASELINES_BEAT_NAIVE = {"lstm", "tft"}
