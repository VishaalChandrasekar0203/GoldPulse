"""Champion promotion rules.

A non-naive model is promoted only when it beats naive persistence on the
latest chronological holdout *and* on at least `min_regimes` of the
historical expanding-window regimes. LSTM/Transformers are out of scope.
"""

from __future__ import annotations

from typing import Any


def beats_promotion_bar(
    holdout_mae: float,
    naive_holdout_mae: float,
    regimes_beat: int,
    regimes_evaluated: int,
    min_regimes: int = 3,
) -> bool:
    if not (holdout_mae < naive_holdout_mae):
        return False
    if regimes_evaluated < min_regimes:
        return False
    return int(regimes_beat) >= int(min_regimes)


def pipeline_version_of(params: Any, default: str = "v1-price-level") -> str:
    if isinstance(params, str):
        import json

        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return default
    if not isinstance(params, dict):
        return default
    return str(params.get("pipeline_version") or default)
