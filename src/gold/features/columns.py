"""Feature-column policy for the log-return pipeline.

Price *levels* are the reconstruction anchor only. Tree models must not see
lagged closes or rolling min/mean/max of price. Return lags, realized vol,
oscillators, calendar, and *changes* in macro series are eligible.
"""

from __future__ import annotations

import pandas as pd

MACRO_SERIES = [
    "usd_inr",
    "dxy",
    "us_real_yield_10y",
    "us_10y_yield",
    "wti_usd",
    "india_cpi",
    "india_call_rate",
    "gold_import_duty",
    "rbi_repo_rate",
]

# Known at time t and required to reconstruct price = close * exp(ŷ_return).
# Also used by ARIMA to recover the unshifted daily log-return series.
ANCHOR_COLUMNS = ("close",)

LEVEL_COLUMNS = {
    "close",
    "log_close",
    "gold_etf_inr_10g",
    *MACRO_SERIES,
}

LEVEL_PREFIXES = (
    "close_lag_",
    "roll_mean_",
    "roll_std_",
    "roll_min_",
    "roll_max_",
)

META_COLUMNS = ("obs_date", "as_of", "target_date", "target_price")


def is_level_column(name: str) -> bool:
    if name in LEVEL_COLUMNS:
        return True
    return name.startswith(LEVEL_PREFIXES)


def tree_feature_frame(X: pd.DataFrame | None) -> pd.DataFrame:
    """Numeric model matrix with price levels and raw macro levels removed."""
    if X is None or X.empty:
        raise ValueError("Tree models require a feature matrix")
    keep = [c for c in X.columns if not is_level_column(c) and c not in META_COLUMNS]
    if not keep:
        raise ValueError("No eligible return/vol/macro-change features remain")
    frame = X.loc[:, keep].apply(pd.to_numeric, errors="coerce")
    nonempty = [c for c in frame.columns if frame[c].notna().any()]
    return frame.loc[:, nonempty] if nonempty else frame
