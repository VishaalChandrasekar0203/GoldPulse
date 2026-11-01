from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gold.config import AppConfig
from gold.data.calendar import IndiaMarketCalendar
from gold.database.store import GoldStore, new_run_id
from gold.exceptions import InsufficientDataError
from gold.features.alignment import asof_join
from gold.features.columns import MACRO_SERIES
from gold.features.technical import macd, rsi
from gold.logging import get_logger

log = get_logger("features")


class FeatureBuilder:
    def __init__(self, cfg: AppConfig, store: GoldStore):
        self.cfg = cfg
        self.store = store
        self.calendar = IndiaMarketCalendar()
        spec = cfg.get("features", {}) or {}
        self.lags = list(spec.get("lags", [1, 2, 3, 5, 10, 21]))
        self.windows = list(spec.get("rolling_windows", [5, 21, 63]))
        self.rsi_period = int(spec.get("rsi_period", 14))
        self.macd_fast = int(spec.get("macd_fast", 12))
        self.macd_slow = int(spec.get("macd_slow", 26))
        self.macd_signal = int(spec.get("macd_signal", 9))
        self.fest_window = int(spec.get("festival_window_days", 7))

    def _target_frame(self) -> pd.DataFrame:
        target = self.store.load_series(self.cfg.primary_series)
        if target.empty:
            raise InsufficientDataError(
                f"Primary series {self.cfg.primary_series} is empty. Run `gold data update` first."
            )
        target = target.sort_values("obs_date")
        target["obs_date"] = pd.to_datetime(target["obs_date"])
        target = target.set_index("obs_date")
        return target

    def _festival_table(self) -> pd.DataFrame:
        path = self.cfg.project_root / "config" / "festivals.csv"
        return pd.read_csv(path, parse_dates=["event_date"])

    def build(self) -> pd.DataFrame:
        target = self._target_frame()
        close = target["value"].astype(float).rename("close")
        idx = close.index
        feats = pd.DataFrame(index=idx)
        feats["close"] = close
        feats["log_close"] = np.log(close)
        feats["log_return"] = np.log(close).diff()

        for lag in self.lags:
            feats[f"ret_lag_{lag}"] = feats["log_return"].shift(lag)

        for win in self.windows:
            feats[f"roll_vol_{win}"] = feats["log_return"].rolling(win).std() * np.sqrt(252)

        # Technicals on close[t], which is known when forecasting t → t+h.
        feats[f"rsi_{self.rsi_period}"] = rsi(close, self.rsi_period)
        feats = feats.join(macd(close, self.macd_fast, self.macd_slow, self.macd_signal))

        for sid in MACRO_SERIES:
            series = self.store.load_series(sid)
            feats[sid] = asof_join(idx, series, "value", sid)
            feats[f"{sid}_chg"] = feats[sid].diff()

        secondary = self.store.load_series(self.cfg.secondary_series)
        if not secondary.empty:
            feats["gold_etf_inr_10g"] = asof_join(idx, secondary, "value", "gold_etf_inr_10g")
            feats["etf_parity_gap_pct"] = (feats["gold_etf_inr_10g"] - feats["close"]) / feats["close"]
            feats["etf_log_return"] = np.log(feats["gold_etf_inr_10g"]).diff()

        dates = pd.Series(idx.date, index=idx)
        feats["dow"] = idx.day_of_week
        feats["month"] = idx.month
        feats["woy"] = idx.isocalendar().week.astype(int)
        feats["is_month_end"] = idx.is_month_end.astype(int)
        feats["is_trading_day"] = [int(self.calendar.is_trading_day(d)) for d in dates]

        festivals = self._festival_table()
        event_dates = pd.to_datetime(festivals["event_date"])
        days_to = []
        in_window = []
        for d in idx:
            future = event_dates[event_dates >= d]
            if future.empty:
                days_to.append(np.nan)
                in_window.append(0)
            else:
                delta = (future.min() - d).days
                days_to.append(delta)
                in_window.append(int(delta <= self.fest_window))
        feats["days_to_next_gold_festival"] = days_to
        feats["festival_window"] = in_window
        for event_name in ("diwali", "dhanteras", "akshaya_tritiya"):
            event_days = set(
                pd.to_datetime(festivals.loc[festivals["event"] == event_name, "event_date"]).dt.date
            )
            feats[f"is_{event_name}"] = [int(d.date() in event_days) for d in idx]

        feats["obs_date"] = idx
        feats["as_of"] = idx
        feats = feats.reset_index(drop=True)

        min_hist = max(self.lags + self.windows + [self.macd_slow + self.macd_signal])
        feats = feats.iloc[min_hist:].copy()

        path = self.store.save_features(feats)
        run_id = new_run_id("feat")
        self.store.con.execute(
            "INSERT OR REPLACE INTO feature_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                datetime.now(tz=timezone.utc).replace(tzinfo=None),
                datetime.now(tz=timezone.utc).replace(tzinfo=None),
                len(feats),
                len(feats.columns),
                feats["obs_date"].max().date() if not feats.empty else None,
                str(path),
            ],
        )
        log.info("features built: %s rows x %s cols -> %s", len(feats), feats.shape[1], path)
        return feats

    def supervised(self, horizon_trading_days: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        feats = self.store.load_features()
        if feats.empty:
            feats = self.build()
        feats = feats.sort_values("obs_date").reset_index(drop=True)
        close = pd.to_numeric(feats["close"], errors="coerce")
        future_close = close.shift(-horizon_trading_days)
        y = pd.Series(np.log(future_close.to_numpy() / close.to_numpy()), index=feats.index)
        y_date = feats["obs_date"].shift(-horizon_trading_days)
        valid = y.notna() & np.isfinite(y) & future_close.notna()
        X = feats.loc[valid].copy()
        meta_cols = [c for c in ["obs_date", "as_of"] if c in X.columns]
        feature_cols = [c for c in X.columns if c not in meta_cols]
        X_model = X[feature_cols].apply(pd.to_numeric, errors="coerce")
        return (
            X_model.reset_index(drop=True),
            y.loc[valid].reset_index(drop=True),
            pd.DataFrame(
                {
                    "obs_date": pd.to_datetime(feats.loc[valid, "obs_date"]).reset_index(drop=True),
                    "target_date": pd.to_datetime(y_date.loc[valid]).reset_index(drop=True),
                    "close": close.loc[valid].reset_index(drop=True),
                    "target_price": future_close.loc[valid].reset_index(drop=True),
                }
            ),
        )

    def live_row(self, columns: list[str] | None = None) -> pd.DataFrame:
        feats = self.store.load_features()
        if feats.empty:
            feats = self.build()
        last = feats.sort_values("obs_date").iloc[[-1]].copy()
        if columns is not None:
            missing = [c for c in columns if c not in last.columns]
            for c in missing:
                last[c] = np.nan
            last = last[columns]
        return last.apply(pd.to_numeric, errors="coerce")


if __name__ == "__main__":
    from gold.config import load_config
    from gold.database.store import GoldStore
    from gold.logging import setup_logging

    cfg = load_config()
    setup_logging(cfg.log_dir)
    print(FeatureBuilder(cfg, GoldStore(cfg)).build().tail())
