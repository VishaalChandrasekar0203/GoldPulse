"""Daily market intelligence. Contemporaneous stats are not treated as forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gold.config import AppConfig
from gold.database.store import GoldStore
from gold.features.alignment import asof_join
from gold.forecasting.volatility import ewma_variance
from gold.logging import get_logger

log = get_logger("intelligence")

DASHBOARD_SERIES = [
    ("gold_usd_oz", "XAU/USD"),
    ("usd_inr", "USD/INR"),
    ("dxy", "DXY"),
    ("us_10y_yield", "US 10y nominal"),
    ("us_real_yield_10y", "US 10y real"),
    ("wti_usd", "WTI crude"),
    ("gold_etf_nse", "GOLDBEES (NSE)"),
    ("gold_etf_inr_10g", "GOLDBEES INR/10g"),
    ("gold_inr_10g", "Parity INR/10g"),
]


def _returns(values: pd.Series) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    return np.log(v.clip(lower=1e-12)).diff()


def _last_return(values: pd.Series, n: int) -> float | None:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) <= n:
        return None
    return float(np.log(v.iloc[-1] / v.iloc[-1 - n]))


def market_dashboard(cfg: AppConfig, store: GoldStore) -> dict:
    cards = []
    levels = {}
    gold_usd = store.load_series("gold_usd_oz").sort_values("obs_date")
    as_of_gold = pd.to_datetime(gold_usd["obs_date"].iloc[-1]) if not gold_usd.empty else None
    for sid, label in DASHBOARD_SERIES:
        df = store.load_series(sid).sort_values("obs_date")
        if df.empty:
            cards.append(
                {
                    "series_id": sid,
                    "label": label,
                    "status": "empty",
                    "note": "No observations. Not fabricated."
                    if sid in {"mcx_gold_futures", "ibja_gold"}
                    else "No observations stored.",
                }
            )
            continue
        df = df.copy()
        df["obs_date"] = pd.to_datetime(df["obs_date"])
        if as_of_gold is not None:
            df = df[df["obs_date"] <= as_of_gold]
        if df.empty:
            cards.append({"series_id": sid, "label": label, "status": "empty", "note": "No print on or before last gold session."})
            continue
        val = df["value"].astype(float)
        last = float(val.iloc[-1])
        as_of = pd.to_datetime(df["obs_date"].iloc[-1])
        rets = _returns(val)
        sigma = float("nan")
        if rets.notna().sum() > 30:
            sigma = float(np.sqrt(ewma_variance(rets.fillna(0.0).to_numpy())[-1]))
        levels[sid] = df
        cards.append(
            {
                "series_id": sid,
                "label": label,
                "as_of": str(as_of.date()),
                "last": last,
                "ret_1d": _last_return(val, 1),
                "ret_5d": _last_return(val, 5),
                "ret_21d": _last_return(val, 21),
                "ret_63d": _last_return(val, 63),
                "ewma_vol": sigma,
                "status": "ok",
            }
        )

    gold = store.load_series("gold_inr_10g").sort_values("obs_date")
    contemp = []
    predictive = []
    if not gold.empty:
        idx = pd.DatetimeIndex(pd.to_datetime(gold["obs_date"]))
        gold_ret = pd.Series(np.log(gold.set_index(idx)["value"]).diff(), index=idx)
        for sid, label in DASHBOARD_SERIES:
            if sid == "gold_inr_10g":
                continue
            series = store.load_series(sid)
            if series.empty:
                continue
            level = asof_join(idx, series, "value", sid)
            x = np.log(level.clip(lower=1e-12)).diff()
            data = pd.DataFrame({"y": gold_ret, "x": x}).dropna()
            if len(data) < 80:
                continue
            contemp.append(
                {
                    "pair": f"{label} vs INR gold (same day)",
                    "kind": "contemporaneous",
                    "pearson": float(data["y"].corr(data["x"])),
                    "n": int(len(data)),
                    "note": "Descriptive same-day comovement. Not a forecast.",
                }
            )
            pred = pd.DataFrame({"y": gold_ret.shift(-1), "x": x}).dropna()
            if len(pred) >= 80:
                predictive.append(
                    {
                        "pair": f"{label}[t] vs next-day INR gold",
                        "kind": "predictive",
                        "pearson": float(pred["y"].corr(pred["x"])),
                        "n": int(len(pred)),
                        "note": "Correlation of today's move with tomorrow's gold. Stage-2 found these too weak to beat naive.",
                    }
                )

    regime = {}
    if not gold.empty and len(gold) > 80:
        val = gold["value"].astype(float)
        r63 = float(np.log(val.iloc[-1] / val.iloc[-64])) if len(val) > 64 else None
        hist = np.log(val / val.shift(63)).dropna()
        pct = float((hist < r63).mean()) if r63 is not None and len(hist) else None
        era = "bull" if r63 is not None and r63 > 0 else "bear_or_flat"
        if pct is not None and pct >= 0.9:
            era = "historically extreme bull (63d)"
        elif pct is not None and pct <= 0.1:
            era = "historically extreme drawdown (63d)"
        regime = {
            "ret_63d": r63,
            "percentile_vs_history": pct,
            "label": era,
            "note": "Trailing 63-session log return vs the full sample. A regime label, not a forecast.",
        }

    restricted = [
        {
            "series_id": "mcx_gold_futures",
            "label": "MCX Gold Futures",
            "status": "unavailable",
            "note": "Licensed. Not fabricated. Parity is not MCX.",
        },
        {
            "series_id": "ibja_gold",
            "label": "IBJA gold",
            "status": "unavailable",
            "note": "Licensed. Not fabricated. GOLDBEES is not IBJA.",
        },
    ]
    log.info("market dashboard cards=%s", len(cards))
    return {
        "cards": cards,
        "contemporaneous": contemp,
        "predictive": predictive,
        "regime": regime,
        "restricted": restricted,
    }
