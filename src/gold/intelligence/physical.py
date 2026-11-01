"""Retail gold scorecard: parity, tax-on-parity physical, premium, rule-based stance.

Not a forecast model. Buy / Hold / Wait / Review are transparent checks on
stored series plus optional user quotes. MCX/IBJA are never filled.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gold.config import AppConfig
from gold.database.store import GoldStore
from gold.forecasting.volatility import ewma_variance

DEFAULT_GST_PCT = 3.0
DEFAULT_MAKING_PCT = 0.0
# COMEX / international parity is treated as 999 fine. Indian jewellery is usually BIS 22K / 916.
FINE_PER_MILLE = 999
DEFAULT_HALLMARK_FINENESS = 916
HALLMARK_CHOICES = {
    999: "24K / 999 — fine bar or coin",
    916: "22K / 916 — BIS hallmarked jewellery",
    750: "18K / 750 — BIS hallmarked jewellery",
    585: "14K / 585 — BIS hallmarked jewellery",
}
SESSION_WINDOWS = (7, 30, 90)
STALE_CALENDAR_DAYS = 4

# Shown in the UI so the rules are inspectable, not a black box.
STANCE_RULES = [
    {
        "id": "off_90d_high",
        "side": "buy",
        "label": "Off the 90-session high",
        "rule": "Drawdown from the 90-session high is 5% or deeper.",
    },
    {
        "id": "below_longer_mas",
        "side": "buy",
        "label": "Below 30- and 90-session averages",
        "rule": "Parity is below both the 30-session and 90-session means.",
    },
    {
        "id": "tight_premium",
        "side": "buy",
        "label": "Dealer premium not above tax wedge",
        "rule": "Dealer quote is set and is within 0.5% of hallmark-adjusted physical (purity × duty × GST × making).",
    },
    {
        "id": "calm_vol",
        "side": "buy",
        "label": "Volatility not in the top quintile",
        "rule": "Current EWMA daily vol is below the 80th percentile of its own history.",
    },
    {
        "id": "near_90d_high",
        "side": "wait",
        "label": "Near the 90-session high",
        "rule": "Drawdown from the 90-session high is shallower than 1.5%.",
    },
    {
        "id": "above_all_mas",
        "side": "wait",
        "label": "Above 7-, 30-, and 90-session averages",
        "rule": "Parity is above all three trailing session means.",
    },
    {
        "id": "fat_premium",
        "side": "wait",
        "label": "Fat dealer premium",
        "rule": "Dealer quote is set and exceeds hallmark-adjusted physical (purity × duty × GST × making) by more than 1%.",
    },
    {
        "id": "fx_driven_spike",
        "side": "wait",
        "label": "INR gold bid is mostly USD/INR",
        "rule": "21-session INR gold return is positive and the USD/INR contribution is larger than the COMEX contribution.",
    },
    {
        "id": "data_gap",
        "side": "review",
        "label": "Insufficient or stale prints",
        "rule": "Fewer than 90 gold sessions, missing USD/INR, or last gold print older than 4 calendar days.",
    },
    {
        "id": "extreme_vol",
        "side": "review",
        "label": "Extreme volatility",
        "rule": "EWMA daily vol is at or above the 95th percentile of its history.",
    },
    {
        "id": "conflict",
        "side": "review",
        "label": "Buy and Wait both fire",
        "rule": "At least two Buy checks and two Wait checks are true at the same time.",
    },
]


def tracker_path(cfg: AppConfig) -> Path:
    return Path(cfg.data_dir) / "user_tracker.json"


def load_user_tracker(cfg: AppConfig) -> dict[str, Any]:
    path = tracker_path(cfg)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def save_user_tracker(cfg: AppConfig, payload: dict[str, Any]) -> Path:
    path = tracker_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = load_user_tracker(cfg)
    current.update({k: v for k, v in payload.items() if v is not None})
    current["updated_at"] = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
    path.write_text(json.dumps(current, indent=2, sort_keys=True))
    return path


def _sorted(store: GoldStore, series_id: str) -> pd.DataFrame:
    df = store.load_series(series_id)
    if df.empty:
        return df
    out = df.copy()
    out["obs_date"] = pd.to_datetime(out["obs_date"])
    return out.sort_values("obs_date")


def _ffill_on(index: pd.DatetimeIndex, series: pd.DataFrame) -> pd.Series:
    if series.empty:
        return pd.Series(index=index, dtype=float)
    s = series.copy()
    s["obs_date"] = pd.to_datetime(s["obs_date"])
    s = s.dropna(subset=["value"]).drop_duplicates("obs_date", keep="last")
    return s.set_index("obs_date")["value"].astype(float).reindex(index, method="ffill")


def _log_ret(values: pd.Series, n: int) -> float | None:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) <= n or float(v.iloc[-1 - n]) <= 0:
        return None
    return float(np.log(float(v.iloc[-1]) / float(v.iloc[-1 - n])))


def _mean(values: pd.Series, n: int) -> float | None:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < n:
        return None
    return float(v.iloc[-n:].mean())


def _high(values: pd.Series, n: int) -> float | None:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < n:
        return None
    return float(v.iloc[-n:].max())


def _years(start, end) -> float | None:
    if start is None or end is None:
        return None
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return round((e - s).days / 365.25, 2)


def tax_wedge_pct(duty_pct: float | None, gst_pct: float, making_pct: float) -> float:
    duty = float(duty_pct or 0.0)
    gst = float(gst_pct or 0.0)
    making = float(making_pct or 0.0)
    factor = (1.0 + duty / 100.0) * (1.0 + gst / 100.0) * (1.0 + making / 100.0)
    return 100.0 * (factor - 1.0)


def hallmark_factor(fineness: int, fine_ref: int = FINE_PER_MILLE) -> float:
    return float(fineness) / float(fine_ref)


def hallmarked_metal(parity: float, fineness: int = DEFAULT_HALLMARK_FINENESS) -> float:
    """Fine-gold parity scaled to a BIS hallmark (e.g. 22K 916 / 999)."""
    return float(parity) * hallmark_factor(int(fineness))


def estimated_physical(
    parity: float,
    duty_pct: float | None,
    gst_pct: float,
    making_pct: float,
    hallmark_fineness: int = FINE_PER_MILLE,
) -> float:
    metal = hallmarked_metal(parity, int(hallmark_fineness))
    wedge = tax_wedge_pct(duty_pct, gst_pct, making_pct)
    return metal * (1.0 + wedge / 100.0)


def physical_gold_premium(dealer_price: float | None, parity: float | None) -> float | None:
    """Actual dealer price − international parity. None if either side is missing."""
    if dealer_price is None or parity is None:
        return None
    if float(dealer_price) <= 0 or float(parity) <= 0:
        return None
    return float(dealer_price) - float(parity)


def physical_base(store: GoldStore, cfg: AppConfig) -> dict[str, Any]:
    """Series-derived scorecard. Does not use dealer quotes or average cost."""
    gold = _sorted(store, cfg.primary_series)
    xau = _sorted(store, "gold_usd_oz")
    fx = _sorted(store, "usd_inr")
    duty_s = _sorted(store, "gold_import_duty")
    etf = _sorted(store, cfg.secondary_series)
    today = datetime.now(tz=timezone.utc).date()
    stale_days = int(
        (cfg.get("ops", {}) or {}).get("freshness_calendar_days")
        or (cfg.get("monitor", {}) or {}).get("stale_calendar_days", STALE_CALENDAR_DAYS)
    )

    empty = {
        "data_ok": False,
        "stale": True,
        "n_obs": 0,
        "history_start": None,
        "history_end": None,
        "history_years": None,
        "configured_start": cfg.start_date,
        "parity_inr_10g": None,
        "history_note": (
            f"Configured ingest start is {cfg.start_date}. No gold_inr_10g prints stored yet. "
            "This is not a 1-year dataset by design — run `gold data update`."
        ),
        "incomplete": ["gold_inr_10g"],
    }
    if gold.empty:
        return empty

    idx = pd.DatetimeIndex(pd.to_datetime(gold["obs_date"]))
    parity_s = gold.set_index(idx)["value"].astype(float)
    last = float(parity_s.iloc[-1])
    as_of = pd.Timestamp(idx[-1]).date()
    start = pd.Timestamp(idx[0]).date()
    n_obs = int(len(parity_s))
    years = _years(start, as_of)
    lag = (today - as_of).days
    stale = lag > stale_days
    xau_s = _ffill_on(idx, xau)
    fx_s = _ffill_on(idx, fx)
    duty_aligned = _ffill_on(idx, duty_s)
    etf_s = _ffill_on(idx, etf)
    duty_pct = float(duty_aligned.iloc[-1]) if duty_aligned.notna().any() and pd.notna(duty_aligned.iloc[-1]) else None
    xau_last = float(xau_s.iloc[-1]) if pd.notna(xau_s.iloc[-1]) else None
    fx_last = float(fx_s.iloc[-1]) if pd.notna(fx_s.iloc[-1]) else None
    etf_last = float(etf_s.iloc[-1]) if etf_s.notna().any() and pd.notna(etf_s.iloc[-1]) else None
    etf_prem = None if etf_last is None else float(etf_last - last)

    mas: dict[str, float | None] = {}
    vs_mas: dict[str, float | None] = {}
    highs: dict[str, float | None] = {}
    dds: dict[str, float | None] = {}
    incomplete: list[str] = []
    for w in SESSION_WINDOWS:
        key = str(w)
        m = _mean(parity_s, w)
        h = _high(parity_s, w)
        mas[key] = m
        highs[key] = h
        vs_mas[key] = None if m in (None, 0) else last / m - 1.0
        dds[key] = None if h in (None, 0) else last / h - 1.0
        if m is None:
            incomplete.append(f"ma_{w}")
        if h is None:
            incomplete.append(f"high_{w}")

    rets = np.log(parity_s.clip(lower=1e-12)).diff()
    ewma = float("nan")
    ewma_pct = None
    if rets.notna().sum() > 40:
        var = ewma_variance(rets.fillna(0.0).to_numpy())
        sigma = np.sqrt(var)
        ewma = float(sigma[-1])
        hist = sigma[np.isfinite(sigma)]
        if len(hist) >= 40:
            ewma_pct = float((hist <= ewma).mean())

    r1_p = _log_ret(parity_s, 1)
    r1_x = _log_ret(xau_s, 1)
    r1_f = _log_ret(fx_s, 1)
    r21_p = _log_ret(parity_s, 21)
    r21_x = _log_ret(xau_s, 21)
    r21_f = _log_ret(fx_s, 21)

    hist_21 = np.log(parity_s / parity_s.shift(21)).dropna()
    p05_21 = float(np.quantile(hist_21, 0.05)) if len(hist_21) >= 40 else None

    downside: dict[str, dict[str, float | None]] = {}
    if math.isfinite(ewma) and ewma > 0:
        for h, label in ((1, "1d"), (21, "30d"), (63, "90d")):
            downside[label] = {
                "sigma_1": float(last * math.exp(-1.0 * ewma * math.sqrt(h))),
                "sigma_2": float(last * math.exp(-2.0 * ewma * math.sqrt(h))),
                "sessions": h,
            }
    else:
        downside = {"1d": {}, "30d": {}, "90d": {}}

    data_ok = n_obs >= 90 and xau_last is not None and fx_last is not None and not stale
    history_note = (
        f"Configured ingest start is {cfg.start_date}. Stored {cfg.primary_series} runs "
        f"{start.isoformat()} → {as_of.isoformat()} ({n_obs:,} sessions, ~{years} years). "
        "Holdout evaluation uses the last 252 sessions, not a 1-year sample. "
        "The chart can zoom; zooming does not drop history from the database."
    )
    return {
        "data_ok": data_ok,
        "stale": stale,
        "stale_lag_days": lag,
        "n_obs": n_obs,
        "history_start": start.isoformat(),
        "history_end": as_of.isoformat(),
        "history_years": years,
        "configured_start": cfg.start_date,
        "as_of": as_of.isoformat(),
        "parity_inr_10g": last,
        "gold_usd_oz": xau_last,
        "usd_inr": fx_last,
        "duty_pct": duty_pct,
        "etf_inr_10g": etf_last,
        "etf_premium_inr": etf_prem,
        "etf_premium_pct": None if etf_last is None else etf_last / last - 1.0,
        "ma": mas,
        "vs_ma": vs_mas,
        "high": highs,
        "drawdown": dds,
        "ret_1d_parity": r1_p,
        "ret_1d_xau": r1_x,
        "ret_1d_fx": r1_f,
        "ret_21d_parity": r21_p,
        "ret_21d_xau": r21_x,
        "ret_21d_fx": r21_f,
        "ewma_vol": ewma if math.isfinite(ewma) else None,
        "ewma_vol_percentile": ewma_pct,
        "downside": downside,
        "hist_p05_21d": p05_21,
        "incomplete": incomplete,
        "history_note": history_note,
        "identity_note": "dP/P ≈ dXAU/XAU + dUSDINR/USDINR on international parity.",
    }


def _fires(signal_id: str, base: dict[str, Any], quotes: dict[str, Any]) -> bool | None:
    """True/False, or None when the check is not applicable."""
    dd90 = (base.get("drawdown") or {}).get("90")
    vs = base.get("vs_ma") or {}
    ewma_pct = base.get("ewma_vol_percentile")
    dealer = quotes.get("dealer_price")
    premium_pct = quotes.get("physical_premium_pct")
    wedge = quotes.get("tax_wedge_pct")
    r21_p = base.get("ret_21d_parity")
    r21_x = base.get("ret_21d_xau")
    r21_f = base.get("ret_21d_fx")

    if signal_id == "off_90d_high":
        return dd90 is not None and dd90 <= -0.05
    if signal_id == "below_longer_mas":
        a, b = vs.get("30"), vs.get("90")
        if a is None or b is None:
            return None
        return a < 0 and b < 0
    if signal_id == "tight_premium":
        physical = quotes.get("estimated_physical_inr_10g")
        if dealer is None or physical in (None, 0):
            return None
        return dealer / float(physical) - 1.0 <= 0.005
    if signal_id == "calm_vol":
        if ewma_pct is None:
            return None
        return ewma_pct < 0.80
    if signal_id == "near_90d_high":
        return dd90 is not None and dd90 > -0.015
    if signal_id == "above_all_mas":
        vals = [vs.get("7"), vs.get("30"), vs.get("90")]
        if any(v is None for v in vals):
            return None
        return all(v > 0 for v in vals)
    if signal_id == "fat_premium":
        physical = quotes.get("estimated_physical_inr_10g")
        if dealer is None or physical in (None, 0):
            return None
        return dealer / float(physical) - 1.0 > 0.01
    if signal_id == "fx_driven_spike":
        if r21_p is None or r21_x is None or r21_f is None:
            return None
        return r21_p > 0 and abs(r21_f) > abs(r21_x) and r21_f > 0
    if signal_id == "data_gap":
        return not bool(base.get("data_ok"))
    if signal_id == "extreme_vol":
        if ewma_pct is None:
            return None
        return ewma_pct >= 0.95
    return None


def apply_quotes(
    base: dict[str, Any],
    *,
    dealer_price: float | None = None,
    avg_cost: float | None = None,
    gst_pct: float = DEFAULT_GST_PCT,
    making_pct: float = DEFAULT_MAKING_PCT,
    hallmark_fineness: int = DEFAULT_HALLMARK_FINENESS,
) -> dict[str, Any]:
    """Fold user quotes into the scorecard and evaluate Buy / Hold / Wait / Review."""
    parity = base.get("parity_inr_10g")
    duty = base.get("duty_pct")
    dealer = float(dealer_price) if dealer_price and float(dealer_price) > 0 else None
    avg = float(avg_cost) if avg_cost and float(avg_cost) > 0 else None
    fineness = int(hallmark_fineness or DEFAULT_HALLMARK_FINENESS)
    wedge = tax_wedge_pct(duty, gst_pct, making_pct)
    metal = hallmarked_metal(parity, fineness) if parity else None
    physical = (
        estimated_physical(parity, duty, gst_pct, making_pct, hallmark_fineness=fineness)
        if parity
        else None
    )
    premium = physical_gold_premium(dealer, parity)
    premium_pct = None if premium is None or not parity else premium / float(parity)
    premium_vs_hallmark = None if dealer is None or physical is None else dealer - physical
    vs_avg_parity = None if avg is None or not parity else float(parity) / avg - 1.0
    vs_avg_physical = None if avg is None or physical is None else physical / avg - 1.0
    label = HALLMARK_CHOICES.get(fineness, f"{fineness}‰")

    quotes = {
        "dealer_price": dealer,
        "avg_cost": avg,
        "gst_pct": float(gst_pct),
        "making_pct": float(making_pct),
        "duty_pct": duty,
        "tax_wedge_pct": wedge,
        "hallmark_fineness": fineness,
        "hallmark_label": label,
        "hallmark_factor": hallmark_factor(fineness),
        "hallmarked_metal_inr_10g": metal,
        "estimated_physical_inr_10g": physical,
        "physical_premium_inr": premium,
        "physical_premium_pct": premium_pct,
        "premium_over_tax_inr": premium_vs_hallmark,
        "vs_avg_parity": vs_avg_parity,
        "vs_avg_physical": vs_avg_physical,
        "premium_formula": "Physical Gold Premium = Actual Dealer Price − International Gold Parity Price",
        "physical_formula": (
            f"Estimated physical = parity × ({fineness}/{FINE_PER_MILLE}) × (1 + duty) × (1 + GST) × (1 + making). "
            "BIS hallmark purity only — not a stamped invoice, not MCX/IBJA."
        ),
    }

    signals = []
    for spec in STANCE_RULES:
        if spec["id"] == "conflict":
            continue
        fired = _fires(spec["id"], base, quotes)
        signals.append({**spec, "fires": fired})

    buy_n = sum(1 for s in signals if s["side"] == "buy" and s["fires"] is True)
    wait_n = sum(1 for s in signals if s["side"] == "wait" and s["fires"] is True)
    review_direct = any(s["side"] == "review" and s["fires"] is True for s in signals)
    conflict = buy_n >= 2 and wait_n >= 2
    signals.append({**STANCE_RULES[-1], "fires": conflict})

    if review_direct or conflict:
        label = "Review"
        why = "Data quality, extreme vol, or Buy and Wait both have two or more hits. Not a trade signal."
    elif wait_n >= 3 and wait_n > buy_n:
        label = "Sell"
        why = f"{wait_n} Wait checks fired vs {buy_n} Buy. Stretched vs recent highs/premium — Sell on the checklist, not a predicted drop."
    elif wait_n >= 2 and wait_n > buy_n:
        label = "Wait"
        why = f"{wait_n} Wait checks fired vs {buy_n} Buy. Stretch/premium/FX rules, not a model call."
    elif buy_n >= 2 and buy_n > wait_n:
        label = "Buy"
        why = f"{buy_n} Buy checks fired vs {wait_n} Wait. Relative cheapness vs recent history, not a predicted rise."
    else:
        label = "Hold"
        why = f"Mixed or quiet tape ({buy_n} Buy / {wait_n} Wait). Default is Hold until two checks agree on one side."

    stance = {
        "label": label,
        "why": why,
        "buy_hits": buy_n,
        "wait_hits": wait_n,
        "signals": signals,
        "disclaimer": (
            "Buy / Hold / Wait / Review is a checklist on parity, BIS hallmark purity, "
            "averages, drawdowns, USD/INR contribution, volatility rank, and optional dealer quotes. "
            "It is not an ML prediction, not investment advice, and not an MCX/IBJA call."
        ),
    }
    out = dict(base)
    out["quotes"] = quotes
    out["stance"] = stance
    return out
