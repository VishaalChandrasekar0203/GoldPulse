from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from gold.config import AppConfig, TROY_OUNCE_GRAMS
from gold.database.store import GoldStore
from gold.exceptions import InsufficientDataError
from gold.ingestion.base import apply_availability, utcnow
from gold.logging import get_logger

log = get_logger("ingestion.derived")


def _pair(store: GoldStore, left: str, right: str) -> pd.DataFrame:
    a = store.load_series(left)
    b = store.load_series(right)
    if a.empty or b.empty:
        raise InsufficientDataError(f"Cannot derive: {left} or {right} is empty")
    a = a[["obs_date", "value"]].rename(columns={"value": left, "obs_date": "obs_date"})
    b = b[["obs_date", "value"]].rename(columns={"value": right})
    merged = pd.merge(a, b, on="obs_date", how="inner")
    if merged.empty:
        raise InsufficientDataError(f"No overlapping dates for {left} and {right}")
    return merged


def derive_gold_inr_10g(store: GoldStore, cfg: AppConfig, run_id: str) -> pd.DataFrame:
    """International gold * USDINR converted to INR per 10g. Inner join only — no fill."""
    ounces = cfg.troy_ounce_grams or TROY_OUNCE_GRAMS
    merged = _pair(store, "gold_usd_oz", "usd_inr")
    merged["value"] = merged["gold_usd_oz"] * merged["usd_inr"] * (10.0 / ounces)
    out = pd.DataFrame(
        {
            "series_id": "gold_inr_10g",
            "obs_date": merged["obs_date"],
            "value": merged["value"],
            "frequency": "daily",
            "source": "derived",
            "ingested_at": utcnow(),
            "run_id": run_id,
            "is_derived": True,
            "quality_flag": "ok",
            "provenance": [
                {
                    "formula": "gold_usd_oz * usd_inr * (10 / troy_ounce_grams)",
                    "inputs": ["gold_usd_oz", "usd_inr"],
                    "troy_ounce_grams": ounces,
                    "note": "International parity, not MCX/IBJA official print",
                    "derived_at": utcnow().isoformat(),
                }
            ]
            * len(merged),
        }
    )
    out = apply_availability(out, lag_days=0)
    log.info("derived gold_inr_10g: %s overlapping observations", len(out))
    return out


def derive_gold_etf_inr_10g(store: GoldStore, cfg: AppConfig, run_id: str) -> pd.DataFrame:
    etf = store.load_series("gold_etf_nse")
    if etf.empty:
        raise InsufficientDataError("GOLDBEES series empty; cannot derive ETF-implied INR/10g")
    factor = float(
        ((cfg.sources.get("derived") or {}).get("gold_etf_inr_10g") or {}).get("etf_units_per_10g", 1000)
    )
    out = pd.DataFrame(
        {
            "series_id": "gold_etf_inr_10g",
            "obs_date": etf["obs_date"],
            "value": etf["value"].astype(float) * factor,
            "frequency": "daily",
            "source": "derived",
            "ingested_at": utcnow(),
            "run_id": run_id,
            "is_derived": True,
            "quality_flag": "ok",
            "provenance": [
                {
                    "formula": "gold_etf_nse * etf_units_per_10g",
                    "inputs": ["gold_etf_nse"],
                    "etf_units_per_10g": factor,
                    "note": "ETF-implied India gold price; not IBJA/MCX",
                    "derived_at": utcnow().isoformat(),
                }
            ]
            * len(etf),
        }
    )
    out = apply_availability(out, lag_days=0)
    log.info("derived gold_etf_inr_10g: %s observations (factor=%s)", len(out), factor)
    return out
