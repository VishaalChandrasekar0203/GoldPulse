"""Source freshness and health. Restricted series are labelled, never filled."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from gold.config import AppConfig
from gold.data.quality import SERIES_KIND
from gold.database.store import GoldStore, new_run_id
from gold.logging import get_logger

log = get_logger("ops.health")

DEFAULT_CRITICAL = ("gold_usd_oz", "usd_inr", "gold_inr_10g")
BLOCKING = {"missing", "stale", "error", "invalid"}


def _ops(cfg: AppConfig) -> dict:
    return cfg.get("ops", {}) or {}


def critical_series(cfg: AppConfig) -> list[str]:
    raw = _ops(cfg).get("critical_series") or list(DEFAULT_CRITICAL)
    return [str(x) for x in raw]


def assess_source_health(cfg: AppConfig, store: GoldStore, *, persist: bool = True) -> pd.DataFrame:
    today = datetime.now(tz=timezone.utc).date()
    checked = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    check_id = new_run_id("health")
    stale_days = int(_ops(cfg).get("freshness_calendar_days") or (cfg.get("monitor", {}) or {}).get("stale_calendar_days", 4))
    crit = set(critical_series(cfg))
    catalog = store.catalog()
    rows = []
    if catalog.empty:
        return pd.DataFrame()
    for rec in catalog.itertuples():
        df = store.load_series(rec.series_id)
        last = max(df["obs_date"]) if not df.empty else None
        lag = (today - last).days if last is not None else None
        freq = (rec.frequency or "daily").lower()
        status_cat = rec.status or "active"
        if status_cat in {"unavailable", "restricted"}:
            status = "restricted"
            message = f"Restricted/unavailable. Not fabricated. {SERIES_KIND.get(rec.series_id, rec.notes or '')}"
        elif status_cat == "fallback" and df.empty:
            status = "fallback_empty"
            message = "Fallback unused (empty). Not fabricated."
        elif df.empty:
            status = "missing"
            message = "No observations stored."
        else:
            last_row = df.sort_values("obs_date").iloc[-1]
            flag = str(last_row["quality_flag"]) if "quality_flag" in df.columns else "ok"
            try:
                finite = bool(pd.notna(last_row["value"]) and float(last_row["value"]) == float(last_row["value"]))
            except (TypeError, ValueError):
                finite = False
            if not finite or flag == "range":
                status = "invalid"
                message = f"Last print {last} is unusable (flag={flag})."
            elif freq == "daily" and status_cat == "active" and lag is not None and lag > stale_days:
                status = "stale"
                message = f"Last print {last} is {lag} calendar days behind (threshold {stale_days})."
            else:
                status = "ok"
                message = f"Last print {last}."
        rows.append(
            {
                "check_id": f"{check_id}_{rec.series_id}",
                "checked_at": checked,
                "series_id": rec.series_id,
                "status": status,
                "last_obs": last,
                "lag_days": lag,
                "critical": rec.series_id in crit,
                "message": message,
            }
        )
    out = pd.DataFrame(rows)
    if persist and not out.empty:
        store.save_source_health(out)
    return out


def forecast_allowed(health: pd.DataFrame, cfg: AppConfig) -> tuple[bool, str]:
    """Gate live forecasts on critical-series freshness. Does not invent missing prints."""
    if not bool(_ops(cfg).get("skip_forecast_on_critical_failure", True)):
        return True, "forecast gate disabled"
    if health is None or health.empty:
        return False, "no source-health rows; run gold init / gold data update"
    present = set(health["series_id"].astype(str))
    missing_ids = [sid for sid in critical_series(cfg) if sid not in present]
    if missing_ids:
        return False, f"forecast skipped: critical series not in catalog ({', '.join(missing_ids)})"
    crit = health[health["critical"] == True]  # noqa: E712
    if crit.empty:
        return False, "critical series are not in the catalog"
    blocked = crit[crit["status"].isin(BLOCKING)]
    if blocked.empty:
        return True, "critical series are present and fresh"
    detail = ", ".join(f"{r.series_id}={r.status}" for r in blocked.itertuples())
    return False, f"forecast skipped: critical series not ready ({detail})"
