from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gold.config import AppConfig
from gold.logging import get_logger

log = get_logger("quality")


def _bounds_for(series_id: str, cfg: AppConfig) -> tuple[float | None, float | None]:
    v = cfg.get("validation", {}) or {}
    if series_id in {"gold_inr_10g", "gold_etf_inr_10g"}:
        return v.get("min_gold_inr_10g", 1000.0), v.get("max_gold_inr_10g", 500000.0)
    if series_id in {"usd_inr", "usd_inr_fred"}:
        return v.get("min_usd_inr", 20.0), v.get("max_usd_inr", 150.0)
    if series_id in {"gold_usd_oz"}:
        return 100.0, 20000.0
    if series_id in {"wti_usd", "wti_fred"}:
        return 0.0, 500.0
    return None, None


def validate_observations(df: pd.DataFrame, series_id: str, cfg: AppConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag missing, out-of-range, and statistical anomalies. Never fills fabricated prices."""
    if df.empty:
        return df, pd.DataFrame()

    out = df.copy()
    out["obs_date"] = pd.to_datetime(out["obs_date"]).dt.date
    out = out.sort_values("obs_date")
    out = out.drop_duplicates(subset=["series_id", "obs_date"], keep="last")

    # Drop non-finite values rather than inventing replacements.
    numeric = pd.to_numeric(out["value"], errors="coerce")
    invalid = ~np.isfinite(numeric)
    if invalid.any():
        log.warning("%s dropped %s non-finite rows (not imputed)", series_id, int(invalid.sum()))
        out = out.loc[~invalid].copy()
        numeric = pd.to_numeric(out["value"], errors="coerce")
    out["value"] = numeric.astype(float)

    lo, hi = _bounds_for(series_id, cfg)
    flags = np.full(len(out), "ok", dtype=object)
    anomalies = []
    if lo is not None:
        bad = (out["value"] < lo) | (out["value"] > hi)
        flags = np.where(bad, "range", flags)
        for row in out.loc[bad].itertuples():
            anomalies.append(
                {
                    "series_id": series_id,
                    "obs_date": row.obs_date,
                    "value": row.value,
                    "zscore": np.nan,
                    "reason": "range",
                    "detected_at": datetime.now(tz=timezone.utc).replace(tzinfo=None),
                }
            )

    skip_z = series_id in {"gold_import_duty", "rbi_repo_rate", "india_festivals", "india_cpi"}
    positive = out["value"] > 0
    returns = np.log(out["value"].where(positive)).diff()
    z = (returns - returns.mean()) / (returns.std(ddof=0) + 1e-12)
    z_thresh = float((cfg.get("validation", {}) or {}).get("zscore_anomaly_threshold", 8.0))
    extreme = z.abs() > z_thresh if not skip_z else pd.Series(False, index=out.index)
    flags = np.where(extreme.fillna(False) & (flags == "ok"), "anomaly", flags)
    for row, zval in zip(out.loc[extreme.fillna(False)].itertuples(), z.loc[extreme.fillna(False)]):
        anomalies.append(
            {
                "series_id": series_id,
                "obs_date": row.obs_date,
                "value": row.value,
                "zscore": float(zval) if pd.notna(zval) else np.nan,
                "reason": "return_zscore",
                "detected_at": datetime.now(tz=timezone.utc).replace(tzinfo=None),
            }
        )

    out["quality_flag"] = flags
    # Range violations are unusable; statistical anomalies are kept but flagged.
    keep = out["quality_flag"] != "range"
    dropped = int((~keep).sum())
    if dropped:
        log.warning("%s dropped %s out-of-range rows", series_id, dropped)
    out = out.loc[keep].copy()
    anom_df = pd.DataFrame(anomalies)
    return out, anom_df


def timestamp_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Flag available_at before obs_date or provenance missing. Never fills values."""
    if df.empty:
        return pd.DataFrame(columns=["obs_date", "issue"])
    out = df.copy()
    out["obs_date"] = pd.to_datetime(out["obs_date"])
    out["available_at"] = pd.to_datetime(out["available_at"], errors="coerce")
    rows = []
    early = out["available_at"].notna() & (out["available_at"] < out["obs_date"])
    for rec in out.loc[early].itertuples():
        rows.append({"obs_date": rec.obs_date.date(), "issue": "available_at_before_obs_date"})
    if "provenance" in out.columns:
        missing = out["provenance"].isna() | (out["provenance"].astype(str).isin(["", "null", "None", "{}"]))
        for rec in out.loc[missing].itertuples():
            rows.append({"obs_date": rec.obs_date.date(), "issue": "missing_provenance"})
    return pd.DataFrame(rows)


def is_stale(last_obs, today, max_calendar_days: int) -> bool:
    if last_obs is None:
        return True
    last = pd.to_datetime(last_obs).date()
    return (today - last).days > int(max_calendar_days)


def series_quality_row(store, cfg: AppConfig, series_id: str, meta: dict, today) -> dict:
    df = store.load_series(series_id)
    freq = (meta or {}).get("frequency") or "daily"
    status = (meta or {}).get("status") or "active"
    gaps = gap_report(df, freq)
    issues = timestamp_issues(df)
    last = max(df["obs_date"]) if not df.empty else None
    last_av = max(df["available_at"]) if not df.empty else None
    stale_days = int((cfg.get("monitor", {}) or {}).get("stale_calendar_days", 4))
    if freq != "daily":
        stale_days = max(stale_days, 40)
    n_flagged = int((df["quality_flag"] != "ok").sum()) if not df.empty and "quality_flag" in df.columns else 0
    # Empty fallbacks and restricted rows are expected absences, not stale prints.
    stale = False
    if df.empty:
        pass
    elif status in {"unavailable", "restricted", "fallback"} and last is None:
        stale = False
    elif freq == "daily" and status == "active":
        stale = bool(is_stale(last, today, stale_days))
    return {
        "series_id": series_id,
        "n_obs": int(len(df)),
        "start_date": min(df["obs_date"]) if not df.empty else None,
        "end_date": last,
        "last_available_at": last_av,
        "n_missing_provenance": int((issues["issue"] == "missing_provenance").sum()) if not issues.empty else 0,
        "n_timestamp_issues": int((issues["issue"] == "available_at_before_obs_date").sum()) if not issues.empty else 0,
        "n_flagged": n_flagged,
        "n_gaps": int(len(gaps)),
        "stale": stale,
        "notes": (meta or {}).get("notes") or "",
    }


SERIES_KIND = {
    "gold_inr_10g": "International parity (not MCX/IBJA)",
    "gold_etf_inr_10g": "GOLDBEES ETF-implied 10g (not IBJA)",
    "gold_etf_nse": "GOLDBEES NSE unit price",
    "mcx_gold_futures": "MCX Gold Futures (unavailable)",
    "ibja_gold": "IBJA gold (unavailable)",
}


def read_quality_table(cfg: AppConfig, store) -> pd.DataFrame:
    """Read-only quality view. Does not write snapshots or fill restricted series."""
    from datetime import datetime, timezone

    today = datetime.now(tz=timezone.utc).date()
    catalog = store.catalog()
    if catalog.empty:
        return pd.DataFrame()
    rows = []
    for rec in catalog.itertuples():
        meta = {"frequency": rec.frequency, "notes": rec.notes, "status": rec.status}
        row = series_quality_row(store, cfg, rec.series_id, meta, today)
        row["status"] = rec.status
        row["source"] = rec.source
        row["kind"] = SERIES_KIND.get(rec.series_id, getattr(rec, "category", "") or "")
        if rec.status in {"unavailable", "restricted"}:
            row["stale"] = False
            row["notes"] = f"Restricted/unavailable. Not fabricated. {rec.notes or ''}".strip()
        elif rec.status == "fallback" and row["n_obs"] == 0:
            row["stale"] = False
            row["notes"] = f"Fallback unused (empty). Not fabricated. {rec.notes or ''}".strip()
        rows.append(row)
    return pd.DataFrame(rows)


def capture_quality_snapshot(cfg: AppConfig, store) -> pd.DataFrame:
    """Persist a source-level quality snapshot. Restricted series stay empty — never fabricated."""
    from datetime import datetime, timezone

    from gold.database.store import new_run_id

    today = datetime.now(tz=timezone.utc).date()
    captured = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    snap_id = new_run_id("qual")
    catalog = store.catalog()
    rows = []
    if catalog.empty:
        return pd.DataFrame()
    for rec in catalog.itertuples():
        meta = {
            "frequency": rec.frequency,
            "notes": rec.notes,
            "status": rec.status,
        }
        row = series_quality_row(store, cfg, rec.series_id, meta, today)
        if rec.status in {"unavailable", "restricted"}:
            row["stale"] = False
            row["notes"] = (
                f"Restricted/unavailable. Not fabricated. {rec.notes or ''}"
            ).strip()
        elif rec.status == "fallback" and row["n_obs"] == 0:
            row["stale"] = False
            row["notes"] = f"Fallback unused (empty). Not fabricated. {rec.notes or ''}".strip()
        row["snapshot_id"] = f"{snap_id}_{rec.series_id}"
        row["captured_at"] = captured
        rows.append(row)
    out = pd.DataFrame(rows)
    store.save_quality_snapshots(out)
    return out


def gap_report(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if df.empty or frequency != "daily":
        return pd.DataFrame(columns=["gap_start", "gap_end", "n_missing_calendar_days"])
    dates = sorted(pd.to_datetime(df["obs_date"]).dt.date.unique())
    gaps = []
    for prev, nxt in zip(dates, dates[1:]):
        delta = (nxt - prev).days
        if delta > 4:  # more than a long weekend
            gaps.append(
                {
                    "gap_start": prev,
                    "gap_end": nxt,
                    "n_missing_calendar_days": delta - 1,
                }
            )
    return pd.DataFrame(gaps)
