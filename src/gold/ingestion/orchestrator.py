from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

from gold.config import AppConfig, ensure_directories
from gold.data.quality import gap_report, validate_observations
from gold.database.store import GoldStore, new_run_id
from gold.exceptions import InsufficientDataError, SourceUnavailable
from gold.ingestion.derived import derive_gold_etf_inr_10g, derive_gold_inr_10g
from gold.ingestion.fred_source import FredSource
from gold.ingestion.retry import retry_call
from gold.ingestion.static_source import StaticTableSource
from gold.ingestion.treasury_source import TreasurySource
from gold.ingestion.worldbank_source import WorldBankSource
from gold.ingestion.yfinance_source import YahooFinanceSource
from gold.logging import get_logger

log = get_logger("ingestion")


class IngestionOrchestrator:
    def __init__(self, cfg: AppConfig, store: GoldStore | None = None):
        self.cfg = cfg
        ensure_directories(cfg)
        self.store = store or GoldStore(cfg)
        self.sources = {
            "yfinance": YahooFinanceSource(cfg),
            "fred": FredSource(cfg),
            "treasury": TreasurySource(cfg),
            "static": StaticTableSource(cfg),
            "worldbank": WorldBankSource(cfg),
        }
        self.pause = float(cfg.get("ingestion.min_interval_seconds", 0.8))

    def _ops(self) -> dict[str, Any]:
        return self.cfg.get("ops", {}) or {}

    def _attempts_for(self, meta: dict[str, Any]) -> int:
        if meta.get("source") == "fred" or meta.get("status") == "fallback":
            return 1
        return int(self._ops().get("ingest_retry_attempts", 3))

    def _backoff(self) -> float:
        return float(self._ops().get("ingest_retry_backoff_seconds", 1.5))

    def _ingest_one(self, series_id: str, meta: dict[str, Any], start: date | None, end: date | None, incremental: bool, run_id: str) -> dict[str, Any]:
        src_name = meta.get("source")
        source = self.sources.get(src_name)
        if source is None:
            raise SourceUnavailable(f"unknown source {src_name}")
        s, e = self._window(series_id, None if incremental else start, end)
        if incremental and start is not None:
            s = min(s, start) if isinstance(start, date) else s
        if s > e:
            log.info("%s already current through %s", series_id, e)
            return {"fetched": 0, "inserted": 0, "updated": 0, "skipped": True}
        raw = retry_call(
            lambda: source.fetch(series_id, meta, s, e),
            attempts=self._attempts_for(meta),
            backoff=self._backoff(),
            retry_on=(SourceUnavailable, TimeoutError, ConnectionError),
            label=f"{src_name}:{series_id}",
        )
        finalized = source.finalize(raw, series_id, meta, run_id)
        cleaned, anomalies = validate_observations(finalized, series_id, self.cfg)
        if not anomalies.empty:
            self.store.record_anomalies(anomalies)
        stats = self.store.upsert_observations(cleaned, series_id)
        gaps = gap_report(cleaned, meta.get("frequency", "daily"))
        if not gaps.empty:
            log.warning("%s calendar gaps: %s", series_id, len(gaps))
        return {
            "fetched": len(raw),
            "inserted": stats.get("inserted", 0),
            "updated": stats.get("updated", 0),
            "skipped": False,
        }

    def seed_catalog(self) -> None:
        rows = []
        for series_id, meta in (self.cfg.sources.get("series") or {}).items():
            rows.append(
                {
                    "series_id": series_id,
                    "name": meta.get("name"),
                    "description": meta.get("notes") or meta.get("name"),
                    "unit": meta.get("unit"),
                    "frequency": meta.get("frequency"),
                    "source": meta.get("source"),
                    "source_url": meta.get("source_url"),
                    "license": meta.get("license"),
                    "publication_lag_days": int(meta.get("publication_lag_days") or 0),
                    "category": meta.get("category"),
                    "status": meta.get("status", "active"),
                    "notes": meta.get("notes"),
                }
            )
        for series_id, meta in (self.cfg.sources.get("derived") or {}).items():
            rows.append(
                {
                    "series_id": series_id,
                    "name": meta.get("name"),
                    "description": meta.get("notes"),
                    "unit": "INR per 10g",
                    "frequency": "daily",
                    "source": "derived",
                    "source_url": None,
                    "license": "Derived from licensed public inputs; see input series",
                    "publication_lag_days": 0,
                    "category": "price",
                    "status": "active",
                    "notes": meta.get("notes"),
                }
            )
        for series_id, meta in (self.cfg.sources.get("restricted") or {}).items():
            rows.append(
                {
                    "series_id": series_id,
                    "name": meta.get("name"),
                    "description": meta.get("reason"),
                    "unit": "INR per 10g",
                    "frequency": "daily",
                    "source": "restricted",
                    "source_url": None,
                    "license": None,
                    "publication_lag_days": 0,
                    "category": "price",
                    "status": meta.get("status", "unavailable"),
                    "notes": f"{meta.get('reason')} Fallback: {meta.get('fallback')}",
                }
            )
        self.store.upsert_catalog(rows)

    def _window(self, series_id: str, start: date | None, end: date | None) -> tuple[date, date]:
        default_start = date.fromisoformat(self.cfg.start_date)
        last = self.store.last_observation_date(series_id)
        if start is None:
            start = last + timedelta(days=1) if last else default_start
        if end is None:
            end = datetime.now(tz=timezone.utc).date()
        return start, end

    def update(self, start: date | None = None, end: date | None = None, incremental: bool = True) -> dict[str, Any]:
        self.seed_catalog()
        run_id = new_run_id("ingest")
        started = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        fetched = inserted = updated = 0
        errors: list[dict[str, str]] = []
        skipped: list[dict[str, str]] = []
        series_meta = self.cfg.sources.get("series") or {}

        failed: list[tuple[str, dict[str, Any]]] = []
        for series_id, meta in series_meta.items():
            if meta.get("status") not in (None, "active"):
                skipped.append({"series_id": series_id, "reason": meta.get("status")})
                continue
            src_name = meta.get("source")
            if self.sources.get(src_name) is None:
                skipped.append({"series_id": series_id, "reason": f"unknown source {src_name}"})
                continue
            try:
                stats = self._ingest_one(series_id, meta, start, end, incremental, run_id)
                fetched += int(stats.get("fetched") or 0)
                inserted += int(stats.get("inserted") or 0)
                updated += int(stats.get("updated") or 0)
            except SourceUnavailable as exc:
                log.warning("skip %s: %s", series_id, exc)
                skipped.append({"series_id": series_id, "reason": str(exc)})
                failed.append((series_id, meta))
            except Exception as exc:  # noqa: BLE001
                log.exception("failed %s", series_id)
                errors.append({"series_id": series_id, "error": str(exc)})
                failed.append((series_id, meta))
            time.sleep(self.pause)

        if failed and bool(self._ops().get("recovery_pass", True)):
            log.info("recovery pass for %s failed series", len(failed))
            recovered: list[tuple[str, dict[str, Any]]] = []
            for series_id, meta in failed:
                try:
                    stats = self._ingest_one(series_id, meta, start, end, incremental, run_id)
                    fetched += int(stats.get("fetched") or 0)
                    inserted += int(stats.get("inserted") or 0)
                    updated += int(stats.get("updated") or 0)
                    skipped = [s for s in skipped if s.get("series_id") != series_id]
                    errors = [e for e in errors if e.get("series_id") != series_id]
                    log.info("recovered %s", series_id)
                except Exception as exc:  # noqa: BLE001
                    recovered.append((series_id, meta))
                    log.warning("recovery still failed for %s: %s", series_id, exc)
            failed = recovered

        for derive_fn, sid in (
            (derive_gold_inr_10g, "gold_inr_10g"),
            (derive_gold_etf_inr_10g, "gold_etf_inr_10g"),
        ):
            try:
                derived = derive_fn(self.store, self.cfg, run_id)
                cleaned, anomalies = validate_observations(derived, sid, self.cfg)
                if not anomalies.empty:
                    self.store.record_anomalies(anomalies)
                stats = self.store.upsert_observations(cleaned, sid)
                inserted += stats.get("inserted", 0)
                updated += stats.get("updated", 0)
            except InsufficientDataError as exc:
                skipped.append({"series_id": sid, "reason": str(exc)})
            except Exception as exc:  # noqa: BLE001
                errors.append({"series_id": sid, "error": str(exc)})

        finished = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        critical = set(self._ops().get("critical_series") or ["gold_usd_oz", "usd_inr", "gold_inr_10g"])
        fetch_fail_ids = {item[0] for item in failed}
        crit_err = [e for e in errors if e.get("series_id") in critical]
        crit_skip = [s for s in skipped if s.get("series_id") in critical and s.get("series_id") in fetch_fail_ids]
        if crit_err or crit_skip:
            status = "error"
        elif errors or fetch_fail_ids:
            status = "partial"
        else:
            status = "ok"
        payload = {
            "run_id": run_id,
            "started_at": started,
            "finished_at": finished,
            "status": status,
            "records_fetched": fetched,
            "records_inserted": inserted,
            "records_updated": updated,
            "skipped_sources": skipped,
            "errors": errors,
            "notes": "incremental" if incremental else "full",
        }
        self.store.record_ingestion_run(payload)
        try:
            from gold.ops.audit import record_event

            record_event(
                self.store,
                "ingest",
                {
                    "status": status,
                    "records_fetched": fetched,
                    "records_inserted": inserted,
                    "records_updated": updated,
                    "errors": errors,
                    "skipped": skipped,
                },
                ref_id=run_id,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("ingest audit failed: %s", exc)
        return payload

    def history(self) -> dict[str, Any]:
        self.seed_catalog()
        return {
            "catalog": self.store.catalog(),
            "coverage": self.store.coverage(),
            "restricted": self.store.restricted_sources(),
        }

    def validate(self) -> dict[str, Any]:
        reports = []
        for _, row in self.store.catalog().iterrows():
            if row["status"] != "active":
                continue
            df = self.store.load_series(row["series_id"])
            gaps = gap_report(df, row["frequency"] or "daily")
            reports.append(
                {
                    "series_id": row["series_id"],
                    "n_obs": len(df),
                    "start": min(df["obs_date"]) if not df.empty else None,
                    "end": max(df["obs_date"]) if not df.empty else None,
                    "n_flagged": int((df["quality_flag"] != "ok").sum()) if not df.empty else 0,
                    "n_gaps": int(len(gaps)),
                }
            )
        anomalies = self.store.con.execute("SELECT * FROM anomalies ORDER BY detected_at DESC LIMIT 50").df()
        return {"series": reports, "recent_anomalies": anomalies}


if __name__ == "__main__":
    from gold.config import load_config
    from gold.logging import setup_logging

    cfg = load_config()
    setup_logging(cfg.log_dir)
    orch = IngestionOrchestrator(cfg)
    print(orch.update())
