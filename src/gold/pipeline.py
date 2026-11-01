from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from gold.config import AppConfig, ensure_directories
from gold.database.store import GoldStore, new_run_id
from gold.exceptions import LockBusy
from gold.features.builder import FeatureBuilder
from gold.forecasting.engine import ForecastEngine
from gold.ingestion.orchestrator import IngestionOrchestrator
from gold.logging import get_logger
from gold.ops.audit import record_event, set_actor, set_run_id
from gold.ops.backup import backup
from gold.ops.health import assess_source_health, forecast_allowed
from gold.ops.lock import RunLock
from gold.reports.generator import ReportGenerator

log = get_logger("pipeline")


class Pipeline:
    """Daily operational loop. Never retrains. Champions change only via `gold train --all`."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        ensure_directories(cfg)
        self.store = GoldStore(cfg)

    def run(self, force_train: bool = False, horizons: list[int] | None = None) -> dict[str, Any]:
        set_actor("pipeline")
        if force_train:
            log.warning("gold run ignores --force-train; use gold train --all to change champions")
        try:
            with RunLock(self.cfg):
                return self._run(horizons=horizons)
        except LockBusy as exc:
            log.error("%s", exc)
            return {
                "status": "locked",
                "ingestion": {},
                "new_obs": 0,
                "retrained": False,
                "forecasted": False,
                "n_forecasts": 0,
                "n_outcomes": 0,
                "report": None,
                "notes": str(exc),
            }

    def _run(self, horizons: list[int] | None = None) -> dict[str, Any]:
        run_id = new_run_id("daily")
        set_run_id(run_id)
        started = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        ops = self.cfg.get("ops", {}) or {}
        backup_path = None
        if bool(ops.get("backup_before_daily", True)):
            try:
                backup_path = str(backup(self.cfg, self.store, already_locked=True))
                record_event(self.store, "backup", {"path": backup_path}, ref_id=run_id, run_id=run_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("pre-run backup failed: %s", exc)

        ingest = IngestionOrchestrator(self.cfg, self.store)
        ingest.seed_catalog()
        before = (
            {r.series_id: r.end_date for r in self.store.coverage().itertuples()}
            if not self.store.coverage().empty
            else {}
        )

        result = ingest.update(incremental=True)
        log.info("ingest run=%s status=%s inserted=%s", result["run_id"], result["status"], result["records_inserted"])

        new_obs = int(result.get("records_inserted") or 0) + int(result.get("records_updated") or 0)
        health = assess_source_health(self.cfg, self.store, persist=True)
        allowed, gate_reason = forecast_allowed(health, self.cfg)
        log.info("forecast gate: %s (%s)", allowed, gate_reason)
        log.info("modeling frozen: gold run does not retrain (gold train --all to change champions)")

        forecasts = None
        forecasted = False
        reused = False
        if allowed:
            feats = FeatureBuilder(self.cfg, self.store).build()
            log.info("features %s rows", len(feats))
            before_n = len(self.store.all_forecasts())
            forecasts = ForecastEngine(self.cfg, self.store).generate(horizons=horizons)
            after_n = len(self.store.all_forecasts())
            reused = after_n == before_n
            forecasted = True
            log.info("forecasts %s rows reused=%s", len(forecasts), reused)
        else:
            log.warning("skipping forecast generation: %s", gate_reason)

        from gold.evaluation.monitor import run_monitor

        monitor = run_monitor(self.cfg, self.store, backfill=True)
        log.info(
            "monitor outcomes=%s alerts=%s live=%s",
            monitor["n_outcomes"],
            len(monitor["alerts"]),
            monitor["n_live_scored"],
        )
        record_event(
            self.store,
            "outcome",
            {
                "n_live_scored": monitor["n_live_scored"],
                "n_backfill": monitor["n_backfill"],
                "n_outcomes": monitor["n_outcomes"],
            },
            ref_id=run_id,
            run_id=run_id,
        )

        report = ReportGenerator(self.cfg, self.store).write()
        finished = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        status = (
            "ok"
            if allowed and result.get("status") in {"ok", "partial"}
            else ("degraded" if result.get("status") != "error" else "error")
        )
        if not allowed:
            status = "degraded"
        notes = gate_reason if not allowed else "daily run; champions unchanged; no retrain"
        pipe_payload = {
            "run_id": run_id,
            "started_at": started,
            "finished_at": finished,
            "status": status,
            "ingest_status": result.get("status"),
            "forecasted": forecasted,
            "n_forecasts": int(len(forecasts)) if forecasts is not None else 0,
            "n_outcomes_scored": int(monitor.get("n_live_scored") or 0),
            "backup_path": backup_path,
            "notes": notes,
            "payload": {
                "new_obs": new_obs,
                "retrained": False,
                "gate": gate_reason,
                "reused_forecasts": reused,
            },
        }
        self.store.record_pipeline_run(pipe_payload)
        record_event(self.store, "daily", pipe_payload, ref_id=run_id, run_id=run_id)
        log.info("daily %s %s", status, notes)
        return {
            "status": status,
            "run_id": run_id,
            "ingestion": result,
            "new_obs": new_obs,
            "retrained": False,
            "forecasted": forecasted,
            "reused_forecasts": reused,
            "n_forecasts": int(len(forecasts)) if forecasts is not None else 0,
            "n_outcomes": monitor["n_outcomes"],
            "n_live_scored": monitor["n_live_scored"],
            "report": str(report),
            "prior_end_dates": before,
            "backup_path": backup_path,
            "health": health,
            "gate": gate_reason,
            "notes": notes,
        }
