from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import duckdb
import pandas as pd

from gold.config import AppConfig
from gold.database.schema import SCHEMA_SQL
from gold.logging import get_logger

log = get_logger("database")

UTC = timezone.utc


def _utcnow() -> datetime:
    return datetime.now(tz=UTC).replace(tzinfo=None)


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S')}_{uuid4().hex[:8]}"


class GoldStore:
    """DuckDB catalog + Parquet-backed observation store."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.cfg.duckdb_path))
        self.con.execute("SET TimeZone='UTC'")
        self.init_schema()

    def init_schema(self) -> None:
        self.con.execute(SCHEMA_SQL)
        self._migrate_audit_run_id()

    def _migrate_audit_run_id(self) -> None:
        try:
            cols = [r[1] for r in self.con.execute("PRAGMA table_info('audit_events')").fetchall()]
        except Exception:  # noqa: BLE001
            return
        if cols and "run_id" not in cols:
            self.con.execute("ALTER TABLE audit_events ADD COLUMN run_id VARCHAR")

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "GoldStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- catalog -----------------------------------------------------
    def upsert_catalog(self, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        df = pd.DataFrame(rows)
        df["updated_at"] = _utcnow()
        self.con.register("catalog_in", df)
        self.con.execute(
            """
            INSERT OR REPLACE INTO series_catalog
            SELECT series_id, name, description, unit, frequency, source, source_url,
                   license, publication_lag_days, category, status, notes, updated_at
            FROM catalog_in
            """
        )
        self.con.unregister("catalog_in")

    def catalog(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM series_catalog ORDER BY series_id").df()

    def restricted_sources(self) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM series_catalog WHERE status IN ('unavailable', 'restricted') ORDER BY series_id"
        ).df()

    # --- observations ------------------------------------------------
    def parquet_path(self, series_id: str) -> Path:
        path = self.cfg.parquet_dir / "observations" / f"{series_id}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def load_series(self, series_id: str) -> pd.DataFrame:
        path = self.parquet_path(series_id)
        if not path.exists():
            return pd.DataFrame(
                columns=[
                    "series_id",
                    "obs_date",
                    "value",
                    "available_at",
                    "frequency",
                    "source",
                    "ingested_at",
                    "run_id",
                    "is_derived",
                    "quality_flag",
                    "provenance",
                ]
            )
        df = pd.read_parquet(path)
        df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.date
        df["available_at"] = pd.to_datetime(df["available_at"]).dt.date
        return df

    def last_observation_date(self, series_id: str) -> date | None:
        df = self.load_series(series_id)
        if df.empty:
            return None
        return max(df["obs_date"])

    def upsert_observations(self, df: pd.DataFrame, series_id: str) -> dict[str, int]:
        if df.empty:
            return {"inserted": 0, "updated": 0, "skipped": 0}
        required = {
            "series_id",
            "obs_date",
            "value",
            "available_at",
            "frequency",
            "source",
            "ingested_at",
            "run_id",
            "is_derived",
            "quality_flag",
            "provenance",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Observation frame missing columns: {missing}")

        incoming = df.copy()
        incoming["obs_date"] = pd.to_datetime(incoming["obs_date"]).dt.date
        incoming["available_at"] = pd.to_datetime(incoming["available_at"]).dt.date
        incoming = incoming.dropna(subset=["value", "obs_date"])
        incoming = incoming.drop_duplicates(subset=["series_id", "obs_date"], keep="last")

        existing = self.load_series(series_id)
        if existing.empty:
            combined = incoming
            inserted, updated = len(incoming), 0
        else:
            existing_idx = existing.set_index("obs_date")
            incoming_idx = incoming.set_index("obs_date")
            new_dates = incoming_idx.index.difference(existing_idx.index)
            overlap = incoming_idx.index.intersection(existing_idx.index)
            changed = []
            for d in overlap:
                old = existing_idx.loc[d, "value"]
                new = incoming_idx.loc[d, "value"]
                if pd.isna(old) or pd.isna(new) or float(old) != float(new):
                    changed.append(d)
            inserted, updated = len(new_dates), len(changed)
            keep_old = existing_idx.drop(index=list(overlap), errors="ignore")
            combined = pd.concat(
                [keep_old.reset_index(), incoming.reset_index(drop=True)], ignore_index=True
            )
            combined = combined.drop_duplicates(subset=["obs_date"], keep="last")

        combined = combined.sort_values("obs_date").reset_index(drop=True)
        path = self.parquet_path(series_id)
        out = combined.copy()
        out["obs_date"] = pd.to_datetime(out["obs_date"])
        out["available_at"] = pd.to_datetime(out["available_at"])
        out["ingested_at"] = pd.to_datetime(out["ingested_at"])
        if "provenance" in out.columns:
            out["provenance"] = out["provenance"].map(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else (x if pd.notna(x) else "{}")
            )
        out.to_parquet(path, index=False)

        self.con.execute("DELETE FROM observations WHERE series_id = ?", [series_id])
        self.con.register("obs_in", out)
        self.con.execute(
            """
            INSERT INTO observations
            SELECT series_id, obs_date, value, available_at, frequency, source,
                   ingested_at, run_id, is_derived, quality_flag, provenance
            FROM obs_in
            """
        )
        self.con.unregister("obs_in")
        log.info(
            "%s upserted: inserted=%s updated=%s total=%s",
            series_id,
            inserted,
            updated,
            len(combined),
        )
        return {"inserted": inserted, "updated": updated, "skipped": 0, "total": len(combined)}

    def observations(
        self,
        series_id: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        q = "SELECT * FROM observations WHERE 1=1"
        params: list[Any] = []
        if series_id:
            q += " AND series_id = ?"
            params.append(series_id)
        if start:
            q += " AND obs_date >= ?"
            params.append(start)
        if end:
            q += " AND obs_date <= ?"
            params.append(end)
        q += " ORDER BY series_id, obs_date"
        return self.con.execute(q, params).df()

    def wide_prices(self, series_ids: list[str] | None = None) -> pd.DataFrame:
        df = self.observations()
        if df.empty:
            return df
        if series_ids:
            df = df[df["series_id"].isin(series_ids)]
        wide = df.pivot_table(index="obs_date", columns="series_id", values="value", aggfunc="last")
        wide.index = pd.to_datetime(wide.index)
        return wide.sort_index()

    def coverage(self) -> pd.DataFrame:
        return self.con.execute(
            """
            SELECT series_id,
                   min(obs_date) AS start_date,
                   max(obs_date) AS end_date,
                   count(*) AS n_obs,
                   count(value) AS n_non_null,
                   max(ingested_at) AS last_ingested
            FROM observations
            GROUP BY series_id
            ORDER BY series_id
            """
        ).df()

    # --- runs / anomalies / experiments / forecasts ------------------
    def record_ingestion_run(self, payload: dict[str, Any]) -> None:
        self.con.execute(
            """
            INSERT OR REPLACE INTO ingestion_runs
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                payload["run_id"],
                payload.get("started_at"),
                payload.get("finished_at"),
                payload.get("status"),
                payload.get("records_fetched", 0),
                payload.get("records_inserted", 0),
                payload.get("records_updated", 0),
                json.dumps(payload.get("skipped_sources", [])),
                json.dumps(payload.get("errors", [])),
                payload.get("notes"),
            ],
        )

    def record_anomalies(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.con.register("anom_in", df)
        self.con.execute(
            """
            INSERT OR REPLACE INTO anomalies
            SELECT series_id, obs_date, value, zscore, reason, detected_at FROM anom_in
            """
        )
        self.con.unregister("anom_in")

    def save_features(self, df: pd.DataFrame) -> Path:
        path = self.cfg.parquet_dir / "features" / "features.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return path

    def load_features(self) -> pd.DataFrame:
        path = self.cfg.parquet_dir / "features" / "features.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def record_experiment(self, payload: dict[str, Any]) -> None:
        self.con.execute(
            """
            INSERT OR REPLACE INTO experiments
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                payload["experiment_id"],
                payload.get("model_name"),
                payload.get("horizon"),
                payload.get("trained_at"),
                payload.get("cutoff_date"),
                json.dumps(payload.get("metrics", {})),
                json.dumps(payload.get("params", {})),
                payload.get("artifact_path"),
                payload.get("is_champion", False),
                payload.get("notes"),
            ],
        )
        path = self.cfg.parquet_dir / "experiments" / "experiments.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        current = self.con.execute("SELECT * FROM experiments").df()
        current.to_parquet(path, index=False)

    def experiments(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM experiments ORDER BY trained_at DESC").df()

    def record_training_run(self, payload: dict[str, Any]) -> None:
        self.con.execute(
            """
            INSERT OR REPLACE INTO training_runs
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                payload["run_id"],
                payload.get("pipeline_version"),
                payload.get("started_at"),
                payload.get("finished_at"),
                payload.get("n_experiments", 0),
                json.dumps(payload.get("comparison_json", [])),
                payload.get("notes"),
            ],
        )
        path = self.cfg.parquet_dir / "experiments" / "training_runs.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        current = self.con.execute("SELECT * FROM training_runs ORDER BY started_at").df()
        current.to_parquet(path, index=False)

    def training_runs(self) -> pd.DataFrame:
        try:
            return self.con.execute("SELECT * FROM training_runs ORDER BY started_at DESC").df()
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def champions(self) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM experiments WHERE is_champion = TRUE ORDER BY horizon"
        ).df()

    def save_forecasts(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.con.register("fc_in", df)
        self.con.execute(
            """
            INSERT OR IGNORE INTO forecasts
            SELECT forecast_id, generated_at, model_name, model_version, target_series,
                   cutoff_date, horizon_calendar_days, horizon_trading_days, target_date,
                   target_is_trading_day, point, lower, upper, interval_low, interval_high,
                   data_cutoff, notes
            FROM fc_in
            """
        )
        self.con.unregister("fc_in")
        path = self.cfg.parquet_dir / "forecasts" / "forecasts.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        all_fc = self.con.execute("SELECT * FROM forecasts ORDER BY generated_at DESC").df()
        all_fc.to_parquet(path, index=False)

    def forecasts_for_cutoff(
        self,
        cutoff: date,
        horizons: list[int] | None = None,
        series: str | None = None,
    ) -> pd.DataFrame:
        """Earliest live book for a cutoff. Used to skip duplicate daily generation."""
        df = self.all_forecasts()
        if df.empty:
            return df
        series = series or self.cfg.primary_series
        out = df.copy()
        out["cutoff_date"] = pd.to_datetime(out["cutoff_date"]).dt.date
        cut = pd.to_datetime(cutoff).date()
        hit = out[(out["cutoff_date"] == cut) & (out["target_series"] == series)]
        if horizons is not None:
            want = {int(h) for h in horizons}
            hit = hit[hit["horizon_calendar_days"].astype(int).isin(want)]
        if hit.empty:
            return hit
        hit = hit.sort_values(["horizon_calendar_days", "generated_at"])
        return hit.drop_duplicates(subset=["horizon_calendar_days"], keep="first").reset_index(drop=True)

    def latest_forecasts(self) -> pd.DataFrame:
        return self.con.execute(
            """
            SELECT * FROM forecasts
            WHERE generated_at = (SELECT max(generated_at) FROM forecasts)
            ORDER BY horizon_calendar_days, target_date
            """
        ).df()

    def all_forecasts(self) -> pd.DataFrame:
        try:
            return self.con.execute(
                "SELECT * FROM forecasts ORDER BY generated_at DESC, horizon_calendar_days"
            ).df()
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def save_forecast_outcomes(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.con.register("out_in", df)
        self.con.execute(
            """
            INSERT OR IGNORE INTO forecast_outcomes
            SELECT outcome_id, forecast_id, source, model_name, model_version, target_series,
                   cutoff_date, target_date, horizon_calendar_days, interval_method,
                   origin_close, point, lower, upper, realized, abs_error, signed_error,
                   covered, winkler, direction_correct, scored_at
            FROM out_in
            """
        )
        self.con.unregister("out_in")
        path = self.cfg.parquet_dir / "forecasts" / "forecast_outcomes.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con.execute("SELECT * FROM forecast_outcomes").df().to_parquet(path, index=False)

    def forecast_outcomes(self) -> pd.DataFrame:
        try:
            return self.con.execute(
                "SELECT * FROM forecast_outcomes ORDER BY cutoff_date, horizon_calendar_days"
            ).df()
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def save_alerts(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.con.register("al_in", df)
        self.con.execute(
            """
            INSERT OR REPLACE INTO alerts
            SELECT alert_id, created_at, severity, kind, horizon, series_id, message, payload
            FROM al_in
            """
        )
        self.con.unregister("al_in")

    def alerts(self, limit: int = 50) -> pd.DataFrame:
        try:
            return self.con.execute(
                """
                SELECT * FROM alerts
                WHERE created_at = (SELECT max(created_at) FROM alerts)
                ORDER BY severity DESC, kind
                LIMIT ?
                """,
                [limit],
            ).df()
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def save_quality_snapshots(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.con.register("qs_in", df)
        self.con.execute(
            """
            INSERT OR REPLACE INTO quality_snapshots
            SELECT snapshot_id, captured_at, series_id, n_obs, start_date, end_date,
                   last_available_at, n_missing_provenance, n_timestamp_issues,
                   n_flagged, n_gaps, stale, notes
            FROM qs_in
            """
        )
        self.con.unregister("qs_in")
        path = self.cfg.parquet_dir / "quality" / "quality_snapshots.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con.execute("SELECT * FROM quality_snapshots ORDER BY captured_at").df().to_parquet(
            path, index=False
        )

    def quality_snapshots(self) -> pd.DataFrame:
        try:
            return self.con.execute(
                "SELECT * FROM quality_snapshots ORDER BY captured_at DESC"
            ).df()
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def append_audit(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        actor: str = "system",
        ref_id: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """Append-only. Never UPDATE/DELETE audit_events."""
        event_id = new_run_id("audit")
        ts = _utcnow()
        body = json.dumps(payload or {})
        self.con.execute(
            """
            INSERT INTO audit_events (event_id, ts, actor, kind, ref_id, run_id, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [event_id, ts, actor, kind, ref_id, run_id, body],
        )
        path = self.cfg.parquet_dir / "audit" / "audit_events.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            all_ev = self.con.execute("SELECT * FROM audit_events ORDER BY ts").df()
            all_ev.to_parquet(path, index=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("audit parquet write failed: %s", exc)
        return event_id

    def audit_events(self, kind: str | None = None, limit: int = 100) -> pd.DataFrame:
        try:
            if kind:
                return self.con.execute(
                    "SELECT * FROM audit_events WHERE kind = ? ORDER BY ts DESC LIMIT ?",
                    [kind, limit],
                ).df()
            return self.con.execute(
                "SELECT * FROM audit_events ORDER BY ts DESC LIMIT ?",
                [limit],
            ).df()
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def save_source_health(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        self.con.register("sh_in", df)
        self.con.execute(
            """
            INSERT OR REPLACE INTO source_health
            SELECT check_id, checked_at, series_id, status, last_obs, lag_days, critical, message
            FROM sh_in
            """
        )
        self.con.unregister("sh_in")

    def source_health(self) -> pd.DataFrame:
        try:
            return self.con.execute(
                """
                SELECT * FROM source_health
                WHERE checked_at = (SELECT max(checked_at) FROM source_health)
                ORDER BY critical DESC, series_id
                """
            ).df()
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def record_pipeline_run(self, payload: dict[str, Any]) -> None:
        self.con.execute(
            """
            INSERT OR REPLACE INTO pipeline_runs
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                payload["run_id"],
                payload.get("started_at"),
                payload.get("finished_at"),
                payload.get("status"),
                payload.get("ingest_status"),
                bool(payload.get("forecasted", False)),
                payload.get("n_forecasts", 0),
                payload.get("n_outcomes_scored", 0),
                payload.get("backup_path"),
                payload.get("notes"),
                json.dumps(payload.get("payload", payload.get("detail", {}))),
            ],
        )

    def pipeline_runs(self, limit: int = 20) -> pd.DataFrame:
        try:
            return self.con.execute(
                "SELECT * FROM pipeline_runs ORDER BY started_at DESC LIMIT ?",
                [limit],
            ).df()
        except Exception:  # noqa: BLE001
            return pd.DataFrame()

    def health(self) -> dict[str, Any]:
        cov = self.coverage()
        n_series = int(self.con.execute("SELECT count(*) FROM series_catalog").fetchone()[0])
        n_obs = int(self.con.execute("SELECT count(*) FROM observations").fetchone()[0])
        n_fc = int(self.con.execute("SELECT count(*) FROM forecasts").fetchone()[0])
        try:
            n_out = int(self.con.execute("SELECT count(*) FROM forecast_outcomes").fetchone()[0])
        except Exception:  # noqa: BLE001
            n_out = 0
        try:
            n_audit = int(self.con.execute("SELECT count(*) FROM audit_events").fetchone()[0])
        except Exception:  # noqa: BLE001
            n_audit = 0
        last_run = self.con.execute(
            "SELECT * FROM ingestion_runs ORDER BY started_at DESC LIMIT 1"
        ).df()
        last_pipe = self.pipeline_runs(limit=1)
        last_audit = self.audit_events(limit=5)
        return {
            "duckdb": str(self.cfg.duckdb_path),
            "n_series": n_series,
            "n_observations": n_obs,
            "n_forecasts": n_fc,
            "n_outcomes": n_out,
            "n_audit_events": n_audit,
            "coverage": cov,
            "last_ingestion_run": last_run.to_dict(orient="records")[0] if not last_run.empty else None,
            "last_pipeline_run": last_pipe.to_dict(orient="records")[0] if not last_pipe.empty else None,
            "recent_audit": last_audit,
            "source_health": self.source_health(),
        }
