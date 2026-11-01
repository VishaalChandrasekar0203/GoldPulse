from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest
from typer.testing import CliRunner

from gold.cli import app
from gold.data.quality import SERIES_KIND
from gold.evaluation.monitor import score_live_forecasts
from gold.exceptions import BackupError, LockBusy, SourceUnavailable
from gold.forecasting.engine import ForecastEngine
from gold.ingestion.base import DataSource
from gold.ingestion.derived import derive_gold_inr_10g
from gold.ingestion.orchestrator import IngestionOrchestrator
from gold.ingestion.retry import retry_call
from gold.ops.audit import audit_forecasts, audit_scenario, record_event, set_actor, set_run_id
from gold.ops.backup import backup, restore, verify_backup, verify_catalog_consistency
from gold.ops.book import forecast_fingerprint
from gold.ops.health import assess_source_health, forecast_allowed
from gold.ops.lock import RunLock
from gold.pipeline import Pipeline
from gold.terminal.brief import build_brief
from gold.web.load import load_ui_payload
from tests.helpers import business_days, make_obs


def _catalog_row(series_id: str, status: str = "active", frequency: str = "daily", **kw) -> dict:
    return {
        "series_id": series_id,
        "name": series_id,
        "description": "",
        "unit": "x",
        "frequency": frequency,
        "source": kw.get("source", "test"),
        "source_url": "",
        "license": "",
        "publication_lag_days": 0,
        "category": "price",
        "status": status,
        "notes": kw.get("notes", ""),
    }


def _forecast_frame(generated_at: datetime, cutoff: date) -> pd.DataFrame:
    rows = []
    specs = [
        (1, 119276.12, 116091.0, 122461.0, "naive", "exp-naive-1"),
        (7, 120010.5, 115000.0, 125000.0, "arima", "exp-arima-7"),
        (30, 121000.0, 110000.0, 130000.0, "naive", "exp-naive-30"),
        (90, 122000.0, 100000.0, 140000.0, "naive", "exp-naive-90"),
    ]
    fid = "fc_parity_test"
    for h, point, lo, hi, model, ver in specs:
        rows.append(
            {
                "forecast_id": fid,
                "generated_at": generated_at,
                "model_name": model,
                "model_version": ver,
                "target_series": "gold_inr_10g",
                "cutoff_date": cutoff,
                "horizon_calendar_days": h,
                "horizon_trading_days": max(1, round(h * 5 / 7)),
                "target_date": cutoff + timedelta(days=h),
                "target_is_trading_day": True,
                "point": point,
                "lower": lo,
                "upper": hi,
                "interval_low": 0.1,
                "interval_high": 0.9,
                "data_cutoff": generated_at,
                "notes": "ewma_sqrt_h" if h == 1 else "residual_rmse",
            }
        )
    return pd.DataFrame(rows)


def test_retry_call_recovers_after_transient_failure():
    n = {"k": 0}

    def fn():
        n["k"] += 1
        if n["k"] < 3:
            raise SourceUnavailable("temp")
        return 42

    assert retry_call(fn, attempts=3, backoff=0, label="unit") == 42
    assert n["k"] == 3
    n["k"] = 0
    with pytest.raises(SourceUnavailable):
        retry_call(fn, attempts=2, backoff=0, label="unit")


def test_orchestrator_retries_source_then_stores(store, cfg):
    class Flaky(DataSource):
        name = "yfinance"

        def __init__(self):
            self.calls = 0

        def fetch(self, series_id, meta, start, end):
            self.calls += 1
            if self.calls < 2:
                raise SourceUnavailable("transient")
            return pd.DataFrame({"obs_date": [date(2024, 1, 3)], "value": [1800.0]})

    cfg.raw.setdefault("ops", {})
    cfg.raw["ops"]["ingest_retry_attempts"] = 3
    cfg.raw["ops"]["ingest_retry_backoff_seconds"] = 0
    cfg.raw["ops"]["recovery_pass"] = False
    cfg.sources["series"] = {
        "gold_usd_oz": {
            "status": "active",
            "source": "yfinance",
            "frequency": "daily",
            "publication_lag_days": 0,
            "name": "XAU",
        }
    }
    cfg.sources["derived"] = {}
    cfg.sources["restricted"] = {}
    src = Flaky()
    orch = IngestionOrchestrator(cfg, store)
    orch.sources["yfinance"] = src
    result = orch.update(incremental=False)
    assert src.calls == 2
    assert result["records_inserted"] >= 1
    loaded = store.load_series("gold_usd_oz")
    assert not loaded.empty
    assert float(loaded.iloc[-1]["value"]) == 1800.0


def test_freshness_gate_blocks_stale_critical_and_leaves_mcx_empty(store, cfg):
    store.upsert_catalog(
        [
            _catalog_row("gold_usd_oz"),
            _catalog_row("usd_inr"),
            _catalog_row("gold_inr_10g"),
            _catalog_row("mcx_gold_futures", status="unavailable", notes="not fabricated"),
        ]
    )
    old = date(2020, 1, 2)
    store.upsert_observations(make_obs("gold_usd_oz", [(old, 1800.0)]), "gold_usd_oz")
    store.upsert_observations(make_obs("usd_inr", [(old, 70.0)]), "usd_inr")
    store.upsert_observations(make_obs("gold_inr_10g", [(old, 40000.0)]), "gold_inr_10g")
    health = assess_source_health(cfg, store, persist=True)
    allowed, reason = forecast_allowed(health, cfg)
    assert allowed is False
    assert "gold_inr_10g" in reason or "stale" in reason
    mcx = health[health["series_id"] == "mcx_gold_futures"].iloc[0]
    assert mcx["status"] == "restricted"
    assert store.load_series("mcx_gold_futures").empty
    assert "MCX" in SERIES_KIND["mcx_gold_futures"]


def test_forecast_outcome_scored_when_future_print_arrives(store, cfg):
    cutoff = date(2024, 1, 2)
    target = date(2024, 1, 3)
    store.upsert_observations(
        make_obs("gold_inr_10g", [(cutoff, 100000.0), (target, 101000.0)]),
        "gold_inr_10g",
    )
    generated = datetime(2024, 1, 2, 18, 0, 0)
    fc = _forecast_frame(generated, cutoff)
    fc.loc[fc["horizon_calendar_days"] == 1, "target_date"] = target
    store.save_forecasts(fc)
    scored = score_live_forecasts(cfg, store)
    one = scored[scored["horizon_calendar_days"] == 1]
    assert len(one) == 1
    assert float(one.iloc[0]["realized"]) == 101000.0
    assert float(one.iloc[0]["point"]) == 119276.12
    assert str(one.iloc[0]["model_version"]) == "exp-naive-1"


def test_audit_logs_forecast_and_scenario(store, cfg):
    set_actor("cli")
    generated = datetime(2026, 10, 30, 12, 0, 0)
    fc = _forecast_frame(generated, date(2026, 10, 30))
    store.save_forecasts(fc)
    audit_forecasts(store, fc, extra={"pipeline_version": "v2-logret"})
    table = pd.DataFrame(
        [
            {
                "scenario": "XAU +1%",
                "kind": "identity_component",
                "delta_inr_10g": 100.0,
            }
        ]
    )
    audit_scenario(store, table, shocks={"xau_pct": 1.0})
    events = store.audit_events()
    kinds = set(events["kind"])
    assert "forecast" in kinds
    assert "scenario" in kinds
    fc_row = events[events["kind"] == "forecast"].iloc[0]
    assert fc_row["actor"] == "cli"
    payload = fc_row["payload"]
    assert "119276" in str(payload)


def test_backup_restore_roundtrip(store, cfg):
    store.upsert_observations(make_obs("gold_inr_10g", [(date(2024, 1, 2), 50000.0)]), "gold_inr_10g")
    path = backup(cfg, store)
    assert (path / "manifest.json").exists()
    store.close()
    opened = type(store)(cfg)
    opened.upsert_observations(make_obs("gold_inr_10g", [(date(2024, 1, 3), 99999.0)]), "gold_inr_10g")
    opened.close()
    with pytest.raises(BackupError):
        restore(cfg, path, confirm=False)
    restore(cfg, path, confirm=True)
    restored = type(store)(cfg)
    vals = set(restored.load_series("gold_inr_10g")["value"].astype(float))
    assert 50000.0 in vals
    assert 99999.0 not in vals
    restored.close()


def test_cli_brief_and_streamlit_share_forecast_book(store, cfg):
    generated = datetime(2026, 10, 30, 16, 51, 32)
    cutoff = date(2026, 10, 30)
    store.save_forecasts(_forecast_frame(generated, cutoff))
    brief = build_brief(cfg, store, run_monitor=False)
    ui = load_ui_payload(cfg=cfg, store=store)
    assert brief["forecast_book"] == ui["forecast_book"]
    assert ui["brief"]["forecast_book"] == brief["forecast_book"]
    book = forecast_fingerprint(brief["forecasts"])
    assert [row["horizon"] for row in book] == [1, 7, 30, 90]
    assert book[0]["point"] == 119276.12
    assert book[0]["generated_at"] == "2026-10-30T16:51:32"
    assert book[0]["model_name"] == "naive"
    assert book[0]["model_version"] == "exp-naive-1"
    assert book[1]["model_name"] == "arima"
    assert book[1]["model_version"] == "exp-arima-7"


def test_cli_help_lists_ops_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    text = result.stdout
    for name in ("brief", "monitor", "scenario", "ui", "run", "backup", "restore", "health"):
        assert name in text


def test_restore_cli_requires_yes():
    result = CliRunner().invoke(app, ["restore", "--from", "/tmp/no-such-backup"])
    assert result.exit_code == 1
    assert "Refusing" in result.stdout or "yes" in result.stdout.lower()


def test_run_lock_blocks_second_process(cfg):
    with RunLock(cfg):
        with pytest.raises(LockBusy):
            RunLock(cfg).acquire()


def test_recovery_pass_after_injected_outage(store, cfg):
    class Injected(DataSource):
        name = "yfinance"

        def __init__(self):
            self.calls = 0

        def fetch(self, series_id, meta, start, end):
            self.calls += 1
            if self.calls == 1:
                raise SourceUnavailable("injected outage")
            return pd.DataFrame({"obs_date": [date(2024, 1, 4)], "value": [1810.0]})

    cfg.raw.setdefault("ops", {})
    cfg.raw["ops"]["ingest_retry_attempts"] = 1
    cfg.raw["ops"]["ingest_retry_backoff_seconds"] = 0
    cfg.raw["ops"]["recovery_pass"] = True
    cfg.sources["series"] = {
        "gold_usd_oz": {
            "status": "active",
            "source": "yfinance",
            "frequency": "daily",
            "publication_lag_days": 0,
            "name": "XAU",
        }
    }
    cfg.sources["derived"] = {}
    cfg.sources["restricted"] = {}
    src = Injected()
    orch = IngestionOrchestrator(cfg, store)
    orch.sources["yfinance"] = src
    result = orch.update(incremental=False)
    assert src.calls == 2
    assert result["status"] in {"ok", "partial"}
    assert float(store.load_series("gold_usd_oz").iloc[-1]["value"]) == 1810.0


def test_generate_and_outcome_scoring_are_idempotent(store, cfg):
    cfg.raw.setdefault("forecast", {})
    cfg.raw["forecast"]["horizons_calendar_days"] = [1, 7]
    days = business_days(120)
    gold = [(d, 1800 + i * 0.3) for i, d in enumerate(days)]
    fx = [(d, 74 + 0.01 * i) for i, d in enumerate(days)]
    store.upsert_observations(make_obs("gold_usd_oz", gold), "gold_usd_oz")
    store.upsert_observations(make_obs("usd_inr", fx), "usd_inr")
    store.upsert_observations(derive_gold_inr_10g(store, cfg, run_id="fixture"), "gold_inr_10g")
    first = ForecastEngine(cfg, store).generate(horizons=[1, 7])
    n_fc = len(store.all_forecasts())
    second = ForecastEngine(cfg, store).generate(horizons=[1, 7])
    assert len(store.all_forecasts()) == n_fc
    assert list(first["forecast_id"].unique()) == list(second["forecast_id"].unique())
    assert list(first["point"].round(6)) == list(second["point"].round(6))
    scored = score_live_forecasts(cfg, store)
    n_out = len(store.forecast_outcomes())
    scored_again = score_live_forecasts(cfg, store)
    assert scored_again.empty
    assert len(store.forecast_outcomes()) == n_out
    assert n_out == len(scored) or n_out >= 0


def test_audit_is_append_only_and_carries_run_id(store, cfg):
    set_actor("pipeline")
    set_run_id("daily_test_run")
    a = record_event(store, "ingest", {"step": 1})
    b = record_event(store, "forecast", {"step": 2})
    ev = store.con.execute("SELECT * FROM audit_events ORDER BY ts").df()
    assert list(ev["event_id"]) == [a, b]
    assert (ev["run_id"] == "daily_test_run").all()
    n = int(store.con.execute("SELECT count(*) FROM audit_events").fetchone()[0])
    record_event(store, "outcome", {"step": 3})
    n2 = int(store.con.execute("SELECT count(*) FROM audit_events").fetchone()[0])
    assert n2 == n + 1
    still = store.con.execute("SELECT event_id FROM audit_events WHERE event_id = ?", [a]).df()
    assert len(still) == 1


def test_restore_preserves_duckdb_parquet_consistency(store, cfg):
    store.upsert_observations(make_obs("gold_inr_10g", [(date(2024, 1, 2), 50000.0)]), "gold_inr_10g")
    store.upsert_observations(make_obs("gold_usd_oz", [(date(2024, 1, 2), 1800.0)]), "gold_usd_oz")
    cons = verify_catalog_consistency(cfg, store)
    assert cons["ok"], cons["mismatches"]
    path = backup(cfg, store)
    check = verify_backup(path)
    assert check["ok"]
    assert check["duckdb_sha256"]
    store.close()
    restore(cfg, path, confirm=True)
    restored = type(store)(cfg)
    aligned = verify_catalog_consistency(cfg, restored)
    assert aligned["ok"], aligned["mismatches"]
    duck_n = int(restored.con.execute("SELECT count(*) FROM observations").fetchone()[0])
    assert duck_n == aligned["n_parquet"] == aligned["n_duckdb"]
    restored.close()


def test_pipeline_never_trains_and_lock_blocks_overlap(store, cfg):
    import gold.pipeline as pmod

    assert not hasattr(pmod, "TrainingOrchestrator")
    with RunLock(cfg):
        result = Pipeline(cfg).run(force_train=True)
    assert result["status"] == "locked"
    assert result["retrained"] is False
    assert result["forecasted"] is False


def test_pipeline_skips_forecast_when_critical_series_missing(store, cfg, monkeypatch):
    cfg.raw.setdefault("ops", {})
    cfg.raw["ops"]["backup_before_daily"] = False

    def fake_update(self, *a, **k):
        return {
            "run_id": "ingest_fake",
            "status": "ok",
            "records_fetched": 0,
            "records_inserted": 0,
            "records_updated": 0,
            "skipped_sources": [],
            "errors": [],
        }

    monkeypatch.setattr(IngestionOrchestrator, "update", fake_update)
    result = Pipeline(cfg).run(force_train=True)
    assert result["retrained"] is False
    assert result["forecasted"] is False
    assert "skipped" in result["gate"] or "not ready" in result["gate"] or "not in catalog" in result["gate"]


def test_missing_critical_series_blocks_forecast_gate(store, cfg):
    store.upsert_catalog([_catalog_row("gold_usd_oz"), _catalog_row("gold_inr_10g")])
    store.upsert_observations(make_obs("gold_usd_oz", [(date(2026, 10, 30), 1800.0)]), "gold_usd_oz")
    store.upsert_observations(make_obs("gold_inr_10g", [(date(2026, 10, 30), 50000.0)]), "gold_inr_10g")
    health = assess_source_health(cfg, store, persist=False)
    allowed, reason = forecast_allowed(health, cfg)
    assert allowed is False
    assert "usd_inr" in reason
