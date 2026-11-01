# Stage 5 — Operational hardening

Date: 2026-11-01. Modeling architecture is frozen. This milestone makes the terminal a daily-running application. It does **not** change features, training, or production champions.

## Production policy (unchanged)

| Horizon | Champion | Interval |
|---|---|---|
| 1d | naive | causal EWMA × √h |
| 7d | arima | residual RMSE |
| 30d | naive | residual RMSE |
| 90d | naive | residual RMSE |

`gold run` never calls `TrainingOrchestrator`. `--force-train` is ignored. `gold train --all` remains the only promotion path. MCX and IBJA stay empty catalog rows. Scenarios remain identity/elasticity what-ifs, never stored as forecasts.

## Daily path (`gold run`)

1. Atomic exclusive lock (`O_EXCL`). Overlap returns status `locked` and exit code 1.
2. Pre-run `CHECKPOINT` + DuckDB/parquet backup (`data/backups/<UTC>/`, SHA-256 in manifest).
3. Incremental ingest with per-source retries/backoff. FRED fallbacks stay fail-fast. A recovery pass retries remaining failures.
4. Source-health snapshot. Forecast gate requires `gold_usd_oz`, `usd_inr`, and `gold_inr_10g` present, finite, and fresh. Restricted series are labelled, never filled.
5. Leak-free features, then forecast generation. Same cutoff + horizons reuses the existing live book (`INSERT OR IGNORE`).
6. Outcome scoring of forecasts whose target dates have now printed (`INSERT OR IGNORE`; already-scored ids are skipped).
7. Append-only `audit_events` with unique `event_id` and shared `run_id`.

Commands: `gold health`, `gold backup`, `gold restore --from <dir> --yes`. Restore refuses without `--yes`, verifies the manifest, then checks DuckDB vs parquet observation counts.

CLI `gold brief` and Streamlit `load_ui_payload` both expose `forecast_book` (forecast_id, generated_at, model_name, model_version, cutoff, horizon, point, lower, upper).

## Test results

Complete suite: **66 passed** (2026-11-01, pytest `-q`). Stage 5 file: **17 passed**.

| Test | What it proves |
|---|---|
| `test_retry_call_recovers_after_transient_failure` | Retry helper succeeds on the third attempt; two attempts still fail. |
| `test_orchestrator_retries_source_then_stores` | Ingest retries a flaky yfinance source and stores the recovered print. |
| `test_recovery_pass_after_injected_outage` | Failure injection: first fetch raises `SourceUnavailable`; recovery pass stores 1810. |
| `test_freshness_gate_blocks_stale_critical_and_leaves_mcx_empty` | Stale critical series block forecasts; MCX remains empty. |
| `test_missing_critical_series_blocks_forecast_gate` | Missing `usd_inr` blocks the gate. |
| `test_pipeline_skips_forecast_when_critical_series_missing` | `gold run` with empty critical series does not forecast and does not train. |
| `test_pipeline_never_trains_and_lock_blocks_overlap` | Pipeline has no `TrainingOrchestrator`; overlapping `gold run` returns `locked`. |
| `test_generate_and_outcome_scoring_are_idempotent` | Second generate keeps the same forecast_id/points; second score inserts nothing. |
| `test_forecast_outcome_scored_when_future_print_arrives` | A later parity print scores the live 1d row once. |
| `test_audit_is_append_only_and_carries_run_id` | Three events, counts only increase, original `event_id` remains, `run_id` is set. |
| `test_backup_restore_roundtrip` | Restore without `--yes` raises; confirmed restore drops post-backup mutations. |
| `test_restore_preserves_duckdb_parquet_consistency` | Manifest SHA-256 present; DuckDB observation count equals parquet count after restore. |
| `test_cli_brief_and_streamlit_share_forecast_book` | Identical fingerprints: 119276.12, Naive/ARIMA versions, `2026-10-30T16:51:32`. |
| `test_cli_help_lists_ops_commands` | `run`, `health`, `backup`, `restore` are registered. |
| `test_restore_cli_requires_yes` | CLI restore without `--yes` exits 1. |
| `test_run_lock_blocks_second_process` | Second `RunLock.acquire` raises `LockBusy`. |

Offline pipeline, Stage 3/4, web UI, and reconstruction tests remain green. Champions in those tests are fixture-local and do not alter the production registry.

## Failure cases (enforced)

- Public feed blip: retry then recovery pass; critical fetch still failing marks ingest `error` and skips the forecast.
- Stale/missing/invalid `gold_usd_oz`, `usd_inr`, or `gold_inr_10g`: no new forecast; outcomes still scored if older targets have printed.
- Overlapping `gold run` / backup / restore: `LockBusy`.
- Duplicate daily run on the same cutoff: reused live book, no second forecast_id.
- Duplicate outcome scoring: `INSERT OR IGNORE` + seen `(forecast_id, horizon)` / `outcome_id`.
- Restore without `--yes`: refused.
- Restore of a backup whose DuckDB and parquet counts disagree: `BackupError`.
- `--force-train` on `gold run`: warning only; no training.

## Backup / restore verification

`gold backup` writes `manifest.json` with `duckdb_sha256`, observation/forecast counts, and a DuckDB-vs-parquet consistency snapshot. After `CHECKPOINT`, WAL is not copied. Restore copies DuckDB + parquet together, then reopens the catalog and asserts observation counts match. Tests confirmed a post-backup extra print (`99999`) is discarded on restore, and DuckDB `count(*)` equals the parquet observation total.

## Outstanding operational risks

- **Weekend/holiday freshness:** the gate uses calendar days (default 4). A long Indian holiday week can skip forecasts even when the last COMEX session is valid. That is safer than generating on stale inputs.
- **FRED fallbacks remain fail-fast and often empty.** They are not critical; an empty fallback is `fallback_empty`, not stale. Yahoo `^TNX` is the live yield.
- **Lock is per-machine file lock**, not a network lock. Two hosts sharing the same `data/` directory over NFS can still race. This app is specified as local-only.
- **Pre-run backup failure is logged and the run continues.** Disk-full backup should be treated as degraded; the catalog is not rolled back automatically.
- **`gold ui` does not take the run lock.** Reading during `gold run` is usually safe (DuckDB); restore while the UI is open can confuse Streamlit’s cache — stop the UI before restore (documented).
- **Naive live refit** on generate still fits persistence on the current window. That is not a champion change; 1d/30d/90d remain naive and 7d remains ARIMA when the artifact loads. If the ARIMA artifact is missing, generate falls back to naive and labels `naive-fallback` (audited).
- **No cloud watchdog.** Cron/launchd must be installed by the operator (`scripts/daily.sh`). A silently dead cron job is outside the process.
- **MCX/IBJA remain unavailable.** Indian retail/premium vs IBJA is still not in the book.

Do not treat daily forecasts as trading advice.
