from __future__ import annotations

from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from gold.config import ensure_directories, load_config
from gold.database.store import GoldStore
from gold.logging import setup_logging
from gold.terminal.render import (
    console,
    render_brief,
    render_compare,
    render_coverage,
    render_forecasts,
    render_health,
    render_monitor,
    render_scenarios,
    render_validation,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Indian gold price forecasting system (INR per 10g).",
)
data_app = typer.Typer(help="Data ingestion and validation.")
app.add_typer(data_app, name="data")


def _boot():
    cfg = load_config()
    ensure_directories(cfg)
    setup_logging(cfg.log_dir)
    from gold.ops.audit import set_actor

    set_actor("cli")
    return cfg


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    ui: bool = typer.Option(False, "--ui", help="Launch the local Streamlit UI (localhost)."),
) -> None:
    if ctx.invoked_subcommand is None:
        if ui:
            _launch_streamlit()
        else:
            typer.echo(ctx.get_help())


@app.command("init")
def init() -> None:
    """Create local directories, DuckDB schema, and series catalog."""
    cfg = _boot()
    from gold.ingestion.orchestrator import IngestionOrchestrator

    store = GoldStore(cfg)
    IngestionOrchestrator(cfg, store).seed_catalog()
    console.print(f"[green]Initialized[/green] DuckDB at {cfg.duckdb_path}")
    console.print(f"Data dir: {cfg.data_dir}")
    console.print("MCX and IBJA official histories are marked unavailable — they will not be fabricated.")
    store.close()


@data_app.command("update")
def data_update(
    full: bool = typer.Option(False, "--full", help="Re-download from start_date instead of incremental."),
) -> None:
    """Download/update historical series and derived INR/10g gold."""
    cfg = _boot()
    from gold.ingestion.orchestrator import IngestionOrchestrator

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        progress.add_task("Ingesting public series (rate-limited)...", total=None)
        orch = IngestionOrchestrator(cfg)
        result = orch.update(incremental=not full)
    console.print(
        f"[green]Ingest {result['status']}[/green]  fetched={result['records_fetched']}  "
        f"inserted={result['records_inserted']}  updated={result['records_updated']}"
    )
    if result["skipped_sources"]:
        console.print("[yellow]Skipped sources (not fabricated):[/yellow]")
        for item in result["skipped_sources"]:
            console.print(f"  - {item['series_id']}: {item['reason']}")
    if result["errors"]:
        console.print("[red]Errors:[/red]")
        for item in result["errors"]:
            console.print(f"  - {item['series_id']}: {item['error']}")


@data_app.command("history")
def data_history() -> None:
    """Show catalog, coverage, and restricted sources."""
    cfg = _boot()
    from gold.ingestion.orchestrator import IngestionOrchestrator

    render_coverage(IngestionOrchestrator(cfg).history())


@data_app.command("validate")
def data_validate() -> None:
    """Validate stored series (gaps, flags, anomalies)."""
    cfg = _boot()
    from gold.ingestion.orchestrator import IngestionOrchestrator

    render_validation(IngestionOrchestrator(cfg).validate())


@app.command("features")
def features(
    action: str = typer.Argument("build", help="Currently only 'build'."),
) -> None:
    """Build leak-free feature matrix."""
    if action != "build":
        raise typer.BadParameter("Only `gold features build` is supported")
    cfg = _boot()
    from gold.features.builder import FeatureBuilder

    store = GoldStore(cfg)
    df = FeatureBuilder(cfg, store).build()
    console.print(f"[green]Features[/green] {len(df)} rows x {df.shape[1]} columns")
    store.close()


@app.command("train")
def train(
    all_models: bool = typer.Option(False, "--all", help="Train/compare the full model set."),
) -> None:
    """Walk-forward train on h-day log returns; promote only if holdout and 3 regimes beat naive."""
    cfg = _boot()
    from gold.training.orchestrator import TrainingOrchestrator

    store = GoldStore(cfg)
    names = ["naive", "moving_average", "arima", "xgboost", "lightgbm", "ensemble"]
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        progress.add_task("Training with walk-forward validation...", total=None)
        df = TrainingOrchestrator(cfg, store).train_all(models=names)
    from gold.terminal.render import render_compare
    from gold.evaluation.backtest import Backtester

    render_compare(Backtester(cfg, store).run())
    store.close()


@app.command("backtest")
def backtest() -> None:
    """Summarize stored chronological holdout results."""
    cfg = _boot()
    from gold.evaluation.backtest import Backtester

    store = GoldStore(cfg)
    render_compare(Backtester(cfg, store).run())
    store.close()


@app.command("forecast")
def forecast(
    horizon: str = typer.Option("7d", "--horizon", help="1d, 7d, 30d, 90d, or all"),
) -> None:
    """Generate point forecasts and prediction intervals."""
    cfg = _boot()
    from gold.forecasting.engine import ForecastEngine

    mapping = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}
    if horizon in ("all", "*"):
        horizons = [1, 7, 30, 90]
    elif horizon in mapping:
        horizons = [mapping[horizon]]
    else:
        raise typer.BadParameter("horizon must be 1d, 7d, 30d, 90d, or all")
    store = GoldStore(cfg)
    df = ForecastEngine(cfg, store).generate(horizons=horizons)
    render_forecasts(df)
    store.close()


@app.command("brief")
def brief(
    refresh_monitor: bool = typer.Option(False, "--refresh-monitor", help="Re-score outcomes and backfill monitor history."),
) -> None:
    """Daily intelligence summary: parity, forecasts, 1d interval, drivers, scenarios, audit."""
    cfg = _boot()
    from gold.terminal.brief import build_brief

    store = GoldStore(cfg)
    payload = build_brief(cfg, store, run_monitor=refresh_monitor)
    render_brief(payload)
    store.close()


@app.command("monitor")
def monitor(
    no_backfill: bool = typer.Option(False, "--no-backfill", help="Score live forecasts only."),
) -> None:
    """Score realized outcomes, rolling MAE/coverage/Winkler, and calibration alerts."""
    cfg = _boot()
    from gold.evaluation.monitor import run_monitor

    store = GoldStore(cfg)
    result = run_monitor(cfg, store, backfill=not no_backfill)
    render_monitor(result)
    store.close()


@app.command("scenario")
def scenario(
    xau: float = typer.Option(1.0, "--xau", help="XAU/USD percent shock (identity)."),
    usdinr: float = typer.Option(1.0, "--usdinr", help="USD/INR percent shock (identity)."),
    dxy: float = typer.Option(1.0, "--dxy", help="DXY percent shock (sample elasticity, not a forecast)."),
    yield_bp: float = typer.Option(25.0, "--yield-bp", help="US 10y yield shock in basis points (elasticity)."),
    real_yield_bp: float = typer.Option(25.0, "--real-yield-bp", help="US 10y TIPS shock in basis points (elasticity)."),
    duty_pp: float = typer.Option(1.0, "--duty-pp", help="Import duty change in percentage points (onshore add-on)."),
    premium_pp: float = typer.Option(1.0, "--premium-pp", help="ETF–parity gap change in percentage points (onshore add-on)."),
    pack: bool = typer.Option(False, "--pack", help="Also print bull/bear/stress identity packs."),
) -> None:
    """Conditional INR/10g scenarios. Not statistically validated forecasts."""
    cfg = _boot()
    from gold.scenarios.engine import SCENARIO_DISCLAIMER, build, build_packs

    store = GoldStore(cfg)
    snap, _elas, table = build(
        cfg,
        store,
        xau_pct=xau,
        usdinr_pct=usdinr,
        dxy_pct=dxy,
        yield_bp=yield_bp,
        real_yield_bp=real_yield_bp,
        duty_pp=duty_pp,
        premium_pp=premium_pp,
    )
    from gold.ops.audit import audit_scenario

    audit_scenario(
        store,
        table,
        shocks={
            "xau_pct": xau,
            "usdinr_pct": usdinr,
            "dxy_pct": dxy,
            "yield_bp": yield_bp,
            "real_yield_bp": real_yield_bp,
            "duty_pp": duty_pp,
            "premium_pp": premium_pp,
        },
    )
    render_scenarios(snap, table, SCENARIO_DISCLAIMER)
    out = cfg.reports_dir / "latest_scenario.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    console.print(f"[green]Wrote[/green] {out}")
    if pack:
        _snap, _e, packs = build_packs(cfg, store)
        from gold.ops.audit import audit_scenario as _audit_pack

        _audit_pack(store, packs, pack=True)
        packs_path = cfg.reports_dir / "latest_scenario_packs.csv"
        packs.to_csv(packs_path, index=False)
        from gold.terminal.render import _table_from_df

        view = packs[["pack_label", "scenario", "kind", "delta_inr_10g", "delta_pct"]]
        console.print(_table_from_df(view, "Bull / bear / stress packs (NOT forecasts)"))
        console.print(f"[green]Wrote[/green] {packs_path}")
    store.close()


@app.command("compare")
def compare() -> None:
    """Compare models on independent chronological holdout."""
    backtest()


@app.command("status")
def status() -> None:
    """System health snapshot."""
    cfg = _boot()
    from gold.ops.health import assess_source_health

    store = GoldStore(cfg)
    assess_source_health(cfg, store, persist=True)
    render_health(store.health())
    store.close()


@app.command("health")
def health_cmd() -> None:
    """Source freshness and critical-series gate (does not fetch)."""
    cfg = _boot()
    from gold.ops.health import assess_source_health, forecast_allowed
    from gold.terminal.render import _table_from_df

    store = GoldStore(cfg)
    table = assess_source_health(cfg, store, persist=True)
    allowed, reason = forecast_allowed(table, cfg)
    render_health(store.health())
    if not table.empty:
        view = table[["series_id", "status", "last_obs", "lag_days", "critical", "message"]]
        console.print(_table_from_df(view, "Source health"))
    color = "green" if allowed else "yellow"
    console.print(f"[{color}]Forecast gate:[/{color}] {reason}")
    store.close()


@app.command("backup")
def backup_cmd() -> None:
    """Copy DuckDB and parquet into data/backups/<timestamp>."""
    cfg = _boot()
    from gold.ops.backup import backup

    store = GoldStore(cfg)
    path = backup(cfg, store)
    store.close()
    console.print(f"[green]Backup[/green] {path}")


@app.command("restore")
def restore_cmd(
    source: str = typer.Option(..., "--from", help="Backup directory produced by gold backup."),
    yes: bool = typer.Option(False, "--yes", help="Required. Overwrites the local DuckDB catalog."),
) -> None:
    """Restore DuckDB and parquet from a local backup. Stop gold ui first."""
    cfg = _boot()
    from gold.exceptions import BackupError
    from gold.ops.backup import restore

    if not yes:
        console.print("[red]Refusing to overwrite.[/red] Re-run with --yes after stopping gold ui / gold run.")
        raise typer.Exit(code=1)
    try:
        path = restore(cfg, source, confirm=True)
    except BackupError as exc:
        console.print(f"[red]Restore failed:[/red] {exc}")
        raise typer.Exit(code=1)
    console.print(f"[green]Restored[/green] {path}")
    console.print("MCX/IBJA remain unavailable and are not fabricated on restore.")


@app.command("report")
def report() -> None:
    """Write a markdown report under reports/output."""
    cfg = _boot()
    from gold.reports.generator import ReportGenerator

    store = GoldStore(cfg)
    path = ReportGenerator(cfg, store).write()
    console.print(f"[green]Report[/green] {path}")
    store.close()


@app.command("run")
def run(
    force_train: bool = typer.Option(False, "--force-train"),
    horizon: str = typer.Option("all", "--horizon"),
) -> None:
    """Daily loop: lock → backup → ingest (retry) → freshness gate → forecast → score outcomes.

    Never retrains. `gold train --all` is the only champion-promotion command.
    """
    cfg = _boot()
    from gold.pipeline import Pipeline

    if force_train:
        console.print(
            "[yellow]Ignoring --force-train.[/yellow] gold run never retrains. Use gold train --all."
        )
    mapping = {"1d": [1], "7d": [7], "30d": [30], "90d": [90], "all": [1, 7, 30, 90]}
    horizons = mapping.get(horizon, [1, 7, 30, 90])
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        progress.add_task("Running daily operational pipeline...", total=None)
        result = Pipeline(cfg).run(force_train=False, horizons=horizons)
    if result.get("status") == "locked":
        console.print(f"[red]Locked[/red] {result.get('notes')}")
        raise typer.Exit(code=1)
    color = "green" if result.get("status") == "ok" else "yellow"
    console.print(
        f"[{color}]Pipeline {result.get('status')}[/{color}] retrained={result['retrained']} "
        f"forecasted={result.get('forecasted')} new_obs={result['new_obs']} "
        f"forecasts={result['n_forecasts']} outcomes={result.get('n_outcomes')}"
    )
    if result.get("gate"):
        console.print(f"Gate: {result['gate']}")
    if result.get("backup_path"):
        console.print(f"Backup: {result['backup_path']}")
    if result.get("report"):
        console.print(f"Report: {result['report']}")


def _launch_streamlit(host: str = "127.0.0.1", port: int = 8501) -> None:
    import os
    import subprocess

    cfg = _boot()
    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print("[red]Streamlit is not installed.[/red] Optional extra: [bold]pip install -e '.[ui]'[/bold]")
        console.print("CLI commands keep working without it: gold brief · gold monitor · gold scenario")
        raise typer.Exit(code=1)
    if host not in {"127.0.0.1", "localhost", "::1"}:
        console.print("[red]gold ui is local-only.[/red] Use --host 127.0.0.1")
        raise typer.Exit(code=1)
    from gold.web import APP_PATH, streamlit_command

    if not APP_PATH.exists():
        console.print(f"[red]UI app missing:[/red] {APP_PATH}")
        raise typer.Exit(code=1)
    env = os.environ.copy()
    env["GOLD_HOME"] = str(cfg.project_root)
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    cmd = streamlit_command(APP_PATH, host=host, port=port)
    console.print(f"[green]Gold Pulse[/green] → http://{host}:{port}")
    console.print("[dim]CLI remains available in another terminal. Ctrl+C stops the UI only.[/dim]")
    raise SystemExit(subprocess.call(cmd, env=env))


@app.command("tui")
def tui() -> None:
    """Launch the optional Textual terminal dashboard."""
    _boot()
    from gold.terminal.app import GoldTUI

    GoldTUI().run()


@app.command("ui")
def ui_cmd(
    port: int = typer.Option(8501, "--port", help="Local port."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (localhost only)."),
) -> None:
    """Launch the optional local Streamlit UI. Binds to localhost; does not change champions."""
    _launch_streamlit(host=host, port=port)


if __name__ == "__main__":
    app()
