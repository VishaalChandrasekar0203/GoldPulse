from __future__ import annotations

from typing import Any

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def _table_from_df(df: pd.DataFrame, title: str, max_rows: int = 40) -> Table:
    table = Table(title=title, show_lines=False, header_style="bold gold1")
    if df.empty:
        table.add_column("status")
        table.add_row("empty")
        return table
    view = df.head(max_rows)
    for col in view.columns:
        table.add_column(str(col))
    for _, row in view.iterrows():
        table.add_row(*[str(row[c])[:48] if pd.notna(row[c]) else "" for c in view.columns])
    return table


def render_coverage(history: dict[str, Any]) -> None:
    catalog = history["catalog"]
    coverage = history["coverage"]
    restricted = history["restricted"]
    console.print(_table_from_df(coverage, "Observation coverage"))
    if not restricted.empty:
        console.print(
            Panel(
                "\n".join(f"[red]{r.series_id}[/red]: {r.notes}" for r in restricted.itertuples()),
                title="Restricted / unavailable (not fabricated)",
            )
        )
    console.print(_table_from_df(catalog[["series_id", "name", "source", "status", "license"]], "Catalog"))


def render_validation(report: dict[str, Any]) -> None:
    series = pd.DataFrame(report["series"])
    console.print(_table_from_df(series, "Validation"))
    anoms = report["recent_anomalies"]
    if not anoms.empty:
        console.print(_table_from_df(anoms, "Recent anomalies (kept unless range-invalid)"))


def render_forecasts(df: pd.DataFrame) -> None:
    if df.empty:
        console.print("[yellow]No forecasts. Run gold forecast --horizon 7d[/yellow]")
        return
    table = Table(title="Forecasts (INR / 10g, international parity proxy)", header_style="bold gold1")
    for col in [
        "horizon_calendar_days",
        "horizon_trading_days",
        "target_date",
        "target_is_trading_day",
        "point",
        "lower",
        "upper",
        "model_name",
        "model_version",
        "cutoff_date",
    ]:
        table.add_column(col)
    for _, r in df.iterrows():
        table.add_row(
            str(int(r["horizon_calendar_days"])),
            str(int(r["horizon_trading_days"])),
            str(r["target_date"])[:10],
            "yes" if r["target_is_trading_day"] else "no (rolled to next session)",
            f"{r['point']:.2f}",
            f"{r['lower']:.2f}",
            f"{r['upper']:.2f}",
            str(r["model_name"]),
            str(r["model_version"])[:22],
            str(r["cutoff_date"])[:10],
        )
    console.print(table)
    console.print(
        Text(
            "Intervals are not guaranteed coverage. 1d uses causal EWMA × √h when configured; other horizons use residual RMSE. Target is not an official MCX/IBJA print.",
            style="dim",
        )
    )


def render_compare(df: pd.DataFrame) -> None:
    if df.empty:
        console.print("[yellow]No experiments. Run gold train --all[/yellow]")
        return
    table = Table(title="Model comparison (chronological holdout)", header_style="bold gold1")
    for col in ["horizon", "model", "mae", "rmse", "mape", "smape", "directional_accuracy", "mae_vs_naive", "is_champion"]:
        table.add_column(col)
    for _, r in df.iterrows():
        champ = "[bold green]yes[/bold green]" if r.get("is_champion") else ""
        table.add_row(
            str(int(r["horizon"])),
            str(r["model"]),
            _fmt(r.get("mae")),
            _fmt(r.get("rmse")),
            _fmt(r.get("mape")),
            _fmt(r.get("smape")),
            _fmt(r.get("directional_accuracy"), 3),
            _fmt(r.get("mae_vs_naive"), 3),
            champ,
        )
    console.print(table)
    console.print("[dim]mae_vs_naive < 1 means better than persistence. Champions require beating naive.[/dim]")


def _fmt(val: Any, digits: int = 2) -> str:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return f"{float(val):.{digits}f}"
    except Exception:
        return str(val)


def render_scenarios(snap, table: pd.DataFrame, disclaimer: str) -> None:
    console.print(
        Panel(
            (
                f"As-of {snap.as_of}\n"
                f"XAU/USD {snap.gold_usd_oz:,.2f}   USD/INR {snap.usd_inr:,.4f}   "
                f"parity {snap.parity_inr_10g:,.2f} INR/10g\n"
                f"DXY {snap.dxy if snap.dxy is not None else 'n/a'}   "
                f"10y {snap.us_10y_yield if snap.us_10y_yield is not None else 'n/a'}   "
                f"real 10y {snap.us_real_yield_10y if snap.us_real_yield_10y is not None else 'n/a'}\n"
                f"Duty {snap.gold_import_duty if snap.gold_import_duty is not None else 'n/a'}%   "
                f"ETF–parity gap "
                f"{(snap.etf_parity_gap_pct * 100):.2f}%" if snap.etf_parity_gap_pct is not None else "ETF–parity gap n/a"
            ),
            title="Market snapshot (international parity)",
            border_style="gold1",
        )
    )
    table_view = table[
        ["scenario", "kind", "scenario_parity_inr_10g", "delta_inr_10g", "delta_pct"]
    ].copy()
    table_view["scenario_parity_inr_10g"] = table_view["scenario_parity_inr_10g"].map(lambda x: f"{x:,.2f}")
    table_view["delta_inr_10g"] = table_view["delta_inr_10g"].map(lambda x: f"{x:,.2f}")
    table_view["delta_pct"] = table_view["delta_pct"].map(lambda x: f"{x:+.3f}%" if pd.notna(x) else "")
    console.print(_table_from_df(table_view, "Conditional INR/10g impact (not a forecast)"))
    for note in table["notes"].tolist():
        console.print(f"[dim]- {note}[/dim]")
    console.print(Panel(disclaimer, title="Disclaimer", border_style="red"))


def render_brief(brief: dict[str, Any]) -> None:
    snap = brief.get("snapshot")
    hero = brief.get("target_note") or ""
    if snap is not None:
        vol = brief.get("parity_vol_ewma")
        vol_txt = f"{vol * 100:.2f}% daily EWMA" if vol else "n/a"
        console.print(
            Panel(
                (
                    f"[b]Parity {snap.parity_inr_10g:,.2f} INR/10g[/b]   as-of {pd.to_datetime(snap.as_of).date()}\n"
                    f"XAU/USD {snap.gold_usd_oz:,.2f}   USD/INR {snap.usd_inr:,.4f}   vol {vol_txt}\n"
                    f"DXY {snap.dxy if snap.dxy is not None else 'n/a'}   "
                    f"10y {snap.us_10y_yield if snap.us_10y_yield is not None else 'n/a'}%   "
                    f"real {snap.us_real_yield_10y if snap.us_real_yield_10y is not None else 'n/a'}%\n"
                    f"{hero}"
                ),
                title="Daily gold intelligence brief",
                border_style="gold1",
            )
        )
    else:
        console.print(Panel("No market snapshot. Run gold data update.", title="Daily brief"))

    fc = brief.get("forecasts")
    if isinstance(fc, pd.DataFrame) and not fc.empty:
        render_forecasts(fc)
        row1 = brief.get("forecast_1d")
        if row1:
            console.print(
                Text(
                    f"1d interval method recorded in notes. "
                    f"Point {float(row1['point']):,.2f}  "
                    f"[{float(row1['lower']):,.2f}, {float(row1['upper']):,.2f}]  "
                    f"model={row1.get('model_name')} {str(row1.get('model_version'))[:22]}",
                    style="dim",
                )
            )

    champs = brief.get("champions")
    if isinstance(champs, pd.DataFrame) and not champs.empty:
        view = champs[["horizon", "model_name", "experiment_id"]].copy() if "horizon" in champs.columns else champs
        console.print(_table_from_df(view, "Production champions (unchanged in Stage 4)"))

    roll = brief.get("rolling")
    if isinstance(roll, pd.DataFrame) and not roll.empty:
        console.print(_table_from_df(roll, "Rolling forecast monitor (MAE / DA / coverage / Winkler)"))

    alerts = brief.get("alerts")
    if isinstance(alerts, pd.DataFrame) and not alerts.empty:
        console.print(
            Panel(
                "\n".join(f"[{r.severity}] {r.message}" for r in alerts.head(8).itertuples()),
                title="Alerts",
                border_style="red",
            )
        )
    else:
        console.print("[dim]No open monitor alerts.[/dim]")

    intel = brief.get("intelligence") or {}
    cards = pd.DataFrame(intel.get("cards") or [])
    if not cards.empty:
        show = cards[[c for c in ["label", "as_of", "last", "ret_1d", "ret_21d", "ewma_vol", "status"] if c in cards.columns]]
        console.print(_table_from_df(show, "Market drivers"))
    contemp = pd.DataFrame(intel.get("contemporaneous") or [])
    pred = pd.DataFrame(intel.get("predictive") or [])
    if not contemp.empty:
        console.print(_table_from_df(contemp[["pair", "pearson", "n", "kind"]], "Contemporaneous correlations (not forecasts)"))
    if not pred.empty:
        console.print(_table_from_df(pred[["pair", "pearson", "n", "kind"]], "Predictive correlations (too weak to beat naive)"))
    regime = intel.get("regime") or {}
    if regime:
        console.print(
            Text(
                f"Regime: {regime.get('label')}  63d log return={regime.get('ret_63d')}  "
                f"historical percentile={regime.get('percentile_vs_history')}  — {regime.get('note')}",
                style="dim",
            )
        )

    packs = brief.get("scenario_packs")
    if isinstance(packs, pd.DataFrame) and not packs.empty:
        view = packs[packs["kind"].isin(["identity_combined", "identity_component"])][
            ["pack_label", "scenario", "kind", "delta_inr_10g", "delta_pct"]
        ]
        console.print(_table_from_df(view, "Named scenario packs (NOT forecasts)"))
        console.print(Panel(brief.get("disclaimer_scenarios") or "", title="Scenario disclaimer", border_style="red"))


def render_monitor(result: dict[str, Any]) -> None:
    roll = result.get("rolling")
    if isinstance(roll, pd.DataFrame):
        console.print(_table_from_df(roll, "Rolling monitor"))
    alerts = result.get("alerts")
    if isinstance(alerts, pd.DataFrame) and not alerts.empty:
        console.print(_table_from_df(alerts[["severity", "kind", "message"]], "Alerts"))
    console.print(
        f"live scored={result.get('n_live_scored')}  backfill={result.get('n_backfill')}  "
        f"outcomes={result.get('n_outcomes')}"
    )
    for p in result.get("charts") or []:
        console.print(f"chart: {p}")


def render_health(health: dict[str, Any], extra: str | None = None) -> None:
    last = health.get("last_ingestion_run") or {}
    pipe = health.get("last_pipeline_run") or {}
    body = (
        f"DuckDB: {health['duckdb']}\n"
        f"Series: {health['n_series']}   Observations: {health['n_observations']}   "
        f"Forecasts: {health['n_forecasts']}   Outcomes: {health.get('n_outcomes', 0)}   "
        f"Audit: {health.get('n_audit_events', 0)}\n"
        f"Last ingest: {last.get('run_id', 'none')}  status={last.get('status', 'n/a')}\n"
        f"Last daily: {pipe.get('run_id', 'none')}  status={pipe.get('status', 'n/a')}  "
        f"forecasted={pipe.get('forecasted', 'n/a')}"
    )
    if extra:
        body += f"\n{extra}"
    console.print(Panel(body, title="System health", border_style="gold1"))
    src = health.get("source_health")
    if isinstance(src, pd.DataFrame) and not src.empty:
        view = src[[c for c in ["series_id", "status", "last_obs", "lag_days", "critical", "message"] if c in src.columns]]
        console.print(_table_from_df(view, "Source health"))
    cov = health.get("coverage")
    if isinstance(cov, pd.DataFrame) and not cov.empty:
        console.print(_table_from_df(cov, "Coverage"))
    audit = health.get("recent_audit")
    if isinstance(audit, pd.DataFrame) and not audit.empty:
        view = audit[[c for c in ["ts", "actor", "kind", "ref_id"] if c in audit.columns]]
        console.print(_table_from_df(view, "Recent audit"))
