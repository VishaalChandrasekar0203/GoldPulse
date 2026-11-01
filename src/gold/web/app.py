"""Local Streamlit UI. Reads DuckDB via existing gold services. Not a second pipeline."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from gold.web.load import UiDataError, load_ui_payload

st.set_page_config(
    page_title="Gold Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #111111; }
      [data-testid="stHeader"] { background: #111111; }
      [data-testid="stSidebar"] { background: #16120a; }
      [data-testid="stMetricValue"] { color: #f3d38a; font-variant-numeric: tabular-nums; }
      .gold-kicker { color: #c9a227; letter-spacing: 0.12em; font-size: 0.72rem; text-transform: uppercase; }
      .gold-title { color: #f3d38a; font-size: 1.7rem; font-weight: 600; margin: 0.15rem 0 0.6rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _inr(v) -> str:
    try:
        return f"₹{float(v):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _pct(v, digits: int = 2) -> str:
    try:
        return f"{100.0 * float(v):+.{digits}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _dt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return str(v)[:19]


def _frame(df, **kwargs):
    """Streamlit 1.66 wants width='stretch'; 1.36 still uses use_container_width."""
    try:
        return st.dataframe(df, hide_index=True, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(df, hide_index=True, use_container_width=True, **kwargs)


@st.cache_data(ttl=30, show_spinner="Loading local DuckDB…")
def _payload(epoch: int):
    # epoch is a cache key so Reload always bypasses the 30s TTL.
    return load_ui_payload()


@st.cache_data(ttl=120, show_spinner=False)
def _scenario_inputs(epoch: int):
    from gold.config import load_config
    from gold.database.store import GoldStore
    from gold.scenarios.engine import sample_elasticities, snapshot

    cfg = load_config()
    store = GoldStore(cfg)
    try:
        return snapshot(store, cfg), sample_elasticities(store)
    finally:
        store.close()


def _require_payload():
    epoch = int(st.session_state.get("db_epoch", 0))
    try:
        return _payload(epoch)
    except UiDataError as exc:
        st.error(str(exc))
        st.info("CLI commands (`gold brief`, `gold monitor`, `gold scenario`) do not need this UI.")
        st.stop()
        raise


def _overview(data: dict) -> None:
    brief = data["brief"]
    snap = brief.get("snapshot")
    st.markdown('<div class="gold-kicker">International gold parity · INR / 10g</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-title">Gold Intelligence · Local</div>', unsafe_allow_html=True)
    if snap is None:
        st.warning("No market snapshot. Run `gold data update`. Parity is not fabricated from MCX/IBJA.")
        return
    fc1 = brief.get("forecast_1d") or {}
    vol = brief.get("parity_vol_ewma")
    champs = brief.get("champions")
    if champs is not None and not champs.empty:
        names = []
        for h in (1, 7, 30, 90):
            hit = champs[champs["horizon"] == h]
            names.append(str(hit.iloc[0]["model_name"]).title() if not hit.empty else "—")
        labels = " / ".join(names)
    else:
        labels = "untrained"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest parity", _inr(snap.parity_inr_10g), help="gold_inr_10g = GC=F × USDINR × (10/31.1034768). Not MCX or IBJA.")
    c2.metric("Daily EWMA volatility", f"{100 * vol:.2f}%" if vol else "n/a")
    if fc1:
        c3.metric("1d forecast interval", f"{_inr(fc1.get('lower'))} – {_inr(fc1.get('upper'))}")
    else:
        c3.metric("1d forecast interval", "n/a")
        st.caption("No stored 1d forecast. Run `gold forecast --horizon all`.")
    c4.metric("Forecast champions", labels)
    c4.caption("1d / 7d / 30d / 90d")

    health = data.get("health") or {}
    last = (health.get("last_ingestion_run") or {}) if isinstance(health, dict) else {}
    fc = brief.get("forecasts")
    gen = None
    if isinstance(fc, pd.DataFrame) and not fc.empty and "generated_at" in fc.columns:
        gen = fc["generated_at"].iloc[0]
    st.caption(
        f"Data cutoff (gold session): {_dt(snap.as_of)} · last ingest: {last.get('run_id', 'n/a')} "
        f"({last.get('status', 'n/a')}) · forecast generated: {_dt(gen)} · {brief.get('target_note')}"
    )

    st.subheader("Horizon forecasts")
    if isinstance(fc, pd.DataFrame) and not fc.empty:
        view = fc.copy()
        view["interval"] = view.apply(
            lambda r: "EWMA × √h" if int(r["horizon_calendar_days"]) == 1 else "Residual RMSE",
            axis=1,
        )
        show = pd.DataFrame(
            {
                "Horizon": view["horizon_calendar_days"].map(lambda h: f"{int(h)}d"),
                "Model": view["model_name"],
                "Version": view["model_version"].astype(str).str[:28],
                "Point": view["point"].map(_inr),
                "Lower": view["lower"].map(_inr),
                "Upper": view["upper"].map(_inr),
                "Interval": view["interval"],
                "Target": pd.to_datetime(view["target_date"]).dt.date.astype(str),
            }
        )
        _frame(show)
    else:
        st.info("No forecasts stored. The CLI command `gold forecast --horizon all` writes them to DuckDB.")

    parity = None
    try:
        from gold.config import load_config
        from gold.database.store import GoldStore

        cfg = load_config()
        store = GoldStore(cfg)
        parity = store.load_series("gold_inr_10g")
        store.close()
    except Exception:  # noqa: BLE001
        parity = None
    if parity is not None and not parity.empty:
        p = parity.sort_values("obs_date").tail(500).copy()
        p["obs_date"] = pd.to_datetime(p["obs_date"])
        st.caption("International parity, last 500 sessions (INR/10g). Not an official MCX/IBJA print.")
        st.line_chart(p.set_index("obs_date")["value"], height=240)


def _monitor(data: dict) -> None:
    st.subheader("Forecast monitor")
    st.caption(
        "Production-policy history uses a naive point. 1d intervals are causal EWMA × √h. "
        "7d/30d/90d monitor bands are train-only σ; live 7d champion remains ARIMA."
    )
    brief = data["brief"]
    roll = brief.get("rolling")
    alerts = brief.get("alerts")
    if isinstance(alerts, pd.DataFrame) and not alerts.empty:
        for rec in alerts.itertuples():
            st.warning(f"{rec.severity}: {rec.message}")
    else:
        st.success("No current calibration alerts.")
    if isinstance(roll, pd.DataFrame) and not roll.empty:
        show = roll.copy()
        show["mae"] = show["mae"].map(lambda x: f"{x:,.0f}")
        show["coverage"] = show["coverage"].map(lambda x: f"{100 * x:.1f}%")
        show["winkler"] = show["winkler"].map(lambda x: f"{x:,.0f}")
        _frame(show)
    outcomes = data.get("outcomes")
    if outcomes is None or outcomes.empty:
        st.info("No scored outcomes yet. Run `gold monitor` from the CLI (does not change champions).")
        return
    horizon = st.selectbox("Horizon", sorted(outcomes["horizon_calendar_days"].unique().tolist()), index=0)
    g = outcomes[outcomes["horizon_calendar_days"] == horizon].sort_values("cutoff_date").copy()
    g["cutoff_date"] = pd.to_datetime(g["cutoff_date"])
    chart = g.set_index("cutoff_date")[["point", "realized", "lower", "upper"]]
    st.caption(f"{int(horizon)}d predicted vs realized parity (INR/10g)")
    st.line_chart(chart, height=280)
    g["rolling_mae"] = g["abs_error"].rolling(21, min_periods=5).mean()
    g["rolling_coverage"] = g["covered"].astype(float).rolling(21, min_periods=5).mean()
    m1, m2 = st.columns(2)
    with m1:
        st.caption("21-origin rolling MAE (INR/10g)")
        st.line_chart(g.set_index("cutoff_date")[["abs_error", "rolling_mae"]], height=220)
    with m2:
        st.caption("21-origin rolling interval coverage (nominal 80%)")
        st.line_chart(g.set_index("cutoff_date")[["rolling_coverage"]], height=220)
    champs = brief.get("champions")
    if champs is not None and not champs.empty:
        st.subheader("Model audit")
        _frame(
            champs[["horizon", "model_name", "experiment_id"]].rename(
                columns={"experiment_id": "model_version"}
            )
        )


def _market(data: dict) -> None:
    st.subheader("Market intelligence")
    intel = data["brief"].get("intelligence") or {}
    cards = pd.DataFrame(intel.get("cards") or [])
    if cards.empty:
        st.info("No driver series loaded.")
    else:
        show = cards.copy()
        if "ret_1d" in show.columns:
            show["Daily"] = show["ret_1d"].map(_pct)
        if "ret_21d" in show.columns:
            show["~Monthly (21d)"] = show["ret_21d"].map(_pct)
        if "ewma_vol" in show.columns:
            show["EWMA vol"] = show["ewma_vol"].map(lambda v: _pct(v).lstrip("+") if v == v else "n/a")
        cols = [c for c in ["label", "as_of", "last", "Daily", "~Monthly (21d)", "EWMA vol", "status"] if c in show.columns]
        _frame(show[cols])
        st.caption("Levels aligned to the last COMEX gold session. GOLDBEES is not IBJA.")

    left, right = st.columns(2)
    contemp = pd.DataFrame(intel.get("contemporaneous") or [])
    pred = pd.DataFrame(intel.get("predictive") or [])
    with left:
        st.markdown("**Contemporaneous** (same-day, descriptive — not a forecast)")
        if not contemp.empty:
            plot = contemp.set_index("pair")["pearson"]
            st.bar_chart(plot, height=280)
            _frame(contemp[["pair", "pearson", "n"]])
    with right:
        st.markdown("**Predictive** (x[t] vs next-day INR gold — too weak to beat naive)")
        if not pred.empty:
            st.bar_chart(pred.set_index("pair")["pearson"], height=280)
            _frame(pred[["pair", "pearson", "n"]])
    regime = intel.get("regime") or {}
    if regime:
        st.caption(
            f"Regime: {regime.get('label')} · 63d log return={regime.get('ret_63d')} · "
            f"historical percentile={regime.get('percentile_vs_history')} — {regime.get('note')}"
        )
    for row in intel.get("restricted") or []:
        st.info(f"{row.get('label')}: {row.get('note')}")


def _scenarios(data: dict) -> None:
    st.subheader("Scenario calculator")
    st.error(data["brief"].get("disclaimer_scenarios") or "SCENARIO, NOT A FORECAST.")
    snap = data["brief"].get("snapshot")
    if snap is None:
        st.warning("Need gold_usd_oz and usd_inr to compute scenarios.")
        return
    st.caption(
        f"Baseline { _dt(snap.as_of) } · XAU/USD {snap.gold_usd_oz:,.2f} · USD/INR {snap.usd_inr:,.4f} · "
        f"parity {_inr(snap.parity_inr_10g)}. Methodology: identity P = XAU × USDINR × (10/31.1034768). "
        "DXY/yield are sample elasticities and are not added to the identity. Duty/premium do not change parity."
    )
    packs = data["brief"].get("scenario_packs")
    if isinstance(packs, pd.DataFrame) and not packs.empty:
        st.markdown("**Named packs (identity combined)**")
        comb = packs[packs["kind"] == "identity_combined"].copy()
        comb["Δ INR/10g"] = comb["delta_inr_10g"].map(_inr)
        comb["Δ%"] = comb["delta_pct"].map(lambda x: f"{x:+.2f}%")
        _frame(comb[["pack_label", "Δ INR/10g", "Δ%", "notes"]])

    st.markdown("**Custom shocks**")
    a, b, c = st.columns(3)
    xau = a.number_input("XAU/USD %", value=1.0, step=0.5)
    usdinr = b.number_input("USD/INR %", value=1.0, step=0.5)
    dxy = c.number_input("DXY % (elasticity overlay)", value=1.0, step=0.5)
    d, e, f = st.columns(3)
    ybp = d.number_input("US 10y bp", value=25.0, step=5.0)
    rbp = e.number_input("US 10y real bp", value=25.0, step=5.0)
    duty = f.number_input("Import duty pp (onshore add-on)", value=1.0, step=0.5)
    prem = st.number_input("ETF–parity gap pp (onshore add-on, not IBJA)", value=1.0, step=0.5)

    from gold.web.load import run_and_audit_scenario

    try:
        snap_obj, elas = _scenario_inputs(int(st.session_state.get("db_epoch", 0)))
        table = run_and_audit_scenario(
            snap_obj,
            elas,
            {
                "xau_pct": float(xau),
                "usdinr_pct": float(usdinr),
                "dxy_pct": float(dxy),
                "yield_bp": float(ybp),
                "real_yield_bp": float(rbp),
                "duty_pp": float(duty),
                "premium_pp": float(prem),
            },
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Scenario engine could not run: {exc}")
        return
    show = table[["scenario", "kind", "scenario_parity_inr_10g", "delta_inr_10g", "delta_pct", "notes"]].copy()
    show["scenario_parity_inr_10g"] = show["scenario_parity_inr_10g"].map(_inr)
    show["delta_inr_10g"] = show["delta_inr_10g"].map(_inr)
    show["delta_pct"] = show["delta_pct"].map(lambda x: f"{x:+.3f}%" if pd.notna(x) else "")
    _frame(show)


def _quality(data: dict) -> None:
    st.subheader("Data quality and provenance")
    st.caption(
        "International parity, GOLDBEES, MCX, and IBJA are separate series. "
        "MCX and IBJA are licensed/unavailable and are never filled with parity."
    )
    q = data.get("quality")
    if q is None or q.empty:
        st.info("Catalog is empty. Run `gold init` and `gold data update`.")
        return
    show = q.copy()
    show["validation"] = show.apply(
        lambda r: (
            "restricted — not fabricated"
            if r.get("status") in {"unavailable", "restricted"}
            else (
                "stale"
                if r.get("stale")
                else ("flagged" if int(r.get("n_flagged") or 0) else "ok")
            )
        ),
        axis=1,
    )
    cols = [
        c
        for c in [
            "series_id",
            "kind",
            "source",
            "status",
            "n_obs",
            "end_date",
            "last_available_at",
            "n_gaps",
            "n_flagged",
            "n_timestamp_issues",
            "n_missing_provenance",
            "validation",
            "notes",
        ]
        if c in show.columns
    ]
    _frame(show[cols])
    highlight = show[show["series_id"].isin(["gold_inr_10g", "gold_etf_inr_10g", "mcx_gold_futures", "ibja_gold"])]
    if not highlight.empty:
        st.markdown("**Target labelling**")
        _frame(
            highlight[[c for c in ["series_id", "kind", "n_obs", "end_date", "validation", "notes"] if c in highlight.columns]]
        )


def main() -> None:
    if "db_epoch" not in st.session_state:
        st.session_state.db_epoch = 0
    with st.sidebar:
        st.markdown("**Gold Intelligence**")
        st.caption("Local · DuckDB · no cloud")
        page = st.radio(
            "Dashboard navigation",
            [
                "Overview",
                "Forecast monitor",
                "Market intelligence",
                "Scenarios",
                "Data quality",
            ],
            label_visibility="visible",
        )
        st.divider()
        if st.button("Reload from DuckDB", help="Force a new read of gold.duckdb. The UI does not poll."):
            st.session_state.db_epoch = int(st.session_state.db_epoch) + 1
            st.session_state.reload_flash = True
            st.cache_data.clear()
            st.rerun()
        if st.session_state.pop("reload_flash", False):
            st.success("Re-read local DuckDB")
        data_preview = None
        try:
            data_preview = _require_payload()
            st.caption(f"Loaded {data_preview.get('loaded_at', 'n/a')} (UTC) · epoch {st.session_state.db_epoch}")
        except Exception:
            data_preview = None
        st.caption("Does not poll. Click Reload after gold data update / gold forecast / gold run.")
        st.caption("CLI: gold brief · gold monitor · gold scenario")

    data = data_preview if data_preview is not None else _require_payload()
    if page == "Overview":
        _overview(data)
    elif page == "Forecast monitor":
        _monitor(data)
    elif page == "Market intelligence":
        _market(data)
    elif page == "Scenarios":
        _scenarios(data)
    else:
        _quality(data)


main()
