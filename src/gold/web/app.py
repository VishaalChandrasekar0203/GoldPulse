"""Local Streamlit UI. Reads DuckDB via existing gold services. Not a second pipeline."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from gold.web.load import UiDataError, load_ui_payload

st.set_page_config(
    page_title="Gold Pulse",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #141416; color: #d8d5cf; }
      [data-testid="stHeader"] { background: #141416; }
      [data-testid="stToolbar"] { visibility: hidden; }
      [data-testid="stDecoration"] { display: none; }
      [data-testid="stSidebar"] { background: #18181b; border-right: 1px solid #26262b; }
      [data-testid="stSidebar"] * { color: #c9c6bf; }
      .block-container { padding-top: 1.35rem; padding-bottom: 2.5rem; max-width: 1240px; }
      footer { visibility: hidden; }
      [data-testid="stMetricValue"] {
        color: #eceae4;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
        overflow: visible !important;
        text-overflow: clip !important;
        font-size: 1.02rem !important;
      }
      div[data-testid="stMetric"] { overflow: visible; }
      .gold-kicker {
        color: #8d8a84;
        letter-spacing: 0.08em;
        font-size: 0.68rem;
        text-transform: uppercase;
        font-weight: 500;
      }
      .gold-title {
        color: #eceae4;
        font-size: 1.55rem;
        font-weight: 560;
        letter-spacing: -0.02em;
        margin: 0.1rem 0 0.35rem 0;
      }
      .gold-card {
        background: #1c1c20;
        border: 1px solid #2a2a30;
        border-radius: 8px;
        padding: 0.8rem 0.95rem 0.85rem 0.95rem;
        margin-bottom: 0.55rem;
        overflow: visible;
      }
      .gold-value {
        color: #eceae4;
        font-size: 1.18rem;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
        overflow: visible;
        line-height: 1.35;
        font-weight: 550;
      }
      .gold-value-wrap { white-space: normal; }
      .gold-sub { color: #9a978f; font-size: 0.75rem; margin-top: 0.22rem; line-height: 1.4; }
      .gold-stance-label { font-size: 1.35rem; font-weight: 600; margin: 0.12rem 0; }
      .stance-Buy {
        background: #d8f3dc;
        border: 1px solid #b7e4c7;
      }
      .stance-Buy .gold-kicker, .stance-Buy .gold-sub { color: #2d6a4f; }
      .stance-Buy .gold-stance-label { color: #1b4332; }
      .stance-Wait {
        background: #fff3bf;
        border: 1px solid #ffe066;
      }
      .stance-Wait .gold-kicker, .stance-Wait .gold-sub { color: #8a6d1b; }
      .stance-Wait .gold-stance-label { color: #5c4b12; }
      .stance-Sell, .stance-Review {
        background: #ffd6d6;
        border: 1px solid #f5b5b5;
      }
      .stance-Sell .gold-kicker, .stance-Sell .gold-sub,
      .stance-Review .gold-kicker, .stance-Review .gold-sub { color: #8b2e2e; }
      .stance-Sell .gold-stance-label, .stance-Review .gold-stance-label { color: #6b1d1d; }
      .stance-Hold {
        background: #eceae4;
        border: 1px solid #d4d1c9;
      }
      .stance-Hold .gold-kicker, .stance-Hold .gold-sub { color: #5c5a55; }
      .stance-Hold .gold-stance-label { color: #2f2e2b; }
      h3, h2 { color: #eceae4 !important; font-weight: 550 !important; letter-spacing: -0.01em; }
      .stCaption, [data-testid="stCaptionContainer"] { color: #8d8a84 !important; }
      div[data-testid="stExpander"] {
        background: #1c1c20;
        border: 1px solid #2a2a30;
        border-radius: 8px;
      }
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


def _card(title: str, value: str, sub: str = "", stance: str | None = None, wrap: bool = False) -> None:
    klass = "gold-card"
    if stance:
        klass += f" stance-{stance}"
    vklass = "gold-value gold-value-wrap" if wrap else "gold-value"
    st.markdown(
        f'<div class="{klass}"><div class="gold-kicker">{title}</div>'
        f'<div class="{vklass}">{value}</div>'
        f'<div class="gold-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=30, show_spinner="Loading DuckDB…")
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
    from gold.config import load_config
    from gold.intelligence.physical import (
        DEFAULT_GST_PCT,
        DEFAULT_HALLMARK_FINENESS,
        DEFAULT_MAKING_PCT,
        HALLMARK_CHOICES,
        apply_quotes,
        save_user_tracker,
    )

    brief = data["brief"]
    snap = brief.get("snapshot")
    base = data.get("physical") or {}
    tracker = data.get("tracker") or {}
    st.markdown('<div class="gold-kicker">International parity · INR / 10g</div>', unsafe_allow_html=True)
    st.markdown('<div class="gold-title">Gold Pulse</div>', unsafe_allow_html=True)
    if snap is None and not base.get("parity_inr_10g"):
        st.warning("No market snapshot. Run `gold data update`. Parity is not fabricated from MCX/IBJA.")
        return

    st.caption(base.get("history_note") or brief.get("target_note"))

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

    default_gst = float(tracker.get("gst_pct") or DEFAULT_GST_PCT)
    default_making = float(tracker.get("making_pct") or DEFAULT_MAKING_PCT)
    default_dealer = float(tracker.get("dealer_price") or 0.0)
    default_avg = float(tracker.get("avg_cost") or 0.0)
    default_hallmark = int(tracker.get("hallmark_fineness") or DEFAULT_HALLMARK_FINENESS)
    if default_hallmark not in HALLMARK_CHOICES:
        default_hallmark = DEFAULT_HALLMARK_FINENESS
    hallmark_labels = list(HALLMARK_CHOICES.values())
    hallmark_keys = list(HALLMARK_CHOICES.keys())

    with st.expander("Dealer quote, hallmark, and average cost", expanded=default_dealer > 0 or default_avg > 0):
        q1, q2 = st.columns(2)
        dealer = q1.number_input(
            "Dealer quote (INR / 10g)",
            min_value=0.0,
            value=default_dealer,
            step=100.0,
            help="What a jeweller or bank is asking today for 10g. Leave 0 if you do not have a quote.",
        )
        avg_cost = q2.number_input(
            "Your average purchase cost (INR / 10g)",
            min_value=0.0,
            value=default_avg,
            step=100.0,
            help="Book cost of gold you already hold, per 10g. Leave 0 to skip the position tracker.",
        )
        q3, q4, q5 = st.columns(3)
        hallmark_label = q3.selectbox(
            "BIS hallmark",
            hallmark_labels,
            index=hallmark_keys.index(default_hallmark),
            help="International parity is 24K / 999. Indian jewellery is usually 22K / 916.",
        )
        hallmark_fineness = hallmark_keys[hallmark_labels.index(hallmark_label)]
        gst_pct = q4.number_input("GST %", min_value=0.0, value=default_gst, step=0.5)
        making_pct = q5.number_input("Making %", min_value=0.0, value=default_making, step=0.5)
        if st.button("Save quotes"):
            save_user_tracker(
                load_config(),
                {
                    "dealer_price": dealer or None,
                    "avg_cost": avg_cost or None,
                    "gst_pct": gst_pct,
                    "making_pct": making_pct,
                    "hallmark_fineness": hallmark_fineness,
                },
            )
            st.success("Saved to data/user_tracker.json on this machine.")

    layer = apply_quotes(
        base,
        dealer_price=dealer if dealer > 0 else None,
        avg_cost=avg_cost if avg_cost > 0 else None,
        gst_pct=gst_pct,
        making_pct=making_pct,
        hallmark_fineness=hallmark_fineness,
    )
    quotes = layer.get("quotes") or {}
    stance = layer.get("stance") or {}
    parity_v = layer.get("parity_inr_10g") or (snap.parity_inr_10g if snap is not None else None)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        _card(
            "International parity / 10g",
            _inr(parity_v),
            f"24K / 999 · GC=F × USD/INR × (10/31.1034768) · {layer.get('as_of') or _dt(getattr(snap, 'as_of', None))}",
        )
    with r2:
        _card(
            "Estimated physical / 10g",
            _inr(quotes.get("estimated_physical_inr_10g")),
            (
                f"{quotes.get('hallmark_label', '22K / 916')} metal {_inr(quotes.get('hallmarked_metal_inr_10g'))} "
                f"then +{quotes.get('duty_pct') or 0:.1f}% duty +{gst_pct:.1f}% GST +{making_pct:.1f}% making. "
                "Purity overlay only — not a hallmarked invoice or MCX/IBJA."
            ),
        )
    with r3:
        prem = quotes.get("physical_premium_inr")
        prem_pct = quotes.get("physical_premium_pct")
        if prem is None:
            etf_p = layer.get("etf_premium_inr")
            _card(
                "Physical gold premium",
                "Enter a dealer quote",
                "Premium = dealer price − international parity. "
                + (
                    f"GOLDBEES vs parity {_inr(etf_p)} ({_pct(layer.get('etf_premium_pct'))}) is an ETF proxy, not a jeweller."
                    if etf_p is not None
                    else "GOLDBEES ETF gap unavailable."
                ),
            )
        else:
            _card(
                "Physical gold premium",
                f"{_inr(prem)}  ({_pct(prem_pct)})",
                "Dealer price − international parity. Positive means you pay above COMEX×FX.",
            )
    with r4:
        if fc1:
            _card(
                "1-day forecast interval",
                f"{_inr(fc1.get('lower'))}  →  {_inr(fc1.get('upper'))}",
                f"Point {_inr(fc1.get('point'))} · {fc1.get('model_name', 'n/a')} · EWMA × √h",
            )
        else:
            _card("1-day forecast interval", "n/a", "No stored 1d forecast. Run gold forecast --horizon all.")

    i1, i2, i3, i4 = st.columns(4)
    vs = layer.get("vs_ma") or {}
    dd = layer.get("drawdown") or {}
    with i1:
        _card(
            "Vs 7 / 30 / 90 session averages",
            f"{_pct(vs.get('7'))}   {_pct(vs.get('30'))}   {_pct(vs.get('90'))}",
            "Trailing session means. Weekends have no COMEX print.",
            wrap=True,
        )
    with i2:
        _card(
            "Drawdown vs 30d / 90d highs",
            f"{_pct(dd.get('30'))}   /   {_pct(dd.get('90'))}",
            "Last print vs the high of the last 30 and 90 sessions.",
        )
    with i3:
        _card(
            "USD/INR impact",
            f"1d {_pct(layer.get('ret_1d_fx'))} · 21d {_pct(layer.get('ret_21d_fx'))}",
            (
                f"Parity 1d {_pct(layer.get('ret_1d_parity'))} ≈ XAU {_pct(layer.get('ret_1d_xau'))} + FX {_pct(layer.get('ret_1d_fx'))}. "
                f"{layer.get('identity_note')}"
            ),
        )
    with i4:
        vol_txt = f"{100 * vol:.2f}%" if vol else "n/a"
        pctile = layer.get("ewma_vol_percentile")
        down = (layer.get("downside") or {}).get("30d") or {}
        _card(
            "Historical vol & 30d downside",
            f"EWMA {vol_txt}" + (f"  (p{100 * pctile:.0f})" if pctile is not None else ""),
            (
                f"Illustrative −1σ / −2σ 21-session: {_inr(down.get('sigma_1'))} / {_inr(down.get('sigma_2'))}. "
                "EWMA×√h overlay — SCENARIO, NOT A FORECAST."
            ),
        )

    if quotes.get("avg_cost"):
        _card(
            "Average purchase cost tracker",
            f"Book {_inr(quotes.get('avg_cost'))} · vs parity {_pct(quotes.get('vs_avg_parity'))}",
            f"Vs estimated physical {_pct(quotes.get('vs_avg_physical'))}. Positive vs parity means the international print is above your book.",
        )

    st.markdown("**Buy / Wait / Sell**")
    st.caption(stance.get("disclaimer") or "")
    sl, sr = st.columns([1, 2])
    with sl:
        st.markdown(
            f'<div class="gold-card stance-{stance.get("label", "Hold")}"><div class="gold-kicker">Allocation stance</div>'
            f'<div class="gold-stance-label">{stance.get("label", "Hold")}</div>'
            f'<div class="gold-sub">{stance.get("why", "")}</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Buy hits {stance.get('buy_hits', 0)} · Wait hits {stance.get('wait_hits', 0)} · champions {labels}")
    with sr:
        rows = []
        for sig in stance.get("signals") or []:
            fired = sig.get("fires")
            mark = "yes" if fired is True else ("n/a" if fired is None else "no")
            rows.append(
                {
                    "Side": sig.get("side", "").title(),
                    "Check": sig.get("label"),
                    "Fires": mark,
                    "Rule": sig.get("rule"),
                }
            )
        if rows:
            _frame(pd.DataFrame(rows))

    health = data.get("health") or {}
    last = (health.get("last_ingestion_run") or {}) if isinstance(health, dict) else {}
    fc = brief.get("forecasts")
    gen = None
    if isinstance(fc, pd.DataFrame) and not fc.empty and "generated_at" in fc.columns:
        gen = fc["generated_at"].iloc[0]
    st.caption(
        f"Data cutoff (gold session): {layer.get('as_of') or _dt(getattr(snap, 'as_of', None))} · last ingest: {last.get('run_id', 'n/a')} "
        f"({last.get('status', 'n/a')}) · forecast generated: {_dt(gen)}"
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

    parity = data.get("parity")
    if parity is not None and not getattr(parity, "empty", True):
        p = parity.sort_values("obs_date").copy()
        p["obs_date"] = pd.to_datetime(p["obs_date"])
        n = len(p)
        window = st.selectbox(
            "Parity chart window",
            ["Last 90 sessions", "Last 1 year (~252 sessions)", "Last 500 sessions", "Full history (2010–)"],
            index=2,
        )
        if window.startswith("Last 90"):
            p = p.tail(90)
        elif "1 year" in window:
            p = p.tail(252)
        elif "500" in window:
            p = p.tail(500)
        st.caption(
            f"International parity INR/10g · showing {len(p):,} of {n:,} stored sessions "
            f"({layer.get('history_start')} → {layer.get('history_end')}, ~{layer.get('history_years')} years). "
            "Not an official MCX/IBJA print."
        )
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
        st.markdown("**Gold Pulse**")
        st.caption("INR / 10g parity")
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
            st.success("Reloaded from DuckDB")
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
