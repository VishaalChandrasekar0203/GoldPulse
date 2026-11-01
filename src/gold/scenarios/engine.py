"""Conditional INR/10g scenario calculator.

These are identity and sample-elasticity what-ifs, not statistically validated
forecasts. International parity is never relabeled as MCX or IBJA.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm

from gold.config import AppConfig, TROY_OUNCE_GRAMS
from gold.database.store import GoldStore
from gold.features.alignment import asof_join
from gold.logging import get_logger

log = get_logger("scenarios")

SCENARIO_DISCLAIMER = (
    "SCENARIO, NOT A FORECAST. Identity shocks (XAU/USD, USD/INR) follow "
    "gold_inr_10g = gold_usd_oz × usd_inr × (10/31.1034768). DXY and yield "
    "rows apply sample elasticities estimated on contemporaneous daily returns "
    "and are not Diebold–Mariano-validated predictions. Duty and premium rows "
    "do not change international parity; they illustrate an onshore add-on. "
    "MCX/IBJA official prints are not used."
)


@dataclass
class MarketSnapshot:
    as_of: pd.Timestamp
    gold_usd_oz: float
    usd_inr: float
    parity_inr_10g: float
    dxy: float | None
    us_10y_yield: float | None
    us_real_yield_10y: float | None
    gold_import_duty: float | None
    etf_parity_gap_pct: float | None
    etf_inr_10g: float | None


def parity_inr_10g(gold_usd_oz: float, usd_inr: float, troy: float = TROY_OUNCE_GRAMS) -> float:
    return float(gold_usd_oz) * float(usd_inr) * (10.0 / float(troy))


def _last(store: GoldStore, series_id: str) -> tuple[pd.Timestamp | None, float | None]:
    df = store.load_series(series_id)
    if df.empty:
        return None, None
    df = df.sort_values("obs_date")
    row = df.iloc[-1]
    return pd.to_datetime(row["obs_date"]), float(row["value"])


def _asof(store: GoldStore, series_id: str, as_of: pd.Timestamp) -> float | None:
    df = store.load_series(series_id)
    if df.empty:
        return None
    df = df.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df[df["obs_date"] <= pd.to_datetime(as_of)]
    if df.empty:
        return None
    return float(df.sort_values("obs_date").iloc[-1]["value"])


def snapshot(store: GoldStore, cfg: AppConfig) -> MarketSnapshot:
    troy = cfg.troy_ounce_grams or TROY_OUNCE_GRAMS
    as_of, xau = _last(store, "gold_usd_oz")
    if xau is None or as_of is None:
        raise RuntimeError("Need gold_usd_oz to build a scenario snapshot")
    fx = _asof(store, "usd_inr", as_of)
    if fx is None:
        raise RuntimeError("Need usd_inr on or before the last gold session")
    dxy = _asof(store, "dxy", as_of)
    y10 = _asof(store, "us_10y_yield", as_of)
    real = _asof(store, "us_real_yield_10y", as_of)
    duty = _asof(store, "gold_import_duty", as_of)
    etf = _asof(store, cfg.secondary_series, as_of)
    parity = parity_inr_10g(xau, fx, troy)
    gap = None if etf is None else (etf / parity - 1.0)
    return MarketSnapshot(
        as_of=as_of or pd.Timestamp.utcnow(),
        gold_usd_oz=xau,
        usd_inr=fx,
        parity_inr_10g=parity,
        dxy=dxy,
        us_10y_yield=y10,
        us_real_yield_10y=real,
        gold_import_duty=duty,
        etf_parity_gap_pct=gap,
        etf_inr_10g=etf,
    )


def _beta(y: pd.Series, x: pd.Series) -> dict:
    data = pd.DataFrame({"y": y, "x": x}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 80 or float(data["x"].std()) < 1e-12:
        return {"beta": None, "n": int(len(data)), "pvalue": None, "r2": None}
    res = sm.OLS(data["y"], sm.add_constant(data["x"], has_constant="add")).fit()
    return {
        "beta": float(res.params["x"]),
        "n": int(res.nobs),
        "pvalue": float(res.pvalues["x"]),
        "r2": float(res.rsquared),
    }


def sample_elasticities(store: GoldStore) -> dict:
    """Contemporaneous daily OLS elasticities. Descriptive, not a forecast model."""
    gold = store.load_series("gold_usd_oz").sort_values("obs_date")
    if gold.empty:
        return {}
    idx = pd.DatetimeIndex(pd.to_datetime(gold["obs_date"]))
    gold_ret = pd.Series(np.log(gold.set_index(idx)["value"]).diff(), index=idx)
    out = {}
    for sid, name, use_log in (
        ("dxy", "gold_usd_vs_dxy", True),
        ("usd_inr", "gold_usd_vs_usd_inr", True),
        ("us_10y_yield", "gold_usd_vs_us_10y_yield", False),
        ("us_real_yield_10y", "gold_usd_vs_real_yield", False),
    ):
        series = store.load_series(sid)
        level = asof_join(idx, series, "value", sid)
        x = np.log(level).diff() if use_log else level.diff()
        out[name] = _beta(gold_ret, x)
    inr = store.load_series("gold_inr_10g")
    if not inr.empty:
        inr = inr.sort_values("obs_date")
        iidx = pd.DatetimeIndex(pd.to_datetime(inr["obs_date"]))
        inr_ret = pd.Series(np.log(inr.set_index(iidx)["value"]).diff(), index=iidx)
        fx = asof_join(iidx, store.load_series("usd_inr"), "value", "usd_inr")
        out["gold_inr_vs_usd_inr"] = _beta(inr_ret, np.log(fx).diff())
    return out


def run_scenarios(
    snap: MarketSnapshot,
    cfg: AppConfig,
    elasticities: dict,
    xau_pct: float = 0.0,
    usdinr_pct: float = 0.0,
    dxy_pct: float = 0.0,
    yield_bp: float = 0.0,
    real_yield_bp: float = 0.0,
    duty_pp: float = 0.0,
    premium_pp: float = 0.0,
) -> pd.DataFrame:
    troy = cfg.troy_ounce_grams or TROY_OUNCE_GRAMS
    gst = float((cfg.get("scenario", {}) or {}).get("gst_rate", 0.03))
    p0 = snap.parity_inr_10g
    duty0 = snap.gold_import_duty if snap.gold_import_duty is not None else 0.0
    gap0 = snap.etf_parity_gap_pct if snap.etf_parity_gap_pct is not None else 0.0
    rows = []

    def add(name, kind, new_parity, note, extra=None):
        rows.append(
            {
                "scenario": name,
                "kind": kind,
                "base_parity_inr_10g": p0,
                "scenario_parity_inr_10g": new_parity,
                "delta_inr_10g": new_parity - p0,
                "delta_pct": (new_parity / p0 - 1.0) * 100 if p0 else None,
                "notes": note,
                **(extra or {}),
            }
        )

    xau1 = snap.gold_usd_oz * (1.0 + xau_pct / 100.0)
    fx1 = snap.usd_inr * (1.0 + usdinr_pct / 100.0)
    add(
        f"XAU/USD {xau_pct:+.2f}%",
        "identity",
        parity_inr_10g(xau1, snap.usd_inr, troy),
        "Holds USDINR fixed. Identity on international parity.",
    )
    add(
        f"USD/INR {usdinr_pct:+.2f}%",
        "identity",
        parity_inr_10g(snap.gold_usd_oz, fx1, troy),
        "Holds XAU/USD fixed. Identity on international parity.",
    )
    add(
        f"XAU {xau_pct:+.2f}% and USDINR {usdinr_pct:+.2f}% combined",
        "identity",
        parity_inr_10g(xau1, fx1, troy),
        "Log-additive identity: dP/P ≈ dXAU/XAU + dUSDINR/USDINR.",
    )

    b_dxy = (elasticities.get("gold_usd_vs_dxy") or {}).get("beta")
    if b_dxy is not None:
        xau_from_dxy = snap.gold_usd_oz * (1.0 + b_dxy * (dxy_pct / 100.0))
        add(
            f"DXY {dxy_pct:+.2f}% (elasticity)",
            "elasticity",
            parity_inr_10g(xau_from_dxy, snap.usd_inr, troy),
            f"Applies sample beta={b_dxy:.2f} of USD gold vs DXY. Not a validated forecast. "
            "Do not add to an XAU identity shock (double-counts gold USD).",
        )
    b_y = (elasticities.get("gold_usd_vs_us_10y_yield") or {}).get("beta")
    if b_y is not None:
        xau_from_y = snap.gold_usd_oz * (1.0 + b_y * (yield_bp / 100.0))
        add(
            f"US 10y {yield_bp:+.0f} bp (elasticity)",
            "elasticity",
            parity_inr_10g(xau_from_y, snap.usd_inr, troy),
            f"Applies sample beta={b_y:.3f} of USD gold vs 10y yield (decimal percent). "
            "Contemporaneous OLS, not a forecast.",
        )
    b_r = (elasticities.get("gold_usd_vs_real_yield") or {}).get("beta")
    if b_r is not None:
        xau_from_r = snap.gold_usd_oz * (1.0 + b_r * (real_yield_bp / 100.0))
        add(
            f"US 10y real yield {real_yield_bp:+.0f} bp (elasticity)",
            "elasticity",
            parity_inr_10g(xau_from_r, snap.usd_inr, troy),
            f"Applies sample beta={b_r:.3f} of USD gold vs TIPS 10y. Contemporaneous OLS, not a forecast.",
        )

    onshore0 = p0 * (1.0 + duty0 / 100.0) * (1.0 + gst)
    onshore1 = p0 * (1.0 + (duty0 + duty_pp) / 100.0) * (1.0 + gst)
    rows.append(
        {
            "scenario": f"Import duty {duty_pp:+.2f} pp (onshore illustration)",
            "kind": "onshore_addon",
            "base_parity_inr_10g": p0,
            "scenario_parity_inr_10g": p0,
            "delta_inr_10g": onshore1 - onshore0,
            "delta_pct": (onshore1 / onshore0 - 1.0) * 100 if onshore0 else None,
            "notes": (
                f"Parity unchanged at {p0:.2f}. Illustrative onshore = parity×(1+duty)×(1+GST={gst:.0%}). "
                f"Duty {duty0:.2f}% → {duty0 + duty_pp:.2f}%. Not MCX/IBJA. GST is a config assumption."
            ),
            "onshore_base": onshore0,
            "onshore_scenario": onshore1,
        }
    )
    etf0 = p0 * (1.0 + gap0)
    etf1 = p0 * (1.0 + gap0 + premium_pp / 100.0)
    rows.append(
        {
            "scenario": f"Domestic ETF-parity gap {premium_pp:+.2f} pp",
            "kind": "onshore_addon",
            "base_parity_inr_10g": p0,
            "scenario_parity_inr_10g": p0,
            "delta_inr_10g": etf1 - etf0,
            "delta_pct": (etf1 / etf0 - 1.0) * 100 if etf0 else None,
            "notes": (
                f"Parity unchanged. GOLDBEES-implied 10g vs parity gap {gap0 * 100:.2f}% → "
                f"{(gap0 + premium_pp / 100.0) * 100:.2f}%. Not IBJA local premium."
            ),
            "etf_base": etf0,
            "etf_scenario": etf1,
        }
    )
    out = pd.DataFrame(rows)
    out["generated_at"] = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    out["as_of"] = snap.as_of
    out["disclaimer"] = SCENARIO_DISCLAIMER
    return out


NAMED_PACK_DEFAULTS = {
    "bull": {
        "label": "Bull (identity)",
        "xau_pct": 5.0,
        "usdinr_pct": 1.0,
        "dxy_pct": -2.0,
        "yield_bp": -25.0,
        "notes": "Gold up, INR weaker. Combined parity uses XAU+USDINR identity only.",
    },
    "bear": {
        "label": "Bear (identity)",
        "xau_pct": -5.0,
        "usdinr_pct": -1.0,
        "dxy_pct": 2.0,
        "yield_bp": 25.0,
        "notes": "Gold down, INR stronger. Combined parity uses identity only.",
    },
    "stress": {
        "label": "Stress (identity)",
        "xau_pct": -8.0,
        "usdinr_pct": 2.0,
        "dxy_pct": 3.0,
        "yield_bp": 50.0,
        "notes": "Risk-off gold drop with INR depreciation. Combined parity uses identity only.",
    },
}


def _pack_specs(cfg: AppConfig) -> dict:
    configured = (cfg.get("scenario", {}) or {}).get("packs") or {}
    out = {}
    for name, default in NAMED_PACK_DEFAULTS.items():
        merged = dict(default)
        merged.update(configured.get(name) or {})
        out[name] = merged
    return out


def run_named_packs(snap: MarketSnapshot, cfg: AppConfig, elasticities: dict) -> pd.DataFrame:
    """Bull/bear/stress packs. Combined INR impact is identity only — never labelled a forecast."""
    troy = cfg.troy_ounce_grams or TROY_OUNCE_GRAMS
    p0 = snap.parity_inr_10g
    rows = []
    for name, spec in _pack_specs(cfg).items():
        xau_pct = float(spec.get("xau_pct", 0.0))
        fx_pct = float(spec.get("usdinr_pct", 0.0))
        dxy_pct = float(spec.get("dxy_pct", 0.0))
        yield_bp = float(spec.get("yield_bp", 0.0))
        xau1 = snap.gold_usd_oz * (1.0 + xau_pct / 100.0)
        fx1 = snap.usd_inr * (1.0 + fx_pct / 100.0)
        p_xau = parity_inr_10g(xau1, snap.usd_inr, troy)
        p_fx = parity_inr_10g(snap.gold_usd_oz, fx1, troy)
        p_both = parity_inr_10g(xau1, fx1, troy)
        label = spec.get("label") or name
        method = (
            "Identity P = XAU × USDINR × (10/31.1034768). "
            "DXY/yield rows are contemporaneous elasticities and are NOT added to the combined identity. "
            "Not a forecast."
        )
        def row(scenario, kind, new_p, note):
            return {
                "pack": name,
                "pack_label": label,
                "scenario": scenario,
                "kind": kind,
                "base_parity_inr_10g": p0,
                "scenario_parity_inr_10g": new_p,
                "delta_inr_10g": new_p - p0,
                "delta_pct": (new_p / p0 - 1.0) * 100 if p0 else None,
                "notes": note,
                "methodology": method,
                "baseline_as_of": snap.as_of,
                "assumptions": spec.get("notes") or "",
            }

        rows.append(row(f"{label}: XAU {xau_pct:+.1f}%", "identity_component", p_xau, "USDINR held."))
        rows.append(row(f"{label}: USDINR {fx_pct:+.1f}%", "identity_component", p_fx, "XAU held."))
        rows.append(
            row(
                f"{label}: combined identity",
                "identity_combined",
                p_both,
                "SCENARIO, NOT A FORECAST. Sum of log identity shocks.",
            )
        )
        b_dxy = (elasticities.get("gold_usd_vs_dxy") or {}).get("beta")
        if b_dxy is not None:
            p_dxy = parity_inr_10g(snap.gold_usd_oz * (1.0 + b_dxy * (dxy_pct / 100.0)), snap.usd_inr, troy)
            rows.append(
                row(
                    f"{label}: DXY {dxy_pct:+.1f}% elasticity overlay",
                    "elasticity_overlay",
                    p_dxy,
                    f"Do not add to combined identity (double-counts USD gold). beta={b_dxy:.2f}.",
                )
            )
        b_y = (elasticities.get("gold_usd_vs_us_10y_yield") or {}).get("beta")
        if b_y is not None:
            p_y = parity_inr_10g(snap.gold_usd_oz * (1.0 + b_y * (yield_bp / 100.0)), snap.usd_inr, troy)
            rows.append(
                row(
                    f"{label}: 10y {yield_bp:+.0f} bp elasticity overlay",
                    "elasticity_overlay",
                    p_y,
                    f"Do not add to combined identity. beta={b_y:.3f}. Contemporaneous OLS.",
                )
            )
    out = pd.DataFrame(rows)
    out["generated_at"] = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    out["disclaimer"] = SCENARIO_DISCLAIMER
    return out


def build(cfg: AppConfig, store: GoldStore, **shocks) -> tuple[MarketSnapshot, dict, pd.DataFrame]:
    snap = snapshot(store, cfg)
    elas = sample_elasticities(store)
    table = run_scenarios(snap, cfg, elas, **shocks)
    log.info("scenario table %s rows as-of %s", len(table), snap.as_of)
    return snap, elas, table


def build_packs(cfg: AppConfig, store: GoldStore) -> tuple[MarketSnapshot, dict, pd.DataFrame]:
    snap = snapshot(store, cfg)
    elas = sample_elasticities(store)
    table = run_named_packs(snap, cfg, elas)
    log.info("named scenario packs %s rows as-of %s", len(table), snap.as_of)
    return snap, elas, table
