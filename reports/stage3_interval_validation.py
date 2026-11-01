#!/usr/bin/env python3
"""Stage-3 leak-free rolling-origin interval validation. Does not promote models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from gold.config import load_config
from gold.database.store import GoldStore, new_run_id
from gold.evaluation.metrics import directional_accuracy, mae, reconstruct_price
from gold.forecasting.volatility import (
    Z_SCORES,
    completed_h_day_return,
    empirical_h_exponent,
    ewma_variance,
    interval_coverage,
    price_interval,
    winkler_score,
)
from gold.training.orchestrator import HORIZON_MAP

OUT = Path("reports/output/stage3_interval_validation.json")
CHART_DIR = Path("reports/output")
LAMBDAS = (0.90, 0.94, 0.97)
LEVELS = (0.80, 0.90, 0.95)
TEST_LEN = 126
STEP = 63
MIN_TRAIN = 504


def nw_se(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    if n < 8:
        return float("nan")
    var = float(np.dot(x, x) / n)
    lmax = max(1, min(int(lags), n - 2))
    for lag in range(1, lmax + 1):
        w = 1.0 - lag / (lmax + 1)
        var += 2.0 * w * float(np.dot(x[lag:], x[:-lag]) / n)
    return float(np.sqrt(max(var, 0.0) / n))


def dm_ae(err_a, err_b, h) -> dict:
    d = np.abs(err_a) - np.abs(err_b)
    d = d[np.isfinite(d)]
    if len(d) < 20:
        return {"n": int(len(d)), "stat": None, "pvalue": None}
    from scipy.stats import norm

    se = nw_se(d, lags=max(1, int(h)))
    stat = float(np.mean(d) / se) if se and se > 0 else float("nan")
    p = float(2.0 * (1.0 - norm.cdf(abs(stat)))) if np.isfinite(stat) else float("nan")
    return {"n": int(len(d)), "mean_d": float(np.mean(d)), "stat": stat, "pvalue": p}


def score_band(price_true, lo, hi, alpha) -> dict:
    return {
        "coverage": interval_coverage(price_true, lo, hi),
        "mean_width": float(np.mean(hi - lo)),
        "median_width": float(np.median(hi - lo)),
        "winkler": winkler_score(price_true, lo, hi, alpha),
        "n": int(len(price_true)),
    }


def main():
    cfg = load_config()
    store = GoldStore(cfg)
    target = store.load_series(cfg.primary_series).sort_values("obs_date")
    target["obs_date"] = pd.to_datetime(target["obs_date"])
    close = target["value"].astype(float).to_numpy()
    dates = pd.to_datetime(target["obs_date"]).to_numpy()
    daily = np.log(close / np.roll(close, 1))
    daily[0] = np.nan
    daily = np.nan_to_num(daily, nan=0.0)

    ewma_by_lam = {lam: ewma_variance(daily, lam=lam) for lam in LAMBDAS}
    cutoffs = list(range(MIN_TRAIN, len(close) - TEST_LEN - 90, STEP))

    windows = []
    scaling = []
    point_rows = []
    method_rollups = []

    for h_cal, h_td in HORIZON_MAP.items():
        y = np.full(len(close), np.nan)
        y[: len(close) - h_td] = np.log(close[h_td:] / close[: len(close) - h_td])
        completed = completed_h_day_return(close, h_td)
        ewma_h = ewma_variance(np.nan_to_num(completed, nan=0.0), lam=0.94)

        # Scaling diagnostic on expanding prefixes at each cutoff.
        # y[t] uses close[t+h], so train y must stop at origin i - h_td.
        lambdas = []
        for i in cutoffs:
            known = max(i - h_td, 1)
            lam_hat = empirical_h_exponent(daily[:known], h_td, y[:known])
            lambdas.append(lam_hat)
        scaling.append(
            {
                "horizon": h_cal,
                "h_trading": h_td,
                "sqrt_h": float(np.sqrt(h_td)),
                "median_empirical_lambda": float(np.median(lambdas)),
                "mean_empirical_lambda": float(np.mean(lambdas)),
                "p10_lambda": float(np.percentile(lambdas, 10)),
                "p90_lambda": float(np.percentile(lambdas, 90)),
                "sqrt_h_overstates_if_lambda_below_half": bool(np.median(lambdas) < 0.5),
            }
        )

        acc: dict[str, list] = {}
        for i in cutoffs:
            te0, te1 = i, min(i + TEST_LEN, len(close) - h_td)
            if te1 - te0 < 40:
                continue
            known = max(i - h_td, 1)
            y_tr = y[:known]
            y_te = y[te0:te1]
            close_te = close[te0:te1]
            mask = np.isfinite(y_te)
            y_te, close_te = y_te[mask], close_te[mask]
            idx = np.arange(te0, te1)[mask]
            if len(y_te) < 30:
                continue
            price_true = reconstruct_price(close_te, y_te)
            naive_price = close_te
            point_rows.append(
                {
                    "horizon": h_cal,
                    "cutoff": str(pd.Timestamp(dates[i - 1]).date()),
                    "n": int(len(y_te)),
                    "naive_mae": mae(price_true, naive_price),
                    "naive_da": directional_accuracy(price_true, naive_price, close_te),
                }
            )
            const_sigma = float(np.nanstd(y_tr, ddof=1)) if np.isfinite(y_tr).sum() > 5 else 0.02
            emp_lambda = empirical_h_exponent(daily[:known], h_td, y_tr)
            ewma_d = np.sqrt(ewma_by_lam[0.94][idx])
            methods = {
                "const_train": np.full(len(y_te), const_sigma),
                "ewma_sqrt_h": ewma_d * np.sqrt(h_td),
                "ewma_emp_h": ewma_d * (h_td ** emp_lambda),
                "ewma_completed_h": np.sqrt(np.clip(ewma_h[idx], 1e-12, None)),
            }
            for lam in LAMBDAS:
                methods[f"ewma_sqrt_h_lam{lam:.2f}"] = np.sqrt(ewma_by_lam[lam][idx]) * np.sqrt(h_td)

            for level in LEVELS:
                z = Z_SCORES[level]
                alpha = 1.0 - level
                for mname, sigma in methods.items():
                    lo, hi = price_interval(close_te, np.zeros_like(y_te), sigma, z)
                    sc = score_band(price_true, lo, hi, alpha)
                    rec = {
                        "horizon": h_cal,
                        "cutoff": str(pd.Timestamp(dates[i - 1]).date()),
                        "level": level,
                        "method": mname,
                        "emp_lambda": emp_lambda,
                        **sc,
                    }
                    windows.append(rec)
                    acc.setdefault((mname, level), []).append(sc)

        for (mname, level), scores in acc.items():
            cov = np.array([s["coverage"] for s in scores])
            wink = np.array([s["winkler"] for s in scores])
            width = np.array([s["mean_width"] for s in scores])
            method_rollups.append(
                {
                    "horizon": h_cal,
                    "level": level,
                    "method": mname,
                    "n_windows": int(len(scores)),
                    "mean_coverage": float(np.mean(cov)),
                    "coverage_error": float(np.mean(np.abs(cov - level))),
                    "median_winkler": float(np.median(wink)),
                    "mean_winkler": float(np.mean(wink)),
                    "mean_width": float(np.mean(width)),
                    "frac_windows_within_5pp": float(np.mean(np.abs(cov - level) <= 0.05)),
                }
            )

    # Rank the four named methods (lambda sweep is sensitivity only).
    named = {"const_train", "ewma_sqrt_h", "ewma_emp_h", "ewma_completed_h"}
    recommendations = []
    for h_cal in HORIZON_MAP:
        subset80 = [
            r
            for r in method_rollups
            if r["horizon"] == h_cal and r["level"] == 0.8 and r["method"] in named
        ]
        if not subset80:
            continue
        ranked = sorted(subset80, key=lambda r: (r["coverage_error"], r["median_winkler"]))
        best = ranked[0]
        const = next(r for r in subset80 if r["method"] == "const_train")
        ewma = next(r for r in subset80 if r["method"] == "ewma_sqrt_h")
        ewma90 = next(
            (
                r
                for r in method_rollups
                if r["horizon"] == h_cal and r["level"] == 0.9 and r["method"] == "ewma_sqrt_h"
            ),
            None,
        )
        const90 = next(
            (
                r
                for r in method_rollups
                if r["horizon"] == h_cal and r["level"] == 0.9 and r["method"] == "const_train"
            ),
            None,
        )
        better_80 = ewma["coverage_error"] + 0.02 < const["coverage_error"] and ewma[
            "median_winkler"
        ] <= const["median_winkler"] * 1.05
        better_90 = True
        if ewma90 and const90:
            better_90 = ewma90["coverage_error"] + 0.02 < const90["coverage_error"] and ewma90[
                "median_winkler"
            ] <= const90["median_winkler"] * 1.05
        recommendations.append(
            {
                "horizon": h_cal,
                "best_method": best["method"],
                "best_coverage": best["mean_coverage"],
                "best_coverage_error": best["coverage_error"],
                "ewma_sqrt_h_coverage": ewma["mean_coverage"],
                "ewma_sqrt_h_coverage_error": ewma["coverage_error"],
                "const_coverage": const["mean_coverage"],
                "const_coverage_error": const["coverage_error"],
                "winkler_ewma_vs_const": ewma["median_winkler"] / const["median_winkler"]
                if const["median_winkler"]
                else None,
                "justifies_production_change": bool(better_80 and better_90 and h_cal in (1, 7)),
            }
        )

    report = {
        "pipeline_version": "research-stage3-intervals",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "promotion": "none — point-forecast champions unchanged",
        "look_ahead": "EWMA var[t] uses daily returns through t only; constant sigma and empirical λ are train-only; y[t] is the future h-day return used solely as the evaluation target",
        "n_cutoffs": len(cutoffs),
        "test_len": TEST_LEN,
        "step": STEP,
        "scaling": scaling,
        "method_rollups": method_rollups,
        "recommendations": recommendations,
        "n_window_scores": len(windows),
        "naive_point_mae_by_horizon": {
            str(h): float(np.mean([r["naive_mae"] for r in point_rows if r["horizon"] == h]))
            for h in HORIZON_MAP
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))

    # Compact window file for charts (not every lambda×level dump in the summary).
    Path("reports/output/stage3_interval_windows.json").write_text(
        json.dumps(
            [w for w in windows if w["method"] in {"const_train", "ewma_sqrt_h", "ewma_emp_h", "ewma_completed_h"} and w["level"] == 0.8],
            indent=2,
            default=str,
        )
    )

    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", str(CHART_DIR / ".mpl"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for h_cal in HORIZON_MAP:
            fig, ax = plt.subplots(figsize=(9, 4))
            for method, label in (
                ("const_train", "Train residual RMSE"),
                ("ewma_sqrt_h", "Causal EWMA × √h"),
                ("ewma_emp_h", "Causal EWMA × h^λ"),
                ("ewma_completed_h", "EWMA of completed h-day returns"),
            ):
                pts = [w for w in windows if w["horizon"] == h_cal and w["method"] == method and w["level"] == 0.8]
                if not pts:
                    continue
                ax.plot([p["cutoff"] for p in pts], [p["coverage"] for p in pts], label=label, lw=1.2)
            ax.axhline(0.8, color="black", ls="--", lw=0.8, label="80% target")
            ax.set_title(f"{h_cal}d 80% interval coverage by cutoff")
            ax.set_ylabel("Empirical coverage")
            ax.set_xlabel("Train cutoff")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(CHART_DIR / f"stage3_coverage_{h_cal}d.png", dpi=120)
            plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print("chart skip", exc)

    cfg_store = GoldStore(cfg)
    cfg_store.record_training_run(
        {
            "run_id": new_run_id("stage3"),
            "pipeline_version": "research-stage3-intervals",
            "started_at": datetime.now(tz=timezone.utc).replace(tzinfo=None),
            "finished_at": datetime.now(tz=timezone.utc).replace(tzinfo=None),
            "n_experiments": len(windows),
            "comparison_json": recommendations,
            "notes": "interval validation only; no champion promotion",
        }
    )
    cfg_store.close()
    print(f"wrote {OUT}")
    for rec in recommendations:
        print(rec)


if __name__ == "__main__":
    main()
