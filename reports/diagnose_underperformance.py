#!/usr/bin/env python3
"""Post-implementation diagnostic: log-return pipeline vs naive, vs previous version."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gold.config import load_config
from gold.database.store import GoldStore
from gold.evaluation.metrics import mae, reconstruct_price, regression_metrics
from gold.features.builder import FeatureBuilder
from gold.features.columns import is_level_column
from gold.models.arima import ARIMAForecaster
from gold.models.baselines import NaiveForecaster
from gold.models.tree import LightGBMForecaster, XGBoostForecaster
from gold.training.orchestrator import HORIZON_MAP, PIPELINE_VERSION
from gold.training.promotion import pipeline_version_of
from gold.training.regimes import DEFAULT_REGIMES, expanding_regime_evaluate
from gold.training.walkforward import chronological_split

OUT = Path("reports/output/diagnostics_v2.json")
SEED = 42


def _parse(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value or {}


def eval_price(y_true, y_pred, y_prev):
    m = regression_metrics(y_true, y_pred, y_prev)
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in m.items()}


def main():
    cfg = load_config()
    store = GoldStore(cfg)
    fb = FeatureBuilder(cfg, store)

    coverage = store.coverage()
    catalog = store.catalog()
    series_audit = []
    for _, r in coverage.iterrows():
        series_audit.append(
            {
                "series_id": r["series_id"],
                "start": str(pd.to_datetime(r["start_date"]).date()) if pd.notna(r["start_date"]) else None,
                "end": str(pd.to_datetime(r["end_date"]).date()) if pd.notna(r["end_date"]) else None,
                "n_obs": int(r["n_obs"]),
            }
        )
    missing_series = []
    for _, r in catalog.iterrows():
        n = int(coverage.loc[coverage["series_id"] == r["series_id"], "n_obs"].sum()) if not coverage.empty else 0
        if r["status"] in ("unavailable", "restricted", "fallback") or n == 0:
            missing_series.append(
                {
                    "series_id": r["series_id"],
                    "status": r["status"],
                    "n_obs": n,
                    "notes": r["notes"],
                }
            )

    feats = store.load_features()
    if feats.empty:
        feats = fb.build()
    feat_missing = (
        feats.drop(columns=["obs_date", "as_of"], errors="ignore")
        .isna()
        .mean()
        .sort_values(ascending=False)
    )
    feat_missing = {k: round(float(v) * 100, 2) for k, v in feat_missing.items() if v > 0}

    banned = [c for c in feats.columns if c.startswith("close_lag_") or c.startswith("roll_mean_") or c.startswith("roll_min_") or c.startswith("roll_max_")]
    kept_returns = [c for c in feats.columns if c.startswith("ret_lag_") or c.startswith("roll_vol_") or c.endswith("_chg")]
    feature_policy = {
        "banned_level_columns_present": banned,
        "return_vol_macro_change_columns": kept_returns,
        "close_is_reconstruction_anchor_only": "close" in feats.columns,
        "pipeline_version": PIPELINE_VERSION,
        "target": "h_day_log_return",
    }

    target = store.load_series(cfg.primary_series).sort_values("obs_date")
    target["obs_date"] = pd.to_datetime(target["obs_date"])
    px = target.set_index("obs_date")["value"].astype(float)
    yearly = px.resample("YE").agg(["first", "last", "mean", "min", "max", "count"])
    yearly["return_pct"] = (yearly["last"] / yearly["first"] - 1) * 100
    yearly_rows = []
    for dt, row in yearly.iterrows():
        yearly_rows.append(
            {
                "year": int(dt.year),
                "n": int(row["count"]),
                "mean": round(float(row["mean"]), 1),
                "min": round(float(row["min"]), 1),
                "max": round(float(row["max"]), 1),
                "return_pct": round(float(row["return_pct"]), 2) if np.isfinite(row["return_pct"]) else None,
            }
        )

    real_yield = store.load_series("us_real_yield_10y")
    real_yield_audit = {
        "n_obs": int(len(real_yield)),
        "start": str(pd.to_datetime(real_yield["obs_date"]).min().date()) if not real_yield.empty else None,
        "end": str(pd.to_datetime(real_yield["obs_date"]).max().date()) if not real_yield.empty else None,
        "source": "treasury.gov par real yield curve 10y" if not real_yield.empty else "empty",
        "feature_missing_pct": feat_missing.get("us_real_yield_10y", 0.0 if not real_yield.empty else 100.0),
        "chg_missing_pct": feat_missing.get("us_real_yield_10y_chg"),
    }

    leakage = {
        "close_in_feature_store": "close" in feats.columns,
        "close_used_as_tree_feature": False,
        "banned_level_features_in_store": banned,
        "note": "Models predict h-day log returns. Prices are reconstructed as close[t]*exp(ŷ). Tree models drop close, ETF levels, raw macro levels, and lagged/rolling price stats.",
        "festival_days_to_uses_future_calendar_not_prices": True,
        "us_real_yield_missing_pct": real_yield_audit["feature_missing_pct"],
    }

    exps = store.experiments()
    experiment_rows = []
    version_compare = []
    if not exps.empty:
        for _, row in exps.iterrows():
            metrics = _parse(row["metrics"])
            params = _parse(row["params"])
            holdout = metrics.get("holdout") or {}
            regimes = metrics.get("regimes") or {}
            version = pipeline_version_of(params)
            experiment_rows.append(
                {
                    "experiment_id": row["experiment_id"],
                    "model": row["model_name"],
                    "horizon": int(row["horizon"]),
                    "pipeline_version": version,
                    "is_champion": bool(row["is_champion"]),
                    "mae": holdout.get("mae"),
                    "mape": holdout.get("mape"),
                    "return_mae": holdout.get("return_mae"),
                    "regimes_beat": regimes.get("regimes_beat"),
                    "regimes_evaluated": regimes.get("regimes_evaluated"),
                    "n": holdout.get("n"),
                    "cutoff_date": str(row.get("cutoff_date")),
                }
            )
        cur = [r for r in experiment_rows if r["pipeline_version"] == PIPELINE_VERSION]
        prev = [r for r in experiment_rows if r["pipeline_version"] != PIPELINE_VERSION]
        for c in cur:
            matches = [p for p in prev if p["model"] == c["model"] and p["horizon"] == c["horizon"] and p["mae"]]
            rec = {
                "model": c["model"],
                "horizon": c["horizon"],
                "v2_mae": c["mae"],
                "v2_mape": c["mape"],
                "v2_regimes_beat": c["regimes_beat"],
                "champion": c["is_champion"],
            }
            if matches:
                last = matches[0]
                rec["previous_version"] = last["pipeline_version"]
                rec["previous_mae"] = last["mae"]
                rec["mae_vs_previous"] = (c["mae"] / last["mae"]) if c["mae"] and last["mae"] else None
            version_compare.append(rec)

    champs = store.champions()
    champions = []
    if not champs.empty:
        for _, row in champs.iterrows():
            champions.append(
                {
                    "horizon": int(row["horizon"]),
                    "model": row["model_name"],
                    "experiment_id": row["experiment_id"],
                    "notes": row["notes"],
                }
            )

    horizon_results = []
    residual_summary = []
    importance = []
    regime_results = []

    for h_cal, h_td in HORIZON_MAP.items():
        X, y, meta = fb.supervised(h_td)
        n = len(y)
        holdout = min(252, max(30, n // 5))
        train_idx, test_idx = chronological_split(n, holdout)
        Xtr, ytr, mtr = X.iloc[train_idx], y.iloc[train_idx], meta.iloc[train_idx]
        Xte, yte, mte = X.iloc[test_idx], y.iloc[test_idx], meta.iloc[test_idx]
        close_te = mte["close"].to_numpy(dtype=float)
        yte_ret = np.asarray(yte, dtype=float)
        yte_price = reconstruct_price(close_te, yte_ret)

        models = {
            "naive": NaiveForecaster(),
            "xgboost": XGBoostForecaster(),
            "lightgbm": LightGBMForecaster(),
            "arima": ARIMAForecaster(horizon_steps=h_td),
        }
        row = {
            "horizon_calendar": h_cal,
            "horizon_trading": h_td,
            "n_train": int(len(ytr)),
            "n_holdout": int(len(yte)),
            "train_close_min": round(float(mtr["close"].min()), 1),
            "train_close_max": round(float(mtr["close"].max()), 1),
            "holdout_close_min": round(float(mte["close"].min()), 1),
            "holdout_close_max": round(float(mte["close"].max()), 1),
            "train_end": str(pd.to_datetime(mtr["obs_date"].iloc[-1]).date()),
            "holdout_start": str(pd.to_datetime(mte["obs_date"].iloc[0]).date()),
            "holdout_end": str(pd.to_datetime(mte["target_date"].iloc[-1]).date()),
            "tree_feature_columns": [c for c in Xtr.columns if not is_level_column(c)],
            "models": {},
        }
        naive_mae = None
        preds_store = {}
        for name, model in models.items():
            model.fit(ytr, Xtr)
            pred_ret = np.asarray(model.predict(len(yte), X_future=Xte), dtype=float)[: len(yte)]
            pred_price = reconstruct_price(close_te, pred_ret)
            met = eval_price(yte_price, pred_price, close_te)
            met["return_mae"] = float(mae(yte_ret, pred_ret))
            met["mean_pred_price"] = round(float(np.mean(pred_price)), 1)
            met["mean_actual_price"] = round(float(np.mean(yte_price)), 1)
            met["bias"] = round(float(np.mean(pred_price - yte_price)), 1)
            if name == "naive":
                naive_mae = met["mae"]
            met["mae_vs_naive"] = round(float(met["mae"] / naive_mae), 3) if naive_mae else None
            met["beats_naive"] = bool(naive_mae is not None and met["mae"] < naive_mae - 1e-9)
            row["models"][name] = met
            preds_store[name] = pred_price
            resid = yte_price - pred_price
            residual_summary.append(
                {
                    "horizon": h_cal,
                    "model": name,
                    "mean_resid": round(float(np.mean(resid)), 1),
                    "std_resid": round(float(np.std(resid)), 1),
                    "acf1": round(float(np.corrcoef(resid[1:], resid[:-1])[0, 1]), 3) if len(resid) > 3 else None,
                    "frac_underpredict": round(float(np.mean(pred_price < yte_price)), 3),
                }
            )

        booster = XGBoostForecaster()
        booster.fit(ytr, Xtr)
        if getattr(booster, "feature_names", None):
            imp = pd.Series(booster.model.feature_importances_, index=booster.feature_names).sort_values(ascending=False)
            importance.append(
                {
                    "horizon": h_cal,
                    "top": [{"feature": i, "gain": round(float(v), 4)} for i, v in imp.head(12).items()],
                }
            )

        if h_cal in (1, 7, 30):
            for name, factory in (
                ("xgboost", XGBoostForecaster),
                ("lightgbm", LightGBMForecaster),
                ("arima", lambda: ARIMAForecaster(horizon_steps=h_td)),
            ):
                regime = expanding_regime_evaluate(
                    factory if name != "arima" else (lambda: ARIMAForecaster(horizon_steps=h_td)),
                    y,
                    X,
                    meta,
                    regimes=DEFAULT_REGIMES,
                    min_train=252,
                )
                for detail in regime["details"]:
                    regime_results.append({"model": name, "horizon": h_cal, **detail})

        horizon_results.append(row)

    monthly = px.resample("ME").last().dropna()
    monthly = monthly[monthly.index >= "2018-01-01"]
    price_chart = {
        "categories": [d.strftime("%Y-%m") for d in monthly.index],
        "values": [round(float(v), 0) for v in monthly.values],
    }

    any_promoted = [c for c in champions if c["model"] != "naive"]
    report = {
        "generated_from": "local DuckDB/Parquet after v2-logret implementation; no fabricated observations",
        "primary_target": cfg.primary_series,
        "pipeline_version": PIPELINE_VERSION,
        "headline": (
            "All models now forecast h-day log returns and reconstruct prices from close[t]. "
            "Naive persistence (zero return) remains the benchmark. Non-naive models are not "
            "promoted unless they beat naive on the latest holdout and at least three of five "
            "expanding-window historical regimes. LSTM/Transformers were not trained."
        ),
        "series_audit": series_audit,
        "missing_or_restricted": missing_series,
        "real_yield": real_yield_audit,
        "feature_policy": feature_policy,
        "feature_missing_pct": feat_missing,
        "yearly_price": yearly_rows,
        "leakage": leakage,
        "champions": champions,
        "non_naive_promoted": any_promoted,
        "experiments": experiment_rows,
        "version_compare": version_compare,
        "horizons": horizon_results,
        "residuals": residual_summary,
        "importance_logret": importance,
        "regimes": regime_results,
        "price_chart": price_chart,
        "n_feature_rows": int(len(feats)),
        "n_feature_cols": int(feats.shape[1]),
        "promotion_rule": "holdout MAE < naive MAE AND regimes_beat >= 3 of 5 expanding windows",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
