#!/usr/bin/env python3
"""Stage-2 research: leak-free predictability, regimes, intervals, significance.

Does not promote production champions. LSTM/Transformers are not trained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from gold.config import load_config
from gold.database.store import GoldStore
from gold.evaluation.metrics import mae, reconstruct_price
from gold.features.alignment import asof_join
from gold.features.builder import FeatureBuilder
from gold.training.orchestrator import HORIZON_MAP
from gold.training.regimes import DEFAULT_REGIMES

OUT = Path("reports/output/stage2_research.json")
SEED = 42
Z80 = 1.2815515655446004
REL_GAIN_BAR = 0.02  # 2% MAE reduction vs naive required to count as material

CORE = [
    "usd_inr_chg",
    "dxy_chg",
    "us_real_yield_10y_chg",
    "us_10y_yield_chg",
    "log_return",
    "ret_lag_1",
    "ret_lag_5",
    "ret_lag_21",
    "roll_vol_21",
]
FX = ["usd_inr_chg", "dxy_chg"]
YIELDS = ["us_real_yield_10y_chg", "us_10y_yield_chg"]
MOM = ["log_return", "ret_lag_1", "ret_lag_5", "ret_lag_21"]
VOL = ["roll_vol_5", "roll_vol_21", "roll_vol_63"]
DOMESTIC = [
    "etf_parity_gap_pct",
    "etf_log_return",
    "gold_import_duty_chg",
    "rbi_repo_rate_chg",
    "festival_window",
]
INTERACTIONS = [
    ("log_return", "us_real_yield_10y_chg", "mom_x_real_yield"),
    ("log_return", "roll_vol_21", "mom_x_vol"),
    ("usd_inr_chg", "dxy_chg", "inr_x_dxy"),
    ("us_real_yield_10y_chg", "roll_vol_21", "real_yield_x_vol"),
    ("etf_parity_gap_pct", "festival_window", "gap_x_festival"),
]

CUTOFFS = [
    {"id": "2016", "train_end": "2016-12-30", "test_end": "2017-12-29"},
    {"id": "2018", "train_end": "2018-12-31", "test_end": "2019-12-31"},
    {"id": "2020", "train_end": "2020-12-31", "test_end": "2021-12-31"},
    {"id": "2022", "train_end": "2022-12-30", "test_end": "2023-12-29"},
    {"id": "2024", "train_end": "2024-12-31", "test_end": "2025-12-31"},
    {"id": "holdout_2026", "train_end": "2025-10-28", "test_end": "2026-10-30"},
]


def _jsonable(x):
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not np.isfinite(v) else round(v, 6)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def nw_se(x: np.ndarray, lags: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 8:
        return float("nan")
    gamma0 = float(np.dot(x, x) / n)
    var = gamma0
    lmax = max(1, min(int(lags), n - 2))
    for lag in range(1, lmax + 1):
        w = 1.0 - lag / (lmax + 1)
        gamma = float(np.dot(x[lag:], x[:-lag]) / n)
        var += 2.0 * w * gamma
    return float(np.sqrt(max(var, 0.0) / n))


def diebold_mariano(err_naive: np.ndarray, err_model: np.ndarray, h: int, loss: str = "ae") -> dict:
    e1 = np.asarray(err_naive, dtype=float)
    e2 = np.asarray(err_model, dtype=float)
    d = (np.abs(e1) - np.abs(e2)) if loss == "ae" else (e1**2 - e2**2)
    d = d[np.isfinite(d)]
    if len(d) < 20:
        return {"n": int(len(d)), "mean_d": None, "stat": None, "pvalue": None}
    se = nw_se(d, lags=max(1, int(h)))
    stat = float(np.mean(d) / se) if se and np.isfinite(se) and se > 0 else float("nan")
    p = float(2.0 * (1.0 - norm.cdf(abs(stat)))) if np.isfinite(stat) else float("nan")
    return {
        "n": int(len(d)),
        "mean_d": float(np.mean(d)),
        "stat": _jsonable(stat),
        "pvalue": _jsonable(p),
        "model_better": bool(np.mean(d) > 0),
        "significant_05": bool(np.isfinite(p) and p < 0.05 and np.mean(d) > 0),
    }


def clark_west(err_naive: np.ndarray, err_model: np.ndarray, pred_naive: np.ndarray, pred_model: np.ndarray, h: int) -> dict:
    e1 = np.asarray(err_naive, dtype=float)
    e2 = np.asarray(err_model, dtype=float)
    adj = (np.asarray(pred_model, dtype=float) - np.asarray(pred_naive, dtype=float)) ** 2
    f = e1**2 - e2**2 + adj
    f = f[np.isfinite(f)]
    if len(f) < 20:
        return {"n": int(len(f)), "stat": None, "pvalue": None}
    se = nw_se(f, lags=max(1, int(h)))
    stat = float(np.mean(f) / se) if se and np.isfinite(se) and se > 0 else float("nan")
    p = float(1.0 - norm.cdf(stat)) if np.isfinite(stat) else float("nan")
    return {"n": int(len(f)), "stat": _jsonable(stat), "pvalue": _jsonable(p), "significant_05": bool(np.isfinite(p) and p < 0.05)}


def ridge():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=SEED)),
        ]
    )


def shallow_xgb():
    return XGBRegressor(
        n_estimators=80,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=1,
        random_state=SEED,
        verbosity=0,
    )


def present(cols: list[str], frame: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in frame.columns]


def add_interactions(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for a, b, name in INTERACTIONS:
        if a in out.columns and b in out.columns:
            out[name] = out[a] * out[b]
    return out


def matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = present(cols, frame)
    return frame.loc[:, cols].apply(pd.to_numeric, errors="coerce")


def fit_predict(model, Xtr, ytr, Xte) -> np.ndarray:
    Xtr_c = Xtr.fillna(0.0)
    Xte_c = Xte.reindex(columns=Xtr_c.columns).fillna(0.0)
    model.fit(Xtr_c, np.asarray(ytr, dtype=float))
    return np.asarray(model.predict(Xte_c), dtype=float)


def ols_hac(y: pd.Series, X: pd.DataFrame, lags: int) -> list[dict]:
    cols = list(X.columns)
    Z = X.replace([np.inf, -np.inf], np.nan)
    data = pd.concat([y.rename("y"), Z], axis=1).dropna()
    if len(data) < 40 or data["y"].std() == 0:
        return []
    Y = data["y"]
    XX = sm.add_constant(data[cols], has_constant="add")
    try:
        res = sm.OLS(Y, XX).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, int(lags))})
    except Exception:
        return []
    rows = []
    for name in ["const", *cols]:
        if name not in res.params.index:
            continue
        rows.append(
            {
                "variable": name,
                "coef": _jsonable(res.params[name]),
                "se": _jsonable(res.bse[name]),
                "t": _jsonable(res.tvalues[name]),
                "pvalue": _jsonable(res.pvalues[name]),
                "n": int(res.nobs),
            }
        )
    return rows


def corr_pair(a, b) -> dict:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(m) < 30:
        return {"n": int(len(m)), "pearson": None, "spearman": None}
    pr = float(m["x"].corr(m["y"]))
    sp = float(spearmanr(m["x"], m["y"]).correlation)
    return {"n": int(len(m)), "pearson": _jsonable(pr), "spearman": _jsonable(sp)}


def winkler(y, lo, hi, alpha=0.2) -> float:
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def coverage(y, lo, hi) -> float:
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    return float(np.mean((y >= lo) & (y <= hi)))


def load_aligned_macro(store: GoldStore, idx: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=idx)
    for sid, name in (
        ("gold_usd_oz", "gold_usd"),
        ("usd_inr", "usd_inr"),
        ("dxy", "dxy"),
        ("us_real_yield_10y", "real_yield"),
    ):
        series = store.load_series(sid)
        out[name] = asof_join(idx, series, "value", name)
        out[f"{name}_ret"] = np.log(out[name]).diff() if name in {"gold_usd", "usd_inr", "dxy"} else out[name].diff()
    return out


def build_frame(fb: FeatureBuilder, store: GoldStore, h_td: int) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X, y, meta = fb.supervised(h_td)
    frame = X.copy()
    frame["y"] = np.asarray(y, dtype=float)
    frame["close"] = meta["close"].to_numpy(dtype=float)
    frame["obs_date"] = pd.to_datetime(meta["obs_date"]).to_numpy()
    frame["target_date"] = pd.to_datetime(meta["target_date"]).to_numpy()
    frame["ret_63"] = frame["log_return"].rolling(63, min_periods=21).sum() if "log_return" in frame.columns else np.nan
    if "roll_vol_21" in frame.columns:
        frame["vol_high"] = (frame["roll_vol_21"] >= frame["roll_vol_21"].expanding(252).median()).astype(float)
    else:
        frame["vol_high"] = np.nan
    frame["bull"] = (frame["ret_63"] > 0).astype(float)
    return frame, y, meta


def split_by_dates(frame: pd.DataFrame, train_end: str, test_end: str):
    obs = pd.to_datetime(frame["obs_date"])
    te = pd.Timestamp(train_end)
    xe = pd.Timestamp(test_end)
    train = frame.loc[obs <= te]
    test = frame.loc[(obs > te) & (obs <= xe)]
    return train, test


def eval_model(train, test, cols, h_td, kind: str) -> dict:
    cols = present(cols, train)
    if not cols or len(train) < 80 or len(test) < 20:
        return {"skipped": True, "reason": "insufficient rows or columns"}
    Xtr = matrix(train, cols)
    Xte = matrix(test, cols)
    ytr = train["y"]
    yte = test["y"].to_numpy(dtype=float)
    close = test["close"].to_numpy(dtype=float)
    naive = np.zeros_like(yte)
    if kind == "ridge":
        pred = fit_predict(ridge(), Xtr, ytr, Xte)
    elif kind == "ridge_interact":
        pred = fit_predict(ridge(), add_interactions(Xtr), ytr, add_interactions(Xte))
    elif kind == "xgb_shallow":
        pred = fit_predict(shallow_xgb(), Xtr, ytr, Xte)
    elif kind == "mean":
        pred = np.full_like(yte, float(np.mean(ytr)))
    else:
        raise ValueError(kind)
    price_true = reconstruct_price(close, yte)
    price_naive = reconstruct_price(close, naive)
    price_pred = reconstruct_price(close, pred)
    mae_n = mae(price_true, price_naive)
    mae_m = mae(price_true, price_pred)
    rel = (mae_n - mae_m) / mae_n if mae_n else 0.0
    err_n = price_true - price_naive
    err_m = price_true - price_pred
    dm = diebold_mariano(err_n, err_m, h=h_td, loss="ae")
    cw = clark_west(yte - naive, yte - pred, naive, pred, h=h_td)
    return {
        "skipped": False,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(len(cols)),
        "mae_price": _jsonable(mae_m),
        "naive_mae_price": _jsonable(mae_n),
        "mae_vs_naive": _jsonable(mae_m / mae_n if mae_n else None),
        "rel_gain": _jsonable(rel),
        "material_gain": bool(rel > REL_GAIN_BAR),
        "dm_ae": dm,
        "clark_west": cw,
        "reproducible_flag": bool(rel > REL_GAIN_BAR and dm.get("significant_05")),
    }


def regime_model(train, test, cols, h_td, flag: str) -> dict:
    cols = present(cols, train)
    if flag not in train.columns or len(test) < 20:
        return {"skipped": True}
    pred = np.zeros(len(test), dtype=float)
    used = {"0": 0, "1": 0}
    for value in (0.0, 1.0):
        tr = train[train[flag] == value]
        te_idx = np.flatnonzero(test[flag].to_numpy() == value)
        if len(tr) < 80 or len(te_idx) == 0:
            continue
        local = fit_predict(ridge(), matrix(tr, cols), tr["y"], matrix(test.iloc[te_idx], cols))
        pred[te_idx] = local
        used[str(int(value))] = int(len(te_idx))
    yte = test["y"].to_numpy(dtype=float)
    close = test["close"].to_numpy(dtype=float)
    naive = np.zeros_like(yte)
    mae_n = mae(reconstruct_price(close, yte), reconstruct_price(close, naive))
    mae_m = mae(reconstruct_price(close, yte), reconstruct_price(close, pred))
    rel = (mae_n - mae_m) / mae_n if mae_n else 0.0
    dm = diebold_mariano(
        reconstruct_price(close, yte) - reconstruct_price(close, naive),
        reconstruct_price(close, yte) - reconstruct_price(close, pred),
        h=h_td,
        loss="ae",
    )
    return {
        "skipped": False,
        "flag": flag,
        "n_test_by_regime": used,
        "mae_price": _jsonable(mae_m),
        "naive_mae_price": _jsonable(mae_n),
        "mae_vs_naive": _jsonable(mae_m / mae_n if mae_n else None),
        "rel_gain": _jsonable(rel),
        "dm_ae": dm,
        "reproducible_flag": bool(rel > REL_GAIN_BAR and dm.get("significant_05")),
    }


def causal_ewma_sigma(train: pd.DataFrame, test: pd.DataFrame, h_td: int, lam: float = 0.94) -> np.ndarray:
    """EWMA daily variance using returns known at each origin t, then √h scaling."""
    hist = pd.concat([train, test], ignore_index=True)
    daily = pd.to_numeric(hist["log_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n_train = len(train)
    var = float(np.var(daily[: min(60, max(n_train, 1))])) if n_train else 1e-4
    sigmas = np.empty(len(test), dtype=float)
    for i, r in enumerate(daily):
        var = lam * var + (1.0 - lam) * r * r
        if i >= n_train:
            sigmas[i - n_train] = float(np.sqrt(max(var, 1e-12) * h_td))
    return sigmas


def interval_eval(train, test, h_td) -> dict:
    ytr = train["y"].to_numpy(dtype=float)
    yte = test["y"].to_numpy(dtype=float)
    close = test["close"].to_numpy(dtype=float)
    yhat = np.zeros_like(yte)
    sigma_const = float(np.std(ytr, ddof=1)) if len(ytr) > 2 else 0.01
    sigma_ewma_frozen = None
    daily = train["log_return"].dropna().to_numpy(dtype=float) if "log_return" in train.columns else ytr
    lam = 0.94
    var = float(np.var(daily[: min(60, len(daily))])) if len(daily) else 1e-4
    for r in daily:
        var = lam * var + (1 - lam) * r * r
    sigma_ewma_frozen = float(np.sqrt(max(var, 1e-12) * h_td))
    sigma_ewma_causal = causal_ewma_sigma(train, test, h_td)
    vol_cols = present(VOL, train)
    if vol_cols and len(train) > 80:
        har = ridge()
        scale = fit_predict(har, matrix(train, vol_cols), np.abs(train["y"]), matrix(test, vol_cols))
        scale = np.clip(scale, 1e-6, None)
    else:
        scale = np.full_like(yte, sigma_const)
    methods = {
        "constant_residual": np.full_like(yte, sigma_const),
        "ewma_frozen_train_end": np.full_like(yte, sigma_ewma_frozen),
        "ewma_causal_sqrt_h": sigma_ewma_causal,
        "har_abs_return": scale,
    }
    out = {}
    price_true = reconstruct_price(close, yte)
    for name, sig in methods.items():
        lo_r = yhat - Z80 * sig
        hi_r = yhat + Z80 * sig
        lo_p = reconstruct_price(close, lo_r)
        hi_p = reconstruct_price(close, hi_r)
        out[name] = {
            "coverage_80": _jsonable(coverage(price_true, lo_p, hi_p)),
            "mean_width_inr": _jsonable(float(np.mean(hi_p - lo_p))),
            "winkler_80": _jsonable(winkler(price_true, lo_p, hi_p, alpha=0.2)),
            "mean_sigma_return": _jsonable(float(np.mean(sig))),
        }
    return out


def contemporaneous_and_predictive(frame: pd.DataFrame, aligned: pd.DataFrame, h_cal: int, h_td: int) -> dict:
    idx = pd.to_datetime(frame["obs_date"])
    aligned = aligned.copy()
    if len(aligned) != len(frame):
        aligned = aligned.reindex(pd.DatetimeIndex(idx))
    aligned = aligned.reset_index(drop=True)
    y = frame["y"]
    predictors = {
        "usd_inr_chg": frame.get("usd_inr_chg"),
        "dxy_chg": frame.get("dxy_chg"),
        "us_real_yield_10y_chg": frame.get("us_real_yield_10y_chg"),
        "gold_momentum_1d": frame.get("log_return"),
        "gold_momentum_21d": frame.get("ret_lag_21"),
        "roll_vol_21": frame.get("roll_vol_21"),
        "etf_parity_gap_pct": frame.get("etf_parity_gap_pct"),
    }
    predictive = {k: corr_pair(v, y) for k, v in predictors.items() if v is not None}
    same_day = {}
    if "gold_usd_ret" in aligned.columns:
        same_day["gold_usd_ret_vs_dxy_ret"] = corr_pair(aligned["gold_usd_ret"], aligned["dxy_ret"])
        same_day["gold_usd_ret_vs_real_yield_chg"] = corr_pair(aligned["gold_usd_ret"], aligned["real_yield_ret"])
        same_day["gold_inr_ret_vs_usd_inr_ret"] = corr_pair(frame.get("log_return"), aligned["usd_inr_ret"])
        same_day["gold_usd_ret_vs_usd_inr_ret"] = corr_pair(aligned["gold_usd_ret"], aligned["usd_inr_ret"])
    era_rows = []
    for spec in DEFAULT_REGIMES:
        mask = (idx >= spec["test_start"]) & (idx <= spec["test_end"])
        if int(mask.sum()) < 40:
            continue
        sub = frame.loc[mask.to_numpy() if getattr(mask, "to_numpy", None) else mask]
        era_rows.append(
            {
                "regime": spec["id"],
                "n": int(mask.sum()),
                "corr_usd_inr": corr_pair(sub.get("usd_inr_chg"), sub["y"]),
                "corr_dxy": corr_pair(sub.get("dxy_chg"), sub["y"]),
                "corr_real_yield": corr_pair(sub.get("us_real_yield_10y_chg"), sub["y"]),
                "corr_momentum": corr_pair(sub.get("log_return"), sub["y"]),
            }
        )
    ols = {
        "core": ols_hac(y, matrix(frame, CORE), lags=h_td),
        "fx": ols_hac(y, matrix(frame, FX), lags=h_td),
        "yields": ols_hac(y, matrix(frame, YIELDS), lags=h_td),
        "momentum": ols_hac(y, matrix(frame, MOM), lags=h_td),
        "domestic": ols_hac(y, matrix(frame, present(DOMESTIC, frame)), lags=h_td),
        "interactions": ols_hac(y, add_interactions(matrix(frame, CORE + present(DOMESTIC, frame))), lags=h_td),
    }
    cond = []
    for flag, label in (("vol_high", "volatility"), ("bull", "trend")):
        if flag not in frame.columns:
            continue
        for value, name in ((0.0, "low/bear"), (1.0, "high/bull")):
            sub = frame[frame[flag] == value]
            if len(sub) < 80:
                continue
            cond.append(
                {
                    "split": label,
                    "bucket": name,
                    "n": int(len(sub)),
                    "ols_core": ols_hac(sub["y"], matrix(sub, CORE), lags=h_td),
                    "corr_real_yield": corr_pair(sub.get("us_real_yield_10y_chg"), sub["y"]),
                    "corr_usd_inr": corr_pair(sub.get("usd_inr_chg"), sub["y"]),
                    "corr_momentum": corr_pair(sub.get("log_return"), sub["y"]),
                }
            )
    return {
        "horizon_calendar": h_cal,
        "predictive_corr": predictive,
        "contemporaneous_corr_not_for_forecasting": same_day,
        "era_predictive_corr": era_rows,
        "ols_hac": ols,
        "conditional_ols": cond,
    }


def premium_study(frame: pd.DataFrame, h_td: int) -> dict:
    if "etf_parity_gap_pct" not in frame.columns:
        return {"available": False, "reason": "etf_parity_gap_pct missing"}
    gap = pd.to_numeric(frame["etf_parity_gap_pct"], errors="coerce")
    future_gap = gap.shift(-h_td)
    gap_change = future_gap - gap
    return {
        "available": True,
        "note": "GOLDBEES-implied 10g vs international parity. Not IBJA and not MCX. Corporate actions can distort the gap.",
        "gap_summary": {
            "n": int(gap.notna().sum()),
            "median": _jsonable(float(gap.median())) if gap.notna().any() else None,
            "p10": _jsonable(float(gap.quantile(0.1))) if gap.notna().any() else None,
            "p90": _jsonable(float(gap.quantile(0.9))) if gap.notna().any() else None,
        },
        "predict_parity_return_from_gap": corr_pair(gap, frame["y"]),
        "gap_mean_reversion": corr_pair(gap, gap_change),
        "ols_gap_to_return": ols_hac(frame["y"], matrix(frame, present(["etf_parity_gap_pct", "etf_log_return", "gold_import_duty_chg", "festival_window", "rbi_repo_rate_chg"], frame)), lags=h_td),
        "ols_gap_change": ols_hac(gap_change, matrix(frame, present(["etf_parity_gap_pct", "gold_import_duty_chg", "festival_window", "rbi_repo_rate_chg"], frame)), lags=h_td),
        "restricted_sources": ["mcx_gold_futures", "ibja_gold"],
    }


def main():
    cfg = load_config()
    store = GoldStore(cfg)
    fb = FeatureBuilder(cfg, store)
    feats = store.load_features()
    if feats.empty:
        feats = fb.build()
    idx = pd.to_datetime(feats["obs_date"])
    aligned = load_aligned_macro(store, pd.DatetimeIndex(idx))
    aligned.index = pd.DatetimeIndex(idx)

    relationships = []
    model_grid = []
    regime_grid = []
    intervals = []
    sensitivity = []
    premiums = []
    significance_summary = []

    feature_sets = {
        "core": CORE,
        "fx": FX,
        "yields": YIELDS,
        "momentum": MOM,
        "core_plus_domestic": CORE + DOMESTIC,
        "vol": VOL,
    }

    for h_cal, h_td in HORIZON_MAP.items():
        frame, _, _ = build_frame(fb, store, h_td)
        aligned_h = aligned.reindex(pd.DatetimeIndex(pd.to_datetime(frame["obs_date"])))
        relationships.append(contemporaneous_and_predictive(frame, aligned_h, h_cal, h_td))
        premiums.append({"horizon": h_cal, **premium_study(frame, h_td)})

        latest_train, latest_test = split_by_dates(frame, "2025-10-28", "2026-10-30")
        for set_name, cols in feature_sets.items():
            for kind in ("ridge", "ridge_interact"):
                if kind == "ridge_interact" and set_name not in {"core", "core_plus_domestic"}:
                    continue
                rec = eval_model(latest_train, latest_test, cols, h_td, kind)
                rec.update({"horizon": h_cal, "feature_set": set_name, "model": kind, "window": "holdout_2026"})
                model_grid.append(rec)
        xgb_rec = eval_model(latest_train, latest_test, CORE + present(DOMESTIC, frame), h_td, "xgb_shallow")
        xgb_rec.update({"horizon": h_cal, "feature_set": "core_plus_domestic", "model": "xgb_shallow", "window": "holdout_2026"})
        model_grid.append(xgb_rec)
        mean_rec = eval_model(latest_train, latest_test, CORE, h_td, "mean")
        mean_rec.update({"horizon": h_cal, "feature_set": "mean_only", "model": "mean", "window": "holdout_2026"})
        model_grid.append(mean_rec)

        for flag in ("vol_high", "bull"):
            rec = regime_model(latest_train, latest_test, CORE, h_td, flag)
            rec.update({"horizon": h_cal, "window": "holdout_2026"})
            regime_grid.append(rec)

        if len(latest_train) > 80 and len(latest_test) > 20:
            iv = interval_eval(latest_train, latest_test, h_td)
            intervals.append({"horizon": h_cal, "n_test": int(len(latest_test)), "methods": iv})

        for spec in CUTOFFS:
            tr, te = split_by_dates(frame, spec["train_end"], spec["test_end"])
            rec = eval_model(tr, te, CORE, h_td, "ridge")
            rec.update({"horizon": h_cal, "cutoff": spec["id"], "model": "ridge_core"})
            sensitivity.append(rec)
            rec_x = eval_model(tr, te, CORE, h_td, "xgb_shallow")
            rec_x.update({"horizon": h_cal, "cutoff": spec["id"], "model": "xgb_shallow_core"})
            sensitivity.append(rec_x)
            rec_d = eval_model(tr, te, CORE + DOMESTIC, h_td, "ridge")
            rec_d.update({"horizon": h_cal, "cutoff": spec["id"], "model": "ridge_core_domestic"})
            sensitivity.append(rec_d)

    # Reproducible-gain rollup: ridge_core across cutoffs.
    for h_cal in HORIZON_MAP:
        for model in ("ridge_core", "xgb_shallow_core", "ridge_core_domestic"):
            rows = [r for r in sensitivity if r.get("horizon") == h_cal and r.get("model") == model and not r.get("skipped")]
            wins = [r for r in rows if r.get("reproducible_flag")]
            material = [r for r in rows if r.get("material_gain")]
            significance_summary.append(
                {
                    "horizon": h_cal,
                    "model": model,
                    "n_windows": len(rows),
                    "windows_material_gain": len(material),
                    "windows_dm_sig_and_material": len(wins),
                    "meets_reproducible_bar": bool(len(wins) >= 3),
                    "median_mae_vs_naive": _jsonable(float(np.median([r["mae_vs_naive"] for r in rows if r.get("mae_vs_naive")]))) if rows else None,
                }
            )

    report = {
        "pipeline_version": "research-stage2",
        "generated_from": "local DuckDB/Parquet; leak-free as-of features; no fabricated MCX/IBJA",
        "promotion": "none — research only; production champions unchanged",
        "lstm_transformers": "not trained; no residual structure found that requires sequence models",
        "bars": {
            "material_relative_mae_gain": REL_GAIN_BAR,
            "dm_alpha": 0.05,
            "reproducible": "material MAE gain AND Diebold-Mariano p<0.05 on at least 3 of 6 cutoff windows",
        },
        "relationships": relationships,
        "holdout_models": model_grid,
        "regime_models": regime_grid,
        "intervals": intervals,
        "sensitivity": sensitivity,
        "premiums": premiums,
        "significance_summary": significance_summary,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
