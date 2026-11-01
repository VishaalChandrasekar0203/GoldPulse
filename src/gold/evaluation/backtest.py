from __future__ import annotations

import json

import pandas as pd

from gold.config import AppConfig
from gold.database.store import GoldStore
from gold.logging import get_logger
from gold.training.promotion import pipeline_version_of

log = get_logger("backtest")


def _parse(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value or {}


class Backtester:
    def __init__(self, cfg: AppConfig, store: GoldStore):
        self.cfg = cfg
        self.store = store

    def run(self) -> pd.DataFrame:
        exps = self.store.experiments()
        if exps.empty:
            raise RuntimeError("No experiments recorded. Run `gold train --all` first.")
        rows = []
        for _, row in exps.iterrows():
            metrics = _parse(row["metrics"])
            params = _parse(row["params"])
            holdout = metrics.get("holdout") or {}
            regimes = metrics.get("regimes") or {}
            h = int(row["horizon"])
            run_id = params.get("run_id") or metrics.get("run_id")
            version = pipeline_version_of(params)
            rows.append(
                {
                    "experiment_id": row["experiment_id"],
                    "model": row["model_name"],
                    "horizon": h,
                    "pipeline_version": version,
                    "run_id": run_id,
                    "cutoff_date": row["cutoff_date"],
                    "mae": holdout.get("mae"),
                    "rmse": holdout.get("rmse"),
                    "mape": holdout.get("mape"),
                    "smape": holdout.get("smape"),
                    "directional_accuracy": holdout.get("directional_accuracy"),
                    "return_mae": holdout.get("return_mae"),
                    "regimes_beat": regimes.get("regimes_beat"),
                    "regimes_evaluated": regimes.get("regimes_evaluated"),
                    "n": holdout.get("n"),
                    "is_champion": bool(row["is_champion"]),
                    "notes": row["notes"],
                }
            )
        out = pd.DataFrame(rows)
        if not out.empty:
            naive = out[out["model"] == "naive"].copy()
            naive_by_key = {}
            for _, nrow in naive.iterrows():
                key = (nrow["horizon"], nrow["run_id"] or nrow["pipeline_version"])
                naive_by_key[key] = nrow["mae"]
            latest_naive = (
                naive.sort_values("experiment_id")
                .groupby("horizon", as_index=False)
                .tail(1)
                .set_index("horizon")["mae"]
                .to_dict()
            )

            def vs_naive(r):
                key = (r["horizon"], r["run_id"] or r["pipeline_version"])
                base = naive_by_key.get(key, latest_naive.get(r["horizon"]))
                if base and r["mae"]:
                    return r["mae"] / base
                return None

            out["mae_vs_naive"] = out.apply(vs_naive, axis=1)
        log.info("backtest summary: %s experiment rows", len(out))
        return out.sort_values(["horizon", "pipeline_version", "mae"])
