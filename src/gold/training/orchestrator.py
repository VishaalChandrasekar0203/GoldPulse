from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gold.config import AppConfig
from gold.database.store import GoldStore, new_run_id
from gold.evaluation.metrics import score_log_return_forecast
from gold.features.builder import FeatureBuilder
from gold.logging import get_logger
from gold.models.arima import ARIMAForecaster, SARIMAForecaster
from gold.models.ensemble import EnsembleForecaster
from gold.models.registry import MODEL_FACTORIES, SKIP_UNTIL_BASELINES_BEAT_NAIVE, available_models
from gold.training.promotion import beats_promotion_bar, pipeline_version_of
from gold.training.regimes import DEFAULT_REGIMES, expanding_regime_evaluate
from gold.training.walkforward import chronological_split, walk_forward_evaluate

log = get_logger("training")

# Calendar-day horizons mapped to trading-day shifts used for supervised labels.
HORIZON_MAP = {1: 1, 7: 5, 30: 21, 90: 63}
PIPELINE_VERSION = "v2-logret"


def _mlflow_run(cfg: AppConfig, name: str, params: dict, metrics: dict, artifact: Path | None) -> None:
    try:
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        import mlflow

        tracking = cfg.get("mlflow.tracking_uri") or f"file:{cfg.mlflow_dir}"
        mlflow.set_tracking_uri(tracking)
        mlflow.set_experiment(cfg.get("mlflow.experiment_name", "gold-forecasting"))
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
            if artifact and artifact.exists():
                mlflow.log_artifact(str(artifact))
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow logging skipped: %s", exc)


def _parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if value is not None else {}


def _holdout_block(metrics: Any) -> dict:
    parsed = _parse_json(metrics)
    return parsed.get("holdout") or {}


class TrainingOrchestrator:
    def __init__(self, cfg: AppConfig, store: GoldStore):
        self.cfg = cfg
        self.store = store
        self.features = FeatureBuilder(cfg, store)
        t = cfg.get("training", {}) or {}
        self.holdout = int(t.get("holdout_trading_days", 252))
        self.min_train = int(t.get("min_train_trading_days", 504))
        self.step = int(t.get("walk_forward_step", 21))
        self.test_size = int(t.get("walk_forward_test_size", 21))
        self.seed = int(t.get("random_seed", 42))
        self.pipeline_version = str(t.get("pipeline_version", PIPELINE_VERSION))
        self.min_regimes = int(t.get("min_regimes_to_promote", 3))
        self.regimes = t.get("regimes") or DEFAULT_REGIMES

    def _dataset(self, horizon_calendar: int):
        h_td = HORIZON_MAP[horizon_calendar]
        X, y, meta = self.features.supervised(h_td)
        if len(y) < self.min_train + 30:
            raise ValueError(f"Not enough rows ({len(y)}) to train horizon {horizon_calendar}d")
        return X, y, meta, h_td

    def _factory(self, name: str, horizon_trading_days: int):
        if name == "arima":
            return lambda: ARIMAForecaster(horizon_steps=horizon_trading_days)
        if name == "sarima":
            return lambda: SARIMAForecaster(horizon_steps=horizon_trading_days)
        return MODEL_FACTORIES[name]

    def train_all(self, models: list[str] | None = None) -> pd.DataFrame:
        names = models or available_models(include_optional=False)
        names = [n for n in names if n not in SKIP_UNTIL_BASELINES_BEAT_NAIVE]
        run_id = new_run_id("train")
        started = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        previous = self.store.experiments()
        rows = []
        # History is preserved. Champions for this run are chosen only among this run_id.

        for horizon in self.cfg.get("forecast.horizons_calendar_days", [1, 7, 30, 90]):
            X, y, meta, h_td = self._dataset(int(horizon))
            n = len(y)
            holdout = min(self.holdout, max(30, n // 5))
            train_idx, test_idx = chronological_split(n, holdout)
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            meta_train, meta_test = meta.iloc[train_idx], meta.iloc[test_idx]
            close_test = meta_test["close"]
            fitted: dict[str, Any] = {}
            member_holdout: dict[str, np.ndarray] = {}
            min_train_wf = min(self.min_train, max(60, len(y_train) // 2))
            regime_min_train = min(252, max(60, min_train_wf))

            for name in names:
                if name == "ensemble":
                    continue
                factory = self._factory(name, h_td)
                log.info("Training %s horizon=%sd (log-return + expanding regimes)", name, horizon)
                wf_metrics, _ = walk_forward_evaluate(
                    factory,
                    y_train,
                    X_train,
                    meta_train,
                    min_train=min_train_wf,
                    test_size=min(self.test_size, max(5, len(y_train) // 10)),
                    step=self.step,
                )
                regime = expanding_regime_evaluate(
                    factory,
                    y,
                    X,
                    meta,
                    regimes=self.regimes,
                    min_train=regime_min_train,
                )
                model = factory()
                model.fit(y_train, X_train)
                pred = np.asarray(model.predict(len(y_test), X_future=X_test), dtype=float)[: len(y_test)]
                if len(pred) < len(y_test):
                    fill = pred[-1] if len(pred) else 0.0
                    pred = np.pad(pred, (0, len(y_test) - len(pred)), constant_values=fill)
                scored = score_log_return_forecast(close_test, y_test, pred)
                holdout_metrics = dict(scored["price"])
                holdout_metrics["return_mae"] = scored["return"]["mae"]
                holdout_metrics["return_rmse"] = scored["return"]["rmse"]
                holdout_metrics["return_mape"] = scored["return"]["mape"]
                skip = getattr(model, "skip_reason", None)
                exp_id = new_run_id(f"{name}_h{horizon}")
                artifact = self.cfg.model_dir / f"{name}_h{horizon}.joblib"
                try:
                    joblib.dump(model, artifact)
                except Exception:
                    artifact = Path("")
                params = dict(getattr(model, "params", {}) or {})
                params.update(
                    {
                        "pipeline_version": self.pipeline_version,
                        "run_id": run_id,
                        "target": "h_day_log_return",
                        "horizon_trading_days": h_td,
                    }
                )
                notes = skip or (
                    "target=h-day log return; prices reconstructed from close[t]; "
                    "chronological holdout; expanding-window regimes; no random split"
                )
                payload = {
                    "experiment_id": exp_id,
                    "model_name": name,
                    "horizon": int(horizon),
                    "trained_at": datetime.now(tz=timezone.utc).replace(tzinfo=None),
                    "cutoff_date": pd.to_datetime(meta_train["obs_date"].iloc[-1]).date(),
                    "metrics": {
                        "walk_forward": wf_metrics,
                        "holdout": holdout_metrics,
                        "regimes": regime,
                        "target": "h_day_log_return",
                        "pipeline_version": self.pipeline_version,
                        "run_id": run_id,
                    },
                    "params": params,
                    "artifact_path": str(artifact) if artifact else None,
                    "is_champion": False,
                    "notes": notes,
                }
                self.store.record_experiment(payload)
                _mlflow_run(
                    self.cfg,
                    f"{name}_h{horizon}_{self.pipeline_version}",
                    {"model": name, "horizon": horizon, **{str(k): v for k, v in params.items()}},
                    {
                        **{f"holdout_{k}": v for k, v in holdout_metrics.items() if isinstance(v, (int, float))},
                        "regimes_beat": float(regime.get("regimes_beat") or 0),
                        "regimes_evaluated": float(regime.get("regimes_evaluated") or 0),
                    },
                    artifact if artifact and artifact.exists() else None,
                )
                fitted[name] = model
                member_holdout[name] = pred
                rows.append(
                    {
                        "model": name,
                        "horizon": horizon,
                        "holdout_mae": holdout_metrics["mae"],
                        "holdout_rmse": holdout_metrics["rmse"],
                        "holdout_mape": holdout_metrics["mape"],
                        "holdout_smape": holdout_metrics["smape"],
                        "holdout_da": holdout_metrics.get("directional_accuracy"),
                        "walk_forward_mae": wf_metrics.get("mae"),
                        "regimes_beat": regime.get("regimes_beat"),
                        "regimes_evaluated": regime.get("regimes_evaluated"),
                        "n_holdout": holdout_metrics["n"],
                        "pipeline_version": self.pipeline_version,
                        "run_id": run_id,
                        "notes": skip or "",
                    }
                )

            usable = [
                n
                for n, p in member_holdout.items()
                if np.isfinite(p).all() and n not in {"naive", "moving_average"}
            ]
            if usable:
                blend = np.nanmean(np.vstack([member_holdout[n] for n in usable]), axis=0)
                ens_scored = score_log_return_forecast(close_test, y_test, blend)
                ens_metrics = dict(ens_scored["price"])
                ens_metrics["return_mae"] = ens_scored["return"]["mae"]
                members = [fitted[n] for n in usable]
                ens = EnsembleForecaster(members=members)
                ens.last = 0.0
                artifact = self.cfg.model_dir / f"ensemble_h{horizon}.joblib"
                joblib.dump(ens, artifact)
                member_names = list(usable)
                ens_regime = expanding_regime_evaluate(
                    lambda names=member_names, steps=h_td: EnsembleForecaster(
                        members=[self._factory(n, steps)() for n in names]
                    ),
                    y,
                    X,
                    meta,
                    regimes=self.regimes,
                    min_train=regime_min_train,
                )
                self.store.record_experiment(
                    {
                        "experiment_id": new_run_id(f"ensemble_h{horizon}"),
                        "model_name": "ensemble",
                        "horizon": int(horizon),
                        "trained_at": datetime.now(tz=timezone.utc).replace(tzinfo=None),
                        "cutoff_date": pd.to_datetime(meta_train["obs_date"].iloc[-1]).date(),
                        "metrics": {
                            "holdout": ens_metrics,
                            "members": usable,
                            "regimes": ens_regime,
                            "target": "h_day_log_return",
                            "pipeline_version": self.pipeline_version,
                            "run_id": run_id,
                        },
                        "params": {
                            "members": usable,
                            "pipeline_version": self.pipeline_version,
                            "run_id": run_id,
                            "target": "h_day_log_return",
                        },
                        "artifact_path": str(artifact),
                        "is_champion": False,
                        "notes": "equal-weight blend of finite non-naive members; log-return then reconstruct",
                    }
                )
                rows.append(
                    {
                        "model": "ensemble",
                        "horizon": horizon,
                        "holdout_mae": ens_metrics["mae"],
                        "holdout_rmse": ens_metrics["rmse"],
                        "holdout_mape": ens_metrics["mape"],
                        "holdout_smape": ens_metrics["smape"],
                        "holdout_da": ens_metrics.get("directional_accuracy"),
                        "walk_forward_mae": None,
                        "regimes_beat": ens_regime.get("regimes_beat"),
                        "regimes_evaluated": ens_regime.get("regimes_evaluated"),
                        "n_holdout": ens_metrics["n"],
                        "pipeline_version": self.pipeline_version,
                        "run_id": run_id,
                        "notes": f"members={usable}",
                    }
                )

            self._promote_champions(int(horizon), run_id)

        result = pd.DataFrame(rows)
        comparison = self._compare_versions(result, previous)
        finished = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        self.store.record_training_run(
            {
                "run_id": run_id,
                "pipeline_version": self.pipeline_version,
                "started_at": started,
                "finished_at": finished,
                "n_experiments": int(len(result)),
                "comparison_json": comparison,
                "notes": "log-return targets; expanding-window regimes; history preserved",
            }
        )
        return result

    def _compare_versions(self, current: pd.DataFrame, previous: pd.DataFrame) -> list[dict[str, Any]]:
        if current.empty:
            return []
        prev_rows = []
        if previous is not None and not previous.empty:
            for _, row in previous.iterrows():
                version = pipeline_version_of(row.get("params"))
                if version == self.pipeline_version:
                    continue
                holdout = _holdout_block(row.get("metrics"))
                mae = holdout.get("mae")
                if mae is None:
                    continue
                prev_rows.append(
                    {
                        "model": row["model_name"],
                        "horizon": int(row["horizon"]),
                        "pipeline_version": version,
                        "holdout_mae": float(mae),
                        "trained_at": str(row.get("trained_at")),
                    }
                )
        prev_df = pd.DataFrame(prev_rows)
        out = []
        for _, row in current.iterrows():
            rec = {
                "model": row["model"],
                "horizon": int(row["horizon"]),
                "current_version": self.pipeline_version,
                "current_mae": float(row["holdout_mae"]) if pd.notna(row["holdout_mae"]) else None,
            }
            if not prev_df.empty:
                match = prev_df[
                    (prev_df["model"] == row["model"]) & (prev_df["horizon"] == int(row["horizon"]))
                ].sort_values("trained_at")
                if not match.empty:
                    last = match.iloc[-1]
                    rec["previous_version"] = last["pipeline_version"]
                    rec["previous_mae"] = float(last["holdout_mae"])
                    if rec["current_mae"] is not None and rec["previous_mae"]:
                        rec["mae_vs_previous"] = rec["current_mae"] / rec["previous_mae"]
            out.append(rec)
        if out:
            log.info("experiment comparison vs previous versions: %s rows", len(out))
        return out

    def _promote_champions(self, horizon: int, run_id: str) -> None:
        exps = self.store.experiments()
        if exps.empty:
            return
        subset = exps[exps["horizon"] == horizon].copy()
        if subset.empty:
            return
        subset["params_obj"] = subset["params"].map(_parse_json)
        subset["run_id"] = subset["params_obj"].map(lambda p: (p or {}).get("run_id"))
        current = subset[subset["run_id"] == run_id].copy()
        if current.empty:
            current = subset.copy()

        def holdout_mae(row) -> float:
            holdout = _holdout_block(row["metrics"])
            return float(holdout.get("mae", np.inf))

        def regime_counts(row) -> tuple[int, int]:
            metrics = _parse_json(row["metrics"])
            regimes = metrics.get("regimes") or {}
            return int(regimes.get("regimes_beat") or 0), int(regimes.get("regimes_evaluated") or 0)

        current["holdout_mae"] = current.apply(holdout_mae, axis=1)
        naive_rows = current[current["model_name"] == "naive"]
        naive_mae = float(naive_rows["holdout_mae"].iloc[0]) if not naive_rows.empty else np.inf
        naive_id = naive_rows["experiment_id"].iloc[0] if not naive_rows.empty else None
        winner_id = naive_id
        winner_name = "naive"
        winner_mae = naive_mae

        eligible = current[np.isfinite(current["holdout_mae"]) & (current["model_name"] != "naive")]
        for _, row in eligible.sort_values("holdout_mae").iterrows():
            beat, evaluated = regime_counts(row)
            if beats_promotion_bar(
                float(row["holdout_mae"]),
                naive_mae,
                beat,
                evaluated,
                min_regimes=self.min_regimes,
            ):
                winner_id = row["experiment_id"]
                winner_name = row["model_name"]
                winner_mae = float(row["holdout_mae"])
                break

        if winner_name != "naive":
            log.info(
                "Champion h=%sd: %s MAE=%.4f (naive=%.4f) after holdout + %s-regime bar",
                horizon,
                winner_name,
                winner_mae,
                naive_mae,
                self.min_regimes,
            )
        else:
            log.info(
                "No model beat naive MAE %.4f on %sd holdout AND %s regimes; champion remains naive",
                naive_mae,
                horizon,
                self.min_regimes,
            )

        if winner_id is None:
            return
        self.store.con.execute("UPDATE experiments SET is_champion = FALSE WHERE horizon = ?", [horizon])
        self.store.con.execute(
            "UPDATE experiments SET is_champion = TRUE WHERE experiment_id = ?", [winner_id]
        )
        current_tbl = self.store.con.execute("SELECT * FROM experiments").df()
        path = self.cfg.parquet_dir / "experiments" / "experiments.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        current_tbl.to_parquet(path, index=False)


if __name__ == "__main__":
    from gold.config import load_config
    from gold.database.store import GoldStore
    from gold.logging import setup_logging

    cfg = load_config()
    setup_logging(cfg.log_dir)
    print(TrainingOrchestrator(cfg, GoldStore(cfg)).train_all())
