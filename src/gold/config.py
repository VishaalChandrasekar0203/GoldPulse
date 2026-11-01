from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from gold.exceptions import ConfigError

TROY_OUNCE_GRAMS = 31.1034768


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "config").exists():
            return candidate
    env_home = Path(__file__).resolve().parents[2]
    return env_home if (env_home / "config").exists() else start


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GOLD_", extra="ignore")

    home: Path | None = None
    fred_api_key: str | None = None
    goldapi_key: str | None = None


class AppConfig(BaseModel):
    raw: dict[str, Any]
    sources: dict[str, Any]
    project_root: Path
    data_dir: Path
    parquet_dir: Path
    duckdb_path: Path
    mlflow_dir: Path
    reports_dir: Path
    log_dir: Path
    model_dir: Path
    timezone: str = "Asia/Kolkata"
    primary_series: str = "gold_inr_10g"
    secondary_series: str = "gold_etf_inr_10g"
    troy_ounce_grams: float = TROY_OUNCE_GRAMS
    start_date: str = "2010-01-01"
    seed: int = 42

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Missing config file: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a mapping: {path}")
    return data


def load_config(project_root: str | None = None) -> AppConfig:
    settings = Settings()
    root = Path(project_root) if project_root else (settings.home or find_project_root())
    default = _load_yaml(root / "config" / "default.yaml")
    sources = _load_yaml(root / "config" / "sources.yaml")
    paths = default.get("paths", {})
    data_dir = root / paths.get("data_dir", "data")
    return AppConfig(
        raw=default,
        sources=sources,
        project_root=root,
        data_dir=data_dir,
        parquet_dir=root / paths.get("parquet_dir", "data/parquet"),
        duckdb_path=root / paths.get("duckdb_path", "data/gold.duckdb"),
        mlflow_dir=root / paths.get("mlflow_dir", "mlruns"),
        reports_dir=root / paths.get("reports_dir", "reports/output"),
        log_dir=root / paths.get("log_dir", "data/logs"),
        model_dir=root / paths.get("model_dir", "data/models"),
        timezone=default.get("project", {}).get("timezone", "Asia/Kolkata"),
        primary_series=default.get("target", {}).get("primary_series", "gold_inr_10g"),
        secondary_series=default.get("target", {}).get("secondary_series", "gold_etf_inr_10g"),
        troy_ounce_grams=float(
            default.get("target", {}).get("troy_ounce_grams", TROY_OUNCE_GRAMS)
        ),
        start_date=default.get("ingestion", {}).get("start_date", "2010-01-01"),
        seed=int(default.get("training", {}).get("random_seed", 42)),
    )


def ensure_directories(cfg: AppConfig) -> None:
    for path in (
        cfg.data_dir,
        cfg.parquet_dir,
        cfg.mlflow_dir,
        cfg.reports_dir,
        cfg.log_dir,
        cfg.model_dir,
        cfg.parquet_dir / "observations",
        cfg.parquet_dir / "features",
        cfg.parquet_dir / "forecasts",
        cfg.parquet_dir / "experiments",
        cfg.parquet_dir / "quality",
        cfg.parquet_dir / "audit",
        cfg.data_dir / "backups",
    ):
        path.mkdir(parents=True, exist_ok=True)
