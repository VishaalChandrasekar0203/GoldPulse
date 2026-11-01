from __future__ import annotations

import copy
from pathlib import Path

import pytest

from gold.config import AppConfig, load_config
from gold.database.store import GoldStore


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    base = load_config()
    data_dir = tmp_path / "data"
    return AppConfig(
        raw=copy.deepcopy(base.raw),
        sources=copy.deepcopy(base.sources),
        project_root=base.project_root,
        data_dir=data_dir,
        parquet_dir=data_dir / "parquet",
        duckdb_path=data_dir / "gold.duckdb",
        mlflow_dir=data_dir / "mlruns",
        reports_dir=data_dir / "reports",
        log_dir=data_dir / "logs",
        model_dir=data_dir / "models",
        timezone=base.timezone,
        primary_series=base.primary_series,
        secondary_series=base.secondary_series,
        troy_ounce_grams=base.troy_ounce_grams,
        start_date=base.start_date,
        seed=base.seed,
    )


@pytest.fixture
def store(cfg: AppConfig) -> GoldStore:
    from gold.config import ensure_directories

    ensure_directories(cfg)
    return GoldStore(cfg)
