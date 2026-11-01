"""Read-only adapters for the Streamlit UI. No separate pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

from gold.config import AppConfig, load_config
from gold.data.quality import read_quality_table
from gold.database.store import GoldStore
from gold.ops.audit import set_actor
from gold.ops.book import brief_forecast_fingerprint
from gold.terminal.brief import build_brief


class UiDataError(RuntimeError):
    """Raised when the local DuckDB catalog is missing or unreadable."""


def load_ui_payload(cfg: AppConfig | None = None, store: GoldStore | None = None) -> dict:
    """Same forecast book as `gold brief` when pointed at the same DuckDB."""
    owns_store = store is None
    cfg = cfg or load_config()
    set_actor("ui")
    if not cfg.duckdb_path.exists():
        raise UiDataError(
            f"DuckDB not found at {cfg.duckdb_path}. Run `gold init` then `gold data update`."
        )
    store = store or GoldStore(cfg)
    try:
        brief = build_brief(cfg, store, run_monitor=False)
        quality = read_quality_table(cfg, store)
        health = store.health()
        outcomes = store.forecast_outcomes()
        catalog = store.catalog()
        brief.setdefault("forecast_book", brief_forecast_fingerprint(brief))
        loaded_at = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
        return {
            "brief": brief,
            "quality": quality,
            "health": health,
            "outcomes": outcomes,
            "catalog": catalog,
            "forecast_book": brief["forecast_book"],
            "loaded_at": loaded_at,
        }
    except UiDataError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise UiDataError(
            f"Could not read the local database ({exc}). "
            "The CLI still works independently. Try `gold status` or `gold data update`."
        ) from exc
    finally:
        if owns_store and store is not None:
            store.close()


def run_and_audit_scenario(snap, elasticities: dict, shocks: dict, cfg: AppConfig | None = None):
    """UI path for custom shocks. Same engine as `gold scenario`, with an audit row."""
    from gold.ops.audit import audit_scenario, set_actor
    from gold.scenarios.engine import run_scenarios

    cfg = cfg or load_config()
    set_actor("ui")
    table = run_scenarios(snap, cfg, elasticities, **shocks)
    db = GoldStore(cfg)
    try:
        audit_scenario(db, table, shocks=shocks)
    finally:
        db.close()
    return table
