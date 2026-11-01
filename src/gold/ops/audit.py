from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from gold.database.store import GoldStore
from gold.ops.book import forecast_fingerprint
from gold.ops.jsonutil import jsonable

_actor: ContextVar[str] = ContextVar("gold_audit_actor", default="system")
_run_id: ContextVar[str | None] = ContextVar("gold_audit_run_id", default=None)


def current_actor() -> str:
    return _actor.get()


def set_actor(name: str) -> None:
    _actor.set(str(name))


def current_run_id() -> str | None:
    return _run_id.get()


def set_run_id(run_id: str | None) -> None:
    _run_id.set(run_id)


def record_event(
    store: GoldStore,
    kind: str,
    payload: dict[str, Any] | None = None,
    *,
    actor: str | None = None,
    ref_id: str | None = None,
    run_id: str | None = None,
) -> str:
    rid = run_id or current_run_id()
    body = dict(payload or {})
    if rid and "run_id" not in body:
        body["run_id"] = rid
    return store.append_audit(
        kind=kind,
        payload=jsonable(body),
        actor=actor or current_actor(),
        ref_id=ref_id or rid,
        run_id=rid,
    )


def audit_forecasts(store: GoldStore, forecasts, *, extra: dict[str, Any] | None = None) -> str:
    import pandas as pd

    if forecasts is None or (isinstance(forecasts, pd.DataFrame) and forecasts.empty):
        return record_event(store, "forecast", {"n": 0, **(extra or {})})
    fid = str(forecasts.iloc[0]["forecast_id"]) if "forecast_id" in forecasts.columns else None
    payload = {
        "n": int(len(forecasts)),
        "rows": forecast_fingerprint(forecasts),
        **(extra or {}),
    }
    return record_event(store, "forecast", payload, ref_id=fid)


def audit_scenario(
    store: GoldStore,
    table,
    *,
    shocks: dict[str, Any] | None = None,
    pack: bool = False,
) -> str:
    import pandas as pd

    n = int(len(table)) if isinstance(table, pd.DataFrame) else 0
    kind = "scenario_pack" if pack else "scenario"
    summary = []
    if isinstance(table, pd.DataFrame) and not table.empty:
        cols = [c for c in ["scenario", "kind", "delta_inr_10g", "pack"] if c in table.columns]
        for rec in table.head(24).itertuples():
            row = {}
            for c in cols:
                row[c] = getattr(rec, c, None)
            summary.append(row)
    return record_event(
        store,
        kind,
        {
            "n": n,
            "shocks": shocks or {},
            "disclaimer": "SCENARIO, NOT A FORECAST",
            "rows": summary,
        },
    )
