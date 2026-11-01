"""Series catalog helpers (status of public vs restricted sources)."""

from __future__ import annotations

from typing import Any

from gold.config import AppConfig


def iter_active_series(cfg: AppConfig) -> list[tuple[str, dict[str, Any]]]:
    return [
        (sid, meta)
        for sid, meta in (cfg.sources.get("series") or {}).items()
        if meta.get("status", "active") == "active"
    ]


def restricted_series(cfg: AppConfig) -> dict[str, Any]:
    return dict(cfg.sources.get("restricted") or {})
