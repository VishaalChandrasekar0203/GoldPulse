"""DuckDB + Parquet backup and restore. Local files only."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gold.config import AppConfig, ensure_directories
from gold.database.store import GoldStore
from gold.exceptions import BackupError
from gold.logging import get_logger
from gold.ops.jsonutil import jsonable

log = get_logger("ops.backup")


def backup_root(cfg: AppConfig) -> Path:
    return cfg.data_dir / "backups"


def _stamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _copy_if_exists(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observation_counts_parquet(cfg: AppConfig) -> dict[str, int]:
    folder = cfg.parquet_dir / "observations"
    out: dict[str, int] = {}
    if not folder.exists():
        return out
    for path in sorted(folder.glob("*.parquet")):
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            out[path.stem] = -1
            continue
        out[path.stem] = int(len(df))
    return out


def observation_counts_duckdb(store: GoldStore) -> dict[str, int]:
    cov = store.coverage()
    if cov.empty:
        return {}
    return {str(r.series_id): int(r.n_obs) for r in cov.itertuples()}


def verify_catalog_consistency(cfg: AppConfig, store: GoldStore) -> dict:
    """DuckDB observations table must match parquet files. Restricted empties are ok."""
    duck = observation_counts_duckdb(store)
    parq = observation_counts_parquet(cfg)
    mismatches = []
    for sid in sorted(set(duck) | set(parq)):
        d = int(duck.get(sid) or 0)
        p = int(parq.get(sid) or 0)
        if d != p:
            mismatches.append({"series_id": sid, "duckdb": d, "parquet": p})
    return {
        "ok": not mismatches,
        "mismatches": mismatches,
        "n_duckdb": int(sum(max(0, v) for v in duck.values())),
        "n_parquet": int(sum(max(0, v) for v in parq.values())),
    }


def verify_backup(path: Path) -> dict:
    src = Path(path)
    manifest_path = src / "manifest.json"
    if not manifest_path.exists():
        raise BackupError(f"Backup missing manifest.json: {src}")
    manifest = json.loads(manifest_path.read_text())
    duck_name = manifest.get("duckdb")
    duck_src = src / duck_name if duck_name else None
    digest = sha256_file(duck_src) if duck_src else None
    expected = manifest.get("duckdb_sha256")
    duck_ok = bool(duck_src and duck_src.exists()) and (not expected or digest == expected)
    parquet_dir = src / "parquet"
    return {
        "ok": duck_ok and parquet_dir.exists() and manifest_path.exists(),
        "duckdb_ok": duck_ok,
        "duckdb_sha256": digest,
        "expected_sha256": expected,
        "has_parquet": parquet_dir.exists(),
        "manifest": manifest,
    }


def backup(cfg: AppConfig, store: GoldStore | None = None, *, already_locked: bool = False) -> Path:
    """CHECKPOINT DuckDB and copy catalog + parquet. Callers that hold RunLock pass already_locked."""
    from gold.ops.lock import RunLock

    if already_locked:
        return _backup(cfg, store)
    with RunLock(cfg):
        return _backup(cfg, store)


def _backup(cfg: AppConfig, store: GoldStore | None) -> Path:
    ensure_directories(cfg)
    checkpointed = False
    if store is not None:
        try:
            store.con.execute("CHECKPOINT")
            checkpointed = True
        except Exception as exc:  # noqa: BLE001
            log.warning("CHECKPOINT before backup failed: %s", exc)
    dest = backup_root(cfg) / _stamp()
    dest.mkdir(parents=True, exist_ok=True)
    copied_db = _copy_if_exists(cfg.duckdb_path, dest / cfg.duckdb_path.name)
    wal = Path(str(cfg.duckdb_path) + ".wal")
    if not checkpointed:
        _copy_if_exists(wal, dest / wal.name)
    parquet_src = cfg.parquet_dir
    if parquet_src.exists():
        shutil.copytree(parquet_src, dest / "parquet", dirs_exist_ok=True)
    n_obs = n_fc = None
    consistency = None
    if store is not None:
        try:
            n_obs = int(store.con.execute("SELECT count(*) FROM observations").fetchone()[0])
            n_fc = int(store.con.execute("SELECT count(*) FROM forecasts").fetchone()[0])
            consistency = verify_catalog_consistency(cfg, store)
        except Exception as exc:  # noqa: BLE001
            log.warning("backup consistency snapshot failed: %s", exc)
    duck_sha = sha256_file(dest / cfg.duckdb_path.name) if copied_db else None
    manifest = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "duckdb": cfg.duckdb_path.name if copied_db else None,
        "duckdb_sha256": duck_sha,
        "parquet": "parquet",
        "n_observations": n_obs,
        "n_forecasts": n_fc,
        "consistency": consistency,
        "pipeline_version": (cfg.get("training", {}) or {}).get("pipeline_version"),
        "note": "Local backup. Restricted MCX/IBJA series are not fabricated on restore.",
    }
    (dest / "manifest.json").write_text(json.dumps(jsonable(manifest), indent=2))
    _prune(cfg)
    log.info("backup written %s sha256=%s", dest, duck_sha)
    return dest


def _prune(cfg: AppConfig) -> None:
    retain = int((cfg.get("ops", {}) or {}).get("backup_retain", 7))
    root = backup_root(cfg)
    if not root.exists() or retain < 1:
        return
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for old in dirs[:-retain]:
        shutil.rmtree(old, ignore_errors=True)
        log.info("pruned old backup %s", old.name)


def restore(cfg: AppConfig, source: str | Path, *, confirm: bool = False) -> Path:
    """Replace DuckDB and parquet from a backup directory. Requires confirm=True."""
    from gold.ops.lock import RunLock

    if not confirm:
        raise BackupError("Restore refuses to overwrite without confirm=True / --yes")
    src = Path(source).expanduser().resolve()
    check = verify_backup(src)
    if not check["ok"]:
        raise BackupError(f"Backup failed verification: {check}")
    with RunLock(cfg):
        _restore_files(cfg, src, check["manifest"])
    store = GoldStore(cfg)
    try:
        cons = verify_catalog_consistency(cfg, store)
        if not cons["ok"]:
            raise BackupError(f"Restored catalog is inconsistent: {cons['mismatches']}")
    finally:
        store.close()
    log.info("restored catalog from %s", src)
    return cfg.duckdb_path


def _restore_files(cfg: AppConfig, src: Path, manifest: dict) -> None:
    duck_name = manifest.get("duckdb") or cfg.duckdb_path.name
    duck_src = src / duck_name
    if not duck_src.exists():
        raise BackupError(f"Backup missing DuckDB file {duck_name}")
    ensure_directories(cfg)
    cfg.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(duck_src, cfg.duckdb_path)
    wal_src = src / (duck_name + ".wal")
    wal_dest = Path(str(cfg.duckdb_path) + ".wal")
    if wal_src.exists():
        shutil.copy2(wal_src, wal_dest)
    elif wal_dest.exists():
        wal_dest.unlink()
    parquet_src = src / "parquet"
    if parquet_src.exists():
        if cfg.parquet_dir.exists():
            shutil.rmtree(cfg.parquet_dir)
        shutil.copytree(parquet_src, cfg.parquet_dir)
