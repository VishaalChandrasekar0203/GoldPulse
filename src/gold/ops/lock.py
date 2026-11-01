from __future__ import annotations

import json
import os
import time
from pathlib import Path

from gold.config import AppConfig
from gold.exceptions import LockBusy
from gold.logging import get_logger

log = get_logger("ops.lock")


def lock_path(cfg: AppConfig) -> Path:
    name = str((cfg.get("ops", {}) or {}).get("lock_file") or "gold.run.lock")
    return cfg.data_dir / name


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class RunLock:
    """Atomic exclusive file lock (O_EXCL) so gold run / backup / restore cannot overlap."""

    def __init__(self, cfg: AppConfig):
        self.path = lock_path(cfg)
        self.stale_seconds = int((cfg.get("ops", {}) or {}).get("lock_stale_seconds", 10800))
        self._held = False
        self._fd: int | None = None

    def _try_create(self) -> bool:
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(self.path, flags, 0o644)
        except FileExistsError:
            return False
        payload = json.dumps({"pid": os.getpid(), "ts": time.time()})
        os.write(fd, payload.encode("utf-8"))
        self._fd = fd
        return True

    def _clear_stale(self) -> bool:
        if not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text())
        except Exception:  # noqa: BLE001
            payload = {}
        pid = int(payload.get("pid") or 0)
        ts = float(payload.get("ts") or 0)
        age = time.time() - ts if ts else 1e9
        if _pid_alive(pid) and age < self.stale_seconds:
            return False
        log.warning("removing stale lock pid=%s age=%.0fs", pid, age)
        self.path.unlink(missing_ok=True)
        return True

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._try_create():
            self._held = True
            return
        if self._clear_stale() and self._try_create():
            self._held = True
            return
        try:
            payload = json.loads(self.path.read_text()) if self.path.exists() else {}
        except Exception:  # noqa: BLE001
            payload = {}
        pid = payload.get("pid", "?")
        raise LockBusy(
            f"gold run already in progress (pid={pid}, lock={self.path}). "
            "Wait for it to finish or delete a stale lock only if that process is dead."
        )

    def release(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        if self._held:
            self.path.unlink(missing_ok=True)
            self._held = False

    def __enter__(self) -> "RunLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
