from __future__ import annotations

import time
from typing import Callable, TypeVar

from gold.exceptions import SourceUnavailable
from gold.logging import get_logger

log = get_logger("ingestion.retry")

T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    backoff: float = 1.5,
    retry_on: tuple[type[BaseException], ...] = (SourceUnavailable,),
    label: str = "fetch",
) -> T:
    """Retry a fetch. Never fabricates a result on failure."""
    attempts = max(1, int(attempts))
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return fn()
        except retry_on as exc:
            last = exc
            log.warning("%s attempt %s/%s failed: %s", label, i + 1, attempts, exc)
            if i + 1 >= attempts:
                raise
            delay = float(backoff) * (2**i)
            if delay > 0:
                time.sleep(delay)
    assert last is not None
    raise last
