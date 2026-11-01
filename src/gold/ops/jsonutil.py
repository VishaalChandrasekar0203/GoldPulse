from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd


def jsonable(obj: Any) -> Any:
    """Convert nested values into JSON-serializable Python types."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, float):
        return obj if obj == obj else None  # NaN → null
    try:
        import numpy as np

        if isinstance(obj, (np.floating,)):
            val = float(obj)
            return val if val == val else None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    return str(obj)
