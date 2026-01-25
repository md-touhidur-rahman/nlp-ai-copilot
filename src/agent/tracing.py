# src/agent/tracing.py
import json
import os
import time
from typing import Any, Dict, Optional

TRACE_DIR = os.path.join("data", "traces")


def ensure_trace_dir() -> None:
    os.makedirs(TRACE_DIR, exist_ok=True)


def _safe(obj: Any) -> Any:
    """JSON-safe conversion for trace payloads."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(x) for x in obj]
    # DataFrames and other objects should not be dumped wholesale
    return str(obj)


def save_trace(payload: Dict[str, Any], filename: Optional[str] = None) -> str:
    """
    Persist a trace JSON to data/traces.
    Returns the filepath.
    """
    ensure_trace_dir()

    if filename is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        filename = f"trace-{ts}-{int(time.time() * 1000)}.json"

    path = os.path.join(TRACE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_safe(payload), f, ensure_ascii=False, indent=2)

    return path
