from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_trace_dir(workflow: dict[str, object]) -> Path:
    trace_dir = Path(workflow["paths"]["trace_dir"]).expanduser()
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def write_trace(workflow: dict[str, object], name: str, payload: dict[str, Any]) -> Path:
    target = ensure_trace_dir(workflow) / f"{name}.json"
    with target.open("w") as handle:
        json.dump(_to_json_ready(payload), handle, indent=2, sort_keys=True)
    return target


def frame_summary(frame: pd.DataFrame, *, name: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "name": name,
        "row_count": int(len(frame)),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    if "process" in frame.columns:
        summary["process_counts"] = {
            str(key): int(value)
            for key, value in frame["process"].astype(str).value_counts(dropna=False).sort_index().items()
        }
    if "pollutant" in frame.columns:
        summary["pollutant_counts"] = {
            str(key): int(value)
            for key, value in frame["pollutant"].astype(str).value_counts(dropna=False).sort_index().items()
        }
    if "county" in frame.columns:
        summary["county_count"] = int(frame["county"].nunique(dropna=True))
    elif "sub_area" in frame.columns:
        summary["county_count"] = int(frame["sub_area"].nunique(dropna=True))
    return summary


def assert_row_count(expected: int, actual: int, *, label: str) -> None:
    if expected != actual:
        raise ValueError(f"{label} changed row count unexpectedly: expected {expected}, got {actual}")


def _to_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_ready(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value
