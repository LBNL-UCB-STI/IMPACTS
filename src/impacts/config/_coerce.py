from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Optional


def _required_string(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing required value: {label}")
    return text


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_int(value: Any, label: str) -> int:
    if value is None:
        raise ValueError(f"Missing required value: {label}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {label}: {value}") from exc


def _optional_int(value: Any) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value: {value}") from exc


def _required_float(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing required value: {label}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for {label}: {value}") from exc


def _required_float_or_string(value: Any, label: str) -> float | str:
    if value is None:
        raise ValueError(f"Missing required value: {label}")
    if isinstance(value, bool):
        raise ValueError(f"Invalid float-or-string for {label}: {value}")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError(f"Missing required value: {label}")
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def _optional_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float value: {value}") from exc


def _required_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError(f"Missing required value: {label}")
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean for {label}: {value}")


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _reject_unknown_keys(payload: Dict[str, Any], allowed: set, label: str) -> None:
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unsupported keys under {label}: {unknown}")
