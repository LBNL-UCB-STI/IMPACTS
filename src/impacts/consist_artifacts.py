from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

logger = logging.getLogger(__name__)

BEAM_EVENTS_PREFIX = "events_parquet"
BEAM_NETWORK_PREFIX = "beam_network_final"


def _serialize_artifact(artifact: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "repr": repr(artifact),
        "type": type(artifact).__name__,
    }
    for field_name in ("id", "key", "name", "path", "uri", "container_uri"):
        value = getattr(artifact, field_name, None)
        if value is not None:
            payload[field_name] = str(value)
    return payload


def _artifact_path_candidates(artifact: Any) -> list[str]:
    if artifact is None:
        return []
    if isinstance(artifact, (str, Path)):
        return [str(artifact)]
    candidates: list[str] = []
    for field_name in ("path", "uri", "container_uri", "local_path", "artifact_path", "value"):
        value = getattr(artifact, field_name, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            candidates.append(text)
    return candidates


def _resolve_artifact_path(artifact: Any) -> Optional[str]:
    candidates = _artifact_path_candidates(artifact)
    for raw in candidates:
        if Path(raw).expanduser().exists():
            return str(Path(raw).expanduser().resolve())
    for raw in candidates:
        if "://" not in raw:
            return str(Path(raw).expanduser().resolve())
    return candidates[0] if candidates else None


def _entry_from_artifact(
    *,
    key: str,
    artifact: Any,
    optional: bool = False,
    source_key: Optional[str] = None,
    local_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    resolved_path = _resolve_artifact_path(artifact) or (str(Path(local_path).resolve()) if local_path else None)
    if not resolved_path:
        return None
    return {
        "kind": "consist",
        "source_path": resolved_path,
        "staged_path": resolved_path,
        "optional": optional,
        "exists": Path(resolved_path).exists() if "://" not in resolved_path else True,
        "storage_mode": "reference",
        "consist": {
            "enabled": True,
            "artifact": _serialize_artifact(artifact),
            "artifact_key": key,
            "source_key": source_key or key,
        },
    }


def _call_consist_logger(logger_fn: Any, *, path: str, key: str, metadata: Dict[str, Any]) -> Any:
    signature = inspect.signature(logger_fn)
    accepted = set(signature.parameters)
    candidate_kwargs = [
        {"path": path, "key": key, "metadata": metadata},
        {"path": path, "name": key, "metadata": metadata},
        {"artifact_path": path, "key": key, "metadata": metadata},
        {"uri": path, "key": key, "metadata": metadata},
    ]
    failures: list[str] = []
    for raw_kwargs in candidate_kwargs:
        kwargs = {name: value for name, value in raw_kwargs.items() if name in accepted}
        if not kwargs:
            continue
        try:
            return logger_fn(**kwargs)
        except TypeError as exc:
            failures.append(str(exc))
            continue
    raise TypeError(
        f"Could not find a supported Consist logger signature for {logger_fn}. "
        f"Attempted kwargs failed with: {failures}"
    )


def _consist_module() -> Any:
    try:
        import consist
    except ImportError as exc:
        raise ImportError(
            "The 'consist' package is required but is not installed. "
            "Install it with: pip install consist"
        ) from exc
    return consist


def _is_missing_active_run(error: RuntimeError) -> bool:
    message = str(error)
    return "No active Consist run found" in message


def log_input_reference(
    *,
    key: str,
    source_path: str,
    artifact_key: Optional[str] = None,
    optional: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        consist = _consist_module()
    except ImportError:
        return None
    publish_key = artifact_key or key
    logger_fn = getattr(consist, "log_input", None)
    if logger_fn is None:
        raise AttributeError(
            f"Consist is installed but does not expose 'log_input'. "
            f"Check your consist version."
        )

    normalized_path = str(Path(source_path).resolve())
    artifact_metadata = {
        "model": "impacts",
        "artifact_key": publish_key,
        "manifest_key": key,
        **(metadata or {}),
    }
    try:
        artifact = _call_consist_logger(
            logger_fn,
            path=normalized_path,
            key=publish_key,
            metadata=artifact_metadata,
        )
    except RuntimeError as exc:
        if _is_missing_active_run(exc):
            logger.info(
                "Consist has no active run; recording %s as a local input reference.",
                publish_key,
            )
            return None
        raise

    return _entry_from_artifact(
        key=publish_key,
        artifact=artifact,
        optional=optional,
        source_key=publish_key,
        local_path=normalized_path,
    )


def log_output_reference(
    *,
    key: str,
    path: str,
    artifact_key: Optional[str] = None,
    optional: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    consist = _consist_module()
    publish_key = artifact_key or key
    logger_fn = getattr(consist, "log_output", None)
    if logger_fn is None:
        raise AttributeError(
            f"Consist is installed but does not expose 'log_output'. "
            f"Check your consist version."
        )

    normalized_path = str(Path(path).resolve())
    artifact_metadata = {
        "model": "impacts",
        "artifact_key": publish_key,
        "manifest_key": key,
        **(metadata or {}),
    }
    artifact = _call_consist_logger(
        logger_fn,
        path=normalized_path,
        key=publish_key,
        metadata=artifact_metadata,
    )

    return _entry_from_artifact(
        key=publish_key,
        artifact=artifact,
        optional=optional,
        source_key=publish_key,
        local_path=normalized_path,
    )


def resolve_logged_path(entry: Dict[str, Any]) -> str:
    staged_path = str((entry or {}).get("staged_path") or "").strip()
    if staged_path:
        return staged_path
    source_path = str((entry or {}).get("source_path") or "").strip()
    if source_path:
        return source_path
    raise ValueError("Could not resolve a usable path from logged entry")
