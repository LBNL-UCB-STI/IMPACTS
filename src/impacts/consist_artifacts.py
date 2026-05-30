from __future__ import annotations

import inspect
import logging
from pathlib import Path
import re
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional

logger = logging.getLogger(__name__)

BEAM_R5_OSM_FILE_KEY = "beam_r5_osm_file"
BEAM_EVENTS_PREFIX = "events_parquet"
BEAM_NETWORK_PREFIX = "beam_network_final"
BEAM_HOUSEHOLDS_PREFIX = "beam_households_final"
BEAM_POPULATION_PREFIX = "beam_population_final"


def consist_available() -> bool:
    try:
        import consist  # noqa: F401
    except Exception:
        return False
    return True


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
    for raw in _artifact_path_candidates(artifact):
        path = Path(raw).expanduser()
        if path.exists():
            return str(path.resolve())
    for raw in _artifact_path_candidates(artifact):
        if "://" not in raw:
            return str(Path(raw).expanduser().resolve())
        return raw
    return None


def _entry_from_artifact(
    *,
    key: str,
    artifact: Any,
    optional: bool = False,
    source_key: Optional[str] = None,
    fallback_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    resolved_path = _resolve_artifact_path(artifact) or (str(Path(fallback_path).resolve()) if fallback_path else None)
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


def _call_consist_lookup(query_fn: Any, *, key: str, metadata: Dict[str, Any]) -> Any:
    signature = inspect.signature(query_fn)
    accepted = set(signature.parameters)
    candidate_kwargs = [
        {"key": key, "metadata": metadata},
        {"name": key, "metadata": metadata},
        {"artifact_key": key, "metadata": metadata},
        {"key": key},
        {"name": key},
        {"artifact_key": key},
        {"metadata": metadata},
        {},
    ]
    failures: list[str] = []
    for raw_kwargs in candidate_kwargs:
        kwargs = {name: value for name, value in raw_kwargs.items() if name in accepted}
        try:
            return query_fn(**kwargs)
        except TypeError as exc:
            failures.append(str(exc))
            continue
    raise TypeError(
        f"Could not find a supported Consist lookup signature for {query_fn}. "
        f"Attempted kwargs failed with: {failures}"
    )


def _call_consist_outputs_reader(query_fn: Any, *, metadata: Dict[str, Any]) -> Any:
    signature = inspect.signature(query_fn)
    accepted = set(signature.parameters)
    candidate_kwargs = [
        {"metadata": metadata},
        {"filters": metadata},
        {},
    ]
    failures: list[str] = []
    for raw_kwargs in candidate_kwargs:
        kwargs = {name: value for name, value in raw_kwargs.items() if name in accepted}
        try:
            return query_fn(**kwargs)
        except TypeError as exc:
            failures.append(str(exc))
            continue
    raise TypeError(
        f"Could not find a supported Consist run-output signature for {query_fn}. "
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


def find_input_reference(
    *,
    key: str,
    optional: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    consist = _consist_module()
    artifact_metadata = {"artifact_key": key, **(metadata or {})}
    for attr_name in ("find_artifact", "get_artifact", "find_input"):
        query_fn = getattr(consist, attr_name, None)
        if query_fn is None:
            continue
        try:
            artifact = _call_consist_lookup(query_fn, key=key, metadata=artifact_metadata)
        except Exception as exc:
            logger.warning("Consist %s lookup failed for %s: %s", attr_name, key, exc)
            continue
        if artifact is None:
            continue
        entry = _entry_from_artifact(key=key, artifact=artifact, optional=optional)
        if entry is not None:
            return entry
    return None


def _select_latest_dynamic_key(keys: Iterable[str], prefix: str) -> Optional[str]:
    normalized_keys = [str(key) for key in keys]
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_(\d+)(?:_sub(\d+))?$")
    best_key: Optional[str] = None
    best_rank: tuple[int, int, int, int] = (-1, -1, -1, -1)
    for key in normalized_keys:
        match = pattern.match(key)
        if not match:
            continue
        year = int(match.group(1))
        iteration = int(match.group(2))
        subiteration_raw = match.group(3)
        is_exact = 1 if subiteration_raw is None else 0
        subiteration = -1 if subiteration_raw is None else int(subiteration_raw)
        rank = (year, iteration, is_exact, subiteration)
        if rank > best_rank:
            best_rank = rank
            best_key = key
    if best_key is None and prefix in normalized_keys:
        return prefix
    return best_key


def find_latest_dynamic_input_reference(
    *,
    prefixes: Iterable[str],
    optional: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    consist = _consist_module()
    outputs: Any = None
    for attr_name in ("get_run_outputs",):
        query_fn = getattr(consist, attr_name, None)
        if query_fn is None:
            continue
        try:
            outputs = _call_consist_outputs_reader(
                query_fn,
                metadata=(metadata or {}),
            )
        except Exception as exc:
            logger.warning("Consist %s lookup failed: %s", attr_name, exc)
        if outputs is not None:
            break
    if outputs is None:
        outputs = getattr(consist, "run_outputs", None)
    if outputs is None:
        outputs = getattr(consist, "raw_outputs", None)
    if not isinstance(outputs, dict):
        return None

    for prefix in prefixes:
        best_key = _select_latest_dynamic_key(outputs.keys(), prefix)
        if not best_key:
            continue
        entry = _entry_from_artifact(
            key=best_key,
            artifact=outputs[best_key],
            optional=optional,
            source_key=best_key,
        )
        if entry is not None:
            return entry
    return None


def find_beam_r5_osm_reference(*, optional: bool = False) -> Optional[Dict[str, Any]]:
    return find_input_reference(key=BEAM_R5_OSM_FILE_KEY, optional=optional, metadata={"artifact_family": BEAM_R5_OSM_FILE_KEY})


def find_latest_beam_events_reference(*, optional: bool = False) -> Optional[Dict[str, Any]]:
    return find_latest_dynamic_input_reference(
        prefixes=[BEAM_EVENTS_PREFIX],
        optional=optional,
        metadata={"artifact_family": BEAM_EVENTS_PREFIX},
    )


def find_latest_beam_network_reference(*, optional: bool = False) -> Optional[Dict[str, Any]]:
    return find_latest_dynamic_input_reference(
        prefixes=[BEAM_NETWORK_PREFIX],
        optional=optional,
        metadata={"artifact_family": BEAM_NETWORK_PREFIX},
    )


def find_latest_beam_households_reference(*, optional: bool = False) -> Optional[Dict[str, Any]]:
    return find_latest_dynamic_input_reference(
        prefixes=[BEAM_HOUSEHOLDS_PREFIX],
        optional=optional,
        metadata={"artifact_family": BEAM_HOUSEHOLDS_PREFIX},
    )


def find_latest_beam_population_reference(*, optional: bool = False) -> Optional[Dict[str, Any]]:
    return find_latest_dynamic_input_reference(
        prefixes=[BEAM_POPULATION_PREFIX],
        optional=optional,
        metadata={"artifact_family": BEAM_POPULATION_PREFIX},
    )


def log_input_reference(
    *,
    key: str,
    source_path: str,
    artifact_key: Optional[str] = None,
    optional: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    consist = _consist_module()
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
        fallback_path=normalized_path,
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
        fallback_path=normalized_path,
    )


def resolve_logged_path(entry: Dict[str, Any]) -> str:
    staged_path = str((entry or {}).get("staged_path") or "").strip()
    if staged_path:
        return staged_path
    source_path = str((entry or {}).get("source_path") or "").strip()
    if source_path:
        return source_path
    raise ValueError("Could not resolve a usable path from logged entry")
