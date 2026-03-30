from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any
from typing import Dict
from typing import Optional

from impacts.manifest.file_ops import write_structured_file


def _reject_unknown_keys(payload: Dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unsupported keys under {label}: {unknown}")

def _deep_update_strings(value: Any, resolver) -> Any:
    if isinstance(value, dict):
        return {k: _deep_update_strings(v, resolver) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_update_strings(item, resolver) for item in value]
    if isinstance(value, str):
        return resolver(value)
    return value


def _parse_epsg(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper().startswith("EPSG:"):
        return text.upper()
    try:
        return f"EPSG:{int(text)}"
    except ValueError:
        return text


def _lookup_dotted(source: Dict[str, Any], dotted_key: str) -> Optional[str]:
    current: Any = source
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current.get(part)
    if current is None:
        return None
    text = str(current).strip()
    return text or None


def _expand_impacts_placeholders(
    impacts_section: Dict[str, Any],
    pilates_settings: Dict[str, Any],
) -> Dict[str, Any]:
    expanded = deepcopy(impacts_section)

    def _normalize_path_like(text: str) -> str:
        if "://" in text:
            prefix, rest = text.split("://", 1)
            rest = re.sub(r"/{2,}", "/", rest)
            return f"{prefix}://{rest}"
        return re.sub(r"/{2,}", "/", text)

    def resolve_text(text: str) -> str:
        updated = text
        matches = re.findall(r"\$\{([^}]+)\}", updated)
        for key in matches:
            replacement = _lookup_dotted(expanded, key) or _lookup_dotted(pilates_settings, key)
            if replacement is not None:
                updated = updated.replace(f"${{{key}}}", replacement)
        matches = re.findall(r"\{([^}]+)\}", updated)
        for key in matches:
            replacement = _lookup_dotted(expanded, key) or _lookup_dotted(pilates_settings, key)
            if replacement is not None:
                updated = updated.replace(f"{{{key}}}", replacement)
        return _normalize_path_like(updated)

    for _ in range(5):
        next_expanded = _deep_update_strings(expanded, resolve_text)
        if next_expanded == expanded:
            break
        expanded = next_expanded
    return expanded


def _assert_no_unresolved_placeholders(value: Any, *, label: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _assert_no_unresolved_placeholders(nested, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_no_unresolved_placeholders(nested, label=f"{label}[{index}]")
        return
    if isinstance(value, str) and (re.search(r"\$\{[^}]+\}", value) or re.search(r"\{[^}]+\}", value)):
        raise ValueError(f"Unresolved placeholder in {label}: {value}")


def _validate_run_and_shared_sections(pilates_settings: Dict[str, Any]) -> None:
    run = dict(pilates_settings.get("run", {}) or {})
    shared = dict(pilates_settings.get("shared", {}) or {})
    _reject_unknown_keys(run, {"region", "scenario", "start_year"}, "run")
    _reject_unknown_keys(shared, {"geography"}, "shared")
    geography = dict(shared.get("geography", {}) or {})
    _reject_unknown_keys(geography, {"FIPS", "local_crs"}, "shared.geography")
    fips = dict(geography.get("FIPS", {}) or {})
    _reject_unknown_keys(fips, {"state", "counties"}, "shared.geography.FIPS")


def build_runtime_payload_from_pilates(
    pilates_settings: Dict[str, Any],
    impacts_overlay: Dict[str, Any],
) -> Dict[str, Any]:
    _reject_unknown_keys(impacts_overlay, {"impacts", "__source_root__"}, "impacts overlay root")
    _validate_run_and_shared_sections(pilates_settings)
    run = pilates_settings.get("run", {}) or {}
    shared = pilates_settings.get("shared", {}) or {}
    impacts_section = _expand_impacts_placeholders(
        impacts_overlay.get("impacts", {}) or {},
        pilates_settings,
    )
    _assert_no_unresolved_placeholders(impacts_section, label="impacts")
    local_input_folder = str(impacts_section.get("local_input_folder") or "").strip()
    if not local_input_folder:
        raise ValueError("impacts.local_input_folder must be provided in settings.")
    impacts_output_dir = str(impacts_section.get("local_output_folder") or "").strip()
    if not impacts_output_dir:
        raise ValueError("impacts.local_output_folder must be provided in settings.")
    geography = shared.get("geography", {}) or {}
    return {
        "run": {
            "region": run.get("region"),
            "scenario": run.get("scenario"),
            "start_year": run.get("start_year"),
        },
        "shared": {
            "geography": {
                "FIPS": geography.get("FIPS", {}) or {},
                "local_crs": _parse_epsg(geography.get("local_crs")),
            },
        },
        "impacts": impacts_section,
    }


def derive_runtime_config_from_pilates(
    *,
    pilates_settings_path: str | Path,
    impacts_model_config_path: str | Path,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    from impacts.config.runtime_builder import build_runtime_config_from_pilates

    runtime_config = build_runtime_config_from_pilates(
        pilates_settings=pilates_settings_path,
        impacts_overlay=impacts_model_config_path,
    )
    payload = runtime_config.to_dict()
    if output_path is not None:
        write_structured_file(output_path, payload)
    return payload
