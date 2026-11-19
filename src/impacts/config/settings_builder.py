from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict

from impacts.manifest.file_ops import load_structured_file
from impacts.manifest.file_ops import write_structured_file
from impacts.config._coerce import _reject_unknown_keys
from impacts.config.settings import DEFAULT_CONFIG_PATH
from impacts.config.settings import ImpactsSettings
from impacts.config.settings import normalize_epsg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_mapping(source: Dict[str, Any] | str | Path, *, label: str) -> Dict[str, Any]:
    payload = load_structured_file(source) if isinstance(source, (str, Path)) else dict(source)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping for {label}")
    payload = dict(payload)
    if isinstance(source, (str, Path)):
        payload["__source_root__"] = str(Path(source).resolve().parent)
    return payload


def _validate_pilates_settings_sections(pilates_settings: Dict[str, Any]) -> None:
    run = dict(pilates_settings.get("run", {}) or {})
    shared = dict(pilates_settings.get("shared", {}) or {})
    _reject_unknown_keys(run, {"region", "scenario", "start_year", "output_run_name"}, "run")
    _reject_unknown_keys(shared, {"geography"}, "shared")
    geography = dict(shared.get("geography", {}) or {})
    _reject_unknown_keys(geography, {"FIPS", "local_crs"}, "shared.geography")
    fips = dict(geography.get("FIPS", {}) or {})
    _reject_unknown_keys(fips, {"state", "counties"}, "shared.geography.FIPS")


def load_settings_from_yaml(path: str | Path) -> ImpactsSettings:
    payload = load_structured_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in settings file: {path}")
    payload = dict(payload)
    payload["__source_root__"] = str(Path(path).resolve().parent)
    return ImpactsSettings.from_dict(payload)


def build_settings_from_pilates(
    pilates_settings: Dict[str, Any] | str | Path,
) -> ImpactsSettings:
    pilates_payload = _load_mapping(pilates_settings, label="PILATES settings")
    _validate_pilates_settings_sections(pilates_payload)
    default_payload = _load_mapping(DEFAULT_CONFIG_PATH, label="default impacts settings")

    run = dict(pilates_payload.get("run", {}) or {})
    shared = dict(pilates_payload.get("shared", {}) or {})
    geography = dict(shared.get("geography", {}) or {})
    pilates_impacts = dict(pilates_payload.get("impacts", {}) or {})
    impacts_section = _deep_merge(
        dict(default_payload.get("impacts", {}) or {}),
        pilates_impacts,
    )
    start_year = run.get("start_year")
    if start_year is not None and "scenario" not in pilates_impacts:
        impacts_section["scenario"] = f"{start_year}-Baseline"
    beam_section = dict(pilates_payload.get("beam", {}) or {})

    settings_payload = {
        "run": {
            "region": run.get("region"),
            "scenario": run.get("scenario"),
            "start_year": start_year,
            "output_run_name": run.get("output_run_name"),
        },
        "shared": {
            "geography": {
                "FIPS": geography.get("FIPS", {}) or {},
                "local_crs": normalize_epsg(geography.get("local_crs")),
            },
        },
        "beam": beam_section,
        "impacts": impacts_section,
    }
    return ImpactsSettings.from_dict(settings_payload)


def derive_settings_from_pilates(
    *,
    pilates_settings_path: str | Path,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    settings = build_settings_from_pilates(pilates_settings=pilates_settings_path)
    payload = settings.to_dict()
    if output_path is not None:
        write_structured_file(output_path, payload)
    return payload


def write_settings(settings: ImpactsSettings, output_path: str | Path) -> Dict[str, Any]:
    payload = settings.to_dict()
    write_structured_file(output_path, payload)
    return payload
