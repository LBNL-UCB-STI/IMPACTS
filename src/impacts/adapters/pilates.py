from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any
from typing import Dict
from typing import Optional

from impacts.manifest.file_ops import write_structured_file


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


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


def _first_present_dict(*candidates: Any) -> Dict[str, Any]:
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _legacy_region(pilates_settings: Dict[str, Any]) -> Optional[str]:
    return str(pilates_settings.get("region") or "").strip() or None


def _legacy_geography(pilates_settings: Dict[str, Any]) -> Dict[str, Any]:
    region = _legacy_region(pilates_settings)
    fips_map = pilates_settings.get("FIPS", {}) or {}
    local_crs_map = pilates_settings.get("local_crs", {}) or {}
    if not region:
        return {}
    return {
        "FIPS": fips_map.get(region, {}) or {},
        "local_crs": local_crs_map.get(region),
    }


def _join_path(base_dir: str, filename: str) -> str:
    return str(PurePosixPath(base_dir) / filename)


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


def _find_preferred_file(root: str, names: list[str]) -> Optional[str]:
    path = Path(root)
    if not path.exists():
        return None
    for name in names:
        direct = path / name
        if direct.exists():
            return str(direct)
    for name in names:
        matches = sorted(path.rglob(name))
        if matches:
            return str(matches[0])
    return None


def _find_first_matching(root: str, pattern: str) -> Optional[str]:
    path = Path(root)
    if not path.exists():
        return None
    matches = sorted(path.glob(pattern))
    if matches:
        return str(matches[0])
    recursive = sorted(path.rglob(pattern))
    if recursive:
        return str(recursive[0])
    return None


def _resolve_search_root(path_str: str, *source_roots: Optional[str]) -> str:
    if not path_str:
        return path_str
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return str(path)
    for source_root in source_roots:
        if not source_root:
            continue
        candidate = Path(source_root) / path
        if candidate.exists():
            return str(candidate)
    return path_str


def _derive_output_settings(pilates_settings: Dict[str, Any]) -> Dict[str, Any]:
    run = pilates_settings.get("run", {}) or {}
    output_directory = str(run.get("output_directory") or "").strip()
    output_run_name = str(run.get("output_run_name") or "").strip()
    if not output_directory:
        return {}
    output_root = PurePosixPath(output_directory)
    if output_run_name:
        output_root = output_root / output_run_name
    return {"output_dir": str(output_root / "impacts")}


def _find_latest_iters_dir(local_output_folder: str) -> Optional[Path]:
    root = Path(local_output_folder)
    if not root.exists():
        return None
    candidates = [path for path in root.rglob("ITERS") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _find_latest_iteration_skims(iters_dir: Path) -> Optional[Path]:
    patterns = [
        ("*.skimsEmissionsTotals.csv.gz", r"/it\.(\d+)/(\d+)\.skimsEmissionsTotals\.csv\.gz$"),
        ("*.skimsEmissions.csv.gz", r"/it\.(\d+)/(\d+)\.skimsEmissions\.csv\.gz$"),
        ("*.skimsEmissions.parquet", r"/it\.(\d+)/(\d+)\.skimsEmissions\.parquet$"),
    ]
    latest_iter = -1
    latest_skims: Optional[Path] = None
    for glob_pattern, regex_pattern in patterns:
        for candidate in iters_dir.glob(f"it.*/{glob_pattern}"):
            match = re.search(regex_pattern, str(candidate))
            if not match:
                continue
            dir_iter = int(match.group(1))
            file_iter = int(match.group(2))
            if dir_iter != file_iter:
                continue
            if dir_iter > latest_iter:
                latest_iter = dir_iter
                latest_skims = candidate
    return latest_skims


def _find_latest_iteration_events(iters_dir: Path) -> Optional[Path]:
    patterns = [
        ("*.events.csv.gz", r"/it\.(\d+)/(\d+)\.events\.csv\.gz$"),
        ("*.events.parquet", r"/it\.(\d+)/(\d+)\.events\.parquet$"),
    ]
    latest_iter = -1
    latest_events: Optional[Path] = None
    for glob_pattern, regex_pattern in patterns:
        for candidate in iters_dir.glob(f"it.*/{glob_pattern}"):
            match = re.search(regex_pattern, str(candidate))
            if not match:
                continue
            dir_iter = int(match.group(1))
            file_iter = int(match.group(2))
            if dir_iter != file_iter:
                continue
            if dir_iter > latest_iter:
                latest_iter = dir_iter
                latest_events = candidate
    return latest_events


def _derive_beam_inputs(
    pilates_settings: Dict[str, Any],
    impacts_section: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    beam = pilates_settings.get("beam", {}) or {}
    impacts_section = impacts_section or {}
    emissions = impacts_section.get("emissions", {}) or {}
    local_input_folder = str(
        beam.get("local_input_folder") or pilates_settings.get("beam_local_input_folder") or ""
    ).strip()
    local_output_folder = str(
        beam.get("local_output_folder") or pilates_settings.get("beam_local_output_folder") or ""
    ).strip()
    router_directory = str(
        beam.get("router_directory") or pilates_settings.get("beam_router_directory") or ""
    ).strip()
    derived: Dict[str, Any] = {}
    search_roots = [
        _lookup_dotted(impacts_section, "__source_root__"),
        _lookup_dotted(pilates_settings, "__source_root__"),
    ]

    simulation_network_folder = str(
        emissions.get("simulation_network_folder") or local_output_folder or ""
    ).strip()
    osm_network_folder = str(
        emissions.get("osm_network_folder") or impacts_section.get("local_input_folder") or ""
    ).strip()
    emissions_rates_folder = str(emissions.get("emissions_rates_folder") or "").strip()
    simulation_network_folder = _resolve_search_root(simulation_network_folder, *search_roots)
    osm_network_folder = _resolve_search_root(osm_network_folder, *search_roots)
    emissions_rates_folder = _resolve_search_root(emissions_rates_folder, *search_roots)

    if simulation_network_folder:
        derived["simulation_network_folder"] = simulation_network_folder
    elif local_output_folder:
        derived["simulation_network_folder"] = _resolve_search_root(local_output_folder, *search_roots)

    if osm_network_folder:
        derived["osm_network_folder"] = osm_network_folder
    elif local_input_folder and router_directory:
        router_path = PurePosixPath(router_directory)
        router_name = router_path.name
        if router_name.endswith(".osm.pbf"):
            osm_root = PurePosixPath(local_input_folder) / router_path.parent
        else:
            osm_root = PurePosixPath(local_input_folder) / router_path
        derived["osm_network_folder"] = str(osm_root)

    if emissions_rates_folder:
        derived["emissions_rates_folder"] = emissions_rates_folder

    return derived


def _derive_population_inputs_from_legacy(pilates_settings: Dict[str, Any]) -> Dict[str, Any]:
    asim_output_dir = str(pilates_settings.get("asim_local_output_folder") or "").strip()
    if not asim_output_dir:
        return {}

    output_tables = pilates_settings.get("asim_output_tables", {}) or {}
    prefix = str(output_tables.get("prefix") or "").strip()
    tables = set(output_tables.get("tables", []) or [])

    derived: Dict[str, Any] = {}
    table_map = {
        "households_asim_out": "households",
        "persons_asim_out": "persons",
    }
    for runtime_key, table_name in table_map.items():
        if tables and table_name not in tables:
            continue
        derived[runtime_key] = _join_path(asim_output_dir, f"{prefix}{table_name}.csv.gz")
    return derived


def _derive_population_inputs(pilates_settings: Dict[str, Any]) -> Dict[str, Any]:
    shared = pilates_settings.get("shared", {}) or {}
    shared_population = shared.get("population", {}) or {}
    activitysim = pilates_settings.get("activitysim", {}) or {}
    urbansim = pilates_settings.get("urbansim", {}) or {}
    source = _first_present_dict(shared_population, activitysim, urbansim)

    derived_inputs: Dict[str, Any] = {}
    key_map = {
        "households_asim_out": ["households_asim_out", "households_path"],
        "persons_asim_out": ["persons_asim_out", "persons_path"],
    }
    for runtime_key, candidate_keys in key_map.items():
        for candidate_key in candidate_keys:
            value = source.get(candidate_key)
            if value:
                derived_inputs[runtime_key] = value
                break
    if not derived_inputs and activitysim:
        output_dir = str(activitysim.get("local_output_folder") or "").strip()
        output_tables = activitysim.get("output_tables", {}) or {}
        prefix = str(output_tables.get("prefix") or "").strip()
        tables = set(output_tables.get("tables", []) or [])
        file_format = str(activitysim.get("file_format") or "csv").strip().lower()
        suffix = ".parquet" if file_format == "parquet" else ".csv.gz"
        if output_dir:
            table_map = {
                "households_asim_out": "households",
                "persons_asim_out": "persons",
            }
            for runtime_key, table_name in table_map.items():
                if tables and table_name not in tables:
                    continue
                derived_inputs[runtime_key] = _join_path(output_dir, f"{prefix}{table_name}{suffix}")
    if not derived_inputs:
        derived_inputs.update(_derive_population_inputs_from_legacy(pilates_settings))
    return derived_inputs


def _extract_impacts_overrides(impacts_section: Dict[str, Any]) -> Dict[str, Any]:
    overrides = dict(impacts_section)
    overrides.pop("local_input_folder", None)
    overrides.pop("local_output_folder", None)
    return overrides


def build_runtime_payload_from_pilates(
    pilates_settings: Dict[str, Any],
    impacts_overlay: Dict[str, Any],
) -> Dict[str, Any]:
    run = pilates_settings.get("run", {}) or {}
    shared = pilates_settings.get("shared", {}) or {}
    impacts_section = _expand_impacts_placeholders(
        impacts_overlay.get("impacts", {}) or {},
        pilates_settings,
    )
    geography = shared.get("geography", {}) or {}
    legacy_geography = _legacy_geography(pilates_settings)
    if not geography:
        geography = legacy_geography
    fips = geography.get("FIPS", {}) or {}
    zones = geography.get("zones", {}) or {}
    alternative_zones = geography.get("alternative_zones", {}) or {}
    skims = shared.get("skims", {}) or {}
    derived_inputs = _derive_population_inputs(pilates_settings)
    derived_inputs = _deep_merge(
        derived_inputs,
        _derive_beam_inputs(pilates_settings, impacts_section),
    )
    impacts_output_dir = str(impacts_section.get("local_output_folder") or "").strip() or None
    derived_outputs = (
        {"output_dir": impacts_output_dir}
        if impacts_output_dir
        else _derive_output_settings(pilates_settings)
    )

    impacts_overrides = _extract_impacts_overrides(impacts_section)

    derived = {
        "shared": {
            "region": run.get("region") or _legacy_region(pilates_settings),
            "start_year": run.get("start_year", pilates_settings.get("start_year")),
            "geography": {
                "fips": {
                    "state": fips.get("state"),
                    "counties": list(fips.get("counties", []) or []),
                },
                "local_crs": _parse_epsg(geography.get("local_crs")),
                "zones": {
                    "zone_type": zones.get("zone_type"),
                    "source_file": zones.get("source_file"),
                    "source_crs": _parse_epsg(zones.get("source_crs")),
                    "canonical_id_col": zones.get("canonical_id_col"),
                    "activitysim_index_col": zones.get("activitysim_index_col"),
                }
                if zones
                else None,
                "alternative_zones": {
                    "zone_type": alternative_zones.get("zone_type"),
                    "source_file": alternative_zones.get("source_file"),
                    "source_crs": _parse_epsg(alternative_zones.get("source_crs")),
                    "canonical_id_col": alternative_zones.get("canonical_id_col"),
                    "activitysim_index_col": alternative_zones.get("activitysim_index_col"),
                }
                if alternative_zones
                else None,
            },
            "skims": {
                "zone_type": skims.get("zone_type"),
                "fname": skims.get("fname"),
                "origin_fname": skims.get("origin_fname"),
                "geoms_fname": skims.get("geoms_fname"),
                "geoms_index_col": skims.get("geoms_index_col"),
            }
            if skims
            else None,
        },
        "inputs": derived_inputs,
        "outputs": derived_outputs,
    }

    return _deep_merge(derived, impacts_overrides)


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
