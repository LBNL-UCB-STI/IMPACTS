from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any
from typing import Dict
from typing import Optional


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


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
    latest_iter = -1
    latest_skims: Optional[Path] = None
    for candidate in iters_dir.glob("it.*/*.skimsEmissionsTotals.csv.gz"):
        match = re.search(r"/it\.(\d+)/(\d+)\.skimsEmissionsTotals\.csv\.gz$", str(candidate))
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


def _derive_beam_inputs(pilates_settings: Dict[str, Any]) -> Dict[str, Any]:
    beam = pilates_settings.get("beam", {}) or {}
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

    if local_output_folder:
        latest_iters_dir = _find_latest_iters_dir(local_output_folder)
        if latest_iters_dir:
            run_root = latest_iters_dir.parent
            derived["beam_network"] = str(run_root / "network.csv.gz")
            latest_skims = _find_latest_iteration_skims(latest_iters_dir)
            if latest_skims:
                derived["emissions_skims"] = str(latest_skims)
        else:
            derived["beam_network"] = _join_path(local_output_folder, "network.csv.gz")

    if local_input_folder and router_directory:
        router_path = PurePosixPath(router_directory)
        router_name = router_path.name
        if router_name.endswith(".osm.pbf"):
            osm_pbf = PurePosixPath(local_input_folder) / router_path
        else:
            osm_pbf = PurePosixPath(local_input_folder) / router_path / f"{router_name}.osm.pbf"
        derived["osm_pbf"] = str(osm_pbf)

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


def _normalize_runtime_overrides(runtime_overrides: Dict[str, Any]) -> Dict[str, Any]:
    if "processing" in runtime_overrides or "inputs" in runtime_overrides:
        return runtime_overrides

    emissions = runtime_overrides.get("emissions", {}) or {}
    dispersions = runtime_overrides.get("dispersions", {}) or {}
    outputs = runtime_overrides.get("outputs", {}) or {}
    inmap = dispersions.get("inmap", {}) or {}
    aermod = dispersions.get("aermod", {}) or {}

    isrm_directory = str(inmap.get("isrm_zarr_directory") or "").strip()
    isrm_s3bucket = str(inmap.get("isrm_zarr_s3bucket") or "").strip() or None
    isrm_zarr = None
    if isrm_directory and Path(isrm_directory).exists():
        isrm_zarr = isrm_directory
    else:
        isrm_zarr = isrm_s3bucket

    normalized = {
        "emissions": {
            "annualization_days": emissions.get("annualization_days"),
            "activity_correction_factors_file": emissions.get("activity_correction_factors_file"),
            "pollutants": emissions.get("pollutants"),
        },
        "dispersions": {
            "inmap": {
                "isrm_zarr": isrm_zarr,
                "grid_path": inmap.get("grid_path"),
                "grid_epsg": inmap.get("grid_epsg"),
                "grid_id": inmap.get("grid_id"),
            },
            "aermod": {
                "grid_path": aermod.get("grid_path"),
                "grid_epsg": aermod.get("grid_epsg"),
                "grid_id": aermod.get("grid_id"),
            },
        },
        "outputs": {
            "output_dir": outputs.get("output_dir"),
        },
    }
    return normalized


def build_runtime_payload_from_pilates(
    pilates_settings: Dict[str, Any],
    impacts_overlay: Dict[str, Any],
) -> Dict[str, Any]:
    run = pilates_settings.get("run", {}) or {}
    shared = pilates_settings.get("shared", {}) or {}
    geography = shared.get("geography", {}) or {}
    legacy_geography = _legacy_geography(pilates_settings)
    if not geography:
        geography = legacy_geography
    fips = geography.get("FIPS", {}) or {}
    zones = geography.get("zones", {}) or {}
    alternative_zones = geography.get("alternative_zones", {}) or {}
    skims = shared.get("skims", {}) or {}
    derived_inputs = _derive_population_inputs(pilates_settings)
    derived_inputs = _deep_merge(derived_inputs, _derive_beam_inputs(pilates_settings))
    derived_outputs = _derive_output_settings(pilates_settings)

    impacts_section = impacts_overlay.get("impacts", {}) or {}
    runtime_overrides = _normalize_runtime_overrides(
        impacts_section.get("runtime_overrides", {}) or {}
    )

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

    return _deep_merge(derived, runtime_overrides)
