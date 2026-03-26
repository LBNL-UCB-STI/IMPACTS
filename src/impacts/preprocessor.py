from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import geopandas as gpd
from .config.builders import build_runtime_config_from_runtime_yaml
from .contract_utils import copy_path
from .contract_utils import file_entry
from .contract_utils import is_remote_path
from .contract_utils import parquet_available
from .contract_utils import resolve_path
from .contract_utils import write_structured_file
from .defaults import DEFAULT_HOUSEHOLDS_COLUMNS
from .defaults import DEFAULT_PERSONS_COLUMNS
from .manifest_models import InputsManifest


CONTRACT_VERSION = "1"


def _parse_epsg(value: Any) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(
            "geography.local_crs (output EPSG) must be set in the runtime config. "
            "Example: local_crs: 26910"
        )
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if ":" in text:
        _, _, suffix = text.rpartition(":")
        text = suffix
    return int(text)


def _required_local_path(path: Optional[str], label: str) -> str:
    if not path:
        raise ValueError(f"Missing required config path: {label}")
    if is_remote_path(path):
        raise ValueError(f"{label} must be a local path during preprocessing: {path}")
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(resolved)


def _optional_local_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if is_remote_path(path):
        return path
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Configured path not found: {path}")
    return str(resolved)


def _infer_vector_epsg(path: Optional[str]) -> Optional[int]:
    if not path or is_remote_path(path):
        return None
    target = Path(path)
    if not target.exists():
        return None
    try:
        if target.suffix.lower() == ".parquet":
            gdf = gpd.read_parquet(target)
        else:
            gdf = gpd.read_file(target)
    except Exception:
        return None
    if gdf.crs is None:
        return None
    try:
        return gdf.crs.to_epsg()
    except Exception:
        return None


def _read_vector(path: str) -> gpd.GeoDataFrame:
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        return gpd.read_parquet(target)
    return gpd.read_file(target)


def _write_vector(gdf: gpd.GeoDataFrame, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        gdf.to_parquet(target, index=False)
    else:
        gdf.to_file(target)


def _ensure_grid_cell_id(
    staged_path: str,
    cell_id_col: str,
    source_col: Optional[str] = None,
) -> tuple[str, str]:
    """Ensure the staged grid file has a ``cell_id_col`` column.

    If ``source_col`` is given, it must exist in the file — raises otherwise.
    If ``source_col`` is not given, ``cell_id_col`` is created from row numbers.
    When the staged input is a shapefile and a rewrite is needed, the normalized
    contract copy is written as parquet so full field names are preserved.

    Returns ``(normalized_path, cell_id_col)``.
    """
    gdf = _read_vector(staged_path)
    if cell_id_col in gdf.columns:
        return staged_path, cell_id_col
    if source_col:
        if source_col not in gdf.columns:
            raise ValueError(
                f"Configured grid_id column '{source_col}' not found in {staged_path}. "
                f"Available columns: {list(gdf.columns)}"
            )
        gdf[cell_id_col] = gdf[source_col]
    else:
        gdf[cell_id_col] = range(len(gdf))
    normalized_path = staged_path
    if Path(staged_path).suffix.lower() != ".parquet":
        normalized_path = str(Path(staged_path).with_suffix(".parquet"))
    _write_vector(gdf, normalized_path)
    return normalized_path, cell_id_col


def _stage_local_input(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    key: str,
    source_path: str,
    relative_target: str,
    optional: bool = False,
) -> str:
    destination = input_root / relative_target
    staged = copy_path(source_path, destination)
    manifest_inputs[key] = file_entry(
        kind="local",
        path=source_path,
        staged_path=staged,
        optional=optional,
    )
    return str(staged)


def _stage_optional_input(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    key: str,
    source_path: Optional[str],
    relative_target: str,
) -> Optional[str]:
    if not source_path:
        return None
    if is_remote_path(source_path):
        manifest_inputs[key] = {
            "kind": "remote",
            "source_path": source_path,
            "staged_path": None,
            "optional": True,
            "exists": True,
        }
        return source_path
    return _stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key=key,
        source_path=source_path,
        relative_target=relative_target,
        optional=True,
    )


def _prepared_table_target(input_root: Path, stem: str) -> Path:
    suffix = ".parquet" if parquet_available() else ".csv.gz"
    return input_root / "skims" / f"{stem}{suffix}"


def _stage_county_boundaries(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    state_fips: str,
    county_fips_codes: list[str],
    year: int,
    area_name: str,
    target_epsg: int,
) -> str:
    from osm_chordify.utils.data_collection import collect_geographic_boundaries

    county_gdf = collect_geographic_boundaries(
        state_fips_code=str(state_fips),
        county_fips_codes=[str(code) for code in county_fips_codes],
        year=int(year),
        area_name=str(area_name),
        geo_level="county",
        work_dir=str(input_root / "county"),
        target_epsg=int(target_epsg),
    )
    destination = input_root / "county" / "county_boundaries.gpkg"
    _write_vector(county_gdf, str(destination))
    manifest_inputs["county_boundaries"] = file_entry(
        kind="local",
        path=str(destination),
        staged_path=str(destination),
        optional=False,
    )
    return str(destination)


def build_inputs_manifest(
    runtime_config_path: str | Path,
    staging_dir: str | Path,
) -> Dict[str, Any]:
    config_path = Path(runtime_config_path).resolve()
    runtime_config = build_runtime_config_from_runtime_yaml(config_path)
    geography = runtime_config.shared_context.geography
    inputs = runtime_config.inputs
    processing = runtime_config.processing
    grid = processing.grid
    outputs = runtime_config.outputs

    workspace_root = Path(staging_dir).resolve()
    input_root = workspace_root / "input"
    input_root.mkdir(parents=True, exist_ok=True)

    manifest_inputs: Dict[str, Any] = {}
    runtime_copy = _stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="runtime_config",
        source_path=str(config_path),
        relative_target="config/runtime.yaml",
    )

    skims_input_path = _optional_local_path(resolve_path(inputs.emissions_skims, config_path))
    if not skims_input_path:
        raise ValueError("Preprocess requires inputs.emissions_skims")

    staged_skims_input = _stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="skims_input",
        source_path=skims_input_path,
        relative_target=f"skims/{Path(skims_input_path).name}",
        optional=True,
    )

    from impacts.emissions.emissions_grid_mapping import annualize_prepared_skims_for_grid_allocation
    from impacts.emissions.emissions_grid_mapping import prepare_skims_for_grid_allocation

    prepared_grouped_skims_path = _prepared_table_target(input_root, "prepared_skims_grouped_for_grid_allocation")
    prepare_skims_for_grid_allocation(
        skims_path=staged_skims_input,
        output_path=str(prepared_grouped_skims_path),
        group_cols=list(processing.prepared_skims_group_cols),
        required_pollutants=list(processing.pollutants),
    )
    prepared_skims_path = _prepared_table_target(input_root, "prepared_skims_for_grid_allocation")
    annualize_prepared_skims_for_grid_allocation(
        prepared_skims_path=str(prepared_grouped_skims_path),
        output_path=str(prepared_skims_path),
        group_cols=list(processing.prepared_skims_group_cols),
        required_pollutants=list(processing.pollutants),
        annualization_days=float(processing.annualization_days),
    )
    manifest_inputs["prepared_skims_grouped"] = file_entry(
        kind="local",
        path=str(prepared_grouped_skims_path),
        staged_path=str(prepared_grouped_skims_path),
        optional=True,
    )
    manifest_inputs["prepared_skims_input"] = file_entry(
        kind="local",
        path=str(prepared_skims_path),
        staged_path=str(prepared_skims_path),
        optional=True,
    )
    staged_prepared_skims_input = str(prepared_skims_path)

    activity_corrections_path = resolve_path(inputs.activity_corrections, config_path)
    staged_activity_corrections = None
    if activity_corrections_path and not is_remote_path(activity_corrections_path):
        staged_activity_corrections = _stage_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="activity_corrections",
            source_path=activity_corrections_path,
            relative_target=f"activity/{Path(activity_corrections_path).name}",
            optional=True,
        )

    beam_network_source = _required_local_path(
        resolve_path(inputs.beam_network, config_path),
        "inputs.beam_network",
    )
    staged_link_lengths = _stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="link_lengths",
        source_path=beam_network_source,
        relative_target=f"network/{Path(beam_network_source).name}",
        optional=True,
    )

    osm_source = resolve_path(inputs.osm_links, config_path)
    osm_pbf_source = resolve_path(inputs.osm_pbf, config_path)
    if osm_source:
        osm_source = _required_local_path(osm_source, "inputs.osm_links")
    elif osm_pbf_source:
        osm_source = _required_local_path(osm_pbf_source, "inputs.osm_pbf")
    else:
        raise ValueError("Either inputs.osm_links or inputs.osm_pbf is required.")

    inmap_grid_source = _required_local_path(
        resolve_path(grid.inmap_grid_path, config_path),
        "processing.grid.inmap_grid_path",
    )
    aermod_grid_source = _optional_local_path(resolve_path(grid.aermod_grid_path, config_path))
    local_output_epsg = _parse_epsg(geography.local_crs)
    inmap_grid_epsg = (
        grid.inmap_grid_epsg
        or _infer_vector_epsg(inmap_grid_source)
        or local_output_epsg
    )
    aermod_grid_epsg = (
        grid.aermod_grid_epsg
        or _infer_vector_epsg(aermod_grid_source)
        or local_output_epsg
    )

    staged_osm = _stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="osm_links",
        source_path=osm_source,
        relative_target=f"osm/{Path(osm_source).name}",
    )
    staged_beam_network = _stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="beam_network",
        source_path=beam_network_source,
        relative_target=f"network/{Path(beam_network_source).name}",
    )
    staged_inmap_grid = _stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="inmap_grid",
        source_path=inmap_grid_source,
        relative_target=f"inmap_grid/{Path(inmap_grid_source).name}",
    )
    staged_aermod_grid = None
    resolved_aermod_grid_id = None
    if aermod_grid_source:
        staged_aermod_grid = _stage_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="aermod_grid",
            source_path=aermod_grid_source,
            relative_target=f"aermod_grid/{Path(aermod_grid_source).name}",
            optional=True,
        )
        staged_aermod_grid, resolved_aermod_grid_id = _ensure_grid_cell_id(
            staged_aermod_grid,
            "srv_cell_id",
            source_col=grid.aermod_grid_id,
        )
        manifest_inputs["aermod_grid"]["staged_path"] = staged_aermod_grid
    staged_inmap_grid, resolved_inmap_grid_id = _ensure_grid_cell_id(
        staged_inmap_grid,
        "srm_cell_id",
        source_col=grid.inmap_grid_id,
    )
    manifest_inputs["inmap_grid"]["staged_path"] = staged_inmap_grid
    staged_county_boundaries = _stage_county_boundaries(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        state_fips=geography.fips.state,
        county_fips_codes=list(geography.fips.counties),
        year=int(runtime_config.shared_context.start_year or 2023),
        area_name=runtime_config.shared_context.region or processing.county_area_name,
        target_epsg=int(local_output_epsg),
    )

    staged_isrm = _stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="isrm",
        source_path=resolve_path(inputs.isrm_zarr, config_path),
        relative_target="isrm",
    )
    staged_persons = _stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="persons",
        source_path=_optional_local_path(resolve_path(inputs.persons_asim_out, config_path)),
        relative_target="population/persons.csv",
    )
    staged_households = _stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="households",
        source_path=_optional_local_path(resolve_path(inputs.households_asim_out, config_path)),
        relative_target="population/households.csv",
    )
    staged_osm_pbf = _stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="osm_pbf",
        source_path=_optional_local_path(resolve_path(inputs.osm_pbf, config_path)),
        relative_target="osm/source.pbf",
    )
    staged_beam_mapdb = _stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="beam_mapdb",
        source_path=_optional_local_path(resolve_path(inputs.beam_mapdb, config_path)),
        relative_target="osm/beam.mapdb",
    )

    mapping_columns = {k: v for k, v in processing.mapping_columns.__dict__.items() if v}
    mapping_columns["grid_id"] = resolved_inmap_grid_id

    manifest: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "model": "impacts",
        "runtime_config_source": str(config_path),
        "staging_dir": str(workspace_root),
        "input_dir": str(input_root),
        "inputs_manifest_path": str(workspace_root / "inputs_manifest.yaml"),
        "maintained_execution_path": [
            "impacts.emissions.events_to_skims_emissions",
            "impacts.emissions.emissions_grid_mapping",
            "impacts.dispersion.isrm_dispersion",
            "impacts.network2grid.network_grid_clipping",
        ],
        "inputs": manifest_inputs,
        "pipeline": {
            "events_path": None,
            "skims_input_path": staged_skims_input,
            "prepared_skims_input_path": staged_prepared_skims_input,
            "link_length_path": staged_link_lengths,
            "rates_dir": None,
            "mapping_input_path": None,
            "use_precomputed_mapping": False,
            "osm_links_path": staged_osm,
            "osm_pbf_path": staged_osm_pbf,
            "beam_mapdb_path": staged_beam_mapdb,
            "beam_network_path": staged_beam_network,
            "inmap_grid_path": staged_inmap_grid,
            "aermod_grid_path": staged_aermod_grid,
            "isrm_url": staged_isrm,
            "iterations": 0,
            "use_rates": False,
            "beam_osm_id_col": processing.beam_osm_id_col,
            "beam_length_col": processing.beam_length_col,
            "beam_osm_epsg": int(local_output_epsg),
            "inmap_grid_epsg": int(inmap_grid_epsg),
            "aermod_grid_epsg": int(aermod_grid_epsg),
            "aermod_grid_id": resolved_aermod_grid_id,
            "output_epsg": int(local_output_epsg),
            "region": runtime_config.shared_context.region,
            "start_year": runtime_config.shared_context.start_year,
            "county_state_fips": geography.fips.state,
            "county_fips_codes": list(geography.fips.counties),
            "county_area_name": runtime_config.shared_context.region or processing.county_area_name,
            "county_boundaries_path": staged_county_boundaries,
            "concentration_factor": float(processing.dispersion.concentration_factor),
            "include_bc": bool(processing.dispersion.include_bc),
            "include_health": bool(processing.dispersion.include_health),
            "mapping_columns": mapping_columns,
            "prepared_skims_grouped_path": str(prepared_grouped_skims_path),
            "prepared_skims_group_cols": list(processing.prepared_skims_group_cols),
            "prepared_pollutants": list(processing.pollutants),
            "activity_corrections_path": staged_activity_corrections,
            "activity_corrections_columns": processing.activity_corrections_columns.__dict__.copy(),
            "annualization_days": float(processing.annualization_days),
        },
        "pilates_contract": {
            "stage": "terminal_postprocessing",
            "runs_after": ["beam", "supply_demand_loop"],
            "upstream_dependency_only": True,
            "publishes_final_artifact_only": True,
            "local_input_dir": str(input_root),
            "local_output_dir": str(outputs.output_dir),
            "container_input_dir": "/input",
            "container_output_dir": "/output",
            "entrypoint": "python -m impacts run",
            "command_template": "python -m impacts run --input-manifest {input_manifest} --output-dir {output_dir}",
            "canonical_output_filenames": ["impacts_exposure_table.parquet", "impacts_exposure_table.csv.gz"],
            "manifest_filenames": ["inputs_manifest.yaml", "run_manifest.yaml", "postprocess_manifest.yaml"],
        },
        "population_inputs": {
            "persons_path": staged_persons,
            "households_path": staged_households,
            "persons_columns": list(DEFAULT_PERSONS_COLUMNS),
            "households_columns": list(DEFAULT_HOUSEHOLDS_COLUMNS),
        },
        "notes": [
            "Only maintained modules are part of this contract.",
            "ActivitySim population integration is currently best-effort and relies on staged population tables carrying usable cell ids.",
            f"Runtime config staged at {runtime_copy}",
        ],
    }
    return manifest


def preprocess_workflow(
    runtime_config_path: str | Path,
    staging_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    manifest = build_inputs_manifest(runtime_config_path=runtime_config_path, staging_dir=staging_dir)
    output_manifest = Path(manifest_path) if manifest_path else Path(manifest["inputs_manifest_path"])
    manifest["inputs_manifest_path"] = str(output_manifest)
    typed_manifest = InputsManifest.from_dict(manifest)
    write_structured_file(output_manifest, typed_manifest.to_dict())
    return typed_manifest.to_dict()
