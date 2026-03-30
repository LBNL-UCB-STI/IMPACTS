from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict
from .config.runtime_builder import build_runtime_config_from_runtime_yaml
from .manifest.file_ops import resolve_path
from .manifest.file_ops import write_structured_file
from .manifest.schema import InputsManifest
from .common import infer_vector_epsg
from .common import parse_epsg
from .common import required_local_path
from .common import resolve_beam_network_local_path
from .common import resolve_osm_pbf_local_path
from .common import stage_local_input


CONTRACT_VERSION = "1"
logger = logging.getLogger(__name__)
PERSONS_COLUMNS = [
    "household_id",
    "cell_id",
    "age",
    "sex",
    "income",
]
HOUSEHOLDS_COLUMNS = [
    "household_id",
    "cell_id",
    "income",
    "income_category",
]


def _validate_configured_local_path(
    path: str | None,
    label: str,
    *,
    allow_remote: bool = False,
) -> str | None:
    if not path:
        return None
    if allow_remote and "://" in str(path):
        return path
    return required_local_path(path, label)


def build_inputs_manifest(
    runtime_config_path: str | Path,
    staging_dir: str | Path,
) -> Dict[str, Any]:
    config_path = Path(runtime_config_path).resolve()
    runtime_config = build_runtime_config_from_runtime_yaml(config_path)
    geography = runtime_config.shared.geography
    emissions = runtime_config.impacts.emissions
    inmap = runtime_config.impacts.dispersions.inmap

    workspace_root = Path(staging_dir).resolve()
    input_root = workspace_root / "staged"
    input_root.mkdir(parents=True, exist_ok=True)

    manifest_inputs: Dict[str, Any] = {}
    runtime_copy = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="runtime_config",
        source_path=str(config_path),
        relative_target="config/runtime.yaml",
    )
    from .preprocessing.step1_stage_inputs import run as preprocess_step1
    from .preprocessing.step2_prepare_grids import run as preprocess_step2
    simulation_network_folder = required_local_path(
        resolve_path(emissions.simulation_network_folder, config_path),
        "impacts.emissions.simulation_network_folder",
    )
    beam_network_source = resolve_beam_network_local_path(simulation_network_folder)
    _validate_configured_local_path(
        resolve_path(emissions.emissions_rates_folder, config_path),
        "impacts.emissions.emissions_rates_folder",
    )
    osm_network_folder = required_local_path(
        resolve_path(emissions.osm_network_folder, config_path),
        "impacts.emissions.osm_network_folder",
    )
    osm_source = resolve_osm_pbf_local_path(osm_network_folder)
    if not osm_source:
        raise ValueError("impacts.emissions.osm_network_folder is configured but no local .osm.pbf file could be found under it.")

    inmap_grid_source = required_local_path(
        resolve_path(inmap.grid_path, config_path),
        "impacts.dispersions.inmap.grid_path",
    )
    local_output_epsg = parse_epsg(geography.local_crs)
    inmap_grid_epsg = (
        inmap.grid_epsg
        or infer_vector_epsg(inmap_grid_source)
        or local_output_epsg
    )

    step1_outputs = preprocess_step1(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        impacts=runtime_config.impacts,
        config_path=config_path,
        beam_network_source=beam_network_source,
        osm_source=osm_source,
        inmap_grid_source=inmap_grid_source,
    )
    staged_osm = step1_outputs["staged_osm"]
    staged_activity_totals = step1_outputs["staged_activity_totals"]
    staged_isrm = step1_outputs["staged_isrm"]
    staged_isrm_nox_to_no2_matrix_npz = step1_outputs["staged_isrm_nox_to_no2_matrix_npz"]

    step2_outputs = preprocess_step2(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        runtime_config=runtime_config,
        inmap=inmap,
        local_output_epsg=int(local_output_epsg),
    )
    staged_inmap_grid = step2_outputs["staged_inmap_grid"]
    staged_aermod_grid = step2_outputs["staged_aermod_grid"]
    resolved_inmap_grid_id = step2_outputs["resolved_inmap_grid_id"]
    resolved_aermod_grid_id = step2_outputs["resolved_aermod_grid_id"]
    mapping_columns = {**emissions.mapping_columns, "grid_id": resolved_inmap_grid_id}

    manifest: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "model": "impacts",
        "runtime_config_source": str(config_path),
        "staging_dir": str(workspace_root),
        "input_dir": str(input_root),
        "inputs_manifest_path": str(workspace_root / "inputs_manifest.yaml"),
        "maintained_execution_path": [
            "impacts.preprocessing.step3_integrate_grids",
            "impacts.runtime.step1_process_emissions",
            "impacts.runtime.step2_compute_inmap_concentrations",
        ],
        "inputs": manifest_inputs,
        "pipeline": {
            "inmap_grid_path": staged_inmap_grid,
            "aermod_grid_path": staged_aermod_grid,
            "isrm_url": staged_isrm,
            "isrm_nox_to_no2_matrix_npz_path": staged_isrm_nox_to_no2_matrix_npz,
            "iterations": 0,
            "beam_osm_id_col": emissions.beam_osm_id_col,
            "beam_length_col": emissions.beam_length_col,
            "beam_osm_epsg": int(local_output_epsg),
            "inmap_grid_epsg": int(local_output_epsg),
            "aermod_grid_epsg": int(local_output_epsg),
            "aermod_grid_id": resolved_aermod_grid_id,
            "output_epsg": int(local_output_epsg),
            "simulation_network_folder": simulation_network_folder,
            "region": runtime_config.run.region,
            "start_year": runtime_config.run.start_year,
            "county_state_fips": geography.fips.state,
            "county_fips_codes": list(geography.fips.counties),
            "concentration_factor": float(emissions.concentration_factor),
            "mapping_columns": mapping_columns,
            "prepared_skims_group_cols": list(emissions.prepared_skims_group_cols),
            "pollutants": list(emissions.pollutants),
            "pollutants_map": dict(emissions.pollutants_map),
            "activity_totals_file": staged_activity_totals,
            "activity_totals_columns": dict(emissions.activity_totals_columns),
            "annualization_days": float(emissions.annualization_days),
            "population_sample": float(emissions.population_sample),
        },
        "pilates_contract": {
            "stage": "terminal_postprocessing",
            "runs_after": ["beam", "supply_demand_loop"],
            "upstream_dependency_only": True,
            "publishes_final_artifact_only": True,
            "local_input_dir": str(input_root),
            "local_output_dir": str(runtime_config.impacts.local_output_folder),
            "container_input_dir": "/input",
            "container_output_dir": "/output",
            "entrypoint": "python -m impacts run",
            "command_template": "python -m impacts run --input-manifest {input_manifest} --output-dir {output_dir}",
            "canonical_output_filenames": ["impacts_exposure_table.parquet", "impacts_exposure_table.csv.gz"],
            "manifest_filenames": ["inputs_manifest.yaml", "run_manifest.yaml", "postprocess_manifest.yaml"],
        },
        "population_inputs": {},
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=False,
    )
    manifest = build_inputs_manifest(runtime_config_path=runtime_config_path, staging_dir=staging_dir)
    output_manifest = Path(manifest_path) if manifest_path else Path(manifest["inputs_manifest_path"])
    manifest["inputs_manifest_path"] = str(output_manifest)
    typed_manifest = InputsManifest.from_dict(manifest)
    write_structured_file(output_manifest, typed_manifest.to_dict())
    return typed_manifest.to_dict()
