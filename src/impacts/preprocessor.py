from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict
from .config.runtime_builder import build_runtime_config_from_runtime_yaml
from .manifest.file_ops import file_entry
from .manifest.file_ops import resolve_path
from .manifest.file_ops import write_structured_file
from .manifest.schema import InputsManifest
from .workflow_preprocess.common import infer_vector_epsg
from .workflow_preprocess.common import optional_local_path
from .workflow_preprocess.common import parse_epsg
from .workflow_preprocess.common import required_local_path
from .workflow_preprocess.common import resolve_beam_network_local_path
from .workflow_preprocess.common import resolve_emissions_skims_local_path
from .workflow_preprocess.common import resolve_osm_pbf_local_path
from .workflow_preprocess.common import stage_local_input


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

    from .workflow_preprocess.step1_prepare_skims import run as preprocess_step1
    from .workflow_preprocess.step2_stage_network_mapping import run as preprocess_step2
    from .workflow_preprocess.step3_stage_spatial_support import run as preprocess_step3
    from .workflow_preprocess.step4_stage_optional_inputs import run as preprocess_step4

    simulation_network_folder = required_local_path(
        resolve_path(inputs.simulation_network_folder, config_path),
        "inputs.simulation_network_folder",
    )
    beam_network_source = resolve_beam_network_local_path(simulation_network_folder)
    skims_input_source = resolve_emissions_skims_local_path(simulation_network_folder)
    osm_network_folder = required_local_path(
        resolve_path(inputs.osm_network_folder, config_path),
        "inputs.osm_network_folder",
    )
    osm_source = resolve_osm_pbf_local_path(osm_network_folder)
    if not osm_source:
        raise ValueError("inputs.osm_network_folder is configured but no local .osm.pbf file could be found under it.")

    inmap_grid_source = required_local_path(
        resolve_path(grid.inmap_grid_path, config_path),
        "processing.grid.inmap_grid_path",
    )
    aermod_grid_source = optional_local_path(resolve_path(grid.aermod_grid_path, config_path))
    local_output_epsg = parse_epsg(geography.local_crs)
    inmap_grid_epsg = (
        grid.inmap_grid_epsg
        or infer_vector_epsg(inmap_grid_source)
        or local_output_epsg
    )
    aermod_grid_epsg = (
        grid.aermod_grid_epsg
        or infer_vector_epsg(aermod_grid_source)
        or local_output_epsg
    )

    step1_outputs = preprocess_step1(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        processing=processing,
        skims_input_source=skims_input_source,
        network_path=beam_network_source,
    )

    step2_outputs = preprocess_step2(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        processing=processing,
        beam_network_source=beam_network_source,
        osm_source=osm_source,
        local_output_epsg=int(local_output_epsg),
    )
    staged_osm = step2_outputs["staged_osm"]

    step3_outputs = preprocess_step3(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        runtime_config=runtime_config,
        processing=processing,
        grid=grid,
        inmap_grid_source=inmap_grid_source,
        aermod_grid_source=aermod_grid_source,
        local_output_epsg=int(local_output_epsg),
    )
    staged_inmap_grid = step3_outputs["staged_inmap_grid"]
    staged_aermod_grid = step3_outputs["staged_aermod_grid"]
    resolved_inmap_grid_id = step3_outputs["resolved_inmap_grid_id"]
    resolved_aermod_grid_id = step3_outputs["resolved_aermod_grid_id"]

    inputs_resolved = runtime_config.inputs
    step4_outputs = preprocess_step4(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        inputs=inputs_resolved,
        config_path=config_path,
    )
    staged_activity_totals = step4_outputs["staged_activity_totals"]
    staged_isrm = step4_outputs["staged_isrm"]
    staged_isrm_nox_to_no2_matrix_npz = step4_outputs["staged_isrm_nox_to_no2_matrix_npz"]
    staged_persons = step4_outputs["staged_persons"]
    staged_households = step4_outputs["staged_households"]
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
            "impacts.workflow_runtime.step3_inmap_dispersion",
            "impacts.workflow_runtime.step1_grid_intersection",
            "impacts.workflow_runtime.step2_emissions_distribution",
        ],
        "inputs": manifest_inputs,
        "pipeline": {
            "inmap_grid_path": staged_inmap_grid,
            "aermod_grid_path": staged_aermod_grid,
            "isrm_url": staged_isrm,
            "isrm_nox_to_no2_matrix_npz_path": staged_isrm_nox_to_no2_matrix_npz,
            "iterations": 0,
            "beam_osm_id_col": processing.beam_osm_id_col,
            "beam_length_col": processing.beam_length_col,
            "beam_osm_epsg": int(local_output_epsg),
            "inmap_grid_epsg": int(local_output_epsg),
            "aermod_grid_epsg": int(local_output_epsg),
            "aermod_grid_id": resolved_aermod_grid_id,
            "output_epsg": int(local_output_epsg),
            "region": runtime_config.shared_context.region,
            "start_year": runtime_config.shared_context.start_year,
            "county_state_fips": geography.fips.state,
            "county_fips_codes": list(geography.fips.counties),
            "county_area_name": runtime_config.shared_context.region or processing.county_area_name,
            "concentration_factor": float(processing.concentrations.concentration_factor),
            "mapping_columns": mapping_columns,
            "prepared_skims_group_cols": list(processing.prepared_skims_group_cols),
            "pollutants": list(processing.pollutants),
            "pollutants_map": dict(processing.pollutants_map),
            "activity_totals_file": staged_activity_totals,
            "activity_totals_columns": dict(processing.activity_totals_columns),
            "annualization_days": float(processing.annualization_days),
            "population_sample": float(processing.population_sample),
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
            "persons_columns": list(PERSONS_COLUMNS),
            "households_columns": list(HOUSEHOLDS_COLUMNS),
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
