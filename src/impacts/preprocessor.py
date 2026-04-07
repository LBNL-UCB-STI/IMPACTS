from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict
from .config.settings_builder import load_settings_from_yaml
from .manifest.file_ops import resolve_path
from .manifest.file_ops import write_structured_file
from .manifest.schema import InputsManifest
from .common import infer_vector_epsg
from .common import parse_epsg
from .common import register_local_input


CONTRACT_VERSION = "1"
logger = logging.getLogger(__name__)


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
    settings_path: str | Path,
    staging_dir: str | Path,
) -> Dict[str, Any]:
    config_path = Path(settings_path).resolve()
    settings = load_settings_from_yaml(config_path)
    geography = settings.shared.geography
    emissions = settings.impacts.emissions
    inmap = settings.impacts.dispersions.inmap
    aermod = settings.impacts.dispersions.aermod

    workspace_root = Path(staging_dir).resolve()
    input_root = Path(resolve_path(settings.impacts.local_input_folder, config_path)).resolve()
    input_root.mkdir(parents=True, exist_ok=True)

    manifest_inputs: Dict[str, Any] = {}
    settings_entry_path = register_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="settings",
        source_path=str(config_path),
    )
    from .preprocessing.step1_collect_inputs import run as preprocess_step1
    from .preprocessing.step2_prepare_grids import run as preprocess_step2
    local_output_epsg = parse_epsg(geography.local_crs)
    step1_outputs = preprocess_step1(
        manifest_inputs=manifest_inputs,
        settings=settings,
        input_root=input_root,
        config_path=config_path,
    )
    staged_activity_totals = step1_outputs["staged_activity_totals"]
    staged_isrm = step1_outputs["staged_isrm"]
    staged_isrm_nox_to_no2_ratios_file = step1_outputs["staged_isrm_nox_to_no2_ratios_file"]
    staged_asrv_patterns_file = step1_outputs["staged_asrv_patterns_file"]
    staged_inmap_grid_source = step1_outputs["staged_inmap_grid"]
    inmap_grid_epsg = None
    if inmap.enabled and staged_inmap_grid_source:
        inmap_grid_epsg = (
            infer_vector_epsg(staged_inmap_grid_source)
            or inmap.grid_epsg
        )
        if inmap_grid_epsg is None:
            raise ValueError(
                "Could not determine EPSG for impacts.dispersions.inmap.grid_path. "
                "Set impacts.dispersions.inmap.grid_epsg explicitly or provide CRS metadata in the file."
            )

    step2_outputs = preprocess_step2(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        settings=settings,
        inmap=inmap,
        local_output_epsg=int(local_output_epsg),
    )
    staged_inmap_grid = step2_outputs["staged_inmap_grid"]
    staged_aermod_grid = step2_outputs["staged_aermod_grid"]
    staged_aermod_full_grid = step2_outputs["staged_aermod_full_grid"]
    resolved_inmap_grid_id = step2_outputs["resolved_inmap_grid_id"]
    resolved_aermod_grid_id = step2_outputs["resolved_aermod_grid_id"]
    mapping_columns = dict(emissions.mapping_columns)
    if resolved_inmap_grid_id:
        mapping_columns["grid_id"] = resolved_inmap_grid_id
    asrv_patterns_epsg = None
    if aermod.enabled and staged_asrv_patterns_file:
        asrv_patterns_epsg = (
            infer_vector_epsg(staged_asrv_patterns_file)
            or aermod.asrv_patterns_epsg
        )
    if aermod.enabled and staged_asrv_patterns_file and asrv_patterns_epsg is None:
        raise ValueError(
            "Could not determine EPSG for impacts.dispersions.aermod.asrv_patterns_file. "
            "Set impacts.dispersions.aermod.asrv_patterns_epsg explicitly or provide CRS metadata in the file."
        )

    maintained_execution_path = [
        "impacts.preprocessing.step3_integrate_grids",
        "impacts.workflow.step1_process_emissions",
    ]
    if inmap.enabled:
        maintained_execution_path.append("impacts.workflow.step2_compute_inmap_concentrations")
    if aermod.enabled:
        maintained_execution_path.append("impacts.workflow.step3_compute_aermod_concentrations")

    manifest: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "model": "impacts",
        "settings_source": str(config_path),
        "staging_dir": str(workspace_root),
        "input_dir": str(input_root),
        "inputs_manifest_path": str(workspace_root / "inputs_manifest.yaml"),
        "maintained_execution_path": maintained_execution_path,
        "inputs": manifest_inputs,
        "pipeline": {
            "inmap_enabled": bool(inmap.enabled),
            "aermod_enabled": bool(aermod.enabled),
            "inmap_grid_path": staged_inmap_grid,
            "aermod_full_grid_path": staged_aermod_full_grid,
            "aermod_grid_path": staged_aermod_grid,
            "isrm_url": staged_isrm,
            "isrm_nox_to_no2_ratios_file": staged_isrm_nox_to_no2_ratios_file,
            "asrv_nox_to_no2_ratios_file": aermod.asrv_nox_to_no2_ratios_file,
            "asrv_patterns_file": staged_asrv_patterns_file,
            "asrv_patterns_epsg": int(asrv_patterns_epsg) if asrv_patterns_epsg is not None else None,
            "grid_size_meters": float(aermod.grid_size_meters) if aermod.grid_size_meters is not None else None,
            "beam_osm_id_col": emissions.beam_osm_id_col,
            "beam_length_col": emissions.beam_length_col,
            "inmap_grid_epsg": int(inmap_grid_epsg) if inmap_grid_epsg is not None else None,
            "aermod_grid_epsg": int(local_output_epsg) if staged_aermod_grid else None,
            "aermod_grid_id": resolved_aermod_grid_id,
            "output_epsg": int(local_output_epsg),
            "region": settings.run.region,
            "start_year": settings.run.start_year,
            "county_state_fips": geography.fips.state,
            "county_fips_codes": list(geography.fips.counties),
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
            "local_output_dir": str(Path(resolve_path(settings.impacts.local_output_folder, config_path)).resolve()),
            "container_input_dir": "/input",
            "container_output_dir": "/output",
            "entrypoint": "python -m impacts run",
            "command_template": "python -m impacts run --input-manifest {input_manifest} --output-dir {output_dir}",
            "canonical_output_filenames": ["impacts_exposure_table.parquet"],
            "manifest_filenames": ["inputs_manifest.yaml", "run_manifest.yaml", "postprocess_manifest.yaml"],
        },
        "population_inputs": dict(step1_outputs["population_inputs"]),
        "notes": [
            "Only maintained modules are part of this contract.",
            "ActivitySim population integration is currently best-effort and relies on staged population tables carrying usable cell ids.",
            f"Settings file registered at {settings_entry_path}",
        ],
    }
    return manifest


def preprocess_workflow(
    settings_path: str | Path,
    staging_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=False,
    )
    manifest = build_inputs_manifest(settings_path=settings_path, staging_dir=staging_dir)
    output_manifest = Path(manifest_path) if manifest_path else Path(manifest["inputs_manifest_path"])
    manifest["inputs_manifest_path"] = str(output_manifest)
    typed_manifest = InputsManifest.from_dict(manifest)
    write_structured_file(output_manifest, typed_manifest.to_dict())
    return typed_manifest.to_dict()
