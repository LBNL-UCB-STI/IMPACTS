from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any
from typing import Dict
from .config.settings_builder import load_settings_from_yaml
from .manifest.file_ops import resolve_path
from .manifest.file_ops import write_structured_file
from .manifest.schema import PreprocessManifest
from .manifest.schema import PipelineConfig
from .manifest.schema import PipelineManifest
from .common import infer_vector_epsg
from .common import parse_epsg
from .common import prepared_table_target
from .common import register_local_input
from .common import resolve_required_manifest_input


CONTRACT_VERSION = "1"
logger = logging.getLogger(__name__)


def _derive_impacts_tmp_root(output_root: Path) -> Path:
    return output_root / "preprocess"


def build_preprocess_manifest(
    settings_path: str | Path,
    output_root_override: str | Path | None = None,
) -> Dict[str, Any]:
    config_path = Path(settings_path).resolve()
    settings = load_settings_from_yaml(config_path)
    geography = settings.shared.geography
    emissions = settings.impacts.emissions
    beam_processing = settings.impacts.emissions.beam
    inmap = settings.impacts.dispersions.inmap
    aermod = settings.impacts.dispersions.aermod
    pipeline = settings.impacts.pipeline

    output_root = (
        Path(output_root_override).expanduser().resolve()
        if output_root_override is not None
        else Path(resolve_path(settings.impacts.local_output_folder, config_path)).resolve()
    )
    input_root = _derive_impacts_tmp_root(output_root).resolve()
    input_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_inputs: Dict[str, Any] = {}
    import shutil as _shutil
    settings_copy = output_root / "settings.yaml"
    _shutil.copy2(str(config_path), str(settings_copy))
    settings_entry_path = register_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="settings",
        source_path=str(settings_copy),
    )
    from .pipeline.preprocessing.step1_collect_inputs import run as preprocess_step1
    from .pipeline.preprocessing.step2_prepare_grids import run as preprocess_step2
    local_output_epsg = parse_epsg(geography.local_crs)
    step1_outputs = preprocess_step1(
        manifest_inputs=manifest_inputs,
        settings=settings,
        input_root=input_root,
        config_path=config_path,
    )
    staged_passenger_inventory_file = step1_outputs["staged_passenger_inventory_file"]
    staged_freight_inventory_file = step1_outputs["staged_freight_inventory_file"]
    staged_isrm = step1_outputs["staged_isrm"]
    staged_isrm_nox_to_no2_ratios_file = step1_outputs["staged_isrm_nox_to_no2_ratios_file"]
    staged_asrv_patterns_file = step1_outputs["staged_asrv_patterns_file"]
    staged_inmap_grid_source = step1_outputs["staged_inmap_grid"]
    inmap_grid_epsg = None
    if pipeline.inmap and staged_inmap_grid_source:
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
    if pipeline.aermod and staged_asrv_patterns_file:
        asrv_patterns_epsg = (
            infer_vector_epsg(staged_asrv_patterns_file)
            or aermod.asrv_patterns_epsg
        )
    if pipeline.aermod and staged_asrv_patterns_file and asrv_patterns_epsg is None:
        raise ValueError(
            "Could not determine EPSG for impacts.dispersions.aermod.asrv_patterns_file. "
            "Set impacts.dispersions.aermod.asrv_patterns_epsg explicitly or provide CRS metadata in the file."
        )

    pipeline_payload = {
        "emissions_enabled": bool(pipeline.emissions),
        "inmap_enabled": bool(pipeline.inmap),
        "aermod_enabled": bool(pipeline.aermod),
        "exposure_enabled": bool(pipeline.exposure),
        "inmap_grid_path": staged_inmap_grid,
        "aermod_full_grid_path": staged_aermod_full_grid,
        "aermod_grid_path": staged_aermod_grid,
        "isrm_url": staged_isrm,
        "isrm_nox_to_no2_ratios_file": staged_isrm_nox_to_no2_ratios_file,
        "isrm_nox_to_no2_ratios_apply_tons_per_year_to_ug_per_s": bool(
            inmap.isrm_nox_to_no2_ratios_apply_tons_per_year_to_ug_per_s
        ),
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
        "passenger_inventory_file": staged_passenger_inventory_file,
        "freight_inventory_file": staged_freight_inventory_file,
        "enable_passenger_inventory_activity_correction": bool(emissions.inventory.enable_passenger_activity_correction),
        "enable_freight_inventory_activity_correction": bool(emissions.inventory.enable_freight_activity_correction),
        "passenger_vehicle_types_file": resolve_required_manifest_input(manifest_inputs, key="passenger_vehicle_types_input"),
        "freight_vehicle_types_file": resolve_required_manifest_input(manifest_inputs, key="freight_vehicle_types_input"),
        "vehicle_category_metadata_file": (
            resolve_required_manifest_input(manifest_inputs, key="vehicle_category_metadata_file_input")
            if emissions.vehicle_category_metadata_file
            else None
        ),
        "annualization_days": {
            "light_duty": float(emissions.defaults.default_annualization_days.light_duty),
            "medium_heavy_duty": float(emissions.defaults.default_annualization_days.medium_heavy_duty),
        },
        "population_sample": float(settings.impacts.population.population_sample),
        "transit_sample": float(settings.impacts.population.transit_sample),
        "freight_sample": getattr(settings.impacts.population, "freight_sample", None),
        "primary_pm25_integration_strategy": settings.impacts.exposure.primary_pm25_integration_strategy,
        "include_non_osm_car_links": bool(beam_processing.include_non_osm_car_links),
        "include_passenger": bool(beam_processing.include_passenger),
        "include_freight": bool(beam_processing.include_freight),
    }
    pipeline_config = PipelineConfig.from_dict(pipeline_payload)

    from .pipeline.preprocessing.step3_integrate_grids import run as preprocess_step3
    intersection_paths, _ = preprocess_step3(
        pipeline_config,
        input_root,
        manifest_inputs=manifest_inputs,
    )
    for manifest_key, path in (
        ("county_intersection", intersection_paths.get("county")),
        ("inmap_intersection", intersection_paths.get("inmap")),
        ("aermod_intersection", intersection_paths.get("aermod")),
    ):
        if path:
            register_local_input(
                manifest_inputs=manifest_inputs,
                input_root=input_root,
                key=manifest_key,
                source_path=path,
            )

    from .pipeline.preprocessing.step4_assign_raw_population_to_grid import run as preprocess_step4
    (
        staged_population_path,
        population_counts_path,
        population_assignment,
        household_metadata_path,
    ) = preprocess_step4(
        pipeline_config,
        input_root,
        population_inputs=step1_outputs.get("population_inputs"),
    )
    if population_assignment is not None:
        step1_outputs["population_inputs"]["assignment"] = population_assignment
    if staged_population_path:
        register_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="staged_population",
            source_path=staged_population_path,
        )
    if population_counts_path:
        register_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="population_cell_counts",
            source_path=population_counts_path,
        )
    if household_metadata_path:
        register_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="population_households",
            source_path=household_metadata_path,
        )

    from .pipeline.preprocessing.step5_build_aermod_cell_attributes import run as preprocess_step5
    aermod_cell_attributes_path = preprocess_step5(
        pipeline_config,
        input_root,
        population_counts_path=population_counts_path,
    )
    if aermod_cell_attributes_path:
        register_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="aermod_cell_attributes",
            source_path=aermod_cell_attributes_path,
        )

    if household_metadata_path and step1_outputs["population_inputs"].get("acs_tables"):
        from .pipeline.preprocessing.step6_calibrate_acs_household_population import run as calibrate_acs_population

        acs_weights_path, acs_summary_path, acs_diagnostics = calibrate_acs_population(
            household_metadata_path=household_metadata_path,
            acs_tables=step1_outputs["population_inputs"]["acs_tables"],
            state_fips=geography.fips.state,
            county_fips_codes=list(geography.fips.counties),
            year=settings.run.start_year,
            output_root=input_root,
        )
        step1_outputs["population_inputs"]["acs_calibration"] = acs_diagnostics
        for key, path in (
            ("acs_household_weights", acs_weights_path),
            ("acs_household_calibration", acs_summary_path),
        ):
            register_local_input(
                manifest_inputs=manifest_inputs,
                input_root=input_root,
                key=key,
                source_path=path,
            )
        if staged_population_path:
            from .pipeline.preprocessing.step7_assign_acs_household_population_to_grid import run as assign_acs_population

            acs_cell_counts_path, acs_assignment_summary = assign_acs_population(
                staged_population_entry=manifest_inputs["staged_population"],
                household_weights_entry=manifest_inputs["acs_household_weights"],
                output_root=input_root,
            )
            step1_outputs["population_inputs"]["acs_assignment"] = acs_assignment_summary
            register_local_input(
                manifest_inputs=manifest_inputs,
                input_root=input_root,
                key="acs_household_population_cell_counts",
                source_path=acs_cell_counts_path,
            )

    population_inputs = step1_outputs["population_inputs"]
    if population_inputs.get("census_bundle"):
        from .pipeline.preprocessing.step8_assign_group_quarters_to_grid import run as assign_group_quarters

        block_path, count_path, report = assign_group_quarters(
            pipeline_config, input_root, population_inputs=population_inputs
        )
        population_inputs["group_quarters_assignment"] = report
        for key, path in (
            ("group_quarters_block_allocations", block_path),
            ("group_quarters_cell_counts", count_path),
        ):
            register_local_input(
                manifest_inputs=manifest_inputs, input_root=input_root, key=key, source_path=path
            )

    maintained_execution_path = ["impacts.pipeline.workflow.step1_process_emissions"]
    if pipeline.inmap:
        maintained_execution_path.append("impacts.pipeline.workflow.step2_compute_inmap_concentrations")
    if pipeline.aermod:
        maintained_execution_path.append("impacts.pipeline.workflow.step3_compute_aermod_concentrations")

    manifest: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "model": "impacts",
        "settings_source": str(config_path),
        "staging_dir": str(input_root),
        "input_dir": str(input_root),
        "preprocess_manifest_path": str(output_root / "postsim_manifest.yaml"),
        "maintained_execution_path": maintained_execution_path,
        "inputs": manifest_inputs,
        "pipeline": pipeline_config.to_dict(),
        "pilates_contract": {
            "stage": "terminal_postprocessing",
            "runs_after": ["beam", "supply_demand_loop"],
            "upstream_dependency_only": True,
            "publishes_final_artifact_only": True,
            "local_input_dir": str(input_root),
            "local_output_dir": str(output_root),
            "container_input_dir": "/input",
            "container_output_dir": "/output",
            "entrypoint": "python -m impacts emissions",
            "command_template": "python -m impacts emissions --run-manifest {run_manifest}",
            "canonical_output_filenames": ["impacts_exposure_table.parquet"],
            "manifest_filenames": ["presim_manifest.yaml", "postsim_manifest.yaml", "analysis/analysis_manifest.yaml"],
        },
        "population_inputs": dict(step1_outputs["population_inputs"]),
        "notes": [
            "Only maintained modules are part of this contract.",
            f"Settings file registered at {settings_entry_path}",
        ],
    }
    return manifest


def preprocess_workflow(
    settings_path: str | Path,
    manifest_path: str | Path | None = None,
    output_root_override: str | Path | None = None,
) -> Dict[str, Any]:
    manifest = build_preprocess_manifest(
        settings_path=settings_path,
        output_root_override=output_root_override,
    )
    output_manifest = (
        Path(manifest_path)
        if manifest_path
        else Path(manifest["preprocess_manifest_path"]).parent / "postsim_manifest.yaml"
    )
    manifest["preprocess_manifest_path"] = str(output_manifest)
    typed_manifest = PreprocessManifest.from_dict(manifest)
    pipeline_manifest_path = output_manifest.resolve()
    pipeline = PipelineConfig.from_dict(typed_manifest.to_dict()["pipeline"])
    inputs_payload = typed_manifest.to_dict()["inputs"]
    output_root = pipeline_manifest_path.parent.resolve()
    input_root = Path(typed_manifest.to_dict()["input_dir"]).resolve()
    effective_settings = output_root / "settings.yaml"
    if Path(settings_path).resolve() != effective_settings.resolve():
        shutil.copy2(settings_path, effective_settings)
    prepared_skims_candidate = prepared_table_target(output_root / "emissions", "prepared_skims_for_grid_allocation")
    initial_pipeline_manifest = {
        "contract_version": typed_manifest.contract_version,
        "model": "impacts",
        "preprocess_manifest_path": "",
        "output_dir": ".",
        "command": "python -m impacts preprocess",
        "image": "not_recorded",
        "outputs": {
            "skims_emissions": str(prepared_skims_candidate) if prepared_skims_candidate.exists() else None,
            "county_intersection": inputs_payload.get("county_intersection", {}).get("path"),
            "inmap_intersection": inputs_payload.get("inmap_intersection", {}).get("path") if pipeline.inmap_enabled else None,
            "aermod_intersection": inputs_payload.get("aermod_intersection", {}).get("path") if pipeline.aermod_enabled else None,
            "aermod_full_grid": pipeline.aermod_full_grid_path,
            "beam_emissions_by_county_process": None,
            "beam_emissions_for_inmap": None,
            "beam_inmap_study_area_grid": None,
            "beam_emissions_for_aermod": None,
            "beam_inmap_concentrations": None,
            "beam_inmap_concentrations_gpkg": None,
            "beam_aermod_concentrations": None,
            "beam_aermod_concentrations_gpkg": None,
            "beam_concentration_distribution": None,
            "beam_concentration_distribution_gpkg": None,
            "beam_population_distribution": None,
            "beam_population_counts": None,
            "beam_population_counts_gpkg": None,
            "acs_household_population_cell_counts": inputs_payload.get(
                "acs_household_population_cell_counts", {}
            ).get("staged_path"),
            "group_quarters_cell_counts": inputs_payload.get(
                "group_quarters_cell_counts", {}
            ).get("staged_path"),
            "group_quarters_block_allocations": inputs_payload.get(
                "group_quarters_block_allocations", {}
            ).get("staged_path"),
        },
        "pipeline": typed_manifest.to_dict()["pipeline"],
        "population_inputs": typed_manifest.to_dict()["population_inputs"],
        "settings_source": typed_manifest.settings_source,
        "input_dir": typed_manifest.input_dir,
        "inputs": inputs_payload,
        "preprocess": {
            "maintained_execution_path": typed_manifest.maintained_execution_path,
            "pilates_contract": typed_manifest.pilates_contract,
            "notes": typed_manifest.notes,
        },
        "deterministic_contract": {
            "uses_only_manifest_paths": True,
            "uses_baked_work_data": False,
        },
        "execution": {
            "dispersion_completed": False,
            "stage_timings_seconds": {},
            "stopped_after": "preprocess",
        },
        "pipeline_manifest_path": "postsim_manifest.yaml",
        "postsim_manifest_path": "postsim_manifest.yaml",
    }
    typed_pipeline_manifest = PipelineManifest.from_dict(initial_pipeline_manifest)
    write_structured_file(pipeline_manifest_path, typed_pipeline_manifest.to_dict())
    logger.info("Preprocess complete: wrote postsim manifest %s", output_manifest)
    logger.info(
        "Preprocess stage complete: postsim_manifest=%s",
        output_manifest,
    )
    result = typed_manifest.to_dict()
    result["pipeline_manifest_path"] = str(pipeline_manifest_path)
    result["postsim_manifest_path"] = str(pipeline_manifest_path)
    return result
