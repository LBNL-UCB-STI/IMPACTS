from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from .contract_utils import copy_path
from .contract_utils import file_entry
from .contract_utils import is_remote_path
from .contract_utils import load_structured_file
from .contract_utils import resolve_path
from .contract_utils import write_structured_file
from .contract_utils import parquet_available


CONTRACT_VERSION = "1"
DEFAULT_POPULATION_SECTION = "activitysim_population"
DEFAULT_PILATES_SECTION = "pilates_terminal_contract"
DEFAULT_EVENTS_COLUMNS = {
    "type": "type",
    "vehicle": "vehicle",
    "vehicle_type": "vehicleType",
    "departure_time": "departureTime",
    "links": "links",
    "link_travel_time": "linkTravelTime",
    "length": "length",
}
DEFAULT_NETWORK_COLUMNS = {
    "link_id": "linkId",
    "link_length": "linkLength",
}
DEFAULT_BEAM_NETWORK_COLUMNS = {
    "osm_id": "attributeOrigId",
    "length": "linkLength",
}
DEFAULT_GRID_COLUMNS = {
    "cell_id": "grid",
}
DEFAULT_SKIMS_COLUMNS = {
    "hour": "hour",
    "link_id": "linkId",
    "vehicle_type": "vehicleTypeId",
    "process": "process",
    "emissions": "emissions",
    "observations": "observations",
    "iterations": "iterations",
    "travel_time": "travelTimeInSecond",
    "parking_duration": "parkingDurationInSecond",
}
DEFAULT_MAPPING_COLUMNS = {
    "link_id": "edge_linkId",
    "proportion": "proportion",
    "grid_id": "GRID",
}
DEFAULT_DISPERSION_EMISSIONS_COLUMNS = {
    "grid_id": "GRID",
    "rog": "tons_per_year_ROG",
    "nox": "tons_per_year_NOx",
    "nh3": "tons_per_year_NH3",
    "sox": "tons_per_year_SOx",
    "pm25": "tons_per_year_PM2_5",
    "bcv1": "tons_per_year_BCV1",
    "bcv3": "tons_per_year_BCV3",
}
DEFAULT_PERSONS_COLUMNS = {
    "household_id": "household_id",
    "cell_id": "cell_id",
    "age": "age",
    "sex": "sex",
    "income": "income",
}
DEFAULT_HOUSEHOLDS_COLUMNS = {
    "household_id": "household_id",
    "cell_id": "cell_id",
    "income": "income",
    "income_category": "income_category",
}
DEFAULT_POPULATION_MAPPING_COLUMNS = {
    "household_id": "household_id",
    "cell_id": "cell_id",
}


def _with_defaults(defaults: Dict[str, Any], config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = defaults.copy()
    merged.update(dict(config or {}))
    return merged


def _section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    return config.get(name, {}) or {}


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


def build_inputs_manifest(
    workflow_config_path: str | Path,
    staging_dir: str | Path,
) -> Dict[str, Any]:
    config_path = Path(workflow_config_path).resolve()
    config = load_structured_file(config_path)

    main = _section(config, "main")
    events_section = _section(config, main.get("events_section", "emissions_events"))
    rates_section = _section(config, main.get("rates_section", "emissions_rates"))
    mapping_section = _section(config, main.get("mapping_section", "osm_grid"))
    emissions_mapping_section = _section(
        config,
        main.get("emissions_mapping_section", "emissions_grid_mapping"),
    )
    dispersion_section = _section(config, main.get("dispersion_section", "dispersion_isrm"))
    population_section = _section(
        config,
        main.get("population_section", DEFAULT_POPULATION_SECTION),
    )
    pilates_section = _section(
        config,
        main.get("pilates_section", DEFAULT_PILATES_SECTION),
    )

    workspace_root = Path(staging_dir).resolve()
    input_root = workspace_root / "input"
    input_root.mkdir(parents=True, exist_ok=True)

    manifest_inputs: Dict[str, Any] = {}

    workflow_copy = _stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="workflow_config",
        source_path=str(config_path),
        relative_target="config/workflow_config.yaml",
    )

    events_path = resolve_path(events_section.get("events_input_path"), config_path)
    skims_input_path = _optional_local_path(
        resolve_path(emissions_mapping_section.get("skims_input_path"), config_path),
    )
    staged_events = None
    if events_path:
        resolved_events_path = _required_local_path(
            events_path,
            "emissions_events.events_input_path",
        )
        staged_events = _stage_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="events",
            source_path=resolved_events_path,
            relative_target=f"events/{Path(resolved_events_path).name}",
        )
    if not staged_events and not skims_input_path:
        raise ValueError(
            "Preprocess requires either emissions_events.events_input_path "
            "or emissions_grid_mapping.skims_input_path"
        )

    staged_skims_input = None
    staged_prepared_skims_input = None
    staged_county_correction_factors = None
    if skims_input_path and not is_remote_path(skims_input_path):
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
            skims_columns=_with_defaults(DEFAULT_SKIMS_COLUMNS, emissions_mapping_section.get("skims_columns", {})),
            group_cols=list(emissions_mapping_section.get("prepared_skims_group_cols", []) or ["linkId", "vehicleTypeId", "process"]),
            required_pollutants=list(
                emissions_mapping_section.get("prepared_pollutants", [])
                or ["NH3", "NOx", "PM2_5", "SOx", "ROG", "BCh"]
            ),
        )
        prepared_skims_path = _prepared_table_target(input_root, "prepared_skims_for_grid_allocation")
        annualize_prepared_skims_for_grid_allocation(
            prepared_skims_path=str(prepared_grouped_skims_path),
            output_path=str(prepared_skims_path),
            group_cols=list(emissions_mapping_section.get("prepared_skims_group_cols", []) or ["linkId", "vehicleTypeId", "process"]),
            required_pollutants=list(
                emissions_mapping_section.get("prepared_pollutants", [])
                or ["NH3", "NOx", "PM2_5", "SOx", "ROG", "BCh"]
            ),
            annualization_days=float(emissions_mapping_section.get("annualization_days", 330.0) or 330.0),
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

    county_correction_factors_path = resolve_path(
        emissions_mapping_section.get("county_correction_factors_path"),
        config_path,
    )
    if county_correction_factors_path and not is_remote_path(county_correction_factors_path):
        staged_county_correction_factors = _stage_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="county_correction_factors",
            source_path=county_correction_factors_path,
            relative_target=f"county/{Path(county_correction_factors_path).name}",
            optional=True,
        )

    link_length_source = resolve_path(events_section.get("link_length_path"), config_path)
    beam_network_source = resolve_path(mapping_section.get("beam_network_path"), config_path)
    staged_link_lengths = None
    if link_length_source:
        staged_link_lengths = _stage_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="link_lengths",
            source_path=_required_local_path(link_length_source, "emissions_events.link_length_path"),
            relative_target=f"network/{Path(link_length_source).name}",
            optional=True,
        )
    elif beam_network_source:
        beam_network_source = _required_local_path(
            beam_network_source,
            "osm_grid.beam_network_path",
        )
        staged_link_lengths = _stage_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="link_lengths",
            source_path=beam_network_source,
            relative_target=f"network/{Path(beam_network_source).name}",
            optional=True,
        )

    rates_dir = _optional_local_path(resolve_path(rates_section.get("rates_dir"), config_path))
    staged_rates_dir = None
    if rates_dir and not is_remote_path(rates_dir):
        staged_rates_dir = _stage_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="rates_dir",
            source_path=rates_dir,
            relative_target="rates",
            optional=True,
        )

    precomputed_mapping_path = _optional_local_path(
        resolve_path(emissions_mapping_section.get("mapping_input_path"), config_path),
    )
    staged_precomputed_mapping = None
    staged_osm = None
    staged_beam_network = None
    staged_grid = None
    staged_fine_grid = None
    use_precomputed_mapping = bool(precomputed_mapping_path)
    if use_precomputed_mapping and not is_remote_path(str(precomputed_mapping_path)):
        staged_precomputed_mapping = _stage_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="mapping_input",
            source_path=str(precomputed_mapping_path),
            relative_target=f"mapping/{Path(str(precomputed_mapping_path)).name}",
        )
    else:
        osm_source = resolve_path(mapping_section.get("osm_links_path"), config_path)
        osm_pbf_source = resolve_path(mapping_section.get("osm_pbf_path"), config_path)
        if osm_source:
            osm_source = _required_local_path(osm_source, "osm_grid.osm_links_path")
        elif osm_pbf_source:
            osm_source = _required_local_path(osm_pbf_source, "osm_grid.osm_pbf_path")
        else:
            raise ValueError("Either osm_grid.osm_links_path or osm_grid.osm_pbf_path is required.")
        beam_network_source = _required_local_path(
            resolve_path(mapping_section.get("beam_network_path"), config_path),
            "osm_grid.beam_network_path",
        )
        grid_source = _required_local_path(
            resolve_path(
                mapping_section.get("inmap_grid_path"),
                config_path,
            ),
            "osm_grid.inmap_grid_path",
        )
        fine_grid_source = _optional_local_path(
            resolve_path(
                mapping_section.get("aermod_grid_path"),
                config_path,
            ),
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
        staged_grid = _stage_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="grid_cells",
            source_path=grid_source,
            relative_target=f"grid/{Path(grid_source).name}",
        )
        if fine_grid_source:
            staged_fine_grid = _stage_local_input(
                manifest_inputs=manifest_inputs,
                input_root=input_root,
                key="fine_grid_cells",
                source_path=fine_grid_source,
                relative_target=f"fine_grid/{Path(fine_grid_source).name}",
                optional=True,
            )

    isrm_source = resolve_path(dispersion_section.get("isrm_url"), config_path)
    staged_isrm = _stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="isrm",
        source_path=isrm_source,
        relative_target=f"isrm/{Path(str(isrm_source)).name}" if isrm_source and not is_remote_path(str(isrm_source)) else "isrm",
    )

    staged_persons = _stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="persons",
        source_path=_optional_local_path(resolve_path(population_section.get("persons_path"), config_path)),
        relative_target="population/persons.csv",
    )
    staged_households = _stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="households",
        source_path=_optional_local_path(resolve_path(population_section.get("households_path"), config_path)),
        relative_target="population/households.csv",
    )
    staged_land_use = _stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="land_use",
        source_path=_optional_local_path(resolve_path(population_section.get("land_use_path"), config_path)),
        relative_target="population/land_use.csv",
    )
    staged_population_mapping = _stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="population_cell_mapping",
        source_path=_optional_local_path(resolve_path(population_section.get("population_cell_mapping_path"), config_path)),
        relative_target="population/population_cell_mapping.csv",
    )
    staged_osm_pbf = _stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="osm_pbf",
        source_path=_optional_local_path(resolve_path(mapping_section.get("osm_pbf_path"), config_path)),
        relative_target="osm/source.pbf",
    )
    staged_beam_mapdb = _stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="beam_mapdb",
        source_path=_optional_local_path(resolve_path(mapping_section.get("beam_mapdb_path"), config_path)),
        relative_target="osm/beam.mapdb",
    )

    manifest: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "model": "impacts",
        "workflow_config_source": str(config_path),
        "staging_dir": str(workspace_root),
        "input_dir": str(input_root),
        "inputs_manifest_path": str(workspace_root / "inputs_manifest.yaml"),
        "maintained_execution_path": [
            "impacts.emissions.events_to_skims_emissions",
            "impacts.emissions.emissions_grid_mapping",
            "impacts.dispersion.isrm_dispersion",
            "impacts.network2grid.network_grid_clipping",
        ],
        "legacy_paths_not_wired": [
            "src/impacts/tmp",
            "src/impacts/tmp/archive",
            "src/impacts/tmp/deprecated_R",
        ],
        "inputs": manifest_inputs,
        "pipeline": {
            "events_path": staged_events,
            "skims_input_path": staged_skims_input,
            "prepared_skims_input_path": staged_prepared_skims_input,
            "link_length_path": staged_link_lengths,
            "rates_dir": staged_rates_dir,
            "mapping_input_path": staged_precomputed_mapping,
            "use_precomputed_mapping": use_precomputed_mapping,
            "osm_links_path": staged_osm,
            "osm_pbf_path": staged_osm_pbf,
            "beam_mapdb_path": staged_beam_mapdb,
            "beam_network_path": staged_beam_network,
            "inmap_grid_path": staged_grid,
            "aermod_grid_path": staged_fine_grid,
            "isrm_url": staged_isrm,
            "iterations": int(events_section.get("iteration", 0)),
            "use_rates": bool(events_section.get("use_rates", True)),
            "beam_osm_id_col": mapping_section.get("beam_osm_id_col", "attributeOrigId"),
            "beam_length_col": mapping_section.get("beam_length_col", "linkLength"),
            "beam_osm_epsg": int(mapping_section.get("beam_osm_epsg", 4326)),
            "inmap_grid_epsg": int(mapping_section.get("inmap_grid_epsg", 4326) or 4326),
            "aermod_grid_epsg": int(mapping_section.get("aermod_grid_epsg", 4326) or 4326),
            "output_epsg": int(mapping_section.get("output_epsg", 26910)),
            "county_state_fips": mapping_section.get("county_state_fips"),
            "county_fips_codes": list(mapping_section.get("county_fips_codes", []) or []),
            "county_area_name": mapping_section.get("county_area_name", "county"),
            "concentration_factor": float(dispersion_section.get("concentration_factor", 28766.639)),
            "include_bc": bool(dispersion_section.get("include_bc", False)),
            "include_health": bool(dispersion_section.get("include_health", False)),
            "events_columns": _with_defaults(DEFAULT_EVENTS_COLUMNS, events_section.get("events_columns", {})),
            "network_columns": _with_defaults(DEFAULT_NETWORK_COLUMNS, events_section.get("network_columns", {})),
            "beam_network_columns": _with_defaults(DEFAULT_BEAM_NETWORK_COLUMNS, mapping_section.get("beam_network_columns", {})),
            "grid_columns": _with_defaults(DEFAULT_GRID_COLUMNS, mapping_section.get("grid_columns", {})),
            "skims_columns": _with_defaults(DEFAULT_SKIMS_COLUMNS, emissions_mapping_section.get("skims_columns", {})),
            "mapping_columns": _with_defaults(DEFAULT_MAPPING_COLUMNS, emissions_mapping_section.get("mapping_columns", {})),
            "prepared_skims_input_path": staged_prepared_skims_input,
            "prepared_skims_grouped_path": str(manifest_inputs["prepared_skims_grouped"]["staged_path"]) if manifest_inputs.get("prepared_skims_grouped") else None,
            "prepared_skims_group_cols": list(emissions_mapping_section.get("prepared_skims_group_cols", []) or ["linkId", "vehicleTypeId", "process"]),
            "prepared_pollutants": list(
                emissions_mapping_section.get("prepared_pollutants", [])
                or ["NH3", "NOx", "PM2_5", "SOx", "ROG", "BCh"]
            ),
            "county_correction_factors_path": staged_county_correction_factors,
            "county_correction_columns": _with_defaults(
                {
                    "county_fips": "COUNTYFP",
                    "vmt_factor": "corr_VMT_by_county",
                    "trips_factor": "corr_trips_by_county",
                },
                emissions_mapping_section.get("county_correction_columns", {}),
            ),
            "annualization_days": float(emissions_mapping_section.get("annualization_days", 330.0) or 330.0),
            "dispersion_emissions_columns": _with_defaults(DEFAULT_DISPERSION_EMISSIONS_COLUMNS, dispersion_section.get("emissions_columns", {})),
        },
        "pilates_contract": {
            "stage": "terminal_postprocessing",
            "runs_after": ["beam", "supply_demand_loop"],
            "upstream_dependency_only": True,
            "publishes_final_artifact_only": True,
            "local_input_dir": pilates_section.get("local_input_dir", str(input_root)),
            "local_output_dir": pilates_section.get("local_output_dir", str(workspace_root / "output")),
            "container_input_dir": pilates_section.get("container_input_dir", "/input"),
            "container_output_dir": pilates_section.get("container_output_dir", "/output"),
            "entrypoint": pilates_section.get("entrypoint", "python -m impacts run"),
            "command_template": pilates_section.get(
                "command_template",
                "python -m impacts run --input-manifest {input_manifest} --output-dir {output_dir}",
            ),
            "canonical_output_filenames": pilates_section.get(
                "canonical_output_filenames",
                ["impacts_exposure_table.parquet", "impacts_exposure_table.csv.gz"],
            ),
            "manifest_filenames": pilates_section.get(
                "manifest_filenames",
                ["inputs_manifest.yaml", "run_manifest.yaml", "postprocess_manifest.yaml"],
            ),
            "placeholders": {
                "land_use_context": staged_land_use is None,
                "osm_reference": staged_osm is None and staged_osm_pbf is None and staged_beam_mapdb is None,
            },
        },
        "population_inputs": {
            "persons_path": staged_persons,
            "households_path": staged_households,
            "land_use_path": staged_land_use,
            "population_cell_mapping_path": staged_population_mapping,
            "persons_columns": _with_defaults(DEFAULT_PERSONS_COLUMNS, population_section.get("persons_columns", {})),
            "households_columns": _with_defaults(DEFAULT_HOUSEHOLDS_COLUMNS, population_section.get("households_columns", {})),
            "population_mapping_columns": _with_defaults(DEFAULT_POPULATION_MAPPING_COLUMNS, population_section.get("population_mapping_columns", {})),
        },
        "notes": [
            "Only maintained modules are part of this contract.",
            "ActivitySim population integration is currently best-effort and may rely on pre-mapped cell ids or a population_cell_mapping file.",
            f"Workflow config staged at {workflow_copy}",
        ],
    }
    return manifest


def preprocess_workflow(
    workflow_config_path: str | Path,
    staging_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    manifest = build_inputs_manifest(workflow_config_path=workflow_config_path, staging_dir=staging_dir)
    output_manifest = Path(manifest_path) if manifest_path else Path(manifest["inputs_manifest_path"])
    manifest["inputs_manifest_path"] = str(output_manifest)
    write_structured_file(output_manifest, manifest)
    return manifest


def impacts_preprocess(
    workflow_config_path: str | Path,
    staging_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    return preprocess_workflow(
        workflow_config_path=workflow_config_path,
        staging_dir=staging_dir,
        manifest_path=manifest_path,
    )
