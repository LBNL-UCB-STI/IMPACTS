from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import pandas as pd

from ...common import find_preferred_file
from ...common import log_step_banner
from ...common import log_substep_banner
from ...common import read_table
from ...common import required_local_path
from ...common import resolve_beam_network_local_path
from ...common import resolve_emissions_skims_local_path
from ...common import resolve_latest_events_local_path
from ...common import resolve_osm_pbf_local_path
from ...common import register_managed_input
from ...consist_artifacts import BEAM_EVENTS_PREFIX
from ...consist_artifacts import BEAM_HOUSEHOLDS_PREFIX
from ...consist_artifacts import BEAM_NETWORK_PREFIX
from ...consist_artifacts import BEAM_POPULATION_PREFIX
from ...consist_artifacts import BEAM_R5_OSM_FILE_KEY
from ...consist_artifacts import find_beam_r5_osm_reference
from ...consist_artifacts import find_latest_beam_events_reference
from ...consist_artifacts import find_latest_beam_households_reference
from ...consist_artifacts import find_latest_beam_network_reference
from ...consist_artifacts import find_latest_beam_population_reference
from ...consist_artifacts import resolve_logged_path
from ...manifest.file_ops import resolve_path

logger = logging.getLogger(__name__)


def _register_manifest_input(
    manifest_inputs: Dict[str, Any],
    *,
    input_root: Path,
    key: str,
    source_path: str,
    relative_target: str,
    artifact_key: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    optional: bool = False,
) -> str:
    return register_managed_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key=key,
        source_path=source_path,
        relative_target=relative_target,
        artifact_key=artifact_key,
        optional=optional,
        prefer_reference=True,
        metadata=metadata,
    )


def _use_existing_reference(manifest_inputs: Dict[str, Any], key: str, entry: Dict[str, Any]) -> str:
    manifest_inputs[key] = entry
    return resolve_logged_path(entry)


def _locate_exchange_file(folder: Path, stem: str) -> Optional[str]:
    return find_preferred_file(str(folder), [f"{stem}.csv.gz", f"{stem}.csv", f"{stem}.parquet"])


def _resolve_region_or_absolute_path(raw_path: str, *, region_input_root: Path, config_path: Path) -> str:
    raw = str(raw_path).strip()
    if raw.startswith("~") or Path(raw).is_absolute():
        return resolve_path(raw, config_path) or raw
    return str((region_input_root / raw).resolve())


def _build_combined_vehicle_types_input(
    *,
    passenger_vehicle_types_source: str,
    freight_vehicle_types_source: str,
    input_root: Path,
) -> str:
    passenger = read_table(passenger_vehicle_types_source).copy()
    passenger["assignment_group"] = "passenger"
    freight = read_table(freight_vehicle_types_source).copy()
    freight["assignment_group"] = "freight"
    combined = pd.concat([passenger, freight], ignore_index=True, sort=False)
    duplicate_ids = combined.loc[combined["vehicleTypeId"].duplicated(), "vehicleTypeId"].drop_duplicates().tolist()
    if duplicate_ids:
        raise ValueError(
            "Configured passenger and freight vehicle types files contain duplicate vehicleTypeId values: "
            f"{duplicate_ids[:10]}"
        )
    output_path = input_root / "vehicle_types_input" / "vehicleTypes--combined--EM.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return str(output_path)


def run(
    *,
    manifest_inputs: Dict[str, Any],
    settings,
    config_path: Path,
    input_root: Path,
) -> dict[str, Any]:
    log_step_banner("Preprocess Step 1", "Collect Inputs", logger=logger)
    beam = settings.beam
    impacts = settings.impacts
    beam_processing = impacts.beam
    emissions = impacts.emissions
    inmap = impacts.dispersions.inmap
    aermod = impacts.dispersions.aermod
    exposure = impacts.exposure

    beam_output_root = required_local_path(
        resolve_path(beam.local_output_folder, config_path),
        "beam.local_output_folder",
    )
    beam_input_root = required_local_path(
        resolve_path(beam.local_input_folder, config_path),
        "beam.local_input_folder",
    )
    region_input_root = Path(beam_input_root) / settings.run.region
    if not region_input_root.exists():
        raise FileNotFoundError(f"Region input root not found: {region_input_root}")
    input_root.mkdir(parents=True, exist_ok=True)

    log_substep_banner("1.1", "register BEAM output files", logger=logger)
    network_entry = find_latest_beam_network_reference(optional=True)
    if network_entry:
        staged_network = _use_existing_reference(manifest_inputs, "network", network_entry)
    else:
        network_source = resolve_beam_network_local_path(beam_output_root)
        staged_network = _register_manifest_input(
            manifest_inputs,
            input_root=input_root,
            key="network",
            source_path=network_source,
            relative_target=Path(network_source).name,
            artifact_key=BEAM_NETWORK_PREFIX,
            metadata={"artifact_family": BEAM_NETWORK_PREFIX},
        )

    skims_source = resolve_emissions_skims_local_path(beam_output_root)
    staged_skims = _register_manifest_input(
        manifest_inputs,
        input_root=input_root,
        key="emissions_skims_input",
        source_path=skims_source,
        relative_target=Path(skims_source).name,
        metadata={"artifact_family": "emissions_skims_input"},
        optional=True,
    )

    events_entry = find_latest_beam_events_reference(optional=True)
    staged_events = None
    if events_entry:
        staged_events = _use_existing_reference(manifest_inputs, "events_input", events_entry)
    else:
        events_source = resolve_latest_events_local_path(Path(beam_output_root))
        if events_source:
            staged_events = _register_manifest_input(
                manifest_inputs,
                input_root=input_root,
                key="events_input",
                source_path=events_source,
                relative_target=Path(events_source).name,
                artifact_key=BEAM_EVENTS_PREFIX,
                metadata={"artifact_family": BEAM_EVENTS_PREFIX},
                optional=True,
            )

    log_substep_banner("1.2", "register production inputs", logger=logger)
    osm_entry = find_beam_r5_osm_reference(optional=True)
    if osm_entry:
        staged_osm = _use_existing_reference(manifest_inputs, "osm_network", osm_entry)
    else:
        resolved_osm_root = _resolve_region_or_absolute_path(
            emissions.osm_network_folder,
            region_input_root=region_input_root,
            config_path=config_path,
        )
        osm_source = resolve_osm_pbf_local_path(
            resolved_osm_root,
        ) or required_local_path(
            resolved_osm_root,
            "impacts.emissions.osm_network_folder",
        )
        staged_osm = _register_manifest_input(
            manifest_inputs,
            input_root=input_root,
            key="osm_network",
            source_path=osm_source,
            relative_target=Path(osm_source).name,
            artifact_key=BEAM_R5_OSM_FILE_KEY,
            metadata={"artifact_family": BEAM_R5_OSM_FILE_KEY},
        )
    emissions_rates_source = required_local_path(
        _resolve_region_or_absolute_path(
            emissions.emissions_rates_folder,
            region_input_root=region_input_root,
            config_path=config_path,
        ),
        "impacts.emissions.emissions_rates_folder",
    )
    _register_manifest_input(
        manifest_inputs,
        input_root=input_root,
        key="emissions_rates_folder",
        source_path=emissions_rates_source,
        relative_target=emissions.emissions_rates_folder,
        metadata={"artifact_family": "emissions_rates_folder"},
    )
    passenger_inventory_source = required_local_path(
        _resolve_region_or_absolute_path(
            emissions.inventory.passenger_file,
            region_input_root=region_input_root,
            config_path=config_path,
        ),
        "impacts.emissions.inventory.passenger_file",
    )
    staged_passenger_inventory_file = _register_manifest_input(
        manifest_inputs,
        input_root=input_root,
        key="passenger_inventory_file",
        source_path=passenger_inventory_source,
        relative_target=str(emissions.inventory.passenger_file),
        metadata={"artifact_family": "passenger_inventory_file"},
    )
    freight_inventory_source = required_local_path(
        _resolve_region_or_absolute_path(
            emissions.inventory.freight_file,
            region_input_root=region_input_root,
            config_path=config_path,
        ),
        "impacts.emissions.inventory.freight_file",
    )
    staged_freight_inventory_file = _register_manifest_input(
        manifest_inputs,
        input_root=input_root,
        key="freight_inventory_file",
        source_path=freight_inventory_source,
        relative_target=str(emissions.inventory.freight_file),
        metadata={"artifact_family": "freight_inventory_file"},
    )
    passenger_vehicle_types_source = required_local_path(
        _resolve_region_or_absolute_path(
            beam_processing.passenger_vehicle_types_file,
            region_input_root=region_input_root,
            config_path=config_path,
        ),
        "impacts.beam.passenger_vehicle_types_file",
    )
    _register_manifest_input(
        manifest_inputs,
        input_root=input_root,
        key="passenger_vehicle_types_input",
        source_path=passenger_vehicle_types_source,
        relative_target=str(beam_processing.passenger_vehicle_types_file),
        metadata={"artifact_family": "passenger_vehicle_types_input"},
    )
    freight_vehicle_types_source = required_local_path(
        _resolve_region_or_absolute_path(
            beam_processing.freight_vehicle_types_file,
            region_input_root=region_input_root,
            config_path=config_path,
        ),
        "impacts.beam.freight_vehicle_types_file",
    )
    _register_manifest_input(
        manifest_inputs,
        input_root=input_root,
        key="freight_vehicle_types_input",
        source_path=freight_vehicle_types_source,
        relative_target=str(beam_processing.freight_vehicle_types_file),
        metadata={"artifact_family": "freight_vehicle_types_input"},
    )
    combined_vehicle_types_source = _build_combined_vehicle_types_input(
        passenger_vehicle_types_source=passenger_vehicle_types_source,
        freight_vehicle_types_source=freight_vehicle_types_source,
        input_root=input_root,
    )
    _register_manifest_input(
        manifest_inputs,
        input_root=input_root,
        key="vehicle_types_input",
        source_path=combined_vehicle_types_source,
        relative_target=Path(combined_vehicle_types_source).name,
        metadata={"artifact_family": "vehicle_types_input"},
    )
    staged_annualization_days_or_file = emissions.annualization_days_or_file
    if isinstance(emissions.annualization_days_or_file, str):
        annualization_days_source = required_local_path(
            resolve_path(emissions.annualization_days_or_file, config_path),
            "impacts.emissions.annualization_days_or_file",
        )
        staged_annualization_days_or_file = _register_manifest_input(
            manifest_inputs,
            input_root=input_root,
            key="annualization_days_or_file_input",
            source_path=annualization_days_source,
            relative_target=Path(annualization_days_source).name,
            metadata={"artifact_family": "annualization_days_or_file_input"},
        )

    staged_inmap_grid = None
    staged_isrm_nox_to_no2_ratios_file = None
    if inmap.enabled:
        inmap_grid_source = required_local_path(
            str((region_input_root / inmap.grid_path).resolve()),
            "impacts.dispersions.inmap.grid_path",
        )
        staged_inmap_grid = _register_manifest_input(
            manifest_inputs,
            input_root=input_root,
            key="inmap_grid",
            source_path=inmap_grid_source,
            relative_target=inmap.grid_path,
            metadata={"artifact_family": "inmap_grid"},
        )
        no2_matrix_source = required_local_path(
            str((region_input_root / inmap.isrm_nox_to_no2_ratios_file).resolve()),
            "impacts.dispersions.inmap.isrm_nox_to_no2_ratios_file",
        )
        staged_isrm_nox_to_no2_ratios_file = _register_manifest_input(
            manifest_inputs,
            input_root=input_root,
            key="isrm_nox_to_no2_ratios_file",
            source_path=no2_matrix_source,
            relative_target=inmap.isrm_nox_to_no2_ratios_file,
            metadata={"artifact_family": "isrm_nox_to_no2_ratios_file"},
        )
    staged_asrv_patterns_file = None
    if aermod.enabled:
        asrv_source = required_local_path(
            str((region_input_root / aermod.asrv_patterns_file).resolve()),
            "impacts.dispersions.aermod.asrv_patterns_file",
        )
        staged_asrv_patterns_file = _register_manifest_input(
            manifest_inputs,
            input_root=input_root,
            key="asrv_patterns_file",
            source_path=asrv_source,
            relative_target=aermod.asrv_patterns_file,
            metadata={"artifact_family": "asrv_patterns_file"},
        )

    log_substep_banner("1.3", "register external dispersion store", logger=logger)
    staged_isrm = None
    if inmap.enabled:
        isrm_source = required_local_path(resolve_path(inmap.isrm_zarr, config_path), "impacts.dispersions.inmap.isrm_zarr")
        staged_isrm = _register_manifest_input(
            manifest_inputs,
            input_root=input_root,
            key="isrm",
            source_path=isrm_source,
            relative_target=Path(isrm_source).name,
            metadata={"artifact_family": "isrm"},
        )

    population_inputs: Dict[str, Any] = {}
    if exposure.enabled:
        log_substep_banner("1.4", "register exposure population inputs", logger=logger)
        population_root = region_input_root / str(exposure.population_folder)
        population_artifact_entries = {
            "persons": find_latest_beam_population_reference(optional=True),
            "households": find_latest_beam_households_reference(optional=True),
        }
        population_artifact_keys = {
            "persons": BEAM_POPULATION_PREFIX,
            "households": BEAM_HOUSEHOLDS_PREFIX,
        }
        for stem in ("persons", "households"):
            existing_entry = population_artifact_entries[stem]
            if existing_entry is not None:
                population_inputs[stem] = existing_entry
                continue
            source_path = _locate_exchange_file(population_root, stem)
            if source_path:
                register_managed_input(
                    manifest_inputs=population_inputs,
                    input_root=input_root,
                    key=stem,
                    source_path=source_path,
                    relative_target=str(Path(exposure.population_folder) / Path(source_path).name),
                    artifact_key=population_artifact_keys[stem],
                    prefer_reference=True,
                    metadata={"artifact_family": population_artifact_keys[stem]},
                )

    logger.info("Preprocess Step 1 complete")
    return {
        "staged_network": staged_network,
        "staged_osm": staged_osm,
        "staged_skims": staged_skims,
        "staged_vehicle_types": staged_vehicle_types,
        "staged_events": staged_events,
        "staged_inmap_grid": staged_inmap_grid,
        "staged_passenger_inventory_file": staged_passenger_inventory_file,
        "staged_freight_inventory_file": staged_freight_inventory_file,
        "staged_annualization_days_or_file": staged_annualization_days_or_file,
        "staged_isrm": staged_isrm,
        "staged_isrm_nox_to_no2_ratios_file": staged_isrm_nox_to_no2_ratios_file,
        "staged_asrv_patterns_file": staged_asrv_patterns_file,
        "population_inputs": population_inputs,
    }
