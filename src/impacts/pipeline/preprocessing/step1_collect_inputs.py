from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from ...common import find_preferred_file
from ...common import log_step_banner
from ...common import make_progress
from ...common import _set_progress_task
from ...common import log_substep_banner
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
from ...config.path_registry import build_registry
from ...config.settings import presim_activities_inventory_root
from ...config.settings import presim_activities_tmp_root
from ...manifest.file_ops import resolve_path
from ...manifest.file_ops import resolve_required_path

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
    prefer_reference: bool = True,
) -> str:
    return register_managed_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key=key,
        source_path=source_path,
        relative_target=relative_target,
        artifact_key=artifact_key,
        optional=optional,
        prefer_reference=prefer_reference,
        metadata=metadata,
    )


def _use_existing_reference(manifest_inputs: Dict[str, Any], key: str, entry: Dict[str, Any]) -> str:
    manifest_inputs[key] = entry
    return resolve_logged_path(entry)


def _resolve_emfacid_path(inventory_path: str, *, label: str) -> str:
    source = Path(inventory_path).resolve()
    name = source.name
    if name.endswith("-activity-by-model-year.parquet"):
        derived = source.with_name(name.replace("-activity-by-model-year.parquet", "-activity-by-emfacid.parquet"))
    else:
        derived = source.with_name(f"{source.stem}-by-emfacid{source.suffix}")
    if not derived.exists():
        raise FileNotFoundError(
            f"Preprocess Step 1.3 requires the EMFAC {label} activity-by-emfacId file "
            f"at {derived}, but it was not found. "
            f"Re-run the EMFAC activities workflow to generate it: "
            f"python -m impacts activities --config <config>"
        )
    return str(derived)


def _locate_exchange_file(folder: Path, stem: str) -> Optional[str]:
    return find_preferred_file(str(folder), [f"{stem}.csv.gz", f"{stem}.csv", f"{stem}.parquet"])


def _find_emfacid_file(folder: Path, pattern: str) -> str | None:
    if not folder.is_dir():
        return None
    key = f"inventory-final-{pattern}-activity-by-emfacid"
    matches = [str(f) for f in folder.iterdir() if f.is_file() and not f.name.startswith(".") and key in f.name]
    return matches[0] if matches else None


def _glob_vehicle_types_file(folder: Path, source: str) -> str | None:
    if not folder.is_dir():
        return None
    matches = sorted(
        str(f) for f in folder.iterdir()
        if f.is_file() and not f.name.startswith(".") and source in f.name.lower() and "--em" in f.name.lower()
    )
    return matches[0] if matches else None


def _resolve_region_input_root(*, beam_input_root: Path, region: str) -> Path:
    candidate = beam_input_root / region
    if candidate.exists():
        return candidate

    # Some local BEAM datasets already point directly at the region root
    # (for example a beam-data-sfbay checkout with freight/ and vehicle-tech/ at top level).
    direct_layout_markers = ("freight", "vehicle-tech", "urbansim", "shape", "r5")
    if any((beam_input_root / marker).exists() for marker in direct_layout_markers):
        logger.info(
            "Preprocess Step 1: using beam.local_input_folder as the region root because %s was not found under %s",
            region,
            beam_input_root,
        )
        return beam_input_root

    raise FileNotFoundError(f"Region input root not found: {candidate}")


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
    beam_processing = impacts.emissions.beam
    emissions = impacts.emissions
    inmap = impacts.dispersions.inmap
    aermod = impacts.dispersions.aermod
    population = impacts.population
    pipeline = impacts.pipeline

    beam_output_root = resolve_required_path(beam.local_output_folder, config_path, "beam.local_output_folder")
    beam_input_root = resolve_required_path(beam.local_input_folder, config_path, "beam.local_input_folder")
    region_input_root = _resolve_region_input_root(
        beam_input_root=Path(beam_input_root),
        region=settings.run.region,
    )
    registry = build_registry(settings, config_path)
    input_root.mkdir(parents=True, exist_ok=True)
    staged_events = None
    staged_inmap_grid = None
    staged_isrm = None
    staged_isrm_nox_to_no2_ratios_file = None
    staged_asrv_patterns_file = None

    log_substep_banner("1.1", "register BEAM output files", logger=logger)
    progress = make_progress("Preprocess Step 1.1", total=3, unit="task", leave=False)
    try:
        _set_progress_task(progress, "network", step_label="Preprocess Step 1.1")
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
        progress.update(1)

        _set_progress_task(progress, "skims", step_label="Preprocess Step 1.1")
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
        progress.update(1)

        _set_progress_task(progress, "events", step_label="Preprocess Step 1.1")
        events_entry = find_latest_beam_events_reference(optional=True)
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
        progress.update(1)
    finally:
        progress.close()

    log_substep_banner("1.2", "register emissions processing inputs", logger=logger)
    substep_12_total = 4  # osm, rates, passenger vehicle types, freight vehicle types
    if emissions.vehicle_category_metadata_file:
        substep_12_total += 1
    progress = make_progress("Preprocess Step 1.2", total=substep_12_total, unit="task", leave=False)
    try:
        _set_progress_task(progress, "osm", step_label="Preprocess Step 1.2")
        osm_entry = find_beam_r5_osm_reference(optional=True)
        if osm_entry:
            staged_osm = _use_existing_reference(manifest_inputs, "osm_network", osm_entry)
        else:
            osm_root = str(registry.locate_required(
                emissions.osm_network_folder, label="impacts.emissions.osm_network_folder"
            ))
            osm_source = resolve_osm_pbf_local_path(osm_root) or required_local_path(
                osm_root, "impacts.emissions.osm_network_folder"
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
        progress.update(1)

        _set_progress_task(progress, "emissions rates", step_label="Preprocess Step 1.2")
        emissions_rates_source = str(registry.locate_required(
            emissions.rates_folder, label="impacts.emissions.rates_folder"
        ))
        _register_manifest_input(
            manifest_inputs,
            input_root=input_root,
            key="rates_folder",
            source_path=emissions_rates_source,
            relative_target=emissions.rates_folder,
            metadata={"artifact_family": "rates_folder"},
        )
        progress.update(1)

        _set_progress_task(progress, "passenger vehicle types", step_label="Preprocess Step 1.2")
        passenger_vtypes_path = beam_processing.passenger_vehicle_types_file
        if passenger_vtypes_path:
            passenger_vehicle_types_source = str(registry.locate_required(
                passenger_vtypes_path, label="impacts.emissions.passenger_vehicle_types_file"
            ))
        else:
            found = _glob_vehicle_types_file(
                region_input_root / (population.vehicle_folder or ""), "atlas"
            )
            passenger_vehicle_types_source = required_local_path(
                found, "impacts.emissions.passenger_vehicle_types_file"
            )
        _register_manifest_input(
            manifest_inputs,
            input_root=input_root,
            key="passenger_vehicle_types_input",
            source_path=passenger_vehicle_types_source,
            relative_target=Path(passenger_vehicle_types_source).name,
            metadata={"artifact_family": "passenger_vehicle_types_input"},
        )
        progress.update(1)

        _set_progress_task(progress, "freight vehicle types", step_label="Preprocess Step 1.2")
        freight_vtypes_path = beam_processing.freight_vehicle_types_file
        if freight_vtypes_path:
            freight_vehicle_types_source = str(registry.locate_required(
                freight_vtypes_path, label="impacts.emissions.freight_vehicle_types_file"
            ))
        else:
            found = _glob_vehicle_types_file(
                region_input_root / (population.vehicle_folder or ""), "frism"
            )
            freight_vehicle_types_source = required_local_path(
                found, "impacts.emissions.freight_vehicle_types_file"
            )
        _register_manifest_input(
            manifest_inputs,
            input_root=input_root,
            key="freight_vehicle_types_input",
            source_path=freight_vehicle_types_source,
            relative_target=Path(freight_vehicle_types_source).name,
            metadata={"artifact_family": "freight_vehicle_types_input"},
        )
        progress.update(1)

        if emissions.vehicle_category_metadata_file:
            _set_progress_task(progress, "vehicle category metadata", step_label="Preprocess Step 1.2")
            vehicle_category_metadata_source = str(registry.locate_required(
                emissions.vehicle_category_metadata_file,
                label="impacts.emissions.vehicle_category_metadata_file",
            ))
            _register_manifest_input(
                manifest_inputs,
                input_root=input_root,
                key="vehicle_category_metadata_file_input",
                source_path=vehicle_category_metadata_source,
                relative_target=Path(vehicle_category_metadata_source).name,
                metadata={"artifact_family": "vehicle_category_metadata_file_input"},
            )
            progress.update(1)
    finally:
        progress.close()

    log_substep_banner("1.3", "register EMFAC inventory inputs", logger=logger)
    local_output_folder = Path(resolve_path(impacts.local_output_folder, config_path)).resolve()
    _inv_folder = presim_activities_inventory_root(
        local_output_folder,
        impacts.scenario or "",
        region=settings.run.region,
        output_run_name=getattr(settings.run, "output_run_name", None),
        run_scenario=settings.run.scenario,
    )
    passenger_inv_path = emissions.inventory.passenger_file
    freight_inv_path = emissions.inventory.freight_file
    if passenger_inv_path:
        passenger_emfacid_source = _resolve_emfacid_path(
            str(registry.locate_required(passenger_inv_path, label="impacts.emissions.inventory.passenger_file")),
            label="passenger",
        )
    else:
        passenger_emfacid_source = required_local_path(
            _find_emfacid_file(_inv_folder, "passenger"),
            "impacts.emissions.inventory.passenger_file (activity-by-emfacid)",
        )
    if freight_inv_path:
        freight_emfacid_source = _resolve_emfacid_path(
            str(registry.locate_required(freight_inv_path, label="impacts.emissions.inventory.freight_file")),
            label="freight",
        )
    else:
        freight_emfacid_source = required_local_path(
            _find_emfacid_file(_inv_folder, "freight"),
            "impacts.emissions.inventory.freight_file (activity-by-emfacid)",
        )
    staged_passenger_inventory_file = None
    staged_freight_inventory_file = None
    substep_13_total = 2  # passenger emfacid, freight emfacid
    if emissions.inventory.enable_passenger_activity_correction:
        substep_13_total += 1
    if emissions.inventory.enable_freight_activity_correction:
        substep_13_total += 1
    if settings.impacts.analysis.inventory_targets:
        substep_13_total += 1
    progress = make_progress("Preprocess Step 1.3", total=substep_13_total, unit="task", leave=False)
    try:
        if emissions.inventory.enable_passenger_activity_correction:
            _set_progress_task(progress, "passenger inventory", step_label="Preprocess Step 1.3")
            passenger_inventory_source = str(registry.locate_required(
                passenger_inv_path, label="impacts.emissions.inventory.passenger_file"
            ))
            staged_passenger_inventory_file = _register_manifest_input(
                manifest_inputs,
                input_root=input_root,
                key="passenger_inventory_file",
                source_path=passenger_inventory_source,
                relative_target=Path(passenger_inventory_source).name,
                metadata={"artifact_family": "passenger_inventory_file"},
            )
            progress.update(1)

        _set_progress_task(progress, "passenger emfac inventory", step_label="Preprocess Step 1.3")
        _register_manifest_input(
            manifest_inputs,
            input_root=input_root,
            key="passenger_inventory_emfacid_file",
            source_path=passenger_emfacid_source,
            relative_target=Path(passenger_emfacid_source).name,
            metadata={"artifact_family": "passenger_inventory_emfacid_file"},
        )
        progress.update(1)

        if emissions.inventory.enable_freight_activity_correction:
            _set_progress_task(progress, "freight inventory", step_label="Preprocess Step 1.3")
            freight_inventory_source = str(registry.locate_required(
                freight_inv_path, label="impacts.emissions.inventory.freight_file"
            ))
            staged_freight_inventory_file = _register_manifest_input(
                manifest_inputs,
                input_root=input_root,
                key="freight_inventory_file",
                source_path=freight_inventory_source,
                relative_target=Path(freight_inventory_source).name,
                metadata={"artifact_family": "freight_inventory_file"},
            )
            progress.update(1)

        _set_progress_task(progress, "freight emfac inventory", step_label="Preprocess Step 1.3")
        _register_manifest_input(
            manifest_inputs,
            input_root=input_root,
            key="freight_inventory_emfacid_file",
            source_path=freight_emfacid_source,
            relative_target=Path(freight_emfacid_source).name,
            metadata={"artifact_family": "freight_inventory_emfacid_file"},
        )
        progress.update(1)

        if settings.impacts.analysis.inventory_targets:
            _set_progress_task(progress, "emissions inventory", step_label="Preprocess Step 1.3")
            _tmp_root = presim_activities_tmp_root(
                local_output_folder,
                region=settings.run.region,
                output_run_name=getattr(settings.run, "output_run_name", None),
                run_scenario=settings.run.scenario,
            )
            region_slug = str(settings.run.region).lower()
            year = int(settings.run.start_year)
            emissions_inv_path = _tmp_root / f"{region_slug}-emfac-{year}-inventory-intermediate-with-activity.parquet"
            required_local_path(emissions_inv_path, "emissions_inventory_file (inventory-intermediate-with-activity)")
            _register_manifest_input(
                manifest_inputs,
                input_root=input_root,
                key="emissions_inventory_file",
                source_path=str(emissions_inv_path),
                relative_target=emissions_inv_path.name,
                metadata={"artifact_family": "emissions_inventory_file"},
            )
            progress.update(1)
    finally:
        progress.close()

    if pipeline.inmap or pipeline.aermod:
        log_substep_banner("1.4", "register dispersion inputs", logger=logger)
        substep_14_total = 0
        if pipeline.inmap:
            substep_14_total += 3  # grid, nox-to-no2 ratios, isrm store
        if pipeline.aermod:
            substep_14_total += 1  # asrv patterns
        progress = make_progress("Preprocess Step 1.4", total=substep_14_total, unit="task", leave=False)
        try:
            if pipeline.inmap:
                _set_progress_task(progress, "inmap grid", step_label="Preprocess Step 1.4")
                inmap_grid_source = str(registry.locate_required(
                    inmap.grid_path, label="impacts.dispersions.inmap.grid_path"
                ))
                staged_inmap_grid = _register_manifest_input(
                    manifest_inputs,
                    input_root=input_root,
                    key="inmap_grid",
                    source_path=inmap_grid_source,
                    relative_target=inmap.grid_path,
                    metadata={"artifact_family": "inmap_grid"},
                )
                progress.update(1)

                _set_progress_task(progress, "nox-to-no2 ratios", step_label="Preprocess Step 1.4")
                no2_matrix_source = str(registry.locate_required(
                    inmap.isrm_nox_to_no2_ratios_file,
                    label="impacts.dispersions.inmap.isrm_nox_to_no2_ratios_file",
                ))
                staged_isrm_nox_to_no2_ratios_file = _register_manifest_input(
                    manifest_inputs,
                    input_root=input_root,
                    key="isrm_nox_to_no2_ratios_file",
                    source_path=no2_matrix_source,
                    relative_target=inmap.isrm_nox_to_no2_ratios_file,
                    metadata={"artifact_family": "isrm_nox_to_no2_ratios_file"},
                )
                progress.update(1)

                _set_progress_task(progress, "isrm store", step_label="Preprocess Step 1.4")
                isrm_source = str(registry.locate_required(
                    inmap.isrm_zarr, label="impacts.dispersions.inmap.isrm_zarr"
                ))
                staged_isrm = _register_manifest_input(
                    manifest_inputs,
                    input_root=input_root,
                    key="isrm",
                    source_path=isrm_source,
                    relative_target=Path(isrm_source).name,
                    metadata={"artifact_family": "isrm"},
                    prefer_reference=False,
                )
                progress.update(1)

            if pipeline.aermod:
                _set_progress_task(progress, "asrv patterns", step_label="Preprocess Step 1.4")
                asrv_source = str(registry.locate_required(
                    aermod.asrv_patterns_file, label="impacts.dispersions.aermod.asrv_patterns_file"
                ))
                staged_asrv_patterns_file = _register_manifest_input(
                    manifest_inputs,
                    input_root=input_root,
                    key="asrv_patterns_file",
                    source_path=asrv_source,
                    relative_target=aermod.asrv_patterns_file,
                    metadata={"artifact_family": "asrv_patterns_file"},
                )
                progress.update(1)
        finally:
            progress.close()

    population_inputs: Dict[str, Any] = {}
    if pipeline.exposure:
        log_substep_banner("1.5", "register exposure population inputs", logger=logger)
        population_root = region_input_root / str(population.passenger_folder)
        population_artifact_entries = {
            "persons": find_latest_beam_population_reference(optional=True),
            "households": find_latest_beam_households_reference(optional=True),
        }
        population_artifact_keys = {
            "persons": BEAM_POPULATION_PREFIX,
            "households": BEAM_HOUSEHOLDS_PREFIX,
        }
        progress = make_progress("Preprocess Step 1.5", total=2, unit="task", leave=False)
        try:
            for stem in ("persons", "households"):
                _set_progress_task(progress, stem, step_label="Preprocess Step 1.5")
                existing_entry = population_artifact_entries[stem]
                if existing_entry is not None:
                    population_inputs[stem] = existing_entry
                    progress.update(1)
                    continue
                source_path = _locate_exchange_file(population_root, stem)
                if source_path:
                    register_managed_input(
                        manifest_inputs=population_inputs,
                        input_root=input_root,
                        key=stem,
                        source_path=source_path,
                        relative_target=str(Path(population.passenger_folder) / Path(source_path).name),
                        artifact_key=population_artifact_keys[stem],
                        prefer_reference=True,
                        metadata={"artifact_family": population_artifact_keys[stem]},
                    )
                progress.update(1)
        finally:
            progress.close()

    logger.info("Preprocess Step 1 complete")
    return {
        "staged_network": staged_network,
        "staged_osm": staged_osm,
        "staged_skims": staged_skims,
        "staged_events": staged_events,
        "staged_inmap_grid": staged_inmap_grid,
        "staged_passenger_inventory_file": staged_passenger_inventory_file,
        "staged_freight_inventory_file": staged_freight_inventory_file,
        "staged_isrm": staged_isrm,
        "staged_isrm_nox_to_no2_ratios_file": staged_isrm_nox_to_no2_ratios_file,
        "staged_asrv_patterns_file": staged_asrv_patterns_file,
        "population_inputs": population_inputs,
    }
