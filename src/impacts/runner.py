from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from .contract_utils import parquet_available
from .contract_utils import load_structured_file
from .contract_utils import write_structured_file

logger = logging.getLogger(__name__)


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _table_path(parent: Path, stem: str) -> Path:
    suffix = ".parquet" if parquet_available() else ".csv.gz"
    return _ensure_parent(parent / f"{stem}{suffix}")


def _load_rates(rates_dir: Optional[str]):
    if not rates_dir:
        return None
    from impacts.emissions.events_to_skims_emissions import read_rates_directory

    return read_rates_directory(rates_dir)


def _build_mapping_from_staged_inputs(pipeline: Dict[str, Any], raw_dir: Path) -> str:
    from impacts.network2grid.network_grid_clipping import intersect_beam_osm_with_counties
    from impacts.network2grid.network_grid_clipping import intersect_beam_osm_with_grid
    from impacts.network2grid.network_grid_clipping import map_beam_network_to_osm

    beam_osm_path = raw_dir / "beam_osm_mapped.parquet"
    county_mapping_path = raw_dir / "beam_osm_county_intersection.parquet"
    mapping_path = raw_dir / "beam_osm_inmap_grid_intersection.parquet"
    fine_mapping_path = raw_dir / "beam_osm_aermod_grid_intersection.parquet"
    osm_source = pipeline.get("osm_links_path") or pipeline.get("osm_pbf_path")
    if not osm_source:
        raise ValueError("Mapping build requires staged osm_links_path or osm_pbf_path.")
    logger.info("Stage 1/5: mapping BEAM network to OSM using %s", osm_source)
    map_beam_network_to_osm(
        osm_path=osm_source,
        beam_network_path=pipeline["beam_network_path"],
        output_path=str(beam_osm_path),
        network_osm_id_col=pipeline["beam_osm_id_col"],
    )
    logger.info("Stage 1/5 complete: wrote %s", beam_osm_path)

    county_input = str(beam_osm_path)
    county_state_fips = pipeline.get("county_state_fips")
    county_fips_codes = list(pipeline.get("county_fips_codes", []) or [])
    if county_state_fips and county_fips_codes:
        logger.info(
            "Stage 2/5: intersecting mapped BEAM/OSM network with counties state_fips=%s county_fips=%s",
            county_state_fips,
            county_fips_codes,
        )
        intersect_beam_osm_with_counties(
            beam_osm_path=str(beam_osm_path),
            state_fips=str(county_state_fips),
            county_fips_codes=county_fips_codes,
            output_path=str(county_mapping_path),
            beam_osm_epsg=int(pipeline["beam_osm_epsg"]),
            output_epsg=int(pipeline["output_epsg"]),
            area_name=str(pipeline.get("county_area_name", "county")),
        )
        county_input = str(county_mapping_path)
        logger.info("Stage 2/5 complete: wrote %s", county_mapping_path)
    else:
        logger.info("Stage 2/5 skipped: county FIPS settings not configured")

    inmap_grid_path = pipeline["inmap_grid_path"]
    logger.info("Stage 3/5: intersecting mapped network with inmap_grid %s", inmap_grid_path)
    intersect_beam_osm_with_grid(
        beam_osm_path=county_input,
        grid_cells_path=inmap_grid_path,
        output_path=str(mapping_path),
        beam_osm_epsg=int(pipeline["beam_osm_epsg"]),
        grid_epsg=int(pipeline["inmap_grid_epsg"]),
        output_epsg=int(pipeline["output_epsg"]),
        beam_length_col=pipeline["beam_length_col"],
    )
    logger.info("Stage 3/5 complete: wrote %s", mapping_path)
    aermod_grid_path = pipeline.get("aermod_grid_path")
    if aermod_grid_path:
        logger.info("Stage 4/6: intersecting inmap_grid-broken network with aermod_grid %s", aermod_grid_path)
        intersect_beam_osm_with_grid(
            beam_osm_path=str(mapping_path),
            grid_cells_path=aermod_grid_path,
            output_path=str(fine_mapping_path),
            beam_osm_epsg=int(pipeline["output_epsg"]),
            grid_epsg=int(pipeline["aermod_grid_epsg"]),
            output_epsg=int(pipeline["output_epsg"]),
            beam_length_col=pipeline["beam_length_col"],
        )
        logger.info("Stage 4/6 complete: wrote %s", fine_mapping_path)
    return str(mapping_path)


def run_from_input_manifest(
    input_manifest_path: str | Path,
    output_dir: str | Path,
    run_manifest_path: str | Path | None = None,
    run_dispersion: bool = True,
) -> Dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=False,
    )
    manifest = load_structured_file(input_manifest_path)
    pipeline = manifest.get("pipeline", {}) or {}
    population_inputs = manifest.get("population_inputs", {}) or {}

    output_root = Path(output_dir).resolve()
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    county_mapping_candidate = raw_dir / "beam_osm_county_intersection.parquet"
    logger.info("Loaded input manifest: %s", Path(input_manifest_path).resolve())
    logger.info("Output directory: %s", output_root)

    from impacts.emissions.emissions_grid_mapping import aggregate_allocated_intersection_rows
    from impacts.emissions.emissions_grid_mapping import apply_county_process_corrections
    from impacts.emissions.emissions_grid_mapping import map_skims_emissions_to_intersection
    from impacts.emissions.events_to_skims_emissions import build_skims_emissions_from_events
    from impacts.emissions.events_to_skims_emissions import write_skims_emissions
    from impacts.dispersion.isrm_dispersion import run_dispersion_from_file

    mapping_input_path = pipeline.get("mapping_input_path")
    if not mapping_input_path:
        mapping_input_path = _build_mapping_from_staged_inputs(pipeline, raw_dir)
    else:
        logger.info("Using staged mapping input: %s", mapping_input_path)

    if pipeline.get("prepared_skims_input_path") or pipeline.get("skims_input_path"):
        source = Path(pipeline.get("prepared_skims_input_path") or pipeline["skims_input_path"])
        if not source.exists():
            raise FileNotFoundError(f"Configured staged skims input not found: {source}")
        skims_path = _ensure_parent(raw_dir / source.name)
        logger.info("Stage 4/5: using staged skims input %s", source)
        if source.resolve() != skims_path.resolve():
            skims_path.write_bytes(source.read_bytes())
        logger.info("Stage 4/5 complete: raw skims available at %s", skims_path)
    else:
        skims_path = _table_path(raw_dir, "skims_emissions")
        if not pipeline.get("events_path"):
            raise ValueError("Runner requires either staged skims_input_path or staged events_path")
        logger.info("Stage 4/5: building skims from staged events %s", pipeline["events_path"])
        rates_df = None
        if pipeline.get("use_rates", True):
            rates_df = _load_rates(pipeline.get("rates_dir"))
            logger.info("Loaded rates from %s", pipeline.get("rates_dir"))
        skims = build_skims_emissions_from_events(
            events_path=pipeline["events_path"],
            network_path=pipeline.get("link_length_path"),
            rates_df=rates_df,
            iterations=int(pipeline.get("iterations", 0)),
            events_columns=pipeline.get("events_columns"),
            network_columns=pipeline.get("network_columns"),
        )
        write_skims_emissions(skims, str(skims_path))
        logger.info("Stage 4/5 complete: wrote %s", skims_path)

    county_allocated_raw_path = None
    county_allocated_corrected_path = None
    if county_mapping_candidate.exists():
        county_allocated_raw_path = _table_path(raw_dir, "emissions_county_allocated_raw")
        logger.info("Stage 5/7: allocating annualized skims emissions to county-broken links")
        county_allocated_raw = map_skims_emissions_to_intersection(
            skims_path=str(skims_path),
            mapping_path=str(county_mapping_candidate),
            output_path=str(county_allocated_raw_path),
            skims_columns=pipeline.get("skims_columns"),
            mapping_columns=pipeline.get("mapping_columns"),
        )
        logger.info("Stage 5/7 complete: wrote %s", county_allocated_raw_path)

        if pipeline.get("county_correction_factors_path"):
            county_allocated_corrected_path = _table_path(raw_dir, "emissions_county_allocated_corrected")
            logger.info("Stage 6/7: applying county process corrections to county-broken emissions")
            county_allocated_corrected = apply_county_process_corrections(
                county_allocated_raw,
                str(pipeline["county_correction_factors_path"]),
                correction_columns=pipeline.get("county_correction_columns"),
            )
            county_allocated_corrected = aggregate_allocated_intersection_rows(
                county_allocated_corrected,
                group_cols=[
                    "linkId",
                    "vehicleTypeId",
                    "zone_GEOID",
                    "zone_NAME",
                    "zone_COUNTYFP",
                ],
            )
            if county_allocated_corrected_path.suffix.lower() == ".parquet":
                county_allocated_corrected.to_parquet(county_allocated_corrected_path, index=False)
            else:
                county_allocated_corrected.to_csv(county_allocated_corrected_path, index=False, compression="gzip")
            logger.info("Stage 6/7 complete: wrote %s", county_allocated_corrected_path)

    grid_allocated_raw_path = _table_path(raw_dir, "emissions_inmap_grid_allocated_raw")
    grid_allocated_path = _table_path(raw_dir, "emissions_inmap_grid_allocated")
    grid_allocation_input_path = county_allocated_corrected_path or skims_path
    logger.info(
        "%s: allocating %s to inmap_grid intersections",
        "Stage 7/7" if county_allocated_raw_path else "Stage 5/5",
        "county-corrected annual emissions" if county_allocated_corrected_path else "skims emissions",
    )
    grid_allocated = map_skims_emissions_to_intersection(
        skims_path=str(grid_allocation_input_path),
        mapping_path=str(mapping_input_path),
        output_path=str(grid_allocated_raw_path),
        skims_columns=pipeline.get("skims_columns"),
        mapping_columns=pipeline.get("mapping_columns"),
    )
    grid_allocated = aggregate_allocated_intersection_rows(
        grid_allocated,
        group_cols=[
            "linkId",
            "vehicleTypeId",
            "cell_id",
            "GRID",
            "zone_isrm",
        ],
    )
    if grid_allocated_path.suffix.lower() == ".parquet":
        grid_allocated.to_parquet(grid_allocated_path, index=False)
    else:
        grid_allocated.to_csv(grid_allocated_path, index=False, compression="gzip")
    logger.info(
        "%s complete: wrote %s",
        "Stage 7/7" if county_allocated_raw_path else "Stage 5/5",
        grid_allocated_path,
    )

    fine_grid_allocated_raw_path = None
    fine_grid_allocated_path = None
    fine_mapping_candidate = raw_dir / "beam_osm_aermod_grid_intersection.parquet"
    if fine_mapping_candidate.exists():
        fine_grid_allocated_raw_path = _table_path(raw_dir, "emissions_aermod_grid_allocated_raw")
        fine_grid_allocated_path = _table_path(raw_dir, "emissions_aermod_grid_allocated")
        logger.info("Stage 8/8: allocating raw inmap_grid-split emissions to aermod_grid intersections")
        fine_grid_allocated = map_skims_emissions_to_intersection(
            skims_path=str(grid_allocated_raw_path),
            mapping_path=str(fine_mapping_candidate),
            output_path=str(fine_grid_allocated_raw_path),
            skims_columns=pipeline.get("skims_columns"),
            mapping_columns=pipeline.get("mapping_columns"),
        )
        fine_grid_allocated = aggregate_allocated_intersection_rows(
            fine_grid_allocated,
            group_cols=[
                "linkId",
                "vehicleTypeId",
                "cell_id",
                "GRID",
                "zone_grid100",
            ],
        )
        if fine_grid_allocated_path.suffix.lower() == ".parquet":
            fine_grid_allocated.to_parquet(fine_grid_allocated_path, index=False)
        else:
            fine_grid_allocated.to_csv(fine_grid_allocated_path, index=False, compression="gzip")
        logger.info("Stage 8/8 complete: wrote %s", fine_grid_allocated_path)

    concentration_path = None
    if run_dispersion:
        concentration_path = _table_path(raw_dir, "grid_concentration")
        logger.info("Dispersion: computing concentrations from allocated grid emissions")
        run_dispersion_from_file(
            emissions_input_path=str(grid_allocated_path),
            output_path=str(concentration_path),
            isrm_url=pipeline["isrm_url"],
            factor=float(pipeline.get("concentration_factor", 28766.639)),
            include_bc=bool(pipeline.get("include_bc", False)),
            include_health=bool(pipeline.get("include_health", False)),
            emissions_columns=pipeline.get("dispersion_emissions_columns"),
        )
        logger.info("Dispersion complete: wrote %s", concentration_path)
    else:
        logger.info("Dispersion skipped: stopping after emissions allocation")

    run_manifest = {
        "contract_version": manifest.get("contract_version", "1"),
        "model": "impacts",
        "input_manifest_path": str(Path(input_manifest_path).resolve()),
        "output_dir": str(output_root),
        "raw_output_dir": str(raw_dir),
        "command": " ".join(sys.argv),
        "image": os.getenv("IMPACTS_IMAGE", "unknown"),
        "raw_outputs": {
            "skims_emissions": str(skims_path),
            "beam_osm_county_intersection": str(county_mapping_candidate) if county_mapping_candidate.exists() else None,
            "emissions_county_allocated_raw": str(county_allocated_raw_path) if county_allocated_raw_path else None,
            "emissions_county_allocated_corrected": str(county_allocated_corrected_path) if county_allocated_corrected_path else None,
            "mapping_input_used": str(mapping_input_path),
            "beam_osm_inmap_grid_intersection": str(mapping_input_path),
            "emissions_inmap_grid_allocated_raw": str(grid_allocated_raw_path),
            "emissions_inmap_grid_allocated": str(grid_allocated_path),
            "beam_osm_aermod_grid_intersection": str(fine_mapping_candidate) if fine_mapping_candidate.exists() else None,
            "emissions_aermod_grid_allocated_raw": str(fine_grid_allocated_raw_path) if fine_grid_allocated_raw_path else None,
            "emissions_aermod_grid_allocated": str(fine_grid_allocated_path) if fine_grid_allocated_path else None,
            "grid_concentration": str(concentration_path) if concentration_path else None,
        },
        "pipeline": pipeline,
        "population_inputs": population_inputs,
        "deterministic_contract": {
            "uses_only_manifest_paths": True,
            "uses_baked_work_data": False,
        },
        "execution": {
            "dispersion_completed": run_dispersion,
            "stopped_after": "dispersion" if run_dispersion else "emissions_grid_allocation",
        },
    }
    output_manifest = Path(run_manifest_path) if run_manifest_path else output_root / "run_manifest.yaml"
    run_manifest["run_manifest_path"] = str(output_manifest)
    write_structured_file(output_manifest, run_manifest)
    logger.info("Run manifest written: %s", output_manifest)
    return run_manifest


def impacts_run(
    input_manifest_path: str | Path,
    output_dir: str | Path,
    run_manifest_path: str | Path | None = None,
    run_dispersion: bool = True,
) -> Dict[str, Any]:
    return run_from_input_manifest(
        input_manifest_path=input_manifest_path,
        output_dir=output_dir,
        run_manifest_path=run_manifest_path,
        run_dispersion=run_dispersion,
    )
