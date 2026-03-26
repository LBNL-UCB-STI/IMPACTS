from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import geopandas as gpd
import pandas as pd

from .contract_utils import parquet_available
from .contract_utils import load_structured_file
from .contract_utils import write_structured_file
from .manifest_models import InputsManifest
from .manifest_models import PipelineConfig
from .manifest_models import RunManifest

logger = logging.getLogger(__name__)


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _table_path(parent: Path, stem: str) -> Path:
    suffix = ".parquet" if parquet_available() else ".csv.gz"
    return _ensure_parent(parent / f"{stem}{suffix}")


def _bbox_subgrid(
    grid_path: str,
    grid_epsg: int,
    network_gdf: gpd.GeoDataFrame,
    output_path: Path,
    output_epsg: int,
) -> gpd.GeoDataFrame:
    """Step 1: filter grid to network bounding box using zone-zone intersection."""
    import osm_chordify
    from shapely.geometry import box as shapely_box

    network_epsg = network_gdf.crs.to_epsg()
    minx, miny, maxx, maxy = network_gdf.total_bounds
    bbox_gdf = gpd.GeoDataFrame(
        geometry=[shapely_box(minx, miny, maxx, maxy)],
        crs=network_gdf.crs,
    )
    subgrid = osm_chordify.intersect_zones_with_zones(
        grid_path,
        grid_epsg,
        bbox_gdf,
        network_epsg,
        output_epsg=output_epsg,
    )
    # intersect_zones_with_zones prefixes zone_a columns with "zone_a_".
    # Strip that prefix so the subgrid retains the original grid column names
    # and downstream road-grid intersection doesn't produce double-prefixed columns.
    subgrid = subgrid.rename(
        columns={c: c[len("zone_a_"):] for c in subgrid.columns if c.startswith("zone_a_")}
    )
    subgrid.to_parquet(output_path, index=False)
    return subgrid


def _land_subgrid(
    bbox_subgrid: gpd.GeoDataFrame,
    land_mask: gpd.GeoDataFrame,
    output_epsg: int,
) -> gpd.GeoDataFrame:
    """Step 3: filter bbox subgrid to land cells using zone-zone intersection."""
    import osm_chordify

    return osm_chordify.intersect_zones_with_zones(
        bbox_subgrid,
        output_epsg,
        land_mask,
        output_epsg,
        output_epsg=output_epsg,
    )


def _fuse_with_land_subgrid(
    intersection_gdf: gpd.GeoDataFrame,
    land_subgrid: gpd.GeoDataFrame,
    output_epsg: int,
    proportion_col: str = "zone_edge_proportion",
) -> gpd.GeoDataFrame:
    """Step 4: fuse road-grid intersection with land subgrid.

    - Non-voided rows (roads × cells) are kept as-is.
    - Voided rows (cells with no roads) are filtered to only those overlapping
      the land subgrid — land cells with no roads are kept with null link/emission
      columns so they can be rendered separately (greyed out) in visualizations.
    - Water and out-of-study-area voided cells are dropped.
    """
    import osm_chordify

    non_voided = intersection_gdf[intersection_gdf[proportion_col].notna()]
    voided = intersection_gdf[intersection_gdf[proportion_col].isna()].copy()

    if voided.empty:
        return intersection_gdf

    voided = voided.reset_index(drop=True)
    voided["vid"] = voided.index

    land_overlap = osm_chordify.intersect_zones_with_zones(
        voided,
        output_epsg,
        land_subgrid,
        output_epsg,
        output_epsg=output_epsg,
    )

    if land_overlap.empty:
        return non_voided.reset_index(drop=True)

    keep_vids = set(land_overlap["zone_a_vid"])
    voided_on_land = voided[voided["vid"].isin(keep_vids)].drop(columns=["vid"])

    return pd.concat([non_voided, voided_on_land], ignore_index=True)


def _buffer_network_by_lanes(
    network_gdf: gpd.GeoDataFrame,
    output_path: Path,
    lane_width_m: float = 4.0,
    lanes_col: str = "numberOfLanes",
) -> gpd.GeoDataFrame:
    """Buffer each road link into a rectangular polygon corridor.

    Half-width per side = (lanes × lane_width_m) / 2, with flat caps so the
    result is rectangular rather than rounded.
    """
    buffered = network_gdf.copy()
    lanes = pd.to_numeric(buffered[lanes_col], errors="coerce").fillna(1.0).clip(lower=1.0)
    half_width = (lanes * lane_width_m) / 2.0
    buffered.geometry = gpd.GeoSeries(
        [geom.buffer(d, cap_style=2, join_style=2) for geom, d in zip(buffered.geometry, half_width)],
        crs=buffered.crs,
    )
    buffered.to_parquet(output_path, index=False)
    buffered.to_file(output_path.with_suffix(".gpkg"), driver="GPKG")
    return buffered


def _load_grid_geometries(grid_path: str) -> gpd.GeoDataFrame:
    path = Path(grid_path)
    if path.suffix.lower() == ".parquet":
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def _resolve_grid_join_columns(
    allocated_df: pd.DataFrame,
    grid_gdf: gpd.GeoDataFrame,
) -> tuple[str, str]:
    for left_col in allocated_df.columns:
        if left_col in grid_gdf.columns:
            return left_col, left_col
        base_col = left_col[5:] if left_col.startswith("zone_") else left_col
        if base_col in grid_gdf.columns:
            return left_col, base_col

    shared_candidates = ["grid_id", "isrm", "GRID", "cell_id", "OBJECTID", "objectid", "ID", "id"]
    for left_col in ["zone_grid_id", "zone_isrm", "zone_GRID", "zone_grid100", "cell_id", "GRID", "grid_id"]:
        if left_col not in allocated_df.columns:
            continue
        for right_col in shared_candidates:
            if right_col in grid_gdf.columns:
                return left_col, right_col

    raise ValueError(
        f"Could not resolve a grid id join between allocated emissions and grid geometry.\n"
        f"  allocated_df columns: {list(allocated_df.columns)}\n"
        f"  grid_gdf columns: {list(grid_gdf.columns)}"
    )


def _write_geoparquet_allocation(
    allocated_df: pd.DataFrame,
    *,
    grid_path: str,
    output_path: Path,
    output_epsg: int,
) -> None:
    grid_gdf = _load_grid_geometries(grid_path)
    if grid_gdf.crs is None:
        raise ValueError(f"Grid geometry is missing CRS: {grid_path}")
    grid_gdf = grid_gdf.to_crs(epsg=output_epsg)

    left_col, right_col = _resolve_grid_join_columns(allocated_df, grid_gdf)
    geometry_cols = [c for c in grid_gdf.columns if c != "geometry"]
    joined = allocated_df.merge(
        grid_gdf[geometry_cols + ["geometry"]],
        how="left",
        left_on=left_col,
        right_on=right_col,
    )
    geo = gpd.GeoDataFrame(joined, geometry="geometry", crs=grid_gdf.crs)
    geo.to_parquet(output_path, index=False)
    gpkg_path = Path(output_path).with_suffix(".gpkg")
    geo.to_file(gpkg_path, driver="GPKG")


def _load_rates(rates_dir: Optional[str]):
    if not rates_dir:
        return None
    from impacts.emissions.events_to_skims_emissions import read_rates_directory

    return read_rates_directory(rates_dir)


def _build_mapping_from_staged_inputs(pipeline: PipelineConfig, raw_dir: Path) -> str:
    from impacts.network2grid.network_grid_clipping import intersect_beam_osm_with_counties
    from impacts.network2grid.network_grid_clipping import intersect_beam_osm_with_grid
    from impacts.network2grid.network_grid_clipping import map_beam_network_to_osm

    beam_osm_path = raw_dir / "beam_osm_mapped.parquet"
    county_mapping_path = raw_dir / "beam_osm_county_intersection.parquet"
    mapping_path = raw_dir / "beam_osm_inmap_grid_intersection.parquet"
    fine_mapping_path = raw_dir / "beam_osm_aermod_grid_intersection.parquet"

    osm_source = pipeline.osm_links_path or pipeline.osm_pbf_path
    if not osm_source:
        raise ValueError("Mapping build requires staged osm_links_path or osm_pbf_path.")

    # Stage 1: Map BEAM network to OSM
    logger.info("Stage 1/5: mapping BEAM network to OSM using %s", osm_source)
    beam_osm_mapped = map_beam_network_to_osm(
        osm_path=osm_source,
        beam_network_path=pipeline.beam_network_path,
        output_path=str(beam_osm_path),
        network_osm_id_col=pipeline.beam_osm_id_col,
        output_epsg=int(pipeline.output_epsg),
    )
    logger.info("Stage 1/5 complete: wrote %s", beam_osm_path)
    beam_osm_mapped.to_file(beam_osm_path.with_suffix(".gpkg"), driver="GPKG")

    beam_osm_buffered_path = raw_dir / "beam_osm_buffered.parquet"
    logger.info("Stage 1/5: buffering network links by lane width (4m/lane)")
    beam_osm_buffered = _buffer_network_by_lanes(
        network_gdf=beam_osm_mapped,
        output_path=beam_osm_buffered_path,
    )
    logger.info("Stage 1/5: buffered network written to %s", beam_osm_buffered_path)

    # Stage 2: Intersect with counties (optional)
    county_state_fips = pipeline.county_state_fips
    county_fips_codes = list(pipeline.county_fips_codes or [])
    network_gdf = beam_osm_mapped
    county_input = str(beam_osm_path)

    if county_state_fips and county_fips_codes:
        logger.info(
            "Stage 2/5: intersecting mapped BEAM/OSM network with counties state_fips=%s county_fips=%s",
            county_state_fips,
            county_fips_codes,
        )
        county_mapped = intersect_beam_osm_with_counties(
            beam_osm_path=str(beam_osm_path),
            state_fips=str(county_state_fips),
            county_fips_codes=county_fips_codes,
            output_path=str(county_mapping_path),
            beam_osm_epsg=int(pipeline.output_epsg),
            output_epsg=int(pipeline.output_epsg),
            area_name=str(pipeline.region or pipeline.county_area_name),
            boundary_year=int(pipeline.start_year or 2023),
        )
        county_mapped.to_file(county_mapping_path.with_suffix(".gpkg"), driver="GPKG")
        county_input = str(county_mapping_path)
        network_gdf = county_mapped
        logger.info("Stage 2/5 complete: wrote %s", county_mapping_path)
    else:
        logger.info("Stage 2/5 skipped: county FIPS settings not configured")

    # Build land mask (if county FIPS configured)
    land_mask = None
    if county_state_fips and county_fips_codes:
        import osm_chordify
        logger.info("Building land mask from cartographic county boundaries")
        land_mask = osm_chordify.build_area_mask_from_counties(
            state_fips_code=str(county_state_fips),
            county_fips_codes=[str(c) for c in county_fips_codes],
            year=int(pipeline.start_year or 2023),
            work_dir=str(raw_dir),
            output_epsg=int(pipeline.output_epsg),
            include_water=False,
        )

    # Stage 3: inMAP grid — 4-step intersection
    inmap_grid_path = pipeline.inmap_grid_path
    logger.info("Stage 3/5: intersecting mapped network with inmap_grid %s", inmap_grid_path)

    # Step 1 — bbox subgrid: filter inMAP grid to network bounding box
    inmap_bbox_subgrid_path = raw_dir / "inmap_grid_bbox_subgrid.parquet"
    inmap_bbox_subgrid = _bbox_subgrid(
        grid_path=inmap_grid_path,
        grid_epsg=int(pipeline.inmap_grid_epsg),
        network_gdf=network_gdf,
        output_path=inmap_bbox_subgrid_path,
        output_epsg=int(pipeline.output_epsg),
    )

    # Step 2 — road-grid intersection: intersect roads with bbox subgrid
    inmap_raw_intersection_path = raw_dir / "beam_osm_inmap_grid_intersection_raw.parquet"
    inmap_intersection = intersect_beam_osm_with_grid(
        beam_osm_path=county_input,
        grid_cells_path=str(inmap_bbox_subgrid_path),
        output_path=str(inmap_raw_intersection_path),
        beam_osm_epsg=int(pipeline.output_epsg),
        grid_epsg=int(pipeline.output_epsg),
        output_epsg=int(pipeline.output_epsg),
        beam_length_col=pipeline.beam_length_col,
    )

    if land_mask is not None:
        # Step 3 — land subgrid: filter bbox subgrid to land cells only
        inmap_land_subgrid = _land_subgrid(
            bbox_subgrid=inmap_bbox_subgrid,
            land_mask=land_mask,
            output_epsg=int(pipeline.output_epsg),
        )
        # Step 4 — fuse: keep road rows as-is, keep land-only voided cells, drop water/outside
        inmap_mapped = _fuse_with_land_subgrid(
            intersection_gdf=inmap_intersection,
            land_subgrid=inmap_land_subgrid,
            output_epsg=int(pipeline.output_epsg),
        )
    else:
        inmap_mapped = inmap_intersection

    inmap_mapped.to_parquet(mapping_path, index=False)
    inmap_mapped.to_file(mapping_path.with_suffix(".gpkg"), driver="GPKG")
    logger.info("Stage 3/5 complete: wrote %s", mapping_path)

    # Stage 4: AERMOD grid — buffered network polygon intersection (optional)
    aermod_grid_path = pipeline.aermod_grid_path
    if aermod_grid_path:
        import osm_chordify
        logger.info("Stage 4/5: intersecting buffered network with aermod_grid %s", aermod_grid_path)

        # Step 1 — bbox subgrid: filter aermod grid to buffered network extent
        aermod_bbox_subgrid_path = raw_dir / "aermod_grid_bbox_subgrid.parquet"
        _bbox_subgrid(
            grid_path=aermod_grid_path,
            grid_epsg=int(pipeline.aermod_grid_epsg),
            network_gdf=beam_osm_buffered,
            output_path=aermod_bbox_subgrid_path,
            output_epsg=int(pipeline.output_epsg),
        )

        # Step 2 — intersect buffered network polygons (zone_a) with aermod subgrid (zone_b)
        buffered_for_intersect = beam_osm_buffered.copy()
        buffered_for_intersect["link_buffer_area"] = buffered_for_intersect.geometry.area

        aermod_result = osm_chordify.intersect_zones_with_zones(
            buffered_for_intersect,
            int(pipeline.output_epsg),
            aermod_bbox_subgrid_path,
            int(pipeline.output_epsg),
            output_epsg=int(pipeline.output_epsg),
        )

        # Rename: zone_a_* → edge_*, zone_b_* → zone_*
        aermod_result = aermod_result.rename(columns={
            col: "edge_" + col[len("zone_a_"):] if col.startswith("zone_a_") else
                 "zone_" + col[len("zone_b_"):] if col.startswith("zone_b_") else col
            for col in aermod_result.columns
        })

        # zone_edge_proportion = fraction of buffered corridor within each cell
        aermod_result["zone_edge_proportion"] = (
            aermod_result.geometry.area / aermod_result["edge_link_buffer_area"]
        ).clip(0.0, 1.0)
        aermod_result = aermod_result.drop(columns=["edge_link_buffer_area"])

        if "edge_linkLength" in aermod_result.columns:
            aermod_result["edge_link_length_m"] = pd.to_numeric(aermod_result["edge_linkLength"], errors="coerce")
            aermod_result["zone_link_length_m"] = aermod_result["edge_link_length_m"] * aermod_result["zone_edge_proportion"]

        aermod_result.to_parquet(fine_mapping_path, index=False)
        aermod_result.to_file(fine_mapping_path.with_suffix(".gpkg"), driver="GPKG")
        logger.info("Stage 4/5 complete: wrote %s", fine_mapping_path)

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
    manifest = InputsManifest.from_dict(load_structured_file(input_manifest_path)).to_dict()
    pipeline = PipelineConfig.from_dict(manifest.get("pipeline", {}) or {})
    population_inputs = manifest.get("population_inputs", {}) or {}

    output_root = Path(output_dir).resolve()
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    county_mapping_candidate = raw_dir / "beam_osm_county_intersection.parquet"
    logger.info("Loaded input manifest: %s", Path(input_manifest_path).resolve())
    logger.info("Output directory: %s", output_root)

    from impacts.emissions.emissions_grid_mapping import aggregate_allocated_intersection_rows
    from impacts.emissions.emissions_grid_mapping import apply_activity_corrections
    from impacts.emissions.emissions_grid_mapping import map_skims_emissions_to_intersection
    from impacts.emissions.emissions_grid_mapping import plot_county_pm25_comparison
    from impacts.emissions.events_to_skims_emissions import build_skims_emissions_from_events
    from impacts.emissions.events_to_skims_emissions import write_skims_emissions
    from impacts.dispersion.isrm_dispersion import run_dispersion_from_file

    mapping_input_path = pipeline.mapping_input_path
    if not mapping_input_path:
        mapping_input_path = _build_mapping_from_staged_inputs(pipeline, raw_dir)
    else:
        logger.info("Using staged mapping input: %s", mapping_input_path)

    if pipeline.prepared_skims_input_path or pipeline.skims_input_path:
        source = Path(pipeline.prepared_skims_input_path or pipeline.skims_input_path)
        if not source.exists():
            raise FileNotFoundError(f"Configured staged skims input not found: {source}")
        skims_path = _ensure_parent(raw_dir / source.name)
        logger.info("Stage 4/5: using staged skims input %s", source)
        if source.resolve() != skims_path.resolve():
            skims_path.write_bytes(source.read_bytes())
        logger.info("Stage 4/5 complete: raw skims available at %s", skims_path)
    else:
        skims_path = _table_path(raw_dir, "skims_emissions")
        if not pipeline.events_path:
            raise ValueError("Runner requires either staged skims_input_path or staged events_path")
        logger.info("Stage 4/5: building skims from staged events %s", pipeline.events_path)
        rates_df = None
        if pipeline.use_rates:
            rates_df = _load_rates(pipeline.rates_dir)
            logger.info("Loaded rates from %s", pipeline.rates_dir)
        skims = build_skims_emissions_from_events(
            events_path=pipeline.events_path,
            network_path=pipeline.link_length_path,
            rates_df=rates_df,
            iterations=int(pipeline.iterations),
            events_columns=pipeline.events_columns,
            network_columns=pipeline.network_columns,
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
            skims_columns=pipeline.skims_columns,
            mapping_columns=pipeline.mapping_columns,
        )
        logger.info("Stage 5/7 complete: wrote %s", county_allocated_raw_path)

        if pipeline.activity_corrections_path:
            county_allocated_corrected_path = _table_path(raw_dir, "emissions_county_allocated_corrected")
            logger.info("Stage 6/7: applying county process corrections to county-broken emissions")
            county_allocated_corrected = apply_activity_corrections(
                county_allocated_raw,
                str(pipeline.activity_corrections_path),
                correction_columns=pipeline.activity_corrections_columns,
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

            plot_path = raw_dir / "county_pm25_before_after_correction.png"
            logger.info("Stage 6/7: generating county PM2.5 before/after correction plot")
            plot_county_pm25_comparison(
                county_allocated_raw,
                county_allocated_corrected,
                output_path=str(plot_path),
            )
            logger.info("Stage 6/7: county PM2.5 plot saved to %s", plot_path)

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
        skims_columns=pipeline.skims_columns,
        mapping_columns=pipeline.mapping_columns,
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
        _write_geoparquet_allocation(
            grid_allocated,
            grid_path=str(pipeline.inmap_grid_path),
            output_path=grid_allocated_path,
            output_epsg=int(pipeline.output_epsg),
        )
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
            skims_columns=pipeline.skims_columns,
            mapping_columns=pipeline.mapping_columns,
        )
        fine_grid_allocated = aggregate_allocated_intersection_rows(
            fine_grid_allocated,
            group_cols=[
                "linkId",
                "vehicleTypeId",
                "cell_id",
                "GRID",
                "zone_grid_id",
            ],
        )
        if fine_grid_allocated_path.suffix.lower() == ".parquet":
            _write_geoparquet_allocation(
                fine_grid_allocated,
                grid_path=str(pipeline.aermod_grid_path),
                output_path=fine_grid_allocated_path,
                output_epsg=int(pipeline.output_epsg),
            )
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
            isrm_url=pipeline.isrm_url,
            factor=float(pipeline.concentration_factor or 28766.639),
            include_bc=bool(pipeline.include_bc),
            include_health=bool(pipeline.include_health),
            emissions_columns=pipeline.dispersion_emissions_columns,
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
        "pipeline": pipeline.to_dict(),
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
    typed_manifest = RunManifest.from_dict(run_manifest)
    write_structured_file(output_manifest, typed_manifest.to_dict())
    logger.info("Run manifest written: %s", output_manifest)
    return typed_manifest.to_dict()


def run_from_runtime_config(
    runtime_config_path: str | Path,
    workspace: str | Path,
    run_manifest_path: str | Path | None = None,
    run_dispersion: bool = True,
) -> Dict[str, Any]:
    from impacts.preprocessor import preprocess_workflow

    workspace_root = Path(workspace).resolve()
    preprocess_manifest = preprocess_workflow(
        runtime_config_path=runtime_config_path,
        staging_dir=workspace_root,
    )
    return run_from_input_manifest(
        input_manifest_path=preprocess_manifest["inputs_manifest_path"],
        output_dir=workspace_root / "output",
        run_manifest_path=run_manifest_path,
        run_dispersion=run_dispersion,
    )
