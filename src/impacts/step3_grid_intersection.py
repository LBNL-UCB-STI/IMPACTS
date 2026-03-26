"""Step 3 — Grid intersection.

Intersect the buffered road network with the AERMOD grid, the inMAP grid,
and county boundaries to produce a fully-labeled intersection table.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .manifest_models import PipelineConfig

logger = logging.getLogger(__name__)


def _read_vector(path: str) -> gpd.GeoDataFrame:
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        return gpd.read_parquet(target)
    return gpd.read_file(target)


def run(pipeline: PipelineConfig, raw_dir: Path, buffered_network: gpd.GeoDataFrame) -> str:
    """Intersect buffered network with AERMOD, inMAP, and county grids.

    Returns path to the labeled grid intersection parquet.
    """
    import osm_chordify

    grid_intersection_path = raw_dir / "beam_osm_aermod_inmap_county_intersection.parquet"
    epsg = int(pipeline.output_epsg)

    # Step 3.1: buffered network × AERMOD grid → labeled aermod_*
    logger.info("Step 3.1: intersecting network with AERMOD grid %s", pipeline.aermod_grid_path)
    A = osm_chordify.intersect_road_polygons_with_zones(
        buffered_network,
        epsg,
        pipeline.aermod_grid_path,
        output_epsg=epsg,
        zone_label="aermod",
    )
    logger.info("Step 3.1 complete: %d rows", len(A))

    # Step 3.2: A × inMAP grid → labeled inmap_* (aermod_* preserved)
    logger.info("Step 3.2: intersecting with inMAP grid %s", pipeline.inmap_grid_path)
    B = osm_chordify.intersect_polygons_with_zones(
        A,
        epsg,
        pipeline.inmap_grid_path,
        output_epsg=epsg,
        zone_label="inmap",
    )
    logger.info("Step 3.2 complete: %d rows", len(B))

    # Step 3.3: B × county boundaries → labeled county_*
    if not pipeline.county_boundaries_path:
        raise ValueError("Step 3 requires pipeline.county_boundaries_path from preprocess.")
    county_gdf = _read_vector(pipeline.county_boundaries_path)
    if county_gdf.crs is not None:
        county_gdf = county_gdf.to_crs(epsg=epsg)
    logger.info("Step 3.3: intersecting with county boundaries")
    C_matched = osm_chordify.intersect_polygons_with_zones(
        B, epsg, county_gdf, output_epsg=epsg, zone_label="county",
    )
    logger.info("Step 3.3 complete: %d matched rows", len(C_matched))

    # Step 3.4: left join to identify B rows with no county match
    logger.info("Step 3.4: identifying unmatched rows")
    C_left = osm_chordify.spatial_left_join_with_zones(
        B, epsg, county_gdf, output_epsg=epsg, zone_label="county",
    )
    county_cols = [c for c in C_left.columns if c.startswith("county_") and c != "geometry"]
    C_unmatched = C_left[C_left[county_cols].isna().all(axis=1)].copy()
    logger.info("Step 3.4 complete: %d unmatched rows", len(C_unmatched))

    # Step 3.5: union matched + unmatched (null county_*)
    C = gpd.GeoDataFrame(
        pd.concat([C_matched, C_unmatched], ignore_index=True),
        geometry="geometry",
        crs=C_matched.crs,
    )
    C.to_parquet(grid_intersection_path, index=False)
    C.to_file(grid_intersection_path.with_suffix(".gpkg"), driver="GPKG")
    logger.info("Step 3 complete: %d total rows → %s", len(C), grid_intersection_path)

    return str(grid_intersection_path)
