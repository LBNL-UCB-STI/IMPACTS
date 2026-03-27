"""Step 3 — Grid intersection.

Intersect the mapped road network lines with the AERMOD grid, the inMAP
grid, and county boundaries while keeping only road-intersecting rows.
"""
from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Optional
from typing import Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from .manifest_models import PipelineConfig

logger = logging.getLogger(__name__)
_SOURCE_ROW_ID = "__source_row_id"


def _resolve_source_row_col(df: pd.DataFrame) -> str:
    for candidate in (
        _SOURCE_ROW_ID,
        f"edge_{_SOURCE_ROW_ID}",
        f"edge{_SOURCE_ROW_ID}",
    ):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Expected source row id column '{_SOURCE_ROW_ID}' in columns: {list(df.columns)}")


def _read_vector(path: str) -> gpd.GeoDataFrame:
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        return gpd.read_parquet(target)
    return gpd.read_file(target)


def _union_county_matches_with_unmatched(
    source: gpd.GeoDataFrame,
    matched: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if _SOURCE_ROW_ID not in source.columns:
        raise ValueError(f"Source rows must include {_SOURCE_ROW_ID}")

    matched_source_col = _resolve_source_row_col(matched)
    matched_ids = set(pd.to_numeric(matched[matched_source_col], errors="coerce").dropna().astype(int).tolist())
    unmatched = source.loc[~source[_SOURCE_ROW_ID].isin(matched_ids)].copy()

    county_cols = [c for c in matched.columns if c.startswith("county_") and c != "geometry"]
    for col in county_cols:
        if col not in unmatched.columns:
            dtype = matched[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                fill = np.nan
            else:
                fill = pd.NA
            unmatched[col] = pd.Series([fill] * len(unmatched), index=unmatched.index)

    matched_clean = matched.drop(columns=[matched_source_col], errors="ignore")
    unmatched_clean = unmatched.drop(columns=[_SOURCE_ROW_ID], errors="ignore")
    ordered_cols = list(dict.fromkeys(list(matched_clean.columns) + list(unmatched_clean.columns)))

    for col in ordered_cols:
        if col not in matched_clean.columns:
            matched_clean[col] = pd.Series([pd.NA] * len(matched_clean), index=matched_clean.index)
        if col not in unmatched_clean.columns:
            unmatched_clean[col] = pd.Series([pd.NA] * len(unmatched_clean), index=unmatched_clean.index)
    matched_clean = matched_clean[ordered_cols]
    unmatched_clean = unmatched_clean[ordered_cols]

    return gpd.GeoDataFrame(
        pd.concat([matched_clean, unmatched_clean], ignore_index=True),
        geometry="geometry",
        crs=matched.crs,
    )
def run(
    pipeline: PipelineConfig,
    raw_dir: Path,
    mapped_network: gpd.GeoDataFrame,
) -> Tuple[str, Optional[gpd.GeoDataFrame]]:
    """Intersect mapped network lines with AERMOD, inMAP, and county grids.

    Returns path to the labeled grid intersection parquet and the in-memory
    GeoDataFrame when this step built it during the current run.
    """
    from osm_chordify.osm.intersect import intersect_road_network_with_zones

    grid_intersection_path = raw_dir / "beam_osm_aermod_inmap_county_intersection.parquet"
    if grid_intersection_path.exists():
        logger.info("Step 3: reusing existing grid intersection %s", grid_intersection_path)
        return str(grid_intersection_path), None
    epsg = int(pipeline.output_epsg)

    # Step 3.1: mapped line network × AERMOD grid → labeled aermod_*
    logger.info("Step 3.1: intersecting line network with AERMOD grid %s", pipeline.aermod_grid_path)
    A = intersect_road_network_with_zones(
        mapped_network,
        epsg,
        pipeline.aermod_grid_path,
        output_epsg=epsg,
        prefilter_zones_to_network_bbox=True,
        zone_label="aermod",
    )
    logger.info("Step 3.1 complete: %d rows", len(A))

    # Step 3.2: A × inMAP grid → labeled inmap_* (aermod_* preserved)
    logger.info("Step 3.2: intersecting with inMAP grid %s", pipeline.inmap_grid_path)
    B = intersect_road_network_with_zones(
        A,
        epsg,
        pipeline.inmap_grid_path,
        output_epsg=epsg,
        prefilter_zones_to_network_bbox=True,
        zone_label="inmap",
    )
    B = B.reset_index(drop=True).copy()
    B[_SOURCE_ROW_ID] = range(len(B))
    logger.info("Step 3.2 complete: %d rows", len(B))

    # Step 3.3: B × county boundaries → labeled county_*
    if not pipeline.county_boundaries_path:
        raise ValueError("Step 3 requires pipeline.county_boundaries_path from preprocess.")
    county_setup_started = time.perf_counter()
    county_gdf = _read_vector(pipeline.county_boundaries_path)
    if county_gdf.crs is not None:
        county_gdf = county_gdf.to_crs(epsg=epsg)
    logger.info(
        "Step 3.3: intersecting with county boundaries (%d polygons prepared in %.2fs)",
        len(county_gdf),
        time.perf_counter() - county_setup_started,
    )
    county_match_started = time.perf_counter()
    C_matched = intersect_road_network_with_zones(
        B, epsg, county_gdf, output_epsg=epsg, zone_label="county",
    )
    logger.info(
        "Step 3.3 complete: %d matched rows in %.2fs",
        len(C_matched),
        time.perf_counter() - county_match_started,
    )

    # Step 3.4: recover unmatched rows by source id instead of a second spatial join
    logger.info("Step 3.4: identifying unmatched rows from county matches")
    unmatched_started = time.perf_counter()
    matched_source_col = _resolve_source_row_col(C_matched)
    C = _union_county_matches_with_unmatched(B, C_matched)
    logger.info(
        "Step 3.4 complete: %d unmatched rows in %.2fs",
        len(B) - len(C_matched[matched_source_col].drop_duplicates()),
        time.perf_counter() - unmatched_started,
    )

    # Step 3.5: persist union of matched + unmatched rows
    persist_started = time.perf_counter()
    logger.info("Step 3.5: writing county-labeled intersection outputs")
    C.to_parquet(grid_intersection_path, index=False)
    C.to_file(grid_intersection_path.with_suffix(".gpkg"), driver="GPKG")
    logger.info(
        "Step 3.5 complete: wrote outputs in %.2fs",
        time.perf_counter() - persist_started,
    )
    logger.info("Step 3 complete: %d total rows → %s", len(C), grid_intersection_path)

    return str(grid_intersection_path), C
