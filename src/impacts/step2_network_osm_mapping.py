"""Step 2 — Network to OSM mapping.

Match BEAM network links to OSM edges and buffer each link into a
rectangular polygon corridor (half-width = lanes × lane_width / 2).
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from .defaults import DEFAULT_LANE_WIDTH_M
from .manifest_models import PipelineConfig

logger = logging.getLogger(__name__)


def _buffer_network_by_lanes(
    network_gdf: gpd.GeoDataFrame,
    output_path: Path,
    lane_width_m: float = DEFAULT_LANE_WIDTH_M,
    lanes_col: str = "numberOfLanes",
) -> gpd.GeoDataFrame:
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


def run(pipeline: PipelineConfig, raw_dir: Path) -> gpd.GeoDataFrame:
    """Map BEAM network to OSM and buffer links into rectangular polygon corridors.

    Returns the buffered network GeoDataFrame.
    """
    from impacts.network2grid.network_grid_clipping import map_beam_network_to_osm

    osm_source = pipeline.osm_links_path or pipeline.osm_pbf_path
    if not osm_source:
        raise ValueError("Step 2 requires osm_links_path or osm_pbf_path.")

    # Step 2.1: match BEAM network to OSM
    beam_osm_path = raw_dir / "beam_osm_mapped.parquet"
    if beam_osm_path.exists():
        logger.info("Step 2.1: reusing existing BEAM/OSM mapping %s", beam_osm_path)
        beam_osm_mapped = gpd.read_parquet(beam_osm_path)
    else:
        logger.info("Step 2.1: mapping BEAM network to OSM using %s", osm_source)
        beam_osm_mapped = map_beam_network_to_osm(
            osm_path=osm_source,
            beam_network_path=pipeline.beam_network_path,
            output_path=str(beam_osm_path),
            network_osm_id_col=pipeline.beam_osm_id_col,
            output_epsg=int(pipeline.output_epsg),
        )
        beam_osm_mapped.to_file(beam_osm_path.with_suffix(".gpkg"), driver="GPKG")
        logger.info("Step 2.1 complete: wrote %s", beam_osm_path)

    # Step 2.2: buffer links into rectangular polygon corridors
    beam_osm_buffered_path = raw_dir / "beam_osm_buffered.parquet"
    if beam_osm_buffered_path.exists():
        logger.info("Step 2.2: reusing existing buffered network %s", beam_osm_buffered_path)
        buffered = gpd.read_parquet(beam_osm_buffered_path)
    else:
        logger.info("Step 2.2: buffering network links (%gm/lane)", DEFAULT_LANE_WIDTH_M)
        buffered = _buffer_network_by_lanes(beam_osm_mapped, beam_osm_buffered_path)
        logger.info("Step 2.2 complete: wrote %s", beam_osm_buffered_path)

    return buffered
