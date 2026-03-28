"""Step 2 — Network to OSM mapping.

Match BEAM network links to OSM and keep line geometries for downstream
line-based grid intersection.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
from .manifest_models import PipelineConfig

logger = logging.getLogger(__name__)


def run(pipeline: PipelineConfig, raw_dir: Path) -> gpd.GeoDataFrame:
    """Map BEAM network to OSM and keep line geometries.

    Returns the mapped line network GeoDataFrame.
    """
    from impacts.utils.utils_network_grid_clipping import map_beam_network_to_osm

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

    logger.info("Step 2.2: line-based workflow enabled; skipping buffered network generation")
    return beam_osm_mapped
