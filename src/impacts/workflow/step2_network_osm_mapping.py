"""Step 2 — Network to OSM mapping.

Match BEAM network links to OSM and keep line geometries for downstream
line-based grid intersection.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import osm_chordify
from ..manifest.schema import PipelineConfig

logger = logging.getLogger(__name__)


def _is_remote_path(path: str) -> bool:
    return path.startswith(("gs://", "s3://", "http://", "https://"))


def _validate_local_path(path: str, label: str) -> None:
    if _is_remote_path(path):
        return
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _save_geodataframe(gdf: gpd.GeoDataFrame, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".parquet":
        gdf.to_parquet(out, index=False)
        return
    if suffix == ".geojson":
        gdf.to_file(out, driver="GeoJSON")
        return
    if suffix == ".gpkg":
        gdf.to_file(out, driver="GPKG")
        return
    gdf.to_file(out.with_suffix(".geojson"), driver="GeoJSON")


def _load_geodataframe(path_or_gdf):
    if isinstance(path_or_gdf, gpd.GeoDataFrame):
        return path_or_gdf.copy()
    path = Path(path_or_gdf)
    if path.suffix.lower() == ".parquet":
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def map_beam_network_to_osm(
    osm_path: str,
    beam_network_path: str,
    output_path: str,
    network_osm_id_col: str = "attributeOrigId",
    output_epsg: Optional[int] = None,
):
    _validate_local_path(osm_path, "OSM network path")
    _validate_local_path(beam_network_path, "BEAM network path")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    mapped = osm_chordify.map_osm_with_beam_network(
        osm_path=osm_path,
        network_path=beam_network_path,
        network_osm_id_col=network_osm_id_col,
        output_path=output_path,
    )
    if output_epsg is not None:
        mapped_gdf = _load_geodataframe(output_path).to_crs(epsg=output_epsg)
        _save_geodataframe(mapped_gdf, output_path)
        return mapped_gdf
    return mapped


def run(pipeline: PipelineConfig, raw_dir: Path) -> gpd.GeoDataFrame:
    """Map BEAM network to OSM and keep line geometries.

    Returns the mapped line network GeoDataFrame.
    """
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
