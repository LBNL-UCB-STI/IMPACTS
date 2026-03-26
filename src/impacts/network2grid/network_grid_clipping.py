#!/usr/bin/env python
"""Emissions mapping workflow utilities.

This module isolates the network/emissions mapping stage used by the
impacts workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import osm_chordify
import pandas as pd

try:
    from .emissions_grid_mapping import (
        map_skims_emissions_to_intersection as _map_skims_emissions_to_intersection_impl,
    )
except ImportError:
    from impacts.emissions.emissions_grid_mapping import (
        map_skims_emissions_to_intersection as _map_skims_emissions_to_intersection_impl,
    )
@dataclass
class EmissionsMappingConfig:
    """Configuration for BEAM-OSM-GRID mapping workflow."""

    osm_links_path: Optional[str] = None
    inmap_grid_path: str = "data/input/grid_polygon/grid_polygon.shp"
    beam_network_path: Optional[str] = None
    precomputed_beam_osm_path: Optional[str] = None
    output_dir: str = "src/impacts/tmp"
    beam_osm_mapped_output_path: Optional[str] = None
    beam_osm_county_intersection_output_path: Optional[str] = None
    beam_osm_inmap_grid_intersection_output_path: Optional[str] = None
    beam_osm_aermod_grid_intersection_output_path: Optional[str] = None
    beam_osm_id_col: str = "attributeOrigId"
    beam_length_col: str = "linkLength"
    beam_osm_epsg: int = 4326
    inmap_grid_epsg: int = 4326
    aermod_grid_epsg: int = 4326
    output_epsg: int = 26910
    county_state_fips: Optional[str] = None
    county_fips_codes: Optional[list[str]] = None
    county_area_name: str = "county"
    aermod_grid_path: Optional[str] = None


DEFAULT_MAPPING_CONFIG = EmissionsMappingConfig()


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
    if suffix == ".csv":
        df = gdf.copy()
        df["geometry_wkt"] = df.geometry.apply(lambda geom: geom.wkt if geom else None)
        df.drop(columns=["geometry"]).to_csv(out, index=False)
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
    """Map BEAM links to OSM geometries using shared OSM IDs."""
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


def intersect_beam_osm_with_grid(
    beam_osm_path,
    grid_cells_path: str,
    output_path: str,
    beam_osm_epsg: int = 4326,
    grid_epsg: int = 4326,
    output_epsg: int = 26910,
    beam_length_col: str = "linkLength",
):
    """Intersect mapped BEAM+OSM links with grid cells.

    Produces per-cell segments and computes edge length within each GRID
    cell via the intersection proportion. The grid is expected to be
    pre-filtered to the study area before calling this function.
    """
    if isinstance(beam_osm_path, str):
        _validate_local_path(beam_osm_path, "BEAM+OSM mapped path")
    _validate_local_path(grid_cells_path, "grid cells path")

    road_network = beam_osm_path if isinstance(beam_osm_path, gpd.GeoDataFrame) else str(beam_osm_path)

    result = osm_chordify.intersect_road_network_with_zones(
        road_network=road_network,
        road_network_epsg=beam_osm_epsg,
        zones=grid_cells_path,
        output_path=None,
        output_epsg=output_epsg,
    )
    if "zone_link_length_m" in result.columns and "edge_length_in_cell_m" not in result.columns:
        result["edge_length_in_cell_m"] = result["zone_link_length_m"]
    if "zone_link_length_m" in result.columns and "proportional_length_m" not in result.columns:
        result["proportional_length_m"] = result["zone_link_length_m"]

    edge_beam_length_col = f"edge_{beam_length_col}"
    beam_length_source_col = None
    if edge_beam_length_col in result.columns:
        beam_length_source_col = edge_beam_length_col
    elif "edge_link_length_m" in result.columns:
        beam_length_source_col = "edge_link_length_m"

    if beam_length_source_col and "zone_edge_proportion" in result.columns:
        result["beam_length_in_cell"] = (
            pd.to_numeric(result[beam_length_source_col], errors="coerce").fillna(0.0)
            * pd.to_numeric(result["zone_edge_proportion"], errors="coerce").fillna(0.0)
        )

    _save_geodataframe(result, output_path)
    return result


def intersect_beam_osm_with_counties(
    beam_osm_path,
    *,
    state_fips: str,
    county_fips_codes: list[str],
    output_path: str,
    beam_osm_epsg: int = 4326,
    output_epsg: int = 26910,
    area_name: str = "county",
    boundary_year: int = 2023,
):
    """Intersect mapped BEAM+OSM links with county zones via FIPS codes."""
    if isinstance(beam_osm_path, str):
        _validate_local_path(beam_osm_path, "BEAM+OSM mapped path")

    road_network = beam_osm_path if isinstance(beam_osm_path, gpd.GeoDataFrame) else str(beam_osm_path)
    county_work_dir = str(Path(output_path).resolve().parent)
    result = osm_chordify.intersect_road_network_with_county_zones(
        road_network=road_network,
        road_network_epsg=beam_osm_epsg,
        state_fips_code=str(state_fips),
        county_fips_codes=[str(code) for code in county_fips_codes],
        year=int(boundary_year),
        work_dir=county_work_dir,
        output_path=None,
        output_epsg=output_epsg,
        area_name=area_name,
    )
    if "zone_link_length_m" in result.columns and "edge_length_in_cell_m" not in result.columns:
        result["edge_length_in_cell_m"] = result["zone_link_length_m"]
    if "zone_link_length_m" in result.columns and "proportional_length_m" not in result.columns:
        result["proportional_length_m"] = result["zone_link_length_m"]

    _save_geodataframe(result, output_path)
    return result


def map_skims_emissions_to_intersection(*args, **kwargs):
    """Map skims emissions to BEAM+OSM+GRID intersection."""
    return _map_skims_emissions_to_intersection_impl(*args, **kwargs)
