#!/usr/bin/env python
"""Emissions mapping workflow utilities.

This module isolates the network/emissions mapping stage used by the
impacts workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import osm_chordify

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
    grid_cells_path: str = "data/input/grid_polygon/grid_polygon.shp"
    beam_network_path: Optional[str] = None
    precomputed_beam_osm_path: Optional[str] = None
    output_dir: str = "src/impacts/tmp"
    beam_osm_id_col: str = "attributeOrigId"
    beam_length_col: str = "linkLength"
    beam_osm_epsg: int = 4326
    grid_epsg: int = 4326
    output_epsg: int = 26910


DEFAULT_MAPPING_CONFIG = EmissionsMappingConfig()


def _parse_scalar(raw: str):
    value = raw.strip()
    if value in ("", "null", "Null", "NULL", "none", "None", "~"):
        return None
    if value in ("true", "True", "TRUE"):
        return True
    if value in ("false", "False", "FALSE"):
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str):
    """Minimal YAML parser for nested dicts with scalar values."""
    data = {}
    current_section = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            continue
        key, raw_val = line.split(":", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if indent == 0:
            if raw_val == "":
                data[key] = {}
                current_section = key
            else:
                data[key] = _parse_scalar(raw_val)
                current_section = None
        else:
            if current_section is None:
                continue
            data[current_section][key] = _parse_scalar(raw_val)
    return data


def load_workflow_config(config_path: str = "src/impacts/config/workflow.yaml"):
    """Load workflow config from YAML (with fallback simple parser)."""
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        return {"main": {}, "osm_grid": {}}

    text = cfg_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
    except Exception:
        loaded = _simple_yaml_load(text)

    if not isinstance(loaded, dict):
        try:
            loaded = json.loads(text)
        except Exception:
            loaded = {}
    return loaded


def load_mapping_config(config_path: str = "src/impacts/config/workflow.yaml"):
    """Build `EmissionsMappingConfig` from workflow YAML."""
    workflow = load_workflow_config(config_path)
    main = workflow.get("main", {}) or {}
    section_name = main.get("mapping_section", "osm_grid")
    section = workflow.get(section_name, {}) or {}

    allowed = {f.name for f in fields(EmissionsMappingConfig)}
    config_kwargs = {k: v for k, v in section.items() if k in allowed}
    cfg = EmissionsMappingConfig(**config_kwargs)
    return cfg


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


def map_beam_network_to_osm(
    osm_path: str,
    beam_network_path: str,
    output_path: Optional[str] = None,
    network_osm_id_col: str = "attributeOrigId",
):
    """Map BEAM links to OSM geometries using shared OSM IDs."""
    _validate_local_path(osm_path, "OSM network path")
    _validate_local_path(beam_network_path, "BEAM network path")

    if output_path is None:
        output_path = "src/impacts/tmp/beam_osm_mapped.geojson"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    mapped = osm_chordify.map_osm_with_beam_network(
        osm_path=osm_path,
        network_path=beam_network_path,
        network_osm_id_col=network_osm_id_col,
        output_path=output_path,
    )
    return mapped


def intersect_beam_osm_with_grid(
    beam_osm_path,
    grid_cells_path: str,
    output_path: Optional[str] = None,
    beam_osm_epsg: int = 4326,
    grid_epsg: int = 4326,
    output_epsg: int = 26910,
    beam_length_col: str = "linkLength",
):
    """Intersect mapped BEAM+OSM links with grid cells.

    Produces per-cell segments and computes edge length within each GRID
    cell via the intersection proportion.
    """
    if isinstance(beam_osm_path, str):
        _validate_local_path(beam_osm_path, "BEAM+OSM mapped path")
    _validate_local_path(grid_cells_path, "grid cells path")

    if output_path is None:
        output_path = "src/impacts/tmp/beam_osm_grid_intersection.geojson"

    if isinstance(beam_osm_path, gpd.GeoDataFrame):
        beam_osm_gdf = beam_osm_path
    else:
        beam_osm_gdf = gpd.read_file(beam_osm_path)

    proportional_cols = None
    if beam_length_col in beam_osm_gdf.columns:
        proportional_cols = [beam_length_col]

    result = osm_chordify.intersect_road_network_with_zones(
        road_network=beam_osm_gdf,
        road_network_epsg=beam_osm_epsg,
        zones=grid_cells_path,
        zones_epsg=grid_epsg,
        proportional_cols=proportional_cols,
        output_path=None,
        output_epsg=output_epsg,
    )
    result["edge_length_in_cell_m"] = result["proportional_length_m"]

    if proportional_cols:
        col = proportional_cols[0]
        p_col = f"proportional_{col}"
        if p_col in result.columns:
            result["beam_length_in_cell"] = result[p_col]

    _save_geodataframe(result, output_path)
    return result


def map_skims_emissions_to_intersection(*args, **kwargs):
    """Map skims emissions to BEAM+OSM+GRID intersection."""
    return _map_skims_emissions_to_intersection_impl(*args, **kwargs)
