"""Near-source suppression for selected traffic ISRM responses."""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_NEAR_SOURCE_RADIUS_METERS = 3_000.0
DEFAULT_INTERSECTION_FRACTION_THRESHOLD = 0.5


def build_near_source_mask(
    *,
    grid_gdf: gpd.GeoDataFrame,
    source_cells: np.ndarray,
    receptor_cells: np.ndarray,
    grid_id_col: str = "inmap_cell_id",
    radius_meters: float = DEFAULT_NEAR_SOURCE_RADIUS_METERS,
    intersection_fraction_threshold: float = DEFAULT_INTERSECTION_FRACTION_THRESHOLD,
) -> np.ndarray:
    """Return a source-by-receptor mask for the configured near-source rule.

    A pair is selected when a circle centered on the source-cell centroid
    covers more than the configured fraction of the receptor polygon.
    """
    if radius_meters <= 0.0 or not np.isfinite(radius_meters):
        raise ValueError(f"radius_meters must be finite and positive, got {radius_meters!r}")
    if not 0.0 <= intersection_fraction_threshold <= 1.0:
        raise ValueError(
            "intersection_fraction_threshold must be in [0, 1], got "
            f"{intersection_fraction_threshold!r}"
        )
    if grid_gdf.crs is None or not grid_gdf.crs.is_projected:
        raise ValueError("ISRM grid must use a projected CRS with metre units.")
    axis_info = grid_gdf.crs.axis_info
    if axis_info and not np.isclose(float(axis_info[0].unit_conversion_factor), 1.0):
        raise ValueError("ISRM grid projected CRS must use metre units.")
    if grid_id_col not in grid_gdf.columns:
        raise ValueError(f"ISRM grid is missing id column {grid_id_col!r}.")

    source_ids = np.asarray(source_cells, dtype=np.int64)
    receptor_ids = np.asarray(receptor_cells, dtype=np.int64)
    if source_ids.ndim != 1 or receptor_ids.ndim != 1:
        raise ValueError("source_cells and receptor_cells must be one-dimensional arrays.")

    grid = grid_gdf[[grid_id_col, "geometry"]].copy()
    grid[grid_id_col] = pd.to_numeric(grid[grid_id_col], errors="raise").astype(np.int64)
    if grid[grid_id_col].duplicated().any():
        raise ValueError(f"ISRM grid contains duplicate {grid_id_col} values.")
    if grid.geometry.isna().any() or grid.geometry.is_empty.any() or (~grid.geometry.is_valid).any():
        raise ValueError("ISRM grid contains missing, empty, or invalid geometries.")

    indexed = grid.set_index(grid_id_col)
    missing_sources = source_ids[~np.isin(source_ids, indexed.index.to_numpy())]
    missing_receptors = receptor_ids[~np.isin(receptor_ids, indexed.index.to_numpy())]
    if missing_sources.size or missing_receptors.size:
        raise ValueError(
            "ISRM grid is missing requested source/receptor ids: "
            f"sources={missing_sources[:10].tolist()} receptors={missing_receptors[:10].tolist()}"
        )

    source_geometry = gpd.GeoSeries(
        indexed.loc[source_ids, "geometry"].to_numpy(), index=np.arange(source_ids.size), crs=grid.crs
    )
    receptor_geometry = gpd.GeoSeries(
        indexed.loc[receptor_ids, "geometry"].to_numpy(), index=np.arange(receptor_ids.size), crs=grid.crs
    )
    receptor_areas = receptor_geometry.area.to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(receptor_areas)) or np.any(receptor_areas <= 0.0):
        raise ValueError("ISRM receptor grid contains non-positive or invalid polygon areas.")

    mask = np.zeros((source_ids.size, receptor_ids.size), dtype=bool)
    receptor_sindex = receptor_geometry.sindex
    for source_position, source_polygon in enumerate(source_geometry):
        source_circle = source_polygon.centroid.buffer(float(radius_meters))
        candidate_positions = np.asarray(
            receptor_sindex.query(source_circle, predicate="intersects"), dtype=np.int64
        )
        if candidate_positions.size == 0:
            continue
        intersection_areas = (
            receptor_geometry.iloc[candidate_positions].intersection(source_circle).area.to_numpy(dtype=np.float64)
        )
        fractions = intersection_areas / receptor_areas[candidate_positions]
        selected = candidate_positions[fractions > float(intersection_fraction_threshold)]
        mask[source_position, selected] = True

    logger.info(
        "Built traffic ISRM near-source mask: radius=%.1fm threshold=>%.3f sources=%d receptors=%d suppressed_pairs=%d",
        float(radius_meters), float(intersection_fraction_threshold), int(source_ids.size),
        int(receptor_ids.size), int(mask.sum()),
    )
    return mask


def zero_near_source_isrm(isrm: np.ndarray, *, near_source_mask: np.ndarray) -> np.ndarray:
    """Return an ISRM copy with selected source/receptor responses set to zero."""
    adjusted = np.asarray(isrm).copy()
    mask = np.asarray(near_source_mask, dtype=bool)
    if adjusted.ndim != 2 or adjusted.shape != mask.shape:
        raise ValueError(
            "ISRM transfer matrix and near-source mask must be two-dimensional with the same shape: "
            f"isrm={adjusted.shape} mask={mask.shape}"
        )
    adjusted[mask] = 0.0
    return adjusted
