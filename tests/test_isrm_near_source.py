from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from impacts.pipeline.workflow.isrm_near_source import build_near_source_mask
from impacts.pipeline.workflow.isrm_near_source import zero_near_source_isrm
from impacts.pipeline.workflow.step2_compute_inmap_concentrations import _compute_custom_receptor_response
from impacts.pipeline.workflow.step2_compute_inmap_concentrations import compute_isrm_concentrations


class _OrthogonalArray:
    def __init__(self, values: np.ndarray):
        self.values = np.asarray(values)
        self.shape = self.values.shape

    @property
    def oindex(self):
        return self

    def __getitem__(self, key):
        layers, sources, receptors = key
        return self.values[np.ix_(np.asarray(layers), np.asarray(sources), np.asarray(receptors))]


def _two_cell_grid() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "inmap_cell_id": [0, 1],
            "geometry": [box(-50.0, -50.0, 50.0, 50.0), box(9_950.0, -50.0, 10_050.0, 50.0)],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )


def test_build_near_source_mask_uses_receptor_area_fraction() -> None:
    grid = gpd.GeoDataFrame(
        {
            "inmap_cell_id": [1, 2, 3],
            "geometry": [
                box(-50.0, -50.0, 50.0, 50.0),
                box(500.0, -50.0, 600.0, 50.0),
                box(-10_000.0, -10_000.0, 10_000.0, 10_000.0),
            ],
        }, geometry="geometry", crs="EPSG:26910",
    )
    mask = build_near_source_mask(
        grid_gdf=grid, source_cells=np.array([1]), receptor_cells=np.array([1, 2, 3]),
        radius_meters=1_000.0, intersection_fraction_threshold=0.5,
    )
    assert mask.tolist() == [[True, True, False]]


def test_zero_near_source_isrm_returns_adjusted_copy() -> None:
    original = np.array([[1.0, 2.0], [3.0, 4.0]])
    adjusted = zero_near_source_isrm(original, near_source_mask=np.eye(2, dtype=bool))
    assert adjusted.tolist() == [[0.0, 2.0], [3.0, 0.0]]
    assert original.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_default_radius_is_3km_not_4500m() -> None:
    grid = gpd.GeoDataFrame({'inmap_cell_id': [0, 1, 2], 'geometry': [
        box(-50, -50, 50, 50), box(2800, -50, 2900, 50),
        box(3500, -50, 3600, 50)]}, crs='EPSG:26910')
    mask = build_near_source_mask(grid_gdf=grid, source_cells=np.array([0]), receptor_cells=np.array([1, 2]))
    assert mask.tolist() == [[True, False]]


def test_exact_half_receptor_overlap_is_retained() -> None:
    from shapely.geometry import Point
    circle = Point(0, 0).buffer(3000)
    # Disjoint equal-area circle plus remote translated circle gives exactly 50%.
    from shapely.affinity import translate
    receptor = circle.union(translate(circle, xoff=20000))
    grid = gpd.GeoDataFrame({'inmap_cell_id': [0, 1], 'geometry': [box(-50,-50,50,50),receptor]}, crs='EPSG:26910')
    fraction = receptor.intersection(circle).area/receptor.area
    mask = build_near_source_mask(grid_gdf=grid, source_cells=np.array([0]), receptor_cells=np.array([1]), intersection_fraction_threshold=fraction)
    assert mask.tolist() == [[False]]


def test_step2_suppresses_only_traffic_primary_bc_and_no2(tmp_path: Path) -> None:
    matrix = np.ones((1, 2, 2), dtype=float)
    sr = {name: _OrthogonalArray(matrix) for name in ("SOA", "pNO3", "pNH4", "pSO4", "PrimaryPM25", "NO2")}
    emissions = pd.DataFrame(
        {
            "inmap_cell_id": [0, 1],
            "tons_per_year_NH3_inmap_allocated": [1.0, 1.0],
            "tons_per_year_NOx_inmap_allocated": [1.0, 1.0],
            "tons_per_year_PM25_inmap_allocated": [1.0, 1.0],
            "tons_per_year_SOx_inmap_allocated": [1.0, 1.0],
            "tons_per_year_ROG_inmap_allocated": [1.0, 1.0],
            "tons_per_year_BC_inmap_allocated": [1.0, 1.0],
        }
    )
    result = compute_isrm_concentrations(
        grid_emissions_df=emissions, sr=sr, factor=1.0,
        requested_pollutants=["NH3", "NOx", "PM25", "SOx", "ROG", "BC"],
        receptor_cells=np.array([0, 1]), working_dir=tmp_path,
        pollutants=["NH3", "NOx", "PM25", "SOx", "ROG", "BC"],
        isrm_grid_gdf=_two_cell_grid(),
    )
    for component in ("SOA", "pNO3", "pNH4", "pSO4"):
        assert result[component].tolist() == [2.0, 2.0]
    assert result["PrimaryPM25"].tolist() == [1.0, 1.0]
    assert result["BC"].tolist() == [1.0, 1.0]
    assert result["NO2"].tolist() == [1.0, 1.0]
    assert result["TotalPM25"].tolist() == [9.0, 9.0]
    assert result["InMAPOnly_PrimaryPM25"].tolist() == [2.0, 2.0]
    assert result["InMAPOnly_BC"].tolist() == [2.0, 2.0]
    assert result["InMAPOnly_NO2"].tolist() == [2.0, 2.0]
    assert result["InMAPOnly_TotalPM25"].tolist() == [10.0, 10.0]


def test_sparse_no2_response_excludes_near_source_pairs() -> None:
    transfer = {
        "source_ids": np.array([0, 0, 1, 1]), "receptor_ids": np.array([0, 1, 0, 1]),
        "values": np.ones(4), "source_dim": 2, "receptor_dim": 2,
    }
    source_indexed = pd.DataFrame(
        {"tons_per_year_NOx_inmap_allocated": [1.0, 1.0]},
        index=pd.Index([0, 1], name="inmap_cell_id"),
    )
    result = _compute_custom_receptor_response(
        transfer_matrix=transfer, emissions_key="tons_per_year_NOx_inmap_allocated",
        source_cells=np.array([0, 1]), source_indexed=source_indexed,
        receptor_cells=np.array([0, 1]), result_key="NO2",
        near_source_mask=np.eye(2, dtype=bool),
    )
    assert result.tolist() == [1.0, 1.0]
