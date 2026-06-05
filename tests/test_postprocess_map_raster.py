from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from shapely.geometry import box

from impacts.pipeline.postprocess._common import _grid_raster_layout
from impacts.pipeline.postprocess._common import _rasterize_grid_values
from impacts.pipeline.postprocess.step5_plot_exposure import _aggregate_exposure_to_inmap
from impacts.pipeline.postprocess.step5_plot_exposure import _merge_population_concentration
from impacts.pipeline.postprocess.step6_plot_delta_concentrations import _build_concentration_delta
from impacts.pipeline.postprocess.step7_plot_delta_exposure import _aggregate_delta_exposure_to_inmap


def test_grid_raster_layout_preserves_native_cell_values() -> None:
    gdf = gpd.GeoDataFrame(
        {"value": [1.0, 2.0, 3.0]},
        geometry=[
            box(0, 0, 100, 100),
            box(100, 0, 200, 100),
            box(0, 100, 100, 200),
        ],
        crs="EPSG:26910",
    )

    layout = _grid_raster_layout(gdf)
    raster = _rasterize_grid_values(gdf["value"], layout, threshold=1.5)

    assert layout.shape == (2, 2)
    assert layout.extent == (0.0, 200.0, 0.0, 200.0)
    assert layout.padded_extent == (-2000.0, 2200.0, -2000.0, 2200.0)
    assert np.ma.is_masked(raster[0, 0])
    assert raster[0, 1] == 2.0
    assert raster[1, 0] == 3.0
    assert np.ma.is_masked(raster[1, 1])


def test_concentration_delta_aligns_by_aermod_cell_id_and_subtracts_baseline(tmp_path: Path) -> None:
    current = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [1, 2],
            "inmap_cell_id": [100, 100],
            "TotalPM25": [3.0, 8.0],
            "PrimaryPM25": [1.0, 4.0],
            "SecondaryPM25": [2.0, 4.0],
            "BC": [0.3, 0.8],
            "NO2": [10.0, 20.0],
        },
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)],
        crs="EPSG:26910",
    )
    baseline = pd.DataFrame(
        {
            "aermod_cell_id": [2, 1],
            "TotalPM25": [5.0, 4.0],
            "PrimaryPM25": [3.0, 1.5],
            "SecondaryPM25": [2.0, 2.5],
            "BC": [0.2, 0.1],
            "NO2": [12.0, 11.0],
        }
    )
    baseline_path = tmp_path / "beam_concentration_distribution.parquet"
    baseline.to_parquet(baseline_path, index=False)

    delta = _build_concentration_delta(current, baseline_path).set_index("aermod_cell_id")

    assert delta.loc[1, "TotalPM25_delta"] == -1.0
    assert delta.loc[2, "TotalPM25_delta"] == 3.0
    assert delta.loc[1, "PrimaryPM25_delta"] == -0.5
    assert delta.loc[2, "PrimaryPM25_delta"] == 1.0
    assert delta.loc[1, "SecondaryPM25_delta"] == -0.5
    assert delta.loc[2, "SecondaryPM25_delta"] == 2.0
    assert delta.loc[1, "BC_delta"] == pytest.approx(0.2)
    assert delta.loc[2, "NO2_delta"] == 8.0


def test_delta_exposure_aggregates_current_population_weighted_delta_to_inmap() -> None:
    pop_gdf = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [1, 2, 3],
            "person_count": [10, 30, 0],
        },
        geometry=[
            box(0, 0, 100, 100),
            box(100, 0, 200, 100),
            box(200, 0, 300, 100),
        ],
        crs="EPSG:26910",
    )
    delta_df = pd.DataFrame(
        {
            "aermod_cell_id": [1, 2, 3],
            "inmap_cell_id": [100, 100, 200],
            "TotalPM25_delta": [-1.0, 3.0, 100.0],
        }
    )
    inmap_gdf = gpd.GeoDataFrame(
        {"inmap_cell_id": [100, 200]},
        geometry=[box(0, 0, 200, 100), box(200, 0, 300, 100)],
        crs="EPSG:26910",
    )

    pwc_delta = _aggregate_delta_exposure_to_inmap(pop_gdf, delta_df, inmap_gdf).sort_values("inmap_cell_id")

    assert pwc_delta["inmap_cell_id"].tolist() == [100]
    assert pwc_delta["population"].tolist() == [40]
    assert pwc_delta["exposure_burden_delta"].tolist() == [80.0]
    assert pwc_delta["pwc_pm25_delta"].tolist() == [2.0]


def test_exposure_burden_is_computed_before_inmap_aggregation() -> None:
    pop_gdf = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [1, 2],
            "person_count": [10, 30],
        },
        geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)],
        crs="EPSG:26910",
    )
    conc_df = pd.DataFrame(
        {
            "aermod_cell_id": [1, 2],
            "inmap_cell_id": [100, 100],
            "TotalPM25": [2.0, 4.0],
        }
    )

    merged = _merge_population_concentration(pop_gdf, conc_df)

    assert merged["pm25_exposure_burden"].tolist() == [20.0, 120.0]


def test_inmap_exposure_aggregates_population_weighted_cells_not_counties() -> None:
    pop_gdf = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [1, 2, 3],
            "person_count": [10, 30, 0],
        },
        geometry=[
            box(0, 0, 100, 100),
            box(100, 0, 200, 100),
            box(200, 0, 300, 100),
        ],
        crs="EPSG:26910",
    )
    conc_df = pd.DataFrame(
        {
            "aermod_cell_id": [1, 2, 3],
            "inmap_cell_id": [100, 100, 200],
            "TotalPM25": [2.0, 4.0, 100.0],
        }
    )
    inmap_gdf = gpd.GeoDataFrame(
        {"inmap_cell_id": [100, 200]},
        geometry=[box(0, 0, 200, 100), box(200, 0, 300, 100)],
        crs="EPSG:26910",
    )

    pwc = _aggregate_exposure_to_inmap(pop_gdf, conc_df, inmap_gdf).sort_values("inmap_cell_id")

    assert pwc["inmap_cell_id"].tolist() == [100]
    assert pwc["population"].tolist() == [40]
    assert pwc["exposure_burden"].tolist() == [140.0]
    assert pwc["pwc_pm25"].tolist() == [3.5]
