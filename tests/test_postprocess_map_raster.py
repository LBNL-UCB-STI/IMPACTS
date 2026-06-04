from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from impacts.pipeline.postprocess._common import _grid_raster_layout
from impacts.pipeline.postprocess._common import _rasterize_grid_values
from impacts.pipeline.postprocess.step5_plot_exposure import _aggregate_exposure_to_inmap
from impacts.pipeline.postprocess.step5_plot_exposure import _merge_population_concentration


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
