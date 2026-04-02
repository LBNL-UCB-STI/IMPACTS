from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from impacts.workflow.step4_prepare_exposure import _aggregate_population_by_aermod_cell
from impacts.workflow.step4_prepare_exposure import _build_population_exposure_distribution


def test_aggregate_population_by_aermod_cell_counts_people() -> None:
    population_table = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "aermod_cell_id": [10, 10, 20, 20, 20],
        }
    )

    result = _aggregate_population_by_aermod_cell(population_table)

    assert result.to_dict(orient="records") == [
        {"aermod_cell_id": 10, "person_count": 2},
        {"aermod_cell_id": 20, "person_count": 3},
    ]


def test_build_population_exposure_distribution_keeps_geometry() -> None:
    population_table = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "aermod_cell_id": [10, 10, 20],
        }
    )
    exposure_grid = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [10, 20, 30],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )

    result = _build_population_exposure_distribution(
        population_table=population_table,
        exposure_grid=exposure_grid,
    )

    assert result.drop(columns="geometry").to_dict(orient="records") == [
        {"aermod_cell_id": 10, "person_count": 2},
        {"aermod_cell_id": 20, "person_count": 1},
    ]
