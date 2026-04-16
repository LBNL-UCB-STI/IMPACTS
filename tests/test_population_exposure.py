from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from impacts.workflow.step4_prepare_exposure import _aggregate_population_by_aermod_cell
from impacts.workflow.step4_prepare_exposure import _build_population_exposure_distribution
from impacts.workflow.step4_prepare_exposure import _prepare_population_table


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


def test_prepare_population_table_accepts_identity_fields_from_named_indexes() -> None:
    persons_df = pd.DataFrame(
        {
            "household_id": [10, 20],
            "home_x": [-122.0, -122.1],
            "home_y": [37.0, 37.1],
        },
        index=pd.Index([1001, 1002], name="person_id"),
    )
    households_df = pd.DataFrame(
        {
            "income": [50000, 75000],
        },
        index=pd.Index([10, 20], name="household_id"),
    )

    result = _prepare_population_table(
        persons_df=persons_df,
        households_df=households_df,
        target_epsg=4326,
    ).drop(columns="geometry")

    assert "person_id" in result.columns
    assert "household_id" in result.columns
    assert result["person_id"].tolist() == [1001, 1002]
    assert result["household_id"].tolist() == [10, 20]
    assert result["income"].tolist() == [50000, 75000]
