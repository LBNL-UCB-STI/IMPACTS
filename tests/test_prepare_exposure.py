from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from impacts.manifest.schema import PipelineConfig
from impacts.pipeline.workflow.step4_prepare_exposure import _prepare_aermod_exposure_inputs
from impacts.pipeline.workflow.step4_prepare_exposure import _build_full_exposure_grid


def _pipeline(
    tmp_path: Path,
    grid_path: Path,
    *,
    primary_pm25_integration_strategy: str = "impute_inmap_primary_in_aermod_domain",
) -> PipelineConfig:
    return PipelineConfig.from_dict(
        {
            "beam_osm_id_col": "attributeOrigId",
            "beam_length_col": "linkLength",
            "output_epsg": 26910,
            "emissions_enabled": True,
            "inmap_enabled": True,
            "aermod_enabled": True,
            "exposure_enabled": True,
            "inmap_grid_path": str(tmp_path / "inmap_grid.parquet"),
            "inmap_grid_epsg": 26910,
            "mapping_columns": {"link_id": "linkId", "grid_id": "isrm"},
            "isrm_url": str(tmp_path / "isrm.zarr"),
            "isrm_nox_to_no2_ratios_file": str(tmp_path / "matrix.npz"),
            "isrm_nox_to_no2_ratios_apply_tons_per_year_to_ug_per_s": False,
            "grid_size_meters": 100.0,
            "asrv_patterns_file": str(tmp_path / "asrv.parquet"),
            "asrv_patterns_epsg": 4326,
            "aermod_full_grid_path": str(grid_path),
            "aermod_grid_path": str(tmp_path / "aermod_grid.parquet"),
            "aermod_grid_epsg": 26910,
            "aermod_grid_id": "aermod_cell_id",
            "region": "sfbay",
            "start_year": 2018,
            "county_state_fips": "06",
            "county_fips_codes": ["001"],
            "passenger_inventory_file": str(tmp_path / "passenger_inventory.parquet"),
            "freight_inventory_file": str(tmp_path / "freight_inventory.parquet"),
            "enable_passenger_inventory_activity_correction": True,
            "enable_freight_inventory_activity_correction": True,
            "passenger_vehicle_types_file": str(tmp_path / "vehicleTypes--atlas.csv"),
            "freight_vehicle_types_file": str(tmp_path / "vehicleTypes--frism.csv"),
            "prepared_skims_group_cols": ["hour", "linkId"],
            "pollutants": ["NOx", "PM2_5", "BC"],
            "source_pollutants": ["NOx", "PM2_5", "BC"],
            "vehicle_category_metadata_file": str(tmp_path / "vehicle_category_metadata.csv"),
            "annualization_days": {"light_duty": 327.0, "medium_heavy_duty": 312.0},
            "population_sample": 0.1,
            "primary_pm25_integration_strategy": primary_pm25_integration_strategy,
        }
    )


def test_build_full_exposure_grid_uses_inmap_secondary_and_aermod_primary(tmp_path: Path) -> None:
    full_grid = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11, 22],
            "inmap_cell_id": [101, 202],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )
    grid_path = tmp_path / "full_grid.parquet"
    full_grid.to_parquet(grid_path, index=False)

    prepared_inmap = gpd.GeoDataFrame(
        {
            "inmap_cell_id": [101, 202],
            "inmap_PrimaryPM25": [0.0, 0.0],
            "inmap_SecondaryPM25": [1.5, 2.5],
            "geometry": full_grid.geometry,
        },
        geometry="geometry",
        crs=full_grid.crs,
    )
    prepared_aermod = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11],
            "aermod_PrimaryPM25": [0.25],
            "aermod_SecondaryPM25": [0.0],
            "aermod_BC": [0.5],
            "aermod_NO2": [3.0],
            "has_aermod_primarypm25": [True],
            "has_aermod_bc": [True],
            "has_aermod_no2": [True],
            "geometry": [full_grid.geometry.iloc[0]],
        },
        geometry="geometry",
        crs=full_grid.crs,
    )

    result = _build_full_exposure_grid(
        pipeline=_pipeline(tmp_path, grid_path),
        prepared_inmap=prepared_inmap,
        prepared_aermod=prepared_aermod,
    ).drop(columns="geometry")

    by_id = result.set_index("aermod_cell_id")
    assert by_id.loc[11, "TotalPM25"] == 1.75   # aermod primary 0.25 + inmap secondary 1.5
    assert by_id.loc[11, "BC"] == 0.5
    assert by_id.loc[11, "NO2"] == 3.0
    assert by_id.loc[22, "TotalPM25"] == 2.5    # no aermod → inmap primary 0.0 + inmap secondary 2.5
    assert by_id.loc[22, "BC"] == 0.0
    assert by_id.loc[22, "NO2"] == 0.0
    assert bool(by_id.loc[11, "has_aermod_primarypm25"]) is True
    assert bool(by_id.loc[11, "has_aermod_bc"]) is True
    assert bool(by_id.loc[11, "has_aermod_no2"]) is True


def test_build_full_exposure_grid_can_scale_aermod_primary_to_inmap_domain_budget(tmp_path: Path) -> None:
    full_grid = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11, 22, 33],
            "inmap_cell_id": [101, 101, 202],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )
    grid_path = tmp_path / "full_grid.parquet"
    full_grid.to_parquet(grid_path, index=False)

    prepared_inmap = gpd.GeoDataFrame(
        {
            "inmap_cell_id": [101, 202],
            "inmap_PrimaryPM25": [4.0, 7.0],
            "inmap_SecondaryPM25": [1.0, 2.0],
            "geometry": [full_grid.geometry.iloc[0].union(full_grid.geometry.iloc[1]), full_grid.geometry.iloc[2]],
        },
        geometry="geometry",
        crs=full_grid.crs,
    )
    prepared_aermod = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11, 33],
            "aermod_PrimaryPM25": [1.0, 3.0],
            "aermod_SecondaryPM25": [0.0, 0.0],
            "aermod_BC": [0.5, 0.7],
            "aermod_NO2": [3.0, 5.0],
            "has_aermod_primarypm25": [True, True],
            "has_aermod_bc": [True, True],
            "has_aermod_no2": [True, True],
            "geometry": [full_grid.geometry.iloc[0], full_grid.geometry.iloc[2]],
        },
        geometry="geometry",
        crs=full_grid.crs,
    )

    result = _build_full_exposure_grid(
        pipeline=_pipeline(
            tmp_path,
            grid_path,
            primary_pm25_integration_strategy="scale_aermod_to_inmap_domain_primary",
        ),
        prepared_inmap=prepared_inmap,
        prepared_aermod=prepared_aermod,
    )

    by_id = result.drop(columns="geometry").set_index("aermod_cell_id")
    assert by_id.loc[11, "aermod_PrimaryPM25_scale_factor"] == 2.75
    assert pd.isna(by_id.loc[22, "aermod_PrimaryPM25_scale_factor"])
    assert by_id.loc[33, "aermod_PrimaryPM25_scale_factor"] == 2.75
    assert by_id.loc[11, "PrimaryPM25"] == 2.75
    assert by_id.loc[22, "PrimaryPM25"] == 4.0
    assert by_id.loc[33, "PrimaryPM25"] == 8.25
    assert by_id.loc[11, "TotalPM25"] == 3.75
    assert by_id.loc[22, "TotalPM25"] == 5.0
    assert by_id.loc[33, "TotalPM25"] == 10.25
    aermod_domain = result[result["has_aermod_primarypm25"]]
    primary_budget = float((aermod_domain["PrimaryPM25"] * aermod_domain.geometry.area).sum())
    assert primary_budget == 11.0


def test_build_full_exposure_grid_can_ignore_aermod_primary_with_inmap_only_strategy(tmp_path: Path) -> None:
    full_grid = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11, 22],
            "inmap_cell_id": [101, 202],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )
    grid_path = tmp_path / "full_grid.parquet"
    full_grid.to_parquet(grid_path, index=False)

    prepared_inmap = gpd.GeoDataFrame(
        {
            "inmap_cell_id": [101, 202],
            "inmap_PrimaryPM25": [0.1, 0.2],
            "inmap_SecondaryPM25": [1.0, 2.0],
            "geometry": full_grid.geometry,
        },
        geometry="geometry",
        crs=full_grid.crs,
    )
    prepared_aermod = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11],
            "aermod_PrimaryPM25": [10.0],
            "aermod_SecondaryPM25": [0.0],
            "has_aermod_primarypm25": [True],
            "geometry": [full_grid.geometry.iloc[0]],
        },
        geometry="geometry",
        crs=full_grid.crs,
    )

    result = _build_full_exposure_grid(
        pipeline=_pipeline(
            tmp_path,
            grid_path,
            primary_pm25_integration_strategy="inmap_only",
        ),
        prepared_inmap=prepared_inmap,
        prepared_aermod=prepared_aermod,
    ).drop(columns="geometry")

    by_id = result.set_index("aermod_cell_id")
    assert by_id.loc[11, "aermod_PrimaryPM25"] == 10.0
    assert by_id.loc[11, "aermod_PrimaryPM25_scaled"] == 0.0
    assert by_id.loc[11, "PrimaryPM25"] == 0.1
    assert by_id.loc[11, "TotalPM25"] == 1.1
    assert by_id.loc[22, "PrimaryPM25"] == 0.2
    assert by_id.loc[22, "TotalPM25"] == 2.2


def test_prepare_aermod_exposure_inputs_allows_partial_aermod_outputs(tmp_path: Path) -> None:
    aermod_gdf = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11],
            "PrimaryPM25": [0.25],
            "has_aermod_primarypm25": [True],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )

    prepared = _prepare_aermod_exposure_inputs(aermod_gdf).drop(columns="geometry")

    assert list(prepared.columns) == [
        "aermod_cell_id",
        "aermod_PrimaryPM25",
        "aermod_SecondaryPM25",
        "aermod_BC",
        "aermod_NO2",
        "has_aermod_primarypm25",
    ]
    assert prepared.loc[0, "aermod_PrimaryPM25"] == 0.25
    assert pd.isna(prepared.loc[0, "aermod_BC"])
    assert pd.isna(prepared.loc[0, "aermod_NO2"])
    assert bool(prepared.loc[0, "has_aermod_primarypm25"]) is True


def test_build_full_exposure_grid_handles_missing_aermod_bc_and_no2_columns(tmp_path: Path) -> None:
    full_grid = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11, 22],
            "inmap_cell_id": [101, 202],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )
    grid_path = tmp_path / "full_grid.parquet"
    full_grid.to_parquet(grid_path, index=False)

    prepared_inmap = gpd.GeoDataFrame(
        {
            "inmap_cell_id": [101, 202],
            "inmap_PrimaryPM25": [0.0, 0.0],
            "inmap_SecondaryPM25": [1.5, 2.5],
            "inmap_BC": [0.2, 0.4],
            "inmap_NO2": [1.2, 2.4],
            "geometry": full_grid.geometry,
        },
        geometry="geometry",
        crs=full_grid.crs,
    )
    prepared_aermod = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11],
            "aermod_PrimaryPM25": [0.25],
            "aermod_SecondaryPM25": [0.0],
            "has_aermod_primarypm25": [True],
            "geometry": [full_grid.geometry.iloc[0]],
        },
        geometry="geometry",
        crs=full_grid.crs,
    )

    result = _build_full_exposure_grid(
        pipeline=_pipeline(tmp_path, grid_path),
        prepared_inmap=prepared_inmap,
        prepared_aermod=prepared_aermod,
    ).drop(columns="geometry")

    by_id = result.set_index("aermod_cell_id")
    assert by_id.loc[11, "PrimaryPM25"] == 0.25
    assert by_id.loc[11, "BC"] == 0.2
    assert by_id.loc[11, "NO2"] == 1.2
    assert bool(by_id.loc[11, "has_aermod_primarypm25"]) is True
    assert bool(by_id.loc[11, "has_aermod_bc"]) is False
    assert bool(by_id.loc[11, "has_aermod_no2"]) is False


def test_build_full_exposure_grid_uses_per_pollutant_aermod_increment_masks(tmp_path: Path) -> None:
    full_grid = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11, 22],
            "inmap_cell_id": [101, 202],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )
    grid_path = tmp_path / "full_grid.parquet"
    full_grid.to_parquet(grid_path, index=False)

    prepared_inmap = gpd.GeoDataFrame(
        {
            "inmap_cell_id": [101, 202],
            "inmap_PrimaryPM25": [0.1, 0.2],
            "inmap_SecondaryPM25": [1.0, 2.0],
            "inmap_BC": [0.3, 0.4],
            "inmap_NO2": [5.0, 6.0],
            "geometry": full_grid.geometry,
        },
        geometry="geometry",
        crs=full_grid.crs,
    )
    prepared_aermod = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11],
            "aermod_PrimaryPM25": [0.9],
            "aermod_SecondaryPM25": [0.0],
            "aermod_BC": [0.8],
            "aermod_NO2": [7.0],
            "has_aermod_primarypm25": [True],
            "has_aermod_bc": [False],
            "has_aermod_no2": [True],
            "geometry": [full_grid.geometry.iloc[0]],
        },
        geometry="geometry",
        crs=full_grid.crs,
    )

    result = _build_full_exposure_grid(
        pipeline=_pipeline(tmp_path, grid_path),
        prepared_inmap=prepared_inmap,
        prepared_aermod=prepared_aermod,
    ).drop(columns="geometry")

    by_id = result.set_index("aermod_cell_id")
    assert by_id.loc[11, "PrimaryPM25"] == 0.9
    assert by_id.loc[11, "SecondaryPM25"] == 1.0
    assert by_id.loc[11, "TotalPM25"] == 1.9
    assert by_id.loc[11, "BC"] == 0.3
    assert by_id.loc[11, "NO2"] == 12.0
    assert by_id.loc[22, "PrimaryPM25"] == 0.2
    assert by_id.loc[22, "SecondaryPM25"] == 2.0
    assert by_id.loc[22, "TotalPM25"] == 2.2
    assert by_id.loc[22, "BC"] == 0.4
    assert by_id.loc[22, "NO2"] == 6.0
