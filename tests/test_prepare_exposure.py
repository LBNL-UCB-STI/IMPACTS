from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from impacts.manifest.schema import PipelineConfig
from impacts.workflow.step4_prepare_exposure import _build_full_exposure_grid


def _pipeline(tmp_path: Path, grid_path: Path) -> PipelineConfig:
    return PipelineConfig.from_dict(
        {
            "beam_osm_id_col": "attributeOrigId",
            "beam_length_col": "linkLength",
            "output_epsg": 26910,
            "inmap_enabled": True,
            "aermod_enabled": True,
            "inmap_grid_path": str(tmp_path / "inmap_grid.parquet"),
            "inmap_grid_epsg": 26910,
            "mapping_columns": {"link_id": "linkId", "grid_id": "isrm"},
            "isrm_url": str(tmp_path / "isrm.zarr"),
            "isrm_nox_to_no2_ratios_file": str(tmp_path / "matrix.npz"),
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
            "activity_totals_file": str(tmp_path / "activity.parquet"),
            "activity_totals_columns": {"county": "countyfp", "year": "year"},
            "prepared_skims_group_cols": ["hour", "linkId"],
            "pollutants": ["NOx", "PM2_5", "BC"],
            "pollutants_map": {"NOx": "NOx", "PM2_5": "PM2_5", "BC": "BC"},
            "annualization_days": 330.0,
            "population_sample": 0.1,
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
