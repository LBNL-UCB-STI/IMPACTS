from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from impacts.manifest.schema import PipelineConfig
from impacts.pipeline.preprocessing.step4_build_cell_attributes import _classify_urban
from impacts.pipeline.preprocessing.step4_build_cell_attributes import _counts_from_joined
from impacts.pipeline.preprocessing.step4_build_cell_attributes import run as run_cell_attribute_step


def _pipeline(tmp_path: Path, grid_path: Path) -> PipelineConfig:
    return PipelineConfig.from_dict(
        {
            "beam_osm_id_col": "attributeOrigId",
            "beam_length_col": "linkLength",
            "output_epsg": 26910,
            "emissions_enabled": True,
            "inmap_enabled": False,
            "aermod_enabled": True,
            "exposure_enabled": False,
            "mapping_columns": {"link_id": "linkId", "grid_id": "isrm"},
            "grid_size_meters": 100.0,
            "asrv_patterns_file": str(tmp_path / "asrv.parquet"),
            "asrv_patterns_epsg": 4326,
            "aermod_full_grid_path": str(grid_path),
            "aermod_grid_path": str(grid_path),
            "aermod_grid_epsg": 26910,
            "aermod_grid_id": "aermod_cell_id",
            "region": "sfbay",
            "start_year": 2018,
            "county_state_fips": "06",
            "county_fips_codes": ["001"],
            "passenger_inventory_file": str(tmp_path / "passenger_inventory.parquet"),
            "freight_inventory_file": str(tmp_path / "freight_inventory.parquet"),
            "passenger_vehicle_types_file": str(tmp_path / "vehicleTypes--atlas.csv"),
            "freight_vehicle_types_file": str(tmp_path / "vehicleTypes--frism.csv"),
            "prepared_skims_group_cols": ["hour", "linkId"],
            "pollutants": ["NOx", "PM25"],
            "source_pollutants": ["NOx", "PM25"],
            "annualization_days": {"light_duty": 327.0, "medium_heavy_duty": 312.0},
            "population_sample": 0.1,
            "primary_pm25_integration_strategy": "impute_inmap_primary_in_aermod_domain",
        }
    )


def test_counts_from_joined_keeps_unpopulated_domain_cells(tmp_path: Path) -> None:
    joined = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "aermod_cell_id": [10, 10, 20, 20, 20],
        }
    )
    aermod_grid = pd.DataFrame({"aermod_cell_id": [10, 20, 30]})

    result = _counts_from_joined(joined, aermod_grid=aermod_grid, working_dir=tmp_path)

    assert result["aermod_cell_id"].tolist() == [10, 20, 30]
    assert result["person_count"].tolist() == [2, 3, 0]
    assert result["source_urban_class"].tolist() == [0, 0, 0]


def test_population_step_writes_zero_attributes_without_population_inputs(tmp_path: Path) -> None:
    grid_path = tmp_path / "aermod_grid.parquet"
    grid = gpd.GeoDataFrame(
        {"aermod_cell_id": [10, 20]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:26910",
    )
    grid.to_parquet(grid_path, index=False)

    cell_path, staged_path = run_cell_attribute_step(
        _pipeline(tmp_path, grid_path),
        tmp_path / "preprocess",
        population_inputs={},
    )

    assert staged_path is None
    result = pd.read_parquet(cell_path)
    assert result.to_dict(orient="records") == [
        {"aermod_cell_id": 10, "person_count": 0, "source_urban_class": 0},
        {"aermod_cell_id": 20, "person_count": 0, "source_urban_class": 0},
    ]


def test_classify_urban_thresholds() -> None:
    counts = pd.Series([0, 999, 1000, 9999, 10000, 50000])
    result = _classify_urban(counts).tolist()
    assert result == [0, 0, 1000, 1000, 10000, 10000]
