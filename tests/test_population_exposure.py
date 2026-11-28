from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from impacts.manifest.file_ops import file_entry
from impacts.manifest.schema import PipelineConfig
from impacts.pipeline.preprocessing.step1_collect_inputs import _resolve_exposure_population_h5
from impacts.pipeline.preprocessing.step4_assign_raw_population_to_grid import run as run_population_assignment_step
from impacts.pipeline.preprocessing.step5_build_aermod_cell_attributes import _classify_urban
from impacts.pipeline.preprocessing.step5_build_aermod_cell_attributes import _complete_cell_attributes
from impacts.pipeline.preprocessing.step5_build_aermod_cell_attributes import run as run_cell_attribute_step


def _pipeline(
    tmp_path: Path,
    grid_path: Path,
    *,
    exposure_enabled: bool = False,
    output_epsg: int = 26910,
) -> PipelineConfig:
    return PipelineConfig.from_dict(
        {
            "beam_osm_id_col": "attributeOrigId",
            "beam_length_col": "linkLength",
            "output_epsg": output_epsg,
            "emissions_enabled": True,
            "inmap_enabled": False,
            "aermod_enabled": True,
            "exposure_enabled": exposure_enabled,
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


def test_cell_attributes_keep_unpopulated_domain_cells() -> None:
    aermod_grid = pd.DataFrame({"aermod_cell_id": [10, 20, 30]})
    population_counts = pd.DataFrame(
        {"aermod_cell_id": [10, 20], "person_count": [2, 3]}
    )

    result = _complete_cell_attributes(aermod_grid, population_counts)

    assert result["aermod_cell_id"].tolist() == [10, 20, 30]
    assert result["person_count"].tolist() == [2, 3, 0]
    assert result["source_urban_class"].tolist() == [0, 0, 0]


def test_population_and_aermod_steps_have_separate_outputs(tmp_path: Path) -> None:
    grid_path = tmp_path / "aermod_grid.parquet"
    grid = gpd.GeoDataFrame(
        {"aermod_cell_id": [10, 20]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:26910",
    )
    grid.to_parquet(grid_path, index=False)

    staged_path, counts_path, assignment, household_metadata_path = run_population_assignment_step(
        _pipeline(tmp_path, grid_path),
        tmp_path / "preprocess",
        population_inputs={},
    )

    assert staged_path is None
    assert counts_path is None
    assert assignment is None
    assert household_metadata_path is None
    cell_path = run_cell_attribute_step(
        _pipeline(tmp_path, grid_path),
        tmp_path / "preprocess",
    )
    result = pd.read_parquet(cell_path)
    assert result.to_dict(orient="records") == [
        {"aermod_cell_id": 10, "person_count": 0, "source_urban_class": 0},
        {"aermod_cell_id": 20, "person_count": 0, "source_urban_class": 0},
    ]


def test_raw_urbansim_h5_assigns_all_residents_once(tmp_path: Path) -> None:
    grid_path = tmp_path / "aermod_grid.parquet"
    grid = gpd.GeoDataFrame(
        {"aermod_cell_id": [10, 20]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    grid.to_parquet(grid_path, index=False)

    population_h5 = tmp_path / "model_data_2018.h5"
    with pd.HDFStore(population_h5, mode="w") as store:
        store.put(
            "/2018/persons",
            pd.DataFrame(
                {"person_id": [1, 2, 3, 4], "household_id": [7, 7, 8, 9]}
            ).set_index("person_id"),
        )
        store.put(
            "/2018/households",
            pd.DataFrame(
                {"household_id": [7, 8, 9], "block_id": [70, 80, 90], "persons": [2, 1, 1]}
            ).set_index("household_id"),
        )
        store.put(
            "/2018/blocks",
            pd.DataFrame(
                {
                    "block_id": [70, 80, 90],
                    "county_id": ["06001", "06001", "06001"],
                    "x": [0.5, None, 3.0],
                    "y": [0.5, None, 3.0],
                }
            ).set_index("block_id"),
        )

    inputs = {
        "h5": file_entry(
            kind="local", path=population_h5, staged_path=population_h5
        ),
        "year": 2018,
        "acs_tables": {"1yr": {}, "5yr": {}},
    }
    staged_path, counts_path, assignment, household_metadata_path = run_population_assignment_step(
        _pipeline(tmp_path, grid_path, exposure_enabled=True, output_epsg=4326),
        tmp_path / "preprocess",
        population_inputs=inputs,
    )

    assert assignment == {
        "residents_supplied": 4,
        "residents_assigned_to_cell": 2,
        "residents_missing_home_coordinates": 1,
        "residents_outside_grid": 1,
    }
    assert staged_path is not None
    assert counts_path is not None
    assert household_metadata_path is not None
    assert not (tmp_path / "preprocess" / "aermod_cell_attributes.parquet").exists()
    assigned = pd.read_parquet(staged_path)
    assert assigned["person_id"].tolist() == [1, 2]
    assert assigned["aermod_cell_id"].tolist() == [10, 10]
    households = pd.read_parquet(household_metadata_path)
    assert households[["household_id", "county_geoid", "household_size"]].to_dict("records") == [
        {"household_id": 7, "county_geoid": "06001", "household_size": 2},
        {"household_id": 8, "county_geoid": "06001", "household_size": 1},
        {"household_id": 9, "county_geoid": "06001", "household_size": 1},
    ]
    cell_path = run_cell_attribute_step(
        _pipeline(tmp_path, grid_path, exposure_enabled=True, output_epsg=4326),
        tmp_path / "preprocess",
        population_counts_path=counts_path,
    )
    counts = pd.read_parquet(cell_path).set_index("aermod_cell_id")["person_count"]
    assert counts.loc[10] == 2


def test_missing_exposure_h5_fails_even_if_beam_population_exists(tmp_path: Path) -> None:
    beam_population = tmp_path / "beam" / "input" / "urbansim"
    beam_population.mkdir(parents=True)
    (beam_population / "persons.parquet").touch()
    (beam_population / "households.parquet").touch()
    settings = SimpleNamespace(
        impacts=SimpleNamespace(exposure=SimpleNamespace(population_h5=None))
    )

    try:
        _resolve_exposure_population_h5(
            settings, config_path=tmp_path / "settings.yaml"
        )
    except ValueError as exc:
        assert "impacts.exposure.population_h5" in str(exc)
    else:
        raise AssertionError(
            "missing exposure HDF5 unexpectedly used BEAM population files"
        )


def test_classify_urban_thresholds() -> None:
    counts = pd.Series([0, 999, 1000, 9999, 10000, 50000])
    result = _classify_urban(counts).tolist()
    assert result == [0, 0, 1000, 1000, 10000, 10000]
