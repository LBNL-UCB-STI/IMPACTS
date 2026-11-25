from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from impacts.manifest.schema import PipelineConfig
from impacts.manifest.file_ops import file_entry
from impacts.pipeline.preprocessing.step1_collect_inputs import _resolve_exposure_population_inputs
from impacts.pipeline.preprocessing.step4_build_cell_attributes import _classify_urban
from impacts.pipeline.preprocessing.step4_build_cell_attributes import _counts_from_joined
from impacts.pipeline.preprocessing.step4_build_cell_attributes import run as run_cell_attribute_step


def _pipeline(tmp_path: Path, grid_path: Path, *, exposure_enabled: bool = False, output_epsg: int = 26910) -> PipelineConfig:
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

    cell_path, staged_path, assignment = run_cell_attribute_step(
        _pipeline(tmp_path, grid_path),
        tmp_path / "preprocess",
        population_inputs={},
    )

    assert staged_path is None
    assert assignment is None
    result = pd.read_parquet(cell_path)
    assert result.to_dict(orient="records") == [
        {"aermod_cell_id": 10, "person_count": 0, "source_urban_class": 0},
        {"aermod_cell_id": 20, "person_count": 0, "source_urban_class": 0},
    ]


def test_explicit_exposure_population_assigns_planless_residents_once(tmp_path: Path) -> None:
    grid_path = tmp_path / "aermod_grid.parquet"
    grid = gpd.GeoDataFrame(
        {"aermod_cell_id": [10, 20]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    grid.to_parquet(grid_path, index=False)

    persons_path = tmp_path / "exposure_persons.parquet"
    households_path = tmp_path / "exposure_households.parquet"
    pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "household_id": [7, 7, 8, 9],
            "home_x": [0.5, 0.5, None, 3.0],
            "home_y": [0.5, 0.5, None, 3.0],
            "has_beam_plan": [False, False, False, False],
        }
    ).to_parquet(persons_path, index=False)
    pd.DataFrame({"household_id": [7, 8, 9]}).to_parquet(households_path, index=False)
    source_manifest_path = tmp_path / "population-source.yaml"
    source_manifest_path.write_text("source: full-geocoded-urbansim\n", encoding="utf-8")

    inputs = {
        "persons": file_entry(kind="local", path=persons_path, staged_path=persons_path),
        "households": file_entry(kind="local", path=households_path, staged_path=households_path),
        "source_manifest": {
            "path": str(source_manifest_path),
            "document": {"source": "full-geocoded-urbansim"},
        },
    }
    cell_path, staged_path, assignment = run_cell_attribute_step(
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
    assigned = pd.read_parquet(staged_path)
    assert assigned["person_id"].tolist() == [1, 2]
    assert assigned["aermod_cell_id"].tolist() == [10, 10]
    counts = pd.read_parquet(cell_path).set_index("aermod_cell_id")["person_count"]
    assert counts.loc[10] == 2


def test_missing_explicit_exposure_files_fail_even_if_beam_population_exists(tmp_path: Path) -> None:
    beam_population = tmp_path / "beam" / "input" / "urbansim"
    beam_population.mkdir(parents=True)
    (beam_population / "persons.parquet").touch()
    (beam_population / "households.parquet").touch()
    settings = SimpleNamespace(
        impacts=SimpleNamespace(
            exposure=SimpleNamespace(persons_file=None, households_file=None, source_manifest=None)
        )
    )

    try:
        _resolve_exposure_population_inputs(settings, config_path=tmp_path / "settings.yaml")
    except ValueError as exc:
        assert "explicit impacts.exposure.persons_file" in str(exc)
    else:
        raise AssertionError("missing explicit exposure inputs unexpectedly used BEAM population files")


def test_classify_urban_thresholds() -> None:
    counts = pd.Series([0, 999, 1000, 9999, 10000, 50000])
    result = _classify_urban(counts).tolist()
    assert result == [0, 0, 1000, 1000, 10000, 10000]


if __name__ == "__main__":
    with TemporaryDirectory() as temp_dir:
        test_explicit_exposure_population_assigns_planless_residents_once(Path(temp_dir))
    with TemporaryDirectory() as temp_dir:
        test_missing_explicit_exposure_files_fail_even_if_beam_population_exists(Path(temp_dir))
