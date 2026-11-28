from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import tarfile
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from impacts.census_bundle import stage_census_bundle
from impacts.pipeline.preprocessing.step1_collect_inputs import _resolve_census_bundle_path
from impacts.manifest.file_ops import file_entry
from impacts.manifest.schema import PipelineConfig
from impacts.pipeline.preprocessing.step1_collect_inputs import _resolve_exposure_population_h5
from impacts.pipeline.preprocessing.step4_assign_raw_population_to_grid import run as run_population_assignment_step
from impacts.pipeline.preprocessing.step5_build_aermod_cell_attributes import _classify_urban
from impacts.pipeline.preprocessing.step5_build_aermod_cell_attributes import _complete_cell_attributes
from impacts.pipeline.preprocessing.step5_build_aermod_cell_attributes import run as run_cell_attribute_step
from impacts.pipeline.preprocessing.step8_assign_group_quarters_to_grid import _allocate_blocks
from impacts.pipeline.preprocessing.step8_assign_group_quarters_to_grid import run as run_group_quarters


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


def test_offline_census_bundle_allocates_tract_targets_to_the_grid(tmp_path: Path) -> None:
    source = tmp_path / "source"
    manifest = {
        "bundle_id": "fixture",
        "baseline_year": 2018,
        "scenario_region": "sfbay",
        "files": [],
        "references": [],
    }

    def add_file(name: str, data: bytes, **metadata) -> None:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest["files"].append({
            "path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), **metadata,
        })

    for table in ("B01003", "B11001", "B11002", "B26001"):
        add_file(
            f"acs/2018/1yr/{table.lower()}.dat",
            f"#GEO_ID|{table}_001E|{table}_001M\n0500000US06001|150|1\n".encode(),
            dataset="ACS",
            table=table,
            vintage="2018",
            product="1yr",
        )
    add_file(
        "acs/2018/5yr/b26001.dat",
        b"#GEO_ID|B26001_001E|B26001_001M\n1400000US06001000100|40|1\n1400000US06001000200|60|2\n",
    )
    p42_path = tmp_path / "ca2010.sgq.zip"
    a, b, c = "060010001001001", "060010002001001", "060010002001002"
    header = bytearray(b" " * 500)
    header[8:11] = b"750"
    header[18:25] = b"0000001"
    header[27:32] = a[:5].encode()
    header[54:60] = a[5:11].encode()
    header[61:65] = a[11:].encode()
    with ZipFile(p42_path, "w") as archive:
        archive.writestr("cageo2010.sgq", header + b"\n")
        archive.writestr("ca000012010.sgq", "GQ,CA,000,01,0000001,40\n")
    add_file("census/2010/ca2010.sgq.zip", p42_path.read_bytes(), table="P42")
    add_file(
        "census/luca/2018/2020LUCA_ST06_address_countlist.csv",
        f"GEOID,CURRENT_COUNT_OF_GQs\n{a},1\n{b},1\n{c},2\nUnable to Geocode,2\nTOTAL,6\n".encode(),
        dataset="2020 Census LUCA Address Count List",
        vintage="2018-01",
        field="CURRENT_COUNT_OF_GQs",
    )
    reference_path = source / "census/reference/luca.pdf"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_bytes(b"documentation")
    manifest["references"].append({
        "path": "census/reference/luca.pdf",
        "bytes": reference_path.stat().st_size,
        "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    })
    manifest_path = source / "census/fixture.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    archive_path = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in (
            manifest_path,
            *(source / entry["path"] for entry in manifest["files"]),
            *(source / entry["path"] for entry in manifest["references"]),
        ):
            archive.add(path, arcname=path.relative_to(source))

    cache = tmp_path / "cache"
    bundle = stage_census_bundle(archive_path, cache, year=2018, region="sfbay", state_fips="06")
    assert stage_census_bundle(archive_path, cache, year=2018, region="sfbay", state_fips="06") == bundle
    assert bundle["archive_sha256"] == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert bundle["source_vintages"] == ["2018", "2018-01"]
    assert {entry.get("vintage") for entry in bundle["source_files"] if entry.get("dataset") == "ACS"} == {"2018"}
    with pytest.raises(ValueError, match="baseline year 2019"):
        stage_census_bundle(archive_path, cache, year=2019, region="sfbay", state_fips="06")
    h5 = tmp_path / "population.h5"
    pd.DataFrame({"block_id": [a, b, c], "x": [0.5, 1.5, 3.0], "y": [0.5, 0.5, 0.5]}).to_hdf(
        h5, key="/2018/blocks"
    )
    grid_path = tmp_path / "grid.parquet"
    gpd.GeoDataFrame(
        {"aermod_cell_id": [10, 20]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs=4326,
    ).to_parquet(grid_path)
    output = tmp_path / "preprocess"
    output.mkdir()
    block_path, cell_path, report = run_group_quarters(
        _pipeline(tmp_path, grid_path, output_epsg=4326),
        output,
        population_inputs={
            "census_bundle": bundle,
            "acs_tables": bundle["acs_tables"],
            "year": 2018,
            "h5": file_entry(kind="local", path=h5, staged_path=h5),
        },
    )
    blocks = pd.read_parquet(block_path)
    assert blocks["group_quarters_population"].tolist() == [60, 30, 60]
    assert blocks["method"].tolist() == ["p42_residents", "luca_address_count", "luca_address_count"]
    assert pd.read_parquet(cell_path)["group_quarters_population"].tolist() == [60, 30]
    assert report["acs_tract_target"] == 150
    assert report["assigned_to_grid"] == 90
    assert report["blocks_missing_coordinates"] == 0
    assert report["outside_grid"] == 60
    assert report["blocks_outside_grid"] == 1
    Path(bundle["luca"]).write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        stage_census_bundle(archive_path, cache, year=2018, region="sfbay", state_fips="06")
    Path(bundle["luca"]).write_bytes(
        (source / "census/luca/2018/2020LUCA_ST06_address_countlist.csv").read_bytes()
    )
    cached_reference = Path(bundle["manifest"]).parent / "reference/luca.pdf"
    cached_reference.write_bytes(b"corrupt documentation")
    with pytest.raises(ValueError, match="checksum mismatch"):
        stage_census_bundle(archive_path, cache, year=2018, region="sfbay", state_fips="06")


def test_census_bundle_path_is_resolved_from_beam_region_root(tmp_path: Path) -> None:
    region_root = tmp_path / "beam_input" / "sfbay"
    bundle = region_root / "impacts/census/bundle.tar.gz"
    bundle.parent.mkdir(parents=True)
    bundle.touch()

    assert _resolve_census_bundle_path("impacts/census/bundle.tar.gz", region_root) == bundle.resolve()
    with pytest.raises(FileNotFoundError, match="BEAM input region"):
        _resolve_census_bundle_path("impacts/census/missing.tar.gz", region_root)
    with pytest.raises(ValueError, match="stay inside the BEAM input region"):
        _resolve_census_bundle_path("../../outside.tar.gz", region_root)


def test_multiple_luca_candidates_use_occupancy_when_all_are_verified() -> None:
    tract = "06001000100"
    a, b = tract + "1001", tract + "1002"
    target = pd.DataFrame({
        "tract_geoid": [tract], "tract_target": [100], "acs_5yr_estimate": [105], "acs_5yr_moe": [5],
    })
    luca = pd.Series({a: 1, b: 1})
    p42 = pd.Series({a: 0})
    result = _allocate_blocks(target, p42, luca, pd.Series({a: 60, b: 20}))
    assert result["group_quarters_population"].tolist() == [75, 25]
    assert result["method"].unique().tolist() == ["luca_verified_occupancy"]
    result = _allocate_blocks(target, p42, luca, pd.Series({a: 60}))
    assert result["group_quarters_population"].tolist() == [50, 50]
    assert result["method"].unique().tolist() == ["luca_address_count"]


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
