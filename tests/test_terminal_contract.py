from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import osm_chordify
import pandas as pd
import pytest
import zarr
from shapely.geometry import LineString
from shapely.geometry import Polygon

from impacts.__main__ import main
from impacts.contract_utils import load_structured_file
from impacts.contract_utils import parquet_available
from impacts.config.builders import build_runtime_config_from_pilates
from impacts.config.builders import build_runtime_config_from_runtime_yaml
from impacts.postprocessor import create_canonical_exposure_table
from impacts.postprocessor import _read_table
from impacts.postprocessor import postprocess_from_runtime_config
from impacts.postprocessor import postprocess_from_run_manifest
from impacts.preprocessor import preprocess_workflow
from impacts.runner import _write_geoparquet_allocation
from impacts.runner import run_from_runtime_config
from impacts.runner import run_from_input_manifest
from impacts.emissions.emissions_grid_mapping import distribute_to_intersection
from impacts.emissions.emissions_grid_mapping import annualize_prepared_skims_for_grid_allocation
from impacts.emissions.emissions_grid_mapping import aggregate_allocated_intersection_rows
from impacts.emissions.emissions_grid_mapping import apply_activity_corrections
from impacts.emissions.emissions_grid_mapping import prepare_skims_for_grid_allocation
from impacts.manifest_models import InputsManifest
from impacts.manifest_models import PipelineConfig
from impacts.manifest_models import PostprocessManifest
from impacts.manifest_models import RunManifest
from impacts.network2grid.network_grid_clipping import intersect_beam_osm_with_grid
from impacts.runtime_config import ImpactsRuntimeConfig

def _write_csv(path: Path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_test_isrm_store(path: Path, n_cells: int = 4) -> Path:
    root = zarr.open(str(path), mode="w")
    for key in ["SOA", "pNO3", "pNH4", "pSO4", "PrimaryPM25"]:
        arr = root.create_array(key, shape=(1, n_cells, n_cells), dtype="f8")
        arr[:] = 0.0
        for idx in range(n_cells):
            arr[0, idx, idx] = 1.0
    total_pop = root.create_array("TotalPop", shape=(n_cells,), dtype="f8")
    total_pop[:] = 100.0
    mortality = root.create_array("MortalityRate", shape=(n_cells,), dtype="f8")
    mortality[:] = 50.0
    return path


def _build_test_workspace(tmp_path: Path) -> Path:
    inputs = tmp_path / "upstream"
    rates_dir = inputs / "rates"
    inputs.mkdir()
    rates_dir.mkdir()
    isrm_path = _build_test_isrm_store(inputs / "isrm.zarr")

    _write_csv(
        inputs / "events.csv",
        [
            {
                "type": "PathTraversal",
                "vehicle": "veh-1",
                "vehicleType": "car",
                "departureTime": 0,
                "links": "1,2",
                "linkTravelTime": "10,20",
                "length": 3000,
            }
        ],
    )
    _write_csv(
        inputs / "network.csv",
        [
            {"linkId": 1, "linkLength": 1000},
            {"linkId": 2, "linkLength": 2000},
        ],
    )
    _write_csv(
        rates_dir / "car.csv",
        [
            {"vehicleTypeId": "car", "process": "RUNEX", "pollutant": "NOx", "rate": 1.0, "rate_basis": "per_mile"},
            {"vehicleTypeId": "car", "process": "RUNLOSS", "pollutant": "PM2_5", "rate": 0.5, "rate_basis": "per_event"},
            {"vehicleTypeId": "car", "process": "PMBW", "pollutant": "PM2_5", "rate": 0.2, "rate_basis": "per_mile"},
            {"vehicleTypeId": "car", "process": "PMTW", "pollutant": "PM2_5", "rate": 0.1, "rate_basis": "per_mile"},
        ],
    )
    pd.DataFrame(
        [
            {"linkId": 1, "GRID": 1, "proportion": 1.0},
            {"linkId": 2, "GRID": 2, "proportion": 1.0},
        ]
    ).to_csv(inputs / "mapping.csv", index=False)
    _write_csv(
        inputs / "households.csv",
        [
            {"household_id": 1, "cell_id": 1, "income": "low"},
            {"household_id": 2, "cell_id": 2, "income": "high"},
        ],
    )
    _write_csv(
        inputs / "persons.csv",
        [
            {"person_id": 11, "household_id": 1, "age": 12, "sex": "F"},
            {"person_id": 12, "household_id": 1, "age": 42, "sex": "M"},
            {"person_id": 21, "household_id": 2, "age": 71, "sex": "F"},
        ],
    )

    runtime_config = tmp_path / "runtime.yaml"
    runtime_config.write_text(
        "\n".join(
            [
                "shared:",
                "  region: sfbay",
                "  geography:",
                "    fips:",
                '      state: "06"',
                "      counties:",
                '        - "001"',
                "    local_crs: EPSG:26910",
                "inputs:",
                "  beam_network: upstream/network.csv",
                "  emissions_skims: upstream/skims.csv",
                "  osm_pbf: upstream/network.osm.pbf",
                "emissions:",
                "  annualization_days: 365",
                "  pollutants: [NOx, PM2_5]",
                "dispersions:",
                "  inmap:",
                f"    isrm_zarr_directory: {isrm_path}",
                "    grid_path: upstream/mapping.csv",
                "  aermod:",
                "    grid_path: null",
                "  persons_asim_out: upstream/persons.csv",
                "  households_asim_out: upstream/households.csv",
                "outputs:",
                "  output_dir: output",
            ]
        ),
        encoding="utf-8",
    )
    return runtime_config


def _build_skims_first_workspace(tmp_path: Path) -> Path:
    inputs = tmp_path / "upstream"
    inputs.mkdir()
    isrm_path = _build_test_isrm_store(inputs / "isrm.zarr")
    pd.DataFrame(
        [
            {
                "hour": 0,
                "linkId": 1,
                "vehicleTypeId": "car",
                "process": "RUNEX",
                "emissions": "NOx:1.5;PM2_5:0.2",
                "travelTimeInSecond": 10.0,
                "parkingDurationInSecond": 0.0,
                "observations": 1,
                "iterations": 1,
            }
        ]
    ).to_csv(inputs / "skims.csv", index=False)
    _write_csv(
        inputs / "mapping.csv",
        [
            {"linkId": 1, "GRID": 1, "proportion": 1.0},
        ],
    )
    _write_csv(
        inputs / "households.csv",
        [
            {"household_id": 1, "cell_id": 1, "income": "low"},
        ],
    )
    _write_csv(
        inputs / "persons.csv",
        [
            {"person_id": 11, "household_id": 1, "age": 12, "sex": "F"},
        ],
    )
    runtime = tmp_path / "runtime_skims.yaml"
    runtime.write_text(
        "\n".join(
            [
                "shared:",
                "  region: sfbay",
                "  geography:",
                "    fips:",
                '      state: "06"',
                "      counties:",
                '        - "001"',
                "inputs:",
                "  beam_network: upstream/skims.csv",
                "  emissions_skims: upstream/skims.csv",
                "  osm_pbf: upstream/skims.csv",
                "  persons_asim_out: upstream/persons.csv",
                "  households_asim_out: upstream/households.csv",
                "emissions:",
                "  annualization_days: 365",
                "  pollutants: [NOx, PM2_5]",
                "dispersions:",
                "  inmap:",
                f"    isrm_zarr_directory: {isrm_path}",
                "    grid_path: upstream/mapping.csv",
                "  aermod:",
                "    grid_path: null",
                "outputs:",
                "  output_dir: output",
            ]
        ),
        encoding="utf-8",
    )
    return runtime


def test_preprocess_manifest_generation(tmp_path: Path):
    workflow = _build_test_workspace(tmp_path)

    manifest = preprocess_workflow(workflow, tmp_path / "workspace")

    assert Path(manifest["inputs_manifest_path"]).exists()
    assert Path(manifest["pipeline"]["events_path"]).exists()
    assert Path(manifest["pipeline"]["mapping_input_path"]).exists()
    assert Path(manifest["population_inputs"]["persons_path"]).exists()
    assert manifest["pipeline"]["use_precomputed_mapping"] is True
    assert manifest["pipeline"]["network_columns"]["link_id"] == "linkId"
    assert manifest["pipeline"]["dispersion_emissions_columns"]["grid_id"] == "GRID"
    assert manifest["pipeline"]["prepared_skims_input_path"] is None


def test_impacts_runtime_config_native_schema_round_trip():
    payload = {
        "shared": {
            "region": "sfbay",
            "geography": {
                "fips": {"state": "06", "counties": ["001", "013"]},
                "local_crs": "EPSG:7131",
                "zones": {
                    "zone_type": "taz",
                    "source_file": "/abs/path/taz.geojson",
                    "source_crs": "EPSG:4326",
                    "canonical_id_col": "taz1454",
                    "activitysim_index_col": "TAZ",
                },
                "alternative_zones": {
                    "zone_type": "taz",
                    "source_file": "/abs/path/alt_taz.shp",
                    "source_crs": "EPSG:26910",
                    "canonical_id_col": "taz1454",
                    "activitysim_index_col": "TAZ",
                },
            },
            "skims": {
                "zone_type": "taz",
                "fname": "as-base-skims-sfbay-taz.omx",
                "origin_fname": "as-origin-skims-sfbay-taz.csv.gz",
                "geoms_fname": "clipped_tazs.csv",
                "geoms_index_col": "taz1454",
            },
        },
        "inputs": {
            "beam_network": "/abs/path/network.parquet",
            "emissions_skims": "/abs/path/skims.parquet",
            "osm_pbf": "/abs/path/network.osm.pbf",
            "activity_corrections": "/abs/path/activity_corrections.csv",
            "households_asim_out": "/abs/path/households.parquet",
            "persons_asim_out": "/abs/path/persons.parquet",
        },
        "emissions": {
            "annualization_days": 365,
            "activity_correction_factors_file": "/abs/path/activity_corrections.csv",
            "pollutants": ["PM2_5", "NOx"],
        },
        "dispersions": {
            "inmap": {
                "isrm_zarr_directory": "/abs/path/isrm.zarr",
                "isrm_zarr_s3bucket": None,
                "grid_path": "/abs/path/inmap_grid.parquet",
                "grid_epsg": 4326,
                "grid_id": "zone_isrm",
            },
            "aermod": {
                "grid_path": "/abs/path/aermod_grid.parquet",
                "grid_epsg": 26910,
                "grid_id": None,
            },
        },
        "outputs": {
            "output_dir": "/abs/path/output",
            "exposure_table": "/abs/path/output/exposure_table.parquet",
        },
    }

    config = ImpactsRuntimeConfig.from_dict(payload)

    assert config.shared_context.region == "sfbay"
    assert config.shared_context.geography.fips.state == "06"
    assert config.shared_context.geography.fips.counties == ["001", "013"]
    assert config.shared_context.geography.alternative_zones.source_file == "/abs/path/alt_taz.shp"
    assert config.shared_context.skims.fname == "as-base-skims-sfbay-taz.omx"
    assert config.inputs.beam_network == "/abs/path/network.parquet"
    assert config.inputs.activity_corrections == "/abs/path/activity_corrections.csv"
    assert config.inputs.isrm_zarr == "/abs/path/isrm.zarr"
    assert config.processing.grid.inmap_grid_path == "/abs/path/inmap_grid.parquet"
    assert config.processing.grid.aermod_grid_epsg == 26910
    assert config.processing.mapping_columns.proportion == "zone_edge_proportion"
    assert config.processing.dispersion.emissions_columns.pm25 == "tons_per_year_PM2_5"
    assert config.outputs.output_dir == "/abs/path/output"
    assert config.to_dict()["shared"]["geography"]["fips"]["state"] == "06"
    assert config.to_dict()["emissions"]["annualization_days"] == 365
    assert config.to_dict()["dispersions"]["inmap"]["grid_path"] == "/abs/path/inmap_grid.parquet"


def test_impacts_runtime_config_requires_core_sections():
    with pytest.raises(ValueError, match="shared_context.geography.fips.state"):
        ImpactsRuntimeConfig.from_dict(
            {
                "shared": {"geography": {"fips": {}}},
                "inputs": {
                    "beam_network": "/abs/path/network.parquet",
                    "emissions_skims": "/abs/path/skims.parquet",
                    "osm_pbf": "/abs/path/network.osm.pbf",
                },
                "processing": {
                    "pollutants": ["PM2_5"],
                    "annualization_days": 365,
                    "grid": {"inmap_grid_path": "/abs/path/inmap_grid.parquet"},
                },
                "outputs": {"output_dir": "/abs/path/output"},
            }
        )


def test_manifest_models_round_trip():
    pipeline = PipelineConfig.from_dict(
        {
            "beam_network_path": "/tmp/network.csv",
            "beam_osm_id_col": "attributeOrigId",
            "beam_length_col": "linkLength",
            "beam_osm_epsg": 4326,
            "output_epsg": 7131,
            "inmap_grid_path": "/tmp/inmap_grid.shp",
            "inmap_grid_epsg": 4326,
            "mapping_columns": {"link_id": "edge_linkId", "proportion": "zone_edge_proportion", "grid_id": "zone_isrm"},
            "skims_columns": {"link_id": "linkId", "vehicle_type": "vehicleTypeId"},
            "isrm_url": "/tmp/isrm.zarr",
        }
    )
    inputs_manifest = InputsManifest.from_dict(
        {
            "contract_version": "1",
            "model": "impacts",
            "runtime_config_source": "/tmp/runtime.yaml",
            "staging_dir": "/tmp/workspace",
            "input_dir": "/tmp/workspace/input",
            "inputs_manifest_path": "/tmp/workspace/inputs_manifest.yaml",
            "maintained_execution_path": ["impacts.runner"],
            "inputs": {"runtime_config": {"path": "/tmp/runtime.yaml"}},
            "pipeline": pipeline.to_dict(),
            "pilates_contract": {"stage": "terminal_postprocessing"},
            "population_inputs": {},
            "notes": ["ok"],
        }
    )
    run_manifest = RunManifest.from_dict(
        {
            "contract_version": "1",
            "model": "impacts",
            "input_manifest_path": "/tmp/workspace/inputs_manifest.yaml",
            "output_dir": "/tmp/workspace/output",
            "raw_output_dir": "/tmp/workspace/output/raw",
            "command": "python -m impacts run",
            "image": "unknown",
            "raw_outputs": {"skims_emissions": "/tmp/workspace/output/raw/skims.parquet"},
            "pipeline": pipeline.to_dict(),
            "population_inputs": {},
            "deterministic_contract": {"uses_only_manifest_paths": True},
            "execution": {"stopped_after": "dispersion"},
            "run_manifest_path": "/tmp/workspace/output/run_manifest.yaml",
        }
    )
    post_manifest = PostprocessManifest.from_dict(
        {
            "contract_version": "1",
            "model": "impacts",
            "run_manifest_path": "/tmp/workspace/output/run_manifest.yaml",
            "output_dir": "/tmp/workspace/output",
            "canonical_artifact": {"path": "/tmp/workspace/output/canonical/exposure.parquet"},
            "validation": {"grid_concentration_exists": True},
            "notes": ["ok"],
            "postprocess_manifest_path": "/tmp/workspace/output/postprocess_manifest.yaml",
        }
    )

    assert inputs_manifest.to_dict()["model"] == "impacts"
    assert inputs_manifest.to_dict()["pipeline"]["beam_osm_id_col"] == "attributeOrigId"
    assert run_manifest.to_dict()["raw_outputs"]["skims_emissions"].endswith("skims.parquet")
    assert post_manifest.to_dict()["canonical_artifact"]["path"].endswith("exposure.parquet")


def test_run_manifest_model_requires_raw_skims_output():
    with pytest.raises(ValueError, match="raw_outputs.skims_emissions"):
        RunManifest.from_dict(
            {
                "contract_version": "1",
                "model": "impacts",
                "input_manifest_path": "/tmp/workspace/inputs_manifest.yaml",
                "output_dir": "/tmp/workspace/output",
                "raw_output_dir": "/tmp/workspace/output/raw",
                "command": "python -m impacts run",
                "image": "unknown",
                "raw_outputs": {},
                "pipeline": {},
                "population_inputs": {},
                "deterministic_contract": {},
                "execution": {},
                "run_manifest_path": "/tmp/workspace/output/run_manifest.yaml",
            }
        )


def test_derive_runtime_config_from_pilates_uses_shared_geography_and_runtime_overrides(tmp_path: Path):
    beam_output_dir = tmp_path / "beam_output"
    run_dir = beam_output_dir / "sfbay" / "year-2017-iteration-1"
    iters_dir = run_dir / "ITERS" / "it.3"
    iters_dir.mkdir(parents=True)
    (iters_dir / "3.skimsEmissionsTotals.csv.gz").write_text("placeholder", encoding="utf-8")
    (run_dir / "network.csv.gz").write_text("placeholder", encoding="utf-8")

    pilates_settings = tmp_path / "pilates_settings.yaml"
    pilates_settings.write_text(
        "\n".join(
            [
                "run:",
                "  region: sfbay",
                "shared:",
                "  geography:",
                "    FIPS:",
                '      state: "06"',
                "      counties:",
                '        - "001"',
                '        - "013"',
                "    local_crs: EPSG:7131",
                "  skims:",
                "    zone_type: taz",
                "    fname: as-base-skims-sfbay-taz.omx",
                "    origin_fname: as-origin-skims-sfbay-taz.csv.gz",
                "    geoms_fname: clipped_tazs.csv",
                "    geoms_index_col: taz1454",
                "activitysim:",
                "  file_format: parquet",
                "  local_output_folder: pilates/activitysim/output/",
                "  output_tables:",
                "    prefix: final_",
                "    tables:",
                    "      - households",
                    "      - persons",
                "beam:",
                "  local_input_folder: pilates/beam/production/",
                f"  local_output_folder: {beam_output_dir}",
                "  router_directory: r5/sfbay-cbg5500-weakConn-network",
            ]
        ),
        encoding="utf-8",
    )

    model_config = tmp_path / "pilates_model_config.yaml"
    model_config.write_text(
        "\n".join(
            [
                "impacts:",
                "  runtime_overrides:",
                "    emissions:",
                "      annualization_days: 365",
                "      pollutants: [PM2_5, NOx]",
                "      activity_correction_factors_file: null",
                "    dispersions:",
                "      inmap:",
                "        isrm_zarr_directory: /path/that/does/not/exist",
                "        isrm_zarr_s3bucket: s3://example/isrm.zarr",
                "        grid_path: upstream/isrm_polygon.shp",
                "        grid_epsg: 4326",
                "        grid_id: zone_isrm",
                "      aermod:",
                "        grid_path: null",
                "        grid_epsg: 4326",
                "        grid_id: null",
                "    outputs:",
                "      output_dir: /tmp/impacts-out",
                "      output_epsg: 7131",
            ]
        ),
        encoding="utf-8",
    )

    runtime_config = build_runtime_config_from_pilates(
        pilates_settings=pilates_settings,
        impacts_overlay=model_config,
    )

    assert runtime_config.shared_context.region == "sfbay"
    assert runtime_config.shared_context.geography.fips.state == "06"
    assert runtime_config.shared_context.geography.fips.counties == ["001", "013"]
    assert runtime_config.shared_context.geography.local_crs == "EPSG:7131"
    assert runtime_config.shared_context.skims.fname == "as-base-skims-sfbay-taz.omx"
    assert runtime_config.inputs.beam_network == str(run_dir / "network.csv.gz")
    assert runtime_config.inputs.emissions_skims == str(iters_dir / "3.skimsEmissionsTotals.csv.gz")
    assert runtime_config.inputs.isrm_zarr == "s3://example/isrm.zarr"
    assert (
        runtime_config.inputs.osm_pbf
        == "pilates/beam/production/r5/sfbay-cbg5500-weakConn-network/sfbay-cbg5500-weakConn-network.osm.pbf"
    )
    assert runtime_config.inputs.households_asim_out == "pilates/activitysim/output/final_households.parquet"
    assert runtime_config.inputs.persons_asim_out == "pilates/activitysim/output/final_persons.parquet"
    assert runtime_config.processing.grid.inmap_grid_path == "upstream/isrm_polygon.shp"
    assert runtime_config.processing.grid.inmap_grid_epsg == 4326
    assert runtime_config.processing.mapping_columns.grid_id == "zone_isrm"
    assert runtime_config.outputs.output_dir == "/tmp/impacts-out"


def test_derive_runtime_config_output_dir_from_pilates_run_settings(tmp_path: Path):
    pilates_settings = tmp_path / "pilates_settings.yaml"
    pilates_settings.write_text(
        "\n".join(
            [
                "run:",
                "  region: sfbay",
                "  output_directory: /tmp/pilates-output",
                "  output_run_name: scenario-a",
                "shared:",
                "  geography:",
                "    FIPS:",
                '      state: "06"',
                "      counties:",
                '        - "001"',
                "    local_crs: EPSG:7131",
            ]
        ),
        encoding="utf-8",
    )

    model_config = tmp_path / "pilates_model_config.yaml"
    model_config.write_text(
        "\n".join(
            [
                "impacts:",
                "  runtime_overrides:",
                "    inputs:",
                "      osm_pbf: upstream/network.osm.pbf",
                "      beam_network: upstream/network.csv.gz",
                "      emissions_skims: upstream/skims.csv.gz",
                "    processing:",
                "      pollutants: [PM2_5]",
                "      annualization_days: 365",
                "      grid:",
                "        inmap_grid_path: upstream/isrm_polygon.shp",
            ]
        ),
        encoding="utf-8",
    )

    runtime_config = build_runtime_config_from_pilates(
        pilates_settings=pilates_settings,
        impacts_overlay=model_config,
    )

    assert runtime_config.outputs.output_dir == "/tmp/pilates-output/scenario-a/impacts"


def test_build_runtime_config_from_runtime_yaml(tmp_path: Path):
    runtime_yaml = tmp_path / "runtime.yaml"
    runtime_yaml.write_text(
        "\n".join(
            [
                "shared:",
                "  region: sfbay",
                "  geography:",
                "    fips:",
                '      state: "06"',
                "      counties:",
                '        - "001"',
                '        - "013"',
                "inputs:",
                "  beam_network: /abs/path/network.parquet",
                "  emissions_skims: /abs/path/skims.parquet",
                "  osm_pbf: /abs/path/network.osm.pbf",
                "emissions:",
                "  annualization_days: 365",
                "  pollutants:",
                "    - PM2_5",
                "dispersions:",
                "  inmap:",
                "    grid_path: /abs/path/inmap_grid.parquet",
                "  aermod:",
                "    grid_path: null",
                "outputs:",
                "  output_dir: /abs/path/output",
            ]
        ),
        encoding="utf-8",
    )

    config = build_runtime_config_from_runtime_yaml(runtime_yaml)

    assert config.shared_context.region == "sfbay"
    assert config.inputs.osm_pbf == "/abs/path/network.osm.pbf"
    assert config.processing.annualization_days == 365


def test_example_runtime_yaml_is_native_runtime_config():
    runtime_yaml = Path(__file__).resolve().parents[1] / "examples" / "pilates" / "runtime.yaml"

    config = build_runtime_config_from_runtime_yaml(runtime_yaml)

    assert config.shared_context.region == "sfbay"
    assert config.inputs.beam_network.endswith("upstream/network.csv.gz")
    assert config.inputs.emissions_skims.endswith("0.skimsEmissionsTotals_5pct_sample.csv.gz")
    assert config.processing.grid.inmap_grid_path.endswith("upstream/isrm_polygon/isrm_polygon.shp")
    assert config.inputs.isrm_zarr == "s3://inmap-model/isrm_v1.2.1.zarr/"


def test_preprocess_accepts_native_runtime_config(tmp_path: Path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    skims_path = upstream / "skims_totals.csv"
    beam_network_path = upstream / "network.csv"
    osm_pbf_path = upstream / "network.osm.pbf"
    inmap_grid_path = upstream / "inmap_grid.shp"
    households_path = upstream / "households.csv"
    persons_path = upstream / "persons.csv"

    pd.DataFrame(
        [
            {"linkId": 101, "vehicleTypeId": "truck", "process": "RUNEX", "NOx": 10.0, "PM2_5": 1.0, "NH3": 0.0, "SOx": 0.5, "ROG": 0.2, "BCh": 0.1},
        ]
    ).to_csv(skims_path, index=False)
    pd.DataFrame([{"linkId": 101, "linkLength": 1000.0}]).to_csv(beam_network_path, index=False)
    osm_pbf_path.write_bytes(b"placeholder")
    gpd.GeoDataFrame(
        [{"geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])}],
        geometry="geometry",
        crs="EPSG:4326",
    ).to_file(inmap_grid_path, driver="GeoJSON")
    pd.DataFrame([{"household_id": 1, "cell_id": 1}]).to_csv(households_path, index=False)
    pd.DataFrame([{"person_id": 1, "household_id": 1}]).to_csv(persons_path, index=False)

    runtime_yaml = tmp_path / "runtime.yaml"
    runtime_yaml.write_text(
        "\n".join(
            [
                "shared:",
                "  region: sfbay",
                "  geography:",
                "    fips:",
                '      state: "06"',
                "      counties:",
                '        - "001"',
                '        - "013"',
                "    local_crs: EPSG:7131",
                "inputs:",
                f"  beam_network: {beam_network_path}",
                f"  emissions_skims: {skims_path}",
                f"  osm_pbf: {osm_pbf_path}",
                f"  households_asim_out: {households_path}",
                f"  persons_asim_out: {persons_path}",
                "emissions:",
                "  pollutants:",
                "    - PM2_5",
                "    - NOx",
                "  annualization_days: 365",
                "dispersions:",
                "  inmap:",
                f"    grid_path: {inmap_grid_path}",
                "  aermod:",
                "    grid_path: null",
                "outputs:",
                f"  output_dir: {tmp_path / 'output'}",
            ]
        ),
        encoding="utf-8",
    )

    manifest = preprocess_workflow(runtime_yaml, tmp_path / "workspace")
    validated = InputsManifest.from_dict(manifest)

    assert manifest["pipeline"]["county_state_fips"] == "06"
    assert manifest["pipeline"]["county_fips_codes"] == ["001", "013"]
    assert manifest["pipeline"]["beam_osm_epsg"] == 7131
    assert manifest["pipeline"]["output_epsg"] == 7131
    assert manifest["pipeline"]["prepared_pollutants"] == ["PM2_5", "NOx"]
    assert manifest["pipeline"]["annualization_days"] == 365.0
    assert Path(manifest["pipeline"]["beam_network_path"]).exists()
    assert Path(manifest["pipeline"]["inmap_grid_path"]).exists()
    assert Path(manifest["pipeline"]["prepared_skims_input_path"]).exists()
    assert manifest["pipeline"]["mapping_columns"]["grid_id"] == "grid_id"
    assert validated.inputs_manifest_path.endswith("inputs_manifest.yaml")


def test_preprocess_inferrs_grid_epsg_from_file_crs_and_falls_back_to_local_crs(tmp_path: Path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    skims_path = upstream / "skims_totals.csv"
    beam_network_path = upstream / "network.csv"
    osm_pbf_path = upstream / "network.osm.pbf"
    households_path = upstream / "households.csv"
    persons_path = upstream / "persons.csv"
    inmap_grid_path = upstream / "inmap_grid.geojson"

    pd.DataFrame(
        [
            {"linkId": 101, "vehicleTypeId": "truck", "process": "RUNEX", "NOx": 10.0, "PM2_5": 1.0, "NH3": 0.0, "SOx": 0.5, "ROG": 0.2, "BCh": 0.1},
        ]
    ).to_csv(skims_path, index=False)
    pd.DataFrame([{"linkId": 101, "linkLength": 1000.0}]).to_csv(beam_network_path, index=False)
    osm_pbf_path.write_bytes(b"placeholder")
    pd.DataFrame([{"household_id": 1, "cell_id": 1}]).to_csv(households_path, index=False)
    pd.DataFrame([{"person_id": 1, "household_id": 1}]).to_csv(persons_path, index=False)
    gpd.GeoDataFrame(
        [{"grid": 1, "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])}],
        geometry="geometry",
        crs="EPSG:3857",
    ).to_file(inmap_grid_path, driver="GeoJSON")

    runtime_yaml = tmp_path / "runtime.yaml"
    runtime_yaml.write_text(
        "\n".join(
            [
                "shared:",
                "  geography:",
                "    fips:",
                '      state: "06"',
                "    local_crs: EPSG:7131",
                "inputs:",
                f"  beam_network: {beam_network_path}",
                f"  emissions_skims: {skims_path}",
                f"  osm_pbf: {osm_pbf_path}",
                f"  households_asim_out: {households_path}",
                f"  persons_asim_out: {persons_path}",
                "emissions:",
                "  pollutants:",
                "    - PM2_5",
                "    - NOx",
                "  annualization_days: 365",
                "dispersions:",
                "  inmap:",
                f"    grid_path: {inmap_grid_path}",
                "  aermod:",
                "    grid_path: null",
                "outputs:",
                f"  output_dir: {tmp_path / 'output'}",
            ]
        ),
        encoding="utf-8",
    )

    manifest = preprocess_workflow(runtime_yaml, tmp_path / "workspace")

    assert manifest["pipeline"]["beam_osm_epsg"] == 7131
    assert manifest["pipeline"]["inmap_grid_epsg"] == 3857
    assert manifest["pipeline"]["aermod_grid_epsg"] == 7131
    assert manifest["pipeline"]["output_epsg"] == 7131


def test_run_from_runtime_config_delegates_through_preprocess(monkeypatch, tmp_path: Path):
    calls = {}

    def _fake_preprocess(runtime_config_path, staging_dir, manifest_path=None):
        calls["preprocess"] = {
            "runtime_config_path": str(runtime_config_path),
            "staging_dir": str(staging_dir),
            "manifest_path": manifest_path,
        }
        return {"inputs_manifest_path": str(tmp_path / "workspace" / "inputs_manifest.yaml")}

    def _fake_run(input_manifest_path, output_dir, run_manifest_path=None, run_dispersion=True):
        calls["run"] = {
            "input_manifest_path": str(input_manifest_path),
            "output_dir": str(output_dir),
            "run_manifest_path": run_manifest_path,
            "run_dispersion": run_dispersion,
        }
        return {"run_manifest_path": str(tmp_path / "workspace" / "output" / "run_manifest.yaml")}

    monkeypatch.setattr("impacts.preprocessor.preprocess_workflow", _fake_preprocess)
    monkeypatch.setattr("impacts.runner.run_from_input_manifest", _fake_run)

    result = run_from_runtime_config(
        runtime_config_path=tmp_path / "runtime.yaml",
        workspace=tmp_path / "workspace",
        run_dispersion=False,
    )

    assert result["run_manifest_path"].endswith("run_manifest.yaml")
    assert calls["preprocess"]["runtime_config_path"].endswith("runtime.yaml")
    assert calls["preprocess"]["staging_dir"].endswith("workspace")
    assert calls["run"]["input_manifest_path"].endswith("inputs_manifest.yaml")
    assert calls["run"]["output_dir"].endswith("workspace/output")
    assert calls["run"]["run_dispersion"] is False


def test_cli_run_accepts_runtime_config_workspace(monkeypatch, tmp_path: Path):
    calls = {}

    def _fake_run_from_runtime_config(runtime_config_path, workspace, run_manifest_path=None, run_dispersion=True):
        calls["runtime_run"] = {
            "runtime_config_path": str(runtime_config_path),
            "workspace": str(workspace),
            "run_manifest_path": run_manifest_path,
            "run_dispersion": run_dispersion,
        }
        return {"run_manifest_path": str(tmp_path / "workspace" / "output" / "run_manifest.yaml")}

    monkeypatch.setattr("impacts.runner.run_from_runtime_config", _fake_run_from_runtime_config)

    exit_code = main(
        [
            "run",
            "--config",
            str(tmp_path / "runtime.yaml"),
            "--workspace",
            str(tmp_path / "workspace"),
        ]
    )

    assert exit_code == 0
    assert calls["runtime_run"]["runtime_config_path"].endswith("runtime.yaml")
    assert calls["runtime_run"]["workspace"].endswith("workspace")


def test_postprocess_from_runtime_config_delegates_through_runner(monkeypatch, tmp_path: Path):
    calls = {}

    def _fake_runtime_run(runtime_config_path, workspace, run_manifest_path=None, run_dispersion=True):
        calls["runtime_run"] = {
            "runtime_config_path": str(runtime_config_path),
            "workspace": str(workspace),
            "run_manifest_path": run_manifest_path,
            "run_dispersion": run_dispersion,
        }
        return {"run_manifest_path": str(tmp_path / "workspace" / "output" / "run_manifest.yaml")}

    def _fake_postprocess(run_manifest_path, output_dir, manifest_path=None):
        calls["postprocess"] = {
            "run_manifest_path": str(run_manifest_path),
            "output_dir": str(output_dir),
            "manifest_path": manifest_path,
        }
        return {"postprocess_manifest_path": str(tmp_path / "workspace" / "output" / "postprocess_manifest.yaml")}

    monkeypatch.setattr("impacts.runner.run_from_runtime_config", _fake_runtime_run)
    monkeypatch.setattr("impacts.postprocessor.postprocess_from_run_manifest", _fake_postprocess)

    result = postprocess_from_runtime_config(
        runtime_config_path=tmp_path / "runtime.yaml",
        workspace=tmp_path / "workspace",
    )

    assert result["postprocess_manifest_path"].endswith("postprocess_manifest.yaml")
    assert calls["runtime_run"]["runtime_config_path"].endswith("runtime.yaml")
    assert calls["runtime_run"]["workspace"].endswith("workspace")
    assert calls["postprocess"]["run_manifest_path"].endswith("run_manifest.yaml")
    assert calls["postprocess"]["output_dir"].endswith("workspace/output")


def test_cli_postprocess_accepts_runtime_config_workspace(monkeypatch, tmp_path: Path):
    calls = {}

    def _fake_postprocess_from_runtime_config(runtime_config_path, workspace, manifest_path=None):
        calls["runtime_postprocess"] = {
            "runtime_config_path": str(runtime_config_path),
            "workspace": str(workspace),
            "manifest_path": manifest_path,
        }
        return {"postprocess_manifest_path": str(tmp_path / "workspace" / "output" / "postprocess_manifest.yaml")}

    monkeypatch.setattr("impacts.postprocessor.postprocess_from_runtime_config", _fake_postprocess_from_runtime_config)

    exit_code = main(
        [
            "postprocess",
            "--config",
            str(tmp_path / "runtime.yaml"),
            "--workspace",
            str(tmp_path / "workspace"),
        ]
    )

    assert exit_code == 0
    assert calls["runtime_postprocess"]["runtime_config_path"].endswith("runtime.yaml")
    assert calls["runtime_postprocess"]["workspace"].endswith("workspace")


def test_runner_deterministic_contract(tmp_path: Path):
    workflow = _build_test_workspace(tmp_path)
    manifest = preprocess_workflow(workflow, tmp_path / "workspace")

    first = run_from_input_manifest(manifest["inputs_manifest_path"], tmp_path / "run-a")
    second = run_from_input_manifest(manifest["inputs_manifest_path"], tmp_path / "run-b")

    first_df = _read_table(first["raw_outputs"]["grid_concentration"])
    second_df = _read_table(second["raw_outputs"]["grid_concentration"])
    pd.testing.assert_frame_equal(first_df, second_df)
    assert first["deterministic_contract"]["uses_only_manifest_paths"] is True
    assert first["deterministic_contract"]["uses_baked_work_data"] is False


def test_runner_accepts_staged_skims_directly(tmp_path: Path):
    workflow = _build_skims_first_workspace(tmp_path)
    manifest = preprocess_workflow(workflow, tmp_path / "workspace")

    run_manifest = run_from_input_manifest(manifest["inputs_manifest_path"], tmp_path / "run")

    assert manifest["pipeline"]["skims_input_path"] is not None
    assert manifest["pipeline"]["prepared_skims_input_path"] is not None
    assert manifest["pipeline"]["events_path"] is None
    assert Path(run_manifest["raw_outputs"]["skims_emissions"]).exists()


def test_prepare_skims_for_grid_allocation_preserves_process_and_downselects_pollutants(tmp_path: Path):
    skims_path = tmp_path / "skims.csv"
    pd.DataFrame(
        [
            {
                "hour": 13,
                "linkId": 1184,
                "vehicleTypeId": "truck",
                "process": "PMTW",
                "emissions": "PM2_5:0.1;NOx:1.0;SOx:0.2",
                "travelTimeInSecond": 5.0,
                "parkingDurationInSecond": 0.0,
                "observations": 2,
                "iterations": 1,
            },
            {
                "hour": 21,
                "linkId": 1184,
                "vehicleTypeId": "truck",
                "process": "RUNEX",
                "emissions": "PM2_5:0.3;NOx:2.0;ROG:0.4",
                "travelTimeInSecond": 10.0,
                "parkingDurationInSecond": 0.0,
                "observations": 3,
                "iterations": 1,
            },
        ]
    ).to_csv(skims_path, index=False)

    prepared = prepare_skims_for_grid_allocation(
        str(skims_path),
        str(tmp_path / "prepared.parquet"),
        required_pollutants=["ROG", "NOx", "NH3", "SOx", "PM2_5"],
    )

    assert list(prepared.columns) == [
        "linkId",
        "vehicleTypeId",
        "process",
        "ROG",
        "NOx",
        "NH3",
        "SOx",
        "PM2_5",
    ]
    assert len(prepared) == 2
    pmtw = prepared[prepared["process"] == "PMTW"].iloc[0]
    runex = prepared[prepared["process"] == "RUNEX"].iloc[0]
    assert pmtw["linkId"] == 1184
    assert pmtw["vehicleTypeId"] == "truck"
    assert pmtw["PM2_5"] == pytest.approx(0.2)
    assert pmtw["NOx"] == pytest.approx(2.0)
    assert pmtw["SOx"] == pytest.approx(0.4)
    assert pmtw["ROG"] == pytest.approx(0.0)
    assert pmtw["NH3"] == pytest.approx(0.0)
    assert runex["PM2_5"] == pytest.approx(0.9)
    assert runex["NOx"] == pytest.approx(6.0)
    assert runex["ROG"] == pytest.approx(1.2)
    assert runex["SOx"] == pytest.approx(0.0)
    assert runex["NH3"] == pytest.approx(0.0)


def test_prepare_skims_for_grid_allocation_accepts_totals_input(tmp_path: Path):
    skims_path = tmp_path / "skims_totals.csv"
    pd.DataFrame(
        [
            {
                "linkId": 1184,
                "vehicleTypeId": "truck",
                "process": "PMTW",
                "ROG": 0.0,
                "NOx": 2.0,
                "NH3": 0.0,
                "SOx": 0.4,
                "PM2_5": 0.2,
                "CO2": 99.0,
            },
            {
                "linkId": 1184,
                "vehicleTypeId": "truck",
                "process": "RUNEX",
                "ROG": 1.2,
                "NOx": 6.0,
                "NH3": 0.0,
                "SOx": 0.0,
                "PM2_5": 0.9,
                "CO2": 199.0,
            },
        ]
    ).to_csv(skims_path, index=False)

    prepared = prepare_skims_for_grid_allocation(
        str(skims_path),
        str(tmp_path / "prepared_totals.parquet"),
        required_pollutants=["ROG", "NOx", "NH3", "SOx", "PM2_5"],
    )

    assert list(prepared.columns) == [
        "linkId",
        "vehicleTypeId",
        "process",
        "ROG",
        "NOx",
        "NH3",
        "SOx",
        "PM2_5",
    ]
    assert "CO2" not in prepared.columns
    assert len(prepared) == 2


def test_annualize_prepared_skims_for_grid_allocation_uses_configured_days(tmp_path: Path):
    grouped_path = tmp_path / "prepared_grouped.parquet"
    pd.DataFrame(
        [
            {
                "linkId": 1184,
                "vehicleTypeId": "truck",
                "process": "RUNEX",
                "ROG": 120.0,
                "NOx": 60.0,
                "NH3": 0.0,
                "SOx": 30.0,
                "PM2_5": 12.0,
            }
        ]
    ).to_parquet(grouped_path, index=False)

    annualized = annualize_prepared_skims_for_grid_allocation(
        str(grouped_path),
        str(tmp_path / "prepared_annualized.parquet"),
        required_pollutants=["ROG", "NOx", "NH3", "SOx", "PM2_5"],
        annualization_days=330.0,
    )

    assert list(annualized.columns) == [
        "linkId",
        "vehicleTypeId",
        "process",
        "tons_per_year_ROG",
        "tons_per_year_NOx",
        "tons_per_year_NH3",
        "tons_per_year_SOx",
        "tons_per_year_PM2_5",
    ]
    row = annualized.iloc[0]
    assert row["tons_per_year_ROG"] == pytest.approx(120.0 * 330.0 / 1_000_000.0)
    assert row["tons_per_year_NOx"] == pytest.approx(60.0 * 330.0 / 1_000_000.0)
    assert row["tons_per_year_SOx"] == pytest.approx(30.0 * 330.0 / 1_000_000.0)
    assert row["tons_per_year_PM2_5"] == pytest.approx(12.0 * 330.0 / 1_000_000.0)


def test_runner_can_stop_after_emissions_allocation(tmp_path: Path):
    workflow = _build_skims_first_workspace(tmp_path)
    manifest = preprocess_workflow(workflow, tmp_path / "workspace")

    run_manifest = run_from_input_manifest(
        manifest["inputs_manifest_path"],
        tmp_path / "run",
        run_dispersion=False,
    )

    allocated_path = Path(run_manifest["raw_outputs"]["emissions_inmap_grid_allocated"])
    assert allocated_path.exists()
    assert run_manifest["raw_outputs"]["grid_concentration"] is None
    assert run_manifest["execution"]["dispersion_completed"] is False
    assert run_manifest["execution"]["stopped_after"] == "emissions_grid_allocation"

    allocated = _read_table(str(allocated_path))
    assert "GRID" in allocated.columns
    assert "cell_id" in allocated.columns
    assert "observations_allocated" not in allocated.columns


def test_emissions_allocation_respects_link_cut_proportions():
    weighted = pd.DataFrame(
        [
            {
                "hour": 8,
                "linkId": 100,
                "vehicleTypeId": "car",
                "process": "RUNEX",
                "observations_sum": 10.0,
                "em_NOx": 2.0,
                "em_PM2_5": 0.5,
                "travelTimeInSecond": 30.0,
                "parkingDurationInSecond": 0.0,
            },
            {
                "hour": 8,
                "linkId": 200,
                "vehicleTypeId": "truck",
                "process": "PMBW",
                "observations_sum": 4.0,
                "em_NOx": 0.0,
                "em_PM2_5": 1.2,
                "travelTimeInSecond": 12.0,
                "parkingDurationInSecond": 0.0,
            },
        ]
    )
    mapping = pd.DataFrame(
        [
            {"edge_linkId": 100, "zone_isrm": 1, "zone_edge_proportion": 0.25},
            {"edge_linkId": 100, "zone_isrm": 2, "zone_edge_proportion": 0.75},
            {"edge_linkId": 200, "zone_isrm": 2, "zone_edge_proportion": 0.40},
            {"edge_linkId": 200, "zone_isrm": 3, "zone_edge_proportion": 0.60},
        ]
    )

    allocated = distribute_to_intersection(
        weighted_df=weighted,
        mapping_df=mapping,
        proportion_col="zone_edge_proportion",
        mapping_columns={"link_id": "edge_linkId", "grid_id": "zone_isrm", "proportion": "zone_edge_proportion"},
    )

    assert {"GRID", "cell_id", "observations_allocated", "em_NOx_allocated", "em_PM2_5_allocated"} <= set(allocated.columns)

    link_100 = allocated[allocated["linkId"] == 100].sort_values("GRID").reset_index(drop=True)
    assert list(link_100["GRID"]) == [1, 2]
    assert link_100.loc[0, "observations_allocated"] == 2.5
    assert link_100.loc[1, "observations_allocated"] == 7.5
    assert link_100.loc[0, "em_NOx_allocated"] == 0.5
    assert link_100.loc[1, "em_NOx_allocated"] == 1.5
    assert link_100.loc[0, "em_PM2_5_allocated"] == 0.125
    assert link_100.loc[1, "em_PM2_5_allocated"] == 0.375

    link_200 = allocated[allocated["linkId"] == 200].sort_values("GRID").reset_index(drop=True)
    assert list(link_200["GRID"]) == [2, 3]
    assert link_200.loc[0, "observations_allocated"] == 1.6
    assert link_200.loc[1, "observations_allocated"] == 2.4
    assert link_200.loc[0, "em_PM2_5_allocated"] == 0.48
    assert link_200.loc[1, "em_PM2_5_allocated"] == 0.72

    # Total allocated mass and observations should be conserved per link and pollutant.
    grouped = allocated.groupby("linkId", dropna=False).sum(numeric_only=True)
    assert grouped.loc[100, "observations_allocated"] == weighted.loc[0, "observations_sum"]
    assert grouped.loc[100, "em_NOx_allocated"] == weighted.loc[0, "em_NOx"]
    assert grouped.loc[100, "em_PM2_5_allocated"] == weighted.loc[0, "em_PM2_5"]
    assert grouped.loc[200, "observations_allocated"] == weighted.loc[1, "observations_sum"]
    assert grouped.loc[200, "em_PM2_5_allocated"] == weighted.loc[1, "em_PM2_5"]
    assert all(allocated["GRID"].astype(int) == allocated["cell_id"].astype(int))


def test_annualized_emissions_allocation_respects_county_break_proportions():
    annualized = pd.DataFrame(
        [
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "process": "RUNEX",
                "tons_per_year_NOx": 10.0,
                "tons_per_year_PM2_5": 4.0,
            }
        ]
    )
    county_mapping = pd.DataFrame(
        [
            {
                "edge_linkId": 100,
                "county_fips": "001",
                "zone_edge_proportion": 0.25,
            },
            {
                "edge_linkId": 100,
                "county_fips": "013",
                "zone_edge_proportion": 0.75,
            },
        ]
    )

    allocated = distribute_to_intersection(annualized, county_mapping)

    assert len(allocated) == 2
    assert set(allocated["county_fips"]) == {"001", "013"}
    assert "observations_allocated" not in allocated.columns
    part_001 = allocated[allocated["county_fips"] == "001"].iloc[0]
    part_013 = allocated[allocated["county_fips"] == "013"].iloc[0]
    assert part_001["tons_per_year_NOx_allocated"] == pytest.approx(2.5)
    assert part_013["tons_per_year_NOx_allocated"] == pytest.approx(7.5)
    assert part_001["tons_per_year_PM2_5_allocated"] == pytest.approx(1.0)
    assert part_013["tons_per_year_PM2_5_allocated"] == pytest.approx(3.0)
    assert allocated["tons_per_year_NOx_allocated"].sum() == pytest.approx(10.0)
    assert allocated["tons_per_year_PM2_5_allocated"].sum() == pytest.approx(4.0)


def test_county_allocated_rows_collapse_by_county_dimensions():
    allocated = pd.DataFrame(
        [
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "process": "RUNEX",
                "zone_GEOID": "06001",
                "zone_NAME": "Alameda",
                "zone_COUNTYFP": "001",
                "zone_edge_proportion": 0.10,
                "tons_per_year_NOx_allocated": 1.0,
            },
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "process": "RUNEX",
                "zone_GEOID": "06001",
                "zone_NAME": "Alameda",
                "zone_COUNTYFP": "001",
                "zone_edge_proportion": 0.15,
                "tons_per_year_NOx_allocated": 1.5,
            },
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "process": "RUNEX",
                "zone_GEOID": "06013",
                "zone_NAME": "Contra Costa",
                "zone_COUNTYFP": "013",
                "zone_edge_proportion": 0.75,
                "tons_per_year_NOx_allocated": 7.5,
            },
        ]
    )

    collapsed = aggregate_allocated_intersection_rows(
        allocated,
        group_cols=["linkId", "vehicleTypeId", "process", "zone_GEOID", "zone_NAME", "zone_COUNTYFP"],
    ).sort_values("zone_COUNTYFP").reset_index(drop=True)

    assert len(collapsed) == 2
    assert list(collapsed["zone_COUNTYFP"]) == ["001", "013"]
    assert collapsed.loc[0, "zone_edge_proportion"] == pytest.approx(0.25)
    assert collapsed.loc[0, "tons_per_year_NOx_allocated"] == pytest.approx(2.5)
    assert collapsed.loc[1, "zone_edge_proportion"] == pytest.approx(0.75)
    assert collapsed.loc[1, "tons_per_year_NOx_allocated"] == pytest.approx(7.5)


def test_county_corrected_rows_collapse_across_process_within_county():
    corrected = pd.DataFrame(
        [
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "process": "RUNEX",
                "zone_GEOID": "06001",
                "zone_NAME": "Alameda",
                "zone_COUNTYFP": "001",
                "tons_per_year_NOx_allocated": 2.0,
                "tons_per_year_PM2_5_allocated": 0.5,
            },
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "process": "HOTSOAK",
                "zone_GEOID": "06001",
                "zone_NAME": "Alameda",
                "zone_COUNTYFP": "001",
                "tons_per_year_NOx_allocated": 3.0,
                "tons_per_year_PM2_5_allocated": 0.0,
            },
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "process": "RUNEX",
                "zone_GEOID": "06013",
                "zone_NAME": "Contra Costa",
                "zone_COUNTYFP": "013",
                "tons_per_year_NOx_allocated": 4.0,
                "tons_per_year_PM2_5_allocated": 1.0,
            },
        ]
    )

    collapsed = aggregate_allocated_intersection_rows(
        corrected,
        group_cols=["linkId", "vehicleTypeId", "zone_GEOID", "zone_NAME", "zone_COUNTYFP"],
    ).sort_values("zone_COUNTYFP").reset_index(drop=True)

    assert list(collapsed["zone_COUNTYFP"]) == ["001", "013"]
    assert "process" not in collapsed.columns
    assert collapsed.loc[0, "tons_per_year_NOx_allocated"] == pytest.approx(5.0)
    assert collapsed.loc[0, "tons_per_year_PM2_5_allocated"] == pytest.approx(0.5)
    assert collapsed.loc[1, "tons_per_year_NOx_allocated"] == pytest.approx(4.0)
    assert collapsed.loc[1, "tons_per_year_PM2_5_allocated"] == pytest.approx(1.0)


def test_activity_corrections_use_countyfp_and_process_family(tmp_path: Path):
    correction_path = tmp_path / "activity_corrections.csv"
    pd.DataFrame(
        [
            {"countyfp": "001", "vmt_factor": 2.0, "trips_factor": 4.0},
            {"countyfp": "013", "vmt_factor": 5.0, "trips_factor": 10.0},
        ]
    ).to_csv(correction_path, index=False)

    allocated = pd.DataFrame(
        [
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "process": "RUNEX",
                "zone_COUNTYFP": "001",
                "tons_per_year_NOx_allocated": 10.0,
            },
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "process": "HOTSOAK",
                "zone_COUNTYFP": "001",
                "tons_per_year_NOx_allocated": 10.0,
            },
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "process": "PMBW",
                "zone_COUNTYFP": "013",
                "tons_per_year_NOx_allocated": 10.0,
            },
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "process": "PRDUST",
                "zone_COUNTYFP": "013",
                "tons_per_year_NOx_allocated": 10.0,
            },
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "process": "IDLEX",
                "zone_COUNTYFP": "013",
                "tons_per_year_NOx_allocated": 10.0,
            },
        ]
    )

    corrected = apply_activity_corrections(allocated, str(correction_path))

    def _value(process: str, countyfp: str) -> float:
        row = corrected.loc[
            (corrected["process"] == process) & (corrected["zone_COUNTYFP"] == countyfp),
            "tons_per_year_NOx_allocated",
        ]
        assert len(row) == 1
        return float(row.iloc[0])

    assert _value("RUNEX", "001") == pytest.approx(10.0 / 2.0)
    assert _value("HOTSOAK", "001") == pytest.approx(10.0 / 4.0)
    assert _value("PMBW", "013") == pytest.approx(10.0 / 5.0)
    assert _value("PRDUST", "013") == pytest.approx(10.0 / 5.0)
    assert _value("IDLEX", "013") == pytest.approx(10.0)


def test_grid_allocation_from_county_level_input_matches_on_carried_county_columns():
    county_level = pd.DataFrame(
        [
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "zone_GEOID": "06001",
                "zone_NAME": "Alameda",
                "zone_COUNTYFP": "001",
                "tons_per_year_NOx_allocated": 10.0,
            }
        ]
    )
    grid_mapping = pd.DataFrame(
        [
            {
                "edge_linkId": 100,
                "edge_zone_GEOID": "06001",
                "edge_zone_NAME": "Alameda",
                "edge_zone_COUNTYFP": "001",
                "zone_isrm": 1,
                "zone_edge_proportion": 0.25,
            },
            {
                "edge_linkId": 100,
                "edge_zone_GEOID": "06001",
                "edge_zone_NAME": "Alameda",
                "edge_zone_COUNTYFP": "001",
                "zone_isrm": 2,
                "zone_edge_proportion": 0.75,
            },
            {
                "edge_linkId": 100,
                "edge_zone_GEOID": "06013",
                "edge_zone_NAME": "Contra Costa",
                "edge_zone_COUNTYFP": "013",
                "zone_isrm": 3,
                "zone_edge_proportion": 0.50,
            },
        ]
    )

    allocated = distribute_to_intersection(
        county_level,
        grid_mapping,
        proportion_col="zone_edge_proportion",
        mapping_columns={"link_id": "edge_linkId", "grid_id": "zone_isrm", "proportion": "zone_edge_proportion"},
    ).sort_values("zone_isrm").reset_index(drop=True)

    assert list(allocated["zone_isrm"].astype(int)) == [1, 2]
    assert allocated["tons_per_year_NOx_allocated_allocated"].sum() == pytest.approx(10.0)
    assert allocated.loc[0, "tons_per_year_NOx_allocated_allocated"] == pytest.approx(2.5)
    assert allocated.loc[1, "tons_per_year_NOx_allocated_allocated"] == pytest.approx(7.5)


def test_grid_allocation_uses_latest_mapping_proportion_when_input_already_has_proportion():
    county_level = pd.DataFrame(
        [
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "zone_GEOID": "06001",
                "zone_NAME": "Alameda",
                "zone_COUNTYFP": "001",
                "zone_edge_proportion": 0.40,
                "edge_link_length_m": 123.0,
                "zone_link_length_m": 49.0,
                "tons_per_year_NOx_allocated": 10.0,
            }
        ]
    )
    grid_mapping = pd.DataFrame(
        [
            {
                "edge_linkId": 100,
                "edge_zone_GEOID": "06001",
                "edge_zone_NAME": "Alameda",
                "edge_zone_COUNTYFP": "001",
                "zone_isrm": 7,
                "zone_edge_proportion": 0.25,
                "edge_link_length_m": 100.0,
                "zone_link_length_m": 25.0,
            },
            {
                "edge_linkId": 100,
                "edge_zone_GEOID": "06001",
                "edge_zone_NAME": "Alameda",
                "edge_zone_COUNTYFP": "001",
                "zone_isrm": 8,
                "zone_edge_proportion": 0.75,
                "edge_link_length_m": 100.0,
                "zone_link_length_m": 75.0,
            },
        ]
    )

    allocated = distribute_to_intersection(
        county_level,
        grid_mapping,
        proportion_col="zone_edge_proportion",
        mapping_columns={"link_id": "edge_linkId", "grid_id": "zone_isrm", "proportion": "zone_edge_proportion"},
    ).sort_values("zone_isrm").reset_index(drop=True)

    assert list(allocated["zone_isrm"].astype(int)) == [7, 8]
    assert "zone_edge_proportion_map" not in allocated.columns
    assert "zone_edge_proportion_skims" not in allocated.columns
    assert list(allocated["zone_edge_proportion"]) == [0.25, 0.75]
    assert list(allocated["edge_link_length_m"]) == [100.0, 100.0]
    assert list(allocated["zone_link_length_m"]) == [25.0, 75.0]
    assert allocated.loc[0, "tons_per_year_NOx_allocated_allocated"] == pytest.approx(2.5)
    assert allocated.loc[1, "tons_per_year_NOx_allocated_allocated"] == pytest.approx(7.5)


def test_grid_allocated_rows_collapse_by_link_vehicle_and_cell():
    allocated = pd.DataFrame(
        [
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "cell_id": 1,
                "GRID": 1,
                "zone_isrm": 1,
                "tons_per_year_NOx_allocated_allocated": 2.0,
                "tons_per_year_PM2_5_allocated_allocated": 0.5,
                "zone_edge_proportion": 0.10,
            },
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "cell_id": 1,
                "GRID": 1,
                "zone_isrm": 1,
                "tons_per_year_NOx_allocated_allocated": 3.0,
                "tons_per_year_PM2_5_allocated_allocated": 0.25,
                "zone_edge_proportion": 0.15,
            },
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "cell_id": 2,
                "GRID": 2,
                "zone_isrm": 2,
                "tons_per_year_NOx_allocated_allocated": 5.0,
                "tons_per_year_PM2_5_allocated_allocated": 1.0,
                "zone_edge_proportion": 0.75,
            },
        ]
    )

    collapsed = aggregate_allocated_intersection_rows(
        allocated,
        group_cols=["linkId", "vehicleTypeId", "cell_id", "GRID", "zone_isrm"],
    ).sort_values("cell_id").reset_index(drop=True)

    assert list(collapsed["cell_id"].astype(int)) == [1, 2]
    assert collapsed.loc[0, "tons_per_year_NOx_allocated_allocated"] == pytest.approx(5.0)
    assert collapsed.loc[0, "tons_per_year_PM2_5_allocated_allocated"] == pytest.approx(0.75)
    assert collapsed.loc[0, "zone_edge_proportion"] == pytest.approx(0.25)
    assert collapsed.loc[1, "tons_per_year_NOx_allocated_allocated"] == pytest.approx(5.0)
    assert collapsed.loc[1, "tons_per_year_PM2_5_allocated_allocated"] == pytest.approx(1.0)


def test_fine_grid_allocation_from_isrm_split_input_matches_on_carried_isrm_cell_columns():
    isrm_split = pd.DataFrame(
        [
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "zone_isrm": 1,
                "GRID": 1,
                "cell_id": 1,
                "tons_per_year_NOx_allocated_allocated": 10.0,
            }
        ]
    )
    fine_mapping = pd.DataFrame(
        [
            {
                "edge_linkId": 100,
                "edge_zone_isrm": 1,
                "edge_GRID": 1,
                "edge_cell_id": 1,
                "zone_grid100": 101,
                "zone_edge_proportion": 0.20,
            },
            {
                "edge_linkId": 100,
                "edge_zone_isrm": 1,
                "edge_GRID": 1,
                "edge_cell_id": 1,
                "zone_grid100": 102,
                "zone_edge_proportion": 0.80,
            },
            {
                "edge_linkId": 100,
                "edge_zone_isrm": 2,
                "edge_GRID": 2,
                "edge_cell_id": 2,
                "zone_grid100": 201,
                "zone_edge_proportion": 0.50,
            },
        ]
    )

    allocated = distribute_to_intersection(
        isrm_split,
        fine_mapping,
        proportion_col="zone_edge_proportion",
        mapping_columns={"link_id": "edge_linkId", "grid_id": "zone_grid100", "proportion": "zone_edge_proportion"},
    ).sort_values("zone_grid100").reset_index(drop=True)

    assert list(allocated["zone_grid100"].astype(int)) == [101, 102]
    assert allocated["tons_per_year_NOx_allocated_allocated_allocated"].sum() == pytest.approx(10.0)
    assert allocated.loc[0, "tons_per_year_NOx_allocated_allocated_allocated"] == pytest.approx(2.0)
    assert allocated.loc[1, "tons_per_year_NOx_allocated_allocated_allocated"] == pytest.approx(8.0)


def test_fine_grid_allocated_rows_collapse_by_link_vehicle_and_fine_grid_id():
    allocated = pd.DataFrame(
        [
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "cell_id": 101,
                "GRID": 101,
                "zone_grid100": 101,
                "tons_per_year_NOx_allocated_allocated_allocated": 2.0,
                "zone_edge_proportion": 0.10,
            },
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "cell_id": 101,
                "GRID": 101,
                "zone_grid100": 101,
                "tons_per_year_NOx_allocated_allocated_allocated": 3.0,
                "zone_edge_proportion": 0.15,
            },
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "cell_id": 102,
                "GRID": 102,
                "zone_grid100": 102,
                "tons_per_year_NOx_allocated_allocated_allocated": 5.0,
                "zone_edge_proportion": 0.75,
            },
        ]
    )

    collapsed = aggregate_allocated_intersection_rows(
        allocated,
        group_cols=["linkId", "vehicleTypeId", "cell_id", "GRID", "zone_grid100"],
    ).sort_values("cell_id").reset_index(drop=True)

    assert list(collapsed["cell_id"].astype(int)) == [101, 102]
    assert collapsed.loc[0, "tons_per_year_NOx_allocated_allocated_allocated"] == pytest.approx(5.0)
    assert collapsed.loc[0, "zone_edge_proportion"] == pytest.approx(0.25)


def test_write_geoparquet_allocation_joins_grid_geometry(tmp_path: Path):
    allocated = pd.DataFrame(
        [
            {
                "linkId": 100,
                "vehicleTypeId": "truck",
                "zone_isrm": 7,
                "cell_id": 7,
                "GRID": 7,
                "tons_per_year_NOx_allocated_allocated": 2.5,
            }
        ]
    )
    grid = gpd.GeoDataFrame(
        [
            {"zone_isrm": 7, "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    grid_path = tmp_path / "isrm.geojson"
    output_path = tmp_path / "allocated.parquet"
    grid.to_file(grid_path, driver="GeoJSON")

    _write_geoparquet_allocation(
        allocated,
        grid_path=str(grid_path),
        output_path=output_path,
        output_epsg=3857,
    )

    written = gpd.read_parquet(output_path)
    assert written.crs.to_epsg() == 3857
    assert "geometry" in written.columns
    assert written.loc[0, "zone_isrm"] == 7


def test_grid_intersection_proportions_match_cut_link_lengths(tmp_path: Path):
    beam_osm = gpd.GeoDataFrame(
        [
            {
                "linkId": 10,
                "linkLength": 2.0,
                "attributeOrigId": 10,
                "geometry": LineString([(0.0, 0.0), (2.0, 0.0)]),
            }
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    zones = gpd.GeoDataFrame(
        [
            {"isrm": 1, "geometry": Polygon([(0.0, -1.0), (1.0, -1.0), (1.0, 1.0), (0.0, 1.0)])},
            {"isrm": 2, "geometry": Polygon([(1.0, -1.0), (2.0, -1.0), (2.0, 1.0), (1.0, 1.0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    zones_path = tmp_path / "zones.geojson"
    zones.to_file(zones_path, driver="GeoJSON")

    result = intersect_beam_osm_with_grid(
        beam_osm_path=beam_osm,
        grid_cells_path=str(zones_path),
        output_path=str(tmp_path / "intersection.geojson"),
        beam_osm_epsg=3857,
        grid_epsg=3857,
        output_epsg=3857,
        beam_length_col="linkLength",
    )

    result = result.sort_values("zone_isrm").reset_index(drop=True)
    assert list(result["zone_isrm"].astype(int)) == [1, 2]
    assert result.loc[0, "zone_edge_proportion"] == pytest.approx(0.5)
    assert result.loc[1, "zone_edge_proportion"] == pytest.approx(0.5)
    assert result.loc[0, "proportional_length_m"] == pytest.approx(1.0)
    assert result.loc[1, "proportional_length_m"] == pytest.approx(1.0)
    assert result.loc[0, "beam_length_in_cell"] == pytest.approx(1.0)
    assert result.loc[1, "beam_length_in_cell"] == pytest.approx(1.0)
    assert result["zone_edge_proportion"].sum() == pytest.approx(1.0)


def test_grid_intersection_prefilters_cells_to_network_extent(tmp_path: Path, monkeypatch):
    beam_osm = gpd.GeoDataFrame(
        [
            {
                "linkId": 10,
                "linkLength": 2.0,
                "attributeOrigId": 10,
                "geometry": LineString([(0.0, 0.0), (2.0, 0.0)]),
            }
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    zones = gpd.GeoDataFrame(
        [
            {"zone_code": "near", "geometry": Polygon([(0.0, -1.0), (2.0, -1.0), (2.0, 1.0), (0.0, 1.0)])},
            {
                "zone_code": "far",
                "geometry": Polygon([(100.0, 100.0), (101.0, 100.0), (101.0, 101.0), (100.0, 101.0)]),
            },
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    zones_path = tmp_path / "zones.geojson"
    zones.to_file(zones_path, driver="GeoJSON")

    captured = {}
    original = osm_chordify.intersect_road_network_with_zones

    def wrapped(*args, **kwargs):
        captured["zones"] = kwargs["zones"]
        return original(*args, **kwargs)

    monkeypatch.setattr(osm_chordify, "intersect_road_network_with_zones", wrapped)

    result = intersect_beam_osm_with_grid(
        beam_osm_path=beam_osm,
        grid_cells_path=str(zones_path),
        output_path=str(tmp_path / "intersection.geojson"),
        beam_osm_epsg=3857,
        grid_epsg=3857,
        output_epsg=3857,
        beam_length_col="linkLength",
    )

    assert isinstance(captured["zones"], gpd.GeoDataFrame)
    assert list(captured["zones"]["zone_code"]) == ["near"]
    assert list(result["zone_zone_code"]) == ["near"]


def test_grid_intersection_corridor_prefilter_drops_cells_inside_bbox_but_far_from_roads(
    tmp_path: Path, monkeypatch
):
    beam_osm = gpd.GeoDataFrame(
        [
            {
                "linkId": 10,
                "linkLength": 100.0,
                "attributeOrigId": 10,
                "geometry": LineString([(0.0, 0.0), (100.0, 100.0)]),
            }
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    zones = gpd.GeoDataFrame(
        [
            {"zone_code": "near_diag", "geometry": Polygon([(48.0, 48.0), (52.0, 48.0), (52.0, 52.0), (48.0, 52.0)])},
            {"zone_code": "far_corner", "geometry": Polygon([(0.0, 90.0), (10.0, 90.0), (10.0, 100.0), (0.0, 100.0)])},
        ],
        geometry="geometry",
        crs="EPSG:3857",
    )
    zones_path = tmp_path / "zones.geojson"
    zones.to_file(zones_path, driver="GeoJSON")

    captured = {}
    original = osm_chordify.intersect_road_network_with_zones

    def wrapped(*args, **kwargs):
        captured["zones"] = kwargs["zones"]
        return original(*args, **kwargs)

    monkeypatch.setattr(osm_chordify, "intersect_road_network_with_zones", wrapped)

    result = intersect_beam_osm_with_grid(
        beam_osm_path=beam_osm,
        grid_cells_path=str(zones_path),
        output_path=str(tmp_path / "intersection.geojson"),
        beam_osm_epsg=3857,
        grid_epsg=3857,
        output_epsg=3857,
        beam_length_col="linkLength",
        prefilter_mode="corridor",
        corridor_buffer_m=10.0,
    )

    assert isinstance(captured["zones"], gpd.GeoDataFrame)
    assert list(captured["zones"]["zone_code"]) == ["near_diag"]
    assert list(result["zone_zone_code"]) == ["near_diag"]


def test_postprocess_canonical_exposure_table_generation(tmp_path: Path):
    concentration_path = tmp_path / "grid_concentration.csv"
    pd.DataFrame(
        [
            {"GRID": 1, "TotalPM25": 10.0, "deathsL": 0.1},
            {"GRID": 2, "TotalPM25": 20.0, "deathsL": 0.2},
        ]
    ).to_csv(concentration_path, index=False)
    persons_path = tmp_path / "persons.csv"
    households_path = tmp_path / "households.csv"
    _write_csv(
        households_path,
        [
            {"household_id": 1, "cell_id": 1, "income": "low"},
            {"household_id": 2, "cell_id": 2, "income": "high"},
        ],
    )
    _write_csv(
        persons_path,
        [
            {"person_id": 1, "household_id": 1, "age": 9, "sex": "F"},
            {"person_id": 2, "household_id": 1, "age": 35, "sex": "M"},
            {"person_id": 3, "household_id": 2, "age": 75, "sex": "F"},
        ],
    )

    canonical = create_canonical_exposure_table(
        concentration_path=str(concentration_path),
        persons_path=str(persons_path),
        households_path=str(households_path),
    )

    assert list(canonical["cell_id"]) == [1, 2]
    assert canonical.loc[0, "population_total"] == 2
    assert canonical.loc[1, "households_total"] == 1
    assert json.loads(canonical.loc[0, "population_mix"])["age_group_counts"]["child"] == 1


def test_postprocess_accepts_configured_column_names(tmp_path: Path):
    concentration_path = tmp_path / "grid_concentration.csv"
    pd.DataFrame(
        [
            {"zone_isrm": 7, "TotalPM25": 10.0},
            {"zone_isrm": 8, "TotalPM25": 20.0},
        ]
    ).to_csv(concentration_path, index=False)
    households_path = tmp_path / "households.csv"
    persons_path = tmp_path / "persons.csv"
    _write_csv(
        households_path,
        [
            {"hh_id": 1, "grid_ref": 7, "income_band": "low"},
            {"hh_id": 2, "grid_ref": 8, "income_band": "high"},
        ],
    )
    _write_csv(
        persons_path,
        [
            {"person_id": 1, "hh_ref": 1, "years": 9, "gender_code": "F"},
            {"person_id": 2, "hh_ref": 1, "years": 35, "gender_code": "M"},
            {"person_id": 3, "hh_ref": 2, "years": 75, "gender_code": "F"},
        ],
    )

    canonical = create_canonical_exposure_table(
        concentration_path=str(concentration_path),
        persons_path=str(persons_path),
        households_path=str(households_path),
        concentration_columns={"grid_id": "zone_isrm"},
        persons_columns={
            "household_id": "hh_ref",
            "age": "years",
            "sex": "gender_code",
        },
        households_columns={
            "household_id": "hh_id",
            "cell_id": "grid_ref",
            "income": "income_band",
        },
    )

    assert list(canonical["cell_id"]) == [7, 8]
    assert canonical.loc[0, "population_total"] == 2
    assert json.loads(canonical.loc[1, "population_mix"])["sex_counts"]["F"] == 1


def test_end_to_end_smoke_run(tmp_path: Path):
    workflow = _build_test_workspace(tmp_path)

    workspace = tmp_path / "workspace"
    exit_code = main(["pipeline", "--config", str(workflow), "--workspace", str(workspace)])

    assert exit_code == 0
    inputs_manifest = load_structured_file(workspace / "inputs_manifest.yaml")
    run_manifest = load_structured_file(workspace / "output" / "run_manifest.yaml")
    post_manifest = postprocess_from_run_manifest(
        workspace / "output" / "run_manifest.yaml",
        workspace / "output",
    )

    assert inputs_manifest["model"] == "impacts"
    assert Path(run_manifest["raw_outputs"]["grid_concentration"]).exists()
    assert Path(post_manifest["canonical_artifact"]["path"]).exists()
    canonical = _read_table(post_manifest["canonical_artifact"]["path"])
    assert "population_mix" in canonical.columns
    assert inputs_manifest["pilates_contract"]["stage"] == "terminal_postprocessing"
    assert inputs_manifest["pilates_contract"]["upstream_dependency_only"] is True


def test_standalone_example_run(tmp_path: Path):
    from examples.pilates.run_pilates_example import main as run_example_main

    example_dir = Path(__file__).resolve().parents[1] / "examples" / "pilates"
    workspace = tmp_path / "example-workspace"

    example_isrm = _build_test_isrm_store(tmp_path / "example-isrm.zarr", n_cells=6000)
    workflow_copy = tmp_path / "example-runtime.yaml"
    workflow_text = (example_dir / "runtime.yaml").read_text(encoding="utf-8")
    workflow_text = workflow_text.replace(
        "/Users/haitamlaarabi/Workspace/Simulation/sfbay/inmap/isrm_v1.2.1.zarr",
        str(example_isrm),
    )
    workflow_text = workflow_text.replace("upstream/", f"{example_dir / 'upstream'}/")
    workflow_copy.write_text(workflow_text, encoding="utf-8")

    exit_code = run_example_main(
        [
            "--config",
            str(workflow_copy),
            "--workspace",
            str(workspace),
        ]
    )

    assert exit_code == 0
    assert (workspace / "inputs_manifest.yaml").exists()
    assert (workspace / "output" / "run_manifest.yaml").exists()
    assert (workspace / "output" / "postprocess_manifest.yaml").exists() is False
    expected_name = (
        "emissions_inmap_grid_allocated.parquet" if parquet_available() else "emissions_inmap_grid_allocated.csv.gz"
    )
    assert (workspace / "output" / "raw" / expected_name).exists()
