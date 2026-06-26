from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import yaml
from pathlib import Path
from types import SimpleNamespace

import pytest

from impacts.__main__ import main
from impacts.config.path_registry import PathRegistry
from impacts.config.settings_builder import build_settings_from_pilates
from impacts.config.settings_builder import load_settings_from_yaml
from impacts.manifest.schema import PreprocessManifest
from impacts.manifest.schema import PipelineConfig
from impacts.manifest.schema import PostprocessManifest
from impacts.manifest.schema import PipelineManifest
from impacts.manifest.schema import ActivitiesManifest
from impacts.postprocessor import postprocess_from_pipeline_manifest
from impacts.postprocessor import postprocess_from_settings
from impacts.pipeline.preprocessing.step1_collect_inputs import _resolve_region_input_root
from impacts.runner import run_emissions_from_pipeline_manifest


def _pipeline_payload(tmp_path: Path) -> dict:
    return {
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
        "asrv_patterns_file": str(tmp_path / "asrv_patterns.parquet"),
        "asrv_patterns_epsg": 4326,
        "aermod_full_grid_path": str(tmp_path / "aermod_full_grid.parquet"),
        "aermod_grid_path": str(tmp_path / "aermod_grid.parquet"),
        "aermod_grid_epsg": 26910,
        "aermod_grid_id": "aermod_id",
        "region": "sfbay",
        "start_year": 2017,
        "county_state_fips": "06",
        "county_fips_codes": ["001", "013"],
        "passenger_inventory_file": str(tmp_path / "passenger_inventory.parquet"),
        "freight_inventory_file": str(tmp_path / "freight_inventory.parquet"),
        "enable_passenger_inventory_activity_correction": True,
        "enable_freight_inventory_activity_correction": True,
        "passenger_vehicle_types_file": str(tmp_path / "vehicleTypes--atlas.csv"),
        "freight_vehicle_types_file": str(tmp_path / "vehicleTypes--frism.csv"),
        "prepared_skims_group_cols": ["hour", "linkId"],
        "pollutants": ["NOx", "PM25"],
        "source_pollutants": ["NOx", "PM25"],
        "vehicle_category_metadata_file": str(tmp_path / "vehicle_category_metadata.csv"),
        "annualization_days": {"light_duty": 327.0, "medium_heavy_duty": 312.0},
        "population_sample": 0.1,
        "primary_pm25_integration_strategy": "impute_inmap_primary_in_aermod_domain",
        "include_passenger": True,
        "include_freight": True,
    }


def _preprocess_manifest_payload(tmp_path: Path) -> dict:
    return {
        "contract_version": "1",
        "model": "impacts",
        "settings_source": str(tmp_path / "settings.yaml"),
        "staging_dir": str(tmp_path / "workspace"),
        "input_dir": str(tmp_path / "workspace" / "staged"),
        "preprocess_manifest_path": str(tmp_path / "workspace" / "preprocess_manifest.yaml"),
        "maintained_execution_path": [
            "impacts.pipeline.workflow.step1_process_emissions",
            "impacts.pipeline.workflow.step2_compute_inmap_concentrations",
            "impacts.pipeline.workflow.step3_compute_aermod_concentrations",
        ],
        "inputs": {
            "settings": {"source_path": str(tmp_path / "settings.yaml")},
            "county_intersection": {"source_path": str(tmp_path / "beam_osm_county_intersection.parquet")},
            "inmap_intersection": {"source_path": str(tmp_path / "beam_osm_inmap_intersection.parquet")},
            "aermod_intersection": {"source_path": str(tmp_path / "beam_osm_aermod_intersection.parquet")},
        },
        "pipeline": _pipeline_payload(tmp_path),
        "pilates_contract": {"stage": "terminal_postprocessing"},
        "population_inputs": {},
        "notes": [],
    }


def test_example_settings_yaml_is_current_settings_file():
    settings_yaml = Path(__file__).resolve().parents[1] / "settings.yaml"

    config = load_settings_from_yaml(settings_yaml)

    assert config.run.region == "sfbay"
    assert config.run.scenario == "base"
    assert config.shared.geography.fips.state == "06"
    assert config.shared.geography.fips.counties[0] == "001"
    assert config.beam.local_input_folder == "~/Workspace/Models/beam/beam-data"
    assert config.beam.local_output_folder == "beam/beam_output"
    assert config.beam.router_directory == "r5/sfbay-cbg5500-weakConn-network"
    assert config.impacts.local_input_folder == "impacts/impacts_inputs"
    assert config.impacts.local_output_folder == "impacts/impacts_output"
    assert config.impacts.pipeline.inmap is True
    assert config.impacts.dispersions.inmap.grid_path.endswith("isrm_polygon_wgs84.gpkg")
    assert config.impacts.dispersions.inmap.isrm_nox_to_no2_ratios_apply_tons_per_year_to_ug_per_s is False
    assert config.impacts.pipeline.aermod is True
    assert config.impacts.pipeline.exposure is True
    assert config.impacts.exposure.primary_pm25_integration_strategy == "impute_inmap_primary_in_aermod_domain"
    assert config.impacts.population.passenger_folder == "urbansim/atlas-2019"
    assert config.impacts.emissions.beam.include_passenger is True
    assert config.impacts.emissions.beam.include_freight is True
    assert len(config.impacts.analysis.sector_targets) == 6
    assert config.impacts.emissions.vehicle_category_metadata_file.endswith(
        "vehicle-tech/emissions/emissions_vehicle_categories.csv"
    )
    assert config.impacts.emissions.defaults.default_annualization_days.light_duty == 327.0
    assert config.impacts.emissions.defaults.default_annualization_days.medium_heavy_duty == 312.0
    assert config.impacts.analysis.sector_targets[0].source == "mobile_onroad"
    assert config.impacts.analysis.sector_targets[0].sector == "passenger_cars"
    assert config.impacts.analysis.sector_targets[0].annual_pm25_short_tons == 714.26
    assert config.impacts.analysis.sector_targets[0].annual_nox_short_tons == 3964.37
    assert config.impacts.analysis.sector_targets[-1].source == "road_dust"
    assert config.impacts.analysis.sector_targets[-1].annual_pm25_short_tons == 1499.25
    assert config.impacts.analysis.sector_targets[-1].annual_nox_short_tons is None
    assert len(config.impacts.analysis.inventory_targets) == 1
    assert config.impacts.analysis.inventory_targets[0].name == "mobile_on_road"


def test_resolve_region_input_root_accepts_direct_region_root_layout(tmp_path: Path) -> None:
    beam_input_root = tmp_path / "beam-data-sfbay"
    (beam_input_root / "freight").mkdir(parents=True)
    (beam_input_root / "vehicle-tech").mkdir()

    resolved = _resolve_region_input_root(beam_input_root=beam_input_root, region="sfbay")

    assert resolved == beam_input_root


def test_builtin_settings_source_of_truth_is_current() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_payload = yaml.safe_load((repo_root / "settings.yaml").read_text())
    default_settings_path = repo_root / "src" / "impacts" / "config" / "settings.yaml"
    default_payload = yaml.safe_load(default_settings_path.read_text())
    default_config = load_settings_from_yaml(default_settings_path)

    shared_keys = {"emissions", "dispersions", "exposure", "analysis"}
    assert shared_keys <= set(default_payload["impacts"])
    assert shared_keys <= set(example_payload["impacts"])
    assert default_config.impacts.activities["project_analysis"]["main"]["folder_in_archive"] == "sfbay-emfac-project-analysis"
    assert default_config.impacts.population.vehicle_folder == "vehicle-tech"
    assert default_config.impacts.exposure.primary_pm25_integration_strategy == "impute_inmap_primary_in_aermod_domain"


def test_native_settings_loader_normalizes_directly(tmp_path: Path) -> None:
    settings_yaml = tmp_path / "settings.yaml"
    settings_yaml.write_text(
        "\n".join(
            [
                "run:",
                "  region: sfbay",
                "  scenario: base",
                "  start_year: 2018",
                "shared:",
                "  geography:",
                "    FIPS:",
                '      state: "06"',
                "      counties:",
                '        - "001"',
                "    local_crs: 26910",
                "beam:",
                "  local_input_folder: pilates/beam/production",
                "  local_output_folder: beam/beam_output",
                "impacts:",
                "  local_input_folder: impacts/impacts_inputs",
                "  local_output_folder: impacts/impacts_output",
                "  scenario: 2018-Baseline",
                "  pipeline:",
                "    postsim:",
                "      inmap: false",
                "      aermod: false",
                "  population:",
                "    vehicle_folder: vehicle-tech",
                "  emissions:",
                "    pollutants: [NOx]",
                "    default_annualization_days:",
                "      light_duty: 327.0",
                "      medium_heavy_duty: 312.0",
                "  dispersions:",
                "    inmap: {}",
                "    aermod: {}",
            ]
        ),
        encoding="utf-8",
    )

    config = load_settings_from_yaml(settings_yaml)

    assert config.shared.geography.local_crs == "EPSG:26910"
    assert config.impacts.emissions.inventory.enable_passenger_activity_correction is True
    assert config.impacts.emissions.inventory.enable_freight_activity_correction is True


def test_pipeline_profile_memray_relaunches_under_impacts_output(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("impacts:\n  local_output_folder: impacts_output\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "impacts.__main__.load_settings_from_yaml",
        lambda _: SimpleNamespace(impacts=SimpleNamespace(local_output_folder="impacts_output")),
    )
    monkeypatch.setattr("impacts.__main__.resolve_path", lambda path, _: str(tmp_path / path))

    def _fake_run(command, env=None, check=False, **_kwargs):
        captured["command"] = command
        captured["env"] = env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("impacts.__main__.subprocess.run", _fake_run)

    assert main(["pipeline", "--config", str(config_path), "--profile", "memray"]) == 0
    assert captured["command"] == [
        sys.executable,
        "-m",
        "memray",
        "run",
        "-o",
        str((tmp_path / "impacts_output" / "profiling" / "pipeline.memray.bin").resolve()),
        "-m",
        "impacts",
        "pipeline",
        "--config",
        str(config_path),
        "--profile",
        "none",
    ]
    assert captured["env"]["IMPACTS_PROFILE_ACTIVE"] == "1"


def test_pipeline_profile_time_relaunches_under_impacts_output(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("impacts:\n  local_output_folder: impacts_output\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "impacts.__main__.load_settings_from_yaml",
        lambda _: SimpleNamespace(impacts=SimpleNamespace(local_output_folder="impacts_output")),
    )
    monkeypatch.setattr("impacts.__main__.resolve_path", lambda path, _: str(tmp_path / path))

    def _fake_run(command, env=None, check=False, **_kwargs):
        captured["command"] = command
        captured["env"] = env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("impacts.__main__.subprocess.run", _fake_run)

    assert main(["pipeline", "--config", str(config_path), "--profile", "time"]) == 0
    assert captured["command"] == [
        "/usr/bin/time",
        "-l",
        "-o",
        str((tmp_path / "impacts_output" / "profiling" / "pipeline.time.txt").resolve()),
        sys.executable,
        "-m",
        "impacts",
        "pipeline",
        "--config",
        str(config_path),
        "--profile",
        "none",
    ]
    assert captured["env"]["IMPACTS_PROFILE_ACTIVE"] == "1"


def test_pipeline_profile_cpu_relaunches_with_cprofile_runner(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("impacts:\n  local_output_folder: impacts_output\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "impacts.__main__.load_settings_from_yaml",
        lambda _: SimpleNamespace(impacts=SimpleNamespace(local_output_folder="impacts_output")),
    )
    monkeypatch.setattr("impacts.__main__.resolve_path", lambda path, _: str(tmp_path / path))

    def _fake_run(command, env=None, check=False, **_kwargs):
        captured["command"] = command
        captured["env"] = env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("impacts.__main__.subprocess.run", _fake_run)

    assert main(["pipeline", "--config", str(config_path), "--profile", "cpu"]) == 0
    assert captured["command"] == [
        sys.executable,
        "-m",
        "impacts.profile_runner",
        "--pstats",
        str((tmp_path / "impacts_output" / "profiling" / "pipeline.cpu.pstats").resolve()),
        "--",
        "pipeline",
        "--config",
        str(config_path),
        "--profile",
        "none",
    ]
    assert captured["env"]["IMPACTS_PROFILE_ACTIVE"] == "1"


def test_pipeline_profile_all_relaunches_with_time_memray_and_cpu(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("impacts:\n  local_output_folder: impacts_output\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "impacts.__main__.load_settings_from_yaml",
        lambda _: SimpleNamespace(impacts=SimpleNamespace(local_output_folder="impacts_output")),
    )
    monkeypatch.setattr("impacts.__main__.resolve_path", lambda path, _: str(tmp_path / path))

    def _fake_run(command, env=None, check=False, **_kwargs):
        captured["command"] = command
        captured["env"] = env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("impacts.__main__.subprocess.run", _fake_run)

    assert main(["pipeline", "--config", str(config_path), "--profile", "all"]) == 0
    assert captured["command"] == [
        "/usr/bin/time",
        "-l",
        "-o",
        str((tmp_path / "impacts_output" / "profiling" / "pipeline.time.txt").resolve()),
        sys.executable,
        "-m",
        "memray",
        "run",
        "-o",
        str((tmp_path / "impacts_output" / "profiling" / "pipeline.memray.bin").resolve()),
        "-m",
        "impacts.profile_runner",
        "--pstats",
        str((tmp_path / "impacts_output" / "profiling" / "pipeline.cpu.pstats").resolve()),
        "--",
        "pipeline",
        "--config",
        str(config_path),
        "--profile",
        "none",
    ]
    assert captured["env"]["IMPACTS_PROFILE_ACTIVE"] == "1"


def test_postsim_profile_time_uses_timestamped_output_env(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("impacts:\n  local_output_folder: impacts_output\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.delenv("IMPACTS_POSTSIM_OUTPUT_DIR", raising=False)
    monkeypatch.setattr(
        "impacts.__main__.load_settings_from_yaml",
        lambda _: SimpleNamespace(
            run=SimpleNamespace(region="sfbay", scenario="base", output_run_name=None),
            impacts=SimpleNamespace(local_output_folder="impacts_output"),
        ),
    )
    monkeypatch.setattr("impacts.__main__.resolve_path", lambda path, _: str(tmp_path / path))

    def _fake_run(command, env=None, check=False, **_kwargs):
        captured["command"] = command
        captured["env"] = env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("impacts.__main__.subprocess.run", _fake_run)

    assert main(["postsim", "--config", str(config_path), "--profile", "time"]) == 0
    postsim_output_root = Path(captured["env"]["IMPACTS_POSTSIM_OUTPUT_DIR"])
    assert postsim_output_root.parent == (tmp_path / "impacts_output").resolve()
    assert re.fullmatch(r"impacts-postsim--sfbay--base--\d{8}-\d{6}", postsim_output_root.name)
    assert captured["command"] == [
        "/usr/bin/time",
        "-l",
        "-o",
        str(postsim_output_root / "profiling" / "postsim.time.txt"),
        sys.executable,
        "-m",
        "impacts",
        "postsim",
        "--config",
        str(config_path),
        "--profile",
        "none",
    ]


def test_settings_and_pipeline_use_vehicle_category_metadata_and_annualization_defaults(tmp_path: Path):
    days_csv = tmp_path / "vehicle_operation_days_per_year.csv"
    days_csv.write_text("vehicleCategory,operation_days_per_year\nLDA,347\n", encoding="utf-8")
    settings_yaml = tmp_path / "settings.yaml"
    settings_yaml.write_text(
        "\n".join(
            [
                "run:",
                "  region: sfbay",
                "  scenario: base",
                "  start_year: 2018",
                "shared:",
                "  geography:",
                "    FIPS:",
                '      state: "06"',
                "      counties:",
                '        - "001"',
                "    local_crs: EPSG:26910",
                "beam:",
                "  local_input_folder: pilates/beam/production",
                "  local_output_folder: beam/beam_output",
                "  router_directory: r5/network",
                "impacts:",
                "  local_input_folder: impacts/impacts_inputs",
                "  local_output_folder: impacts/impacts_output",
                "  scenario: 2018-Baseline",
                "  pipeline:",
                "    postsim:",
                "      emissions: true",
                "      inmap: false",
                "      aermod: false",
                "      exposure: false",
                "  population:",
                "    vehicle_folder: vehicle-tech",
                "    atlas_year: 2019",
                "    frism_year: 2018",
                "    population_sample: 0.1",
                "    transit_sample: 1.0",
                "    freight_sample: 0.009900990099009901",
                "  emissions:",
                "    include_passenger: false",
                "    include_freight: true",
                "    enable_passenger_activity_correction: true",
                "    enable_freight_activity_correction: false",
                "    default_annualization_days:",
                "      light_duty: 327.0",
                "      medium_heavy_duty: 312.0",
                "    pollutants: [NOx, PM25]",
                "  dispersions:",
                "    inmap: {}",
                "    aermod: {}",
            ]
        ),
        encoding="utf-8",
    )
    config = load_settings_from_yaml(settings_yaml)
    assert config.impacts.population.freight_sample == 0.009900990099009901
    assert config.impacts.emissions.defaults.default_annualization_days.light_duty == 327.0
    assert config.impacts.emissions.defaults.default_annualization_days.medium_heavy_duty == 312.0
    assert config.impacts.emissions.vehicle_category_metadata_file == "vehicle-tech/emissions/emissions_vehicle_categories.csv"
    assert config.impacts.emissions.rates_folder == "vehicle-tech/emissions/activities/2018-Baseline/rates"
    assert config.impacts.emissions.inventory.inventory_folder == "impacts/impacts_inputs/activities/2018-Baseline/inventory"
    assert config.impacts.emissions.inventory.passenger_file is None
    assert config.impacts.emissions.inventory.freight_file is None
    assert config.impacts.emissions.inventory.enable_passenger_activity_correction is True
    assert config.impacts.emissions.inventory.enable_freight_activity_correction is False
    assert config.impacts.emissions.beam.include_passenger is False
    assert config.impacts.emissions.beam.include_freight is True
    assert config.impacts.emissions.beam.passenger_vehicle_types_file is None
    assert config.impacts.emissions.beam.freight_vehicle_types_file is None

    payload = _pipeline_payload(tmp_path)
    payload["vehicle_category_metadata_file"] = str(days_csv)
    payload["annualization_days"] = {"light_duty": 327.0, "medium_heavy_duty": 312.0}
    payload["freight_sample"] = 0.009900990099009901
    pipeline = PipelineConfig.from_dict(payload)
    assert pipeline.vehicle_category_metadata_file == str(days_csv)
    assert pipeline.annualization_days == {"light_duty": 327.0, "medium_heavy_duty": 312.0}
    assert pipeline.freight_sample == 0.009900990099009901
    assert pipeline.include_passenger is True
    assert pipeline.include_freight is True


def test_build_settings_from_pilates_template_uses_builtin_impacts_settings(tmp_path: Path):
    pilates_settings = tmp_path / "pilates_settings.yaml"
    pilates_settings.write_text(
        "\n".join(
            [
                "run:",
                "  region: sfbay",
                "  scenario: base",
                "  start_year: 2017",
                "  output_run_name: calibration-a",
                "shared:",
                "  geography:",
                "    FIPS:",
                '      state: "06"',
                "      counties:",
                '        - "001"',
                '        - "013"',
                "    local_crs: EPSG:26910",
                "beam:",
                "  local_input_folder: pilates/beam/production",
                "  local_output_folder: beam/beam_output",
                "  router_directory: beam/beam_output/r5/sfbay-cbg5500-weakConn-network",
            ]
        ),
        encoding="utf-8",
    )

    config = build_settings_from_pilates(pilates_settings=pilates_settings)

    assert config.run.region == "sfbay"
    assert config.run.scenario == "base"
    assert config.run.start_year == 2017
    assert config.run.output_run_name == "calibration-a"
    assert config.shared.geography.local_crs == "EPSG:26910"
    assert config.beam.local_input_folder == "pilates/beam/production"
    assert config.beam.local_output_folder == "beam/beam_output"
    assert config.beam.router_directory == "beam/beam_output/r5/sfbay-cbg5500-weakConn-network"
    assert config.impacts.emissions.osm_network_folder.endswith("r5/sfbay-cbg5500-weakConn-network")
    assert config.impacts.scenario == "2017-Baseline"
    assert config.impacts.pipeline.inmap is True
    assert config.impacts.dispersions.inmap.isrm_zarr == "~/Workspace/Simulation/sfbay/inmap/isrm_v1.2.1.zarr"
    assert config.impacts.pipeline.aermod is True
    assert config.impacts.dispersions.aermod.grid_size_meters == 100.0
    assert config.impacts.population.passenger_folder == "urbansim/atlas-2019"
    assert config.impacts.emissions.beam.include_passenger is True
    assert config.impacts.emissions.beam.include_freight is True
    assert config.impacts.population.vehicle_folder == "vehicle-tech"
    assert config.impacts.emissions.beam.passenger_vehicle_types_file is None
    assert config.impacts.emissions.beam.freight_vehicle_types_file is None
    assert len(config.impacts.analysis.sector_targets) == 6
    assert (
        config.impacts.emissions.vehicle_category_metadata_file
        == "vehicle-tech/emissions/emissions_vehicle_categories.csv"
    )
    assert config.impacts.analysis.inventory_targets == []


def test_manifest_models_round_trip_current_shape(tmp_path: Path):
    pipeline = PipelineConfig.from_dict(_pipeline_payload(tmp_path)).to_dict()
    preprocess_manifest = PreprocessManifest.from_dict(_preprocess_manifest_payload(tmp_path)).to_dict()
    activities_manifest = ActivitiesManifest.from_dict(
        {
            "contract_version": "1",
            "model": "impacts",
            "settings_source": str(tmp_path / "settings.yaml"),
            "output_dir": str(tmp_path / "emfac"),
            "region_label": "SFBAY",
            "calendar_year": 2018,
            "scenario": "2018-Baseline",
            "vehicle_category_metadata_file": str(tmp_path / "vehicle_category_metadata.csv"),
            "outputs": {
                "outputs_root": str(tmp_path / "emfac"),
                "activities_output_root": str(tmp_path / "emfac" / "activities"),
                "tmp_root": str(tmp_path / "emfac" / "_tmp"),
                "emissions_store_root": str(tmp_path / "emfac" / "emissions" / "2018-2018-Baseline"),
                "passenger_rates_file": str(tmp_path / "emfac" / "activities" / "passenger-rates.parquet"),
                "passenger_activity_file": str(tmp_path / "emfac" / "_tmp" / "passenger-activity.parquet"),
                "passenger_fleet_file": str(tmp_path / "emfac" / "activities" / "passenger-fleet.parquet"),
                "freight_rates_file": str(tmp_path / "emfac" / "activities" / "freight-rates.parquet"),
                "freight_activity_file": str(tmp_path / "emfac" / "_tmp" / "freight-activity.parquet"),
                "freight_fleet_file": str(tmp_path / "emfac" / "activities" / "freight-fleet.parquet"),
                "final_activity_by_emfacid_output_passenger": str(
                    tmp_path / "emfac" / "activities" / "passenger-activity-by-emfacid.parquet"
                ),
                "final_activity_by_emfacid_output_freight": str(
                    tmp_path / "emfac" / "activities" / "freight-activity-by-emfacid.parquet"
                ),
            },
            "notes": [],
            "activities_manifest_path": str(tmp_path / "emfac" / "activities" / "activities_manifest.yaml"),
        }
    ).to_dict()
    run_manifest = PipelineManifest.from_dict(
        {
            "contract_version": "1",
            "model": "impacts",
            "preprocess_manifest_path": preprocess_manifest["preprocess_manifest_path"],
            "output_dir": str(tmp_path / "workspace"),
            "command": "python -m impacts emissions",
            "image": "unknown",
            "outputs": {"skims_emissions": str(tmp_path / "prepared.parquet")},
            "pipeline": pipeline,
            "population_inputs": {},
            "deterministic_contract": {},
            "execution": {"dispersion_completed": False, "stopped_after": "step1_process_emissions"},
            "pipeline_manifest_path": str(tmp_path / "workspace" / "pipeline_manifest.yaml"),
        }
    ).to_dict()
    postprocess_manifest = PostprocessManifest.from_dict(
        {
            "contract_version": "1",
            "model": "impacts",
            "pipeline_manifest_path": run_manifest["pipeline_manifest_path"],
            "output_dir": str(tmp_path / "impacts"),
            "postprocess_outputs": {},
            "validation": {},
            "notes": [],
            "postprocess_manifest_path": str(tmp_path / "impacts" / "postprocess_manifest.yaml"),
        }
    ).to_dict()

    assert activities_manifest["outputs"]["outputs_root"].endswith("emfac")
    assert preprocess_manifest["pipeline"]["region"] == "sfbay"
    assert run_manifest["execution"]["stopped_after"] == "step1_process_emissions"
    assert postprocess_manifest["postprocess_outputs"] == {}


def test_analysis_accepts_delta_baseline_concentration_distribution_file(tmp_path: Path):
    payload = yaml.safe_load(Path("src/impacts/config/settings.yaml").read_text(encoding="utf-8"))
    baseline_path = tmp_path / "baseline" / "beam_concentration_distribution.parquet"
    payload["impacts"]["analysis"]["delta_baseline_concentration_distribution_file"] = str(baseline_path)
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_settings_from_yaml(settings_path)

    assert config.impacts.analysis.delta_baseline_concentration_distribution_file == str(baseline_path)
    assert (
        config.to_dict()["impacts"]["analysis"]["delta_baseline_concentration_distribution_file"]
        == str(baseline_path)
    )


def test_pipeline_manifest_allows_disabled_inmap_without_inmap_inputs(tmp_path: Path):
    payload = _pipeline_payload(tmp_path)
    payload["inmap_enabled"] = False
    payload["inmap_grid_path"] = None
    payload["inmap_grid_epsg"] = None
    payload["isrm_url"] = None
    payload["isrm_nox_to_no2_ratios_file"] = None

    config = PipelineConfig.from_dict(payload)

    assert config.inmap_enabled is False
    assert config.inmap_grid_path is None
    assert config.isrm_url is None


def test_pipeline_manifest_allows_disabled_aermod_without_aermod_inputs(tmp_path: Path):
    payload = _pipeline_payload(tmp_path)
    payload["aermod_enabled"] = False
    payload["grid_size_meters"] = None
    payload["aermod_full_grid_path"] = None
    payload["aermod_grid_path"] = None
    payload["aermod_grid_epsg"] = None
    payload["aermod_grid_id"] = None
    payload["asrv_patterns_file"] = None
    payload["asrv_patterns_epsg"] = None

    config = PipelineConfig.from_dict(payload)

    assert config.aermod_enabled is False
    assert config.aermod_grid_path is None
    assert config.asrv_patterns_file is None


def test_pipeline_manifest_rejects_unknown_primary_pm25_integration_strategy(tmp_path: Path):
    payload = _pipeline_payload(tmp_path)
    payload["primary_pm25_integration_strategy"] = "unknown"

    with pytest.raises(ValueError, match="primary_pm25_integration_strategy"):
        PipelineConfig.from_dict(payload)


def test_run_emissions_from_pipeline_manifest_uses_staged_intersections_from_preprocess(monkeypatch, tmp_path: Path):
    import impacts.pipeline.workflow.prepare_emissions.from_skims as from_skims_module
    import impacts.pipeline.workflow.step1_process_emissions as step1_process_emissions
    import impacts.runner as runner_module

    for name in (
        "beam_osm_county_intersection.parquet",
        "beam_osm_inmap_intersection.parquet",
        "beam_osm_aermod_intersection.parquet",
    ):
        (tmp_path / name).write_text("", encoding="utf-8")

    run_manifest_path = tmp_path / "workspace" / "pipeline_manifest.yaml"

    def _fake_load_structured_file(path):
        path = Path(path)
        if path == run_manifest_path:
            return {
                "contract_version": "1",
                "model": "impacts",
                "preprocess_manifest_path": str(tmp_path / "workspace" / "preprocess_manifest.yaml"),
                "output_dir": str(tmp_path / "impacts_output"),
                "command": "python -m impacts emissions",
                "image": "unknown",
                "outputs": {"skims_emissions": str(tmp_path / "prepared_skims.parquet")},
                "pipeline": _pipeline_payload(tmp_path),
                "population_inputs": {},
                "deterministic_contract": {},
                "execution": {"dispersion_completed": False, "stopped_after": "preprocess"},
                "pipeline_manifest_path": str(run_manifest_path),
            }
        return _preprocess_manifest_payload(tmp_path)

    monkeypatch.setattr(runner_module, "load_structured_file", _fake_load_structured_file)
    monkeypatch.setattr(
        runner_module,
        "load_settings_from_yaml",
        lambda _: SimpleNamespace(
            impacts=SimpleNamespace(local_output_folder=str(tmp_path / "impacts_output"))
        ),
    )
    captured = {}
    monkeypatch.setattr(
        step1_process_emissions,
        "run",
        lambda pipeline, raw_dir, input_root, grid_intersection_paths, manifest_inputs=None: captured.update(
            {
                "grid_intersection_paths": grid_intersection_paths,
            }
        ) or {
            "beam_emissions_by_county_process": str(tmp_path / "beam_emissions_by_county_process.parquet"),
            "beam_emissions_for_inmap": str(tmp_path / "beam_emissions_for_inmap.parquet"),
        },
    )
    monkeypatch.setattr(
        from_skims_module,
        "resolve_prepared_skims_path",
        lambda input_root: str(tmp_path / "prepared_skims.parquet"),
    )

    result = run_emissions_from_pipeline_manifest(
        run_manifest_path=run_manifest_path,
    )

    assert result["execution"]["dispersion_completed"] is False
    assert result["execution"]["stopped_after"] == "step1_process_emissions"
    assert "step1_process_emissions" in result["execution"]["stage_timings_seconds"]
    assert captured["grid_intersection_paths"]["county"].endswith("beam_osm_county_intersection.parquet")
    assert captured["grid_intersection_paths"]["inmap"].endswith("beam_osm_inmap_intersection.parquet")
    assert captured["grid_intersection_paths"]["aermod"].endswith("beam_osm_aermod_intersection.parquet")


def test_build_preprocess_manifest_runs_step3_and_registers_intersections(monkeypatch, tmp_path: Path):
    import sys
    from types import ModuleType

    import impacts.pipeline.preprocessing.step1_collect_inputs as step1_collect_inputs
    import impacts.pipeline.preprocessing.step2_prepare_grids as step2_prepare_grids
    import impacts.preprocessor as preprocessor_module

    config_path = tmp_path / "settings.yaml"
    config_path.write_text("impacts: {}\n", encoding="utf-8")
    output_root = tmp_path / "impacts_output"
    input_root = output_root / "preprocess"
    calls = {}

    settings = SimpleNamespace(
        run=SimpleNamespace(region="sfbay", start_year=2018),
        shared=SimpleNamespace(
            geography=SimpleNamespace(
                local_crs="EPSG:26910",
                fips=SimpleNamespace(state="06", counties=["001", "013"]),
            )
        ),
        impacts=SimpleNamespace(
            local_output_folder="impacts_output",
            pipeline=SimpleNamespace(
                emissions=True,
                inmap=True,
                aermod=True,
                exposure=True,
            ),
            population=SimpleNamespace(population_sample=0.1, transit_sample=0.25),
            emissions=SimpleNamespace(
                mapping_columns={"link_id": "linkId"},
                beam_osm_id_col="attributeOrigId",
                beam_length_col="linkLength",
                prepared_skims_group_cols=["linkId", "vehicleTypeId", "process"],
                pollutants=["NOx"],
                source_pollutants=["NOx"],
                vehicle_category_metadata_file=str(tmp_path / "vehicle_category_metadata.csv"),
                inventory=SimpleNamespace(
                    enable_passenger_activity_correction=True,
                    enable_freight_activity_correction=True,
                ),
                beam=SimpleNamespace(
                    include_non_osm_car_links=True,
                    include_passenger=True,
                    include_freight=True,
                ),
                defaults=SimpleNamespace(
                    default_annualization_days=SimpleNamespace(light_duty=327.0, medium_heavy_duty=312.0)
                ),
            ),
            dispersions=SimpleNamespace(
                inmap=SimpleNamespace(
                    grid_epsg=26910,
                    isrm_nox_to_no2_ratios_apply_tons_per_year_to_ug_per_s=False,
                ),
                aermod=SimpleNamespace(
                    asrv_patterns_epsg=4326,
                    grid_size_meters=100.0,
                ),
            ),
            exposure=SimpleNamespace(primary_pm25_integration_strategy="impute_inmap_primary_in_aermod_domain"),
        ),
    )

    monkeypatch.setattr(preprocessor_module, "load_settings_from_yaml", lambda _: settings)
    monkeypatch.setattr(preprocessor_module, "resolve_path", lambda path, _: str(tmp_path / path))
    monkeypatch.setattr(preprocessor_module, "parse_epsg", lambda _: 26910)
    monkeypatch.setattr(preprocessor_module, "infer_vector_epsg", lambda _: 26910)

    def _fake_step1(*, manifest_inputs, **kwargs):
        for key, filename in (
            ("passenger_vehicle_types_input", "passenger_vehicle_types.csv"),
            ("freight_vehicle_types_input", "freight_vehicle_types.csv"),
            ("vehicle_category_metadata_file_input", "vehicle_category_metadata.csv"),
            ("county_boundaries", "county_boundaries.parquet"),
            ("network", "network.csv.gz"),
            ("osm_network", "osm_network.parquet"),
        ):
            manifest_inputs[key] = {"source_path": str(tmp_path / filename)}
        return {
            "staged_passenger_inventory_file": str(tmp_path / "passenger_inventory.parquet"),
            "staged_freight_inventory_file": str(tmp_path / "freight_inventory.parquet"),
            "staged_isrm": str(tmp_path / "isrm.zarr"),
            "staged_isrm_nox_to_no2_ratios_file": str(tmp_path / "matrix.npz"),
            "staged_asrv_patterns_file": str(tmp_path / "asrv_patterns.parquet"),
            "staged_inmap_grid": str(tmp_path / "inmap_grid.parquet"),
            "population_inputs": {},
        }

    def _fake_step2(**kwargs):
        import geopandas as gpd
        from shapely.geometry import box

        aermod_grid = gpd.GeoDataFrame(
            {"aermod_id": [1]},
            geometry=[box(0.0, 0.0, 100.0, 100.0)],
            crs="EPSG:26910",
        )
        aermod_grid.to_parquet(tmp_path / "aermod_grid.parquet", index=False)
        return {
            "staged_inmap_grid": str(tmp_path / "inmap_grid.parquet"),
            "staged_aermod_grid": str(tmp_path / "aermod_grid.parquet"),
            "staged_aermod_full_grid": str(tmp_path / "aermod_full_grid.parquet"),
            "resolved_inmap_grid_id": "isrm",
            "resolved_aermod_grid_id": "aermod_id",
        }

    def _fake_step3(pipeline, staged_input_root, manifest_inputs=None):
        calls["step3"] = {
            "input_root": staged_input_root,
            "manifest_inputs": manifest_inputs,
            "pipeline": pipeline,
        }
        return (
            {
                "county": str(output_root / "beam_osm_county_intersection.parquet"),
                "inmap": str(output_root / "beam_osm_inmap_intersection.parquet"),
                "aermod": str(output_root / "beam_osm_aermod_intersection.parquet"),
            },
            {"county": None, "inmap": None, "aermod": None},
        )

    monkeypatch.setattr(step1_collect_inputs, "run", _fake_step1)
    monkeypatch.setattr(step2_prepare_grids, "run", _fake_step2)
    fake_step3_module = ModuleType("impacts.pipeline.preprocessing.step3_integrate_grids")
    fake_step3_module.run = _fake_step3
    monkeypatch.setitem(sys.modules, "impacts.pipeline.preprocessing.step3_integrate_grids", fake_step3_module)

    manifest = preprocessor_module.build_preprocess_manifest(config_path)

    assert calls["step3"]["input_root"] == input_root
    assert manifest["maintained_execution_path"] == [
        "impacts.pipeline.workflow.step1_process_emissions",
        "impacts.pipeline.workflow.step2_compute_inmap_concentrations",
        "impacts.pipeline.workflow.step3_compute_aermod_concentrations",
    ]
    assert manifest["inputs"]["county_intersection"]["source_path"].endswith("beam_osm_county_intersection.parquet")
    assert manifest["inputs"]["inmap_intersection"]["source_path"].endswith("beam_osm_inmap_intersection.parquet")
    assert manifest["inputs"]["aermod_intersection"]["source_path"].endswith("beam_osm_aermod_intersection.parquet")
    assert manifest["pipeline"]["isrm_nox_to_no2_ratios_apply_tons_per_year_to_ug_per_s"] is False


def test_cli_rejects_removed_run_command() -> None:
    with pytest.raises(SystemExit):
        main(["run"])


def test_cli_fleet_uses_activities_manifest(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def _fake_run_fleet(*, activities_manifest_path):
        captured["activities_manifest_path"] = str(activities_manifest_path)

    monkeypatch.setattr("impacts.provisioner.run_fleet", _fake_run_fleet)

    assert main(["fleet", "--activities-manifest", str(tmp_path / "activities_manifest.yaml")]) == 0
    assert captured["activities_manifest_path"].endswith("activities_manifest.yaml")


def test_postprocess_from_settings_delegates_through_runner(monkeypatch, tmp_path: Path):
    calls = {}

    class _Settings:
        class impacts:
            local_output_folder = "impacts"
            pipeline = SimpleNamespace(
                postsim=SimpleNamespace(
                    emissions=True,
                    inmap=False,
                    aermod=False,
                    exposure=False,
                )
            )

    def _fake_preprocess(settings_path):
        calls["preprocess"] = {
            "settings_path": str(settings_path),
        }
        return {"pipeline_manifest_path": str(tmp_path / "workspace" / "pipeline_manifest.yaml")}

    def _fake_emissions_run(run_manifest_path):
        calls["emissions_run"] = {
            "run_manifest_path": str(run_manifest_path),
        }
        return {"pipeline_manifest_path": str(tmp_path / "workspace" / "pipeline_manifest.yaml")}

    def _fake_postprocess(
        run_manifest_path,
        manifest_path=None,
        input_roots=None,
        baseline_concentration_override=None,
    ):
        calls["postprocess"] = {
            "run_manifest_path": str(run_manifest_path),
            "manifest_path": manifest_path,
            "input_roots": tuple(input_roots or ()),
            "baseline_concentration_override": baseline_concentration_override,
        }
        return {"postprocess_manifest_path": str(tmp_path / "impacts" / "postprocess_manifest.yaml")}

    monkeypatch.setattr("impacts.postprocessor.load_settings_from_yaml", lambda _: _Settings())
    monkeypatch.setattr("impacts.preprocessor.preprocess_workflow", _fake_preprocess)
    monkeypatch.setattr("impacts.runner.run_emissions_from_pipeline_manifest", _fake_emissions_run)
    monkeypatch.setattr("impacts.postprocessor.postprocess_from_pipeline_manifest", _fake_postprocess)

    result = postprocess_from_settings(
        settings_path=tmp_path / "settings.yaml",
    )

    assert result["postprocess_manifest_path"].endswith("postprocess_manifest.yaml")
    assert calls["preprocess"]["settings_path"].endswith("settings.yaml")
    assert calls["emissions_run"]["run_manifest_path"].endswith("pipeline_manifest.yaml")
    assert calls["postprocess"]["run_manifest_path"].endswith("pipeline_manifest.yaml")


def test_postprocess_logs_pipeline_complete_after_steps_and_manifest(monkeypatch, caplog, tmp_path: Path):
    output_root = tmp_path / "impacts_output"
    run_manifest_path = output_root / "pipeline_manifest.yaml"
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    preprocess_manifest_path = output_root / "preprocess_manifest.yaml"
    run_manifest_path.write_text(
        json.dumps(
            {
                "contract_version": "1",
                "model": "impacts",
                "preprocess_manifest_path": str(preprocess_manifest_path),
                "output_dir": str(output_root),
                "command": "python -m impacts postsim",
                "image": "unknown",
                "outputs": {"skims_emissions": str(tmp_path / "prepared.parquet")},
                "pipeline": _pipeline_payload(tmp_path),
                "population_inputs": {},
                "deterministic_contract": {},
                "execution": {"dispersion_completed": True, "stopped_after": "step4_prepare_exposure"},
                "pipeline_manifest_path": str(run_manifest_path),
            }
        ),
        encoding="utf-8",
    )
    preprocess_manifest_path.write_text(
        json.dumps(
            {
                "contract_version": "1",
                "model": "impacts",
                "settings_source": str(tmp_path / "settings.yaml"),
                "staging_dir": str(output_root / "preprocess"),
                "input_dir": str(output_root / "preprocess"),
                "preprocess_manifest_path": str(preprocess_manifest_path),
                "maintained_execution_path": [],
                "inputs": {},
                "pipeline": _pipeline_payload(tmp_path),
                "pilates_contract": {"stage": "terminal_postprocessing"},
                "population_inputs": {},
                "notes": [],
            }
        ),
        encoding="utf-8",
    )

    def _fake_postprocess_steps(settings_path, **_kwargs):
        logging.getLogger("impacts.postprocess.fake").info("postprocess steps complete for %s", settings_path)
        return {"postprocess_output": str(tmp_path / "postprocess.parquet")}

    monkeypatch.setattr("impacts.postprocessor._run_postprocess_steps", _fake_postprocess_steps)

    with caplog.at_level(logging.INFO):
        result = postprocess_from_pipeline_manifest(run_manifest_path=run_manifest_path)

    messages = [record.getMessage() for record in caplog.records]
    steps_index = messages.index(f"postprocess steps complete for {tmp_path / 'settings.yaml'}")
    manifest_index = messages.index(f"Postprocess manifest written: {output_root / 'postprocess_manifest.yaml'}")
    complete_index = messages.index(f"Pipeline complete: postprocess_manifest={output_root / 'postprocess_manifest.yaml'}")

    assert result["postprocess_manifest_path"].endswith("postprocess_manifest.yaml")
    assert steps_index < manifest_index < complete_index
    assert messages[-1] == f"Pipeline complete: postprocess_manifest={output_root / 'postprocess_manifest.yaml'}"


def test_postprocess_impact_output_dir_cli_passes_output_root_override_and_impact_input_dir(monkeypatch, tmp_path: Path):
    output_root = tmp_path / "copied_output"
    input_root = tmp_path / "beam-data" / "sfbay"
    output_root.mkdir()
    input_root.mkdir(parents=True)
    (output_root / "pipeline_manifest.yaml").write_text("{}", encoding="utf-8")
    calls = {}

    def _fake_postprocess(
        run_manifest_path,
        manifest_path=None,
        output_root_override=None,
        input_roots=None,
        baseline_concentration_override=None,
    ):
        calls["run_manifest_path"] = str(run_manifest_path)
        calls["manifest_path"] = manifest_path
        calls["output_root_override"] = str(output_root_override)
        calls["input_roots"] = tuple(input_roots or ())
        calls["baseline_concentration_override"] = baseline_concentration_override
        return {"postprocess_manifest_path": str(output_root / "postprocess_manifest.yaml")}

    monkeypatch.setattr("impacts.postprocessor.postprocess_from_pipeline_manifest", _fake_postprocess)

    assert main([
        "postprocess",
        "--impact-output-dir",
        str(output_root),
        "--impact-input-dir",
        str(input_root),
    ]) == 0

    assert calls["run_manifest_path"] == str(output_root / "pipeline_manifest.yaml")
    assert calls["manifest_path"] is None
    assert calls["output_root_override"] == str(output_root)
    assert calls["input_roots"] == (str(input_root),)


def test_top_level_postprocess_help_describes_current_workflow(capsys):
    stale_help = "canonical impacts " + "exposure table " + "artifact"

    with pytest.raises(SystemExit):
        main(["--help"])

    help_text = capsys.readouterr().out
    assert "Run postprocess comparisons and map plots" in help_text
    assert stale_help not in help_text


def test_postprocess_cli_rejects_legacy_output_dir_and_input_root_flags(tmp_path: Path):
    output_root = tmp_path / "copied_output"
    input_root = tmp_path / "beam-data" / "sfbay"
    legacy_output_flag = "--" + "output-dir"
    legacy_input_flag = "--" + "input-root"
    output_root.mkdir()
    input_root.mkdir(parents=True)
    (output_root / "pipeline_manifest.yaml").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        main([
            "postprocess",
            legacy_output_flag,
            str(output_root),
            legacy_input_flag,
            str(input_root),
        ])


def test_build_nox_to_no2_top_level_cli_uses_current_tool_contract(capsys):
    with pytest.raises(SystemExit):
        main(["build_nox_to_no2", "--help"])

    help_text = capsys.readouterr().out
    assert "--output-file" in help_text
    assert "--cmaq-ratio-table" in help_text
    assert "--regional-matrix" in help_text
    assert "--output-name" not in help_text
    assert "--input-dir" not in help_text
    assert "--" + "output-dir" not in help_text


def test_postprocess_output_root_override_localizes_stale_manifest_paths(
    monkeypatch,
    tmp_path: Path,
):
    output_root = tmp_path / "copied_output"
    output_root.mkdir()
    stale_root = Path("/global/scratch/users/hmlaarabi/sources/IMPACTS/examples/pilates/impacts/impacts_output")
    run_manifest_path = output_root / "pipeline_manifest.yaml"
    preprocess_manifest_path = output_root / "preprocess_manifest.yaml"
    local_settings_path = output_root / "settings.yaml"
    skims_emissions_path = output_root / "emissions" / "prepared_skims_for_grid_allocation.parquet"
    skims_emissions_path.parent.mkdir(parents=True)
    skims_emissions_path.write_text("", encoding="utf-8")
    local_settings_path.write_text("impacts:\n  local_output_folder: impacts/impacts_output\n", encoding="utf-8")
    run_manifest_path.write_text(
        json.dumps(
            {
                "contract_version": "1",
                "model": "impacts",
                "preprocess_manifest_path": str(stale_root / "preprocess_manifest.yaml"),
                "output_dir": str(stale_root),
                "command": "python -m impacts postsim",
                "image": "unknown",
                "outputs": {
                    "skims_emissions": str(stale_root / "emissions" / "prepared_skims_for_grid_allocation.parquet"),
                },
                "pipeline": _pipeline_payload(tmp_path),
                "population_inputs": {},
                "deterministic_contract": {},
                "execution": {"dispersion_completed": True, "stopped_after": "step4_prepare_exposure"},
                "pipeline_manifest_path": str(run_manifest_path),
            }
        ),
        encoding="utf-8",
    )
    preprocess_manifest_path.write_text(
        json.dumps(
            {
                "contract_version": "1",
                "model": "impacts",
                "settings_source": str(stale_root.parent.parent / "hpc-settings.yaml"),
                "staging_dir": str(stale_root / "preprocess"),
                "input_dir": str(stale_root / "preprocess"),
                "preprocess_manifest_path": str(stale_root / "preprocess_manifest.yaml"),
                "maintained_execution_path": [],
                "inputs": {
                    "settings": {
                        "kind": "local",
                        "source_path": str(stale_root / "settings.yaml"),
                        "staged_path": str(stale_root / "settings.yaml"),
                        "optional": False,
                        "exists": True,
                    }
                },
                "pipeline": _pipeline_payload(tmp_path),
                "pilates_contract": {"stage": "terminal_postprocessing"},
                "population_inputs": {},
                "notes": [],
            }
        ),
        encoding="utf-8",
    )
    calls = {}

    def _fake_postprocess_steps(
        settings_path,
        *,
        run_manifest_path=None,
        output_root=None,
        allow_missing_source_inputs=False,
        input_roots=None,
        baseline_concentration_override=None,
    ):
        calls["settings_path"] = str(settings_path)
        calls["run_manifest_path"] = str(run_manifest_path)
        calls["output_root"] = str(output_root)
        calls["allow_missing_source_inputs"] = allow_missing_source_inputs
        calls["input_roots"] = tuple(input_roots or ())
        calls["baseline_concentration_override"] = baseline_concentration_override
        return {"postprocess_output": str(output_root / "postprocess" / "dummy.parquet")}

    monkeypatch.setattr("impacts.postprocessor._run_postprocess_steps", _fake_postprocess_steps)

    result = postprocess_from_pipeline_manifest(
        run_manifest_path=run_manifest_path,
        output_root_override=output_root,
    )

    assert result["output_dir"] == str(output_root.resolve())
    assert result["postprocess_manifest_path"] == str(output_root / "postprocess_manifest.yaml")
    assert calls["settings_path"] == str(local_settings_path.resolve())
    assert calls["run_manifest_path"] == str(run_manifest_path)
    assert calls["output_root"] == str(output_root.resolve())
    assert calls["allow_missing_source_inputs"] is True
    assert calls["input_roots"] == ()


def test_postprocess_input_root_localizes_manifest_source_paths(monkeypatch, tmp_path: Path):
    import impacts.postprocessor as postprocessor_module

    output_root = tmp_path / "copied_output"
    input_root = tmp_path / "beam-data" / "sfbay"
    stale_input_root = Path("/global/scratch/users/hmlaarabi/sources/PILATES/pilates/beam/production/sfbay")
    passenger_vt = input_root / "vehicle-tech" / "vehicleTypes--atlas--2019-Baseline--EM.csv"
    freight_vt = input_root / "vehicle-tech" / "vehicleTypes--frism--2018-Baseline--EM.csv"
    metadata = input_root / "vehicle-tech" / "emissions" / "emissions_vehicle_categories.csv"
    for path in (passenger_vt, freight_vt, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    preprocess_manifest = {
        "inputs": {
            "passenger_vehicle_types_input": {
                "staged_path": str(stale_input_root / "vehicle-tech" / passenger_vt.name),
            },
            "freight_vehicle_types_input": {
                "staged_path": str(stale_input_root / "vehicle-tech" / freight_vt.name),
            },
            "vehicle_category_metadata_file_input": {
                "staged_path": str(stale_input_root / "vehicle-tech" / "emissions" / metadata.name),
            },
        }
    }

    monkeypatch.setattr(
        postprocessor_module,
        "_load_context",
        lambda *_args, **_kwargs: (tmp_path / "pipeline_manifest.yaml", {}, preprocess_manifest, preprocess_manifest["inputs"]),
    )
    registry = PathRegistry([input_root])

    passenger_resolved, freight_resolved = postprocessor_module._resolve_vehicle_types_paths(
        tmp_path / "settings.yaml",
        output_root=output_root,
        registry=registry,
    )
    metadata_resolved = postprocessor_module._resolve_vehicle_category_metadata_path(
        tmp_path / "settings.yaml",
        output_root=output_root,
        registry=registry,
    )

    assert passenger_resolved == passenger_vt.resolve()
    assert freight_resolved == freight_vt.resolve()
    assert metadata_resolved == metadata.resolve()


def test_postprocess_delta_baseline_path_can_be_relative_to_impact_input_dir(tmp_path: Path):
    import impacts.postprocessor as postprocessor_module

    input_root = tmp_path / "beam-data" / "sfbay"
    baseline_path = input_root / "baseline" / "beam_concentration_distribution.parquet"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text("", encoding="utf-8")

    resolved = postprocessor_module._resolve_delta_baseline_concentration_path(
        "baseline/beam_concentration_distribution.parquet",
        registry=PathRegistry([input_root]),
    )

    assert resolved == baseline_path.resolve()


def test_postprocess_steps_write_named_dirs_not_analysis_subdir(monkeypatch, tmp_path: Path):
    import impacts.postprocessor as postprocessor_module
    import impacts.pipeline.postprocess.step1_compare_fleet as step1_module
    import impacts.pipeline.postprocess.step2_compare_annual_targets as step2_module
    import impacts.pipeline.postprocess.step3_compare_emissions_inventory as step3_module
    import impacts.pipeline.postprocess.step4_plot_concentrations as step4_module
    import impacts.pipeline.postprocess.step5_plot_exposure as step5_module
    import impacts.pipeline.postprocess.step6_plot_delta_concentrations as step6_module
    import impacts.pipeline.postprocess.step7_plot_delta_exposure as step7_module

    output_root = tmp_path / "impacts_output"
    output_root.mkdir()
    (output_root / "exposure").mkdir()
    (output_root / "preprocess").mkdir()
    (output_root / "concentrations").mkdir()
    (output_root / "exposure" / "beam_concentration_distribution.parquet").touch()
    (output_root / "exposure" / "beam_population_counts.parquet").touch()
    (output_root / "preprocess" / "beam_osm_mapped.parquet").touch()
    (output_root / "concentrations" / "beam_inmap_concentrations.parquet").touch()
    baseline_path = output_root / "baseline_concentration.parquet"
    baseline_path.touch()
    output_dirs: dict[str, Path] = {}
    step4_calls: dict[str, bool] = {}
    step6_calls: dict[str, str] = {}
    step7_calls: dict[str, str] = {}

    class _Settings:
        class impacts:
            local_output_folder = str(output_root)

            class analysis:
                delta_baseline_concentration_distribution_file = str(baseline_path)
                sector_targets = [
                    SimpleNamespace(
                        source="mobile_onroad",
                        sector="all",
                        annual_pm25_short_tons=1.0,
                        annual_nox_short_tons=None,
                        annual_pm10_short_tons=None,
                        annual_tog_short_tons=None,
                        annual_rog_short_tons=None,
                        annual_co_short_tons=None,
                        annual_sox_short_tons=None,
                    )
                ]
                inventory_targets = [
                    SimpleNamespace(
                        name="inventory",
                        pollutants={
                            "PM25": SimpleNamespace(
                                columns=("PM25",),
                            )
                        },
                    )
                ]

        class shared:
            class geography:
                class fips:
                    counties = []

    def _fake_step(name: str):
        def _run(*, output_dir: Path, **_kwargs):
            output_dirs[name] = Path(output_dir)
            return {f"{name}_output": str(Path(output_dir) / f"{name}.png")}

        return _run

    def _fake_step4(*, output_dir: Path, **_kwargs):
        output_dirs["step4"] = Path(output_dir)
        step4_calls["called"] = True
        return {"step4_output": str(Path(output_dir) / "concentration.png")}

    def _fake_step6(*, output_dir: Path, delta_baseline_concentration_path: str, **_kwargs):
        output_dirs["step6"] = Path(output_dir)
        step6_calls["delta_baseline_concentration_path"] = delta_baseline_concentration_path
        delta_table = Path(output_dir) / "concentration_delta.parquet"
        return {"delta_table": str(delta_table), "step6_output": str(Path(output_dir) / "delta.png")}

    def _fake_step7(*, output_dir: Path, concentration_delta_path: str, **_kwargs):
        output_dirs["step7"] = Path(output_dir)
        step7_calls["concentration_delta_path"] = concentration_delta_path
        return {"step7_output": str(Path(output_dir) / "delta_exposure.png")}

    monkeypatch.setattr(postprocessor_module, "load_settings_from_yaml", lambda _: _Settings())
    monkeypatch.setattr(postprocessor_module, "_resolve_modeled_emissions_path", lambda *_args, **_kwargs: output_root / "modeled.parquet")
    monkeypatch.setattr(postprocessor_module, "_resolve_skims_emissions_path", lambda *_args, **_kwargs: output_root / "skims.parquet")
    monkeypatch.setattr(postprocessor_module, "_resolve_vehicle_types_paths", lambda *_args, **_kwargs: (output_root / "passenger.csv", output_root / "freight.csv"))
    monkeypatch.setattr(postprocessor_module, "_resolve_optional_population_assignment_paths", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(postprocessor_module, "_resolve_inventory_emfacid_activity_path", lambda *_args, **_kwargs: output_root / "activity.parquet")
    monkeypatch.setattr(postprocessor_module, "_resolve_vehicle_category_metadata_path", lambda *_args, **_kwargs: output_root / "metadata.csv")
    monkeypatch.setattr(postprocessor_module, "_resolve_county_boundaries_path", lambda *_args, **_kwargs: output_root / "county.gpkg")
    monkeypatch.setattr(postprocessor_module, "_resolve_emissions_inventory_path", lambda *_args, **_kwargs: output_root / "inventory.parquet")
    monkeypatch.setattr(step1_module, "run", _fake_step("step1"))
    monkeypatch.setattr(step2_module, "run", _fake_step("step2"))
    monkeypatch.setattr(step3_module, "run", _fake_step("step3"))
    monkeypatch.setattr(step4_module, "run", _fake_step4)
    monkeypatch.setattr(step5_module, "run", _fake_step("step5"))
    monkeypatch.setattr(step6_module, "run", _fake_step6)
    monkeypatch.setattr(step7_module, "run", _fake_step7)

    outputs = postprocessor_module._run_postprocess_steps(
        tmp_path / "settings.yaml",
        output_root=output_root,
    )

    assert output_dirs == {
        "step1": output_root / "postprocess" / "fleet",
        "step2": output_root / "postprocess" / "annual_targets",
        "step3": output_root / "postprocess" / "emissions_inventory",
        "step4": output_root / "postprocess" / "concentrations",
        "step5": output_root / "postprocess" / "exposure",
        "step6": output_root / "postprocess" / "delta_concentrations",
        "step7": output_root / "postprocess" / "delta_exposure",
    }
    assert step4_calls["called"] is True
    assert step6_calls["delta_baseline_concentration_path"] == str(baseline_path.resolve())
    assert step7_calls["concentration_delta_path"] == str(
        output_root / "postprocess" / "delta_concentrations" / "concentration_delta.parquet"
    )
    analysis_subdir = Path("postprocess") / "analysis"
    assert all(str(analysis_subdir) not in str(path) for path in output_dirs.values())
    assert set(outputs) == {
        "fleet_step1_output",
        "annual_targets_step2_output",
        "inventory_step3_output",
        "concentration_step4_output",
        "exposure_step5_output",
        "delta_concentration_delta_table",
        "delta_concentration_step6_output",
        "delta_exposure_step7_output",
    }


def test_hpc_job_prints_successful_stage_completion():
    job_script = Path("hpc/job.sh").read_text(encoding="utf-8")

    assert "Pipeline job complete: stage=$STAGE" in job_script
    assert "Stage job complete: stage=$STAGE" in job_script


def test_hpc_scripts_default_impacts_dir_to_checkout_root():
    for script_path in (Path("hpc/job.sh"), Path("hpc/job_runner.sh")):
        script = script_path.read_text(encoding="utf-8")
        assert "/global/scratch/users/$USER/sources/impacts" not in script
        assert 'REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"' in script
        assert 'IMPACTS_DIR="${IMPACTS_DIR:-$REPO_ROOT}"' in script


def test_hpc_scripts_do_not_advertise_removed_analysis_stage():
    for script_path in (Path("hpc/job.sh"), Path("hpc/job_runner.sh")):
        script = Path(script_path).read_text(encoding="utf-8")
        assert "python3\" -u -m impacts analysis" not in script
        assert "analysis)" not in script
        assert ", analysis" not in script


def test_hpc_job_runner_accepts_manifest_inputs_for_stage_runs():
    job_runner = Path("hpc/job_runner.sh").read_text(encoding="utf-8")

    assert "-c <settings-or-manifest>" in job_runner
    assert "activities_manifest.yaml for fleet" in job_runner
    assert "pipeline_manifest.yaml for" in job_runner
    assert "_stage_uses_run_manifest" in job_runner
    assert "fleet|emissions|inmap|aermod|exposure|postprocess" in job_runner
    assert 'MANIFEST_OUTPUT_DIR="$(_top_level_yaml_value output_dir)"' in job_runner
    assert "requires a manifest with top-level output_dir" in job_runner
    assert "if ! _stage_uses_run_manifest; then" in job_runner


def test_hpc_profile_is_controlled_by_profile_mode():
    job_runner = Path("hpc/job_runner.sh").read_text(encoding="utf-8")
    job_script = Path("hpc/job.sh").read_text(encoding="utf-8")

    assert 'profile_arg="${IMPACTS_PROFILE:-none}"' in job_runner
    assert "--profile      Profiling mode: none (default), time, cpu, memray, all" in job_runner
    assert "none|time|cpu|memray|all" in job_runner
    assert "IMPACTS_PROFILE=$profile_arg" in job_runner
    assert 'PROFILE_MODE="${IMPACTS_PROFILE:-none}"' in job_script
    assert 'PROFILE_ARGS=(--profile "$PROFILE_MODE")' in job_script
    assert 'PROFILE_MODE="${IMPACTS_PROFILE:-time}"' not in job_script


def test_hpc_dependency_marker_includes_project_metadata():
    job_script = Path("hpc/job.sh").read_text(encoding="utf-8")

    assert 'local setup_file="$IMPACTS_DIR/setup.cfg"' in job_script
    assert 'sha256sum "$setup_file"' in job_script
    assert 'pip install -e "$IMPACTS_DIR" --no-deps' in job_script


def test_hpc_requirements_include_postprocess_basemap_stack():
    requirements = Path("hpc/requirements-hpc.txt").read_text(encoding="utf-8")

    for package in ("contextily", "rasterio", "xyzservices", "mercantile", "pillow"):
        assert package in requirements


def test_hpc_job_uses_venv_geospatial_data_paths():
    job_script = Path("hpc/job.sh").read_text(encoding="utf-8")

    assert "configure_python_geospatial_data_paths" in job_script
    assert 'export PROJ_DATA="$pyproj_data_dir"' in job_script
    assert 'export PROJ_LIB="$pyproj_data_dir"' in job_script
    assert 'export GDAL_DATA="$gdal_data_dir"' in job_script
    assert "pyproj.CRS.from_epsg(3857)" in job_script


def test_postprocessor_resolves_modeled_emissions_from_pipeline_manifest(monkeypatch, tmp_path: Path):
    import impacts.postprocessor as postprocessor_module

    output_root = tmp_path / "impacts"
    emissions_path = output_root / "beam_emissions_by_county_process.parquet"
    emissions_path.parent.mkdir(parents=True, exist_ok=True)
    emissions_path.write_text("", encoding="utf-8")
    run_manifest_path = output_root / "pipeline_manifest.yaml"
    run_manifest_path.write_text(
        json.dumps(
            {
                "contract_version": "1",
                "model": "impacts",
                "preprocess_manifest_path": str(output_root / "preprocess_manifest.yaml"),
                "output_dir": str(output_root),
                "command": "python -m impacts emissions",
                "image": "unknown",
                "outputs": {"beam_emissions_by_county_process": str(emissions_path)},
                "pipeline": _pipeline_payload(tmp_path),
                "population_inputs": {},
                "deterministic_contract": {},
                "execution": {"dispersion_completed": False, "stopped_after": "step1_process_emissions"},
                "pipeline_manifest_path": str(run_manifest_path),
            }
        ),
        encoding="utf-8",
    )

    class _Settings:
        class impacts:
            local_output_folder = str(output_root)

            class emissions:
                passenger_inventory_file = str(tmp_path / "passenger_inventory.parquet")
                freight_inventory_file = str(tmp_path / "freight_inventory.parquet")
                enable_passenger_inventory_activity_correction = True
                enable_freight_inventory_activity_correction = True

        class shared:
            class geography:
                class fips:
                    counties = []

    monkeypatch.setattr(postprocessor_module, "load_settings_from_yaml", lambda _: _Settings())
    monkeypatch.setattr(postprocessor_module, "resolve_path", lambda path, _: path)
    monkeypatch.setattr(
        postprocessor_module.PipelineManifest,
        "from_dict",
        classmethod(lambda cls, payload: SimpleNamespace(to_dict=lambda: payload)),
    )

    resolved = postprocessor_module._resolve_modeled_emissions_path(tmp_path / "settings.yaml")

    assert resolved == emissions_path.resolve()


def test_postprocessor_resolves_county_boundaries_from_input_manifest(monkeypatch, tmp_path: Path):
    import impacts.postprocessor as postprocessor_module

    output_root = tmp_path / "impacts"
    input_root = tmp_path / "input"
    county_path = input_root / "county" / "county_boundaries.gpkg"
    county_path.parent.mkdir(parents=True, exist_ok=True)
    county_path.write_text("", encoding="utf-8")
    run_manifest_path = output_root / "pipeline_manifest.yaml"
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    preprocess_manifest_path = output_root / "preprocess_manifest.yaml"
    run_manifest_path.write_text(
        json.dumps(
            {
                "contract_version": "1",
                "model": "impacts",
                "preprocess_manifest_path": str(preprocess_manifest_path),
                "output_dir": str(output_root),
                "command": "python -m impacts emissions",
                "image": "unknown",
                "outputs": {"skims_emissions": str(input_root / "skims" / "prepared.parquet")},
                "pipeline": _pipeline_payload(tmp_path),
                "population_inputs": {},
                "deterministic_contract": {},
                "execution": {"dispersion_completed": False, "stopped_after": "step1_process_emissions"},
                "pipeline_manifest_path": str(run_manifest_path),
            }
        ),
        encoding="utf-8",
    )
    preprocess_manifest_path.write_text(
        json.dumps(
            {
                "contract_version": "1",
                "model": "impacts",
                "settings_source": str(tmp_path / "settings.yaml"),
                "staging_dir": str(input_root),
                "input_dir": str(input_root),
                "preprocess_manifest_path": str(preprocess_manifest_path),
                "maintained_execution_path": [],
                "inputs": {
                    "county_boundaries": {
                        "path": str(county_path),
                        "staged_path": str(county_path),
                    }
                },
                "pipeline": _pipeline_payload(tmp_path),
                "pilates_contract": {"stage": "terminal_postprocessing"},
                "population_inputs": {},
                "notes": [],
            }
        ),
        encoding="utf-8",
    )

    class _Settings:
        class impacts:
            local_output_folder = str(output_root)

            class emissions:
                passenger_inventory_file = str(tmp_path / "passenger_inventory.parquet")
                freight_inventory_file = str(tmp_path / "freight_inventory.parquet")
                enable_passenger_inventory_activity_correction = True
                enable_freight_inventory_activity_correction = True

        class shared:
            class geography:
                class fips:
                    counties = []

    monkeypatch.setattr(postprocessor_module, "load_settings_from_yaml", lambda _: _Settings())
    monkeypatch.setattr(postprocessor_module, "resolve_path", lambda path, _: path)
    monkeypatch.setattr(
        postprocessor_module.PipelineManifest,
        "from_dict",
        classmethod(lambda cls, payload: SimpleNamespace(to_dict=lambda: payload)),
    )
    monkeypatch.setattr(
        postprocessor_module.PreprocessManifest,
        "from_dict",
        classmethod(lambda cls, payload: SimpleNamespace(to_dict=lambda: payload)),
    )

    resolved = postprocessor_module._resolve_county_boundaries_path(
        tmp_path / "settings.yaml",
        run_manifest_path=run_manifest_path,
        registry=PathRegistry([input_root]),
    )

    assert resolved == county_path.resolve()

def test_settings_loader_rejects_invalid_shape(tmp_path: Path):
    invalid_settings_yaml = tmp_path / "invalid_settings.yaml"
    invalid_settings_yaml.write_text(
        "\n".join(
            [
                "shared:",
                "  region: sfbay",
                "inputs:",
                "  beam_network: beam/network.csv.gz",
                "outputs:",
                "  output_dir: impacts",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported keys under root"):
        load_settings_from_yaml(invalid_settings_yaml)
