from __future__ import annotations

import json
import subprocess
import sys
import yaml
from pathlib import Path
from types import SimpleNamespace

import pytest

from impacts.__main__ import main
from impacts.config.settings_builder import build_settings_from_pilates
from impacts.config.settings_builder import load_settings_from_yaml
from impacts.manifest.schema import InputsManifest
from impacts.manifest.schema import PipelineConfig
from impacts.manifest.schema import PostprocessManifest
from impacts.manifest.schema import RunManifest
from impacts.manifest.schema import ActivitiesManifest
from impacts.postprocessor import postprocess_from_settings
from impacts.pipeline.preprocessing.step1_collect_inputs import _resolve_region_input_root
from impacts.runner import run_emissions_from_run_manifest


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
        "pollutants": ["NOx", "PM2_5"],
        "source_pollutants": ["NOx", "PM2_5"],
        "vehicle_category_metadata_file": str(tmp_path / "vehicle_category_metadata.csv"),
        "annualization_days": {"light_duty": 327.0, "medium_heavy_duty": 312.0},
        "population_sample": 0.1,
        "include_passenger": True,
        "include_freight": True,
    }


def _inputs_manifest_payload(tmp_path: Path) -> dict:
    return {
        "contract_version": "1",
        "model": "impacts",
        "settings_source": str(tmp_path / "settings.yaml"),
        "staging_dir": str(tmp_path / "workspace"),
        "input_dir": str(tmp_path / "workspace" / "staged"),
        "inputs_manifest_path": str(tmp_path / "workspace" / "inputs_manifest.yaml"),
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
    settings_yaml = Path(__file__).resolve().parents[1] / "examples" / "pilates" / "settings.yaml"

    config = load_settings_from_yaml(settings_yaml)

    assert config.run.region == "sfbay"
    assert config.run.scenario == "base"
    assert config.shared.geography.fips.state == "06"
    assert config.shared.geography.fips.counties[0] == "001"
    assert config.beam.local_input_folder == "~/Workspace/Models/beam/beam-data/beam-data-sfbay"
    assert config.beam.local_output_folder == "beam/beam_output/"
    assert config.beam.router_directory == "r5/sfbay-cbg5500-weakConn-network"
    assert config.impacts.pipeline.inmap is True
    assert config.impacts.dispersions.inmap.grid_path.endswith("isrm_polygon_wgs84.gpkg")
    assert config.impacts.pipeline.aermod is True
    assert config.impacts.pipeline.exposure is True
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
    assert config.impacts.analysis.inventory_targets == []


def test_resolve_region_input_root_accepts_direct_region_root_layout(tmp_path: Path) -> None:
    beam_input_root = tmp_path / "beam-data-sfbay"
    (beam_input_root / "freight").mkdir(parents=True)
    (beam_input_root / "vehicle-tech").mkdir()

    resolved = _resolve_region_input_root(beam_input_root=beam_input_root, region="sfbay")

    assert resolved == beam_input_root


def test_pipeline_example_is_source_of_truth_for_builtin_pipeline_impacts_settings() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_payload = yaml.safe_load((repo_root / "examples" / "pilates" / "settings.yaml").read_text())
    default_payload = yaml.safe_load((repo_root / "src" / "impacts" / "config" / "settings.yaml").read_text())
    overlay_payload = yaml.safe_load((repo_root / "src" / "impacts" / "pipeline" / "adapters" / "pilates_overlay.yaml").read_text())

    shared_keys = {"emissions", "dispersions", "analysis"}
    assert shared_keys <= set(default_payload["impacts"])
    assert shared_keys <= set(example_payload["impacts"])
    assert shared_keys <= set(overlay_payload["impacts"])
def test_pipeline_profile_memray_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("impacts:\n  local_output_folder: impacts_output\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        main(["pipeline", "--config", str(config_path), "--profile", "memray"])


def test_pipeline_profile_time_relaunches_under_impacts_output(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("impacts:\n  local_output_folder: impacts_output\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "impacts.__main__.load_settings_from_yaml",
        lambda _: SimpleNamespace(impacts=SimpleNamespace(local_output_folder="impacts_output")),
    )
    monkeypatch.setattr("impacts.__main__.resolve_path", lambda path, _: str(tmp_path / path))

    def _fake_run(command, env=None, check=False):
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
                "  local_input_folder: pilates/beam/production/",
                "  local_output_folder: beam/beam_output/",
                "  router_directory: r5/network",
                "impacts:",
                "  local_input_folder: impacts/impacts_inputs/",
                "  local_output_folder: impacts/impacts_output/",
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
    pipeline = PipelineConfig.from_dict(payload)
    assert pipeline.vehicle_category_metadata_file == str(days_csv)
    assert pipeline.annualization_days == {"light_duty": 327.0, "medium_heavy_duty": 312.0}
    assert pipeline.include_passenger is True
    assert pipeline.include_freight is True


def test_build_settings_from_pilates_template_uses_current_overlay_shape(tmp_path: Path):
    pilates_settings = tmp_path / "pilates_settings.yaml"
    pilates_settings.write_text(
        "\n".join(
            [
                "run:",
                "  region: sfbay",
                "  scenario: base",
                "  start_year: 2017",
                "shared:",
                "  geography:",
                "    FIPS:",
                '      state: "06"',
                "      counties:",
                '        - "001"',
                '        - "013"',
                "    local_crs: EPSG:26910",
                "beam:",
                "  local_input_folder: pilates/beam/production/",
                "  local_output_folder: beam/beam_output/",
                "  router_directory: beam/beam_output/r5/sfbay-cbg5500-weakConn-network",
            ]
        ),
        encoding="utf-8",
    )

    overlay = Path(__file__).resolve().parents[1] / "src" / "impacts" / "pipeline" / "adapters" / "pilates_overlay.yaml"
    config = build_settings_from_pilates(pilates_settings=pilates_settings, impacts_overlay=overlay)

    assert config.run.region == "sfbay"
    assert config.run.scenario == "base"
    assert config.run.start_year == 2017
    assert config.shared.geography.local_crs == "EPSG:26910"
    assert config.beam.local_input_folder == "pilates/beam/production/"
    assert config.beam.local_output_folder == "beam/beam_output/"
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
    inputs_manifest = InputsManifest.from_dict(_inputs_manifest_payload(tmp_path)).to_dict()
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
    run_manifest = RunManifest.from_dict(
        {
            "contract_version": "1",
            "model": "impacts",
            "input_manifest_path": inputs_manifest["inputs_manifest_path"],
            "output_dir": str(tmp_path / "workspace"),
            "command": "python -m impacts emissions",
            "image": "unknown",
            "outputs": {"skims_emissions": str(tmp_path / "prepared.parquet")},
            "pipeline": pipeline,
            "population_inputs": {},
            "deterministic_contract": {},
            "execution": {"dispersion_completed": False, "stopped_after": "step1_process_emissions"},
            "run_manifest_path": str(tmp_path / "workspace" / "run_manifest.yaml"),
        }
    ).to_dict()
    postprocess_manifest = PostprocessManifest.from_dict(
        {
            "contract_version": "1",
            "model": "impacts",
            "run_manifest_path": run_manifest["run_manifest_path"],
            "output_dir": str(tmp_path / "impacts"),
            "analysis_outputs": {},
            "validation": {},
            "notes": [],
            "postprocess_manifest_path": str(tmp_path / "impacts" / "postprocess_manifest.yaml"),
        }
    ).to_dict()

    assert activities_manifest["outputs"]["outputs_root"].endswith("emfac")
    assert inputs_manifest["pipeline"]["region"] == "sfbay"
    assert run_manifest["execution"]["stopped_after"] == "step1_process_emissions"
    assert postprocess_manifest["analysis_outputs"] == {}


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


def test_run_emissions_from_run_manifest_uses_staged_intersections_from_preprocess(monkeypatch, tmp_path: Path):
    import impacts.pipeline.workflow.prepare_emissions_from_skims as prepare_emissions_from_skims
    import impacts.pipeline.workflow.step1_process_emissions as step1_process_emissions
    import impacts.runner as runner_module

    for name in (
        "beam_osm_county_intersection.parquet",
        "beam_osm_inmap_intersection.parquet",
        "beam_osm_aermod_intersection.parquet",
    ):
        (tmp_path / name).write_text("", encoding="utf-8")

    run_manifest_path = tmp_path / "workspace" / "run_manifest.yaml"

    def _fake_load_structured_file(path):
        path = Path(path)
        if path == run_manifest_path:
            return {
                "contract_version": "1",
                "model": "impacts",
                "input_manifest_path": str(tmp_path / "workspace" / "inputs_manifest.yaml"),
                "output_dir": str(tmp_path / "impacts_output"),
                "command": "python -m impacts emissions",
                "image": "unknown",
                "outputs": {"skims_emissions": str(tmp_path / "prepared_skims.parquet")},
                "pipeline": _pipeline_payload(tmp_path),
                "population_inputs": {},
                "deterministic_contract": {},
                "execution": {"dispersion_completed": False, "stopped_after": "preprocess"},
                "run_manifest_path": str(run_manifest_path),
            }
        return _inputs_manifest_payload(tmp_path)

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
        prepare_emissions_from_skims,
        "resolve_prepared_skims_path",
        lambda input_root: str(tmp_path / "prepared_skims.parquet"),
    )

    result = run_emissions_from_run_manifest(
        run_manifest_path=run_manifest_path,
    )

    assert result["execution"]["dispersion_completed"] is False
    assert result["execution"]["stopped_after"] == "step1_process_emissions"
    assert "step1_process_emissions" in result["execution"]["stage_timings_seconds"]
    assert captured["grid_intersection_paths"]["county"].endswith("beam_osm_county_intersection.parquet")
    assert captured["grid_intersection_paths"]["inmap"].endswith("beam_osm_inmap_intersection.parquet")
    assert captured["grid_intersection_paths"]["aermod"].endswith("beam_osm_aermod_intersection.parquet")


def test_build_inputs_manifest_runs_step3_and_registers_intersections(monkeypatch, tmp_path: Path):
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
                inmap=SimpleNamespace(grid_epsg=26910),
                aermod=SimpleNamespace(
                    asrv_patterns_epsg=4326,
                    grid_size_meters=100.0,
                ),
            ),
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

    manifest = preprocessor_module.build_inputs_manifest(config_path)

    assert calls["step3"]["input_root"] == input_root
    assert manifest["maintained_execution_path"] == [
        "impacts.pipeline.workflow.step1_process_emissions",
        "impacts.pipeline.workflow.step2_compute_inmap_concentrations",
        "impacts.pipeline.workflow.step3_compute_aermod_concentrations",
    ]
    assert manifest["inputs"]["county_intersection"]["source_path"].endswith("beam_osm_county_intersection.parquet")
    assert manifest["inputs"]["inmap_intersection"]["source_path"].endswith("beam_osm_inmap_intersection.parquet")
    assert manifest["inputs"]["aermod_intersection"]["source_path"].endswith("beam_osm_aermod_intersection.parquet")

def test_cli_rejects_removed_run_command() -> None:
    with pytest.raises(SystemExit):
        main(["run"])


def test_cli_fleet_uses_activities_manifest(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def _fake_run_fleet_main(*, activities_manifest_path):
        captured["activities_manifest_path"] = str(activities_manifest_path)

    monkeypatch.setattr("impacts.emfac.fleet.main.main", _fake_run_fleet_main)

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
        return {"run_manifest_path": str(tmp_path / "workspace" / "run_manifest.yaml")}

    def _fake_emissions_run(run_manifest_path):
        calls["emissions_run"] = {
            "run_manifest_path": str(run_manifest_path),
        }
        return {"run_manifest_path": str(tmp_path / "workspace" / "run_manifest.yaml")}

    def _fake_postprocess(run_manifest_path, manifest_path=None):
        calls["postprocess"] = {
            "run_manifest_path": str(run_manifest_path),
            "manifest_path": manifest_path,
        }
        return {"postprocess_manifest_path": str(tmp_path / "impacts" / "postprocess_manifest.yaml")}

    monkeypatch.setattr("impacts.postprocessor.load_settings_from_yaml", lambda _: _Settings())
    monkeypatch.setattr("impacts.preprocessor.preprocess_workflow", _fake_preprocess)
    monkeypatch.setattr("impacts.runner.run_emissions_from_run_manifest", _fake_emissions_run)
    monkeypatch.setattr("impacts.postprocessor.postprocess_from_run_manifest", _fake_postprocess)

    result = postprocess_from_settings(
        settings_path=tmp_path / "settings.yaml",
    )

    assert result["postprocess_manifest_path"].endswith("postprocess_manifest.yaml")
    assert calls["preprocess"]["settings_path"].endswith("settings.yaml")
    assert calls["emissions_run"]["run_manifest_path"].endswith("run_manifest.yaml")
    assert calls["postprocess"]["run_manifest_path"].endswith("run_manifest.yaml")


def test_analysis_runner_resolves_modeled_emissions_from_run_manifest(monkeypatch, tmp_path: Path):
    from impacts import runner as analysis_runner

    output_root = tmp_path / "impacts"
    emissions_path = output_root / "beam_emissions_by_county_process.parquet"
    emissions_path.parent.mkdir(parents=True, exist_ok=True)
    emissions_path.write_text("", encoding="utf-8")
    run_manifest_path = output_root / "run_manifest.yaml"
    run_manifest_path.write_text(
        json.dumps(
            {
                "contract_version": "1",
                "model": "impacts",
                "input_manifest_path": str(output_root / "inputs_manifest.yaml"),
                "output_dir": str(output_root),
                "command": "python -m impacts emissions",
                "image": "unknown",
                "outputs": {"beam_emissions_by_county_process": str(emissions_path)},
                "pipeline": _pipeline_payload(tmp_path),
                "population_inputs": {},
                "deterministic_contract": {},
                "execution": {"dispersion_completed": False, "stopped_after": "step1_process_emissions"},
                "run_manifest_path": str(run_manifest_path),
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

    class _AnalysisSettings(_Settings):
        class impacts(_Settings.impacts):
            class analysis:
                sector_targets = []
                targets = []

    monkeypatch.setattr(analysis_runner, "load_settings_from_yaml", lambda _: _AnalysisSettings())
    monkeypatch.setattr(analysis_runner, "resolve_path", lambda path, _: path)
    monkeypatch.setattr(
        analysis_runner.RunManifest,
        "from_dict",
        classmethod(lambda cls, payload: SimpleNamespace(to_dict=lambda: payload)),
    )

    resolved = analysis_runner._resolve_analysis_modeled_emissions_path(tmp_path / "settings.yaml")

    assert resolved == emissions_path.resolve()


def test_analysis_runner_resolves_county_boundaries_from_input_manifest(monkeypatch, tmp_path: Path):
    from impacts import runner as analysis_runner

    output_root = tmp_path / "impacts"
    input_root = tmp_path / "input"
    county_path = input_root / "county" / "county_boundaries.gpkg"
    county_path.parent.mkdir(parents=True, exist_ok=True)
    county_path.write_text("", encoding="utf-8")
    run_manifest_path = output_root / "run_manifest.yaml"
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    input_manifest_path = output_root / "inputs_manifest.yaml"
    run_manifest_path.write_text(
        json.dumps(
            {
                "contract_version": "1",
                "model": "impacts",
                "input_manifest_path": str(input_manifest_path),
                "output_dir": str(output_root),
                "command": "python -m impacts emissions",
                "image": "unknown",
                "outputs": {"skims_emissions": str(input_root / "skims" / "prepared.parquet")},
                "pipeline": _pipeline_payload(tmp_path),
                "population_inputs": {},
                "deterministic_contract": {},
                "execution": {"dispersion_completed": False, "stopped_after": "step1_process_emissions"},
                "run_manifest_path": str(run_manifest_path),
            }
        ),
        encoding="utf-8",
    )
    input_manifest_path.write_text(
        json.dumps(
            {
                "contract_version": "1",
                "model": "impacts",
                "settings_source": str(tmp_path / "settings.yaml"),
                "staging_dir": str(input_root),
                "input_dir": str(input_root),
                "inputs_manifest_path": str(input_manifest_path),
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

    monkeypatch.setattr(analysis_runner, "load_settings_from_yaml", lambda _: _Settings())
    monkeypatch.setattr(analysis_runner, "resolve_path", lambda path, _: path)
    monkeypatch.setattr(
        analysis_runner.RunManifest,
        "from_dict",
        classmethod(lambda cls, payload: SimpleNamespace(to_dict=lambda: payload)),
    )
    monkeypatch.setattr(
        analysis_runner.InputsManifest,
        "from_dict",
        classmethod(lambda cls, payload: SimpleNamespace(to_dict=lambda: payload)),
    )

    resolved = analysis_runner._resolve_analysis_county_boundaries_path(tmp_path / "settings.yaml")

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

    with pytest.raises(ValueError, match="Unsupported keys under shared"):
        load_settings_from_yaml(invalid_settings_yaml)
