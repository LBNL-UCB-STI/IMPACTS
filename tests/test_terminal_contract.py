from __future__ import annotations

import json
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
from impacts.postprocessor import postprocess_from_settings
from impacts.runner import run_from_input_manifest
from impacts.runner import run_from_settings


def _pipeline_payload(tmp_path: Path) -> dict:
    return {
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
        "asrv_patterns_file": str(tmp_path / "asrv_patterns.parquet"),
        "asrv_patterns_epsg": 4326,
        "aermod_default_site": "LIVERMORE_2015",
        "aermod_default_urban_class": 0,
        "aermod_default_temporal": "CITYSTREET",
        "aermod_default_release_height": 1.0,
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
            "impacts.pipeline.preprocessing.step3_integrate_grids",
            "impacts.pipeline.workflow.step1_process_emissions",
            "impacts.pipeline.workflow.step2_compute_inmap_concentrations",
            "impacts.pipeline.workflow.step3_compute_aermod_concentrations",
        ],
        "inputs": {"settings": {"path": str(tmp_path / "settings.yaml")}},
        "pipeline": _pipeline_payload(tmp_path),
        "pilates_contract": {"stage": "terminal_postprocessing"},
        "population_inputs": {},
        "notes": [],
    }


def test_example_settings_yaml_is_current_settings_file():
    settings_yaml = Path(__file__).resolve().parents[1] / "examples" / "pipeline" / "pilates" / "settings.yaml"

    config = load_settings_from_yaml(settings_yaml)

    assert config.run.region == "sfbay"
    assert config.run.scenario == "base"
    assert config.shared.geography.fips.state == "06"
    assert config.shared.geography.fips.counties[0] == "001"
    assert config.beam.local_input_folder == "beam/production/"
    assert config.beam.local_output_folder == "beam/beam_output/"
    assert config.impacts.dispersions.inmap.enabled is True
    assert config.impacts.dispersions.inmap.grid_path.endswith("isrm_polygon_wgs84.gpkg")
    assert config.impacts.dispersions.aermod.enabled is True
    assert config.impacts.dispersions.aermod.default_site == "LIVERMORE_2015"
    assert config.impacts.dispersions.aermod.default_temporal == "CITYSTREET"
    assert config.impacts.exposure.enabled is True
    assert config.impacts.exposure.population_folder == "urbansim/atlas-2019"
    assert config.impacts.beam.include_passenger is True
    assert config.impacts.beam.include_freight is True
    assert config.impacts.beam.passenger_vehicle_types_file == "vehicle-tech/vehicleTypes--atlas--2019-Baseline--EM.csv"
    assert config.impacts.beam.freight_vehicle_types_file == "vehicle-tech/vehicleTypes--frism--2018-Baseline--EM.csv"
    assert len(config.impacts.analysis.sector_targets) == 6
    assert (
        config.impacts.emissions.vehicle_category_metadata_file
        == "~/Workspace/Models/beam-data/beam-data-sfbay/vehicle-tech/_emissions_vehicle_catalog.csv"
    )
    assert config.impacts.emissions.defaults.annualization_days.light_duty == 327.0
    assert config.impacts.emissions.defaults.annualization_days.medium_heavy_duty == 312.0
    assert config.impacts.analysis.sector_targets[0].source == "mobile_onroad"
    assert config.impacts.analysis.sector_targets[0].sector == "passenger_cars"
    assert config.impacts.analysis.sector_targets[0].annual_pm25_short_tons == 714.26
    assert config.impacts.analysis.sector_targets[0].annual_nox_short_tons == 3964.37
    assert config.impacts.analysis.sector_targets[-1].source == "road_dust"
    assert config.impacts.analysis.sector_targets[-1].annual_pm25_short_tons == 1499.25
    assert config.impacts.analysis.sector_targets[-1].annual_nox_short_tons is None
    assert config.impacts.analysis.inventory_targets == []


def test_pipeline_example_is_source_of_truth_for_builtin_pipeline_impacts_settings() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_payload = yaml.safe_load((repo_root / "examples" / "pipeline" / "pilates" / "settings.yaml").read_text())
    default_payload = yaml.safe_load((repo_root / "src" / "impacts" / "config" / "settings.yaml").read_text())
    overlay_payload = yaml.safe_load((repo_root / "src" / "impacts" / "pipeline" / "adapters" / "pilates_overlay.yaml").read_text())

    assert default_payload["impacts"] == example_payload["impacts"]
    assert overlay_payload["impacts"] == example_payload["impacts"]


def test_top_level_emfac_command_defaults_to_all(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "emfac.yaml"
    config_path.write_text("emfac: {}\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake_main(argv=None):
        captured["argv"] = list(argv or [])

    monkeypatch.setattr("impacts.emfac.__main__.main", _fake_main)

    assert main(["emfac", "--config", str(config_path)]) == 0
    assert captured["argv"] == ["--config", str(config_path)]


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
                "impacts:",
                "  local_output_folder: impacts/impacts_output/",
                "  beam:",
                "    passenger_vehicle_types_file: vehicle-tech/vehicleTypes--atlas--2019-Baseline--EM.csv",
                "    freight_vehicle_types_file: vehicle-tech/vehicleTypes--frism--2018-Baseline--EM.csv",
                "    population_sample: 0.1",
                "    include_passenger: false",
                "    include_freight: true",
                "  emissions:",
                "    osm_network_folder: r5/network",
                "    emissions_rates_folder: vehicle-tech/emissions/2018-Baseline",
                "    inventory:",
                "      passenger_file: beam/production/sfbay/vehicle-tech/emissions/passenger_inventory.parquet",
                "      freight_file: beam/production/sfbay/vehicle-tech/emissions/freight_inventory.parquet",
                "      enable_passenger_activity_correction: true",
                "      enable_freight_activity_correction: false",
                f"    vehicle_category_metadata_file: {days_csv.name}",
                "    defaults:",
                "      annualization_days:",
                "        light_duty: 327.0",
                "        medium_heavy_duty: 312.0",
                "    pollutants: [NOx, PM25]",
                "  dispersions:",
                "    inmap:",
                "      enabled: false",
                "    aermod:",
                "      enabled: false",
                "  exposure:",
                "    enabled: false",
            ]
        ),
        encoding="utf-8",
    )
    config = load_settings_from_yaml(settings_yaml)
    assert config.impacts.emissions.defaults.annualization_days.light_duty == 327.0
    assert config.impacts.emissions.defaults.annualization_days.medium_heavy_duty == 312.0
    assert config.impacts.emissions.vehicle_category_metadata_file == days_csv.name
    assert config.impacts.emissions.inventory.passenger_file.endswith("passenger_inventory.parquet")
    assert config.impacts.emissions.inventory.freight_file.endswith("freight_inventory.parquet")
    assert config.impacts.emissions.inventory.enable_passenger_activity_correction is True
    assert config.impacts.emissions.inventory.enable_freight_activity_correction is False
    assert config.impacts.beam.include_passenger is False
    assert config.impacts.beam.include_freight is True
    assert config.impacts.beam.passenger_vehicle_types_file.endswith("vehicleTypes--atlas--2019-Baseline--EM.csv")
    assert config.impacts.beam.freight_vehicle_types_file.endswith("vehicleTypes--frism--2018-Baseline--EM.csv")

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
    assert config.impacts.emissions.osm_network_folder.endswith("r5/sfbay-cbg5500-weakConn-network")
    assert config.impacts.dispersions.inmap.enabled is True
    assert config.impacts.dispersions.inmap.isrm_zarr == "~/Workspace/Simulation/sfbay/inmap/isrm_v1.2.1.zarr"
    assert config.impacts.dispersions.aermod.enabled is True
    assert config.impacts.dispersions.aermod.grid_size_meters == 100.0
    assert config.impacts.dispersions.aermod.default_site == "LIVERMORE_2015"
    assert config.impacts.dispersions.aermod.default_temporal == "CITYSTREET"
    assert config.impacts.exposure.population_folder == "urbansim/atlas-2019"
    assert config.impacts.beam.include_passenger is True
    assert config.impacts.beam.include_freight is True
    assert config.impacts.beam.passenger_vehicle_types_file == "vehicle-tech/vehicleTypes--atlas--2019-Baseline--EM.csv"
    assert config.impacts.beam.freight_vehicle_types_file == "vehicle-tech/vehicleTypes--frism--2018-Baseline--EM.csv"
    assert len(config.impacts.analysis.sector_targets) == 6
    assert (
        config.impacts.emissions.vehicle_category_metadata_file
        == "~/Workspace/Models/beam-data/beam-data-sfbay/vehicle-tech/_emissions_vehicle_catalog.csv"
    )
    assert config.impacts.analysis.inventory_targets == []


def test_manifest_models_round_trip_current_shape(tmp_path: Path):
    pipeline = PipelineConfig.from_dict(_pipeline_payload(tmp_path)).to_dict()
    inputs_manifest = InputsManifest.from_dict(_inputs_manifest_payload(tmp_path)).to_dict()
    run_manifest = RunManifest.from_dict(
        {
            "contract_version": "1",
            "model": "impacts",
            "input_manifest_path": inputs_manifest["inputs_manifest_path"],
            "output_dir": str(tmp_path / "workspace"),
            "outputs_dir": str(tmp_path / "workspace" / "outputs"),
            "command": "python -m impacts run",
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
            "canonical_artifact": {"path": str(tmp_path / "impacts" / "impacts_exposure_table.parquet")},
            "analysis_outputs": {},
            "validation": {},
            "notes": [],
            "postprocess_manifest_path": str(tmp_path / "impacts" / "postprocess_manifest.yaml"),
        }
    ).to_dict()

    assert inputs_manifest["pipeline"]["region"] == "sfbay"
    assert run_manifest["execution"]["stopped_after"] == "step1_process_emissions"
    assert postprocess_manifest["canonical_artifact"]["path"].endswith(".parquet")


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


def test_run_from_input_manifest_uses_current_step_name(monkeypatch, tmp_path: Path):
    import impacts.pipeline.preprocessing.step3_integrate_grids as step3_integrate_grids
    import impacts.pipeline.workflow.prepare_emissions_from_skims as prepare_emissions_from_skims
    import impacts.pipeline.workflow.step1_process_emissions as step1_process_emissions
    import impacts.runner as runner_module

    monkeypatch.setattr(runner_module, "load_structured_file", lambda _: _inputs_manifest_payload(tmp_path))
    monkeypatch.setattr(
        runner_module,
        "load_settings_from_yaml",
        lambda _: SimpleNamespace(
            impacts=SimpleNamespace(local_output_folder=str(tmp_path / "impacts_output"))
        ),
    )
    monkeypatch.setattr(
        step3_integrate_grids,
        "run",
        lambda pipeline, raw_dir, input_root, manifest_inputs=None: (
            {
                "county": str(tmp_path / "beam_osm_county_intersection.parquet"),
                "inmap": str(tmp_path / "beam_osm_inmap_intersection.parquet"),
                "aermod": str(tmp_path / "beam_osm_aermod_intersection.parquet"),
            },
            {"county": None, "inmap": None, "aermod": None},
        ),
    )
    monkeypatch.setattr(
        step1_process_emissions,
        "run",
        lambda pipeline, raw_dir, input_root, grid_intersection_paths, intersection_dfs=None, manifest_inputs=None: {
            "beam_emissions_by_county_process": str(tmp_path / "beam_emissions_by_county_process.parquet"),
            "beam_emissions_for_inmap": str(tmp_path / "beam_emissions_for_inmap.parquet"),
        },
    )
    monkeypatch.setattr(
        prepare_emissions_from_skims,
        "resolve_prepared_skims_path",
        lambda input_root: str(tmp_path / "prepared_skims.parquet"),
    )

    result = run_from_input_manifest(
        input_manifest_path=tmp_path / "workspace" / "inputs_manifest.yaml",
        output_dir=tmp_path / "workspace",
        run_dispersion=False,
    )

    assert result["execution"]["dispersion_completed"] is False
    assert result["execution"]["stopped_after"] == "step1_process_emissions"


def test_run_from_settings_delegates_through_preprocess(monkeypatch, tmp_path: Path):
    calls = {}

    def _fake_preprocess(settings_path, manifest_path=None):
        calls["preprocess"] = {
            "settings_path": str(settings_path),
            "manifest_path": manifest_path,
        }
        return {
            "inputs_manifest_path": str(tmp_path / "impacts" / "inputs_manifest.yaml"),
            "staging_dir": str(tmp_path / "impacts" / "tmp"),
            "input_dir": str(tmp_path / "impacts" / "tmp"),
        }

    def _fake_run(input_manifest_path, output_dir, run_manifest_path=None, run_dispersion=False):
        calls["run"] = {
            "input_manifest_path": str(input_manifest_path),
            "output_dir": str(output_dir),
            "run_manifest_path": run_manifest_path,
            "run_dispersion": run_dispersion,
        }
        return {"run_manifest_path": str(tmp_path / "workspace" / "run_manifest.yaml")}

    monkeypatch.setattr("impacts.preprocessor.preprocess_workflow", _fake_preprocess)
    monkeypatch.setattr("impacts.runner.run_from_input_manifest", _fake_run)

    result = run_from_settings(
        settings_path=tmp_path / "settings.yaml",
        run_dispersion=False,
    )

    assert result["run_manifest_path"].endswith("run_manifest.yaml")
    assert calls["preprocess"]["settings_path"].endswith("settings.yaml")
    assert calls["run"]["input_manifest_path"].endswith("inputs_manifest.yaml")
    assert calls["run"]["output_dir"].endswith("tmp")
    assert calls["run"]["run_dispersion"] is False


def test_cli_run_accepts_settings_only(monkeypatch, tmp_path: Path):
    calls = {}

    def _fake_run_from_settings(settings_path, run_manifest_path=None, run_dispersion=False):
        calls["settings_run"] = {
            "settings_path": str(settings_path),
            "run_manifest_path": run_manifest_path,
            "run_dispersion": run_dispersion,
        }
        return {"run_manifest_path": str(tmp_path / "workspace" / "run_manifest.yaml")}

    monkeypatch.setattr("impacts.runner.run_from_settings", _fake_run_from_settings)

    exit_code = main(
        [
            "run",
            "--config",
            str(tmp_path / "settings.yaml"),
        ]
    )

    assert exit_code == 0
    assert calls["settings_run"]["settings_path"].endswith("settings.yaml")
    assert calls["settings_run"]["run_dispersion"] is False


def test_postprocess_from_settings_delegates_through_runner(monkeypatch, tmp_path: Path):
    calls = {}

    class _Settings:
        class impacts:
            local_output_folder = "impacts"

    def _fake_settings_run(settings_path, run_dispersion=True):
        calls["settings_run"] = {
            "settings_path": str(settings_path),
            "run_dispersion": run_dispersion,
        }
        return {"run_manifest_path": str(tmp_path / "workspace" / "run_manifest.yaml")}

    def _fake_postprocess(run_manifest_path, output_dir, manifest_path=None):
        calls["postprocess"] = {
            "run_manifest_path": str(run_manifest_path),
            "output_dir": str(output_dir),
            "manifest_path": manifest_path,
        }
        return {"postprocess_manifest_path": str(tmp_path / "impacts" / "postprocess_manifest.yaml")}

    monkeypatch.setattr("impacts.postprocessor.load_settings_from_yaml", lambda _: _Settings())
    monkeypatch.setattr("impacts.postprocessor.resolve_path", lambda path, _: path)
    monkeypatch.setattr("impacts.runner.run_from_settings", _fake_settings_run)
    monkeypatch.setattr("impacts.postprocessor.postprocess_from_run_manifest", _fake_postprocess)

    result = postprocess_from_settings(
        settings_path=tmp_path / "settings.yaml",
    )

    assert result["postprocess_manifest_path"].endswith("postprocess_manifest.yaml")
    assert calls["settings_run"]["settings_path"].endswith("settings.yaml")
    assert calls["settings_run"]["run_dispersion"] is True
    assert calls["postprocess"]["run_manifest_path"].endswith("run_manifest.yaml")
    assert calls["postprocess"]["output_dir"].endswith("impacts")


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
                "outputs_dir": str(output_root),
                "command": "python -m impacts run",
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
                "outputs_dir": str(output_root),
                "command": "python -m impacts run",
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
