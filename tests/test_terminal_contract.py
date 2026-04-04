from __future__ import annotations

from pathlib import Path

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
        "isrm_nox_to_no2_matrix_npz_path": str(tmp_path / "matrix.npz"),
        "isrm_nox_to_no2_matrix_factor": 1.0,
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
        "activity_totals_file": str(tmp_path / "activity.parquet"),
        "activity_totals_columns": {"county": "countyfp", "year": "year"},
        "prepared_skims_group_cols": ["hour", "linkId"],
        "pollutants": ["NOx", "PM2_5"],
        "pollutants_map": {"NOx": "NOx", "PM2_5": "PM2_5"},
        "annualization_days": 330.0,
        "population_sample": 0.1,
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
            "impacts.preprocessing.step3_integrate_grids",
            "impacts.workflow.step1_process_emissions",
            "impacts.workflow.step2_compute_inmap_concentrations",
            "impacts.workflow.step3_compute_aermod_concentrations",
        ],
        "inputs": {"settings": {"path": str(tmp_path / "settings.yaml")}},
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
    assert config.beam.local_input_folder == "pilates/beam/production/"
    assert config.beam.local_output_folder == "beam/beam_output/"
    assert config.impacts.local_input_folder == "impacts/input/"
    assert config.impacts.dispersions.inmap.enabled is True
    assert config.impacts.dispersions.inmap.grid_path.endswith("isrm_polygon_wgs84.gpkg")
    assert config.impacts.dispersions.inmap.isrm_nox_to_no2_matrix_factor == 1.0
    assert config.impacts.dispersions.aermod.enabled is True
    assert config.impacts.exposure.enabled is True
    assert config.impacts.exposure.population_folder == "urbansim/2018"


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

    overlay = Path(__file__).resolve().parents[1] / "src" / "impacts" / "adapters" / "pilates_overlay.yaml"
    config = build_settings_from_pilates(pilates_settings=pilates_settings, impacts_overlay=overlay)

    assert config.run.region == "sfbay"
    assert config.run.scenario == "base"
    assert config.run.start_year == 2017
    assert config.shared.geography.local_crs == "EPSG:26910"
    assert config.beam.local_input_folder == "pilates/beam/production/"
    assert config.beam.local_output_folder == "beam/beam_output/"
    assert config.impacts.local_input_folder == "impacts/input/"
    assert config.impacts.emissions.osm_network_folder.endswith("r5/sfbay-cbg5500-weakConn-network")
    assert config.impacts.dispersions.inmap.enabled is True
    assert config.impacts.dispersions.inmap.isrm_zarr == "~/Workspace/Simulation/sfbay/inmap/isrm_v1.2.1.zarr"
    assert config.impacts.dispersions.inmap.isrm_nox_to_no2_matrix_factor == 1.0
    assert config.impacts.dispersions.aermod.enabled is True
    assert config.impacts.dispersions.aermod.grid_size_meters == 100.0
    assert config.impacts.exposure.population_folder == "urbansim/2018"


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
    payload["isrm_nox_to_no2_matrix_npz_path"] = None
    payload["isrm_nox_to_no2_matrix_factor"] = None

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
    import impacts.preprocessing.step3_integrate_grids as step3_integrate_grids
    import impacts.workflow.prepare_emissions_from_skims as prepare_emissions_from_skims
    import impacts.workflow.step1_process_emissions as step1_process_emissions
    import impacts.runner as runner_module

    monkeypatch.setattr(runner_module, "load_structured_file", lambda _: _inputs_manifest_payload(tmp_path))
    monkeypatch.setattr(
        step3_integrate_grids,
        "run",
        lambda pipeline, raw_dir, input_root: (tmp_path / "grid_intersection.parquet", None),
    )
    monkeypatch.setattr(
        step1_process_emissions,
        "run",
        lambda pipeline, raw_dir, input_root, grid_intersection_path, intersection_df=None, manifest_inputs=None: {
            "beam_emissions_for_inmap": str(tmp_path / "beam_emissions_for_inmap.parquet")
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

    def _fake_preprocess(settings_path, staging_dir, manifest_path=None):
        calls["preprocess"] = {
            "settings_path": str(settings_path),
            "staging_dir": str(staging_dir),
            "manifest_path": manifest_path,
        }
        return {"inputs_manifest_path": str(tmp_path / "workspace" / "inputs_manifest.yaml")}

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
        workspace=tmp_path / "workspace",
        run_dispersion=False,
    )

    assert result["run_manifest_path"].endswith("run_manifest.yaml")
    assert calls["preprocess"]["settings_path"].endswith("settings.yaml")
    assert calls["preprocess"]["staging_dir"].endswith("workspace")
    assert calls["run"]["input_manifest_path"].endswith("inputs_manifest.yaml")
    assert calls["run"]["output_dir"].endswith("workspace")
    assert calls["run"]["run_dispersion"] is False


def test_cli_run_accepts_settings_workspace(monkeypatch, tmp_path: Path):
    calls = {}

    def _fake_run_from_settings(settings_path, workspace, run_manifest_path=None, run_dispersion=False):
        calls["settings_run"] = {
            "settings_path": str(settings_path),
            "workspace": str(workspace),
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
            "--workspace",
            str(tmp_path / "workspace"),
        ]
    )

    assert exit_code == 0
    assert calls["settings_run"]["settings_path"].endswith("settings.yaml")
    assert calls["settings_run"]["workspace"].endswith("workspace")
    assert calls["settings_run"]["run_dispersion"] is False


def test_postprocess_from_settings_delegates_through_runner(monkeypatch, tmp_path: Path):
    calls = {}

    class _Settings:
        class impacts:
            local_output_folder = "impacts"

    def _fake_settings_run(settings_path, workspace, run_dispersion=True):
        calls["settings_run"] = {
            "settings_path": str(settings_path),
            "workspace": str(workspace),
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
        workspace=tmp_path / "workspace",
    )

    assert result["postprocess_manifest_path"].endswith("postprocess_manifest.yaml")
    assert calls["settings_run"]["settings_path"].endswith("settings.yaml")
    assert calls["settings_run"]["workspace"].endswith("workspace")
    assert calls["settings_run"]["run_dispersion"] is True
    assert calls["postprocess"]["run_manifest_path"].endswith("run_manifest.yaml")
    assert calls["postprocess"]["output_dir"].endswith("impacts")

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
