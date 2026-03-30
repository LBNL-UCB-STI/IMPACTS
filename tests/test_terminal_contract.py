from __future__ import annotations

from pathlib import Path

import pytest

from impacts.__main__ import main
from impacts.config.runtime_builder import build_runtime_config_from_pilates
from impacts.config.runtime_builder import build_runtime_config_from_runtime_yaml
from impacts.manifest.schema import InputsManifest
from impacts.manifest.schema import PipelineConfig
from impacts.manifest.schema import PostprocessManifest
from impacts.manifest.schema import RunManifest
from impacts.postprocessor import postprocess_from_runtime_config
from impacts.runner import run_from_input_manifest
from impacts.runner import run_from_runtime_config


def _pipeline_payload(tmp_path: Path) -> dict:
    return {
        "beam_osm_id_col": "attributeOrigId",
        "beam_length_col": "linkLength",
        "beam_osm_epsg": 26910,
        "output_epsg": 26910,
        "inmap_grid_path": str(tmp_path / "inmap_grid.parquet"),
        "inmap_grid_epsg": 26910,
        "mapping_columns": {"link_id": "linkId", "grid_id": "isrm"},
        "isrm_url": str(tmp_path / "isrm.zarr"),
        "isrm_nox_to_no2_matrix_npz_path": str(tmp_path / "matrix.npz"),
        "aermod_grid_path": str(tmp_path / "aermod_grid.parquet"),
        "aermod_grid_epsg": 26910,
        "aermod_grid_id": "aermod_id",
        "simulation_network_folder": str(tmp_path / "beam"),
        "region": "sfbay",
        "start_year": 2017,
        "county_state_fips": "06",
        "county_fips_codes": ["001", "013"],
        "activity_totals_file": str(tmp_path / "activity.parquet"),
        "activity_totals_columns": {"county": "countyfp", "year": "year"},
        "concentration_factor": 1.0,
        "iterations": 0,
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
        "runtime_config_source": str(tmp_path / "runtime.yaml"),
        "staging_dir": str(tmp_path / "workspace"),
        "input_dir": str(tmp_path / "workspace" / "staged"),
        "inputs_manifest_path": str(tmp_path / "workspace" / "inputs_manifest.yaml"),
        "maintained_execution_path": [
            "impacts.preprocessing.step3_integrate_grids",
            "impacts.runtime.step1_process_emissions",
            "impacts.runtime.step2_compute_inmap_concentrations",
        ],
        "inputs": {"runtime_config": {"path": str(tmp_path / "runtime.yaml")}},
        "pipeline": _pipeline_payload(tmp_path),
        "pilates_contract": {"stage": "terminal_postprocessing"},
        "population_inputs": {},
        "notes": [],
    }


def test_example_settings_yaml_is_current_runtime_config():
    runtime_yaml = Path(__file__).resolve().parents[1] / "examples" / "pilates" / "settings.yaml"

    config = build_runtime_config_from_runtime_yaml(runtime_yaml)

    assert config.run.region == "sfbay"
    assert config.run.scenario == "base"
    assert config.shared.geography.fips.state == "06"
    assert config.shared.geography.fips.counties[0] == "001"
    assert config.impacts.emissions.simulation_network_folder == "upstream/"
    assert config.impacts.dispersions.inmap.grid_path.endswith("isrm_polygon/isrm_polygon.shp")


def test_build_runtime_config_from_pilates_template_uses_current_overlay_shape(tmp_path: Path):
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
            ]
        ),
        encoding="utf-8",
    )

    overlay = Path(__file__).resolve().parents[1] / "src" / "impacts" / "adapters" / "pilates_settings.yaml"
    config = build_runtime_config_from_pilates(pilates_settings=pilates_settings, impacts_overlay=overlay)

    assert config.run.region == "sfbay"
    assert config.run.scenario == "base"
    assert config.run.start_year == 2017
    assert config.shared.geography.local_crs == "EPSG:26910"
    assert config.impacts.local_input_folder == "pilates/beam/production/"
    assert config.impacts.emissions.osm_network_folder.endswith("r5/sfbay-cbg5500-weakConn-network")
    assert config.impacts.dispersions.inmap.isrm_zarr == "s3://inmap-model/isrm_v1.2.1.zarr/"


def test_manifest_models_round_trip_current_shape(tmp_path: Path):
    pipeline = PipelineConfig.from_dict(_pipeline_payload(tmp_path)).to_dict()
    inputs_manifest = InputsManifest.from_dict(_inputs_manifest_payload(tmp_path)).to_dict()
    run_manifest = RunManifest.from_dict(
        {
            "contract_version": "1",
            "model": "impacts",
            "input_manifest_path": inputs_manifest["inputs_manifest_path"],
            "output_dir": str(tmp_path / "workspace"),
            "raw_output_dir": str(tmp_path / "workspace" / "outputs"),
            "command": "python -m impacts run",
            "image": "unknown",
            "raw_outputs": {"skims_emissions": str(tmp_path / "prepared.parquet")},
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
            "output_dir": str(tmp_path / "downstream"),
            "canonical_artifact": {"path": str(tmp_path / "downstream" / "impacts_exposure_table.parquet")},
            "validation": {},
            "notes": [],
            "postprocess_manifest_path": str(tmp_path / "downstream" / "postprocess_manifest.yaml"),
        }
    ).to_dict()

    assert inputs_manifest["pipeline"]["region"] == "sfbay"
    assert run_manifest["execution"]["stopped_after"] == "step1_process_emissions"
    assert postprocess_manifest["canonical_artifact"]["path"].endswith(".parquet")


def test_run_from_input_manifest_uses_current_step_name(monkeypatch, tmp_path: Path):
    import impacts.preprocessing.step3_integrate_grids as step3_integrate_grids
    import impacts.runtime.prepare_emissions_from_skims as prepare_emissions_from_skims
    import impacts.runtime.step1_process_emissions as step1_process_emissions
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
        lambda pipeline, raw_dir, input_root, grid_intersection_path, intersection_df=None: {
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


def test_run_from_runtime_config_delegates_through_preprocess(monkeypatch, tmp_path: Path):
    calls = {}

    def _fake_preprocess(runtime_config_path, staging_dir, manifest_path=None):
        calls["preprocess"] = {
            "runtime_config_path": str(runtime_config_path),
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

    result = run_from_runtime_config(
        runtime_config_path=tmp_path / "runtime.yaml",
        workspace=tmp_path / "workspace",
        run_dispersion=False,
    )

    assert result["run_manifest_path"].endswith("run_manifest.yaml")
    assert calls["preprocess"]["runtime_config_path"].endswith("runtime.yaml")
    assert calls["preprocess"]["staging_dir"].endswith("workspace")
    assert calls["run"]["input_manifest_path"].endswith("inputs_manifest.yaml")
    assert calls["run"]["output_dir"].endswith("workspace")
    assert calls["run"]["run_dispersion"] is False


def test_cli_run_accepts_runtime_config_workspace(monkeypatch, tmp_path: Path):
    calls = {}

    def _fake_run_from_runtime_config(runtime_config_path, workspace, run_manifest_path=None, run_dispersion=False):
        calls["runtime_run"] = {
            "runtime_config_path": str(runtime_config_path),
            "workspace": str(workspace),
            "run_manifest_path": run_manifest_path,
            "run_dispersion": run_dispersion,
        }
        return {"run_manifest_path": str(tmp_path / "workspace" / "run_manifest.yaml")}

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
    assert calls["runtime_run"]["run_dispersion"] is False


def test_postprocess_from_runtime_config_delegates_through_runner(monkeypatch, tmp_path: Path):
    calls = {}

    class _RuntimeConfig:
        class impacts:
            local_output_folder = "downstream"

    def _fake_runtime_run(runtime_config_path, workspace, run_dispersion=True):
        calls["runtime_run"] = {
            "runtime_config_path": str(runtime_config_path),
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
        return {"postprocess_manifest_path": str(tmp_path / "downstream" / "postprocess_manifest.yaml")}

    monkeypatch.setattr("impacts.postprocessor.build_runtime_config_from_runtime_yaml", lambda _: _RuntimeConfig())
    monkeypatch.setattr("impacts.postprocessor.resolve_path", lambda path, _: path)
    monkeypatch.setattr("impacts.runner.run_from_runtime_config", _fake_runtime_run)
    monkeypatch.setattr("impacts.postprocessor.postprocess_from_run_manifest", _fake_postprocess)

    result = postprocess_from_runtime_config(
        runtime_config_path=tmp_path / "runtime.yaml",
        workspace=tmp_path / "workspace",
    )

    assert result["postprocess_manifest_path"].endswith("postprocess_manifest.yaml")
    assert calls["runtime_run"]["runtime_config_path"].endswith("runtime.yaml")
    assert calls["runtime_run"]["workspace"].endswith("workspace")
    assert calls["runtime_run"]["run_dispersion"] is True
    assert calls["postprocess"]["run_manifest_path"].endswith("run_manifest.yaml")
    assert calls["postprocess"]["output_dir"].endswith("downstream")


def test_example_runtime_yaml_reference_is_removed():
    runtime_yaml = Path(__file__).resolve().parents[1] / "examples" / "pilates" / "runtime.yaml"
    assert not runtime_yaml.exists()


def test_current_runtime_builder_rejects_legacy_shape(tmp_path: Path):
    runtime_yaml = tmp_path / "legacy_runtime.yaml"
    runtime_yaml.write_text(
        "\n".join(
            [
                "shared:",
                "  region: sfbay",
                "inputs:",
                "  beam_network: upstream/network.csv.gz",
                "outputs:",
                "  output_dir: downstream",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported keys under shared"):
        build_runtime_config_from_runtime_yaml(runtime_yaml)
