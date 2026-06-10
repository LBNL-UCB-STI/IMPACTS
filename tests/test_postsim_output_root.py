from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace

import impacts.__main__ as impacts_main
from impacts.config.settings import presim_activities_inventory_root
from impacts.config.settings import presim_activities_manifest_path
from impacts.config.settings import presim_activities_rates_root
from impacts.config.settings import presim_activities_tmp_root
from impacts.provisioner import _expected_output_path
from impacts.runner import run_emissions_from_pipeline_manifest


def _postsim_settings(*, output_run_name: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(
            region="sfbay",
            scenario="base",
            start_year=2018,
            output_run_name=output_run_name,
        ),
        shared=SimpleNamespace(),
        beam=SimpleNamespace(),
        impacts=SimpleNamespace(
            local_output_folder="impacts_output",
            pipeline=SimpleNamespace(
                postsim=SimpleNamespace(
                    emissions=True,
                    inmap=False,
                    aermod=False,
                    exposure=False,
                )
            )
        )
    )


def test_postsim_from_settings_uses_hpc_timestamp_output_root(monkeypatch, tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("impacts:\n  local_output_folder: impacts_output\n", encoding="utf-8")
    postsim_output_root = tmp_path / "impacts_output" / "impacts-postsim--sfbay--base--20260609-123456"
    calls: dict[str, object] = {}

    def _fake_preprocess(*, settings_path, output_root_override=None, **_kwargs):
        calls["preprocess_output_root"] = output_root_override
        return {"pipeline_manifest_path": str(postsim_output_root / "pipeline_manifest.yaml")}

    def _fake_run_emissions(*, run_manifest_path):
        calls["emissions_manifest"] = run_manifest_path
        return {"pipeline_manifest_path": str(postsim_output_root / "pipeline_manifest.yaml")}

    def _fake_postprocess(*, run_manifest_path, output_root_override=None, input_roots=None, **_kwargs):
        calls["postprocess_manifest"] = run_manifest_path
        calls["postprocess_output_root"] = output_root_override
        calls["input_roots"] = tuple(input_roots or ())
        return {"postprocess_manifest_path": str(postsim_output_root / "postprocess_manifest.yaml")}

    monkeypatch.setenv("IMPACTS_POSTSIM_OUTPUT_DIR", str(postsim_output_root))
    monkeypatch.setattr(impacts_main, "load_settings_from_yaml", lambda _: _postsim_settings())
    monkeypatch.setattr("impacts.preprocessor.preprocess_workflow", _fake_preprocess)
    monkeypatch.setattr("impacts.runner.run_emissions_from_pipeline_manifest", _fake_run_emissions)
    monkeypatch.setattr("impacts.postprocessor.postprocess_from_pipeline_manifest", _fake_postprocess)
    monkeypatch.setattr(
        "impacts.config.path_registry.build_registry",
        lambda *_args, **_kwargs: SimpleNamespace(roots=(tmp_path / "input-root",)),
    )

    impacts_main._run_postsim_from_settings(str(settings_path))

    assert postsim_output_root.exists()
    assert calls["preprocess_output_root"] == postsim_output_root
    assert calls["postprocess_output_root"] == postsim_output_root
    assert calls["input_roots"] == (tmp_path / "input-root",)


def test_postsim_from_settings_creates_local_timestamp_output_root(monkeypatch, tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("impacts:\n  local_output_folder: impacts_output\n", encoding="utf-8")
    calls: dict[str, object] = {}

    def _fake_preprocess(*, settings_path, output_root_override=None, **_kwargs):
        calls["preprocess_output_root"] = output_root_override
        return {"pipeline_manifest_path": str(Path(output_root_override) / "pipeline_manifest.yaml")}

    def _fake_run_emissions(*, run_manifest_path):
        return {"pipeline_manifest_path": str(run_manifest_path)}

    def _fake_postprocess(*, run_manifest_path, output_root_override=None, input_roots=None, **_kwargs):
        calls["postprocess_output_root"] = output_root_override
        return {"postprocess_manifest_path": str(Path(output_root_override) / "postprocess_manifest.yaml")}

    monkeypatch.delenv("IMPACTS_POSTSIM_OUTPUT_DIR", raising=False)
    monkeypatch.setattr(impacts_main, "load_settings_from_yaml", lambda _: _postsim_settings())
    monkeypatch.setattr("impacts.preprocessor.preprocess_workflow", _fake_preprocess)
    monkeypatch.setattr("impacts.runner.run_emissions_from_pipeline_manifest", _fake_run_emissions)
    monkeypatch.setattr("impacts.postprocessor.postprocess_from_pipeline_manifest", _fake_postprocess)
    monkeypatch.setattr(
        "impacts.config.path_registry.build_registry",
        lambda *_args, **_kwargs: SimpleNamespace(roots=(tmp_path / "input-root",)),
    )

    impacts_main._run_postsim_from_settings(str(settings_path))

    postsim_output_root = calls["preprocess_output_root"]
    assert isinstance(postsim_output_root, Path)
    assert postsim_output_root.parent == tmp_path / "impacts_output"
    assert re.fullmatch(r"impacts-postsim--sfbay--base--\d{8}-\d{6}", postsim_output_root.name)
    assert calls["postprocess_output_root"] == postsim_output_root
    assert postsim_output_root.exists()


def test_postsim_output_root_uses_output_run_name_when_available(monkeypatch, tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("impacts:\n  local_output_folder: impacts_output\n", encoding="utf-8")

    monkeypatch.delenv("IMPACTS_POSTSIM_OUTPUT_DIR", raising=False)
    monkeypatch.setattr(
        impacts_main,
        "load_settings_from_yaml",
        lambda _: _postsim_settings(output_run_name="calibration-a"),
    )

    postsim_output_root = impacts_main._resolve_postsim_output_root(str(settings_path))

    assert postsim_output_root.parent == tmp_path / "impacts_output"
    assert re.fullmatch(r"impacts-postsim--sfbay--calibration-a--\d{8}-\d{6}", postsim_output_root.name)


def test_presim_activities_paths_are_under_presim_activities(tmp_path: Path) -> None:
    output_root = tmp_path / "impacts_output"
    scenario = "2018-Baseline"
    cfg = {
        "output_root": output_root,
        "scenario": scenario,
        "region_label": "SFBAY",
        "run_region": "sfbay",
        "run_scenario": "base",
        "output_run_name": None,
    }

    presim_kwargs = {
        "region": "sfbay",
        "output_run_name": None,
        "run_scenario": "base",
    }

    assert presim_activities_manifest_path(output_root, **presim_kwargs) == (
        output_root / "impacts_presim--sfbay--base" / "activities" / "activities_manifest.yaml"
    )
    assert presim_activities_inventory_root(output_root, scenario, **presim_kwargs) == (
        output_root / "impacts_presim--sfbay--base" / "activities" / scenario / "inventory"
    )
    assert presim_activities_rates_root(output_root, scenario, **presim_kwargs) == (
        output_root / "impacts_presim--sfbay--base" / "activities" / scenario / "rates"
    )
    assert presim_activities_tmp_root(output_root, **presim_kwargs) == (
        output_root / "impacts_presim--sfbay--base" / "activities" / "_tmp"
    )
    assert _expected_output_path(cfg, 2018) == (
        output_root
        / "impacts_presim--sfbay--base"
        / "activities"
        / scenario
        / "inventory"
        / "sfbay-emfac-2018-inventory-final-passenger-activity-by-emfacid.parquet"
    )


def _pipeline_payload() -> dict[str, object]:
    return {
        "emissions_enabled": True,
        "beam_osm_id_col": "attributeOrigId",
        "beam_length_col": "linkLength",
        "output_epsg": 26910,
        "mapping_columns": {},
        "inmap_enabled": False,
        "aermod_enabled": False,
        "exposure_enabled": False,
        "region": "sfbay",
        "start_year": 2018,
        "county_state_fips": "06",
        "county_fips_codes": ["001"],
        "passenger_inventory_file": None,
        "freight_inventory_file": None,
        "enable_passenger_inventory_activity_correction": False,
        "enable_freight_inventory_activity_correction": False,
        "passenger_vehicle_types_file": "passenger.csv",
        "freight_vehicle_types_file": "freight.csv",
        "vehicle_category_metadata_file": None,
        "prepared_skims_group_cols": ["linkId", "vehicleTypeId", "process"],
        "pollutants": ["NOx"],
        "annualization_days": {"light_duty": 327.0, "medium_heavy_duty": 312.0},
        "population_sample": 1.0,
        "transit_sample": 1.0,
        "freight_sample": None,
        "include_non_osm_car_links": False,
        "include_passenger": True,
        "include_freight": True,
    }


def test_runner_uses_pipeline_manifest_output_dir_for_stage_outputs(monkeypatch, tmp_path: Path) -> None:
    base_output = tmp_path / "impacts_output"
    postsim_output = base_output / "impacts-postsim--sfbay--base--20260609-123456"
    preprocess_manifest_path = postsim_output / "preprocess_manifest.yaml"
    run_manifest_path = postsim_output / "pipeline_manifest.yaml"
    county_intersection = postsim_output / "preprocess" / "county.parquet"
    settings_path = tmp_path / "settings.yaml"
    run_manifest_path.parent.mkdir(parents=True)
    county_intersection.parent.mkdir(parents=True)
    run_manifest_path.write_text("{}", encoding="utf-8")
    county_intersection.write_text("", encoding="utf-8")
    settings_path.write_text("impacts:\n  local_output_folder: impacts_output\n", encoding="utf-8")
    pipeline = _pipeline_payload()
    preprocess_manifest = {
        "contract_version": "1",
        "model": "impacts",
        "settings_source": str(settings_path),
        "staging_dir": str(postsim_output / "preprocess"),
        "input_dir": str(postsim_output / "preprocess"),
        "preprocess_manifest_path": str(preprocess_manifest_path),
        "maintained_execution_path": [],
        "inputs": {
            "county_intersection": {
                "kind": "local",
                "path": str(county_intersection),
                "staged_path": str(county_intersection),
                "optional": False,
                "exists": True,
            }
        },
        "pipeline": pipeline,
        "pilates_contract": {},
        "population_inputs": {},
        "notes": [],
    }
    run_manifest = {
        "contract_version": "1",
        "model": "impacts",
        "preprocess_manifest_path": str(preprocess_manifest_path),
        "output_dir": str(postsim_output),
        "command": "python -m impacts preprocess",
        "image": "not_recorded",
        "outputs": {"skims_emissions": None},
        "pipeline": pipeline,
        "population_inputs": {},
        "deterministic_contract": {},
        "execution": {"dispersion_completed": False, "stopped_after": "preprocess"},
        "pipeline_manifest_path": str(run_manifest_path),
    }
    calls: dict[str, Path] = {}

    def _fake_load_structured_file(path):
        resolved = Path(path).resolve()
        if resolved == run_manifest_path.resolve():
            return run_manifest
        if resolved == preprocess_manifest_path.resolve():
            return preprocess_manifest
        raise AssertionError(f"unexpected structured file: {path}")

    def _fake_run_emissions(_pipeline, raw_dir, output_dir, _grid_intersections, **_kwargs):
        calls["raw_dir"] = Path(raw_dir)
        calls["output_dir"] = Path(output_dir)
        return {
            "beam_emissions_by_county_process": str(Path(output_dir) / "county.parquet"),
            "beam_emissions_for_inmap": None,
            "beam_inmap_study_area_grid": None,
            "beam_emissions_for_aermod": None,
        }

    monkeypatch.setattr("impacts.runner.load_structured_file", _fake_load_structured_file)
    monkeypatch.setattr("impacts.pipeline.workflow.step1_process_emissions.run", _fake_run_emissions)

    result = run_emissions_from_pipeline_manifest(run_manifest_path=run_manifest_path)

    assert result["output_dir"] == str(postsim_output.resolve())
    assert calls["raw_dir"] == postsim_output / "emissions"
    assert calls["output_dir"] == postsim_output / "emissions"
