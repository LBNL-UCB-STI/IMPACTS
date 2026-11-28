import pytest

from impacts.__main__ import main
from impacts.config.settings_builder import _deep_merge
from impacts.config.settings_builder import build_settings_from_pilates
from impacts.manifest.file_ops import load_structured_file, write_structured_file


def test_deep_merge_scalar_override():
    result = _deep_merge({"a": 1, "b": 2}, {"b": 99})
    assert result == {"a": 1, "b": 99}


def test_deep_merge_nested_dict():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 99, "z": 100}}
    result = _deep_merge(base, override)
    assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}


def test_deep_merge_list_override():
    result = _deep_merge({"a": [1, 2]}, {"a": [3, 4]})
    assert result["a"] == [3, 4]


def test_deep_merge_empty_override():
    base = {"a": {"x": 1}}
    assert _deep_merge(base, {}) == base


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"x": 1}}
    _deep_merge(base, {"a": {"x": 99}})
    assert base["a"]["x"] == 1


_MINIMAL_BRIDGE = {
    "run": {"region": "sfbay", "scenario": "test", "start_year": 2030},
    "shared": {"geography": {"FIPS": {"state": "06", "counties": ["001"]}, "local_crs": 26910}},
    "beam": {"local_input_folder": "/beam/input", "local_output_folder": "/beam/output"},
}


def test_impacts_override_passenger_folder():
    bridge = {
        **_MINIMAL_BRIDGE,
        "impacts": {"population": {"passenger_folder": "/some/abs/path"}},
    }
    settings = build_settings_from_pilates(bridge)
    assert settings.impacts.population.passenger_folder == "/some/abs/path"


def test_impacts_override_postsim_inmap():
    bridge = {
        **_MINIMAL_BRIDGE,
        "impacts": {"pipeline": {"postsim": {"inmap": False}}},
    }
    settings = build_settings_from_pilates(bridge)
    assert settings.impacts.pipeline.postsim.inmap is False


def test_impacts_override_does_not_clobber_other_defaults():
    bridge = {
        **_MINIMAL_BRIDGE,
        "impacts": {"population": {"passenger_folder": "/some/abs/path"}},
    }
    settings = build_settings_from_pilates(bridge)
    assert settings.impacts.pipeline.postsim.emissions is True
    assert "presim" not in settings.to_dict()["impacts"]["pipeline"]


def test_presim_does_not_accept_independent_activity_or_fleet_switches():
    bridge = {
        **_MINIMAL_BRIDGE,
        "impacts": {"pipeline": {"presim": {"activities": False, "fleet": False}}},
    }
    with pytest.raises(ValueError, match="Unsupported keys under impacts.pipeline"):
        build_settings_from_pilates(bridge)


def test_presim_opens_relocated_catalog_after_bridge_round_trip(tmp_path, monkeypatch):
    from impacts import provisioner
    from impacts.pipeline.emfac.activities.step1_prepare_emissions_and_activities_tables import (
        _load_operation_days_lookup,
    )

    beam_root = tmp_path / "beam_input"
    relative_path = "impacts/emissions/emissions_vehicle_categories.csv"
    catalog = beam_root / "sfbay" / relative_path
    catalog.parent.mkdir(parents=True)
    catalog.write_text("emfac_vehicle_category,operation_days_per_year\nLDA,327\n", encoding="utf-8")
    (catalog.parent / "emissions_raw.tar.zst").touch()
    config_path = tmp_path / "settings.yaml"
    bridge_path = tmp_path / "bridge.yaml"
    write_structured_file(bridge_path, {
        **_MINIMAL_BRIDGE,
        "beam": {**_MINIMAL_BRIDGE["beam"], "local_input_folder": str(beam_root)},
        "impacts": {
            "local_input_folder": str(tmp_path / "input"),
            "local_output_folder": str(tmp_path / "output"),
            "pipeline": {"presim": {"activities": True, "fleet": True}},
            "emissions": {"vehicle_category_metadata_file": relative_path},
            "exposure": {"census_bundle": "impacts/census/fixture.tar.gz"},
        },
    })
    assert main([
        "derive_settings_from_pilates", "--pilates-settings", str(bridge_path), "--output", str(config_path)
    ]) == 0
    payload = load_structured_file(config_path)["impacts"]
    assert payload["emissions"]["vehicle_category_metadata_file"] == relative_path
    assert payload["exposure"]["census_bundle"] == "impacts/census/fixture.tar.gz"
    assert "presim" not in payload["pipeline"]
    assert not (beam_root / "sfbay/vehicle-tech/emissions/emissions_vehicle_categories.csv").exists()

    calls = []
    def ensure_activities(settings, path):
        inputs = provisioner._resolve_activities_config(settings, path)
        calls.append(_load_operation_days_lookup(str(inputs["vehicle_category_metadata_file"])))
        return {"activities_manifest_path": "fixture-manifest.yaml"}

    monkeypatch.setattr(provisioner, "ensure_emfac_activities_outputs", ensure_activities)
    monkeypatch.setattr(provisioner, "run_fleet", lambda **_: None)
    assert main(["presim", "--config", str(config_path)]) == 0
    assert calls == [{"LDA": 327.0}]


def test_scenario_defaults_to_start_year_when_not_supplied():
    settings = build_settings_from_pilates(_MINIMAL_BRIDGE)
    assert settings.impacts.scenario == "2030-Baseline"


def test_scenario_override_wins_over_start_year():
    bridge = {
        **_MINIMAL_BRIDGE,
        "impacts": {"scenario": "custom"},
    }
    settings = build_settings_from_pilates(bridge)
    assert settings.impacts.scenario == "custom"


def test_derive_settings_preserves_relocated_beam_input_paths():
    overrides = {
        "local_input_folder": "/app/workspace/impacts/input",
        "local_output_folder": "/app/workspace/impacts/output",
        "shared_input_cache_dir": "/app/impacts_cache",
        "population": {"vehicle_folder": "vehicle-tech"},
        "activities": {"emissions_raw_archive": "impacts/emissions/emissions_raw.tar.zst"},
        "fleet": {"assignment_model": "impacts/emissions/fleet_assignment.yaml"},
        "emissions": {
            "vehicle_category_metadata_file": "impacts/emissions/emissions_vehicle_categories.csv"
        },
        "dispersions": {
            "aermod": {"asrv_patterns_file": "impacts/dispersion/aermod/aermod_patterns_wgs84.parquet"},
            "inmap": {
                "grid_path": "impacts/dispersion/inmap/isrm_polygon_wgs84.gpkg",
                "isrm_nox_to_no2_ratios_file": "impacts/dispersion/inmap/nox_to_no2_full_isrm_matrix.npz",
                "isrm_zarr": "/app/inmap/isrm_v1.2.1.zarr",
            },
        },
        "exposure": {"census_bundle": "impacts/census/breathe-sfbay-2018-v2.tar.gz"},
    }
    settings = build_settings_from_pilates({
        **_MINIMAL_BRIDGE,
        "beam": {"local_input_folder": "/app/beam_input", "local_output_folder": "/app/impacts_beam_output"},
        "impacts": overrides,
    })
    derived = settings.to_dict()

    assert derived["beam"]["local_input_folder"] == "/app/beam_input"
    assert derived["impacts"]["shared_input_cache_dir"] == "/app/impacts_cache"
    assert derived["impacts"]["population"]["vehicle_folder"] == "vehicle-tech"
    assert derived["impacts"]["activities"]["emissions_raw_archive"] == overrides["activities"]["emissions_raw_archive"]
    assert derived["impacts"]["fleet"]["assignment_model"] == overrides["fleet"]["assignment_model"]
    assert derived["impacts"]["emissions"]["vehicle_category_metadata_file"] == overrides["emissions"]["vehicle_category_metadata_file"]
    assert derived["impacts"]["dispersions"]["aermod"]["asrv_patterns_file"] == overrides["dispersions"]["aermod"]["asrv_patterns_file"]
    assert derived["impacts"]["dispersions"]["inmap"]["grid_path"] == overrides["dispersions"]["inmap"]["grid_path"]
    assert derived["impacts"]["dispersions"]["inmap"]["isrm_nox_to_no2_ratios_file"] == overrides["dispersions"]["inmap"]["isrm_nox_to_no2_ratios_file"]
    assert derived["impacts"]["dispersions"]["inmap"]["isrm_zarr"] == "/app/inmap/isrm_v1.2.1.zarr"
    assert derived["impacts"]["exposure"]["census_bundle"] == overrides["exposure"]["census_bundle"]
