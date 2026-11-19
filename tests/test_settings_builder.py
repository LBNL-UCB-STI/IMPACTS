from impacts.config.settings_builder import _deep_merge
from impacts.config.settings_builder import build_settings_from_pilates


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
    assert settings.impacts.pipeline.presim.fleet is True


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
