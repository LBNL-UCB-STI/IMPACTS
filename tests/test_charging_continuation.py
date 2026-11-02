"""Geographic filtering must not change the original dispatch allocation."""
import importlib.util
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box


def test_incomplete_charging_requires_explicit_scenario_flag(tmp_path, monkeypatch):
    demand = tmp_path / 'charging'
    demand.mkdir()
    (demand / 'beam_charging_summary.json').write_text(json.dumps({'ready_for_full_electrification_pipeline': False}))
    target = tmp_path / 'results'
    monkeypatch.setattr(continuation, 'DEMAND', demand)
    monkeypatch.setattr(continuation.sys, 'argv', ['continue', '--stage', 'dispatch',
                                                '--demand-dir', str(demand), '--output-dir', str(target)])
    with pytest.raises(SystemExit) as error:
        continuation.main()
    assert error.value.code == 2
    assert not target.exists()


def test_incomplete_charging_permission_recorded(tmp_path, monkeypatch):
    demand = tmp_path / 'charging'
    demand.mkdir()
    (demand / 'beam_charging_summary.json').write_text(json.dumps({'ready_for_full_electrification_pipeline': False}))
    target = tmp_path / 'results'
    monkeypatch.setattr(continuation, 'DEMAND', demand)
    monkeypatch.setattr(continuation, 'dispatch', lambda out: None)
    monkeypatch.setattr(continuation.sys, 'argv', ['continue', '--stage', 'dispatch',
                                                '--demand-dir', str(demand), '--output-dir', str(target),
                                                '--allow-incomplete-charging-scenario'])
    continuation.main()
    status = json.loads((target / 'continuation_status.json').read_text())
    assert status['incomplete_charging_scenario_explicitly_allowed'] is True
    assert status['charging_input'] == str(demand.resolve())
    assert status['dispatch'] == 'completed'

spec = importlib.util.spec_from_file_location(
    'charging_continuation', Path(__file__).parents[1] / 'scripts/continue_beam_charging_pipeline.py'
)
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


def counties():
    return gpd.GeoDataFrame(
        {'STATEFP': ['06'] * 9, 'COUNTYFP': sorted(continuation.BAY_COUNTY_FIPS)},
        geometry=[box(i, 0, i + 1, 1) for i in range(9)], crs=4326,
    )


def test_filter_keeps_dispatch_amounts_and_boundary_plants():
    plants = pd.DataFrame({'LAT': [.5, .5, .5], 'LON': [.5, 9., 12.],
                           'added_generation_MWh': [10., 20., 70.], 'NOx_kg': [1., 2., 7.]})
    selected = continuation.select_sfbay_plants(plants, counties())
    pd.testing.assert_frame_equal(selected, plants.iloc[:2])
    assert selected.added_generation_MWh.sum() == 30.  # Not reallocated to 100.


def test_filter_handles_projected_counties():
    plants = pd.DataFrame({'LAT': [.5, .5], 'LON': [8.5, 12.]})
    pd.testing.assert_frame_equal(
        continuation.select_sfbay_plants(plants, counties().to_crs(3857)), plants.iloc[:1]
    )


def test_filter_requires_all_nine_counties():
    with pytest.raises(ValueError, match='nine'):
        continuation.select_sfbay_plants(pd.DataFrame({'LAT': [0.], 'LON': [0.]}), counties().iloc[:8])


def test_filter_rejects_invalid_coordinates():
    with pytest.raises(ValueError, match='finite'):
        continuation.select_sfbay_plants(pd.DataFrame({'LAT': [float('nan')], 'LON': [0.]}), counties())
