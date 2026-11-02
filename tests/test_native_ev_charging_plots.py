"""Baseline plots must not double-scale or import hypothetical truck charging."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from plot_native_ev_ldv_truck_evi import build_curves


def fixture():
    source = pd.DataFrame(dict(hour_of_day=range(24),
        charging_demand_ev_L1_test_MWh=np.ones(24), charging_demand_MWh=np.ones(24)))
    summary = dict(electric_road_vehicle_counts={'ev-L1-test': 100, 'phev-L1-test': 500},
        fleet_scale_factor=10, daily_energy_MWh=24, truck_charging_sessions=0, phev_charging_sessions=0)
    evi = dict(inputs=dict(fleet_size=410000, pev_dist='BEV'), response=dict(results={
        f'{day}_load_profile': {str(i): [410.] * 96 for i in range(6)}
        for day in ('weekday', 'weekend')}))
    return source, summary, evi


def test_normalization_and_zero_trucks():
    curves, sample, expanded = build_curves(*fixture())
    assert sample == 100 and expanded == 1000
    np.testing.assert_allclose(curves.LDV_native_kW_per_1000_BEV, 1000)
    np.testing.assert_allclose(curves.EVI_weekday_kW_per_1000_BEV, 6)
    np.testing.assert_allclose(curves.EVI_weekday_MW_at_baseline_BEV_count, .006)
    assert curves.filter(regex='[LMH]DT_native').to_numpy().sum() == 0


def test_reject_new_truck_charging_in_zero_baseline():
    source, summary, evi = fixture()
    summary['truck_charging_sessions'] = 1
    with pytest.raises(ValueError, match='truck/PHEV'):
        build_curves(source, summary, evi)


def test_reject_unaccounted_load():
    source, summary, evi = fixture()
    source['charging_demand_MWh'] = 2
    with pytest.raises(AssertionError):
        build_curves(source, summary, evi)
