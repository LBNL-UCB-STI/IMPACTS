import numpy as np
import pandas as pd
import pytest

from impacts.pipeline.workflow.cristian_charging import (
    expand_stop_bins, load_original, pandas2_sampling, run_cristian_charging,
)
from impacts.pipeline.workflow.notebook_charging import assign_stops_indexed, fill_stops


def stops():
    rows = []
    for v in range(40):
        for i, (dep, arr, dwell) in enumerate([(28000., 28800., 32400.), (61200., 62000., 52400.)]):
            rows.append(dict(vehicle=f'v{v:03}', PTID=2*v+i, battery_kWh=16. if v%2 else 60.,
                length=10000., dailyVehicleLength_m=20000., cumulativeTravelLength_m=(i+1)*10000.,
                departureTime=dep, arrivalTime=arr, parkingTime_s=dwell,
                groupedActivityType='Home' if i else 'Work/Education', is_home=bool(i),
                lev1Chargers_cnt=1., lev2Chargers_cnt=1., fastChargers_cnt=1., netEnergyLoadGrid_MW=0.,
                energyConsumption_kWh=.17 if v%2 else 1.7))
    return pd.DataFrame(rows)


def test_direct_original_matches_previously_verified_adapter(charging_notebooks):
    activity = stops()
    original, params, _ = load_original(charging_notebooks)
    assigned = assign_stops_indexed(activity, original)
    expected, audit, curve = fill_stops(assigned)
    actual, vehicles, hourly, settings, provenance = run_cristian_charging(
        activity, notebooks=charging_notebooks, input_layout='stops')
    by_stop = actual.groupby('PTID').agg(energyCharged_kWh=('energyCharged_kWh','sum'),
        timeCharging_s=('timeCharging_s','sum'), socAfterCharging_kWh=('socAfterCharging_kWh','last'),
        attemptCharge=('attemptCharge','first'), chargerPower_kW=('chargerPower_kW','first'))
    for col in by_stop:
        np.testing.assert_allclose(by_stop[col], expected.set_index('PTID').sort_index()[col], atol=1e-9)
    np.testing.assert_array_equal(vehicles.retained, audit.retained)
    np.testing.assert_allclose(hourly.total_sample_kWh, curve, atol=1e-9)
    assert settings['HYBRID_ELECTRICITY_CONSUMPTION_FACTOR'] == .1
    assert settings['AVG_E_CONS_kWh_KM'] == .17
    assert provenance['AssignChargingEvents.ipynb']['sha256']
    assert not any('fuel' in c for c in actual.columns)


def test_expansion_conserves_distance_and_parking():
    original = stops()
    bins = expand_stop_bins(original)
    for col in ['length', 'parkingTime_s']:
        np.testing.assert_allclose(bins.groupby('PTID')[col].sum(), original.set_index('PTID')[col])
    assert bins.parkingTime_s.le(3600).all()
    assert bins.groupby('PTID').cumulativeTravelLength_m.nunique().eq(1).all()


def test_no_stale_energy_proxy_or_home_flags_are_reused(charging_notebooks):
    activity = stops()
    activity['energyConsumption_kWh'] = 999
    activity['is_home'] = ~activity.is_home
    result, vehicles, _, _, _ = run_cristian_charging(activity, notebooks=charging_notebooks, input_layout='stops')
    phev = vehicles.powertrain_proxy.eq('PHEV')
    np.testing.assert_allclose(vehicles.loc[phev, 'traction_kWh'], .34)
    np.testing.assert_allclose(vehicles.loc[~phev, 'traction_kWh'], 3.4)
    assert result.is_home.equals(result.groupedActivityType.eq('Home'))


def test_sampling_compatibility_restores_state_after_failure():
    method = pd.DataFrame.sample
    state = np.random.get_state()
    with pytest.raises(RuntimeError):
        with pandas2_sampling():
            np.random.seed(12)
            raise RuntimeError('test')
    assert pd.DataFrame.sample is method
    actual = np.random.get_state()
    assert actual[0] == state[0]
    np.testing.assert_array_equal(actual[1], state[1])
    assert actual[2:] == state[2:]


def test_original_settings_are_read_from_notebook_copy(tmp_path, charging_notebooks):
    import shutil
    for name in ['AssignChargingEvents.ipynb', 'GenerateVehicleActivity.ipynb']:
        shutil.copy2(charging_notebooks / name, tmp_path / name)
    p = tmp_path / 'AssignChargingEvents.ipynb'
    text = p.read_text(encoding='utf-8')
    assert 'HYBRID_ELECTRICITY_CONSUMPTION_FACTOR = 0.1' in text
    p.write_text(text.replace('HYBRID_ELECTRICITY_CONSUMPTION_FACTOR = 0.1',
                              'HYBRID_ELECTRICITY_CONSUMPTION_FACTOR = 0.15'), encoding='utf-8')
    _, settings, _ = load_original(tmp_path)
    assert settings['HYBRID_ELECTRICITY_CONSUMPTION_FACTOR'] == .15
