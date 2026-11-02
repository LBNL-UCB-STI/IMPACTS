import numpy as np
import pytest

from impacts.pipeline.workflow.home_priority_charging import Schedule, simulate_day, run_periodic


def schedule(dwell=12*3600., energy=10., capacity=32.):
    return Schedule('bev', 'BEV', capacity, capacity, np.array([7.*3600]),
        np.array([18.*3600]), np.array([dwell]), np.array([energy]),
        np.array([True]), np.array(['Home']), 0, True, 7.2)


def test_separate_grid_battery_ledgers_and_duration():
    s=schedule()
    r=simulate_day(s, s.initial, capture=True, efficiency=.9)
    assert r['charge']==pytest.approx(10.)
    assert r['grid_charge']==pytest.approx(10/.9)
    assert r['profile'].sum()==pytest.approx(10/.9)
    assert r['battery_profile'].sum()==pytest.approx(10)
    row=r['sessions'][0]
    assert row['timeCharging_s']==pytest.approx(10/.9/7.2*3600)
    assert row['soc_after_kWh']-row['soc_before_kWh']==pytest.approx(10)
    assert row['charging_loss_kWh']==pytest.approx(10/.9-10)


def test_efficiency_limits_battery_charge_during_short_stop():
    s=schedule(dwell=1800)
    r=simulate_day(s,s.initial,capture=True,efficiency=.9)
    assert r['charge']==pytest.approx(7.2*.5*.9)
    assert r['grid_charge']==pytest.approx(7.2*.5)
    assert r['sessions'][0]['timeCharging_s']==pytest.approx(1800)


def test_periodic_grid_energy_is_not_soc_gain():
    r=run_periodic([schedule()],efficiency=.9)
    v=r['vehicles'].iloc[0]
    assert v.steady_SOC_gain_kWh==pytest.approx(0)
    assert v.steady_grid_charge_kWh*.9==pytest.approx(v.steady_traction_kWh)
    assert r['profiles']['steady_BEV'].sum()==pytest.approx(v.steady_grid_charge_kWh)


def test_infeasible_bev_not_hidden_by_losses_or_late_charging():
    r=run_periodic([schedule(energy=40)],efficiency=.9)
    assert not r['vehicles'].steady_included.iloc[0]
    assert r['profiles']['steady_BEV'].sum()==0


@pytest.mark.parametrize('efficiency',[0,-.1,1.1,np.nan,np.inf])
def test_invalid_efficiency(efficiency):
    with pytest.raises(ValueError,match='efficiency'):
        simulate_day(schedule(),32,efficiency=efficiency)


def test_legacy_default_is_preserved():
    a=run_periodic([schedule()])
    b=run_periodic([schedule()],efficiency=1.)
    assert a['vehicles'].equals(b['vehicles'])
    np.testing.assert_array_equal(a['profiles']['steady_BEV'],b['profiles']['steady_BEV'])
