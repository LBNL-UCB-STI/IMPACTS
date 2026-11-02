from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from impacts.pipeline.workflow.charging.ldv_home_sensitivity import (
    Schedule,
    build_phev_preference_schedules,
    simulate_phev_day,
    run_phev_periodic,
)


def schedule(primary=0):
    return Schedule('p','PHEV',16.,16.,np.array([7.,17.])*3600,
        np.array([8.,18.])*3600,np.array([9.,13.])*3600,np.array([10.,10.]),
        np.array([False,True]),np.array(['Work/Education','Home']),primary,primary==1,7.2)


def test_work_preference_charges_only_work_on_ac_and_keeps_fuel_fallback():
    r=run_phev_periodic([schedule()],charging_policy='primary_preference')
    v=r['vehicles'].iloc[0]
    assert r['sessions'].trip_index.tolist()==[0]
    assert not r['sessions'].is_home.any()
    assert r['sessions'].grid_power_kW.max()==7.2
    assert v.steady_electric_distance_km==pytest.approx(14/.202)
    assert v.steady_fuel_distance_km==pytest.approx(6/.202)
    assert v.steady_grid_charge_kWh==pytest.approx(14/.9)


def test_home_preference_does_not_enable_nonhome_charging():
    r=run_phev_periodic([schedule(1)],charging_policy='primary_preference')
    assert r['sessions'].trip_index.tolist()==[1]
    assert r['sessions'].is_home.all()


def test_no_primary_means_fuel_only_at_equilibrium():
    r=run_phev_periodic([schedule(-1)],charging_policy='primary_preference')
    assert r['vehicles'].steady_grid_charge_kWh.iloc[0]==0
    assert r['vehicles'].steady_fuel_distance_km.iloc[0]==pytest.approx(20/.202)


def test_home_only_default_still_charges_home_despite_work_primary():
    r=simulate_phev_day(schedule(),16.,capture=True)
    assert [x['trip_index'] for x in r['sessions']]==[1]


def test_home80_builder_assigns_preferences_and_removes_proxy():
    rows=[]
    for i in range(10):
        rows.extend([dict(vehicle=f'p{i}',departureTime=7*3600.,arrivalTime=8*3600.,length=10000.,
                          is_home=False,groupedActivityType='Work/Education'),
                     dict(vehicle=f'p{i}',departureTime=17*3600.,arrivalTime=18*3600.,length=10000.,
                          is_home=True,groupedActivityType='Home')])
    audit=pd.DataFrame(rows).assign(powertrain='PHEV',battery_kWh=16.,initialsoc_kWh=8.,energyConsumption_kWh=.34)
    trips=audit[['vehicle','departureTime','arrivalTime','length']]
    schedules=build_phev_preference_schedules(trips,audit,.8)
    assert sum(s.home_preferred for s in schedules)==8
    assert sum(s.primary==0 for s in schedules)==2
    for s in schedules:
        assert s.energy.sum()==pytest.approx(20*.202)
        assert s.capacity==16
    again=build_phev_preference_schedules(trips,audit,.8)
    assert [s.primary for s in schedules]==[s.primary for s in again]


def test_dwell_bound_and_battery_grid_balance():
    s=replace(schedule(),dwell=np.array([1800.,13*3600.]))
    r=simulate_phev_day(s,16.,capture=True,charging_policy='primary_preference')
    assert r['grid_charge_kWh']==pytest.approx(3.6)
    assert r['battery_charge_kWh']==pytest.approx(3.24)
    assert r['sessions'][0]['charging_time_s']==pytest.approx(1800)


def test_bad_policy_raises():
    with pytest.raises(ValueError,match='policy'):
        simulate_phev_day(schedule(),16.,charging_policy='bad')
