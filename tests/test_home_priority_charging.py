import numpy as np
import pytest
import pandas as pd

from impacts.pipeline.workflow.charging.ldv_home_sensitivity import Schedule,simulate_day,run_periodic,uniform,build_schedules


def make(kind='BEV',capacity=10.,initial=6.,energy=(3.,4.,1.),dwell=(7200.,3600.,43200.)):
    return Schedule('v',kind,capacity,initial,np.array([7.,10.,17.])*3600,
        np.array([8.,11.,18.])*3600,np.array(dwell),np.array(energy),
        np.array([False,False,True]),np.array(['Work/Education','Other','Home']),2,True,7.2)


def test_bev_topup_before_home_prevents_shortfall():
    r=simulate_day(make(),6.,capture=True)
    assert r['shortfall']==pytest.approx(0)
    assert r['sessions'][0]['reason']=='necessary_nonhome_topup'
    assert r['sessions'][0]['energyCharged_kWh']==pytest.approx(4)
    assert r['sessions'][-1]['reason']=='primary_home'
    assert r['final']==pytest.approx(10)
    assert r['charge']==pytest.approx(r['traction']+r['final']-6)


def test_phev_no_dc_and_explicit_fuel_substitution():
    r=simulate_day(make(kind='PHEV'),6.,capture=True)
    assert r['shortfall']==pytest.approx(4)
    assert all(x['power_kW']!=50 for x in r['sessions'])
    assert sum(x['PHEV_fuel_substitution_electric_equiv_kWh'] for x in r['trips'])==pytest.approx(4)
    assert sum(x['BEV_unmet_kWh'] for x in r['trips'])==0


def test_cannot_charge_after_arrival_to_erase_failed_trip():
    r=run_periodic([make(energy=(12.,1.,1.))])
    assert not r['vehicles'].steady_included.iloc[0]
    assert r['vehicles'].steady_shortfall_kWh.iloc[0]>0
    assert r['profiles']['steady_feasible'].sum()==0


def test_repeat_day_balance_and_energy_not_initial_soc_gain():
    r=run_periodic([make()])
    v=r['vehicles'].iloc[0]
    assert v.converged
    assert v.steady_SOC_gain_kWh==pytest.approx(0)
    assert v.steady_charge_kWh==pytest.approx(v.steady_traction_kWh)
    assert v.first_SOC_gain_kWh>0
    assert r['profiles']['steady_feasible'].sum()==pytest.approx(v.steady_charge_kWh)


def test_long_dwell_no_overcharge_and_stop_limit():
    r=simulate_day(make(),6.,capture=True)
    for row in r['sessions']:
        assert row['timeCharging_s']<=row['parkingTime_s']
        assert row['soc_after_kWh']<=10
    s=make(dwell=(600.,600.,600.))
    assert simulate_day(s,6.,capture=True)['sessions']==[]


def test_deterministic_vehicle_draws():
    assert uniform('abc',42,'ac')==uniform('abc',42,'ac')
    assert uniform('abc',42,'ac')!=uniform('abc',42,'preference')


@pytest.mark.parametrize('bad',[0,-1,np.nan,np.inf])
def test_invalid_base_energy_coefficient(bad):
    with pytest.raises(ValueError,match='base_kWh_per_km'):
        build_schedules(None,None,1.,base_kWh_per_km=bad)


@pytest.mark.parametrize('kind,factor',[('BEV',1.),('PHEV',.2)])
def test_energy_coefficient_changes_traction_not_capacity_or_preferences(kind,factor):
    trips=pd.DataFrame([dict(vehicle='v',departureTime=7*3600.,arrivalTime=8*3600.,length=10000.)])
    audit=trips.assign(is_home=True,groupedActivityType='Home',powertrain=kind,
                       battery_kWh=16.,initialsoc_kWh=8.,energyConsumption_kWh=1.7*factor)
    old=build_schedules(trips,audit,1.)[0]
    new=build_schedules(trips,audit,1.,base_kWh_per_km=.224)[0]
    assert old.energy.sum()==pytest.approx(1.7*factor)
    assert new.energy.sum()==pytest.approx(2.24*factor)
    assert new.capacity==old.capacity
    assert new.ac_power==old.ac_power
    assert new.initial==old.initial
    assert new.primary==old.primary
