from dataclasses import replace
import numpy as np
import pytest
from impacts.pipeline.workflow.home_priority_charging import Schedule
from impacts.pipeline.workflow.phev_home_charging import simulate_phev_day,run_phev_periodic


def schedule(home=(False,False,True), energy=(6.,6.,6.), dwell=(7200.,7200.,43200.)):
    return Schedule('p','PHEV',16.,16.,np.array([7.,11.,17.])*3600,
        np.array([8.,12.,18.])*3600,np.array(dwell),np.array(energy),np.array(home),
        np.array(['Other','Work','Home']),2,True,7.2)


def test_home_only_and_soc_fuel_split():
    r=simulate_phev_day(schedule(),16.,capture=True)
    assert len(r['sessions'])==1
    assert r['sessions'][0]['trip_index']==2
    assert r['battery_traction_kWh']==pytest.approx(14)
    assert r['fuel_distance_km']==pytest.approx(4/.202)
    assert r['grid_charge_kWh']==pytest.approx(14/.9)
    assert r['final_kWh']==pytest.approx(16)


def test_efficiency_limits_dwell_and_separates_ledgers():
    s=schedule(energy=(0.,0.,10.),dwell=(0.,0.,3600.))
    r=simulate_phev_day(s,16.,capture=True)
    assert r['grid_charge_kWh']==pytest.approx(7.2)
    assert r['battery_charge_kWh']==pytest.approx(6.48)
    assert r['charging_loss_kWh']==pytest.approx(.72)
    assert r['sessions'][0]['charging_time_s']==pytest.approx(3600)
    assert r['final_kWh']==pytest.approx(12.48)


def test_no_home_stays_in_fleet_with_fuel_only_at_equilibrium():
    r=run_phev_periodic([schedule(home=(False,False,False))])
    v=r['vehicles'].iloc[0]
    assert not v.has_eligible_home
    assert v.steady_grid_charge_kWh==0
    assert v.steady_electric_distance_km==0
    assert v.steady_fuel_distance_km==pytest.approx(18/.202)
    assert v.steady_final_kWh==pytest.approx(2)


def test_any_home_return_can_charge_not_just_last():
    r=simulate_phev_day(schedule(home=(True,False,True)),16.,capture=True)
    assert [s['trip_index'] for s in r['sessions']]==[0,2]
    assert r['fuel_distance_km']==0


def test_daily_soc_carryover_and_energy_balance():
    s=replace(schedule(),initial=7.)
    r=run_phev_periodic([s])
    v=r['vehicles'].iloc[0]
    assert v.first_SOC_gain_kWh>0
    assert v.steady_SOC_gain_kWh==pytest.approx(0)
    assert v.steady_grid_charge_kWh*.9==pytest.approx(v.steady_battery_traction_kWh)
    assert v.steady_total_distance_km==pytest.approx(v.steady_electric_distance_km+v.steady_fuel_distance_km)


def test_35_miles_not_arbitrarily_reduced_to_twenty_percent():
    s=schedule(energy=(0.,0.,35*1.609344*.202))
    r=simulate_phev_day(s,16.)
    assert r['fuel_distance_km']==0
    assert r['battery_traction_kWh']==pytest.approx(11.37806208)
    assert r['grid_charge_kWh']==pytest.approx(11.37806208/.9)


@pytest.mark.parametrize('eff',[0,1.1,float('nan')])
def test_invalid_efficiency(eff):
    with pytest.raises(ValueError):
        simulate_phev_day(schedule(),16.,efficiency=eff)
