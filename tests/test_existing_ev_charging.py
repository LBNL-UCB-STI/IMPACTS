from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impacts.pipeline.workflow.charging.ldv_charging import (
    prepare_existing_ev_stops,
    load_notebook_functions,
    assign_stops_indexed,
    fill_stops,
)

ROOT = Path(__file__).resolve().parents[1]


def inputs(charging_notebooks):
    trips = pd.DataFrame([
        dict(vehicle=v, vehicleType=kind, driver=v, departureTime=dep,
             arrivalTime=arr, length=10000.)
        for v, kind in [('b','ev-test'),('p','phev-test')]
        for dep, arr in [(28000.,28800.),(61200.,62000.)]])
    metadata = pd.DataFrame(dict(vehicleTypeId=['ev-test','phev-test'],
        primaryFuelType=['electricity','electricity'], primaryFuelCapacityInJoule=[32.55*3600000,16.03125*3600000]))
    activities = pd.DataFrame([dict(person=v,time=t,actType=label)
        for v in ('b','p') for t,label in [(28900.,'Work'),(62100.,'Home')]])
    original, _ = load_notebook_functions(charging_notebooks)
    return trips, metadata, activities, original


def test_actual_batteries_hybrid_energy_and_activity_labels(charging_notebooks):
    stops, audit = prepare_existing_ev_stops(*inputs(charging_notebooks))
    np.testing.assert_allclose(stops.loc[stops.powertrain.eq('BEV'),'battery_kWh'],32.55)
    np.testing.assert_allclose(stops.loc[stops.powertrain.eq('PHEV'),'energyConsumption_kWh'],.17)
    np.testing.assert_allclose(stops.loc[stops.powertrain.eq('BEV'),'energyConsumption_kWh'],1.7)
    assert stops.loc[stops.arrivalTime.eq(62000),'is_home'].all()
    assert not stops.loc[stops.arrivalTime.eq(28800),'is_home'].any()
    assert audit['activity_source_stops']=={'events_driver_actstart':4}
    original=inputs(charging_notebooks)[-1]
    no_charge=original['add_daily_soc'](original['add_initial_soc'](stops.copy(),42))
    for _,group in no_charge.groupby('vehicle'):
        np.testing.assert_allclose(group.initialsoc_kWh-group.soc_kWh, group.energyConsumption_kWh.cumsum())


def test_do_not_override_known_work_with_longest_stop_home(charging_notebooks):
    trips, meta, acts, original = inputs(charging_notebooks)
    acts['actType']='Work'
    stops,_=prepare_existing_ev_stops(trips,meta,acts,original)
    assert not stops.is_home.any()


def test_reject_ice_and_battery_powertrain_mismatch(charging_notebooks):
    trips, meta, acts, original = inputs(charging_notebooks)
    bad=trips.copy();bad.loc[0,'vehicleType']='cv-test'
    with pytest.raises(ValueError,match='Only native'):
        prepare_existing_ev_stops(bad,meta,acts,original)
    meta.loc[0,'primaryFuelCapacityInJoule']=20*3600000
    with pytest.raises(ValueError,match='hybrid heuristic'):
        prepare_existing_ev_stops(trips,meta,acts,original)


def test_original_soc_integrator_with_small_phev_batteries(charging_notebooks):
    stops,_=prepare_existing_ev_stops(*inputs(charging_notebooks))
    original=inputs(charging_notebooks)[-1]
    assigned=assign_stops_indexed(stops,original)
    filled,vehicles,profile=fill_stops(assigned)
    expected=original['fill_time_bins'](assigned,{})
    for col in ['energyCharged_kWh','timeCharging_s','socAfterCharging_kWh']:
        np.testing.assert_allclose(filled[col],expected[col],atol=1e-10)
    np.testing.assert_allclose(profile.sum(),filled.loc[filled.retained,'energyCharged_kWh'].sum())
    np.testing.assert_allclose(vehicles.initial_soc_kwh+vehicles.charging_kwh-vehicles.traction_kwh,vehicles.final_soc_kwh)


def test_phev20_traction_and_projected_soc_agree(charging_notebooks):
    trips, meta, acts, original = inputs(charging_notebooks)
    original['HYBRID_ELECTRICITY_CONSUMPTION_FACTOR'] = .2
    stops, audit = prepare_existing_ev_stops(trips, meta, acts, original, phev_electricity_factor=.2)
    np.testing.assert_allclose(stops.loc[stops.powertrain.eq('PHEV'),'energyConsumption_kWh'], .34)
    np.testing.assert_allclose(stops.loc[stops.powertrain.eq('BEV'),'energyConsumption_kWh'], 1.7)
    projected = original['add_daily_soc'](original['add_initial_soc'](stops.copy(),42))
    np.testing.assert_allclose(projected.initialsoc_kWh-projected.soc_kWh,
                              projected.groupby('vehicle').energyConsumption_kWh.cumsum())
    assert audit['input_traction_proxy_kwh'] == pytest.approx(4.08)
    assigned = assign_stops_indexed(stops, original)
    filled, vehicles, profile = fill_stops(assigned)
    np.testing.assert_allclose(profile.sum(), filled.loc[filled.retained,'energyCharged_kWh'].sum())
    np.testing.assert_allclose(vehicles.initial_soc_kwh + vehicles.charging_kwh - vehicles.traction_kwh,
                              vehicles.final_soc_kwh)


@pytest.mark.parametrize('factor', [-.1, 1.1, np.nan, np.inf])
def test_invalid_phev_factor(factor, charging_notebooks):
    with pytest.raises(ValueError, match='PHEV electricity factor'):
        prepare_existing_ev_stops(*inputs(charging_notebooks), phev_electricity_factor=factor)
