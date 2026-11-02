import numpy as np
import pandas as pd
import pytest

from impacts.pipeline.workflow.truck_charging import trip_energy, schedule_vehicle, build_truck_profiles


def trips(kind='hdv', miles=20, seconds=3600):
    return pd.DataFrame(dict(vehicle=['v'], vehicleType=[f'ft-{kind}-Dsl'],
                             departureTime=[0.], arrivalTime=[float(seconds)],
                             length=[miles * 1609.344]))


def test_emfac_twenty_mph_and_grid_battery_boundary():
    result = trip_energy(trips())
    assert result.speed_factor.iloc[0] == 1
    assert result.required_grid_kwh.iloc[0] == pytest.approx(20 * 2.07)
    assert result.traction_battery_kwh.iloc[0] == pytest.approx(20 * 2.07 * .85)


def test_speed_interpolation_and_clamping():
    assert trip_energy(trips(miles=22.5)).speed_factor.iloc[0] == pytest.approx(.96)
    assert trip_energy(trips(miles=100)).speed_factor.iloc[0] == .91
    assert trip_energy(trips(miles=2)).speed_factor.iloc[0] == 2.85


def test_light_pickup_not_heavy_truck_speed_curve():
    result = trip_energy(trips(kind='ld1', miles=10))
    assert result.required_grid_kwh.iloc[0] == pytest.approx(4.9)
    assert result.speed_factor.iloc[0] == 1


def test_zero_duration_is_flagged_not_silently_lost():
    result = trip_energy(trips(seconds=0))
    assert result.speed_fallback.iloc[0]
    assert result.speed_mph.iloc[0] == 20
    assert result.required_grid_kwh.iloc[0] > 0


def test_zero_distance_zero_duration_needs_no_energy():
    assert trip_energy(trips(miles=0, seconds=0)).required_grid_kwh.iloc[0] == 0


def test_cyclic_energy_conservation():
    energy = trip_energy(trips())
    result = schedule_vehicle(energy, battery_kwh=291, grid_kw=150)
    assert result['grid_kwh'].sum() == pytest.approx(energy.required_grid_kwh.sum())
    assert result['unmet_battery_kwh'].sum() == 0
    assert result['initial_battery_kwh'] == pytest.approx(result['final_battery_kwh'])


def test_infeasible_vehicle_is_kept_and_missing_energy_reported():
    profiles, sessions, vehicles, energy = build_truck_profiles(trips(miles=1000))
    assert len(vehicles) == 1
    assert vehicles.unmet_grid_kwh.iloc[0] > 0
    assert energy.unmet_battery_kwh.sum() > 0
    assert vehicles.supplied_grid_kwh.iloc[0] + vehicles.unmet_grid_kwh.iloc[0] == pytest.approx(vehicles.required_grid_kwh.iloc[0])
    assert profiles['ft-hdv-Dsl'].sum() == pytest.approx(sessions.representative_daily_kwh.sum())


def test_zero_parking_is_not_a_full_day():
    energy = trip_energy(pd.concat([trips(), trips().assign(departureTime=3600., arrivalTime=7200.)]))
    result = schedule_vehicle(energy, battery_kwh=291, grid_kw=150)
    assert result['parking_hours'][0] == 0
    assert result['grid_kwh'][0] == 0


def test_multiday_not_overlapping_daily_repeats():
    frame = pd.concat([trips(), trips().assign(departureTime=90000., arrivalTime=93600.)])
    profiles, sessions, vehicles, energy = build_truck_profiles(frame)
    assert vehicles.cycle_days.iloc[0] == 2
    assert vehicles.daily_required_grid_kwh.iloc[0] == pytest.approx(energy.required_grid_kwh.sum() / 2)
    assert profiles['ft-hdv-Dsl'].sum() == pytest.approx(vehicles.daily_supplied_grid_kwh.sum())


def test_overlap_rejected():
    energy = trip_energy(pd.concat([trips(), trips().assign(departureTime=100., arrivalTime=4000.)]))
    with pytest.raises(ValueError, match='Overlapping'):
        schedule_vehicle(energy, battery_kwh=291, grid_kw=150)


def test_unmapped_type_rejected():
    with pytest.raises(ValueError, match='Unmapped'):
        trip_energy(trips().assign(vehicleType='ft-new-Dsl'))


def test_negative_trip_time_rejected():
    with pytest.raises(ValueError, match='precedes'):
        trip_energy(trips().assign(arrivalTime=-1, departureTime=0).assign(arrivalTime=1, departureTime=2))
