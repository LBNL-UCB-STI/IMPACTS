from pathlib import Path

import numpy as np
import pandas as pd

from impacts.pipeline.workflow.notebook_charging import (
    load_notebook_functions, prepare_passenger_stops, assign_stops_indexed, fill_stops,
)
from impacts.pipeline.workflow.power_generation import build_beam_hourly_charging_demand

ROOT = Path(__file__).resolve().parents[1]


def fixture_stops():
    rows = []
    rng = np.random.default_rng(123)
    for v in range(40):
        for j, (dep, arr) in enumerate([(20000, 22000), (22000, 22300), (40000, 42000), (64000, 66000)]):
            rows.append(dict(vehicle=f'v{v:03}', vehicleType='cv-L1-17000-2019',
                             departureTime=dep, arrivalTime=arr,
                             length=80000. if v < 5 else float(rng.uniform(500, 70000))))
    return pd.DataFrame(rows)


def test_original_parking_and_distance_helpers_preserve_zero_dwell(charging_notebooks):
    original, _ = load_notebook_functions(charging_notebooks)
    trips = fixture_stops()
    stops, audit = prepare_passenger_stops(trips, original)
    assert audit['zero_dwell_segments'] == 40
    assert stops.parkingTime_s.gt(900).all()
    assert np.isclose(stops.length.sum(), trips.length.sum())
    assert not stops.arrivalTime.eq(22000).any()


def test_indexed_assignment_matches_original_notebook(monkeypatch, charging_notebooks):
    # The supplied notebook predates pandas 3, whose new sample() guard rejects
    # unequal weights when n * max(probability) > 1. Restore pandas 2 / numpy
    # weighted-without-replacement semantics for the reference call only.
    def pandas2_sample(frame, n=1, weights=None, replace=False):
        w = np.asarray(frame[weights] if isinstance(weights, str) else weights, dtype=float)
        return frame.iloc[np.random.choice(len(frame), n, replace=replace, p=w/w.sum())]
    monkeypatch.setattr(pd.DataFrame, 'sample', pandas2_sample)
    original, _ = load_notebook_functions(charging_notebooks)
    stops, _ = prepare_passenger_stops(fixture_stops(), original)
    actual = assign_stops_indexed(stops, original)
    expected = original['add_initial_soc'](stops.copy(), 42)
    expected = original['add_daily_soc'](expected)
    critical = original['pre_assign_charging'](expected, safety_margin_kWh=10, minimumAllowed_kWh=2)
    counts = original['decide_daily_charges'](expected, p1=.45, p2=.04, seed=42)
    expected = original['mark_attemptCharges'](expected, counts, critical, share_home=.8, seed=42)
    expected['chargerPower_kW'] = expected.ChosenChargerType.map(original['POWER_BY_TYPE']).fillna(0)
    for col in ['initialsoc_kWh', 'soc_kWh', 'attemptCharge', 'chargerPower_kW']:
        np.testing.assert_allclose(actual[col], expected[col])
    # Type choice, stop choice, and all SOC/energy results, not just aggregate load.
    assert actual.ChosenChargerType.fillna('').tolist() == expected.ChosenChargerType.replace(0, '').fillna('').tolist()
    expected = original['fill_time_bins'](expected, counts)
    filled, audits, profile = fill_stops(actual)
    for col in ['energyCharged_kWh', 'timeCharging_s', 'socAfterCharging_kWh']:
        np.testing.assert_allclose(filled[col], expected[col], atol=1e-10)
    np.testing.assert_allclose(audits.initial_soc_kwh + audits.charging_kwh - audits.traction_kwh, audits.final_soc_kwh)
    np.testing.assert_allclose(profile.sum(), filled.loc[filled.retained, 'energyCharged_kWh'].sum())


def test_legacy_adapter_does_not_charge_between_contiguous_segments(tmp_path):
    rows = fixture_stops().iloc[:4].copy()
    rows['length'] = 1000.
    rows['type'] = 'PathTraversal'
    rows['mode'] = 'car'
    rows['primaryFuelType'] = 'Gasoline'
    path = tmp_path/'events.parquet'
    rows.to_parquet(path)
    _, _, sessions = build_beam_hourly_charging_demand(
        path, force_all_vehicle_types_electric=True, one_charge_share=1., two_charge_share=0., home_charging_share=1.)
    assert not sessions.time.eq(22000).any()
    assert sessions.time.eq(66000).all()


def test_equal_departure_ties_are_ordered_by_arrival(charging_notebooks):
    original, _ = load_notebook_functions(charging_notebooks)
    trips = pd.DataFrame([
        dict(vehicle='v', vehicleType='cv', departureTime=100, arrivalTime=200, length=1000.),
        dict(vehicle='v', vehicleType='cv', departureTime=100, arrivalTime=100, length=0.),
    ])
    stops, audit = prepare_passenger_stops(trips, original)
    assert audit['excluded_overlap_or_over_24h_vehicles'] == 0
    assert len(stops) == 1


def test_post_charge_filter_checks_first_hour_bin_not_only_full_stop():
    row = dict(vehicle='v', departureTime=3400., arrivalTime=3500., PTID=1,
               parkingTime_s=7200., energyConsumption_kWh=10., initialsoc_kWh=5.,
               battery_kWh=75., chargerPower_kW=7.2)
    result, vehicles, profile = fill_stops(pd.DataFrame([row]))
    assert result.socAfterCharging_kWh.iloc[0] > 2
    assert result.socAfterFirstHourBin_kWh.iloc[0] < 2
    assert not vehicles.retained.iloc[0]
    assert profile.sum() == 0


def test_stop_integrator_matches_original_expanded_hour_bins(charging_notebooks):
    original, _ = load_notebook_functions(charging_notebooks)
    stops, _ = prepare_passenger_stops(fixture_stops(), original)
    assigned = assign_stops_indexed(stops, original)
    result, vehicles, profile = fill_stops(assigned)
    rows = []
    for row in assigned.to_dict('records'):
        left = row['arrivalTime']
        end = left + row['parkingTime_s']
        first = True
        while left < end:
            right = min(end, (np.floor(left/3600)+1)*3600)
            rows.append(dict(row, parkingTime_s=right-left,
                             energyConsumption_kWh=row['energyConsumption_kWh'] if first else 0.,
                             binLabel_s=(np.floor(left/3600)%24)*3600))
            first = False
            left = right
    expected = original['fill_time_bins'](pd.DataFrame(rows), {})
    sums = expected.groupby('PTID').energyCharged_kWh.sum()
    np.testing.assert_allclose(result.set_index('PTID').energyCharged_kWh.sort_index(), sums)
    min_soc = expected.groupby('vehicle').socAfterCharging_kWh.min()
    np.testing.assert_allclose(vehicles.set_index('vehicle').min_after_hour_bin_kwh, min_soc)
    retained = min_soc.index[min_soc.ge(2)]
    expected_curve = expected.loc[expected.vehicle.isin(retained)].groupby('binLabel_s').energyCharged_kWh.sum()
    np.testing.assert_allclose(profile, expected_curve.reindex(np.arange(24)*3600, fill_value=0), atol=1e-9)
