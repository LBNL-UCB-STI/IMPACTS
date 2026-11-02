"""Replace only freight charging, retaining all original non-freight profiles.

Writes a feasibility audit even if batteries/parking cannot serve all itineraries.
An incomplete all-electric scenario is not passed silently to power dispatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from impacts.pipeline.workflow.power_generation import _safe_profile_token, write_charging_curve_outputs
from impacts.pipeline.workflow.truck_charging import PARAMETERS, ASSUMPTIONS, SPEED_MPH, SPEED_FACTORS, build_truck_profiles


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--previous-demand', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    # Refuse overwriting any prior run, including failed/diagnostic runs.
    out.mkdir(parents=True, exist_ok=False)
    connection = duckdb.connect()
    trips = connection.execute(
        "SELECT vehicle, vehicleType, departureTime, arrivalTime, length FROM read_parquet(?) "
        "WHERE type='PathTraversal' AND starts_with(vehicleType, 'ft-')", [str(args.events)]).df()
    connection.close()
    profiles, sessions, vehicles, energy = build_truck_profiles(trips)
    energy.to_parquet(out / 'truck_trip_energy_audit.parquet', index=False)
    vehicles.to_csv(out / 'truck_vehicle_energy_soc_audit.csv', index=False)
    vehicles.loc[vehicles.unmet_grid_kwh > 1e-6].to_csv(out / 'truck_infeasible_vehicles.csv', index=False)
    sessions.to_parquet(out / 'truck_charging_sessions.parquet', index=False)
    parameters = pd.DataFrame.from_dict(PARAMETERS, orient='index').rename_axis('truck_class').reset_index()
    parameters.to_csv(out / 'truck_parameters_with_sources.csv', index=False)
    metadata = dict(ASSUMPTIONS, parameters=PARAMETERS,
                    speed_mph=SPEED_MPH.tolist(), speed_factors=SPEED_FACTORS.tolist(),
                    events_file=str(args.events.resolve()), events_sha256=digest(args.events),
                    previous_demand=str(args.previous_demand.resolve()),
                    previous_demand_sha256=digest(args.previous_demand / 'sfbay_hourly_charging_demand.parquet'))
    (out / 'truck_model_parameters.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    summary_by_class = vehicles.groupby('truck_class').agg(
        vehicles=('vehicle', 'size'), infeasible_vehicles=('unmet_grid_kwh', lambda x: int((x > 1e-6).sum())),
        daily_km=('daily_distance_km', 'sum'), daily_required_kwh=('daily_required_grid_kwh', 'sum'),
        daily_supplied_kwh=('daily_supplied_grid_kwh', 'sum'), daily_unmet_kwh=('daily_unmet_grid_kwh', 'sum'))
    summary_by_class.to_csv(out / 'truck_class_summary.csv')
    truck_demand = pd.DataFrame({'hour_of_year': np.arange(8760)})
    for kind in PARAMETERS:
        daily = sum((p for t, p in profiles.items() if t.split('-')[1] == kind), np.zeros(24))
        truck_demand[f'charging_demand_{kind}_MWh'] = np.tile(daily / 1000., 365)
    truck_demand['charging_demand_MWh'] = truck_demand.filter(like='charging_demand_').sum(axis=1)
    truck_demand.to_parquet(out / 'truck_hourly_supplied_demand.parquet', index=False)
    previous = pd.read_parquet(args.previous_demand / 'sfbay_hourly_charging_demand.parquet')
    nontruck = [c for c in previous if c.startswith('charging_demand_') and c != 'charging_demand_MWh' and not c.startswith('charging_demand_ft_')]
    demand = previous[['hour_of_year'] + nontruck].copy()
    for vehicle_type, daily in profiles.items():
        demand[f'charging_demand_{_safe_profile_token(vehicle_type)}_MWh'] = np.tile(daily / 1000., 365)
    demand['charging_demand_MWh'] = demand.filter(like='charging_demand_').sum(axis=1)
    pd.testing.assert_frame_equal(demand[nontruck], previous[nontruck])
    np.testing.assert_allclose(demand.charging_demand_MWh,
                               previous[nontruck].sum(axis=1) + truck_demand.charging_demand_MWh, atol=1e-9)
    previous_sessions = pd.read_parquet(args.previous_demand / 'beam_charging_sessions.parquet')
    previous_sessions = previous_sessions.loc[~previous_sessions.vehicleType.str.startswith('ft-')].copy()
    previous_sessions['cycle_days'] = 1
    previous_sessions['representative_daily_kwh'] = previous_sessions.energy_needed_kwh
    combined_sessions = pd.concat([previous_sessions, sessions], ignore_index=True)
    combined_sessions.to_parquet(out / 'beam_charging_sessions.parquet', index=False)
    feasible = not (vehicles.unmet_grid_kwh > 1e-6).any()
    # Filename and plot title explicitly distinguish served load from full demand.
    demand.to_parquet(out / 'sfbay_hourly_charging_demand.parquet', index=False)
    summary = dict(
        charging_demand_source='BEAM events: class-specific trucks plus unchanged legacy nontruck load',
        ready_for_full_electrification_pipeline=bool(feasible),
        curve_scope='Full load' if feasible else 'Feasibility diagnostic: supplied load only, excludes unmet truck energy',
        truck_vehicles=int(len(vehicles)), truck_trips=int(len(energy)),
        infeasible_truck_vehicles=int((vehicles.unmet_grid_kwh > 1e-6).sum()),
        truck_cycle_days_counts={str(k): int(v) for k, v in vehicles.cycle_days.value_counts().items()},
        zero_duration_speed_fallback_trips=int(energy.speed_fallback.sum()),
        speed_clamped_trips=int(energy.speed_clamped.sum()),
        daily_truck_required_MWh=float(vehicles.daily_required_grid_kwh.sum() / 1000.),
        daily_truck_supplied_MWh=float(vehicles.daily_supplied_grid_kwh.sum() / 1000.),
        daily_truck_unmet_MWh=float(vehicles.daily_unmet_grid_kwh.sum() / 1000.),
        daily_energy_MWh=float(demand.charging_demand_MWh.sum() / 365.),
        annual_energy_MWh=float(demand.charging_demand_MWh.sum()),
        annual_peak_MW=float(demand.charging_demand_MWh.max()),
        nontruck_profiles_bitwise_unchanged=True, fleet_scale_factor=1.,
        previous_daily_total_MWh=float(previous.charging_demand_MWh.sum() / 365.),
        assumptions=ASSUMPTIONS,
        limitations=['All parking has assumed charger access; no actual charger input.',
                     'Nontruck legacy model unchanged, including nonroad/transit proxy limitations.',
                     'Existing traffic baseline is not reduced for vehicle electrification.'])
    (out / 'beam_charging_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    # Reuse the existing charging figure layout with explicit diagnostic labels.
    write_charging_curve_outputs(demand, out, scenario_label=(
        'updated trucks + unchanged nontruck model' if feasible else 'SUPPLIED LOAD ONLY; unmet truck energy excluded'))
    write_charging_curve_outputs(truck_demand, out / 'truck_curve', scenario_label=(
        'class-specific trucks' if feasible else 'TRUCK SUPPLIED LOAD ONLY; unmet energy excluded'))
    print(summary_by_class.to_string(), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
