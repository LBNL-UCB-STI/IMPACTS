"""Replay Home80 and three collaborator freight power cases using explicit inputs.

External BREATHE allocator/data and collaborator data are user-supplied; only
the code under this checkout is used for IMPACTS charging and dispersion.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import geopandas as gpd


def compare_table(actual, reference, keys):
    a = actual.sort_values(keys).reset_index(drop=True)
    b = reference.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[b.columns], b, check_dtype=False, rtol=1e-9, atol=1e-9)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs-root', type=Path, required=True)
    p.add_argument('--freight-input', type=Path, required=True)
    p.add_argument('--breathe-root', type=Path, required=True)
    p.add_argument('--isrm', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import impacts
    assert Path(impacts.__file__).resolve().is_relative_to(ROOT)
    from impacts.pipeline.workflow.hourly_power import run_hourly_power
    from analyze_collaborator_truck_charging import allocate
    from impacts.pipeline.workflow.charging.ldv_home_sensitivity import build_schedules, run_periodic
    from impacts.pipeline.workflow.charging.ldv_home_sensitivity import build_phev_preference_schedules, run_phev_periodic
    spatial_base = args.outputs_root / 'impacts-postsim--sfbay--base--20260901-165739'
    base = args.outputs_root / 'events-fc07-existing-ev-charging-code10x-20260928-v2'
    ldv_ref = base / 'bev-phev-home80-evi20c-20261012'
    combined_ref = args.outputs_root / 'ldv-home80-freight3-power-combined10x-20261029'
    freight_ref = args.outputs_root / 'baseline0-ldv-home80-freight3-10x-20261026/freight_power'
    trips = pd.read_parquet(base / 'existing_ev_trips.parquet')
    audit = pd.read_parquet(base / 'phev20-sensitivity-20261012/phev20/ev_stop_charging_audit.parquet')
    print('REPLAY Home80 BEV/PHEV schedules and periodic SOC', flush=True)
    bev = run_periodic([s for s in build_schedules(trips, audit, .8, 42,
        base_kWh_per_km=.202) if s.kind == 'BEV'], efficiency=.9)
    phev = run_phev_periodic(build_phev_preference_schedules(trips, audit, .8, 42),
                             charging_policy='primary_preference')
    for kind, result in [('bev_home80', bev), ('phev_home80', phev)]:
        folder = args.output / 'charging' / kind
        folder.mkdir(parents=True)
        for key in ['vehicles', 'sessions', 'trips']:
            actual, expected = result[key], pd.read_parquet(ldv_ref / kind / (key + '.parquet'))
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-10, atol=1e-9)
            actual.to_parquet(folder / (key + '.parquet'), index=False)
    ldv = (bev['profiles']['steady_BEV'] + phev['profiles']['steady_grid']) * .01
    np.testing.assert_allclose(ldv, pd.read_csv(ldv_ref / 'comparison_hourly.csv').Both_Home80_MW,
                               rtol=1e-12, atol=1e-12)
    cases = {'LDV': (ldv, combined_ref / 'LDV')}
    for path in sorted(args.freight_input.glob('charging_events_*.csv')):
        raw = pd.read_csv(path)
        name = raw.scenario.iloc[0].replace('SFBAY-2050-', '')
        data = raw.loc[raw.powerTrain.eq('BE')]
        horizon = max(24, math.ceil((data.time / 3600 + data.energy_needed_kwh / data.actual_charge_speed_kw).max()))
        curve = sum((allocate(r.time / 3600, r.energy_needed_kwh, r.actual_charge_speed_kw, horizon)
                     for r in data.itertuples()), np.zeros(horizon))
        load = curve[:24] * .01
        reference = freight_ref / name
        old = pd.read_csv(reference / 'charging_demand/freight_daily_curve_10x.csv')
        np.testing.assert_allclose(load, old.charging_MW_10x, rtol=1e-12, atol=1e-12)
        cases[name] = (load, reference)
    assert set(cases) == {'LDV', 'BF-06EP', 'BF-1EP', 'HF-1EP'}
    checks = []
    concentrations = {}
    for name, (load, reference) in cases.items():
        target = args.output / name
        demand = target / 'charging_demand'
        demand.mkdir(parents=True)
        pd.DataFrame(dict(hour_of_year=np.arange(8760), charging_demand_MWh=np.tile(load,365))).to_parquet(
            demand / 'sfbay_hourly_charging_demand.parquet', index=False)
        qa = dict(curve_scope=name + ' representative day; 10x', daily_energy_MWh=float(load.sum()),
                  fleet_scale_factor=10, additional_scale_factor=1, annualization_days=365)
        (demand / 'beam_charging_summary.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
        print('DISPATCH + ISRM', name, flush=True)
        run_hourly_power(config=dict(enabled=True,
            allocation_method='cambium_marginal_technology', electricity_year=2040, egrid_year=2023,
            hourly_demand_file=str(demand / 'sfbay_hourly_charging_demand.parquet'),
            charging_summary_file=str(demand / 'beam_charging_summary.json'),
            breathe_root=str(args.breathe_root.resolve()),
            county_boundaries_file=str(spatial_base / 'preprocess/county_boundaries.gpkg'),
            source_layer_index=0), isrm_url=str(args.isrm),
            inmap_grid_path=str(spatial_base / 'preprocess/inmap_study_area.parquet'),
            output_dir=target / 'power_generation')
        newdir, olddir = target / 'power_generation', reference / 'power_generation'
        for rel, keys in [('plant_allocation/plant_emissions_approx.csv', ['ORISPL']),
                          ('plant_allocation/hourly_dispatch_summary.csv', ['hour_of_year']),
                          ('plant_emissions_used_for_concentration.csv', ['ORISPL'])]:
            compare_table(pd.read_csv(newdir / rel), pd.read_csv(olddir / rel), keys)
        actual = gpd.read_parquet(newdir / 'power_generation_isrm_concentrations.parquet')
        expected = gpd.read_parquet(olddir / 'power_generation_isrm_concentrations.parquet')
        compare_table(actual, expected, ['inmap_cell_id'])
        a = actual.sort_values('inmap_cell_id').reset_index(drop=True)
        b = expected.sort_values('inmap_cell_id').reset_index(drop=True)
        cols = [c for c in a if c.startswith('power_')]
        concentrations[name] = a
        checks.append(dict(scenario=name, status='passed', rows=len(a), daily_MWh=float(load.sum()),
            peak_MW=float(load.max()), peak_hour=int(load.argmax()),
            maximum_concentration_absolute_difference=float(np.abs(a[cols].to_numpy()-b[cols].to_numpy()).max()),
            hourly_dispatch_and_annual_emissions_match=True))
    for name in ['BF-06EP', 'BF-1EP', 'HF-1EP']:
        a, b = concentrations['LDV'], concentrations[name]
        result = a[['inmap_cell_id', 'geometry']].copy()
        for col in [c for c in a if c.startswith('power_')]:
            result['ldv_' + col] = a[col]
            result['freight_' + col] = b[col]
            result[col] = a[col] + b[col]
        compare_table(result, gpd.read_parquet(combined_ref / 'combined' / name / 'ldv_freight_power_concentrations.parquet'), ['inmap_cell_id'])
        result.to_parquet(args.output / (name + '_combined.parquet'), index=False)
    (args.output / 'comparison.json').write_text(json.dumps(dict(status='passed',
        imported_code=str(impacts.__file__), charging_vehicle_session_trip_parity=True,
        three_combined_power_cases_match=True, power_layer=0, power_near_source_removal=False,
        external_allocator=str(args.breathe_root / 'scripts/allocate_power_plants_approx.py'), cases=checks), indent=2), encoding='utf-8')
    print('ALL CHARGING / POWER COMPARISONS PASSED', flush=True)


if __name__ == '__main__':
    main()
