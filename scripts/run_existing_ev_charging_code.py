"""Generate charging-code demand for native existing BEV/PHEV trips, then 10x.

Charging-curve stage only: no dispatch, annualization or concentration changes.
"""
from research_paths import ISRM_PATH, BREATHE_ROOT, BEAM_DATA_ROOT, CHARGING_REFERENCE
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from impacts.pipeline.workflow.charging.ldv_charging import (
    prepare_existing_ev_stops,
    load_notebook_functions,
    assign_stops_indexed,
    fill_stops,
)
from impacts.pipeline.workflow.power_generation import allocate_charging_event
from compare_native_evi_15k_weekday import weekday_hourly

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'examples/pilates/impacts/impacts_output'


def save(path, data):
    path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, default=ROOT/'chargingcode/beam-fc07-v260419-it3/3.events.parquet')
    parser.add_argument('--vehicle-types', type=Path, default=(BEAM_DATA_ROOT/'vehicle-tech/vehicletypes--baseline.csv'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phev-electricity-factor', type=float, default=.1,
                        help='PHEV multiplier on 0.17 kWh/km; default preserves original 10%% proxy')
    args = parser.parse_args()
    if not np.isfinite(args.phev_electricity_factor) or not 0 <= args.phev_electricity_factor <= 1:
        parser.error('--phev-electricity-factor must be between 0 and 1')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out/'run_status.json', dict(status='running', stage='charging_only'))
    db = duckdb.connect()
    db.execute('SET threads=4')
    trips = db.execute("""SELECT vehicle, vehicleType, driver,
        departureTime::DOUBLE departureTime, arrivalTime::DOUBLE arrivalTime,
        length::DOUBLE length, primaryFuelLevel::DOUBLE primaryFuelLevel
        FROM read_parquet(?) WHERE type='PathTraversal'
        AND (vehicleType LIKE 'ev-%' OR vehicleType LIKE 'phev-%')
        AND mode IN ('car','car_hov2','car_hov3','ride_hail','ride_hail_pooled')""", [str(args.events)]).df()
    ids = pd.DataFrame({'driver': trips.driver.dropna().unique()})
    db.register('drivers', ids)
    acts = db.execute("""SELECT person,time::DOUBLE AS "time",actType FROM read_parquet(?)
        WHERE type='actstart' AND person IN (SELECT driver FROM drivers)""",[str(args.events)]).df()
    db.close()
    if trips.groupby('vehicle').vehicleType.nunique().gt(1).any():
        raise ValueError('Vehicle ID changes type')
    fleet = trips[['vehicle','vehicleType']].drop_duplicates()
    fleet['powertrain'] = np.where(fleet.vehicleType.str.startswith('phev-'), 'PHEV', 'BEV')
    counts = fleet.groupby('powertrain').vehicle.nunique().to_dict()
    if set(counts) != {'BEV','PHEV'}:
        raise ValueError('Unexpected existing-EV population')
    original, provenance = load_notebook_functions(ROOT/'chargingcode')
    # Original add_daily_soc reads this shared namespace. Keep projected SOC
    # and integrated traction energy consistent when changing the PHEV proxy.
    original['HYBRID_ELECTRICITY_CONSUMPTION_FACTOR'] = args.phev_electricity_factor
    metadata = pd.read_csv(args.vehicle_types, low_memory=False)
    selected = metadata.loc[metadata.vehicleTypeId.isin(fleet.vehicleType)].copy()
    selected['battery_kWh'] = selected.primaryFuelCapacityInJoule / 3_600_000
    observed = trips.groupby('vehicleType').primaryFuelLevel.max() / 3_600_000
    selected['maximum_observed_remaining_kWh'] = selected.vehicleTypeId.map(observed)
    # Rounded event fuel values can differ by a few joules from nominal capacity.
    if (selected.maximum_observed_remaining_kWh > selected.battery_kWh+.01).any():
        raise ValueError('Event SOC exceeds matched nominal capacity')
    selected.to_csv(out/'matched_vehicle_types.csv', index=False)
    fleet.to_parquet(out/'existing_ev_inventory.parquet', index=False)
    trips.to_parquet(out/'existing_ev_trips.parquet', index=False)
    stops, audit = prepare_existing_ev_stops(trips, metadata, acts, original,
        phev_electricity_factor=args.phev_electricity_factor)
    stops.to_parquet(out/'prepared_ev_stops.parquet', index=False)
    assigned = assign_stops_indexed(stops, original, seed=42)
    np.testing.assert_allclose(assigned.initialsoc_kWh-assigned.soc_kWh,
        assigned.groupby('vehicle', sort=False).energyConsumption_kWh.cumsum(), atol=1e-9)
    filled, vehicles, sample_profile = fill_stops(assigned)
    vehicles = vehicles.merge(fleet, on='vehicle', validate='one_to_one')
    vehicles['battery_kWh'] = vehicles.vehicleType.map(selected.set_index('vehicleTypeId').battery_kWh)
    vehicles['initial_soc_pct'] = vehicles.initial_soc_kwh / vehicles.battery_kWh * 100
    vehicles['final_soc_pct'] = vehicles.final_soc_kwh / vehicles.battery_kWh * 100
    filled.to_parquet(out/'ev_stop_charging_audit.parquet', index=False)
    vehicles.to_parquet(out/'ev_vehicle_soc_audit.parquet', index=False)
    vehicles.to_csv(out/'ev_vehicle_soc_audit.csv', index=False)
    sessions = filled.loc[filled.retained & filled.energyCharged_kWh.gt(0)].copy()
    sessions.to_parquet(out/'modeled_ev_charging_sessions.parquet', index=False)
    profiles = {}
    for kind in ('BEV','PHEV'):
        p = np.zeros(24)
        for r in sessions.loc[sessions.powertrain.eq(kind)].itertuples():
            allocate_charging_event(p, start_hour=r.arrivalTime/3600,
                duration_hours=r.timeCharging_s/3600, energy_kwh=r.energyCharged_kWh)
        profiles[kind] = p
    np.testing.assert_allclose(sum(profiles.values()), sample_profile)
    scale = 10.
    daily = pd.DataFrame({'hour_of_day':np.arange(24)})
    for kind, profile in profiles.items():
        daily[f'{kind}_sample_kWh'] = profile
        daily[f'{kind}_MW_10x'] = profile*scale/1000
    daily['total_MW_10x'] = daily.BEV_MW_10x + daily.PHEV_MW_10x
    daily['total_kW_per_1000_PEV'] = daily.total_MW_10x * 1e6/(len(fleet)*scale)
    daily.to_csv(out/'existing_ev_hourly_charging_curve_10x.csv', index=False)
    daily.to_parquet(out/'existing_ev_hourly_charging_curve_10x.parquet', index=False)
    charging_interface = pd.DataFrame({'hour_of_day':range(24),
        'charging_demand_BEV_MWh':daily.BEV_MW_10x,
        'charging_demand_PHEV_MWh':daily.PHEV_MW_10x,
        'charging_demand_MWh':daily.total_MW_10x})
    charging_interface.to_parquet(out/'representative_day_charging_demand.parquet', index=False)
    kept = vehicles.loc[vehicles.retained]
    ledger = []
    for kind, sub in kept.groupby('powertrain'):
        ledger.append(dict(powertrain=kind, input_vehicles=counts[kind],
            modeled_vehicles=int(vehicles.powertrain.eq(kind).sum()),retained_vehicles=len(sub),
            charging_vehicles=int(sessions.loc[sessions.powertrain.eq(kind),'vehicle'].nunique()),
            daily_charging_MWh_10x=float(sub.charging_kwh.sum()*scale/1000),
            daily_traction_MWh_10x=float(sub.traction_kwh.sum()*scale/1000),
            SOC_gain_MWh_10x=float((sub.final_soc_kwh-sub.initial_soc_kwh).sum()*scale/1000),
            mean_initial_SOC_pct=float(sub.initial_soc_pct.mean()),mean_final_SOC_pct=float(sub.final_soc_pct.mean()),
            retained_below_2kWh_before_charging=int(sub.min_before_charge_kwh.lt(2).sum())))
    residual = vehicles.initial_soc_kwh+vehicles.charging_kwh-vehicles.traction_kwh-vehicles.final_soc_kwh
    np.testing.assert_allclose(residual,0,atol=1e-8)
    np.testing.assert_allclose(daily.total_MW_10x.sum(), kept.charging_kwh.sum()*scale/1000)
    pd.DataFrame(ledger).to_csv(out/'powertrain_energy_soc_summary.csv', index=False)
    plt.rcParams.update({'font.size':11})
    fig, ax = plt.subplots(figsize=(12,6),layout='constrained')
    for kind, color, style in [('BEV','#2166ac','-'),('PHEV','#d95f02','--')]:
        ax.plot(daily.hour_of_day,daily[f'{kind}_MW_10x'],lw=2.4,color=color,ls=style,label=f'{kind} ({counts[kind]*10:,} fleet vehicles)')
    ax.plot(daily.hour_of_day,daily.total_MW_10x,color='black',lw=2.8,label='Total modeled EV charging')
    ax.set(title=f'Existing BEV + PHEV trips: PHEV electricity factor {args.phev_electricity_factor:.0%}\n10x sample expansion; no EVI curve fitting',
        xlabel='Hour of day (simulation-local)',ylabel='Hourly-average charging power (MW)',
        xlim=(0,23),xticks=range(24),ylim=(0,None))
    ax.legend(); ax.grid(alpha=.25)
    fig.supxlabel('Native charging records are NOT used. Overnight sessions wrap onto a representative 24-hour profile.\n'
        'Original notebook probabilities retained; charger access assumed; no CAISO load weighting. SOC audit required.',fontsize=9)
    fig.savefig(out/'existing_ev_charging_curve_10x.png',dpi=180);plt.close(fig)
    evi_path = OUTPUTS/'events-fc07-baseline-native-ev10x-20260928/charging_demand/evi_15000_home80_weekday_v2/evi_response_30000.json'
    evi = json.loads(evi_path.read_text())
    if evi['inputs'] != dict(fleet_size=30000,mean_dvmt=35,temp_c=10,pev_dist='BEV',class_dist='Sedan',
        home_access_dist='HA100',home_power_dist='MostL2',work_power_dist='MostL2',pref_dist='Home80',res_charging='min_delay',work_charging='min_delay'):
        raise ValueError('Unexpected saved EVI scenario')
    evi_kw, _ = weekday_hourly(evi['response'])
    compare = pd.DataFrame(dict(hour_of_day=range(24), charging_code_kW_per_1000_PEV=daily.total_kW_per_1000_PEV,
        EVI_weekday_kW_per_1000_PEV=evi_kw/30))
    compare['charging_code_MW_15000'] = compare.charging_code_kW_per_1000_PEV*.015
    compare['EVI_weekday_MW_15000'] = compare.EVI_weekday_kW_per_1000_PEV*.015
    compare.to_csv(out/'evi_weekday_comparison.csv', index=False)
    fig, axes = plt.subplots(2,1,figsize=(12,9),layout='constrained')
    for ax, suffix, title, unit in [(axes[0],'kW_per_1000_PEV','Per 1,000 existing plug-in vehicles','kW / 1,000 PEVs'),
        (axes[1],'MW_15000','Display-only common fleet: 15,000 PEVs','Charging power (MW)')]:
        ax.plot(compare.hour_of_day,compare[f'charging_code_{suffix}'],lw=2.5,color='#2166ac',label='Charging code: existing EV trips')
        ax.plot(compare.hour_of_day,compare[f'EVI_weekday_{suffix}'],lw=2.5,color='#d95f02',ls='--',label='EVI weekday: Home80, 35 mi/day, 10 C')
        ax.set(title=title,xlabel='Hour of day',ylabel=unit,xlim=(0,23),xticks=range(24),ylim=(0,None));ax.legend();ax.grid(alpha=.25)
    fig.suptitle('Charging CODE for existing EVs vs EVI-Pro Lite (weekday only)',fontsize=15)
    fig.supxlabel(f'Code: {counts["BEV"]} BEVs + {counts["PHEV"]} PHEVs; actual 10x fleet = {len(fleet)*10:,}. EVI: 30,000 profile scaled for display.\n'
        'Powertrain mixtures differ (EVI 75% BEV); no calibration. Provisional: see SOC and excluded-vehicle audit.',fontsize=9)
    fig.savefig(out/'existing_ev_vs_evi_weekday.png',dpi=180);plt.close(fig)
    curve_valid = bool(kept.min_before_charge_kwh.ge(2).all() and len(kept)==len(fleet)
        and abs((kept.final_soc_kwh-kept.initial_soc_kwh).sum()) <= .01*kept.traction_kwh.sum())
    summary = dict(input_events=str(args.events),source_kind='Original charging notebook on existing BEV/PHEV travel, NOT native refuel events',
        input_vehicle_counts=counts,input_total_vehicles=len(fleet),expanded_fleet=len(fleet)*10,scale_factor=10,
        modeled_vehicles=len(vehicles),retained_vehicles=len(kept),discarded_vehicles=int((~vehicles.retained).sum()),
        input_vehicles_without_modeled_stops=len(fleet)-len(vehicles),actual_charging_vehicles=int(sessions.vehicle.nunique()),
        daily_energy_MWh_10x=float(daily.total_MW_10x.sum()),peak_MW_10x=float(daily.total_MW_10x.max()),peak_hour=int(daily.total_MW_10x.idxmax()),
        input_audit=audit,powertrain_energy_soc=ledger,max_energy_balance_residual_kWh=float(residual.abs().max()),
        assumptions=dict(P1=.45,P2=.04,home_share=.8,home_L1_share=.2,home_L2_share=.8,
            L1_kW=1.9,L2_kW=7.2,DC_kW=50,BEV_kWh_per_km=.17,
            PHEV_kWh_per_km=.17*args.phev_electricity_factor,
            PHEV_electricity_factor=args.phev_electricity_factor,seed=42,
            battery_source=str(args.vehicle_types),home_source='next driver actstart within 30min and parking interval; longest unmatched stop fallback only',
            charger_availability='Assumed L1/L2 at home and L1/L2/DC at non-home stops; no AFDC TAZ inventory',
            grid_load='Neutral zero; no CAISO hourly weighting',initial_SOC='Original beta-distribution function; not native BEAM SOC',
            daily_window='Original arrival<29h; final parking until first departure+24h; hourly energy folded modulo24',
            charging_losses='No extra charger losses, matching original notebook',minimum_SOC_filter='Original first-hour-bin postcharge >=2kWh; before-charge violations separately reported'),
        calibration_applied=False,
        sensitivity_parameter_override=args.phev_electricity_factor != .1,
        annualization_performed=False,ready_for_power_generation=curve_valid,
        downstream_status='NOT RUN: charging curve and diagnostic stage only',
        EVI_reference=str(evi_path),EVI_inputs=evi['inputs'],EVI_sha256=hashlib.sha256(evi_path.read_bytes()).hexdigest(),
        notebook_provenance=provenance)
    save(out/'charging_code_summary.json',summary)
    save(out/'run_status.json',dict(status='charging_curve_completed',downstream='not_started',ready_for_power_generation=curve_valid))
    (out/'README.md').write_text('# Existing-EV charging CODE run\n\n'
        'Only existing BEV/PHEV vehicle IDs in 3.events.parquet are used. No ICE conversion, no reuse of the all-passenger curve, '
        'and no RefuelSessionEvent/PlugIn power or SOC inputs. Original notebook core functions are reused with seed42. '
        'Capacity matches the explicit source-type metadata. '
        f'PHEV electric-energy factor: {args.phev_electricity_factor:g} (notebook default 0.1). '
        'This is a scenario proxy, not a physics-based fuel-switching model or a validated local value.\n\n'
        'Driver activity starts are matched forward within 30min of arrival and strictly before the next departure. '
        'Unknown home locations may use only the longest unmatched stop, not overwrite observed non-home activity. '
        'TAZ charger inventory and CAISO series remain unavailable: availability is assumed and grid-load score is neutral.\n\n'
        'Actual curve: existing_ev_charging_curve_10x.png; hourly CSV retains sample energy and 10x loads. '
        'EVI comparison: existing_ev_vs_evi_weekday.png; top per1000 ALL input PEVs, bottom display-only 15000. '
        'Actual 10x population remains 15440. Excluded vehicles are not removed from the comparison denominator. '
        'Original modulo24 overnight allocation is kept, not a strict midnight-truncated native-charging curve.\n\n'
        'This is a first uncalibrated modeled curve. Consult charging_code_summary.json and powertrain_energy_soc_summary.csv '
        'for exclusions, precharge violations and SOC drift. Do not treat this as equilibrium annual demand. '
        'No annual repetition, dispatch, traffic or concentration results changed. PHEV traffic exhaust consistency '
        'must be checked before combining a new charging model with old fuel-mode emissions.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,default=str))


if __name__=='__main__': main()
