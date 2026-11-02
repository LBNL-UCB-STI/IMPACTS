"""Generate an uncalibrated passenger curve using original notebook core functions.

Independent audit outputs only: never starts power dispatch or overwrites old runs.
EVI API data are requested AFTER demand generation, solely for comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from impacts.pipeline.workflow.charging.ldv_charging import (
    load_notebook_functions, prepare_passenger_stops, assign_stops_indexed, fill_stops,
)


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def metrics(values):
    v = np.asarray(values, dtype=float)
    return dict(daily_kwh_per_1000=float(v.sum()), peak_kw_per_1000=float(v.max()),
                peak_hour=int(v.argmax()), trough_kw_per_1000=float(v.min()), trough_hour=int(v.argmin()),
                peak_to_mean=float(v.max()/v.mean()),
                peak_to_trough=float(v.max()/v.min()) if v.min() else None)


def figures(out, hourly, evi, count):
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(hours, hourly.passenger_MW, lw=2.5, label='Passenger vehicles: notebook core, uncalibrated')
    ax.set(title='Sample-fleet passenger charging demand (no SFBAY expansion)', xlabel='Hour of day', ylabel='Charging power (MW)', xticks=range(0,24,2))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/'passenger_charging_curve.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11,5.5))
    for col in ['LDT_MW','MDT_MW','HDT_MW']:
        ax.plot(hours, hourly[col], label=col.replace('_MW',''), lw=2)
    ax.set(title='Truck supplied charging load (unchanged opportunity-charging model)', xlabel='Hour of day', ylabel='Charging power (MW)', xticks=range(0,24,2))
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(out/'truck_charging_curve.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5.5))
    for col,label in [('passenger_MW','Passenger'),('truck_MW','Trucks (supplied load)'),('passenger_truck_total_MW','Passenger + freight total')]:
        ax.plot(hours,hourly[col],label=label,lw=2.5)
    ax.set(title='Uncalibrated sample-fleet charging demand\nExcludes bus, rail, bicycle and other non-passenger/non-freight types',xlabel='Hour of day',ylabel='Charging power (MW)',xticks=range(0,24,2))
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'passenger_truck_total_curve.png',dpi=180);plt.close(fig)
    if not evi:
        return
    fig,axes=plt.subplots(2,1,figsize=(11,9),sharex=True)
    curves={'Notebook core (sample, uncalibrated)':hourly.passenger_MW.to_numpy()*1e6/count}
    curves.update(evi)
    for i,(name,v) in enumerate(curves.items()):
        axes[0].plot(hours,v,label=name,lw=2.5 if i==0 else 1.6,ls='-' if i==0 else '--')
        axes[1].plot(hours,v/v.sum()*100,label=name,lw=2.5 if i==0 else 1.6,ls='-' if i==0 else '--')
    axes[0].set(title='Passenger charging: independent EVI-Pro Lite comparison',ylabel='Hourly-average kW / 1,000 vehicles')
    axes[1].set(title='Shape comparison only: each curve divided by its own daily energy',ylabel='Share of daily energy (%)',xlabel='Hour of day',xticks=range(0,24,2))
    for ax in axes:ax.grid(alpha=.25)
    axes[0].legend(fontsize=9)
    fig.tight_layout();fig.savefig(out/'passenger_vs_evi_pro_lite.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--events',type=Path,required=True)
    parser.add_argument('--notebooks',type=Path,default=Path('chargingcode'))
    parser.add_argument('--truck-demand-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    original,provenance=load_notebook_functions(args.notebooks)
    save_json(out/'notebook_provenance.json',provenance)
    print('Reading passenger PathTraversal events',flush=True)
    db=duckdb.connect()
    trips=db.execute("SELECT vehicle, vehicleType, departureTime::DOUBLE AS departureTime, arrivalTime::DOUBLE AS arrivalTime, length::DOUBLE AS length FROM read_parquet(?) WHERE type='PathTraversal' AND regexp_matches(vehicleType, '^(cv|ev|hev|phev)-') AND mode IN ('car','car_hov2','car_hov3','ride_hail','ride_hail_pooled')",[str(args.events)]).df()
    db.close()
    stops,input_audit=prepare_passenger_stops(trips,original)
    save_json(out/'input_audit.json',input_audit)
    del trips
    print(json.dumps(input_audit,indent=2),flush=True)
    assigned=assign_stops_indexed(stops,original)
    del stops
    result,vehicles,profile=fill_stops(assigned)
    del assigned
    result.to_parquet(out/'passenger_stop_charging_audit.parquet',index=False)
    vehicles.to_parquet(out/'passenger_vehicle_soc_audit.parquet',index=False)
    session=result.loc[result.energyCharged_kWh.gt(0)].copy()
    session.to_parquet(out/'passenger_charging_sessions.parquet',index=False)
    kept=vehicles.loc[vehicles.retained]
    count=int(input_audit['modeled_vehicles'])
    per1000=profile*1000/count
    maxres=float(abs(vehicles.initial_soc_kwh+vehicles.charging_kwh-vehicles.traction_kwh-vehicles.final_soc_kwh).max())
    summary=dict(input_audit,created_utc=datetime.now(timezone.utc).isoformat(),
                 events_file=str(args.events.resolve()),calibrated_to_evi=False,fleet_scale_factor=1.,
                 original_core_functions_reused=True,complete_original_input_reproduction=False,
                 modeled_vehicles=count,retained_vehicles=len(kept),
                 discarded_post_charge_soc_vehicles=int((~vehicles.retained).sum()),
                 retained_but_below_2kwh_before_charge=int((kept.min_before_charge_kwh<2).sum()),
                 passenger_daily_supplied_MWh=float(profile.sum()/1000),passenger_peak_MW=float(profile.max()/1000),
                 passenger_peak_hour=int(profile.argmax()),
                 retained_traction_MWh=float(kept.traction_kwh.sum()/1000),
                 retained_soc_change_MWh=float((kept.final_soc_kwh-kept.initial_soc_kwh).sum()/1000),
                 maximum_soc_accounting_residual_kWh=maxres,
                 eligible_session_count=int((result.retained & result.energyCharged_kWh.gt(0)).sum()),
                 zero_dwell_charging_sessions=int(((result.parkingTime_s<=0)&result.energyCharged_kWh.gt(0)).sum()),
                 assumptions={'battery_kwh':75,'energy_kwh_km':.17,'home':'longest eligible stop, not plans Home',
                              'charger_access':'L1/L2 at home; L1/L2/DC at other stops; assumed availability',
                              'grid_load':'neutral; no CAISO load input','P1':.45,'P2':.04,'home_share':.8,
                              'seed':42,'arrival_cutoff_seconds':104400,'min_parking_seconds_exclusive':900,
                              'scope':'passenger sample fleet + previously computed freight supplied load; not geographic SFBAY census',
                              'energy_boundary':'Original passenger notebook directly equates energy delivered and SOC gain; no explicit charging loss'},
                 limitations=['Missing plans, person/path-traversal mapping, vehicle battery table, station inventory and CAISO input.',
                              'Original random charging probabilities retained, no daily SOC-balance calibration.',
                              'Notebook post-charge >=2kWh filter retained; before-charge violations separately flagged.',
                              'Constant-power charging is integrated at stop level and allocated to exact hour overlaps; no geographic bin split.',
                              'No annual repetition or air-quality continuation: this is a charging-method diagnostic.',
                              'Total excludes bus/rail/bicycle/other non-passenger and non-freight types.'])
    hourly=pd.DataFrame({'hour_of_day':np.arange(24),'passenger_MW':profile/1000})
    truck_file=args.truck_demand_dir/'truck_hourly_supplied_demand.parquet'
    t=pd.read_parquet(truck_file).iloc[:24]
    hourly['LDT_MW']=t.charging_demand_ld1_MWh+t.charging_demand_ld3_MWh
    hourly['MDT_MW']=t.charging_demand_mdv_MWh
    hourly['HDT_MW']=t.charging_demand_hdv_MWh+t.charging_demand_hdt_MWh
    hourly['truck_MW']=hourly[['LDT_MW','MDT_MW','HDT_MW']].sum(axis=1)
    hourly['passenger_truck_total_MW']=hourly.passenger_MW+hourly.truck_MW
    summary['truck_curve_source']=str(truck_file.resolve())
    summary['truck_curve_sha256']=hashlib.sha256(truck_file.read_bytes()).hexdigest()
    summary['truck_daily_supplied_MWh']=float(hourly.truck_MW.sum())
    summary['truck_daily_unmet_MWh']=json.loads((args.truck_demand_dir/'beam_charging_summary.json').read_text())['daily_truck_unmet_MWh']
    summary['passenger_truck_daily_supplied_MWh']=float(hourly.passenger_truck_total_MW.sum())
    hourly.to_csv(out/'hourly_charging_curve.csv',index=False)
    hourly.to_parquet(out/'hourly_charging_curve.parquet',index=False)
    save_json(out/'charging_reuse_summary.json',summary)
    print('Demand generation complete. Fetching independent EVI comparison ONLY.',flush=True)
    base=dict(fleet_size=1000,mean_dvmt=35,pev_type='BEV250',pev_dist='BEV',class_dist='Sedan',
              home_access_dist='HA100',home_power_dist='MostL2',work_power_dist='MostL2',pref_dist='Home80',
              res_charging='min_delay',work_charging='min_delay')
    url='https://developer.nlr.gov/api/evi-pro-lite/v1/daily-load-profile'
    evi={};records=[dict(series='Notebook core (sample, uncalibrated)',**metrics(per1000))]
    for temp in [20,10]:
        params=dict(base,temp_c=temp)
        try:
            response=requests.get(url,params=dict(params,api_key='DEMO_KEY'),timeout=45)
            response.raise_for_status()
            data=response.json()
            save_json(out/f'evi_pro_lite_{temp}C_response.json',dict(endpoint=url,inputs=params,retrieved_utc=datetime.now(timezone.utc).isoformat(),response=data))
            for day,parts in data['results'].items():
                arrays=[np.asarray(a,dtype=float) for a in parts.values()]
                if not arrays or any(a.shape!=(96,) for a in arrays):raise ValueError('Expected 96 quarter-hour kW bins')
                v=sum(arrays).reshape(24,4).mean(axis=1)
                label=f'EVI {temp}C '+('weekday' if day.startswith('weekday') else 'weekend')
                evi[label]=v
                records.append(dict(series=label,**metrics(v),
                                    sample_energy_difference_pct=float((per1000.sum()/v.sum()-1)*100),
                                    hourly_correlation=float(np.corrcoef(per1000,v)[0,1]),
                                    shape_total_variation=float(abs(per1000/per1000.sum()-v/v.sum()).sum()/2)))
        except (requests.RequestException,ValueError,KeyError) as exc:
            save_json(out/f'evi_pro_lite_{temp}C_error.json',{'error':str(exc),'inputs':params})
    pd.DataFrame(records).to_csv(out/'evi_comparison_metrics.csv',index=False)
    compare=pd.DataFrame({'hour_of_day':np.arange(24),'sample_kw_per_1000':per1000,**evi})
    compare.to_csv(out/'evi_comparison_hourly.csv',index=False)
    figures(out,hourly,evi,count)
    notes=f'''# Uncalibrated charging-code reuse audit

This is NOT an EVI-calibrated result or a new air-quality run. Prior outputs are untouched.

Original notebook source functions are loaded directly; hashes and names are in notebook_provenance.json.
Indexed stop/type assignment is regression-tested against the original functions. SOC integration uses
the same constant-power equation; actual parking windows replace the erroneous zero-dwell=24h adapter.
Pandas 3 rejects some weighted draws used by the notebook; indexed assignment preserves the original
NumPy/pandas-2 weighted-without-replacement semantics. Reference tests apply that compatibility shim
only to the notebook reference. Stop integration is also checked against the original hour-bin function,
including the first partial-hour post-charge SOC filter, not merely end-of-stop SOC.
Missing plans, battery tables, infrastructure and CAISO are explicitly assumed, so full notebook-input
reproduction has NOT been achieved. The original 29h arrival cutoff and >15min stop rule are used.
Overlapping or >24h vehicle itineraries are excluded, counted in input_audit.json. End-of-itinerary
traction not represented by eligible stops is reported there. No hidden mileage/fleet expansion.

Passenger P1=.45, P2=.04, SH=.8, 75kWh, .17kWh/km, seed42: unchanged, NOT fitted to EVI.
SOC drift is reported, not corrected. A 24h plot does not certify an energy-balanced repeating day.
Before-charge SOC violations and the original post-charge vehicle exclusions are reported separately.
Truck curves reuse the established class-specific opportunity-charging supplied-load outputs; unmet
truck energy remains excluded and explicitly reported. Other transit/nonroad curves are NOT included.

EVI official API: {url}
1,000 BEV250 Sedans, 35 miles/day, 10/20C, HA100, Home80, MostL2, immediate charging.
Each 96-bin returned kW profile is averaged four-at-a-time to 24 hourly kW bins.
Sample denominator: {count:,} modeled passenger vehicles INCLUDING vehicles assigned no charging and
post-charge-discarded vehicles (unserved load is not silently imputed). Sample mean itinerary mileage:
{input_audit['modeled_daily_distance_km']/count/1.609344:.4f} miles/vehicle. EVI is an independent
scenario, not observed SFBAY ground truth. The lower shape panel divides each curve by its own daily
energy for visual comparison only; it is NOT used to alter any charging output.

Files: hourly_charging_curve.csv/parquet; passenger_stop_charging_audit.parquet;
passenger_vehicle_soc_audit.parquet; passenger_charging_sessions.parquet; charging_reuse_summary.json;
evi_pro_lite_*_response.json (raw official data); evi_comparison_metrics.csv; evi_comparison_hourly.csv.
'''
    (out/'README.md').write_text(notes,encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    print(pd.DataFrame(records).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
