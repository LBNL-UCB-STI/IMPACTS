"""Isolated baseline 0.events traffic plus designed LDV and collaborator freight power."""
from research_paths import ISRM_PATH, BREATHE_ROOT, BEAM_DATA_ROOT, CHARGING_REFERENCE
from pathlib import Path
import argparse
import csv
import hashlib
import json
import duckdb
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'examples/pilates/impacts/impacts_output/baseline0-ldv-home80-freight3-10x-20261026'
EVENTS = ROOT/'examples/pilates/beam/beam_output/ITERS/it.0/0.events.csv.gz'
VEH = ROOT/'examples/pilates/beam/production/sfbay/vehicle-tech'


def save(name, value):
    (OUT/name).write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=OUT/'inputs'; inputs.mkdir(exist_ok=True)
    target=inputs/'baseline_events.parquet'
    if not target.exists():
        save('run_status.json',dict(status='running',stage='input_extraction',final_concentrations_ready=False))
        with duckdb.connect() as db:
            db.execute('SET threads=4');db.execute("SET memory_limit='8GB'")
            print('Extracting PathTraversal and actstart records; original CSV is read-only',flush=True)
            destination=target.as_posix().replace("'", "''")
            db.execute(f"""COPY (SELECT type,vehicle,vehicleType,mode,driver,person,actType,
                try_cast(time AS DOUBLE) AS time,try_cast(departureTime AS DOUBLE) AS departureTime,
                try_cast(arrivalTime AS DOUBLE) AS arrivalTime,try_cast(length AS DOUBLE) AS length,
                try_cast(weight AS DOUBLE) AS weight,links,linkTravelTime,
                primaryFuelType,secondaryFuelType,try_cast(primaryFuelLevel AS DOUBLE) AS primaryFuelLevel,
                try_cast(primaryFuel AS DOUBLE) AS primaryFuel,try_cast(secondaryFuel AS DOUBLE) AS secondaryFuel
                FROM read_csv(?,header=true,all_varchar=true,delim=',',quote='"',escape='"')
                WHERE type IN ('PathTraversal','actstart')) TO '{destination}' (FORMAT PARQUET,COMPRESSION ZSTD)""",
                [str(EVENTS)])
    with duckdb.connect() as db:
        d=db.execute("""SELECT vehicleType,mode,primaryFuelType,secondaryFuelType,
            count(*) trips,count(DISTINCT vehicle) vehicles,sum(length)/1000 distance_km
            FROM read_parquet(?) WHERE type='PathTraversal' GROUP BY ALL ORDER BY vehicles DESC""",[str(target)]).df()
        d.to_csv(inputs/'vehicle_type_inventory.csv',index=False)
        fuels=db.execute("""SELECT primaryFuelType,secondaryFuelType,mode,
            count(DISTINCT vehicle) vehicles,count(*) trips FROM read_parquet(?)
            WHERE type='PathTraversal' GROUP BY ALL ORDER BY vehicles DESC""",[str(target)]).df()
        fuels.to_csv(inputs/'fuel_inventory.csv',index=False)
    catalog=pd.concat([pd.read_csv(p,low_memory=False).assign(metadata_file=str(p)) for p in VEH.glob('vehicleTypes*.csv')],ignore_index=True)
    d['exact_metadata_match']=d.vehicleType.isin(catalog.vehicleTypeId)
    d.loc[~d.exact_metadata_match].to_csv(inputs/'unmatched_vehicle_types.csv',index=False)
    catalog.loc[catalog.vehicleTypeId.isin(d.vehicleType)].to_csv(inputs/'matched_vehicle_metadata.csv',index=False)
    ev=d.loc[d.primaryFuelType.fillna('').str.lower().eq('electricity')]
    report=dict(events=str(EVENTS),source_bytes=EVENTS.stat().st_size,source_mtime_ns=EVENTS.stat().st_mtime_ns,
        derived_events=str(target),vehicle_type_count=d.vehicleType.nunique(),
        exact_metadata_match_types=d.loc[d.exact_metadata_match].vehicleType.nunique(),
        electric_types=ev.to_dict('records'),fuel_summary=fuels.to_dict('records'),
        requested=dict(baseline='0.events',traffic_scale=10,ldv_scale=10,freight_scale=10,
            freight_scenarios=['BF-06EP','BF-1EP','HF-1EP'],freight_additive_not_ice_replacement=True,
            ldv_model='Designed BEV Home80/PHEV Home80, applied to baseline0 existing EVs',
            power_isrm_layer=0,aermod_radius_m=3000,isrm_primary_removal_radius_m=3000,
            isrm_primary_removal_receptor_overlap_threshold=.5))
    save('input_audit.json',report)
    print(json.dumps(report,indent=2,default=str),flush=True)


def freight_power():
    import continue_beam_charging_pipeline as power
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=ROOT/'OUTPUT/collaborator-truck-charging-curves-20261026-final/hourly_charging_diagnostics.csv'
    data=pd.read_csv(source)
    summary=[]
    for scenario in ['BF-06EP','BF-1EP','HF-1EP']:
        target=OUT/'freight_power'/scenario
        demand=target/'charging_demand'
        demand.mkdir(parents=True,exist_ok=True)
        d=data.loc[data.scenario.eq(scenario)&data.powertrain.eq('BE')].sort_values('elapsed_hour_start')
        day=d.loc[d.elapsed_hour_start.lt(24)]
        np.testing.assert_array_equal(day.elapsed_hour_start,np.arange(24))
        load=day.energy_kWh.to_numpy()*10/1000
        pd.DataFrame(dict(hour_of_day=np.arange(24),sample_kWh=day.energy_kWh.to_numpy(),
            charging_MW_10x=load)).to_csv(demand/'freight_daily_curve_10x.csv',index=False)
        annual=pd.DataFrame(dict(hour_of_year=np.arange(8760),charging_demand_MWh=np.tile(load,365)))
        np.testing.assert_allclose(annual.charging_demand_MWh.sum(),load.sum()*365)
        annual.to_parquet(demand/'sfbay_hourly_charging_demand.parquet',index=False)
        qa=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            curve_scope=f'Collaborator {scenario} BE recorded first-24h charging only, 10x, additive scenario',
            fleet_scale_factor=10,annualization_days=365,daily_energy_MWh=float(load.sum()),
            annual_energy_MWh=float(load.sum()*365),peak_MW=float(load.max()),peak_hour=int(load.argmax()),
            excluded_after24_MWh_10x=float(d.loc[d.elapsed_hour_start.ge(24),'energy_kWh'].sum()*.01),
            source_vehicle_classes=['LD1','LD3','MDV','HDV','HDT'],H2FC_included=False,
            timing='First 24 elapsed-hour bins retained; time treated as charging start; no midnight folding or timezone shift',
            energy_boundary='Reported charge energy treated as end-use load proxy; no additional charger-loss multiplier',
            caveats=['Successful externalZone1 records only; complete depot coverage unconfirmed',
                'SOC and simultaneous-event flags preserved; supplied-load scenario, not validated full charging needs',
                'Daily profile repeated 365 times; not a year-matched historical forecast',
                '10x scaling explicitly requested; no ICE freight emissions displaced'])
        (demand/'beam_charging_summary.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
        ax.plot(np.arange(24),load,lw=2.3)
        ax.set(xlabel='Elapsed hour (start of 1-hour interval)',ylabel='Charging load (MW)',
            title=f'{scenario}: collaborator BE freight charging, 10x',xlim=(0,23),ylim=(0,None))
        ax.grid(alpha=.25)
        fig.savefig(demand/'freight_charging_10x.png',dpi=160);plt.close(fig)
        power.DEMAND=demand
        result=target/'power_generation/power_generation_isrm_concentrations.parquet'
        if not result.exists():
            print('FREIGHT DISPATCH',scenario,flush=True)
            power.dispatch(target)
            power.power_isrm(target,ISRM_PATH,source_layer_index=0)
        result_summary=json.loads((target/'power_generation/power_generation_summary.json').read_text())
        result_summary.update(traffic_baseline=str(EVENTS),traffic_included=False,
            contribution='freight charging power only',final_integrated_concentration=False)
        (target/'power_generation/power_generation_summary.json').write_text(json.dumps(result_summary,indent=2),encoding='utf-8')
        conc=pd.read_parquet(result)
        summary.append(dict(scenario=scenario,daily_energy_MWh_10x=qa['daily_energy_MWh'],
            annual_energy_MWh_10x=qa['annual_energy_MWh'],peak_load_MW=qa['peak_MW'],
            peak_hour=qa['peak_hour'],selected_emitting_plants=result_summary['selected_emitting_plants'],
            selected_source_cells=result_summary['selected_source_cells'],
            max_power_TotalPM25_ug_m3=float(conc.power_TotalPM25.max()),
            output=str(target)))
        print('FREIGHT COMPLETE',scenario,flush=True)
    pd.DataFrame(summary).to_csv(OUT/'freight_power_summary.csv',index=False)
    save('run_status.json',dict(status='partial_complete_waiting_input',
        completed='Three freight charging -> hourly allocation -> annual emissions -> Layer0 ISRM products',
        missing='0.events matching vehicleTypes / exact EMFAC mapping and LDV battery metadata',
        traffic_completed=False,ldv_completed=False,final_concentrations_ready=False))
    print(json.dumps(summary,indent=2),flush=True)


def mapping_audit():
    """Record distinct EVs and exhaust local vehicle-tech tables before requesting input."""
    paths=set()
    for base in [ROOT/'examples/pilates/beam/production/sfbay/vehicle-tech',
                 ROOT/'examples/pilates/impacts/impacts_input/emfac',
                 BEAM_DATA_ROOT/'vehicle-tech']:
        for p in base.rglob('*.csv'):
            if any(t in p.name.lower() for t in ['vehicletypes','crosswalk']): paths.add(p)
    known=set();searched=[]
    for p in sorted(paths):
        # Identifier discovery only: tolerate ragged trailing descriptive columns,
        # but never use these rows to assign physical or emissions parameters.
        with p.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                for col in ['vehicleTypeId','sourceVehicleTypeId','mappingVehicleTypeId']:
                    if row.get(col): known.add(row[col])
        searched.append(str(p))
    with duckdb.connect() as db:
        ev=db.execute("""SELECT DISTINCT vehicle,vehicleType,
            CASE WHEN lower(secondaryFuelType)='gasoline' THEN 'PHEV' ELSE 'BEV' END powertrain
            FROM read_parquet(?) WHERE type='PathTraversal' AND lower(primaryFuelType)='electricity'
            AND mode IN ('car','car_hov2','car_hov3','ride_hail','ride_hail_pooled')""",
            [str(OUT/'inputs/baseline_events.parquet')]).df()
    ev['metadata_key_found']=ev.vehicleType.isin(known)
    ev.to_parquet(OUT/'inputs/existing_ldv_ev_inventory.parquet',index=False)
    summary=dict(existing_ev_counts=ev.groupby('powertrain').vehicle.nunique().to_dict(),
        missing_ev_types=sorted(set(ev.loc[~ev.metadata_key_found,'vehicleType'])),
        searched_vehicle_metadata_files=searched,
        required_input='Exact vehicleTypes files/crosswalk used by 0.events, including battery capacity and EMFAC category/year/fuel',
        original_skims_not_substituted=True,old_fc07_charging_not_substituted=True)
    save('missing_metadata_audit.json',summary)
    old=json.loads((OUT/'input_audit.json').read_text())
    assert EVENTS.stat().st_size==old['source_bytes'] and EVENTS.stat().st_mtime_ns==old['source_mtime_ns']
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','freight-power','mapping-audit'])
    args=parser.parse_args()
    {'prepare':prepare,'freight-power':freight_power,'mapping-audit':mapping_audit}[args.stage]()
