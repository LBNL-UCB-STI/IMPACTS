"""Independent native-EV baseline: 10x sampled road traffic + recorded charging.

Reuses validated source-resolved traffic products and the existing power allocator,
ISRM, exposure, plots and original-style web exporter. No old output is modified.
"""
from research_paths import ISRM_PATH, BREATHE_ROOT, BEAM_DATA_ROOT, CHARGING_REFERENCE
from pathlib import Path
import argparse
import gc
import hashlib
import json
import logging
import shutil
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import continue_beam_charging_pipeline as continuation
from impacts.pipeline.workflow.native_ev_charging import ROAD_MODES, reconstruct_sessions
from impacts.pipeline.workflow.power_generation import _safe_profile_token, write_charging_curve_outputs

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT/'examples/pilates/impacts/impacts_output'
EVENTS = ROOT/'chargingcode/beam-fc07-v260419-it3/3.events.parquet'
REFERENCE = OUTPUTS/'impacts-postsim--sfbay--base--20260901-165739'
SPLIT = OUTPUTS/'events-fc07-traffic-only-20260925-baseline-mixture-v1-ldv-freight10x'
TOTAL = OUTPUTS/'events-fc07-traffic-only-20260925-baseline-mixture-v1-partial-isrm103-fulltraffic10x'
DEFAULT_OUT = OUTPUTS/'events-fc07-baseline-native-ev10x-20260928'
ISRM = ISRM_PATH


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def charging(out):
    target=out/'charging_demand'
    if target.exists() and any(target.iterdir()): raise FileExistsError(target)
    target.mkdir(exist_ok=True)
    db=duckdb.connect()
    sessions=db.execute("SELECT * FROM read_parquet(?) WHERE type='RefuelSessionEvent'",[str(EVENTS)]).fetchdf()
    plugs=db.execute("SELECT vehicle,parkingZoneId,time FROM read_parquet(?) WHERE type='ChargingPlugInEvent'",[str(EVENTS)]).fetchdf()
    vehicles=db.execute("SELECT vehicle,vehicleType,primaryFuelType,secondaryFuelType,mode,count(*) trips FROM read_parquet(?) WHERE type='PathTraversal' GROUP BY ALL",[str(EVENTS)]).fetchdf()
    road=vehicles.loc[vehicles['mode'].isin(ROAD_MODES)].copy()
    electric=road.loc[road.primaryFuelType.eq('Electricity') | road.secondaryFuelType.eq('Electricity')]
    profiles,audit=reconstruct_sessions(sessions,plugs,set(electric.vehicleType),scale=10.,quarantine_invalid=True)
    # Independently verify delivered energy against the change in battery fuel.
    balance=db.execute("""WITH r AS (SELECT vehicle,parkingZoneId,time,fuel FROM read_parquet(?) WHERE type='RefuelSessionEvent'),
        p AS (SELECT vehicle,parkingZoneId,time,primaryFuelLevel FROM read_parquet(?) WHERE type='ChargingPlugInEvent'),
        o AS (SELECT vehicle,parkingZoneId,time,primaryFuelLevel FROM read_parquet(?) WHERE type='ChargingPlugOutEvent')
        SELECT r.*,p.time AS plug_time,o.primaryFuelLevel-p.primaryFuelLevel-r.fuel AS residual_J
        FROM r ASOF LEFT JOIN p ON r.vehicle=p.vehicle AND r.parkingZoneId=p.parkingZoneId AND r.time>=p.time
        LEFT JOIN o ON r.vehicle=o.vehicle AND r.parkingZoneId=o.parkingZoneId AND r.time=o.time""",[str(EVENTS)]*3).fetchdf()
    bad=balance.loc[balance.residual_J.abs().gt(10)|balance.residual_J.isna()]
    assert set(zip(bad.vehicle,bad.time)).issubset(set(zip(audit.loc[audit.quarantined,'vehicle'],audit.loc[audit.quarantined,'time'])))
    balance.to_csv(target/'native_energy_balance_audit.csv',index=False)
    audit.loc[audit.quarantined].to_csv(target/'quarantined_native_sessions.csv',index=False)
    if audit.vehicle.str.startswith('virtual').any():
        raise ValueError('Native virtual scale-up charging found; investigate before applying 10x')
    audit.to_parquet(target/'native_charging_session_audit.parquet',index=False)
    # CSV is also a lightweight, directly inspectable audit of all 1,909 records.
    audit.to_csv(target/'native_charging_session_audit.csv',index=False)
    electric.to_csv(target/'electric_road_vehicle_inventory.csv',index=False)
    daily=pd.DataFrame({'hour_of_day':range(24)})
    demand=pd.DataFrame({'hour_of_year':range(8760)})
    for kind,energy in sorted(profiles.items()):
        col=f'charging_demand_{_safe_profile_token(kind)}_MWh'
        daily[col]=energy/1000.
        demand[col]=np.tile(energy/1000.,365)
    demand['charging_demand_MWh']=demand.filter(like='charging_demand_').sum(axis=1)
    daily['charging_demand_MWh']=daily.filter(like='charging_demand_').sum(axis=1)
    daily.to_csv(target/'native_ev_hourly_curve_10x.csv',index=False)
    demand.to_parquet(target/'sfbay_hourly_charging_demand.parquet',index=False)
    inventory=electric.groupby('vehicleType').vehicle.nunique().to_dict()
    selected=audit.loc[audit.included_sample_kwh.gt(0)]
    summary=dict(charging_demand_source='Recorded BEAM RefuelSessionEvent + ChargingPlugInEvent, existing electric road vehicles only',
        curve_scope='Recorded supplied charging, clipped to 0-24h; not total traction needs or an equilibrium all-electric scenario',
        events_file=str(EVENTS),fleet_scale_factor=10.,traffic_sample_fraction=.1,
        annualization_days=365,time_window_seconds=[0,86400],
        native_fuel_units='J; divide by 3,600,000 for kWh',
        time_rule='BEAM simulation-local seconds; no UTC conversion; no modulo folding of next-day events',
        within_session_timing='Uniform mean power over matched plug-in to disconnect; exact internal charging cycles unavailable',
        charging_efficiency_assumption=1.,
        energy_boundary='Recorded delivered energy used as end-use electricity proxy, with no additional charger-loss factor; Cambium grid losses applied separately',
        electric_road_vehicle_counts=inventory,
        electric_road_vehicles=int(electric.vehicle.nunique()),
        refuel_sessions=int(len(audit)),zero_energy_sessions=int(audit.delivered_kwh.eq(0).sum()),
        quarantined_sessions=int(audit.quarantined.sum()),
        quarantined_sample_kWh=float(audit.loc[audit.quarantined,'delivered_kwh'].sum()),
        valid_sessions_match_plugin_unplug_energy=True,
        all_recorded_charging_kWh=float(audit.delivered_kwh.sum()),
        included_positive_sessions=int(len(selected)),charging_vehicles=int(selected.vehicle.nunique()),
        included_sample_daily_kWh=float(audit.included_sample_kwh.sum()),
        excluded_energy_kWh=float(audit.excluded_sample_kwh.sum()),
        positive_energy_fallback_sessions=int((audit.delivered_kwh.gt(0)&audit.timing_method.str.contains('fallback')).sum()),
        daily_energy_MWh=float(daily.charging_demand_MWh.sum()),
        annual_energy_MWh=float(demand.charging_demand_MWh.sum()),
        annual_peak_MW=float(demand.charging_demand_MWh.max()),
        peak_hour=int(daily.charging_demand_MWh.idxmax()),
        phev_charging_sessions=int(audit.vehicleType.str.startswith('phev-').sum()),
        truck_charging_sessions=int(audit.vehicleType.str.startswith('ft-').sum()),
        limitations=['Native BEAM simulation records, not observed real-world charging.',
            'Only recorded charging is represented: no inferred overnight top-up or PHEV/truck charging.',
            '365 repetitions annualize one recorded 0-24h window; no claim of annual representativeness or cyclic SOC equilibrium.',
            'Vehicle electricity losses are not independently measured; delivered energy approximates end-use load.',
            'Existing p9 grid scenario uses Cambium 2040 and eGRID 2023; not year-matched historical baseline.'])
    np.testing.assert_allclose(summary['annual_energy_MWh'],summary['included_sample_daily_kWh']*10/1000*365)
    save(target/'beam_charging_summary.json',summary)
    write_charging_curve_outputs(demand,target,scenario_label='Native recorded BEV charging; 10x sample expansion; 0-24h')
    print(json.dumps(summary,indent=2),flush=True)


def combine_concentrations(left,right,target,key):
    """Outer-join independent source products; preserve GeoParquet metadata."""
    db=duckdb.connect();db.execute('SET threads=4');db.execute("SET memory_limit='16GB'")
    db.read_parquet(str(left)).create_view('l')
    db.read_parquet(str(right)).create_view('r')
    schema=pq.read_schema(left)
    if schema.names!=pq.read_schema(right).names:raise ValueError('Concentration schemas differ')
    for alias in ['l','r']:
        assert db.execute(f'SELECT count(*)=count(DISTINCT {key}) FROM {alias}').fetchone()[0]
    assert db.execute(f'SELECT count(*) FROM l JOIN r USING ({key}) WHERE l.geometry!=r.geometry').fetchone()[0]==0
    concentration={'SOA','pNO3','pNH4','pSO4','PrimaryPM25','BC','NO2','TotalPM25',
                   'InMAPOnly_PrimaryPM25','InMAPOnly_TotalPM25','InMAPOnly_BC','InMAPOnly_NO2'}
    expressions=[]
    for col in schema.names:
        if col in concentration:expr=f'coalesce(l."{col}",0)+coalesce(r."{col}",0)'
        elif col.startswith('has_aermod_'):expr=f'coalesce(l."{col}",false) OR coalesce(r."{col}",false)'
        else:expr=f'coalesce(l."{col}",r."{col}")'
        expressions.append(f'{expr} AS "{col}"')
    reader=db.execute(f'SELECT {",".join(expressions)} FROM l FULL OUTER JOIN r USING ({key}) ORDER BY {key}').to_arrow_reader(131072)
    with pq.ParquetWriter(target,schema,compression='zstd') as writer:
        for batch in reader:
            writer.write_table(pa.Table.from_batches([batch]).cast(schema))
    for col in concentration & set(schema.names):
        expected=db.execute(f'SELECT (SELECT sum("{col}") FROM l)+(SELECT sum("{col}") FROM r)').fetchone()[0]
        actual=db.execute(f'SELECT sum("{col}") FROM read_parquet(?)',[str(target)]).fetchone()[0]
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
    db.close()


def traffic(out,scope):
    folder=out/'concentrations';folder.mkdir(exist_ok=True)
    manifest=yaml.safe_load((TOTAL/'pipeline_manifest.yaml').read_text())
    for model,key in [('inmap','inmap_cell_id'),('aermod','aermod_cell_id')]:
        filename=f'beam_{model}_concentrations.parquet'
        target=folder/filename
        if target.exists():raise FileExistsError(target)
        if scope=='ldv-freight':
            combine_concentrations(SPLIT/'LDV/concentrations'/filename,SPLIT/'Freight/concentrations'/filename,target,key)
        else:shutil.copy2(TOTAL/'concentrations'/filename,target)
        manifest['outputs'][f'beam_{model}_concentrations']=str(target)
    save(out/'traffic_reuse_audit.json',dict(traffic_scope=scope,traffic_scale_already_applied=10.,
        further_traffic_scaling_applied=False,method='Sum independent source-resolved LDV and Freight operators' if scope=='ldv-freight' else 'Reuse validated fulltraffic10x',
        source_directory=str(SPLIT if scope=='ldv-freight' else TOTAL),
        approved_unavailable_isrm_sources=103,missing_sources_redistributed=False,
        local_removal='4500m source-neighborhood removal for traffic primary PM2.5, BC, NO2 only; AERMOD added directly; secondary and power unchanged',
        power_generation_in_traffic_inputs=False))
    (out/'traffic_pipeline_manifest.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False),encoding='utf-8')


def integrate(out):
    from impacts.manifest.schema import PipelineConfig
    from impacts.pipeline.workflow.step4_prepare_exposure import run
    manifest=yaml.safe_load((out/'traffic_pipeline_manifest.yaml').read_text())
    preprocess=yaml.safe_load((TOTAL/'preprocess_manifest.yaml').read_text())
    folder=out/'exposure';folder.mkdir(exist_ok=True)
    run(pipeline=PipelineConfig.from_dict(manifest['pipeline']),raw_dir=folder,
        inmap_concentrations_path=manifest['outputs']['beam_inmap_concentrations'],
        aermod_concentrations_path=manifest['outputs']['beam_aermod_concentrations'],
        power_generation_concentrations_path=str(out/'power_generation/power_generation_isrm_concentrations.parquet'),
        manifest_inputs=preprocess['inputs'])


def postprocess(out):
    """Use original layout with equal power added to both comparison methods."""
    from impacts.pipeline.postprocess.step4_plot_concentrations import run as plot
    from impacts.pipeline.postprocess.step5_plot_exposure import run as exposure_plot
    from impacts.pipeline.postprocess.step8_human_exposure import run as human
    main=out/'exposure/beam_concentration_distribution.parquet'
    comparison=out/'exposure/beam_concentration_comparison_with_power.parquet'
    reader=pq.ParquetFile(main)
    metadata=dict(reader.schema_arrow.metadata or {})
    metadata[b'impacts_comparison_scope']=b'InMAP-only includes full traffic ISRM + same power as enhanced; native component file remains traffic-only for inmap_only_*'
    schema=reader.schema_arrow.with_metadata(metadata)
    with pq.ParquetWriter(comparison,schema,compression='zstd') as writer:
        for batch in reader.iter_batches(batch_size=131072):
            table=pa.Table.from_batches([batch])
            for base,power in [('inmap_only_PrimaryPM25','power_PrimaryPM25'),('inmap_only_TotalPM25','power_TotalPM25')]:
                index=table.schema.get_field_index(base)
                values=table[base].to_numpy()+table[power].to_numpy()
                table=table.set_column(index,schema.field(base),pa.array(values))
            writer.write_table(table.cast(schema))
    settings=yaml.safe_load((ROOT/'examples/pilates/settings.yaml').read_text())
    analysis=settings['impacts']['analysis']['human_exposure']
    plants=str(out/'power_generation/plant_emissions_used_for_concentration.csv')
    note='Native EV charging 10x; traffic 10x; partial traffic ISRM; power Layer 0 / Cambium 2040.'
    products=plot(concentration_path=str(comparison),network_path=str(REFERENCE/'preprocess/beam_osm_mapped.parquet'),
        power_plants_path=plants,pm25_component_region_path=analysis['ces_tracts_file'],
        output_dir=out/'postprocess/concentrations',scenario_note=note)
    save(out/'concentration_plot_manifest.json',products);gc.collect()
    products=exposure_plot(population_path=str(out/'exposure/beam_population_counts.parquet'),
        concentration_path=str(comparison),network_path=str(REFERENCE/'preprocess/beam_osm_mapped.parquet'),
        inmap_cells_path=str(out/'concentrations/beam_inmap_concentrations.parquet'),output_dir=out/'postprocess/exposure')
    save(out/'exposure_plot_manifest.json',products);gc.collect()
    products=human(population_path=str(out/'exposure/beam_population_counts.parquet'),
        concentration_path=str(comparison),ces_tracts_path=analysis['ces_tracts_file'],
        sensitive_facilities_path=analysis['sensitive_facilities_file'],schools_layer=analysis['schools_layer'],
        ltc_layer=analysis['ltc_layer'],output_dir=out/'postprocess/human_exposure',power_plants_path=plants)
    save(out/'human_exposure_manifest.json',products)
    html=Path(products['webmap'])
    text=html.read_text(encoding='utf-8')
    warning=f'<div style="padding:8px;background:#fff3cd">{note} Same power contribution included in BOTH comparison methods. Recorded supplied load only; not year-matched historical baseline.</div>'
    html.write_text(text.replace('<body>','<body>'+warning,1),encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'scripts/export_original_style_human_exposure_tables.py'),
        '--input',str(out/'postprocess/human_exposure/human_exposure_ces_tracts.parquet'),
        '--output-dir',str(out/'postprocess/human_exposure/tables')],check=True)


def verify(out):
    digest=hashlib.sha256()
    with EVENTS.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    archived=json.loads((OUTPUTS/'beam-charging-fc07-v260419-it3--20260923-truck-model-v1/charging_demand/truck_model_parameters.json').read_text())
    assert digest.hexdigest()==archived['events_sha256'],'Events changed since existing traffic scenario; rerun traffic'
    db=duckdb.connect()
    road_count=db.execute("SELECT count(*) FROM read_parquet(?) WHERE type='PathTraversal' AND mode IN ('car','car_hov2','car_hov3','bus','ride_hail','ride_hail_pooled')",[str(EVENTS)]).fetchone()[0]
    traffic_audit=json.loads((OUTPUTS/'events-fc07-traffic-only-20260925-baseline-mixture-v1/activity_audit.json').read_text())
    assert road_count==traffic_audit['road_trips']+traffic_audit['excluded_missing_route_records']
    db.close()
    save(out/'source_validation.json',dict(events_sha256=digest.hexdigest(),events_size_bytes=EVENTS.stat().st_size,
        unchanged_from_archived_events=True,road_trip_count=road_count,
        traffic_trips_with_routes=traffic_audit['road_trips'],
        previously_excluded_missing_route_records=traffic_audit['excluded_missing_route_records'],
        road_trip_count_matches_traffic_input=True))
    continuation.verify(out)
    concentration_path=out/'concentration_validation.json'
    concentration_check=json.loads(concentration_path.read_text())
    concentration_check['inmap_only_fields_scope']='Traffic-only ISRM with local responses retained, but previously approved unavailable source rows still omitted'
    concentration_check['traffic_isrm_source_coverage']='Partial; no redistribution of missing emissions; AERMOD sources retained'
    save(concentration_path,concentration_check)
    continuation.audit(out)
    continuation.validate_products(out)
    product_path=out/'product_validation.json'
    product_check=json.loads(product_path.read_text())
    product_check['browser_interaction_test']='Not performed in this run; static HTML raster, layer and plant-marker checks passed.'
    save(product_path,product_check)
    summary=json.loads((out/'charging_demand/beam_charging_summary.json').read_text())
    power=json.loads((out/'power_generation/power_generation_summary.json').read_text())
    assert power['power_source_layer_index']==0 and not power['power_local_local_removal_applied']
    assert summary['truck_charging_sessions']==0 and summary['fleet_scale_factor']==10.
    assert power['hourly_annual_totals_match']
    scope=json.loads((out/'scenario.json').read_text())['traffic_scope']
    if scope=='ldv-freight':
        db=duckdb.connect();db.execute('SET threads=4');db.execute("SET memory_limit='16GB'")
        for alias,path in [('m',out/'exposure/beam_concentration_distribution.parquet'),
                           ('l',SPLIT/'products/LDV_concentrations.parquet'),
                           ('f',SPLIT/'products/Freight_concentrations.parquet')]:
            db.read_parquet(str(path)).create_view(alias)
        expressions=[]
        for field in ['PrimaryPM25','SecondaryPM25','TotalPM25','BC','NO2','inmap_only_TotalPM25']:
            addition=f'+m.power_{field}' if field in ['PrimaryPM25','SecondaryPM25','TotalPM25'] else ''
            expressions.append(f'max(abs(m."{field}"-(l."{field}"+f."{field}"{addition}))) AS "{field}"')
        differences=db.execute(f'SELECT count(*) n,{",".join(expressions)} FROM m JOIN l USING (aermod_cell_id) JOIN f USING (aermod_cell_id)').fetchdf().iloc[0].to_dict()
        assert int(differences.pop('n'))==pq.read_metadata(out/'exposure/beam_concentration_distribution.parquet').num_rows
        assert max(differences.values())<1e-9,differences
        save(out/'traffic_plus_power_identity_check.json',dict(passed=True,max_absolute_differences= differences,
            identity='Final = existing LDV10x + existing Freight10x + independently modeled native-EV10x power; no extra traffic scaling'))
        db.close()
    main=pq.ParquetFile(out/'exposure/beam_concentration_distribution.parquet')
    comp=pq.ParquetFile(out/'exposure/beam_concentration_comparison_with_power.parquet')
    fields=['aermod_cell_id','inmap_only_TotalPM25','power_TotalPM25','TotalPM25']
    for x,y in zip(main.iter_batches(columns=fields),comp.iter_batches(columns=fields),strict=True):
        a,b=x.to_pandas(),y.to_pandas()
        np.testing.assert_array_equal(a.aermod_cell_id,b.aermod_cell_id)
        np.testing.assert_allclose(a.inmap_only_TotalPM25+a.power_TotalPM25,b.inmap_only_TotalPM25)
        np.testing.assert_allclose(a.TotalPM25,b.TotalPM25)
    population=pd.read_parquet(out/'exposure/beam_population_counts.parquet',columns=['aermod_cell_id','person_count']).set_index('aermod_cell_id').person_count
    fields=['TotalPM25','BC','NO2','power_TotalPM25']
    burdens={k:0. for k in fields}
    for batch in main.iter_batches(columns=['aermod_cell_id',*fields]):
        d=batch.to_pandas();w=d.aermod_cell_id.map(population).fillna(0)
        for key in fields:burdens[key]+=float((d[key]*w).sum())
    save(out/'results_summary.json',dict(status='complete',grid_cells=main.metadata.num_rows,
        traffic_scope=json.loads((out/'scenario.json').read_text())['traffic_scope'],
        charging_daily_MWh=summary['daily_energy_MWh'],charging_annual_MWh=summary['annual_energy_MWh'],
        charging_peak_MW=summary['annual_peak_MW'],population=float(population.sum()),
        population_weighted_concentrations_ug_m3={k:v/float(population.sum()) for k,v in burdens.items()},
        power_source_layer=0,traffic_scale=10,charging_scale=10,
        main_concentration=str(out/'exposure/beam_concentration_distribution.parquet'),
        comparison_concentration=str(out/'exposure/beam_concentration_comparison_with_power.parquet'),
        power_only=str(out/'power_generation/power_generation_isrm_concentrations.parquet'),
        power_only_plot=str(out/'power_generation/power_generation_isrm_total_pm25.png'),
        same_power_in_both_comparison_branches=True,existing_results_unchanged=True))
    (out/'README.md').write_text('''# Native existing-EV baseline, 10x sample expansion

This independent scenario retains baseline passenger/freight road emissions
(including existing hybrid fuel emissions and EV non-exhaust emissions) and adds
only the power contribution from recorded EV charging. It does NOT electrify
fuel trucks or infer missing PHEV charging. Bus/rail/nonroad demand is excluded
in the default ldv-freight scope. See scenario.json for the selected scope.

## Main products

- exposure/beam_concentration_distribution.parquet: 100m GeoParquet. TotalPM25,
  PrimaryPM25 and SecondaryPM25 include power. BC and NO2 are traffic-only:
  the power ISRM models primary/secondary PM2.5, not power BC or direct NO2.
- exposure/beam_concentration_distribution.gpkg: equivalent GIS export.
- power_generation/power_generation_isrm_concentrations.parquet and .gpkg:
  power-only concentrations on the native ISRM receptor grid (not a new 100m model).
- power_generation/power_generation_isrm_total_pm25.png: green-to-red power-only
  annual-average PM2.5 map with emitting plant markers.
- postprocess/concentrations/totalpm25.png: enhanced total PM2.5. The original
  plotting code uses a percentile-based color limit, not the absolute maximum;
  raw concentration values are not clipped in the data products.
- charging_demand/native_ev_hourly_curve_10x.csv: expanded recorded charging curve.
- postprocess/human_exposure/web/BayArea_Interactive_Direct_Final_PM25_BC_NO2_Power.html:
  original-style interactive map; 100m source grid.
- postprocess/human_exposure/tables/: original-style CES exposure comparisons.

## Concentration column and comparison semantics

The main concentration file preserves native IMPACTS definitions: inmap_* and
inmap_only_* are TRAFFIC-only components; power_* is a separate increment.
Traffic local-to-local removal affects primary PM2.5, BC and NO2 only. AERMOD is
added directly with a scale factor of 1. Secondary PM2.5 and power are not removed.

exposure/beam_concentration_comparison_with_power.parquet is the explicit plotting
and exposure comparison view. ONLY inmap_only_PrimaryPM25 and inmap_only_TotalPM25
are increased by the same corresponding power increments included in the enhanced
branch. Thus both model-comparison branches include identical power emissions;
their difference isolates traffic-dispersion methodology. BC/NO2 remain traffic-only.

## Scale, time and limitations

- Traffic input concentrations ALREADY contain the user-confirmed 10x expansion.
  They are reused/summed without another multiplication. Native session energy
  receives one 10x expansion; dispatch and ISRM apply no further fleet expansion.
- Charging is reconstructed from RefuelSessionEvent energy (joules) and matched
  plug-in/disconnect records, uniformly over each connected interval. Internal
  charging power cycles are not available. Native simulation-local time is used.
- Only the 0-24h overlap is included, consistent with the traffic window. No
  next-day modulo folding. Four inconsistent/duplicate disconnect records are
  quarantined in charging_demand/quarantined_native_sessions.csv.
- The recorded day is repeated 365 times as an annualized supplied-load scenario.
  This is not a verified equilibrium charging-needs profile or measured annual load.
- Delivered charging energy approximates end-use electricity (charger efficiency
  assumed 1); Cambium distribution losses are applied separately.
- Existing grid scenario: Cambium22 LowRECost p9 2040 plus eGRID 2023 generation
  allocation and supplemental 2021 pollutant factors. Not a year-matched historical
  baseline. All included charging is assigned to p9, without station-based BA splitting.
- Power uses the previously authorized temporary ISRM Layer 0 approximation,
  only for emitting plants in the nine Bay Area counties. Outside-Bay generation
  and unallocated technologies are NOT redistributed to the retained plants.
- Previously approved missing traffic ISRM source rows remain omitted; AERMOD
  retains all sources. Population weights are not scaled.
- Read results_summary.json, power_generation/power_generation_summary.json,
  concentration_validation.json and traffic_plus_power_identity_check.json for QA.

## Reproduce

Run scripts/run_native_ev_baseline.py with a NEW --output directory, using stages
charging, traffic, dispatch, power_isrm, integrate, postprocess, verify in order.
Default --traffic-scope is ldv-freight. Existing completed stages refuse overwrite.
''',encoding='utf-8')


def main():
    for stream in [sys.stdout,sys.stderr]:
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='backslashreplace')
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--stage',required=True,choices=['charging','traffic','dispatch','power_isrm','integrate','postprocess','verify'])
    parser.add_argument('--traffic-scope',choices=['ldv-freight','all-road'],default='ldv-freight')
    args=parser.parse_args();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    scenario_path=out/'scenario.json'
    if scenario_path.exists():
        if json.loads(scenario_path.read_text())['traffic_scope']!=args.traffic_scope:
            raise ValueError('Scenario traffic scope cannot change in-place')
    else:save(scenario_path,dict(events=str(EVENTS),traffic_scope=args.traffic_scope,
        traffic_expansion=10.,charging_expansion=10.,native_existing_ev_only=True,
        truck_electrification=False,phev_charge_imputation=False,
        power_grid='Cambium22 LowRECost p9 2040 + eGRID 2023 annual-generation allocation',
        power_isrm=ISRM,power_layer=0,power_scope='SFBAY-located emitting plants; outside generation not redistributed',
        traffic_partial_isrm='Previously approved 103 unavailable source rows remain omitted',
        annualization='Recorded 0-24h charging window repeated 365 times as a scenario, not full charging needs',
        sources=dict(traffic=str(SPLIT if args.traffic_scope=='ldv-freight' else TOTAL),preprocess=str(REFERENCE))))
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    continuation.BASE=REFERENCE
    continuation.DEMAND=out/'charging_demand'
    status_path=out/'run_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get(args.stage)=='completed':raise ValueError('Stage already completed; refusing overwrite')
    status[args.stage]='running';save(status_path,status)
    try:
        if args.stage=='traffic':traffic(out,args.traffic_scope)
        elif args.stage=='dispatch':
            continuation.dispatch(out)
            path=out/'power_generation/power_generation_summary.json'
            data=json.loads(path.read_text());data.update(traffic_baseline=str(out/'traffic_pipeline_manifest.yaml'),
                charging_scale_factor=10.,additional_dispatch_scale_factor=1.)
            save(path,data)
        elif args.stage=='power_isrm':continuation.power_isrm(out,ISRM,source_layer_index=0)
        else:globals()[args.stage](out)
    except Exception as exc:
        status[args.stage]='failed';status['error']=f'{type(exc).__name__}: {exc}';save(status_path,status);raise
    status[args.stage]='completed';status.pop('error',None);save(status_path,status)


if __name__=='__main__':main()
