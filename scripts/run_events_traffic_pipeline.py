"""Explicit fc07 traffic-only, baseline-fleet-mixture scenario; independent outputs.

No charging loads or power emissions. This is a documented fleet-composition
proxy, not a claim that the raw events identify individual EMFAC model years.
"""
from __future__ import annotations

from research_paths import ISRM_PATH, BREATHE_ROOT, BEAM_DATA_ROOT, CHARGING_REFERENCE
import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from impacts.pipeline.workflow.prepare_emissions.event_emfac import EmfacRateStore, COUNTIES, POLLUTANTS
from impacts.config.defaults import grams_per_short_ton

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'examples/pilates/impacts/impacts_output/impacts-postsim--sfbay--base--20260901-165739'
INPUT = ROOT/'chargingcode/beam-fc07-v260419-it3'
VEH = ROOT/'examples/pilates/beam/production/sfbay/vehicle-tech'
RATES = ROOT/'examples/pilates/impacts/impacts_input/activities/2018-Baseline/rates'
MODES = "('car','car_hov2','car_hov3','bus','ride_hail','ride_hail_pooled')"
POLS = list(POLLUTANTS)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def sqlpath(path):
    return str(Path(path).resolve()).replace("'", "''")


def setup(out):
    if (out/'preprocess_manifest.yaml').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    for folder in ['inputs', 'emissions', 'concentrations', 'exposure', '_tmp']:
        (out/folder).mkdir(exist_ok=True)
    old = pd.read_csv(ROOT/'examples/pilates/beam/beam_output/network.csv.gz')
    new = pd.read_csv(INPUT/'network.csv.gz')
    geometry_fields = [c for c in old if c != 'numberOfLanes']
    pd.testing.assert_frame_equal(old[geometry_fields], new[geometry_fields])
    save(out/'network_validation.json', dict(geometry_and_lengths_identical=True,
         differing_fields=[c for c in old if not old[c].equals(new[c])],
         intersection_reuse='Same link IDs, coordinates, lengths, OSM IDs and road categories; lane field is not used in intersection construction.'))
    settings = yaml.safe_load((ROOT/'examples/pilates/settings.yaml').read_text())
    settings['impacts']['power_generation'] = {'enabled': False}
    settings['impacts']['local_output_folder'] = str(out)
    settings['impacts']['population']['population_sample'] = 1.
    settings_path = out/'settings.yaml'
    settings_path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
    m = yaml.safe_load((BASE/'preprocess_manifest.yaml').read_text())
    m['settings_source'] = str(settings_path)
    m['input_dir'] = str(out/'inputs')
    m['staging_dir'] = str(out/'inputs')
    m['preprocess_manifest_path'] = str(out/'preprocess_manifest.yaml')
    m['inputs'].pop('emissions_skims_input', None)
    for key, path in [('settings', settings_path), ('events_input', INPUT/'3.events.parquet'),
                      ('network', INPUT/'network.csv.gz')]:
        m['inputs'][key] = dict(kind='local', source_path=str(path), staged_path=str(path), exists=True, optional=False)
    p = m['pipeline']
    p.update(population_sample=1., freight_sample=1., transit_sample=1., power_generation={'enabled': False},
             enable_passenger_inventory_activity_correction=False, enable_freight_inventory_activity_correction=False,
             primary_pm25_integration_strategy='additive_inmap_aermod',
             isrm_url=ISRM_PATH)
    (out/'preprocess_manifest.yaml').write_text(yaml.safe_dump(m, sort_keys=False), encoding='utf-8')
    save(out/'scenario.json', dict(events=str(INPUT/'3.events.parquet'), power_generation=False,
         fleet_scale=1, day_window_seconds=[0,86400], emission_factor_year=2018,
         fleet_composition='Existing baseline proxy, conditioned on passenger powertrain/income; freight sourceVehicleTypeId priors',
         limitations=['Not a full-SFBAY fleet expansion; only vehicles represented in events',
                      'Passenger VMT composition approximated by baseline vehicle-count composition within powertrain/income',
                      'Freight VMT priors for driving, population priors for parking/start (proxy)',
                      'PHEV uses mapped baseline EMFAC factors, not a trip-specific fuel-mode model',
                      'No initial cold starts, unobserved overnight parking or hotelling inferred',
                      'One representative 0-24h day; partial link activity clipped at midnight',
                      'Events weight is curb plus payload per BEAM DrivesVehicle.scala; do not add payload again',
                      'Working idle hours = travel hours times mapped idleTimeFraction']))


def mapping(out):
    a = pd.read_csv(VEH/'vehicleTypes--atlas--2019-Baseline--EM.csv', low_memory=False)
    b = pd.read_csv(VEH/'vehicleTypes--frism--2018-Baseline--EM.csv', low_memory=False)
    vehicles = pd.read_parquet(ROOT/'examples/pilates/beam/production/sfbay/urbansim/atlas-2019/vehicles.parquet')
    original = pd.read_csv(str(BEAM_DATA_ROOT/'vehicle-tech/vehicletypes--baseline.csv')).set_index('vehicleTypeId')
    con = duckdb.connect()
    observed = con.execute(f"SELECT DISTINCT vehicleType FROM read_parquet(?) WHERE type='PathTraversal' AND mode IN {MODES}", [str(INPUT/'3.events.parquet')]).fetchdf().vehicleType.tolist()
    con.close()
    frames = []
    for source in sorted(observed):
        if source.startswith('ft-'):
            candidates = b.loc[b.sourceVehicleTypeId.eq(source)].copy()
            candidates['vmt_weight'] = candidates.fleetVmtPrior / candidates.fleetVmtPrior.sum()
            candidates['pop_weight'] = candidates.fleetPopulationPrior / candidates.fleetPopulationPrior.sum()
            basis = 'Existing sourceVehicleTypeId: VMT/population priors, normalized within source type'
        elif source.split('-')[0] in ['cv','hev','ev','phev']:
            if source not in original.index:
                raise ValueError(f'Original passenger metadata not found: {source}')
            prefix = source.split('-')[0]
            fuel = {'cv':'conv','hev':'hybrid','ev':'ev','phev':'phev'}[prefix]
            selected = vehicles.loc[vehicles.adopt_fuel.eq(fuel)]
            label = str(original.loc[source,'sampleProbabilityString'])
            match = re.search(r'income\s*\|\s*([\d.]+)-([\d.]+):', label)
            if match:
                lo, hi = map(float, match.groups())
                selected = selected.loc[selected.income_in_thousands.ge(lo) & selected.income_in_thousands.lt(hi)]
            counts = selected.vehicleTypeId.value_counts(normalize=True)
            candidates = a.loc[a.vehicleTypeId.isin(counts.index)].copy()
            candidates['pop_weight'] = candidates.vehicleTypeId.map(counts)
            candidates['vmt_weight'] = candidates.pop_weight
            basis = f'Baseline observed vehicle counts: {fuel}; income={match.group(0) if match else "all"}'
        else:
            candidates = a.loc[a.vehicleTypeId.eq(source)].copy()
            candidates['pop_weight'] = candidates['vmt_weight'] = 1.
            basis = 'Exact existing transit vehicle mapping'
        if candidates.empty or candidates.emfacId.isna().any():
            raise ValueError(f'No complete candidate mapping for {source}')
        # Collapse identical EMFAC categories/year groups before trip expansion.
        weights = candidates.groupby('emfacId')[['vmt_weight','pop_weight']].sum()
        candidates = candidates.drop_duplicates('emfacId').set_index('emfacId')
        candidates[['vmt_weight','pop_weight']] = weights
        candidates = candidates.reset_index()
        for col in ['vmt_weight','pop_weight']:
            candidates[col] /= candidates[col].sum()
            np.testing.assert_allclose(candidates[col].sum(),1)
        candidates['sourceVehicleTypeId'] = source
        candidates['vehicleTypeId'] = 'fc07--'+source+'--'+candidates.emfacId
        candidates['mapping_basis'] = basis
        candidates['assignment_group'] = 'freight' if source.startswith('ft-') else 'transit' if source.startswith('BUS') else 'passenger'
        frames.append(candidates)
    result = pd.concat(frames, ignore_index=True)
    catalog = pd.read_csv(VEH/'_emissions_vehicle_catalog.csv').set_index('emfac_vehicle_category')
    result['operation_days'] = result.emfacVehicleCategory.map(catalog.operation_days_per_year)
    result['source_release_height'] = result.emfacVehicleCategory.map(catalog.tailpipe_height_meters)
    if result[['operation_days','source_release_height']].isna().any().any():
        raise ValueError('Missing annualization days or release height')
    result.to_csv(out/'inputs/vehicle_emfac_mixture.csv', index=False)
    result.loc[result.assignment_group.ne('freight')].to_csv(out/'inputs/passenger_vehicle_types.csv', index=False)
    result.loc[result.assignment_group.eq('freight')].to_csv(out/'inputs/freight_vehicle_types.csv', index=False)
    m = yaml.safe_load((out/'preprocess_manifest.yaml').read_text())
    for label in ['passenger','freight']:
        path = str(out/f'inputs/{label}_vehicle_types.csv')
        m['inputs'][label+'_vehicle_types_input'] = dict(kind='local', source_path=path, staged_path=path, exists=True, optional=False)
        m['pipeline'][label+'_vehicle_types_file'] = path
    (out/'preprocess_manifest.yaml').write_text(yaml.safe_dump(m, sort_keys=False), encoding='utf-8')
    print('MAPPING',len(observed),'source types ->',len(result),'weighted combinations',flush=True)


def activity(out):
    db = out/'_tmp/traffic.duckdb'
    con = duckdb.connect(str(db)); con.execute("SET threads=4"); con.execute("SET memory_limit='32GB'")
    con.execute('SET preserve_insertion_order=false')
    network = pd.read_csv(INPUT/'network.csv.gz')
    con.register('network_input', network)
    con.execute('CREATE OR REPLACE TABLE network AS SELECT linkId,linkLength,linkFreeSpeed,attributeOrigType AS road FROM network_input')
    zones = pd.read_parquet(BASE/'preprocess/beam_osm_county_intersection.parquet')
    zones = zones.groupby(['linkId','county_COUNTYFP'],as_index=False).county_proportion.sum()
    zones['county'] = zones.county_COUNTYFP.astype(str).str.zfill(3).map(COUNTIES).str.lower()
    if zones.county.isna().any(): raise ValueError('Unknown county')
    zones['county_weight'] = zones.county_proportion/zones.groupby('linkId').county_proportion.transform('sum')
    con.register('zone_input',zones)
    con.execute('CREATE OR REPLACE TABLE zones AS SELECT linkId,county,county_weight FROM zone_input')
    con.execute(f'''CREATE OR REPLACE TABLE trips AS SELECT row_number() OVER () AS trip_id,
        vehicle, vehicleType AS source_type, departureTime AS dep, arrivalTime AS arr,
        try_cast(weight AS DOUBLE) AS weight_kg,
        list_transform(string_split(links,','),x->try_cast(x AS BIGINT)) AS link_list,
        list_transform(string_split(linkTravelTime,','),x->try_cast(x AS DOUBLE)) AS time_list
        FROM read_parquet('{sqlpath(INPUT/'3.events.parquet')}')
        WHERE type='PathTraversal' AND mode IN {MODES}''')
    missing_route = 'len(link_list)=0 OR list_contains(list_transform(link_list,x->x IS NULL),true) OR list_contains(list_transform(time_list,x->x IS NULL),true)'
    missing_route_count = con.execute(f'SELECT count(*) FROM trips WHERE {missing_route}').fetchone()[0]
    if missing_route_count:
        con.execute(f"COPY (SELECT * FROM trips WHERE {missing_route}) TO '{sqlpath(out/'inputs/excluded_missing_route_trips.parquet')}' (FORMAT PARQUET)")
        con.execute(f'DELETE FROM trips WHERE {missing_route}')
        print('Excluded missing-route records (saved for audit):',missing_route_count,flush=True)
    invalid = con.execute('''SELECT count(*) FROM trips WHERE dep<0 OR arr<dep OR weight_kg IS NULL OR weight_kg<=0
        OR len(link_list)!=len(time_list) OR list_contains(list_transform(time_list,x->x<0),true)''').fetchone()[0]
    if invalid: raise ValueError(f'Invalid trips: {invalid}')
    # Prefix time from list cumulative sums: no Python per-link expansion.
    print('Expanding link activity',flush=True)
    con.execute('''CREATE OR REPLACE TABLE traversal AS
       WITH expanded AS (SELECT trip_id,source_type,dep,weight_kg,
           unnest(link_list) AS linkId,unnest(time_list) AS seconds,
           generate_subscripts(link_list,1) AS ord FROM trips WHERE dep<86400),
       timed AS (SELECT *,dep+coalesce(sum(seconds) OVER(PARTITION BY trip_id ORDER BY ord ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS start FROM expanded)
       SELECT t.*,n.linkLength,n.linkFreeSpeed,n.road,
         greatest(0,least(seconds,86400-start)) AS used_seconds,
         CASE WHEN seconds>0 THEN greatest(0,least(1,(86400-start)/seconds)) ELSE 0 END AS day_fraction
       FROM timed t LEFT JOIN network n USING(linkId)''')
    if con.execute('SELECT count(*) FROM traversal WHERE linkLength IS NULL').fetchone()[0]: raise ValueError('Network link missing')
    diagnostics = con.execute('''SELECT count(*) links, count(*) FILTER(WHERE seconds=0) zero_duration_links,
       sum(linkLength) FILTER(WHERE seconds=0) zero_duration_length_m,
       sum(linkLength*day_fraction)/1609.344 day_miles FROM traversal''').fetchdf().to_dict('records')[0]
    diagnostics['excluded_missing_route_records'] = missing_route_count
    # Zero-duration links cannot supply a speed. Their distance is not invented;
    # retain their count/length in QA rather than silently using a speed factor.
    con.execute('''CREATE OR REPLACE TABLE running AS SELECT source_type,linkId,
        greatest(5,least(90,ceil((linkLength/1609.344/(seconds/3600))/5-0.5)*5)) AS speed_bin,
        sum(linkLength/1609.344*day_fraction) AS miles,
        sum(linkLength/1609.344*day_fraction*pow(weight_kg/907.18474,1.02)) AS dust_miles,
        sum(used_seconds)/3600 AS hours,sum(day_fraction) AS observations
        FROM traversal WHERE used_seconds>0 GROUP BY ALL''')
    # Chronological next-trip join; no wraparound and no artificial 24h parking.
    con.execute('''CREATE OR REPLACE TABLE parking_raw AS WITH ordered AS (
       SELECT *,lead(dep) OVER(PARTITION BY vehicle ORDER BY dep,arr,trip_id) AS next_dep,
          lead(source_type) OVER(PARTITION BY vehicle ORDER BY dep,arr,trip_id) AS next_type,
          lead(list_extract(link_list,1)) OVER(PARTITION BY vehicle ORDER BY dep,arr,trip_id) AS next_link
       FROM trips)
       SELECT *,next_dep-arr AS gap FROM ordered WHERE arr<86400 AND next_dep IS NOT NULL''')
    diagnostics['overlapping_trip_pairs'] = con.execute('SELECT count(*) FROM parking_raw WHERE gap<0').fetchone()[0]
    # Overlaps are documented and contribute no inferred parking, never a negative dwell.
    soak = [5,10,20,30,40,50,60,120,180,240,300,360,420,480,540,600,660,720]
    cases = ' '.join(f'WHEN gap/60 <= {(l+r)/2} THEN {l}' for l,r in zip(soak,soak[1:]))
    con.execute(f'''CREATE OR REPLACE TABLE parking AS
       SELECT source_type,list_extract(link_list,-1) AS linkId,'DIURN' AS process,
         CASE {cases} ELSE 720 END AS soak_bin,
         sum(greatest(0,least(next_dep,86400)-arr))/3600 AS hours,count(*)::DOUBLE AS observations
       FROM parking_raw WHERE gap>0 GROUP BY ALL
       UNION ALL
       SELECT source_type,list_extract(link_list,-1),'HOTSOAK',0,0,count(*)::DOUBLE
       FROM parking_raw WHERE gap>0 GROUP BY ALL
       UNION ALL
       SELECT next_type,next_link,'STREX',CASE {cases} ELSE 720 END,0,count(*)::DOUBLE
       FROM parking_raw WHERE gap>0 AND next_dep<86400 GROUP BY ALL''')
    diagnostics.update(con.execute('SELECT count(*) road_trips,count(*) FILTER(WHERE dep<86400) day_departures,max(arr) maximum_arrival_seconds FROM trips').fetchdf().to_dict('records')[0])
    diagnostics['running_groups'] = con.execute('SELECT count(*) FROM running').fetchone()[0]
    save(out/'activity_audit.json',diagnostics)
    con.close();print('ACTIVITY',diagnostics,flush=True)


def emissions(out, *, prepare_only=False):
    target = out/'emissions/prepared_skims_for_grid_allocation.parquet'
    if target.exists():
        raise FileExistsError(target)
    m = pd.read_csv(out/'inputs/vehicle_emfac_mixture.csv',low_memory=False)
    store = EmfacRateStore(RATES,m.set_index('vehicleTypeId',drop=False),POLS)
    frames = []
    for emfac, group in m.groupby('emfacId'):
        try:
            r = store.rates(group.vehicleTypeId.iloc[0]).copy()
        except FileNotFoundError:
            # Recover an exact missing partition from the project's original
            # grouped factors, without modifying the shared rate store.
            meta = group.iloc[0]
            original_rates = pd.read_parquet(RATES.parent/'inventory/sf-emfac-2018-project-analysis-final-freight-rates.parquet')
            recovered = original_rates.loc[
                original_rates.vehicleCategory.eq(meta.emfacVehicleCategory)
                & original_rates.fuel.eq(meta.emfacFuel)
                & original_rates.modelYear.eq(meta.emfacResolvedModelYear)]
            if recovered.empty:
                raise
            partition = out/'inputs/recovered_rates'/f'emfacId={emfac}'
            partition.mkdir(parents=True,exist_ok=True)
            recovered.to_parquet(partition/f'{emfac}.parquet',index=False)
            exact_store = EmfacRateStore(out/'inputs/recovered_rates',m.set_index('vehicleTypeId',drop=False),POLS)
            r = exact_store.rates(meta.vehicleTypeId).copy()
            print('Recovered exact native factor partition:',emfac,len(r),'rows',flush=True)
        r['emfacId'] = emfac
        r['activity_bin'] = pd.to_numeric(r.speedMph_timeMin,errors='raise').fillna(0.)
        frames.append(r)
    rates = pd.concat(frames,ignore_index=True)
    # Some diesel STREX partitions start at 20 min. BEAM selects their
    # nearest available bin for shorter dwell, rather than omitting starts.
    extra = []
    soak_centers = [5,10,20,30,40,50,60,120,180,240,300,360,420,480,540,600,660,720]
    for _, group in rates.loc[rates.process.eq('STREX')].groupby(['emfacId','county','roadCategory'],dropna=False):
        centers = group.activity_bin.to_numpy()
        if 0 in centers:
            continue
        for center in soak_centers:
            if center in centers:
                continue
            if centers.min()<center<centers.max():
                raise ValueError('Interior STREX bin gap requires exact dwell lookup')
            row = group.iloc[np.abs(centers-center).argmin()].copy()
            row['activity_bin'] = center
            extra.append(row)
    if extra:
        rates = pd.concat([rates,pd.DataFrame(extra)],ignore_index=True)
        print('Nearest endpoint STREX factors:',len(extra),flush=True)
    speed = rates.loc[rates.process.isin(['RUNEX','PMBW','PMTW','PTOEX','PRDUST']) & rates.activity_bin.ne(0),'activity_bin']
    if not speed.isin(range(5,91,5)).all():raise ValueError('Unexpected speed-bin centers')
    con = duckdb.connect(str(out/'_tmp/traffic.duckdb'));con.execute('SET threads=4');con.execute("SET memory_limit='32GB'")
    con.register('mix_input',m);con.register('rate_input',rates)
    con.execute('CREATE OR REPLACE TABLE mixture AS SELECT * FROM mix_input')
    con.execute('CREATE OR REPLACE TABLE rates AS SELECT * FROM rate_input')
    # Each component retains its own annual operating days and release height.
    rate_sql = ','.join(f'coalesce(r.{POLLUTANTS[p]},0) AS "{p}"' for p in POLS)
    con.execute(f'''CREATE OR REPLACE TABLE component_rates AS SELECT
       m.sourceVehicleTypeId AS source_type,m.vehicleTypeId,m.operation_days,m.source_release_height,
       m.vmt_weight,m.pop_weight,coalesce(m.idleTimeFraction,0) AS idle_fraction,
       r.county,r.process,r.roadCategory,r.activity_bin,{rate_sql}
       FROM mixture m JOIN rates r USING(emfacId)''')
    # Coverage must exist for every in-domain source/process/county/road combination.
    con.execute('''CREATE OR REPLACE TABLE running_detail AS SELECT a.*,n.road,z.county,z.county_weight
       FROM running a JOIN network n USING(linkId) JOIN zones z USING(linkId)''')
    con.execute('''CREATE OR REPLACE TABLE parking_detail AS SELECT a.*,n.road,z.county,z.county_weight
       FROM parking a JOIN network n USING(linkId) JOIN zones z USING(linkId)''')
    for label, table, bin_col, process_filter in [
        ('running','running_detail','speed_bin',"r.process IN ('RUNEX','RUNLOSS','PMBW','PMTW','PRDUST','PTOEX','IDLEX')"),
        ('parking','parking_detail','soak_bin','a.process=r.process'),
    ]:
        keys = f'source_type,county,road,{bin_col}' + (',process' if label == 'parking' else '')
        missing = con.execute(f'''WITH a AS (SELECT DISTINCT {keys} FROM {table}),
          r AS (SELECT DISTINCT source_type,vehicleTypeId,process FROM component_rates)
          SELECT a.*,r.vehicleTypeId,r.process AS expected_process FROM a JOIN r USING(source_type)
          WHERE {process_filter} AND NOT EXISTS (
            SELECT 1 FROM component_rates c WHERE c.vehicleTypeId=r.vehicleTypeId
            AND c.process=r.process AND c.county=a.county
            AND (c.roadCategory IS NULL OR c.roadCategory='' OR c.roadCategory=lower(a.road))
            AND (c.activity_bin=0 OR c.activity_bin=a.{bin_col})) LIMIT 20''').fetchdf()
        if not missing.empty:
            missing.to_csv(out/f'inputs/missing_{label}_rate_keys.csv',index=False)
            raise ValueError(f'Missing {label} rate keys: {missing.to_dict("records")}')
    if prepare_only:
        con.close()
        print('Native EMFAC component-rate preparation and activity coverage checks complete',flush=True)
        return
    print('Joining activity to weighted EMFAC components',flush=True)
    # All applicable processes from each explicit EMFAC partition are included.
    con.execute('''CREATE OR REPLACE TABLE gram_activity AS
       SELECT a.linkId,r.vehicleTypeId,r.process,r.operation_days,r.source_release_height,
         a.observations*r.vmt_weight*a.county_weight AS obs,
         a.miles*r.vmt_weight*a.county_weight AS miles,
         CASE WHEN r.process='PRDUST' THEN a.dust_miles
              WHEN r.process='IDLEX' THEN a.hours*r.idle_fraction
              WHEN r.process='RUNLOSS' THEN a.hours ELSE a.miles END
              *r.vmt_weight*a.county_weight AS amount,
         r.NH3,r.NOx,r.PM25,r.SOx,r.ROG,r.BC
       FROM running_detail a JOIN component_rates r ON a.source_type=r.source_type AND a.county=r.county
         AND (r.roadCategory IS NULL OR r.roadCategory='' OR lower(a.road)=r.roadCategory)
         AND (r.activity_bin=0 OR a.speed_bin=r.activity_bin)
       WHERE r.process IN ('RUNEX','RUNLOSS','PMBW','PMTW','PRDUST','PTOEX','IDLEX')
       UNION ALL
       SELECT a.linkId,r.vehicleTypeId,r.process,r.operation_days,r.source_release_height,
         a.observations*r.pop_weight*a.county_weight,0.,
         CASE WHEN r.process='DIURN' THEN a.hours ELSE a.observations END*r.pop_weight*a.county_weight,
         r.NH3,r.NOx,r.PM25,r.SOx,r.ROG,r.BC
       FROM parking_detail a JOIN component_rates r ON a.source_type=r.source_type AND a.county=r.county AND a.process=r.process
         AND (r.roadCategory IS NULL OR r.roadCategory='' OR lower(a.road)=r.roadCategory)
         AND (r.activity_bin=0 OR a.soak_bin=r.activity_bin)''')
    totals = ','.join(f'sum(amount*"{p}"*operation_days/{grams_per_short_ton}) AS "tons_per_year_{p}"' for p in POLS)
    partial_target = target.with_suffix('.partial.parquet')
    con.execute(f'''COPY (SELECT linkId,vehicleTypeId,process,source_release_height,
       n.road AS roadCategory,sum(obs*operation_days) AS totTrips,sum(miles*operation_days) AS totVMT,{totals}
       FROM gram_activity JOIN network n USING(linkId) GROUP BY ALL)
       TO '{sqlpath(partial_target)}' (FORMAT PARQUET, COMPRESSION ZSTD)''')
    report = con.execute(f'SELECT count(*) AS row_count,{",".join("sum(tons_per_year_"+p+") AS "+p for p in POLS)} FROM read_parquet(?)',[str(partial_target)]).fetchdf().to_dict('records')[0]
    if any(not np.isfinite(report[p]) or report[p]<0 for p in POLS):
        raise ValueError('Invalid annual emission totals')
    partial_target.replace(target)
    report.update(fleet_expansion=1.,power_generation=False,units='short tons/year',source='new events, baseline fleet mixture')
    save(out/'emissions_audit.json',report);con.close();print('EMISSIONS',report,flush=True)


def run_stage(out, stage):
    from impacts.runner import _run_stages_from_preprocess_manifest
    flags={f'run_{s}':s==stage for s in ['emissions','inmap','aermod','exposure']}
    _run_stages_from_preprocess_manifest(out/'preprocess_manifest.yaml',run_manifest_path=out/'pipeline_manifest.yaml',
                                        output_root_override=out,**flags)


def coverage(out):
    import zarr
    sr = zarr.open_group(ISRM_PATH,mode='r')
    con = duckdb.connect();con.execute('SET threads=4')
    sums = ','.join(f'sum(tons_per_year_{p}) AS {p}' for p in POLS)
    allocated = ','.join(f'sum(e.{p}*i.inmap_proportion) AS {p}' for p in POLS)
    frame = con.execute(f'''WITH e AS (SELECT linkId,{sums} FROM read_parquet(?) GROUP BY linkId)
        SELECT i.inmap_cell_id,{allocated} FROM e JOIN read_parquet(?) i USING(linkId)
        GROUP BY i.inmap_cell_id''',[str(out/'emissions/prepared_skims_for_grid_allocation.parquet'),
        str(BASE/'preprocess/beam_osm_inmap_intersection.parquet')]).fetchdf()
    con.close()
    frame = frame.loc[frame[POLS].sum(axis=1)>0].copy()
    frame['matrix_source_available'] = frame.inmap_cell_id.isin(sr['source_ids'][:])
    frame.to_csv(out/'inputs/isrm_source_coverage.csv',index=False)
    missing = frame.loc[~frame.matrix_source_available]
    missing.to_csv(out/'inputs/missing_isrm_sources.csv',index=False)
    report = dict(required_source_cells=len(frame),available_source_cells=int(frame.matrix_source_available.sum()),
                  missing_source_cells=len(missing), missing_emissions_short_tons_year=missing[POLS].sum().to_dict(),
                  missing_emissions_fraction=(missing[POLS].sum()/frame[POLS].sum()).to_dict())
    save(out/'isrm_coverage_audit.json',report)
    print('ISRM COVERAGE',report,flush=True)
    if len(missing):
        save(out/'run_status.json',dict(status='blocked_missing_isrm_sources',final_concentration_generated=False,
             completed_stages=['setup','mapping','activity','emissions'],power_generation=False,
             missing_source_count=len(missing),missing_source_list=str(out/'inputs/missing_isrm_sources.csv'),
             required_action='Provide ground-layer ISRM transfer rows for missing global IDs, or authorize an explicitly partial-domain scenario.'))
        raise ValueError(f'ISRM lacks {len(missing)} emitting sources; see inputs/missing_isrm_sources.csv')


def audit_emissions(out):
    con = duckdb.connect()
    report = con.execute(f'SELECT count(*) AS row_count,{",".join("sum(tons_per_year_"+p+") AS "+p for p in POLS)} FROM read_parquet(?)',
        [str(out/'emissions/prepared_skims_for_grid_allocation.parquet')]).fetchdf().to_dict('records')[0]
    con.close()
    if any(not np.isfinite(report[p]) or report[p]<0 for p in POLS):
        raise ValueError('Invalid annual emission totals')
    report.update(fleet_expansion=1.,power_generation=False,units='short tons/year',source='new events, baseline fleet mixture')
    save(out/'emissions_audit.json',report)
    print('EMISSIONS',report,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['setup','mapping','activity','emissions','audit_emissions','coverage','allocate','inmap','aermod','exposure'],required=True)
    args=p.parse_args();out=args.output.resolve()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.stage in ['setup','mapping','activity','emissions','audit_emissions','coverage']:
        globals()[args.stage](out)
    else:run_stage(out,'emissions' if args.stage=='allocate' else args.stage)


if __name__=='__main__':main()
