"""Vehicle-ID-resolved baseline freight attribution; no fleet-ratio splitting.

Read the validated original event activity database without modifying it, retain
vehicle IDs through port classification, reaggregate activity and reuse identical
EMFAC factors. Run the existing dispersion/exposure stages for all four outputs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import logging
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

import run_events_traffic_pipeline as traffic
import run_ldv_freight_products as legacy
from impacts.config.defaults import grams_per_short_ton

ROOT = traffic.ROOT
SOURCE = legacy.ORIGINAL
REFERENCE = legacy.OUT / 'Freight'
CLASSIFICATION = ROOT / 'scenarios/all_scenarios_vehicle_classification.csv'
EVENTS = traffic.INPUT / '3.events.parquet'
OUT = SOURCE.parent / 'events-fc07-freight-port-classification10x-20261001'
GROUPS = {
    'Seaport': 'seaport_related',
    'Airport': 'airport_related',
    'NonPort': 'non_port_freight',
    'PortRelated': None,
}
POLS = traffic.POLS
MEASURES = ['totTrips', 'totVMT'] + [f'tons_per_year_{p}' for p in POLS]
CONC_FIELDS = ['TotalPM25', 'PrimaryPM25', 'SecondaryPM25', 'BC', 'NO2',
               'inmap_PrimaryPM25', 'inmap_SecondaryPM25', 'inmap_BC', 'inmap_NO2',
               'aermod_PrimaryPM25', 'aermod_BC', 'aermod_NO2',
               'inmap_only_PrimaryPM25', 'inmap_only_TotalPM25',
               'inmap_only_BC', 'inmap_only_NO2']


def validate_labels(frame):
    d = frame.loc[frame.scenario.eq('2018-Baseline')].copy()
    if d.empty or d.vehicleId.duplicated().any() or d.beam_vehicle_id.duplicated().any():
        raise ValueError('Baseline IDs missing or ambiguous')
    if not d.beam_vehicle_id.eq('body-' + d.vehicleId).all():
        raise ValueError('Unexpected BEAM body-to-truck relationship')
    for col in ['is_port_related', 'is_non_port_freight', 'has_seaport_stop', 'has_airport_stop']:
        if d[col].isna().any() or not d[col].isin([True, False]).all():
            raise ValueError(f'Invalid boolean: {col}')
    if (d.has_seaport_stop & d.has_airport_stop).any():
        raise ValueError('Overlapping airport/seaport membership needs explicit allocation policy')
    if not d.is_port_related.eq(d.has_seaport_stop | d.has_airport_stop).all():
        raise ValueError('Inconsistent port flags')
    if not d.is_non_port_freight.eq(~d.is_port_related).all():
        raise ValueError('Inconsistent non-port flags')
    expected = np.where(d.has_seaport_stop, 'seaport_related',
                        np.where(d.has_airport_stop, 'airport_related', 'non_port_freight'))
    if not d.port_category.eq(expected).all():
        raise ValueError('Category disagrees with indicators')
    return d


def rewritten(value, target):
    if isinstance(value, str):
        return value.replace(str(REFERENCE), str(target)).replace(REFERENCE.as_posix(), target.as_posix())
    if isinstance(value, list):
        return [rewritten(v, target) for v in value]
    if isinstance(value, dict):
        return {k: rewritten(v, target) for k, v in value.items()}
    return value


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def setup(*, group_column='port_category', labels_transform=None):
    if OUT.exists():
        raise FileExistsError(OUT)
    labels = validate_labels(pd.read_csv(CLASSIFICATION))
    if labels_transform is not None:
        labels = labels_transform(labels.copy())
    if group_column not in labels or not group_column.replace('_', '').isalnum():
        raise ValueError('Invalid grouping column')
    for name in ['audit', 'products', '_tmp', 'logs']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    labels.to_csv(OUT / 'audit/baseline_vehicle_classification.csv', index=False)
    con = duckdb.connect(str(OUT / '_tmp/classification.duckdb'))
    con.execute('SET threads=4')
    con.execute("SET memory_limit='24GB'")
    con.execute(f"ATTACH '{traffic.sqlpath(SOURCE / '_tmp/traffic.duckdb')}' AS original (READ_ONLY)")
    con.register('label_input', labels)
    con.execute('CREATE TABLE labels AS SELECT * FROM label_input')
    # Independently re-read the raw events, including the body IDs requested by
    # the collaborator. Body PathTraversal records are walks, not truck trips.
    con.execute('CREATE TABLE event_trucks AS SELECT DISTINCT vehicle,vehicleType FROM read_parquet(?) '
                "WHERE type='PathTraversal' AND starts_with(vehicleType,'ft-')", [str(EVENTS)])
    counts = con.execute('SELECT count(*) FROM event_trucks').fetchone()[0]
    bad = con.execute('SELECT count(*) FROM labels l FULL JOIN event_trucks e ON l.vehicleId=e.vehicle '
                      'WHERE l.vehicleId IS NULL OR e.vehicle IS NULL OR l.vehicleTypeId<>e.vehicleType').fetchone()[0]
    if bad or counts != len(labels):
        raise ValueError(f'Raw event truck classification mismatch: {bad}')
    bodies = con.execute('WITH e AS (SELECT DISTINCT vehicle FROM read_parquet(?)) '
                         'SELECT count(*) FROM labels JOIN e ON beam_vehicle_id=vehicle', [str(EVENTS)]).fetchone()[0]
    if bodies != len(labels):
        raise ValueError('Not all requested beam_vehicle_id values exist in events')
    # Validate the cached freight trajectories against the current event file,
    # including multiplicity, link sequences, timings and vehicle weights.
    missing_route = ('len(link_list)=0 OR list_contains(list_transform(link_list,x->x IS NULL),true) '
                     'OR list_contains(list_transform(time_list,x->x IS NULL),true)')
    con.execute('''CREATE TABLE raw_trips AS SELECT vehicle,vehicleType AS source_type,
        departureTime AS dep,arrivalTime AS arr,try_cast(weight AS DOUBLE) AS weight_kg,
        list_transform(string_split(links,','),x->try_cast(x AS BIGINT)) AS link_list,
        list_transform(string_split(linkTravelTime,','),x->try_cast(x AS DOUBLE)) AS time_list
        FROM read_parquet(?) WHERE type='PathTraversal' AND starts_with(vehicleType,'ft-')''', [str(EVENTS)])
    excluded = con.execute(f'SELECT count(*) FROM raw_trips WHERE {missing_route}').fetchone()[0]
    con.execute(f'DELETE FROM raw_trips WHERE {missing_route}')
    cols = 'vehicle,source_type,dep,arr,weight_kg,link_list,time_list'
    for a, b in [('raw_trips', 'original.trips WHERE starts_with(source_type,\'ft-\')'),
                 ('original.trips WHERE starts_with(source_type,\'ft-\')', 'raw_trips')]:
        diff = con.execute(f'SELECT count(*) FROM (SELECT {cols} FROM {a} EXCEPT ALL SELECT {cols} FROM {b})').fetchone()[0]
        if diff:
            raise ValueError(f'Cached event trajectories differ: {diff}')
    con.execute(f'CREATE TABLE classified_trips AS SELECT t.trip_id,l."{group_column}" AS port_category '
                'FROM original.trips t JOIN labels l ON t.vehicle=l.vehicleId')
    traffic.save(OUT / 'audit/input_validation.json', dict(
        events=str(EVENTS), events_sha256=sha256(EVENTS), classification=str(CLASSIFICATION),
        classification_sha256=sha256(CLASSIFICATION), sample_vehicles=counts,
        body_ids_matched=bodies, truck_ids_and_types_matched=True,
        cached_freight_trajectories_equal_current_events=True, excluded_missing_route_trips=excluded,
        fleet_expansion=10, power_generation=False,
        classification_scope='All modeled trips of classified vehicles, not only trips inside ports',
        grouping_column=group_column,
        gateway_policy='Preserved multi-label IDs; no duplication or arbitrary allocation'))
    print('RAW EVENTS / CACHED TRAJECTORIES / ID CLASSIFICATION VERIFIED', flush=True)
    con.execute('''CREATE TABLE running AS SELECT l.port_category,t.source_type,t.linkId,
        greatest(5,least(90,ceil((linkLength/1609.344/(seconds/3600))/5-0.5)*5)) AS speed_bin,
        sum(linkLength/1609.344*day_fraction) AS miles,
        sum(linkLength/1609.344*day_fraction*pow(weight_kg/907.18474,1.02)) AS dust_miles,
        sum(used_seconds)/3600 AS hours,sum(day_fraction) AS observations
        FROM original.traversal t JOIN classified_trips l USING(trip_id)
        WHERE used_seconds>0 GROUP BY ALL''')
    soak = [5,10,20,30,40,50,60,120,180,240,300,360,420,480,540,600,660,720]
    cases = ' '.join(f'WHEN gap/60 <= {(a+b)/2} THEN {a}' for a,b in zip(soak,soak[1:]))
    con.execute(f'CREATE VIEW classified_parking AS SELECT p.*,l."{group_column}" AS port_category '
                'FROM original.parking_raw p JOIN labels l ON p.vehicle=l.vehicleId')
    con.execute(f'''CREATE TABLE parking AS
        SELECT port_category,source_type,list_extract(link_list,-1) AS linkId,'DIURN' AS process,
        CASE {cases} ELSE 720 END AS soak_bin,
        sum(greatest(0,least(next_dep,86400)-arr))/3600 AS hours,count(*)::DOUBLE AS observations
        FROM classified_parking WHERE gap>0 GROUP BY ALL
        UNION ALL SELECT port_category,source_type,list_extract(link_list,-1),'HOTSOAK',0,0,count(*)::DOUBLE
        FROM classified_parking WHERE gap>0 GROUP BY ALL
        UNION ALL SELECT port_category,next_type,next_link,'STREX',CASE {cases} ELSE 720 END,0,count(*)::DOUBLE
        FROM classified_parking WHERE gap>0 AND next_dep<86400 GROUP BY ALL''')
    for table in ['running', 'parking']:
        con.execute(f'''CREATE VIEW {table}_detail AS SELECT a.*,n.road,z.county,z.county_weight
            FROM {table} a JOIN original.network n USING(linkId) JOIN original.zones z USING(linkId)''')
    # Identical rate selection, operating days, process activity and cohort
    # weights to the validated traffic pipeline. Only add a group key.
    con.execute('''CREATE TABLE gram_activity AS
        SELECT a.port_category,a.linkId,r.vehicleTypeId,r.process,r.operation_days,r.source_release_height,
        a.observations*r.vmt_weight*a.county_weight AS obs,
        a.miles*r.vmt_weight*a.county_weight AS miles,
        CASE WHEN r.process='PRDUST' THEN a.dust_miles
             WHEN r.process='IDLEX' THEN a.hours*r.idle_fraction
             WHEN r.process='RUNLOSS' THEN a.hours ELSE a.miles END*r.vmt_weight*a.county_weight AS amount,
        r.NH3,r.NOx,r.PM25,r.SOx,r.ROG,r.BC
        FROM running_detail a JOIN original.component_rates r
        ON a.source_type=r.source_type AND a.county=r.county
        AND (r.roadCategory IS NULL OR r.roadCategory='' OR lower(a.road)=r.roadCategory)
        AND (r.activity_bin=0 OR a.speed_bin=r.activity_bin)
        WHERE r.process IN ('RUNEX','RUNLOSS','PMBW','PMTW','PRDUST','PTOEX','IDLEX')
        UNION ALL
        SELECT a.port_category,a.linkId,r.vehicleTypeId,r.process,r.operation_days,r.source_release_height,
        a.observations*r.pop_weight*a.county_weight,0.,
        CASE WHEN r.process='DIURN' THEN a.hours ELSE a.observations END*r.pop_weight*a.county_weight,
        r.NH3,r.NOx,r.PM25,r.SOx,r.ROG,r.BC
        FROM parking_detail a JOIN original.component_rates r
        ON a.source_type=r.source_type AND a.county=r.county AND a.process=r.process
        AND (r.roadCategory IS NULL OR r.roadCategory='' OR lower(a.road)=r.roadCategory)
        AND (r.activity_bin=0 OR a.soak_bin=r.activity_bin)''')
    sums = ','.join(f'sum(amount*"{p}"*operation_days/{grams_per_short_ton})*10 AS tons_per_year_{p}' for p in POLS)
    aggregate = OUT / 'audit/classified_link_process_emissions.parquet'
    con.execute(f'''COPY (SELECT port_category,linkId,process,source_release_height,n.road AS roadCategory,
        sum(obs*operation_days)*10 AS totTrips,sum(miles*operation_days)*10 AS totVMT,{sums}
        FROM gram_activity JOIN original.network n USING(linkId) GROUP BY ALL)
        TO '{traffic.sqlpath(aggregate)}' (FORMAT PARQUET,COMPRESSION ZSTD)''')
    con.close()
    audit_emissions()
    totals = []
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        for label, category in GROUPS.items():
            target = OUT / label
            for folder in ['inputs', 'emissions', 'concentrations', 'exposure', '_tmp']:
                (target / folder).mkdir(parents=True, exist_ok=True)
            subset = labels.loc[labels[group_column].eq(category)] if category else labels.loc[labels.is_port_related]
            subset.to_csv(target / 'inputs/vehicle_classification.csv', index=False)
            where = f"port_category='{category}'" if category else "port_category IN ('seaport_related','airport_related')"
            sums = ','.join(f'sum("{c}") AS "{c}"' for c in MEASURES)
            prepared = target / 'emissions/prepared_skims_for_grid_allocation.parquet'
            con.execute(f'''COPY (SELECT linkId,'{label}-height-' || CAST(source_release_height AS VARCHAR) AS vehicleTypeId,
                process,source_release_height,roadCategory,{sums} FROM read_parquet('{traffic.sqlpath(aggregate)}')
                WHERE {where} GROUP BY ALL) TO '{traffic.sqlpath(prepared)}' (FORMAT PARQUET,COMPRESSION ZSTD)''')
            for filename in ['settings.yaml', 'preprocess_manifest.yaml']:
                data = rewritten(yaml.safe_load((REFERENCE / filename).read_text()), target)
                if filename == 'preprocess_manifest.yaml':
                    assert not data['pipeline']['power_generation']['enabled']
                    assert data['pipeline']['primary_pm25_integration_strategy'] == 'additive_inmap_aermod'
                    data.setdefault('notes', []).append(f'{label}: vehicle-ID-based Baseline freight subset; 10x applied once in emissions; no power generation.')
                (target / filename).write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
            scenario = json.loads((REFERENCE / 'scenario.json').read_text())
            scenario.update(product=label, vehicle_group=label, sample_vehicles=len(subset),
                represented_vehicles=len(subset)*10, fleet_scale=10, assignment_group='freight',
                method='Vehicle-ID-classified event activity with original EMFAC factors; independent dispersion',
                source_emissions_file=str(aggregate), classification_source=str(CLASSIFICATION),
                classification_scenario='2018-Baseline', power_generation=False)
            traffic.save(target / 'scenario.json', scenario)
            row = con.execute(f'SELECT {sums} FROM read_parquet(?)', [str(prepared)]).fetchdf().iloc[0].to_dict()
            row.update(vehicle_group=label, sample_vehicles=len(subset), represented_vehicles=len(subset)*10)
            totals.append(row)
            traffic.save(target / 'emissions_audit.json', dict(**{p:row[f'tons_per_year_{p}'] for p in POLS},
                fleet_expansion=10, units='short tons/year', power_generation=False))
            traffic.save(target / 'run_status.json', dict(status='classified_emissions_ready', concentration_generated=False))
    pd.DataFrame(totals).to_csv(OUT / 'audit/annual_emissions_by_group.csv', index=False)
    traffic.save(OUT / 'run_status.json', dict(status='emissions_ready', four_concentration_products_complete=False))
    print(pd.DataFrame(totals).to_string(index=False), flush=True)


def audit_emissions():
    keys = ['linkId', 'process', 'source_release_height', 'roadCategory']
    sums = ','.join(f'SUM("{c}") AS "{c}"' for c in MEASURES)
    joined = ' AND '.join(f'a."{c}" IS NOT DISTINCT FROM b."{c}"' for c in keys)
    invalid = ' OR '.join(f'ABS(a."{c}"-b."{c}")>1e-9+1e-9*ABS(b."{c}")' for c in MEASURES)
    errors = ','.join(f'MAX(ABS(a."{c}"-b."{c}")) AS "{c}"' for c in MEASURES)
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        report = con.execute(f'''WITH a AS (SELECT {','.join(keys)},{sums} FROM read_parquet(?) GROUP BY ALL),
            b AS (SELECT {','.join(keys)},{sums} FROM read_parquet(?) GROUP BY ALL)
            SELECT count(*) source_rows,count(*) FILTER(WHERE a.linkId IS NULL OR b.linkId IS NULL OR {invalid}) mismatched_rows,
            {errors} FROM a FULL JOIN b ON {joined}''',
            [str(OUT / 'audit/classified_link_process_emissions.parquet'),
             str(REFERENCE / 'emissions/prepared_skims_for_grid_allocation.parquet')]).fetchdf().iloc[0].to_dict()
    if report['mismatched_rows'] != 0:
        raise ValueError(f'Freight source emission conservation failed: {report}')
    traffic.save(OUT / 'audit/source_emissions_conservation.json', report)
    print('SOURCE/PROCESS/HEIGHT CONSERVATION PASSED', report, flush=True)


def validate():
    population_reference = pd.read_parquet(legacy.TOTAL / 'exposure/beam_population_counts.parquet',
                                           columns=['aermod_cell_id', 'person_count']).set_index('aermod_cell_id').sort_index()
    for group in GROUPS:
        population = pd.read_parquet(OUT / group / 'exposure/beam_population_counts.parquet',
                                     columns=['aermod_cell_id', 'person_count']).set_index('aermod_cell_id').sort_index()
        pd.testing.assert_frame_equal(population, population_reference)
    paths = [OUT / 'products' / f'{g}_concentrations.parquet' for g in GROUPS]
    paths.append(REFERENCE.parent / 'products/Freight_concentrations.parquet')
    readers = [pq.ParquetFile(p) for p in paths]
    errors = {kind:{c:0. for c in CONC_FIELDS} for kind in ['port_sum', 'freight_sum']}
    for batches in zip(*(p.iter_batches(columns=['aermod_cell_id', 'inmap_cell_id', 'geometry', *CONC_FIELDS], batch_size=131072) for p in readers), strict=True):
        frames = [b.to_pandas() for b in batches]
        for d in frames[1:]:
            for key in ['aermod_cell_id', 'inmap_cell_id', 'geometry']:
                np.testing.assert_array_equal(d[key], frames[0][key])
        sea, air, non, port, freight = frames
        for c in CONC_FIELDS:
            for kind, actual, expected in [('port_sum', sea[c]+air[c], port[c]),
                                           ('freight_sum', sea[c]+air[c]+non[c], freight[c])]:
                np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-7)
                errors[kind][c] = max(errors[kind][c], float(np.max(np.abs(actual-expected))))
    reports = {g:json.loads((OUT / g / 'run_status.json').read_text()) for g in GROUPS}
    denominator = json.loads((REFERENCE / 'run_status.json').read_text())['population_weighted_concentrations']
    rows = []
    for group, report in reports.items():
        scenario = json.loads((OUT / group / 'scenario.json').read_text())
        row = dict(group=group, sample_vehicles=scenario['sample_vehicles'], represented_vehicles=scenario['represented_vehicles'])
        for p, value in report['population_weighted_concentrations'].items():
            row[f'{p}_PWC_ug_m3'] = value
            row[f'{p}_share_of_freight_PWC_percent'] = value/denominator[p]*100
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'products/contribution_summary.csv', index=False)
    result = dict(status='complete', all_four_concentration_products_validated=True, products=reports,
                  concentration_conservation_max_absolute_errors=errors, power_generation=False,
                  fleet_expansion=10, source_emission_conservation_passed=True, population_weights_unchanged=True,
                  warning='Partial ISRM source coverage, within previously approved 103 missing IDs; AERMOD sources intact. PortRelated includes Seaport and Airport; do not sum all four.')
    traffic.save(OUT / 'results_summary.json', result)
    traffic.save(OUT / 'run_status.json', dict(status='complete', four_concentration_products_complete=True))
    (OUT / 'README.txt').write_text(
        'Baseline freight air-quality attribution, 10x expansion\n'
        'Products: Seaport, Airport, NonPort, PortRelated; each has emissions and concentration GeoParquet.\n'
        'PortRelated = Seaport + Airport; total freight = Seaport + Airport + NonPort.\n'
        'Concentrations: ug/m3; emissions: US short tons/year; 100-m receptor grid with geometry.\n'
        'Traffic only, no passenger/bus/power generation. Classification applies to entire vehicle activity.\n'
        'Hybrid traffic primary PM2.5/BC/NO2: 4.5-km source-receptor removal in InMAP + AERMOD.\n'
        'Secondary PM2.5 and InMAP-only do not apply near-source removal.\n'
        'Partial ISRM: approved missing source rows omitted; no redistribution; AERMOD retains all sources.\n'
        'Input labels/gateway IDs retained per group under inputs/vehicle_classification.csv.\n'
        'Multi-gateway tags do not imply duplicated emissions or separately modeled gateway contributions.\n'
        'Audit: raw event trajectories verified against read-only cache; emission conservation by link/process/height; concentration conservation at every receptor.\n', encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True, choices=['all', 'setup', 'allocate', 'filter', 'inmap', 'aermod', 'exposure', 'export', 'validate'])
    p.add_argument('--group', choices=list(GROUPS))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.stage == 'all':
        commands = [('setup', None)] + [(s,g) for g in GROUPS for s in ['allocate','filter','inmap','aermod','exposure','export']] + [('validate', None)]
        for stage, group in commands:
            command = [sys.executable, '-X', 'utf8', __file__, '--stage', stage]
            if group:
                command += ['--group', group]
            # Setup creates the log directory, so report its output to the caller.
            if stage == 'setup':
                subprocess.run(command, check=True)
            else:
                print(f'START {group or "ALL"} {stage}', flush=True)
                with (OUT / 'logs' / f'{group or "ALL"}_{stage}.log').open('w', encoding='utf-8') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
                print(f'DONE {group or "ALL"} {stage}', flush=True)
        return
    if args.stage in ['setup','validate']:
        return globals()[args.stage]()
    if not args.group:
        p.error('--group required')
    # Reuse the existing source-coverage, geometry, population and component QA.
    legacy.OUT = OUT
    if args.stage == 'filter':
        legacy.filter_sources(args.group)
    elif args.stage == 'export':
        legacy.export(args.group)
    else:
        traffic.run_stage(OUT / args.group, 'emissions' if args.stage == 'allocate' else args.stage)


if __name__ == '__main__':
    main()
