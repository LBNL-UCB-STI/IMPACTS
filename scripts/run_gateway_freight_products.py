"""Three main traffic groups plus disjoint port-name freight attribution.

Reuses the prior checked vehicle-ID/EMFAC path and native dispersion operators.
Only the FFT empty-domain work and redundant output formats are optimized.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import gc
import json
import logging
import shutil
import subprocess
import sys

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

import run_port_freight_products as broad
import run_ldv_freight_products as legacy
from impacts.manifest.schema import PipelineConfig
from impacts.pipeline.workflow import step3_compute_aermod_concentrations as aermod
from impacts.pipeline.workflow import step4_prepare_exposure as exposure
from gateway_aermod_crop import apply_cropped

PRIOR = broad.OUT
OUT = PRIOR.with_name('events-fc07-passenger-port-gateway10x-20261001')
MAIN = {
    'Passenger': (legacy.OUT / 'LDV', legacy.OUT / 'products/LDV'),
    'PortRelated': (PRIOR / 'PortRelated', PRIOR / 'products/PortRelated'),
    'NonPort': (PRIOR / 'NonPort', PRIOR / 'products/NonPort'),
}


def add_gateway_group(d):
    def classify(row):
        if not row.is_port_related:
            return 'NonPort'
        if pd.isna(row.gateway_ids) or not str(row.gateway_ids).strip():
            return 'Unknown_gateway'
        parts = sorted(set(str(row.gateway_ids).split('|')))
        return '__and__'.join(parts)
    d['gateway_group'] = d.apply(classify, axis=1)
    return d


def gateway_catalog(d):
    # Lists of IDs and names are independently sorted in the input. Never zip
    # them positionally. Resolve names using unambiguous single-gateway rows.
    singles = d.loc[d.is_port_related & d.gateway_ids.notna() & ~d.gateway_ids.fillna('').str.contains('|', regex=False)]
    if (singles.groupby('gateway_ids').gateway_names.nunique() > 1).any():
        raise ValueError('Conflicting canonical gateway names')
    names = singles.drop_duplicates('gateway_ids').set_index('gateway_ids').gateway_names.to_dict()
    rows = []
    for group, frame in d.loc[d.is_port_related].groupby('gateway_group'):
        ids = [] if group == 'Unknown_gateway' else group.split('__and__')
        if any(g not in names for g in ids):
            raise ValueError('A gateway ID has no unambiguous name')
        rows.append(dict(group=group, gateway_name=' + '.join(names[g] for g in ids) if ids else 'Unknown gateway',
            gateway_ids='|'.join(ids), sample_vehicles=len(frame), represented_vehicles=len(frame)*10,
            has_seaport_stop=bool(frame.has_seaport_stop.any()), has_airport_stop=bool(frame.has_airport_stop.any()),
            multiple_gateways=len(ids)>1))
    return rows


def setup():
    labels = add_gateway_group(broad.validate_labels(pd.read_csv(broad.CLASSIFICATION)))
    catalog = gateway_catalog(labels)
    broad.OUT = OUT
    broad.GROUPS = {r['group']:r['group'] for r in catalog}
    broad.setup(group_column='gateway_group', labels_transform=add_gateway_group)
    broad.traffic.save(OUT / 'gateway_catalog.json', catalog)
    pd.DataFrame(catalog).to_csv(OUT / 'products/gateway_catalog.csv', index=False)
    for row in catalog:
        scenario_file = OUT / row['group'] / 'scenario.json'
        scenario = json.loads(scenario_file.read_text())
        scenario.update(gateway=row, method='Vehicle-ID-resolved disjoint gateway grouping; original EMFAC and dispersion operators',
            optimization='Source-support bounding rectangles for unchanged AERMOD FFT; GeoParquet only; shared population',
            overlap_policy='Multi-gateway vehicles are separate combination groups, never duplicated or split arbitrarily')
        broad.traffic.save(scenario_file, scenario)
    main = OUT / 'products/main_categories'
    main.mkdir()
    references = {}
    for group, (folder, prefix) in MAIN.items():
        report = json.loads((folder / 'run_status.json').read_text())
        if report['status'] != 'complete' or report['power_generation'] or report['fleet_scale'] != 10:
            raise ValueError(f'Invalid main-category reference {group}')
        for kind in ['emissions','concentrations']:
            source = prefix.with_name(prefix.name + f'_{kind}.parquet')
            target = main / f'{group}_{kind}.parquet'
            shutil.copy2(source, target)
            if broad.sha256(source) != broad.sha256(target):
                raise ValueError('Copied reference checksum mismatch')
        references[group] = dict(source_directory=str(folder), report=report)
    broad.traffic.save(OUT / 'audit/reused_main_categories.json', references)
    # Show actual sample trip counts separately from vehicle counts. This is
    # the original road-trip activity table, not pedestrian body events.
    with duckdb.connect(str(broad.SOURCE / '_tmp/traffic.duckdb'), read_only=True) as con:
        con.register('gateway_labels', labels)
        trips = con.execute('''SELECT gateway_group,count(*) sample_trips_0_24h,
            count(distinct t.vehicle) active_vehicles_0_24h
            FROM trips t JOIN gateway_labels g ON t.vehicle=g.vehicleId
            WHERE dep<86400 GROUP BY gateway_group''').fetchdf()
        trips.to_csv(OUT / 'audit/sample_freight_trip_counts.csv', index=False)
        mix = pd.read_csv(broad.SOURCE / 'inputs/vehicle_emfac_mixture.csv')
        passenger = mix.loc[mix.assignment_group.eq('passenger'), ['sourceVehicleTypeId']].drop_duplicates()
        con.register('passenger_types', passenger)
        count = con.execute('''SELECT count(distinct vehicle) sample_vehicles,count(*) sample_trips_0_24h
            FROM trips t JOIN passenger_types p ON t.source_type=p.sourceVehicleTypeId WHERE dep<86400''').fetchdf().iloc[0].to_dict()
        broad.traffic.save(OUT / 'audit/passenger_sample_counts.json', count)
    broad.traffic.save(OUT / 'run_status.json', dict(status='gateway_emissions_ready', gateway_groups=len(catalog), complete=False))
    print('SETUP', len(catalog), 'gateway groups; 3 main categories copied with checksum verification', flush=True)


def parquet_only(gdf, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path, index=False)


def run_group(group):
    for stage in ['allocate','filter','inmap','aermod','merge','export']:
        marker = OUT / group / f'completed_{stage}.json'
        if marker.exists():
            continue
        command = [sys.executable, '-X','utf8',__file__,'--stage',stage,'--group',group]
        print('START',group,stage,flush=True)
        with (OUT / 'logs' / f'{group}_{stage}.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        broad.traffic.save(marker, dict(status='complete',stage=stage))
        print('DONE',group,stage,flush=True)


def merge(group):
    target = OUT / group
    pipeline = PipelineConfig.from_dict(yaml.safe_load((target / 'preprocess_manifest.yaml').read_text())['pipeline'])
    result = exposure._build_full_exposure_grid(pipeline=pipeline,
        prepared_inmap=exposure._prepare_inmap_exposure_inputs(gpd.read_parquet(target/'concentrations/beam_inmap_concentrations.parquet')),
        prepared_aermod=exposure._prepare_aermod_exposure_inputs(gpd.read_parquet(target/'concentrations/beam_aermod_concentrations.parquet')),
        working_dir=target/'exposure')
    result.to_parquet(target/'exposure/beam_concentration_distribution.parquet', index=False)
    broad.traffic.save(target/'exposure/shared_population.json', dict(
        population_counts_file=str(legacy.TOTAL/'exposure/beam_population_counts.parquet'),
        population_scale=1, grid_file=pipeline.aermod_full_grid_path,
        note='Same population and full-grid geometry as verified main categories; not regenerated or scaled.'))


def parity(group, reference=None):
    """Compare optimized real-data AERMOD to an independent original run."""
    if reference:
        target = Path(reference)
        result_dir = OUT / 'audit/real_operator_parity'
    else:
        target = OUT / group
        result_dir = target / 'parity_original'
    result_dir.mkdir(parents=True, exist_ok=True)
    pipeline = PipelineConfig.from_dict(yaml.safe_load((target/'preprocess_manifest.yaml').read_text())['pipeline'])
    aermod._write_outputs = parquet_only
    if reference:
        aermod._apply_kernels = apply_cropped
    result, _, path = aermod.run(pipeline=pipeline, raw_dir=result_dir,
        emissions_input_path=str(target/'emissions/beam_emissions_for_aermod.parquet'))
    del result
    gc.collect()
    a = pq.ParquetFile(path)
    b = pq.ParquetFile(target/'concentrations/beam_aermod_concentrations.parquet')
    fields = ['aermod_cell_id','PrimaryPM25','BC','NO2','has_aermod_primarypm25','has_aermod_bc','has_aermod_no2']
    errors = {c:0. for c in ['PrimaryPM25','BC','NO2']}
    for ba, bb in zip(a.iter_batches(columns=fields),b.iter_batches(columns=fields),strict=True):
        da, db = ba.to_pandas(), bb.to_pandas()
        for c in fields:
            if c in errors:
                np.testing.assert_allclose(da[c],db[c],atol=1e-10,rtol=1e-9)
                errors[c] = max(errors[c], float(np.max(np.abs(da[c]-db[c]))))
            else:
                np.testing.assert_array_equal(da[c],db[c])
    broad.traffic.save(result_dir/'operator_parity.json', dict(passed=True,max_absolute_errors=errors,reference=str(target)))
    print('REAL OPERATOR PARITY PASSED',errors,flush=True)


def validate():
    catalog = json.loads((OUT/'gateway_catalog.json').read_text())
    groups = [r['group'] for r in catalog]
    # Sum every gateway output at every receptor, retaining separate seaport
    # and airport sums so a category-specific error cannot cancel globally.
    refs = {g:pq.ParquetFile(PRIOR/'products'/f'{g}_concentrations.parquet') for g in ['PortRelated','Seaport','Airport']}
    streams = [pq.ParquetFile(OUT/'products'/f'{g}_concentrations.parquet') for g in groups] + list(refs.values())
    fields = ['aermod_cell_id','inmap_cell_id','geometry',*broad.CONC_FIELDS]
    errors = {g:{c:0. for c in broad.CONC_FIELDS} for g in refs}
    for batches in zip(*(f.iter_batches(columns=fields,batch_size=65536) for f in streams),strict=True):
        reference_frames = [b.to_pandas() for b in batches[-3:]]
        sums = {g:np.zeros((len(reference_frames[0]),len(broad.CONC_FIELDS))) for g in refs}
        for info, batch in zip(catalog,batches[:-3],strict=True):
            d = batch.to_pandas()
            for c in ['aermod_cell_id','inmap_cell_id','geometry']:
                np.testing.assert_array_equal(d[c],reference_frames[0][c])
            values = d[broad.CONC_FIELDS].to_numpy()
            assert np.isfinite(values).all() and (values >= -1e-10).all()
            sums['PortRelated'] += values
            sums['Seaport' if info['has_seaport_stop'] else 'Airport'] += values
        for (group, actual), ref in zip(sums.items(),reference_frames,strict=True):
            np.testing.assert_allclose(actual,ref[broad.CONC_FIELDS].to_numpy(),atol=1e-8,rtol=1e-7)
            for c, maximum in zip(broad.CONC_FIELDS,np.abs(actual-ref[broad.CONC_FIELDS].to_numpy()).max(axis=0)):
                errors[group][c] = max(errors[group][c],float(maximum))
    # Independently check source-grid emissions partitioning.
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        files = [str(OUT/'products'/f'{g}_emissions.parquet') for g in groups]
        pcols = [f'tons_per_year_{p}' for p in broad.POLS]
        summed = ','.join(f'sum("{c}") AS "{c}"' for c in pcols)
        left = con.execute(f'SELECT aermod_cell_id,{summed} FROM read_parquet(?) GROUP BY aermod_cell_id ORDER BY aermod_cell_id',[files]).fetchdf()
        right = con.execute(f'SELECT aermod_cell_id,{summed} FROM read_parquet(?) GROUP BY aermod_cell_id ORDER BY aermod_cell_id',
                            [str(PRIOR/'products/PortRelated_emissions.parquet')]).fetchdf()
        np.testing.assert_array_equal(left.aermod_cell_id,right.aermod_cell_id)
        np.testing.assert_allclose(left[pcols],right[pcols],atol=1e-10,rtol=1e-9)
    reference_reports = json.loads((OUT/'audit/reused_main_categories.json').read_text())
    main_rows = []
    denominators = {p:sum(r['report']['population_weighted_concentrations'][p] for r in reference_reports.values()) for p in ['TotalPM25','BC','NO2']}
    for group, item in reference_reports.items():
        row = dict(group=group)
        for p,value in item['report']['population_weighted_concentrations'].items():
            row[f'{p}_PWC_ug_m3'] = value
            row[f'{p}_share_of_selected_traffic_percent'] = value/denominators[p]*100
        main_rows.append(row)
    gateway_rows = []
    trips = pd.read_csv(OUT/'audit/sample_freight_trip_counts.csv').set_index('gateway_group')
    for info in catalog:
        report = json.loads((OUT/info['group']/'run_status.json').read_text())
        assert report['status'] == 'complete'
        row = dict(**info, sample_trips_0_24h=int(trips.loc[info['group'],'sample_trips_0_24h']),
            omitted_isrm_sources=report['omitted_inmap_source_count'])
        for p, value in report['population_weighted_concentrations'].items():
            row[f'{p}_PWC_ug_m3'] = value
            row[f'{p}_share_of_port_PWC_percent'] = value/reference_reports['PortRelated']['report']['population_weighted_concentrations'][p]*100
        gateway_rows.append(row)
    pd.DataFrame(main_rows).to_csv(OUT/'products/main_categories_summary.csv',index=False)
    pd.DataFrame(gateway_rows).to_csv(OUT/'products/gateway_contribution_summary.csv',index=False)
    parity_report = json.loads((OUT/'audit/real_operator_parity/operator_parity.json').read_text())
    if not parity_report['passed']:
        raise ValueError('Missing real-data operator validation')
    broad.traffic.save(OUT/'results_summary.json',dict(status='complete',main_categories=main_rows,gateway_groups=gateway_rows,
        gateway_count=len(groups),fleet_expansion=10,power_generation=False,passenger_excludes_transit=True,
        concentration_sum_max_errors=errors,source_emission_sum_passed=True,real_operator_parity=parity_report,
        population_source=str(legacy.TOTAL/'exposure/beam_population_counts.parquet'),population_scaled=False,
        caveat='Partial ISRM sources as previously approved; vehicle/tour-level association, not stop-resolved gateway trip labels.'))
    broad.traffic.save(OUT/'run_status.json',dict(status='complete',gateway_groups=len(groups),complete=True))
    (OUT/'README.txt').write_text(
        'Baseline traffic attribution, 10x traffic scaling, no power generation, no transit.\n'
        'products/main_categories: Passenger, PortRelated, NonPort emission and concentration GeoParquet; copied unchanged with checksum verification.\n'
        'products: 14 disjoint gateway-group emission and concentration GeoParquet; concentrations ug/m3, emissions US short tons/year.\n'
        'gateway_contribution_summary.csv: PWC weighted by the same Bay Area population; percentages relative to total port-related freight.\n'
        'main_categories_summary.csv: shares relative to passenger + port freight + non-port freight (excludes transit).\n'
        'Single gateways, multi-gateway combinations, and Unknown are mutually exclusive and add to PortRelated.\n'
        'Classification applies to all trips of labeled vehicles; not only port-bound trips or emissions inside port boundaries.\n'
        'Use per-group inputs/vehicle_classification.csv for original seaport/airport flags and gateway tags. IDs/names lists are not positionally zipped.\n'
        'Original EMFAC, InMAP, AERMOD and exposure functions retained. AERMOD crops zero-source rectangles only, retaining full kernel support; validated against original and group sums.\n'
        'InMAP-only has no local removal; hybrid traffic primary PM2.5/BC/NO2 use original 4.5km source-mask rule; secondary unchanged.\n'
        'Previously approved missing ISRM sources remain omitted with per-group audits; all AERMOD sources retained.\n'
        'No redundant GPKG/population copies for gateway groups; shared grid and population; geometry is in every concentration GeoParquet.\n', encoding='utf-8')
    print(pd.DataFrame(main_rows).to_string(index=False),flush=True)
    print(pd.DataFrame(gateway_rows).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True,choices=['all','resume','setup','allocate','filter','inmap','aermod','merge','export','validate','parity'])
    p.add_argument('--group')
    args=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.stage in ['all','resume']:
        if args.stage=='all':
            subprocess.run([sys.executable,'-X','utf8',__file__,'--stage','setup'],check=True)
        if not (OUT/'audit/real_operator_parity/operator_parity.json').exists():
            with (OUT/'logs/operator_parity.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-X','utf8',__file__,'--stage','parity'],check=True,stdout=log,stderr=subprocess.STDOUT)
            print('REAL-DATA CROP PARITY PASSED',flush=True)
        catalog=json.loads((OUT/'gateway_catalog.json').read_text())
        # Two processes, isolated outputs. The largest source group starts first.
        ordered=sorted(catalog,key=lambda r:r['sample_vehicles'],reverse=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run_group,[r['group'] for r in ordered]))
        subprocess.run([sys.executable,'-X','utf8',__file__,'--stage','validate'],check=True)
        return
    if args.stage=='setup':return setup()
    if args.stage=='validate':return validate()
    if args.stage=='parity':return parity(args.group,reference=PRIOR/'Airport')
    catalog=json.loads((OUT/'gateway_catalog.json').read_text())
    if args.group not in {r['group'] for r in catalog}:raise ValueError('Unknown gateway group')
    legacy.OUT=OUT
    target=OUT/args.group
    if args.stage=='filter':return legacy.filter_sources(args.group)
    if args.stage=='export':return legacy.export(args.group)
    if args.stage=='merge':return merge(args.group)
    if args.stage=='aermod':
        aermod._apply_kernels=apply_cropped
        aermod._write_outputs=parquet_only
    broad.traffic.run_stage(target,'emissions' if args.stage=='allocate' else args.stage)


if __name__=='__main__':main()
