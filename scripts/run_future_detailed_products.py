"""Complete the future scenarios at the baseline's vehicle/gateway resolution.

Retains the completed Total products and the fixed-factor/10x/no-power scenario.
Gateway emissions are reaggregated by vehicle ID, never split by fleet shares.
All multi-gateway vehicles remain in disjoint combination groups.
"""
from pathlib import Path
import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from functools import partial

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

import run_future_fixed_factor_scenarios as future
import run_gateway_freight_products as gateway
import run_port_freight_products as broad

OUT = future.OUT
STAGES = ('allocate', 'filter', 'inmap', 'aermod', 'merge', 'validate')
MEASURES = broad.MEASURES
FIELDS = broad.CONC_FIELDS
POLICY = 'baseline45_full_aermod'


def baseline_dispersion_policy():
    """Process-local compatibility with the already-completed October Total.

    Subsequent repository defaults changed to matched circular 3-km operators.
    Do not mix those with the existing Total or mutate shared/global settings.
    """
    from impacts.pipeline.workflow import step2_compute_inmap_concentrations as inmap
    from impacts.pipeline.workflow import step3_compute_aermod_concentrations as aermod
    from impacts.pipeline.workflow.isrm_near_source import build_near_source_mask
    inmap.build_near_source_mask = partial(build_near_source_mask,radius_meters=4500.)
    # Disables only the newly added circular cutoff; original ASRV support stays
    # unchanged. A distinct cache key prevents reusing a circular-cutoff kernel.
    aermod._AERMOD_RESPONSE_RADIUS_METERS = 1e30
    logging.info('Compatibility policy: ISRM radius 4500m, >50%% overlap; original full AERMOD kernel support')


def save(path, value):
    future.traffic.save(path, value)


def catalog(labels):
    # ID/name arrays are independently sorted: never zip their elements.
    full = pd.read_csv(broad.CLASSIFICATION)
    singles = full.loc[full.is_port_related & full.gateway_ids.notna()
        & ~full.gateway_ids.fillna('').str.contains('|', regex=False)]
    if (singles.groupby('gateway_ids').gateway_names.nunique() != 1).any():
        raise ValueError('Ambiguous canonical gateway names')
    names = singles.drop_duplicates('gateway_ids').set_index('gateway_ids').gateway_names.to_dict()
    result = []
    for group, frame in labels.loc[labels.is_port_related].groupby('gateway_group'):
        parts = [] if group == 'Unknown_gateway' else group.split('__and__')
        if any(p not in names for p in parts):
            raise ValueError(f'No unambiguous canonical gateway name: {parts}')
        result.append(dict(group=group, gateway_name=' + '.join(names[p] for p in parts) if parts else 'Unknown gateway',
            gateway_ids='|'.join(parts), sample_vehicles=len(frame), represented_vehicles=10*len(frame),
            has_seaport_stop=bool(frame.has_seaport_stop.any()), has_airport_stop=bool(frame.has_airport_stop.any()),
            multiple_gateways=len(parts)>1))
    return result


def emission_parity(con, aggregate, reference):
    keys = ['linkId', 'process', 'source_release_height', 'roadCategory']
    totals = ','.join(f'SUM("{c}") AS "{c}"' for c in MEASURES)
    join = ' AND '.join(f'a."{k}" IS NOT DISTINCT FROM b."{k}"' for k in keys)
    invalid = ' OR '.join(f'ABS(a."{c}"-b."{c}")>1e-8+1e-9*ABS(b."{c}")' for c in MEASURES)
    errors = ','.join(f'MAX(ABS(a."{c}"-b."{c}")) AS "{c}"' for c in MEASURES)
    row = con.execute(f'''WITH a AS (SELECT {','.join(keys)},{totals} FROM read_parquet(?) GROUP BY ALL),
        b AS (SELECT {','.join(keys)},{totals} FROM read_parquet(?) GROUP BY ALL)
        SELECT count(*) source_rows,count(*) FILTER(WHERE a.linkId IS NULL OR b.linkId IS NULL OR {invalid}) mismatched_rows,
        {errors} FROM a FULL JOIN b ON {join}''', [str(aggregate), str(reference)]).fetchdf().iloc[0].to_dict()
    if row['mismatched_rows']:
        raise ValueError(f'Grouped emissions do not conserve reference: {row}')
    return row


def prepare_gateways(scenario):
    root = OUT/scenario
    marker = root/'audit/detailed_gateway_preparation.json'
    if marker.exists():
        return
    work = future.configure(scenario)
    audit = root/'audit'; audit.mkdir(exist_ok=True)
    labels = gateway.add_gateway_group(pd.read_csv(work/'inputs/vehicle_classification.csv'))
    if labels.vehicleId.duplicated().any() or not labels.beam_vehicle_id.eq('body-'+labels.vehicleId).all():
        raise ValueError('Ambiguous vehicle labels or body-to-vehicle mapping')
    groups = catalog(labels)
    labels.to_csv(audit/'gateway_vehicle_classification.csv', index=False)
    aggregate = audit/'gateway_link_process_emissions_10x.parquet'
    with duckdb.connect() as con:
        con.execute('SET threads=2')
        con.execute("SET memory_limit='8GB'")
        con.execute(f"ATTACH '{future.traffic.sqlpath(work/'_tmp/traffic.duckdb')}' AS original (READ_ONLY)")
        con.register('labels', labels)
        con.execute('''CREATE TEMP TABLE selected_trips AS
            SELECT t.trip_id,l.gateway_group FROM original.trips t JOIN labels l ON t.vehicle=l.vehicleId
            WHERE l.is_port_related''')
        trip_counts = con.execute('''SELECT l.gateway_group,count(*) sample_trips_0_24h,
            count(distinct t.vehicle) active_vehicles_0_24h FROM original.trips t JOIN labels l ON t.vehicle=l.vehicleId
            WHERE dep<86400 GROUP BY 1''').fetchdf()
        trip_counts.to_csv(audit/'gateway_sample_trip_counts.csv', index=False)
        con.execute('''CREATE TEMP TABLE running AS SELECT l.gateway_group,t.source_type,t.linkId,
            greatest(5,least(90,ceil((linkLength/1609.344/(seconds/3600))/5-0.5)*5)) AS speed_bin,
            sum(linkLength/1609.344*day_fraction) miles,
            sum(linkLength/1609.344*day_fraction*pow(weight_kg/907.18474,1.02)) dust_miles,
            sum(used_seconds)/3600 AS hours,sum(day_fraction) AS observations
            FROM original.traversal t JOIN selected_trips l USING(trip_id)
            WHERE used_seconds>0 GROUP BY ALL''')
        soak = [5,10,20,30,40,50,60,120,180,240,300,360,420,480,540,600,660,720]
        cases = ' '.join(f'WHEN gap/60 <= {(a+b)/2} THEN {a}' for a,b in zip(soak,soak[1:]))
        con.execute('''CREATE TEMP VIEW classified_parking AS SELECT p.*,l.gateway_group
            FROM original.parking_raw p JOIN labels l ON p.vehicle=l.vehicleId WHERE l.is_port_related''')
        con.execute(f'''CREATE TEMP TABLE parking AS SELECT gateway_group,source_type,
            list_extract(link_list,-1) linkId,'DIURN' process,CASE {cases} ELSE 720 END soak_bin,
            sum(greatest(0,least(next_dep,86400)-arr))/3600 AS hours,count(*)::DOUBLE AS observations
            FROM classified_parking WHERE gap>0 GROUP BY ALL
            UNION ALL SELECT gateway_group,source_type,list_extract(link_list,-1),'HOTSOAK',0,0,count(*)::DOUBLE
            FROM classified_parking WHERE gap>0 GROUP BY ALL
            UNION ALL SELECT gateway_group,next_type,next_link,'STREX',CASE {cases} ELSE 720 END,0,count(*)::DOUBLE
            FROM classified_parking WHERE gap>0 AND next_dep<86400 GROUP BY ALL''')
        for table in ['running','parking']:
            con.execute(f'''CREATE TEMP VIEW {table}_detail AS SELECT a.*,n.road,z.county,z.county_weight
                FROM {table} a JOIN original.network n USING(linkId) JOIN original.zones z USING(linkId)''')
        selects = []
        for kind, table, amount, bincol in [
            ('run','running_detail',"CASE WHEN r.process='PRDUST' THEN a.dust_miles WHEN r.process IN ('IDLEX','RUNLOSS') THEN a.hours ELSE a.miles END",'speed_bin'),
            ('park','parking_detail',"CASE WHEN r.process='DIURN' THEN a.hours ELSE a.observations END",'soak_bin')]:
            pollutants = ','.join(f'{amount}*r."{p}"*a.county_weight/{future.traffic.grams_per_short_ton} AS tons_per_year_{p}' for p in future.traffic.POLS)
            extra = ' AND a.process=r.process' if kind=='park' else ''
            selects.append(f'''SELECT a.gateway_group,a.linkId,r.process,r.source_release_height,a.road roadCategory,
                a.observations*r.activity_coeff*a.county_weight totTrips,
                {('a.miles' if kind=='run' else '0')}*r.activity_coeff*a.county_weight totVMT,{pollutants}
                FROM {table} a JOIN original.{kind}_coeff r ON a.source_type=r.source_type AND a.county=r.county
                AND (r.roadCategory IS NULL OR r.roadCategory='' OR lower(a.road)=r.roadCategory)
                AND (r.activity_bin=0 OR a.{bincol}=r.activity_bin){extra}''')
        sums = ','.join(f'SUM("{c}")*10 AS "{c}"' for c in MEASURES)
        con.execute(f'''COPY (SELECT gateway_group,linkId,process,source_release_height,roadCategory,{sums}
            FROM ({' UNION ALL '.join(selects)}) GROUP BY ALL)
            TO '{future.traffic.sqlpath(aggregate)}' (FORMAT PARQUET,COMPRESSION ZSTD)''')
        parity = emission_parity(con, aggregate, root/'PortRelated/emissions/prepared_skims_for_grid_allocation.parquet')
        save(audit/'gateway_emission_conservation.json', parity)
        print(scenario,'GATEWAY EMISSION CONSERVATION PASSED',flush=True)
        totals = ','.join(f'SUM("{c}") AS "{c}"' for c in MEASURES)
        reference = root/'PortRelated'
        for item in groups:
            group = item['group']; target = root/'gateways'/group
            for folder in ['inputs','emissions','concentrations','exposure','_tmp']:
                (target/folder).mkdir(parents=True,exist_ok=True)
            prepared = target/'emissions/prepared_skims_for_grid_allocation.parquet'
            con.execute(f'''COPY (SELECT linkId,?||'-height-'||cast(source_release_height AS VARCHAR) vehicleTypeId,
                process,source_release_height,roadCategory,{totals} FROM read_parquet(?) WHERE gateway_group=? GROUP BY ALL)
                TO '{future.traffic.sqlpath(prepared)}' (FORMAT PARQUET,COMPRESSION ZSTD)''',[group,str(aggregate),group])
            for filename in ['settings.yaml','preprocess_manifest.yaml']:
                text = (reference/filename).read_text(encoding='utf-8').replace(str(reference),str(target)).replace(reference.as_posix(),target.as_posix())
                data = yaml.safe_load(text)
                if filename=='preprocess_manifest.yaml':
                    assert not data['pipeline']['power_generation']['enabled']
                    assert data['pipeline']['population_sample']==1
                    data['notes'] = [f'{scenario}/{group}: disjoint vehicle-ID-based gateway subset; 10x already applied; no power or transit.']
                (target/filename).write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
            record = json.loads((reference/'scenario.json').read_text())
            record.update(product=group,gateway=item,classification_scope='All trips of associated vehicles, not only trips inside a port',
                gateway_policy='Separate multi-gateway and unknown groups; no duplicate assignment')
            save(target/'scenario.json',record)
            save(target/'run_status.json',dict(status='emissions_ready',concentration_generated=False))
        con.execute('SELECT gateway_group,'+totals+' FROM read_parquet(?) GROUP BY 1',[str(aggregate)]).fetchdf().to_csv(root/'annual_emissions_by_gateway.csv',index=False)
    save(root/'gateway_catalog.json',groups)
    save(marker,dict(status='complete',gateway_groups=len(groups),source_emissions_conserved=True,fleet_expansion=10,power_generation=False))


def run_gateway_stage(scenario, group, stage):
    target = OUT/scenario/group if group in future.GROUPS else OUT/scenario/'gateways'/group
    marker = target/f'completed_{stage}.json' if stage in ('allocate','filter') else target/f'completed_{POLICY}_{stage}.json'
    if marker.exists(): return
    logdir = OUT/scenario/'logs/detailed'; logdir.mkdir(parents=True,exist_ok=True)
    print('START',scenario,group,stage,flush=True)
    with (logdir/f'{group}_{stage}.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-X','utf8',__file__,'--scenario',scenario,'--action','stage',
            '--group',group,'--stage',stage],check=True,stdout=log,stderr=subprocess.STDOUT)
    save(marker,dict(status='complete',stage=stage,product=group,dispersion_policy=POLICY))
    print('DONE',scenario,group,stage,flush=True)


def paths(scenario):
    root = OUT/scenario
    result = {g:root/g for g in future.GROUPS}
    if (root/'gateway_catalog.json').exists():
        result.update({r['group']:root/'gateways'/r['group'] for r in json.loads((root/'gateway_catalog.json').read_text())})
    return result


def export(scenario, group, target):
    root = OUT/scenario
    dest = root/'products'/('main_categories' if group in future.GROUPS else 'gateways')
    dest.mkdir(parents=True,exist_ok=True)
    marker = target/'completed_export.json'
    if marker.exists(): return
    report = json.loads((target/'run_status.json').read_text())
    if report['status']!='complete': raise ValueError(f'Incomplete product: {target}')
    source = target/'exposure/beam_concentration_distribution.parquet'
    pf = pq.ParquetFile(source)
    output = dest/f'{group}_concentrations.parquet'
    metadata = dict(pf.schema_arrow.metadata or {})
    metadata[b'impacts_scenario'] = json.dumps(dict(scenario=scenario,vehicle_group=group,fleet_scale=10,
        power_generation=False,concentration_units='ug/m3',emission_factor_year=2018)).encode()
    with pq.ParquetWriter(output,pf.schema_arrow.with_metadata(metadata),compression='zstd') as writer:
        for batch in pf.iter_batches(batch_size=131072):
            writer.write_batch(batch.replace_schema_metadata(metadata))
    emissions = gpd.read_parquet(target/'emissions/beam_emissions_for_aermod.parquet')
    cols = [c for c in emissions if c.startswith('tons_per_year_')]
    numeric = emissions.groupby('aermod_cell_id')[cols].sum()
    geometry = emissions.drop_duplicates('aermod_cell_id').set_index('aermod_cell_id').geometry
    product = gpd.GeoDataFrame(numeric,geometry=geometry,crs=emissions.crs).reset_index()
    product.rename(columns={c:c.replace('_aermod_allocated','') for c in cols},inplace=True)
    product['vehicle_group'] = group
    product['emissions_units'] = 'short tons/year (10x already applied)'
    product.to_parquet(dest/f'{group}_emissions.parquet',index=False,compression='zstd')
    for c in cols:
        np.testing.assert_allclose(product[c.replace('_aermod_allocated','')].sum(),emissions[c].sum(),rtol=1e-12)
    if b'geo' not in pq.read_schema(output).metadata: raise ValueError('Lost grid geometry metadata')
    save(marker,dict(status='complete',concentrations=str(output),emissions=str(dest/f'{group}_emissions.parquet'),
        receptor_count=pf.metadata.num_rows,source_grid_cells=len(product)))
    print('EXPORTED',scenario,group,flush=True)


def concentration_sum(targets, reference):
    readers = [pq.ParquetFile(t/'exposure/beam_concentration_distribution.parquet') for t in [reference,*targets]]
    if len({p.metadata.num_rows for p in readers})!=1: raise ValueError('Receptor counts differ')
    errors = {c:0. for c in FIELDS}
    cols = ['aermod_cell_id','inmap_cell_id','geometry',*FIELDS]
    for batches in zip(*(p.iter_batches(columns=cols,batch_size=65536) for p in readers),strict=True):
        for b in batches[1:]:
            for key in cols[:3]:
                if not b[key].equals(batches[0][key]): raise ValueError(f'Receptor/geometry mismatch: {key}')
        for c in FIELDS:
            expected = batches[0][c].to_numpy()
            actual = sum(b[c].to_numpy() for b in batches[1:])
            np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-7,err_msg=c)
            errors[c] = max(errors[c],float(np.max(np.abs(actual-expected))))
    return dict(passed=True,grid_cells=readers[0].metadata.num_rows,max_absolute_errors_ug_m3=errors)


def summarize(scenario,main_only=False):
    root = OUT/scenario
    products = paths(scenario)
    if main_only: products = {g:t for g,t in products.items() if g in future.GROUPS}
    for group,target in products.items(): export(scenario,group,target)
    with duckdb.connect() as con:
        con.execute('SET threads=2')
        source_parity = emission_parity(con,root/'_work/emissions/classified_emissions_10x.parquet',
            root/'Total/emissions/prepared_skims_for_grid_allocation.parquet')
    save(root/'audit/category_emission_conservation.json',source_parity)
    main_parity = concentration_sum([products[g] for g in future.GROUPS],root/'Total')
    gateways = [] if main_only else json.loads((root/'gateway_catalog.json').read_text())
    gateway_parity = None if main_only else concentration_sum([products[r['group']] for r in gateways],root/'PortRelated')
    save(root/'audit/category_concentration_conservation.json',main_parity)
    if gateway_parity is not None: save(root/'audit/gateway_concentration_conservation.json',gateway_parity)
    trip_counts = pd.read_csv(root/'audit/gateway_sample_trip_counts.csv').set_index('gateway_group')
    with duckdb.connect(str(root/'_work/_tmp/traffic.duckdb'),read_only=True) as con:
        main_counts = con.execute("SELECT split_part(source_type,'@@',2) AS category,count(*) AS sample_trips_0_24h,count(distinct vehicle) AS active_vehicles_0_24h FROM trips WHERE dep<86400 GROUP BY 1").fetchdf().set_index('category')
    total = json.loads((root/'Total/run_status.json').read_text())['population_weighted_concentrations']
    rows = []
    for group,target in products.items():
        report = json.loads((target/'run_status.json').read_text())
        row = dict(scenario=scenario,group=group,grid_cells=report['grid_cells'],population=report['population'])
        if group in main_counts.index:
            row.update(main_counts.loc[group].to_dict())
        else:
            row.update(next(r for r in gateways if r['group']==group))
            if group in trip_counts.index: row.update(trip_counts.loc[group].to_dict())
        row['represented_active_vehicles_0_24h'] = 10*row.get('active_vehicles_0_24h',0)
        for p,value in report['population_weighted_concentrations'].items():
            row[f'{p}_PWC_ug_m3'] = value
            row[f'{p}_share_of_Total_PWC_percent'] = 100*value/total[p] if total[p] else 0.
            row[f'{p}_maximum_ug_m3'] = report['maxima_ug_m3'][p]
        rows.append(row)
    prefix = 'main' if main_only else 'detailed'
    pd.DataFrame(rows).to_csv(root/f'products/{prefix}_concentration_summary.csv',index=False)
    result = dict(status='complete',scenario=scenario,main_categories=[r for r in rows if r['group'] in future.GROUPS],
        gateway_groups=[r for r in rows if r['group'] not in future.GROUPS],
        main_category_sum_equals_Total=main_parity,gateway_sum_equals_PortRelated=gateway_parity,
        main_source_emissions_conserved=source_parity,
        power_generation=False,fleet_scale=10,emission_factor_year=2018,population_scaled=False,
        dispersion_policy=POLICY,isrm_removal_radius_meters=4500,aermod='Original full ASRV support; no added circular cutoff')
    save(root/f'{prefix}_results_summary.json',result)
    (root/'products/README.txt').write_text(
        'Detailed future traffic products: fixed 2018 factors; traffic expanded 10x once; no buses/power generation.\n'
        'main_categories/: Passenger, PortRelated, NonPort; each has emissions and concentration GeoParquet.\n'
        'gateways/: disjoint named-port/airport/unknown/multiple-gateway freight groups.\n'
        'All trips of gateway-associated vehicles are included, not just trips near a port.\n'
        'Multiple-gateway vehicles have their own combination groups and are not counted twice.\n'
        'emissions: short tons/year, allocated to source grids. concentrations: ug/m3 on 100-m receptor grids.\n'
        'aermod_*: local traffic; inmap_*: primary nonlocal and unchanged secondary; inmap_only_*: full InMAP response.\n'
        'TotalPM25/BC/NO2: integrated traffic concentrations; not all-source ambient concentrations.\n'
        'Population is fixed, not multiplied by 10. Missing ISRM sources remain the previously approved exclusions.\n'
        'Reproduce with scripts/run_future_detailed_products.py --scenario '+scenario+' --action all\n'
        'detailed_concentration_summary.csv: sample counts, population-weighted concentrations, contribution percentages, maxima.\n'
        '../annual_emissions_by_category.csv and ../annual_emissions_by_gateway.csv: annual source emissions.\n'
        '../audit/: per-link/process/height emission conservation and receptor-level concentration conservation.\n',encoding='utf-8')
    print('DETAILED NUMERIC PRODUCTS VERIFIED',scenario,flush=True)


def readable_pwc_chart(rows,title,path,dpi):
    """Use the original chart, allowing enough row height for long gateway names."""
    import plot_gateway_products as plots
    line_count=max(len(plots.textwrap.wrap(r.get('gateway_name',plots.MAIN.get(r['group'],r['group'])),32)) for r in rows)
    height=max(5.,len(rows)*max(.64,.25*line_count+.2))
    original_finish=plots.finish
    def resized_finish(fig,destination,resolution):
        fig.set_size_inches(22,height)
        fig.tight_layout(rect=(0,.07,1,.94))
        original_finish(fig,destination,resolution)
    plots.finish=resized_finish
    try:
        plots.pwc_chart(rows,title,path,dpi)
    finally:
        plots.finish=original_finish


def plot(scenario,dpi=150,main_only=False):
    import plot_gateway_products as plots
    from PIL import Image
    root = OUT/scenario; out = root/'figures_detailed'
    results = json.loads((root/('main_results_summary.json' if main_only else 'detailed_results_summary.json')).read_text())
    sources = {g:t/'exposure/beam_concentration_distribution.parquet' for g,t in paths(scenario).items()}
    if main_only: sources = {g:t for g,t in sources.items() if g in future.GROUPS}
    names = {**plots.MAIN,**{r['group']:r['gateway_name'] for r in results['gateway_groups']}}
    plots.NOTE = f'{scenario}: fixed 2018 factors; traffic x10; no buses/power; partial ISRM coverage'
    mask,ids,layout,net,counties,context = plots.load_context(root,sources['Passenger'],manifest_path=root/'Passenger/preprocess_manifest.yaml')
    manifest_path = out/('main_figure_manifest.json' if main_only else 'figure_manifest.json')
    summary = {}
    for g,source in sources.items():
        folder = out/('main_categories' if g in future.GROUPS else 'gateways')/g
        marker = folder/'figure_summary.json'
        if marker.exists():
            summary[g] = json.loads(marker.read_text())
            continue
        data = plots.load_values(source,mask,ids)
        limits = {p:plots.positive_limit(data,[p]) for p in plots.POLLUTANTS}
        summary[g] = dict(name=names[g],source=str(source),limits=limits,
            true_sfbay_maxima={p:float(data[p].max()) for p in plots.POLLUTANTS})
        for p,title in plots.POLLUTANTS.items():
            plots.single_map(data,p,title,names[g],layout,net,counties,limits[p],folder/f'{p.lower()}.png',dpi)
        if g in future.GROUPS:
            plots.component_comparison(data,names[g],layout,net,counties,folder/'primary_secondary_pm25.png',dpi)
            vmax = plots.positive_limit(data,[c for c,_,_,_ in plots.native._PM25_COMPONENT_LAYERS])
            summary[g]['component_shared_vmax'] = vmax
            for col,title,filename,_ in plots.native._PM25_COMPONENT_LAYERS:
                plots.single_map(data,col,title,names[g],layout,net,counties,vmax,folder/filename,dpi)
        save(marker,summary[g]); del data; gc.collect()
    load = lambda g:plots.load_values(sources[g],mask,ids)
    scales = {}
    for key,groups in [('main_categories',list(future.GROUPS)),('gateways',[r['group'] for r in results['gateway_groups']])]:
        if not groups: continue
        scales[key] = plots.comparison_figures(groups,summary,layout,net,counties,load,out/'summary'/key,dpi,comparison_title=f'{scenario}: {key.replace("_"," ")}')
    readable_pwc_chart(results['main_categories'],f'{scenario}: population-weighted traffic contributions',out/'summary/main_categories_pwc.png',dpi)
    if not main_only: readable_pwc_chart(results['gateway_groups'],f'{scenario}: gateway freight contributions',out/'summary/gateway_pwc.png',dpi)
    files = []
    for path in out.rglob('*.png'):
        if main_only and ('gateways' in path.parts or path.name=='gateway_pwc.png'): continue
        with Image.open(path) as im:
            pixels=im.size; im.verify()
        files.append(dict(file=str(path.relative_to(out)),pixels=pixels))
    expected = 8*len(future.GROUPS)+3*len(results['gateway_groups'])+(4 if main_only else 8)
    if len(files)!=expected: raise ValueError(f'Expected {expected} images, found {len(files)}')
    save(manifest_path,dict(status='complete',scenario=scenario,context=context,groups=summary,shared_scales=scales,
        files=files,dpi=dpi,units='ug/m3',power_generation=False,fleet_scale_already_applied=10,
        color_limits='Positive-cell P99 (not maxima); common scales on comparison panels',
        native_helpers='plot_gateway_products and IMPACTS postprocess helpers'))
    print('FIGURES COMPLETE',scenario,len(files),flush=True)


def render_ready_gateways(scenario,dpi=150):
    """Render validated groups while the remaining independent groups run."""
    import plot_gateway_products as plots
    root=OUT/scenario
    groups=json.loads((root/'gateway_catalog.json').read_text())
    plots.NOTE=f'{scenario}: fixed 2018 factors; traffic x10; no buses/power; partial ISRM coverage'
    mask,ids,layout,net,counties,_=plots.load_context(root,root/'Total/exposure/beam_concentration_distribution.parquet',
        manifest_path=root/'Total/preprocess_manifest.yaml')
    pending={r['group']:r for r in groups}
    while pending:
        progressed=False
        for group,item in list(pending.items()):
            target=root/'gateways'/group
            folder=root/'figures_detailed/gateways'/group
            marker=folder/'figure_summary.json'
            if marker.exists():
                del pending[group]; progressed=True; continue
            if not (target/f'completed_{POLICY}_validate.json').exists(): continue
            source=target/'exposure/beam_concentration_distribution.parquet'
            data=plots.load_values(source,mask,ids)
            limits={p:plots.positive_limit(data,[p]) for p in plots.POLLUTANTS}
            report=dict(name=item['gateway_name'],source=str(source),limits=limits,
                true_sfbay_maxima={p:float(data[p].max()) for p in plots.POLLUTANTS})
            for p,title in plots.POLLUTANTS.items():
                plots.single_map(data,p,title,item['gateway_name'],layout,net,counties,limits[p],folder/f'{p.lower()}.png',dpi)
            save(marker,report); del data; gc.collect()
            del pending[group]; progressed=True
            print('VALIDATED GATEWAY FIGURES READY',scenario,group,flush=True)
        if not progressed: time.sleep(5)


def finalize():
    import plot_gateway_products as plots
    from PIL import Image
    results = {}
    main_rows, gateway_rows, inventory = [], [], []
    for scenario in future.SCENARIOS:
        root = OUT/scenario
        result = json.loads((root/'detailed_results_summary.json').read_text())
        figures = json.loads((root/'figures_detailed/figure_manifest.json').read_text())
        if result['status']!='complete' or figures['status']!='complete':
            raise ValueError(f'Incomplete detailed result: {scenario}')
        plots.NOTE=f'{scenario}: fixed 2018 factors; traffic x10; no buses/power; partial ISRM coverage'
        readable_pwc_chart(result['gateway_groups'],f'{scenario}: gateway freight contributions',
            root/'figures_detailed/summary/gateway_pwc.png',figures['dpi'])
        for entry in figures['files']:
            with Image.open(root/'figures_detailed'/entry['file']) as im:
                entry['pixels']=im.size
                im.verify()
        save(root/'figures_detailed/figure_manifest.json',figures)
        main_rows.extend(result['main_categories']); gateway_rows.extend(result['gateway_groups'])
        results[scenario] = dict(main_categories=len(result['main_categories']),gateway_groups=len(result['gateway_groups']),
            figures=len(figures['files']),products_directory=str(root/'products'),figures_directory=str(root/'figures_detailed'),
            main_sum_equals_Total=result['main_category_sum_equals_Total']['passed'],
            gateway_sum_equals_PortRelated=result['gateway_sum_equals_PortRelated']['passed'])
        for group in paths(scenario):
            folder=root/'products'/('main_categories' if group in future.GROUPS else 'gateways')
            for kind in ['emissions','concentrations']:
                path=folder/f'{group}_{kind}.parquet'; pf=pq.ParquetFile(path)
                if not pf.metadata.num_rows or b'geo' not in pf.schema_arrow.metadata:
                    raise ValueError(f'Empty/non-geospatial final product: {path}')
                if kind=='concentrations' and pf.metadata.num_rows!=result['main_category_sum_equals_Total']['grid_cells']:
                    raise ValueError(f'Incomplete receptor grid: {path}')
                inventory.append(dict(scenario=scenario,group=group,kind=kind,path=str(path),
                    rows=pf.metadata.num_rows,size_MB=path.stat().st_size/1e6,grid_geometry_included=True,
                    units='ug/m3' if kind=='concentrations' else 'short tons/year',fleet_scale_already_applied=10))
    pd.DataFrame(main_rows).to_csv(OUT/'detailed_main_category_comparison.csv',index=False)
    pd.DataFrame(gateway_rows).to_csv(OUT/'detailed_gateway_comparison.csv',index=False)
    pd.DataFrame(inventory).to_csv(OUT/'detailed_product_inventory.csv',index=False)
    # Compare only matching main groups to the previously delivered baseline.
    baseline_path = gateway.OUT/'results_summary.json'
    baseline = {r['group']:r for r in json.loads(baseline_path.read_text())['main_categories']}
    changes=[]
    for row in main_rows:
        for pollutant in ['TotalPM25','BC','NO2']:
            key = pollutant+'_PWC_ug_m3'; old = baseline[row['group']][key]; new = row[key]
            changes.append(dict(scenario=row['scenario'],group=row['group'],pollutant=pollutant,
                baseline_PWC_ug_m3=old,future_PWC_ug_m3=new,change_ug_m3=new-old,
                percent_change=100*(new-old)/old if old else None))
    pd.DataFrame(changes).to_csv(OUT/'detailed_main_category_vs_baseline.csv',index=False)
    save(OUT/'detailed_results_summary.json',dict(status='complete',scenarios=results,
        main_category_products=len(main_rows),gateway_products=len(gateway_rows),
        geospatial_product_files=len(inventory),all_product_files_nonempty=True,
        total_figures=sum(r['figures'] for r in results.values()),
        fleet_expansion=10,power_generation=False,emission_factor_year=2018,dispersion_policy=POLICY,
        baseline_reference=str(baseline_path),old_Total_results_preserved=True))
    readme = OUT/'README.txt'
    text = readme.read_text(encoding='utf-8')
    old = 'Category subfolders contain prepared emissions; separate category dispersion is not part of this total-scenario run.'
    new = ('Detailed update 2026-10-28: Passenger/PortRelated/NonPort and disjoint gateway dispersion are complete.\n'
        'Each scenario/products/main_categories and products/gateways: emissions and concentration GeoParquet.\n'
        'Each scenario/figures_detailed: category/component maps, gateway maps and common-scale/PWC comparisons.\n'
        'detailed_main_category_comparison.csv, detailed_gateway_comparison.csv: all detailed results.\n'
        'detailed_main_category_vs_baseline.csv: matched-group PWC changes from the original baseline.\n'
        'detailed_results_summary.json: completion, conservation checks and exact output counts.\n'
        'detailed_product_inventory.csv: every final GeoParquet file, row count, size and units.\n'
        'Compatibility: ISRM 4.5-km removal (>50% receptor overlap), original full AERMOD kernel support.\n'
        'This matches the existing October Total and baseline; it is NOT the later 3-km circular-cutoff experiment.\n'
        'The newly generated 3-km diagnostic run is isolated under _diagnostic_3km_20261028, not a final product.\n'
        'Multi-gateway vehicles are separate combination groups. All vehicle trips are attributed, not just port visits.\n'
        'Annual skim totTrips/totVMT fields are process/cohort activity measures, not unique trips; use sample trip counts in detailed summaries.\n'
        'No expanded cohort database or ZIP archive was created for this detailed update.')
    if old in text: readme.write_text(text.replace(old,new),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenario',choices=list(future.SCENARIOS))
    p.add_argument('--action',choices=['prepare','run','main','stage','summarize','plot','ready-plots','main-output','finalize','all'],required=True)
    p.add_argument('--group'); p.add_argument('--stage',choices=STAGES)
    p.add_argument('--dpi',type=int,default=150)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.action=='finalize': return finalize()
    if not args.scenario: p.error('--scenario is required except for finalize')
    if args.action in ('prepare','all'): prepare_gateways(args.scenario)
    if args.action in ('main','all'):
        for group in future.GROUPS:
            for stage in STAGES: run_gateway_stage(args.scenario,group,stage)
    if args.action in ('run','all'):
        for item in json.loads((OUT/args.scenario/'gateway_catalog.json').read_text()):
            for stage in STAGES: run_gateway_stage(args.scenario,item['group'],stage)
    if args.action=='stage':
        future.configure(args.scenario)
        baseline_dispersion_policy()
        product = args.group if args.group in future.GROUPS else 'gateways/'+args.group
        if args.stage=='validate':
            future.validate(args.scenario,product)
            report_path = OUT/args.scenario/product/'run_status.json'
            report = json.loads(report_path.read_text())
            report.update(dispersion_policy=POLICY,isrm_removal_radius_meters=4500,
                aermod_policy='Original full ASRV support; no added circular cutoff')
            save(report_path,report)
        else: future.disperse(args.scenario,product,args.stage)
    if args.action in ('summarize','all'): summarize(args.scenario)
    if args.action in ('plot','all'): plot(args.scenario,args.dpi)
    if args.action=='main-output':
        summarize(args.scenario,main_only=True)
        plot(args.scenario,args.dpi,main_only=True)
    if args.action=='ready-plots': render_ready_gateways(args.scenario,args.dpi)


if __name__=='__main__': main()
