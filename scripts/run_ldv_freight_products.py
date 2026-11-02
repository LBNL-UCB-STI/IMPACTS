"""Source-resolved LDV/Freight products; never split concentrations by a fleet ratio."""
from research_paths import ISRM_PATH, BREATHE_ROOT, BEAM_DATA_ROOT, CHARGING_REFERENCE
from pathlib import Path
import argparse
import gc
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

from run_events_traffic_pipeline import ROOT, BASE, POLS, run_stage, save, sqlpath

ORIGINAL = ROOT/'examples/pilates/impacts/impacts_output/events-fc07-traffic-only-20260925-baseline-mixture-v1'
TOTAL = ORIGINAL.with_name(ORIGINAL.name+'-partial-isrm103-fulltraffic10x')
OUT = ORIGINAL.with_name(ORIGINAL.name+'-ldv-freight10x')
FACTOR = 10.
GROUPS = {'LDV': 'passenger', 'Freight': 'freight'}


def classify_mapping(mapping):
    required = ['vehicleTypeId', 'sourceVehicleTypeId', 'assignment_group']
    if mapping[required].isna().any().any() or mapping.vehicleTypeId.duplicated().any():
        raise ValueError('Ambiguous or missing source vehicle mapping')
    if not set(mapping.assignment_group).issubset({'passenger', 'freight', 'transit'}):
        raise ValueError('Unclassified source vehicle group')
    if not mapping.loc[mapping.sourceVehicleTypeId.str.startswith('ft-'), 'assignment_group'].eq('freight').all():
        raise ValueError('Freight source misclassified')
    return mapping[required].copy()


def replace_paths(value, target):
    if isinstance(value, str):
        return value.replace(str(TOTAL), str(target)).replace(TOTAL.as_posix(), target.as_posix())
    if isinstance(value, list):return [replace_paths(v,target) for v in value]
    if isinstance(value, dict):return {k:replace_paths(v,target) for k,v in value.items()}
    return value


def setup():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    (OUT/'audit').mkdir();(OUT/'_tmp').mkdir();(OUT/'products').mkdir()
    mapping=pd.read_csv(ORIGINAL/'inputs/vehicle_emfac_mixture.csv',low_memory=False)
    lookup=classify_mapping(mapping)
    lookup.to_csv(OUT/'audit/vehicle_group_mapping.csv',index=False)
    con=duckdb.connect();con.execute('SET threads=4');con.execute("SET memory_limit='24GB'")
    con.execute(f"SET temp_directory='{sqlpath(OUT/'_tmp')}'")
    con.register('vehicle_lookup',lookup)
    source=ORIGINAL/'emissions/prepared_skims_for_grid_allocation.parquet'
    measures=['totTrips','totVMT']+[f'tons_per_year_{p}' for p in POLS]
    sums=','.join(f'SUM(e."{c}")*{FACTOR} AS "{c}"' for c in measures)
    print('Classifying original source-resolved EMFAC emissions, then expanding 10x',flush=True)
    aggregate=OUT/'audit/classified_link_process_emissions.parquet'
    con.execute(f'''COPY (SELECT m.assignment_group,e.linkId,e.process,e.source_release_height,e.roadCategory,{sums}
        FROM read_parquet('{sqlpath(source)}') e LEFT JOIN vehicle_lookup m USING(vehicleTypeId)
        GROUP BY ALL) TO '{sqlpath(aggregate)}' (FORMAT PARQUET, COMPRESSION ZSTD)''')
    grouped=con.execute('SELECT DISTINCT assignment_group FROM read_parquet(?)',[str(aggregate)]).fetchall()
    if any(g[0] is None for g in grouped):raise ValueError('Unmapped source rows; refusing incomplete split')
    totals=con.execute(f'''SELECT assignment_group,{','.join('SUM("'+c+'") AS "'+c+'"' for c in measures)}
        FROM read_parquet(?) GROUP BY assignment_group''',[str(aggregate)]).fetchdf()
    reference=json.loads((TOTAL/'emissions_audit.json').read_text())
    for p in POLS:np.testing.assert_allclose(totals[f'tons_per_year_{p}'].sum(),reference[p],rtol=1e-10)
    totals.to_csv(OUT/'audit/annual_emissions_by_group.csv',index=False)
    save(OUT/'audit/split_conservation.json',dict(all_pollutant_totals_match_previous_fulltraffic=True,
        traffic_expansion=FACTOR,partition='passenger + freight + transit = previous all-traffic emissions',
        transit_excluded_from_two_requested_groups=True,source_file=str(source)))
    for label,group in GROUPS.items():
        target=OUT/label
        for name in ['inputs','emissions','concentrations','exposure','_tmp']:(target/name).mkdir(parents=True,exist_ok=True)
        prepared=target/'emissions/prepared_skims_for_grid_allocation.parquet'
        con.execute(f'''COPY (SELECT linkId,'{label}-height-' || CAST(source_release_height AS VARCHAR) AS vehicleTypeId,
            process,source_release_height,roadCategory,{','.join('"'+c+'"' for c in measures)}
            FROM read_parquet('{sqlpath(aggregate)}') WHERE assignment_group='{group}')
            TO '{sqlpath(prepared)}' (FORMAT PARQUET,COMPRESSION ZSTD)''')
        mapping.loc[mapping.assignment_group.eq(group)].to_csv(target/'inputs/vehicle_emfac_mixture.csv',index=False)
        for filename in ['settings.yaml','preprocess_manifest.yaml']:
            data=replace_paths(yaml.safe_load((TOTAL/filename).read_text()),target)
            if filename=='preprocess_manifest.yaml':
                assert not data['pipeline']['power_generation']['enabled']
                assert data['pipeline']['primary_pm25_integration_strategy']=='additive_inmap_aermod'
                assert not data['pipeline']['enable_passenger_inventory_activity_correction']
                assert not data['pipeline']['enable_freight_inventory_activity_correction']
                data.pop('traffic_expansion',None)
                data.setdefault('notes',[]).append(f'{label}: source assignment_group={group}; transit excluded; 10x expansion already applied to prepared emissions. See scenario.json.')
            (target/filename).write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
        scenario=json.loads((TOTAL/'scenario.json').read_text())
        scenario.update(product=label,assignment_group=group,transit_included=False,
            method='Partition original EMFAC cohort emissions by source use, expand 10x once, rerun independent ISRM/AERMOD operators',
            source_emissions_file=str(source))
        save(target/'scenario.json',scenario)
        row=totals.set_index('assignment_group').loc[group]
        save(target/'emissions_audit.json',dict(**{p:float(row[f'tons_per_year_{p}']) for p in POLS},
            units='short tons/year',fleet_expansion=FACTOR,assignment_group=group,power_generation=False))
        save(target/'run_status.json',dict(status='classified_emissions_ready',concentration_generated=False))
    con.close()
    print(totals.to_string(index=False),flush=True)


def filter_sources(label, *, additional_approved_sources=()):
    import geopandas as gpd
    import zarr
    target=OUT/label
    manifest_path=target/'pipeline_manifest.yaml'
    manifest=yaml.safe_load(manifest_path.read_text())
    original=Path(manifest['outputs']['beam_emissions_for_inmap'])
    filtered=target/'emissions/beam_emissions_for_inmap_partial.parquet'
    if filtered.exists():raise FileExistsError(filtered)
    data=gpd.read_parquet(original)
    matrix=zarr.open_group(ISRM_PATH,mode='r')
    keep=data.inmap_cell_id.isin(matrix['source_ids'][:])
    cols=[c for c in data if c.startswith('tons_per_year_')]
    missing=data.loc[~keep & data[cols].abs().sum(axis=1).gt(0),'inmap_cell_id'].unique()
    original_approved=set(pd.read_csv(TOTAL/'inputs/missing_isrm_sources.csv').inmap_cell_id)
    additional_approved=set(map(int,additional_approved_sources))
    approved=original_approved | additional_approved
    if not set(missing).issubset(approved):raise ValueError('New missing sources outside authorized 103')
    data.loc[~keep].to_parquet(target/'emissions/omitted_inmap_source_emissions.parquet',index=False)
    data.loc[keep].to_parquet(filtered,index=False)
    manifest['outputs']['beam_emissions_for_inmap_all_sources']=str(original)
    manifest['outputs']['beam_emissions_for_inmap']=str(filtered)
    manifest_path.write_text(yaml.safe_dump(manifest,sort_keys=False),encoding='utf-8')
    grouped=data.groupby('inmap_cell_id')[cols].sum()
    grouped.loc[grouped.index.isin(missing)].to_csv(target/'inputs/missing_isrm_sources.csv')
    report=dict(excluded_nonzero_source_count=len(missing),approved_missing_source_count=len(approved),
        baseline_approved_source_count=len(original_approved),
        additional_approved_source_ids=sorted(additional_approved),
        excluded_source_ids=sorted(map(int,missing)),excluded_totals=data.loc[~keep,cols].sum().to_dict(),
        missing_emission_fractions=(data.loc[~keep,cols].sum()/data[cols].sum()).to_dict(),
        aermod_sources_filtered=False,receptor_domain_filtered=False,missing_source_redistribution=False)
    save(target/'partial_isrm_audit.json',report)
    scenario=json.loads((target/'scenario.json').read_text())
    scenario['excluded_isrm_source_count']=len(missing);save(target/'scenario.json',scenario)
    print(label,'ISRM omitted sources:',len(missing),flush=True)


def export(label):
    import geopandas as gpd
    target=OUT/label
    emissions=gpd.read_parquet(target/'emissions/beam_emissions_for_aermod.parquet')
    cols=[c for c in emissions if c.startswith('tons_per_year_')]
    totals=emissions.groupby('aermod_cell_id',sort=True)[cols].sum()
    geometry=emissions.drop_duplicates('aermod_cell_id').set_index('aermod_cell_id').geometry
    product=gpd.GeoDataFrame(totals,geometry=geometry.reindex(totals.index),crs=emissions.crs).reset_index()
    product.rename(columns={c:c.replace('_aermod_allocated','') for c in cols},inplace=True)
    product['vehicle_group']=label
    product['emissions_units']='short tons/year'
    for c in cols:np.testing.assert_allclose(product[c.replace('_aermod_allocated','')].sum(),emissions[c].sum(),rtol=1e-12)
    output=OUT/'products'/f'{label}_emissions.parquet'
    if output.exists():raise FileExistsError(output)
    product.to_parquet(output,index=False)
    spatial_totals={p:float(product[f'tons_per_year_{p}'].sum()) for p in POLS}
    del product,emissions;gc.collect()
    conc=target/'exposure/beam_concentration_distribution.parquet'
    pf=pq.ParquetFile(conc)
    maxima={c:0. for c in ['TotalPM25','BC','NO2']}
    cols=[c for c in pf.schema_arrow.names if c not in ['geometry']]
    previous=pq.ParquetFile(TOTAL/'exposure/beam_concentration_distribution.parquet')
    # Validate every receptor and component, and nonnegative residual to all traffic.
    for batch,total_batch in zip(pf.iter_batches(columns=cols,batch_size=131072),previous.iter_batches(columns=cols,batch_size=131072),strict=True):
        d=batch.to_pandas();all_d=total_batch.to_pandas()
        np.testing.assert_array_equal(d.aermod_cell_id,all_d.aermod_cell_id)
        for c in ['TotalPM25','BC','NO2','PrimaryPM25','SecondaryPM25','inmap_only_TotalPM25']:
            assert np.isfinite(d[c]).all() and (d[c]>=-1e-10).all()
            assert (d[c]<=all_d[c]+1e-7).all(),f'{label}/{c} exceeds all-traffic total'
        for c in maxima:maxima[c]=max(maxima[c],float(d[c].max()))
        np.testing.assert_allclose(d.TotalPM25,d.PrimaryPM25+d.SecondaryPM25,atol=1e-10)
        np.testing.assert_allclose(d.PrimaryPM25,d.inmap_PrimaryPM25+d.aermod_PrimaryPM25,atol=1e-10)
        np.testing.assert_allclose(d.SecondaryPM25,d.inmap_SecondaryPM25,atol=1e-10)
        np.testing.assert_allclose(d.BC,d.inmap_BC+d.aermod_BC,atol=1e-10)
        np.testing.assert_allclose(d.NO2,d.inmap_NO2+d.aermod_NO2,atol=1e-10)
        np.testing.assert_allclose(d.inmap_only_TotalPM25,d.inmap_only_PrimaryPM25+d.SecondaryPM25,atol=1e-10)
        assert (d.inmap_only_PrimaryPM25+1e-10>=d.inmap_PrimaryPM25).all()
        np.testing.assert_array_equal(d.aermod_PrimaryPM25_scale_factor,1.)
    # Package identical numeric/geometry data with explicit product metadata.
    output=OUT/'products'/f'{label}_concentrations.parquet'
    if output.exists():raise FileExistsError(output)
    metadata=dict(pf.schema_arrow.metadata or {})
    metadata[b'impacts_scenario']=json.dumps(dict(vehicle_group=label,fleet_scale=FACTOR,power_generation=False,
        transit_included=False,units='ug/m3',partial_isrm=True)).encode()
    schema=pf.schema_arrow.with_metadata(metadata)
    with pq.ParquetWriter(output,schema,compression='zstd') as writer:
        for batch in pf.iter_batches(batch_size=131072):writer.write_batch(batch.replace_schema_metadata(metadata))
    pop=pd.read_parquet(TOTAL/'exposure/beam_population_counts.parquet',columns=['aermod_cell_id','person_count']).set_index('aermod_cell_id').person_count
    burdens={c:0. for c in maxima}
    for batch in pf.iter_batches(columns=['aermod_cell_id',*maxima]):
        d=batch.to_pandas();weights=d.aermod_cell_id.map(pop).fillna(0.)
        for c in maxima:burdens[c]+=float((d[c]*weights).sum())
    report=dict(status='complete',concentration_generated=True,grid_cells=pf.metadata.num_rows,
        vehicle_group=label,fleet_scale=FACTOR,power_generation=False,transit_included=False,
        omitted_inmap_source_count=json.loads((target/'partial_isrm_audit.json').read_text())['excluded_nonzero_source_count'],
        component_identities_passed=True,concentrations_bounded_by_alltraffic=True,
        maxima_ug_m3=maxima,population=float(pop.sum()),
        population_weighted_concentrations={c:v/float(pop.sum()) for c,v in burdens.items()},
        grid_allocated_emissions_short_tons_year=spatial_totals,
        emissions_product=str(OUT/'products'/f'{label}_emissions.parquet'),concentration_product=str(output))
    save(target/'run_status.json',report)
    print(json.dumps(report,indent=2),flush=True)


def validate_combined():
    audit_split()
    columns=['TotalPM25','PrimaryPM25','SecondaryPM25','BC','NO2','inmap_only_TotalPM25','inmap_only_BC','inmap_only_NO2']
    readers=[pq.ParquetFile(OUT/'products'/f'{label}_concentrations.parquet') for label in GROUPS]
    readers.append(pq.ParquetFile(TOTAL/'exposure/beam_concentration_distribution.parquet'))
    minimum={c:np.inf for c in columns}
    for batches in zip(*(r.iter_batches(columns=['aermod_cell_id','geometry',*columns],batch_size=131072) for r in readers),strict=True):
        a,b,t=batches
        for key in ['aermod_cell_id','geometry']:
            assert a.column(a.schema.get_field_index(key)).equals(t.column(t.schema.get_field_index(key)))
            assert b.column(b.schema.get_field_index(key)).equals(t.column(t.schema.get_field_index(key)))
        for c in columns:
            i=a.schema.get_field_index(c)
            residual=t.column(i).to_numpy()-a.column(i).to_numpy()-b.column(i).to_numpy()
            minimum[c]=min(minimum[c],float(residual.min()))
            assert (residual>=-1e-7).all(),f'Negative transit residual for {c}'
    reports={label:json.loads((OUT/label/'run_status.json').read_text()) for label in GROUPS}
    for label,report in reports.items():
        assert report['status']=='complete'
        report['omitted_inmap_source_count']=json.loads((OUT/label/'partial_isrm_audit.json').read_text())['excluded_nonzero_source_count']
        table=pq.read_table(OUT/'products'/f'{label}_emissions.parquet')
        assert b'geo' in table.schema.metadata
        assert table['geometry'].null_count==0
        ids=table['aermod_cell_id'].to_pandas()
        assert ids.notna().all() and ids.is_unique
        for p in POLS:
            values=table[f'tons_per_year_{p}'].to_numpy()
            assert np.isfinite(values).all() and (values>=0).all()
            np.testing.assert_allclose(values.sum(),report['grid_allocated_emissions_short_tons_year'][p],rtol=1e-12)
        report['emission_source_grid_cells']=len(ids)
        save(OUT/label/'run_status.json',report)
    save(OUT/'results_summary.json',dict(status='complete',all_four_products_validated=True,
        products=reports,combined_not_exceeding_all_traffic=True,
        residual_interpretation='Excluded transit contribution plus numerical roundoff; not a fifth delivered product',
        residual_minima=minimum,geometry_and_receptor_ids_identical=True))
    print('ALL FOUR PRODUCTS VALIDATED',flush=True)


def audit_split():
    """Check conservation by road/process/height, not only domain totals."""
    con=duckdb.connect();con.execute('SET threads=4')
    keys=['linkId','process','source_release_height','roadCategory']
    fields=[f'tons_per_year_{p}' for p in POLS]
    new=OUT/'audit/classified_link_process_emissions.parquet'
    old=TOTAL/'emissions/prepared_skims_for_grid_allocation.parquet'
    grouped=','.join(f'SUM("{c}") AS "{c}"' for c in fields)
    join=' AND '.join(f'a."{k}" IS NOT DISTINCT FROM b."{k}"' for k in keys)
    errors=','.join(f'MAX(ABS(a."{c}"-b."{c}")) AS "{c}"' for c in fields)
    invalid=' OR '.join(f'ABS(a."{c}"-b."{c}")>1e-10+1e-9*ABS(b."{c}")' for c in fields)
    report=con.execute(f'''WITH a AS (SELECT {','.join(keys)},{grouped} FROM read_parquet('{sqlpath(new)}') GROUP BY ALL),
        b AS (SELECT {','.join(keys)},{grouped} FROM read_parquet('{sqlpath(old)}') GROUP BY ALL)
        SELECT COUNT(*) AS source_process_height_rows,
        COUNT(*) FILTER (WHERE a.linkId IS NULL OR b.linkId IS NULL OR {invalid}) AS mismatched_rows,
        {errors} FROM a FULL JOIN b ON {join}''').fetchdf().to_dict('records')[0]
    assert report['mismatched_rows']==0,report
    save(OUT/'audit/source_process_height_conservation.json',report)
    con.close();print(report,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True,choices=['setup','run','allocate','filter','inmap','aermod','exposure','export','validate','audit_split'])
    parser.add_argument('--group',choices=list(GROUPS))
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.stage=='setup':return setup()
    if args.stage=='validate':return validate_combined()
    if args.stage=='audit_split':return audit_split()
    if not args.group:parser.error('--group required')
    if args.stage=='run':
        for stage in ['allocate','filter','inmap','aermod','exposure','export']:
            subprocess.run([sys.executable,'-X','utf8',__file__,'--stage',stage,'--group',args.group],check=True)
    elif args.stage=='filter':filter_sources(args.group)
    elif args.stage=='export':export(args.group)
    else:run_stage(OUT/args.group,'emissions' if args.stage=='allocate' else args.stage)


if __name__=='__main__':main()
