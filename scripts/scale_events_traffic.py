"""Expand a verified 10% traffic scenario using the fixed linear model operators.

No congestion/speed or chemistry feedback is simulated. Population, geometry,
source-support flags and dimensionless integration factors are not expanded.
"""
from pathlib import Path
import argparse
import gc
import json
import shutil
import subprocess
import sys
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pandas as pd
import yaml
from run_events_traffic_pipeline import ROOT, POLS, save

SOURCE=ROOT/'examples/pilates/impacts/impacts_output/events-fc07-traffic-only-20260925-baseline-mixture-v1-partial-isrm103'
OUT=SOURCE.with_name(SOURCE.name+'-fulltraffic10x')
FACTOR=10.
SPECIES={'SOA','pNO3','pNH4','pSO4','PrimaryPM25','SecondaryPM25','TotalPM25','BC','NO2'}


def scale_column(name,kind):
    if kind=='emissions':
        return name.startswith('tons_per_year_') or name in {'totTrips','totVMT'} or name.startswith(('totTrips_','totVMT_'))
    if kind!='concentration':raise ValueError(kind)
    if name in SPECIES or name=='aermod_PrimaryPM25_scaled':return True
    return any(name==prefix+species for prefix in ['inmap_','inmap_only_','InMAPOnly_','aermod_','power_'] for species in SPECIES)


def scale_parquet(source,target,kind,factor=FACTOR):
    if target.exists():raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    pf=pq.ParquetFile(source)
    names=[c for c in pf.schema_arrow.names if scale_column(c,kind)]
    if not names:raise ValueError(f'No scalable columns: {source}')
    metadata=dict(pf.schema_arrow.metadata or {})
    metadata[b'impacts_traffic_expansion']=json.dumps(dict(factor=factor,source_file=str(source),columns=names,
        method='fixed-operator linear expansion',population_scaled=False,power_generation=False)).encode()
    schema=pf.schema_arrow.with_metadata(metadata)
    partial=target.with_suffix('.partial.parquet')
    with pq.ParquetWriter(partial,schema,compression='zstd') as writer:
        for batch in pf.iter_batches(batch_size=131072):
            arrays=[]
            for field,values in zip(schema,batch.columns):
                arrays.append(pc.multiply(values,pa.scalar(factor,type=values.type)) if field.name in names else values)
            writer.write_batch(pa.RecordBatch.from_arrays(arrays,schema=schema))
    partial.replace(target)
    # Full column-by-column check, including unchanged geometry and identifiers.
    new=pq.ParquetFile(target)
    assert pf.metadata.num_rows==new.metadata.num_rows
    for old_batch,new_batch in zip(pf.iter_batches(batch_size=131072),new.iter_batches(batch_size=131072),strict=True):
        for name,a,b in zip(pf.schema_arrow.names,old_batch.columns,new_batch.columns):
            if name in names:
                np.testing.assert_allclose(b.to_numpy(zero_copy_only=False),a.to_numpy(zero_copy_only=False)*factor,rtol=1e-12,atol=1e-12,equal_nan=True)
            else:
                assert a.equals(b),f'Non-scaled field changed: {name}'
    return dict(file=str(target),rows=new.metadata.num_rows,scaled_columns=names,all_columns_checked=True)


def remap(value):
    if isinstance(value,str):return value.replace(str(SOURCE),str(OUT)).replace(SOURCE.as_posix(),OUT.as_posix())
    if isinstance(value,list):return [remap(v) for v in value]
    if isinstance(value,dict):return {k:remap(v) for k,v in value.items()}
    return value


def scale():
    if OUT.exists():raise FileExistsError(OUT)
    for folder in ['emissions','concentrations','exposure','inputs','_tmp']:(OUT/folder).mkdir(parents=True,exist_ok=True)
    original=json.loads((SOURCE/'run_status.json').read_text())
    assert original['final_concentration_generated'] and original['component_identities_passed']
    scenario=json.loads((SOURCE/'scenario.json').read_text())
    assert scenario['fleet_scale']==1 and scenario['power_generation'] is False
    scenario.update(fleet_scale=FACTOR,traffic_sample_fraction=0.1,
        user_confirmed_total_traffic_sample_fraction=0.1,population_weights_scaled=False,
        scaling_source_directory=str(SOURCE),method='Exact uniform 10x expansion under existing fixed linear model operators',
        limitations=[s for s in scenario['limitations'] if 'Not a full-SFBAY fleet expansion' not in s]+[
            'Uniform expansion applies to all modeled traffic classes and processes; no new speed, congestion, fleet composition or chemistry feedback.',
            'The 103 missing ISRM sources remain excluded; this is full-traffic scaling, not complete ISRM coverage.'])
    save(OUT/'scenario.json',scenario)
    audit=[]
    for source in sorted((SOURCE/'emissions').glob('*.parquet')):
        if source.name.startswith('_'):continue
        print('Scaling emissions',source.name,flush=True)
        audit.append(scale_parquet(source,OUT/'emissions'/source.name,'emissions'))
    for relative in ['concentrations/beam_inmap_concentrations.parquet','concentrations/beam_aermod_concentrations.parquet','exposure/beam_concentration_distribution.parquet']:
        print('Scaling concentration',relative,flush=True)
        audit.append(scale_parquet(SOURCE/relative,OUT/relative,'concentration'))
    for relative in ['exposure/beam_population_counts.parquet','exposure/beam_population_counts.gpkg',
                     'exposure/beam_population_distribution.parquet','emissions/beam_inmap_study_area_grid.gpkg',
                     'inputs/vehicle_emfac_mixture.csv','network_validation.json','activity_audit.json']:
        shutil.copy2(SOURCE/relative,OUT/relative)
    for name in ['missing_isrm_sources.csv','isrm_source_coverage.csv']:
        frame=pd.read_csv(SOURCE/'inputs'/name);frame[POLS]*=FACTOR;frame.to_csv(OUT/'inputs'/name,index=False)
    report=json.loads((SOURCE/'emissions_audit.json').read_text())
    for p in POLS:report[p]*=FACTOR
    report['fleet_expansion']=FACTOR
    save(OUT/'emissions_audit.json',report)
    report=remap(json.loads((SOURCE/'partial_isrm_audit.json').read_text()))
    report['excluded_totals']={k:v*FACTOR for k,v in report['excluded_totals'].items()}
    save(OUT/'partial_isrm_audit.json',report)
    report=json.loads((SOURCE/'isrm_coverage_audit.json').read_text())
    report['missing_emissions_short_tons_year']={k:v*FACTOR for k,v in report['missing_emissions_short_tons_year'].items()}
    save(OUT/'isrm_coverage_audit.json',report)
    for name in ['settings.yaml','preprocess_manifest.yaml','pipeline_manifest.yaml']:
        data=remap(yaml.safe_load((SOURCE/name).read_text()))
        if name=='settings.yaml':
            data['impacts']['population'].update(population_sample=.1,freight_sample=.1,transit_sample=.1)
        else:
            data['pipeline'].update(population_sample=.1,freight_sample=.1,transit_sample=.1)
            data['traffic_expansion']=dict(factor=FACTOR,sample_fraction=.1,already_applied_to_prepared_emissions=True,
                linear_model_expansion=True,source_directory=str(SOURCE),population_weights_scaled=False)
        if name=='pipeline_manifest.yaml':
            data['command']='python scripts/scale_events_traffic.py --stage scale'
            data['stage_timings']={}
        (OUT/name).write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    save(OUT/'scaling_validation.json',dict(factor=FACTOR,sample_fraction=.1,source_directory=str(SOURCE),
        files=audit,population_weights_unchanged=True,geometry_ids_flags_factors_unchanged=True,
        remaining_isrm_source_emissions_renormalized=False))
    save(OUT/'run_status.json',dict(status='scaled_parquet_complete_gis_and_postprocess_pending',traffic_expansion_factor=FACTOR))
    print('SCALING COMPLETE',OUT,flush=True)


def gis():
    import geopandas as gpd
    for relative in ['concentrations/beam_inmap_concentrations.parquet','concentrations/beam_aermod_concentrations.parquet','exposure/beam_concentration_distribution.parquet']:
        target=(OUT/relative).with_suffix('.gpkg')
        if target.exists():raise FileExistsError(target)
        print('Writing GIS',target.name,flush=True)
        frame=gpd.read_parquet(OUT/relative);frame.to_file(target,driver='GPKG');del frame;gc.collect()


def validate_exposure():
    folder='postprocess/human_exposure'
    checked={}
    for name in ['human_exposure_grid.parquet','human_exposure_ces_tracts.parquet','human_exposure_schools.parquet','human_exposure_ltc.parquet']:
        a=pd.read_parquet(SOURCE/folder/name);b=pd.read_parquet(OUT/folder/name)
        assert a.columns.tolist()==b.columns.tolist() and len(a)==len(b)
        scaled=[]
        for col in a:
            if (col.startswith(('Final_','InMAP_','AERMOD_','Hybrid_','Traffic_','Power_')) or '_Exposure_' in col) and pd.api.types.is_numeric_dtype(a[col]):
                np.testing.assert_allclose(b[col],a[col]*FACTOR,rtol=1e-10,atol=1e-10,equal_nan=True);scaled.append(col)
            else:pd.testing.assert_series_equal(a[col],b[col])
        checked[name]=scaled
    save(OUT/'exposure_scaling_validation.json',dict(factor=FACTOR,scaled_fields=checked,all_other_fields_including_population_unchanged=True))
    print('EXPOSURE SCALE VALIDATED',flush=True)


def validate_gis():
    import sqlite3
    checked=[]
    for relative in ['concentrations/beam_inmap_concentrations.parquet','concentrations/beam_aermod_concentrations.parquet','exposure/beam_concentration_distribution.parquet']:
        source=OUT/relative;target=source.with_suffix('.gpkg')
        pf=pq.ParquetFile(source)
        with sqlite3.connect(target.as_uri()+'?mode=ro',uri=True) as con:
            table=con.execute("SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchone()[0]
            quoted='"'+table.replace('"','""')+'"'
            values=con.execute(f'SELECT COUNT(*), MAX(TotalPM25), MAX(BC), MAX(NO2) FROM {quoted}').fetchone()
            assert values[0]==pf.metadata.num_rows
            expected=pq.read_table(source,columns=['TotalPM25','BC','NO2'])
            for name,value in zip(['TotalPM25','BC','NO2'],values[1:]):
                np.testing.assert_allclose(value,pc.max(expected[name]).as_py(),rtol=1e-12)
            checked.append(dict(file=str(target),rows=values[0],maxima_match_parquet=True))
    save(OUT/'gis_validation.json',dict(files=checked))
    print('GIS VALIDATED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['scale','gis','verify','postprocess','validate_products','validate_exposure','validate_gis'],required=True)
    args=parser.parse_args()
    if args.stage in ['scale','gis','validate_exposure','validate_gis']:globals()[args.stage]()
    else:subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/continue_events_partial_isrm.py'),'--output',str(OUT),'--stage',args.stage],check=True)
