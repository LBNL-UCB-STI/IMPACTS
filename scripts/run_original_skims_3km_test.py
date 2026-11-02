"""Controlled traffic-only dispersion comparison; never rewrite reference outputs."""
from research_paths import ISRM_PATH, BREATHE_ROOT, BEAM_DATA_ROOT, CHARGING_REFERENCE
import argparse
import functools
import json
import logging
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
import zarr

from audit_original_skims_3km_test import BASE, OUT
import run_events_traffic_pipeline as traffic

MATRIX = ISRM_PATH
BRANCHES = ('old_4500m_square', 'new_3000m_circle')
NAMES = dict(zip(BRANCHES, ['ISRM 4.5 km; AERMOD full template', 'ISRM 3 km; AERMOD 3 km circle']))
COLS = ['TotalPM25','PrimaryPM25','SecondaryPM25','BC','NO2',
        'inmap_PrimaryPM25','inmap_SecondaryPM25','aermod_PrimaryPM25',
        'inmap_BC','inmap_NO2','aermod_BC','aermod_NO2',
        'inmap_only_PrimaryPM25','inmap_only_TotalPM25','inmap_only_BC','inmap_only_NO2']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def setup():
    shared = OUT/'inputs'
    shared.mkdir(exist_ok=True)
    destination = shared/'annual_inmap_emissions_available_sources.parquet'
    if destination.exists():
        raise FileExistsError(destination)
    source = BASE/'emissions/beam_emissions_for_inmap.parquet'
    columns = [c for c in pq.read_schema(source).names if c.startswith('tons_per_year_')]
    with duckdb.connect() as con:
        data = con.execute('SELECT inmap_cell_id,'+','.join(f'sum({c}) AS {c}' for c in columns)+
                           ' FROM read_parquet(?) GROUP BY 1 ORDER BY 1', [str(source)]).fetchdf()
    ids = zarr.open_group(MATRIX, mode='r')['source_ids'][:]
    keep = data.inmap_cell_id.isin(ids)
    missing = set(data.loc[~keep & data[columns].abs().sum(axis=1).gt(0), 'inmap_cell_id'])
    approved = set(pd.read_csv(OUT/'missing_isrm_sources.csv').inmap_cell_id)
    assert missing == approved and len(missing) == 114
    data.loc[keep].to_parquet(destination, index=False)
    data.loc[~keep].to_parquet(shared/'omitted_annual_inmap_emissions.parquet', index=False)
    original = yaml.safe_load((BASE/'preprocess_manifest.yaml').read_text())
    reference = yaml.safe_load((BASE/'pipeline_manifest.yaml').read_text())
    for branch in BRANCHES:
        target = OUT/branch
        target.mkdir(exist_ok=True)
        manifest = yaml.safe_load(yaml.safe_dump(original))
        manifest['pipeline']['isrm_url'] = MATRIX
        manifest['pipeline']['power_generation'] = {'enabled': False}
        (target/'preprocess_manifest.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False),encoding='utf-8')
        outputs = {k:v for k,v in reference['outputs'].items() if k in (
            'skims_emissions','beam_emissions_by_county_process','beam_emissions_for_aermod','beam_inmap_study_area_grid')}
        outputs['beam_emissions_for_inmap'] = str(destination)
        run_manifest = dict(reference, outputs=outputs, output_dir=str(target),
            pipeline=manifest['pipeline'], preprocess_manifest_path=str(target/'preprocess_manifest.yaml'),
            pipeline_manifest_path=str(target/'pipeline_manifest.yaml'))
        (target/'pipeline_manifest.yaml').write_text(yaml.safe_dump(run_manifest,sort_keys=False),encoding='utf-8')
    save(OUT/'test_configuration.json', dict(status='running', reference=str(BASE),
        authorization='User continued test after additional 11 missing-source exclusion request, 2026-10-15',
        excluded_nonzero_isrm_sources=sorted(map(int,missing)), aermod_sources_excluded=False,
        power_generation=False, annual_emissions_reused=True, additional_scaling=1,
        source_aggregation='Sum original annual allocated emissions by source cell; no emission factors changed',
        original_population_sample=original['pipeline']['population_sample'],
        original_annualization_days=original['pipeline']['annualization_days'],
        all_source_totals=data[columns].sum().to_dict(), included_source_totals=data.loc[keep,columns].sum().to_dict(),
        branches={BRANCHES[0]:dict(isrm_radius_m=4500,aermod_cutoff_m=None),
                  BRANCHES[1]:dict(isrm_radius_m=3000,aermod_cutoff_m=3000)},
        isrm_overlap_rule='intersection area / receptor area > 0.5'))


def disperse(branch, stage):
    import geopandas as gpd
    from impacts.pipeline.workflow import step2_compute_inmap_concentrations as inmap
    from impacts.pipeline.workflow import step3_compute_aermod_concentrations as aermod
    from impacts.pipeline.workflow import step4_prepare_exposure as exposure
    from impacts.manifest.schema import PipelineConfig
    from gateway_aermod_crop import apply_cropped
    target = OUT/branch
    manifest_path = target/'pipeline_manifest.yaml'
    run_manifest = yaml.safe_load(manifest_path.read_text())
    if 'contract_version' not in run_manifest:
        reference = yaml.safe_load((BASE/'pipeline_manifest.yaml').read_text())
        run_manifest = dict(reference, outputs=run_manifest['outputs'], output_dir=str(target),
            pipeline=yaml.safe_load((target/'preprocess_manifest.yaml').read_text())['pipeline'],
            preprocess_manifest_path=str(target/'preprocess_manifest.yaml'),pipeline_manifest_path=str(manifest_path))
        manifest_path.write_text(yaml.safe_dump(run_manifest,sort_keys=False),encoding='utf-8')
    if branch == BRANCHES[0]:
        inmap.build_near_source_mask = functools.partial(inmap.build_near_source_mask,radius_meters=4500.)
        aermod._AERMOD_RESPONSE_RADIUS_METERS = 1e9
    if stage == 'merge':
        pipeline = PipelineConfig.from_dict(yaml.safe_load((target/'preprocess_manifest.yaml').read_text())['pipeline'])
        data = exposure._build_full_exposure_grid(pipeline=pipeline,
            prepared_inmap=exposure._prepare_inmap_exposure_inputs(gpd.read_parquet(target/'concentrations/beam_inmap_concentrations.parquet')),
            prepared_aermod=exposure._prepare_aermod_exposure_inputs(gpd.read_parquet(target/'concentrations/beam_aermod_concentrations.parquet')),
            working_dir=target/'exposure')
        path = target/'exposure/beam_concentration_distribution.parquet'
        data.to_parquet(path,index=False)
        manifest_path=target/'pipeline_manifest.yaml'
        manifest=yaml.safe_load(manifest_path.read_text())
        manifest['outputs']['beam_concentration_distribution']=str(path)
        manifest['outputs']['beam_population_counts']=str(BASE/'exposure/beam_population_counts.parquet')
        manifest_path.write_text(yaml.safe_dump(manifest,sort_keys=False),encoding='utf-8')
        return
    if stage == 'aermod':
        aermod._apply_kernels = apply_cropped
        original_builder = aermod._build_kernel_for_pattern
        kernel_audit = []
        def audited_builder(*args, **kwargs):
            kernel = original_builder(*args, **kwargs)
            # The test's receptor spacing is verified as 100 m by the workflow.
            distance = np.hypot(kernel['dix'],kernel['diy'])*100.
            outside = distance > 3000.
            nonzero = kernel['response_per_ton'] != 0
            if branch == BRANCHES[1]:
                assert not np.any(outside & nonzero)
            kernel_audit.append(dict(points=len(distance),nonzero_outside_3000m=int((outside & nonzero).sum()),
                max_nonzero_distance_m=float(distance[nonzero].max()) if nonzero.any() else 0.))
            return kernel
        aermod._build_kernel_for_pattern = audited_builder
        def parquet_only(gdf, output_path):
            output_path.parent.mkdir(parents=True,exist_ok=True)
            gdf.to_parquet(output_path,index=False)
        aermod._write_outputs = parquet_only
    traffic.run_stage(target,stage)
    if stage == 'aermod':
        save(target/'kernel_distance_audit.json',dict(kernels_built=len(kernel_audit),
            radius_m=3000 if branch==BRANCHES[1] else None,kernels=kernel_audit))


def validate():
    pop = pd.read_parquet(BASE/'exposure/beam_population_counts.parquet',columns=['aermod_cell_id','person_count']).set_index('aermod_cell_id').person_count
    assert pop.index.is_unique
    files = [pq.ParquetFile(OUT/b/'exposure/beam_concentration_distribution.parquet') for b in BRANCHES]
    assert files[0].metadata.num_rows == files[1].metadata.num_rows
    for f in files:
        assert not any(c.startswith('power_') for c in f.schema_arrow.names)
    burdens = [{c:0. for c in COLS} for _ in files]
    maxima = [{c:0. for c in COLS} for _ in files]
    same = ['SecondaryPM25','inmap_SecondaryPM25',*[c for c in COLS if c.startswith('inmap_only_')]]
    change = {p:dict(increased=0,decreased=0,unchanged=0) for p in ['TotalPM25','BC','NO2']}
    represented_population=0.
    for batches in zip(*(f.iter_batches(columns=['aermod_cell_id',*COLS],batch_size=131072) for f in files),strict=True):
        old,new = [b.to_pandas() for b in batches]
        np.testing.assert_array_equal(old.aermod_cell_id,new.aermod_cell_id)
        weights = old.aermod_cell_id.map(pop).fillna(0.)
        represented_population += float(weights.sum())
        for i,d in enumerate((old,new)):
            assert np.isfinite(d[COLS].to_numpy()).all() and d[COLS].min().min() >= -1e-10
            for actual,left,right in [('TotalPM25','PrimaryPM25','SecondaryPM25'),
                ('PrimaryPM25','inmap_PrimaryPM25','aermod_PrimaryPM25'),('BC','inmap_BC','aermod_BC'),
                ('NO2','inmap_NO2','aermod_NO2'),('inmap_only_TotalPM25','inmap_only_PrimaryPM25','SecondaryPM25')]:
                np.testing.assert_allclose(d[actual],d[left]+d[right],atol=1e-10)
            for c in COLS:
                burdens[i][c] += float((d[c]*weights).sum())
                maxima[i][c] = max(maxima[i][c],float(d[c].max()))
        np.testing.assert_allclose(old[same],new[same],atol=1e-10,rtol=1e-10)
        for c in ['PrimaryPM25','BC','NO2']:
            assert (new['inmap_'+c]+1e-10 >= old['inmap_'+c]).all()
            assert (new['aermod_'+c] <= old['aermod_'+c]+1e-10).all()
        for p in change:
            delta = new[p]-old[p]
            change[p]['increased'] += int((delta>1e-10).sum())
            change[p]['decreased'] += int((delta< -1e-10).sum())
            change[p]['unchanged'] += int((delta.abs()<=1e-10).sum())
    np.testing.assert_allclose(represented_population,pop.sum(),rtol=1e-10)
    rows=[]
    for c in COLS:
        a,b = [d[c]/float(pop.sum()) for d in burdens]
        rows.append(dict(component=c,old_PWC_ug_m3=a,new_PWC_ug_m3=b,change_percent=(b/a-1)*100 if a else None,
                         old_max_ug_m3=maxima[0][c],new_max_ug_m3=maxima[1][c]))
    pd.DataFrame(rows).to_csv(OUT/'concentration_comparison.csv',index=False)
    save(OUT/'validation.json',dict(status='passed',grid_cells=files[0].metadata.num_rows,population=float(pop.sum()),
        same_emissions=True,secondary_and_inmap_only_unchanged=True,new_aermod_never_increases=True,
        new_nonlocal_inmap_never_decreases=True,component_identities=True,cell_change_counts=change,comparison=rows))
    print(json.dumps(rows,indent=2),flush=True)


def plot():
    import plot_gateway_products as plots
    paths={b:OUT/b/'exposure/beam_concentration_distribution.parquet' for b in BRANCHES}
    plots.NOTE='Original skim; identical annual emissions; traffic only; partial ISRM coverage'
    mask,ids,layout,net,counties,context=plots.load_context(OUT,paths[BRANCHES[0]],
        manifest_path=OUT/BRANCHES[0]/'preprocess_manifest.yaml')
    summary={}
    for b in BRANCHES:
        data=plots.load_values(paths[b],mask,ids)
        summary[b]=dict(name=NAMES[b],limits={p:plots.positive_limit(data,[p]) for p in plots.POLLUTANTS})
    limits=plots.comparison_figures(BRANCHES,summary,layout,net,counties,
        load=lambda b:plots.load_values(paths[b],mask,ids),destination=OUT/'figures',dpi=150,
        comparison_title='Original-skims controlled test')
    comparison=pd.read_csv(OUT/'concentration_comparison.csv').set_index('component')
    rows=[dict(group=b,gateway_name=NAMES[b],**{p+'_PWC_ug_m3':float(comparison.loc[p,prefix+'_PWC_ug_m3'])
        for p in plots.POLLUTANTS}) for b,prefix in zip(BRANCHES,['old','new'])]
    plots.pwc_chart(rows,'Original-skims controlled test',OUT/'figures/population_weighted_comparison.png',150)
    save(OUT/'figure_manifest.json',dict(context=context,common_color_limits=limits,figures=sorted(str(p) for p in (OUT/'figures').glob('*.png'))))


def job(stage,branch=None):
    key=f'{branch}_{stage}' if branch else stage
    marker=OUT/f'completed_{key}.json'
    if marker.exists():
        return
    (OUT/'logs').mkdir(exist_ok=True)
    print('START',key,flush=True)
    command=[sys.executable,'-X','utf8',__file__,'--stage',stage]
    if branch:command+=['--branch',branch]
    with (OUT/'logs'/f'{key}.log').open('w',encoding='utf-8') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
    save(marker,dict(status='complete'))
    print('DONE',key,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','setup','inmap','aermod','merge','validate','plot'],default='all')
    parser.add_argument('--branch',choices=BRANCHES)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.stage=='all':
        job('setup')
        for branch in BRANCHES:
            for stage in ['inmap','aermod','merge']:job(stage,branch)
        job('validate');job('plot')
        config=json.loads((OUT/'test_configuration.json').read_text())
        config['status']='complete'
        save(OUT/'test_configuration.json',config)
        audit=json.loads((OUT/'input_coverage_audit.json').read_text())
        audit.setdefault('initial_status',audit['status'])
        audit.update(status='additional_exclusion_authorized_test_complete',concentration_generated=True,
            authorization=config['authorization'])
        save(OUT/'input_coverage_audit.json',audit)
        save(OUT/'run_status.json',dict(status='complete',power_generation=False,comparison=str(OUT/'concentration_comparison.csv')))
    elif args.stage=='setup':setup()
    elif args.stage=='validate':validate()
    elif args.stage=='plot':plot()
    else:
        if not args.branch:parser.error('--branch required')
        disperse(args.branch,args.stage)
