"""User-authorized partial ISRM continuation; preserve complete AERMOD sources."""
from research_paths import ISRM_PATH, BREATHE_ROOT, BEAM_DATA_ROOT, CHARGING_REFERENCE
from pathlib import Path
import argparse
import json
import logging
import shutil
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from run_events_traffic_pipeline import ROOT, BASE, POLS, run_stage, save, sqlpath

SOURCE = ROOT/'examples/pilates/impacts/impacts_output/events-fc07-traffic-only-20260925-baseline-mixture-v1'
OUT = SOURCE.with_name(SOURCE.name+'-partial-isrm103')


def setup():
    if (OUT/'preprocess_manifest.yaml').exists():
        raise FileExistsError('Scenario already initialized')
    for name in ['inputs','emissions','concentrations','exposure','_tmp']:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    for name in ['scenario.json','network_validation.json','activity_audit.json','emissions_audit.json','isrm_coverage_audit.json']:
        shutil.copy2(SOURCE/name,OUT/name)
    for name in ['missing_isrm_sources.csv','isrm_source_coverage.csv','vehicle_emfac_mixture.csv']:
        shutil.copy2(SOURCE/'inputs'/name,OUT/'inputs'/name)
    settings=yaml.safe_load((SOURCE/'settings.yaml').read_text())
    settings['impacts']['local_output_folder']=str(OUT)
    (OUT/'settings.yaml').write_text(yaml.safe_dump(settings,sort_keys=False),encoding='utf-8')
    manifest=yaml.safe_load((SOURCE/'preprocess_manifest.yaml').read_text())
    manifest.update(input_dir=str(OUT/'inputs'),staging_dir=str(OUT/'inputs'),settings_source=str(OUT/'settings.yaml'),preprocess_manifest_path=str(OUT/'preprocess_manifest.yaml'))
    manifest['inputs']['settings'].update(source_path=str(OUT/'settings.yaml'),staged_path=str(OUT/'settings.yaml'))
    manifest['pilates_contract'].update(local_input_dir=str(OUT/'inputs'),local_output_dir=str(OUT))
    if manifest['pipeline']['enable_passenger_inventory_activity_correction'] or manifest['pipeline']['enable_freight_inventory_activity_correction']:
        raise ValueError('Cohort collapse requires inventory corrections disabled')
    # Factors and annual operating days have already been applied per cohort.
    # Dispersion depends on location, process, height and road, not cohort labels.
    con=duckdb.connect();con.execute('SET threads=4');con.execute("SET memory_limit='24GB'")
    columns=['totTrips','totVMT']+[f'tons_per_year_{p}' for p in POLS]
    sums=','.join(f'SUM("{c}") AS "{c}"' for c in columns)
    source=SOURCE/'emissions/prepared_skims_for_grid_allocation.parquet'
    target=OUT/'emissions/prepared_skims_for_grid_allocation.parquet'
    con.execute(f'''COPY (SELECT linkId, 'aggregated-height-' || CAST(source_release_height AS VARCHAR) AS vehicleTypeId,
        process,source_release_height,roadCategory,{sums} FROM read_parquet('{sqlpath(source)}')
        GROUP BY ALL) TO '{sqlpath(target)}' (FORMAT PARQUET,COMPRESSION ZSTD)''')
    before=con.execute(f'SELECT {sums} FROM read_parquet(?)',[str(source)]).fetchnumpy()
    after=con.execute(f'SELECT {sums} FROM read_parquet(?)',[str(target)]).fetchnumpy()
    for col in columns:np.testing.assert_allclose(before[col],after[col],rtol=1e-10,atol=1e-10)
    save(OUT/'cohort_aggregation_audit.json',dict(source_rows=pq.ParquetFile(source).metadata.num_rows,
         prepared_rows=pq.ParquetFile(target).metadata.num_rows,all_totals_preserved=True,original_cohort_file=str(source)))
    con.close()
    (OUT/'preprocess_manifest.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False),encoding='utf-8')
    scenario=json.loads((OUT/'scenario.json').read_text())
    scenario.update(partial_isrm=True,excluded_isrm_source_count=103,
        exclusion_policy='User-authorized omission of 103 unavailable source response rows from ALL InMAP species, including InMAP-only; AERMOD emissions and receptor domain are unchanged.',
        concentration_warning='PARTIAL ISRM SOURCE COVERAGE; not a complete regional concentration estimate')
    save(OUT/'scenario.json',scenario)
    save(OUT/'run_status.json',dict(status='running_partial_isrm',final_concentration_generated=False))
    print('SETUP COMPLETE',OUT,flush=True)


def filter_sources():
    import geopandas as gpd
    import zarr
    manifest_path=OUT/'pipeline_manifest.yaml'
    manifest=yaml.safe_load(manifest_path.read_text())
    original=Path(manifest['outputs']['beam_emissions_for_inmap'])
    target=OUT/'emissions/beam_emissions_for_inmap_partial103.parquet'
    if original==target:raise ValueError('Already filtered')
    data=gpd.read_parquet(original)
    sr=zarr.open_group(ISRM_PATH,mode='r')
    keep=data.inmap_cell_id.isin(sr['source_ids'][:])
    expected=set(pd.read_csv(OUT/'inputs/missing_isrm_sources.csv').inmap_cell_id)
    emission_cols=[c for c in data if c.startswith('tons_per_year_')]
    actual=set(data.loc[~keep & data[emission_cols].abs().sum(axis=1).gt(0),'inmap_cell_id'])
    if actual!=expected:raise ValueError(f'Exclusion mismatch: {actual^expected}')
    data.loc[~keep].to_parquet(OUT/'emissions/omitted_inmap_source_emissions.parquet',index=False)
    data.loc[keep].to_parquet(target,index=False)
    manifest['outputs']['beam_emissions_for_inmap_all_sources']=str(original)
    manifest['outputs']['beam_emissions_for_inmap']=str(target)
    manifest_path.write_text(yaml.safe_dump(manifest,sort_keys=False),encoding='utf-8')
    save(OUT/'partial_isrm_audit.json',dict(excluded_nonzero_source_count=len(actual),
         retained_source_count=int(data.loc[keep,'inmap_cell_id'].nunique()),
         excluded_totals=data.loc[~keep,emission_cols].sum().to_dict(),aermod_sources_filtered=False,
         receptor_domain_filtered=False,original_emissions_file=str(original),partial_emissions_file=str(target)))
    print('Filtered only InMAP source input:',len(actual),'missing sources',flush=True)


def verify():
    fleet_scale=json.loads((OUT/'scenario.json').read_text()).get('fleet_scale',1)
    path=OUT/'exposure/beam_concentration_distribution.parquet'
    pf=pq.ParquetFile(path)
    fields=['PrimaryPM25','SecondaryPM25','TotalPM25','inmap_PrimaryPM25','inmap_SecondaryPM25',
      'aermod_PrimaryPM25','inmap_only_PrimaryPM25','inmap_only_TotalPM25','BC','inmap_BC','aermod_BC',
      'NO2','inmap_NO2','aermod_NO2','power_TotalPM25']
    maxima={c:0. for c in ['TotalPM25','BC','NO2']}
    fields=[c for c in fields if c in pf.schema_arrow.names]
    for batch in pf.iter_batches(columns=fields):
        d=batch.to_pandas()
        assert np.isfinite(d.to_numpy()).all()
        assert (d.to_numpy()>=-1e-10).all()
        np.testing.assert_allclose(d.PrimaryPM25,d.inmap_PrimaryPM25+d.aermod_PrimaryPM25,atol=1e-12)
        np.testing.assert_allclose(d.SecondaryPM25,d.inmap_SecondaryPM25,atol=1e-12)
        np.testing.assert_allclose(d.TotalPM25,d.PrimaryPM25+d.SecondaryPM25,atol=1e-12)
        np.testing.assert_allclose(d.inmap_only_TotalPM25,d.inmap_only_PrimaryPM25+d.inmap_SecondaryPM25,atol=1e-12)
        np.testing.assert_allclose(d.BC,d.inmap_BC+d.aermod_BC,atol=1e-12)
        np.testing.assert_allclose(d.NO2,d.inmap_NO2+d.aermod_NO2,atol=1e-12)
        assert (d.inmap_only_PrimaryPM25+1e-10>=d.inmap_PrimaryPM25).all()
        if 'power_TotalPM25' in d:assert (d.power_TotalPM25==0).all()
        for col in maxima:maxima[col]=max(maxima[col],float(d[col].max()))
    report=dict(status='complete_partial_isrm',final_concentration_generated=True,grid_cells=pf.metadata.num_rows,
        concentration_file=str(path),maxima_ug_m3=maxima,power_generation=False,
        excluded_isrm_sources=103,aermod_sources_filtered=False,component_identities_passed=True,
        retained_source_emissions_rescaled=(fleet_scale!=1),
        missing_source_redistribution_applied=False,inmap_only_has_local_removal=False,
        traffic_expansion_factor=fleet_scale)
    save(OUT/'concentration_validation.json',report);save(OUT/'run_status.json',report)
    save(OUT/'exposure/concentration_scenario.json',dict(**report,
        warning='PARTIAL ISRM: 103 missing source responses omitted in all InMAP species; AERMOD retains all sources. Not a complete-source comparison.',
        fleet_scale=fleet_scale,units='ug/m3',grid_size_meters=100))
    print(report,flush=True)


def postprocess():
    import gc
    import subprocess
    import sys
    from impacts.pipeline.postprocess.step4_plot_concentrations import run as plot
    from impacts.pipeline.postprocess.step5_plot_exposure import run as exposure_plot
    from impacts.pipeline.postprocess.step8_human_exposure import run as human
    settings=yaml.safe_load((OUT/'settings.yaml').read_text())
    analysis=settings['impacts']['analysis']['human_exposure']
    fleet_scale=json.loads((OUT/'scenario.json').read_text()).get('fleet_scale',1)
    conc=str(OUT/'exposure/beam_concentration_distribution.parquet')
    products=plot(concentration_path=conc,network_path=str(BASE/'preprocess/beam_osm_mapped.parquet'),
        power_plants_path=None,pm25_component_region_path=analysis['ces_tracts_file'],
        output_dir=OUT/'postprocess/concentrations',
        scenario_note=f'Partial ISRM: 103 unavailable sources omitted; all AERMOD sources retained. Traffic only; fleet scale = {fleet_scale:g}.')
    save(OUT/'concentration_plot_manifest.json',products);gc.collect()
    products=exposure_plot(population_path=str(OUT/'exposure/beam_population_counts.parquet'),
        concentration_path=conc,network_path=str(BASE/'preprocess/beam_osm_mapped.parquet'),
        inmap_cells_path=str(OUT/'concentrations/beam_inmap_concentrations.parquet'),output_dir=OUT/'postprocess/exposure')
    save(OUT/'exposure_plot_manifest.json',products);gc.collect()
    products=human(population_path=str(OUT/'exposure/beam_population_counts.parquet'),concentration_path=conc,
        ces_tracts_path=analysis['ces_tracts_file'],sensitive_facilities_path=analysis['sensitive_facilities_file'],
        schools_layer=analysis['schools_layer'],ltc_layer=analysis['ltc_layer'],
        output_dir=OUT/'postprocess/human_exposure',power_plants_path=None)
    save(OUT/'human_exposure_manifest.json',products)
    subprocess.run([sys.executable,str(ROOT/'scripts/export_original_style_human_exposure_tables.py'),
       '--input',str(OUT/'postprocess/human_exposure/human_exposure_ces_tracts.parquet'),
       '--output-dir',str(OUT/'postprocess/human_exposure/tables')],check=True)
    # Keep the original visualization layout, but make the approximation visible.
    html=Path(products['webmap'])
    text=html.read_text(encoding='utf-8')
    warning='<div style="background:#fff3cd;color:#664d03;padding:8px;font-family:sans-serif">PARTIAL ISRM SOURCE COVERAGE: 103 unavailable source cells omitted from InMAP; all AERMOD sources retained. Traffic only; no power generation.</div>'
    warning=warning.replace('Traffic only;',f'Traffic expansion = {fleet_scale:g}x; population weights unchanged. Traffic only;')
    if '<body>' not in text:raise ValueError('Cannot locate webmap body for scenario warning')
    html.write_text(text.replace('<body>','<body>'+warning,1),encoding='utf-8')
    report=json.loads((OUT/'run_status.json').read_text())
    report['postprocess_completed']=True;report['webmap']=str(html)
    save(OUT/'run_status.json',report)


def validate_products():
    import re
    from html.parser import HTMLParser
    from PIL import Image
    class Parser(HTMLParser):
        srcdoc=None
        def handle_starttag(self,tag,attrs):
            if tag=='iframe':self.srcdoc=dict(attrs).get('srcdoc')
    status=json.loads((OUT/'run_status.json').read_text())
    assert status['postprocess_completed']
    html=Path(status['webmap']).read_text(encoding='utf-8')
    assert 'PARTIAL ISRM SOURCE COVERAGE' in html
    parser=Parser();parser.feed(html)
    assert parser.srcdoc
    payload={}
    for key in ['RAW','DATA','POWER_PLANTS']:
        match=re.search(r'const\s+'+key+r'\s*=\s*',parser.srcdoc)
        assert match,key
        payload[key]=json.JSONDecoder().raw_decode(parser.srcdoc[match.end():])[0]
    assert payload['RAW']['sourceCellMetres']==100
    assert not payload['POWER_PLANTS']
    assert len(payload['RAW']['rasters'])==9
    pngs=list((OUT/'postprocess').rglob('*.png'))
    for path in pngs:
        with Image.open(path) as img:img.verify()
    csvs=list((OUT/'postprocess/human_exposure/tables').glob('*.csv'))
    assert len(csvs)>=3
    for path in csvs:assert len(pd.read_csv(path))>0
    human=json.loads((OUT/'postprocess/human_exposure/human_exposure_report.json').read_text())
    assert human['power_included_in_final_total_pm25'] is False
    report=dict(png_files_verified=len(pngs),table_files_verified=len(csvs),
        web_raster_keys=list(payload['RAW']['rasters']),web_source_cell_meters=100,
        power_plant_markers=0,partial_coverage_warning_present=True,
        browser_interaction_tested=False)
    save(OUT/'product_validation.json',report)
    status['products_verified']=True;save(OUT/'run_status.json',status)
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['setup','allocate','filter','inmap','aermod','exposure','verify','postprocess','validate_products'])
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output.resolve()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.stage=='setup':setup()
    elif args.stage=='filter':filter_sources()
    elif args.stage=='verify':verify()
    elif args.stage=='postprocess':postprocess()
    elif args.stage=='validate_products':validate_products()
    else:run_stage(OUT,'emissions' if args.stage=='allocate' else args.stage)
