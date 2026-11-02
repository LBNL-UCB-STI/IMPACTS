"""Add six component figures to the original skim baseline, without rerunning AQ."""
import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import yaml

from impacts.pipeline.postprocess import step4_plot_concentrations as native
from impacts.pipeline.postprocess import _common as style

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'examples/pilates/impacts/impacts_output/impacts-postsim--sfbay--base--20260901-165739'


def main():
    source = BASE/'exposure/beam_concentration_distribution.parquet'
    network = BASE/'preprocess/beam_osm_mapped.parquet'
    settings = yaml.safe_load((BASE/'settings.yaml').read_text(encoding='utf-8'))
    region = Path(settings['impacts']['analysis']['human_exposure']['ces_tracts_file'])
    destination = BASE/'postprocess/concentrations'
    outputs = [destination/filename for layers in native._LOCAL_NONLOCAL_COMPONENT_LAYERS.values()
               for _, _, filename in layers]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite existing figure: {path}')
    fingerprint = (source.stat().st_size, source.stat().st_mtime_ns)
    columns = ['BC','NO2','aermod_BC','inmap_BC','aermod_NO2','inmap_NO2']
    errors = {p:0. for p in ['BC','NO2']}
    print('Checking all receptor component sums',flush=True)
    for batch in pq.ParquetFile(source).iter_batches(columns=columns,batch_size=131072):
        d = batch.to_pandas()
        for p in errors:
            total = d[f'aermod_{p}']+d[f'inmap_{p}']
            np.testing.assert_allclose(d[p],total,rtol=1e-10,atol=1e-10)
            errors[p]=max(errors[p],float((d[p]-total).abs().max()))
    print('Loading original baseline geometry and concentrations',flush=True)
    concentrations = gpd.read_parquet(source,columns=['geometry',*columns])
    net = gpd.read_parquet(network,columns=['geometry']).drop_duplicates().to_crs(concentrations.crs)
    print('Using original PM2.5 component-map SF Bay clipping region',region,flush=True)
    concentrations,net = native._clip_pm25_components_to_sfbay(concentrations,net,str(region))
    layout = style._grid_raster_layout(concentrations)
    report = dict(source=str(source),region=str(region),source_signature=fingerprint,
        baseline='Original emission-skim baseline, not Cristian events and not the 3-km test',
        concentration_recomputed=False,additional_scaling=False,power_plants_plotted=False,
        component_sum_error_ug_m3=errors,sfbay_cells=len(concentrations),units='ug/m3',
        resolution_m=100,figures={},color_scales={})
    for p,layers in native._LOCAL_NONLOCAL_COMPONENT_LAYERS.items():
        data=native._add_local_nonlocal_component_columns(concentrations,p)
        vmax=native._shared_positive_vmax(data,[c for c,_,_ in layers],.99)
        report['color_scales'][p]=dict(vmin=0,vmax=vmax,
            rule='Maximum of the three positive-cell 99th percentiles; shared across all three maps')
        for column,title,filename in layers:
            print('Rendering',filename,flush=True)
            output=native._plot_one(data,net,None,layout,column,title+' (µg/m³)',style.CMAP_GREEN_RED,
                .99,destination/filename,vmax_override=vmax,
                scenario_note='Original skim baseline; existing concentrations unchanged. Shared pollutant color scale (positive-cell P99); higher values saturate.')
            if output is None:
                raise ValueError(f'Expected nonempty component map: {column}')
            report['figures'][column]=output
        del data
    assert fingerprint==(source.stat().st_size,source.stat().st_mtime_ns)
    report['source_unchanged']=True
    (destination/'bc_no2_component_figures_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Complete: six figures; original concentration file unchanged.',flush=True)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    main()
