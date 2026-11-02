"""Clip display geometry only; preserve freight-power concentration products."""
from pathlib import Path
import hashlib
import json

import geopandas as gpd
import pandas as pd
import yaml

from impacts.pipeline.postprocess.step4_plot_concentrations import _SF_BAY_COUNTIES
from impacts.pipeline.workflow.power_generation import plot_power_generation_concentration

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'examples/pilates/impacts/impacts_output/impacts-postsim--sfbay--base--20260901-165739'
OUT = ROOT/'examples/pilates/impacts/impacts_output/baseline0-ldv-home80-freight3-10x-20261026/freight_power'


def main():
    settings=yaml.safe_load((BASE/'settings.yaml').read_text(encoding='utf-8'))
    region_path=Path(settings['impacts']['analysis']['human_exposure']['ces_tracts_file'])
    region=gpd.read_parquet(region_path)
    county_column=next(c for c in ['County','NAME','NAMELSAD'] if c in region)
    county=region[county_column].astype(str).str.replace(r'\s+County$','',regex=True)
    region=region.loc[county.isin(_SF_BAY_COUNTIES)]
    assert not region.empty
    for scenario in ['BF-06EP','BF-1EP','HF-1EP']:
        folder=OUT/scenario/'power_generation'
        source=folder/'power_generation_isrm_concentrations.parquet'
        signature=hashlib.sha256(source.read_bytes()).hexdigest()
        data=gpd.read_parquet(source)
        mask=region.to_crs(data.crs).geometry.union_all()
        clipped=gpd.clip(data,mask,keep_geom_type=True)
        assert not clipped.empty and clipped.inmap_cell_id.is_unique
        original=data.set_index('inmap_cell_id')
        for column in data.columns:
            if column.startswith('power_'):
                pd.testing.assert_series_equal(clipped.set_index('inmap_cell_id')[column].sort_index(),
                    original.loc[clipped.inmap_cell_id,column].sort_index())
        plants=pd.read_csv(folder/'plant_emissions_used_for_concentration.csv')
        image=folder/'power_generation_isrm_total_pm25_sfbay_outline.png'
        plot_power_generation_concentration(clipped,plants,image,source_layer_index=0,
            method_note=f'{scenario}; freight charging 10x; SF Bay land-area mask')
        assert hashlib.sha256(source.read_bytes()).hexdigest()==signature
        (folder/'sfbay_outline_plot_audit.json').write_text(json.dumps(dict(
            source=str(source),source_sha256=signature,region=str(region_path),image=str(image),
            source_cells=len(data),display_cells=len(clipped),concentrations_unchanged=True,
            operation='Exact polygon intersection for display; no interpolation or additional scaling',
            includes_aermod=False,includes_traffic=False,
            scope='Freight charging power only; primary + secondary PM2.5'),indent=2),encoding='utf-8')
        print('Created',image,flush=True)


if __name__=='__main__':
    main()
