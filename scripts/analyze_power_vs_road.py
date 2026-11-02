"""Read-only PM2.5 power contribution analysis on existing road receptor grids."""
from pathlib import Path
import gc
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plot_gateway_products as plots
from run_ldv_freight_products import TOTAL as POPULATION_RUN

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'examples/pilates/impacts/impacts_output'
ROAD = BASE / 'events-sfbay-future-scenarios-20261011'
POWER = BASE / 'ldv-home80-freight3-power-combined10x-20261029'
OUT = POWER / 'onroad_impact_analysis'
SCENARIOS = ['BF-06EP', 'BF-1EP', 'HF-1EP']
COMPONENTS = ['TotalPM25', 'PrimaryPM25', 'SecondaryPM25']
NOTE = 'Additive sensitivity: baseline LDV + scenario freight power; both 10x. Not a matched fleet/year forecast.'


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def road_path(scenario):
    return ROAD / scenario / 'Total/exposure/beam_concentration_distribution.parquet'


def weighted_quantile(values, weights, q):
    keep = weights > 0
    v, w = values[keep], weights[keep]
    order = np.argsort(v)
    return float(np.interp(q * w.sum(), np.cumsum(w[order]), v[order]))


def calculate(scenario, mask, ids, pop):
    src = road_path(scenario)
    powerfile = POWER / 'combined' / scenario / 'ldv_freight_power_concentrations.parquet'
    power = pd.read_parquet(powerfile).set_index('inmap_cell_id')
    assert power.index.is_unique and pop.index.is_unique
    report = json.loads((ROAD / scenario / 'Total/run_status.json').read_text())
    target = OUT / scenario
    target.mkdir(parents=True, exist_ok=True)
    selected = []
    burdens = {c: np.zeros(4) for c in COMPONENTS}
    pop_total = 0.
    bay_pop = 0.
    offset = 0
    writer = None
    try:
        for batch in pq.ParquetFile(src).iter_batches(columns=['aermod_cell_id', 'inmap_cell_id', *COMPONENTS], batch_size=131072):
            d = batch.to_pandas()
            end = offset + len(d)
            np.testing.assert_array_equal(d.aermod_cell_id, ids[offset:end])
            pp = power.reindex(d.inmap_cell_id)
            assert not pp.power_TotalPM25.isna().any(), 'Missing power receptors must not be set to zero'
            w = d.aermod_cell_id.map(pop).fillna(0).to_numpy()
            pop_total += w.sum()
            bay_pop += w[mask[offset:end]].sum()
            values = d[['aermod_cell_id', 'inmap_cell_id']].copy()
            for c in COMPONENTS:
                r = d[c].to_numpy()
                l = pp['ldv_power_' + c].to_numpy()
                f = pp['freight_power_' + c].to_numpy()
                p = pp['power_' + c].to_numpy()
                assert np.isfinite(r).all() and (r >= 0).all()
                np.testing.assert_allclose(p, l + f)
                burdens[c] += np.array([np.dot(r, w), np.dot(l, w), np.dot(f, w), np.dot(r+p, w)])
                if c == 'TotalPM25':
                    values['road_PM25_ug_m3'] = r
                    values['ldv_power_PM25_ug_m3'] = l
                    values['freight_power_PM25_ug_m3'] = f
                    values['power_PM25_ug_m3'] = p
                    values['combined_PM25_ug_m3'] = r + p
                    values['power_share_of_combined_percent'] = np.divide(100*p, r+p, out=np.full(len(r), np.nan), where=r+p>0)
                    values['power_relative_to_road_percent'] = np.divide(100*p, r, out=np.full(len(r), np.nan), where=r>0)
            values['person_count'] = w
            values['in_sfbay_display_mask'] = mask[offset:end]
            table = pa.Table.from_pandas(values, preserve_index=False)
            if writer is None:
                metadata = dict(table.schema.metadata or {})
                metadata[b'impacts_analysis'] = json.dumps(dict(geometry_source=str(src),
                    join_key='aermod_cell_id', power_join='inmap_cell_id; native IMPACTS assignment; no interpolation',
                    units='ug/m3, percent, persons; see column suffix', additional_scale=1)).encode()
                writer = pq.ParquetWriter(target / 'power_vs_road_100m_values.parquet', table.schema.with_metadata(metadata), compression='zstd')
            writer.write_table(table)
            selected.append(values.loc[mask[offset:end]].drop(columns=['aermod_cell_id', 'inmap_cell_id', 'in_sfbay_display_mask']))
            offset = end
    finally:
        if writer is not None:
            writer.close()
    np.testing.assert_allclose(pop_total, pop.sum(), atol=1e-6)
    np.testing.assert_allclose(pop_total, report['population'], atol=1e-6)
    rows = []
    for c, b in burdens.items():
        r, l, f, combined = b/pop_total
        np.testing.assert_allclose(r, report['population_weighted_concentrations'][c], rtol=1e-10)
        np.testing.assert_allclose(combined, r+l+f)
        rows.append(dict(scenario=scenario, component=c, population=pop_total,
            road_PWC_ug_m3=r, ldv_power_PWC_ug_m3=l, freight_power_PWC_ug_m3=f,
            power_PWC_ug_m3=l+f, combined_PWC_ug_m3=combined,
            increase_over_road_percent=100*(l+f)/r,
            power_share_of_combined_percent=100*(l+f)/combined,
            ldv_share_of_power_percent=100*l/(l+f)))
    data = pd.concat(selected, ignore_index=True)
    share = data.power_share_of_combined_percent.to_numpy()
    relative = data.power_relative_to_road_percent.to_numpy()
    weights = data.person_count.to_numpy()
    spatial = dict(scenario=scenario, full_receptor_count=offset, mapped_receptors=offset,
        bay_display_receptors=len(data), population=pop_total, bay_display_population=bay_pop,
        bay_max_power_share_percent=float(np.nanmax(share)),
        bay_median_cell_power_share_percent=float(np.nanmedian(share)),
        bay_p95_cell_power_share_percent=float(np.nanquantile(share,.95)),
        bay_p99_cell_power_share_percent=float(np.nanquantile(share,.99)),
        bay_population_weighted_median_share_percent=weighted_quantile(share, weights, .5),
        bay_population_weighted_p95_share_percent=weighted_quantile(share, weights, .95),
        bay_max_increase_over_road_percent=float(np.nanmax(relative)),
        bay_max_power_PM25_ug_m3=float(data.power_PM25_ug_m3.max()),
        bay_max_road_PM25_ug_m3=float(data.road_PM25_ug_m3.max()))
    imax = int(np.nanargmax(share))
    spatial['max_share_cell_road_PM25_ug_m3'] = float(data.road_PM25_ug_m3.iloc[imax])
    spatial['max_share_cell_power_PM25_ug_m3'] = float(data.power_PM25_ug_m3.iloc[imax])
    for threshold in [0.1, 0.5, 1, 5]:
        spatial[f'bay_population_share_where_power_exceeds_{threshold}percent'] = float(100*weights[share>threshold].sum()/bay_pop)
    save(target/'analysis_audit.json', dict(source_road=str(src), source_power=str(powerfile),
        road_PWC_reproduced=True, population_join_verified=True, all_power_receptors_matched=True,
        spatial_statistics=spatial, PWC=rows))
    print('CALCULATED', scenario, rows[0], flush=True)
    return rows, spatial, data


def panel(ax, data, column, layout, net, counties, vmax, title, units, plants=None):
    plots.map_axis(ax, data, column, layout, net, counties, vmax, title, fontsize=15)
    if plants is not None:
        plots.style._add_power_plants(ax, plants)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=plots.style.CMAP_GREEN_RED)
    ax.figure.colorbar(sm, ax=ax, fraction=.035, pad=.02, shrink=.75, label=units)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT/'run_status.json', dict(status='running'))
    mask, ids, layout, net, counties, context = plots.load_context(ROAD/'BF-06EP', road_path('BF-06EP'),
        manifest_path=ROAD/'BF-06EP/Total/preprocess_manifest.yaml')
    popfile = POPULATION_RUN/'exposure/beam_population_counts.parquet'
    pop = pd.read_parquet(popfile, columns=['aermod_cell_id','person_count']).set_index('aermod_cell_id').person_count
    tables, spatial_rows, data_all = [], [], {}
    for scenario in SCENARIOS:
        rows, spatial, data = calculate(scenario, mask, ids, pop)
        tables.extend(rows); spatial_rows.append(spatial); data_all[scenario] = data
        gc.collect()
    summary = pd.DataFrame(tables)
    summary.to_csv(OUT/'population_weighted_contribution_summary.csv', index=False)
    pd.DataFrame(spatial_rows).to_csv(OUT/'spatial_share_summary.csv', index=False)
    total = summary.loc[summary.component.eq('TotalPM25')].set_index('scenario').loc[SCENARIOS]
    # Shared map limits across scenarios. Power has its own scale because it is
    # much smaller; road and road+power use the identical concentration scale.
    roadmax = max(float(d.combined_PM25_ug_m3.quantile(.99)) for d in data_all.values())
    powermax = max(float(d.power_PM25_ug_m3.max()) for d in data_all.values())
    sharemax = max(float(d.power_share_of_combined_percent.max()) for d in data_all.values())
    for scenario, data in data_all.items():
        plant_path = POWER/'combined'/scenario/'combined_sfbay_plant_emissions.csv'
        plants = plots.style._load_emitting_power_plants(str(plant_path), counties.crs)
        fig, axes = plt.subplots(2, 2, figsize=(17, 19))
        specs = [('road_PM25_ug_m3',roadmax,'On-road: AERMOD + InMAP','PM2.5 (µg/m³)',None),
            ('power_PM25_ug_m3',powermax,'Added LDV + freight power contribution','PM2.5 (µg/m³)',plants),
            ('combined_PM25_ug_m3',roadmax,'On-road + power','PM2.5 (µg/m³)',None),
            ('power_share_of_combined_percent',sharemax,'Power / (on-road + power)','Contribution (%)',plants)]
        for ax, (col,vmax,title,units,pp) in zip(axes.flat,specs):
            panel(ax,data,col,layout,net,counties,vmax,title,units,pp)
        fig.suptitle(f'{scenario}: PM2.5 contribution of charging power generation',fontsize=20,y=.985)
        fig.tight_layout(rect=(0,.035,1,.975))
        fig.text(.025,.018,NOTE+'\nRoad/combined: shared P99 color ceiling; power and share: full-range common scales. Power Layer0; SFBAY plants only.',fontsize=10)
        plots.finish(fig,OUT/scenario/'road_power_combined_share_maps.png',150)
    fig,axes=plt.subplots(1,3,figsize=(22,10))
    for ax, scenario in zip(axes, SCENARIOS):
        panel(ax,data_all[scenario],'power_share_of_combined_percent',layout,net,counties,sharemax,
            scenario,'Power share (%)')
    fig.suptitle('PM2.5: power-generation share of on-road + power concentration',fontsize=20)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.text(.02,.015,NOTE+'\nSame full-range percentage scale; a high percentage can reflect a low on-road denominator.',fontsize=11)
    plots.finish(fig,OUT/'power_share_three_scenarios.png',160)
    fig, axes=plt.subplots(1,3,figsize=(17,5.5))
    x=np.arange(3)
    axes[0].bar(x,total.road_PWC_ug_m3,color='#6998b7',label='On-road')
    axes[0].bar(x,total.power_PWC_ug_m3,bottom=total.road_PWC_ug_m3,color='#e28a3c',label='Power')
    axes[0].set(title='Population-weighted PM2.5',ylabel='µg/m³');axes[0].legend()
    axes[1].bar(x,total.ldv_power_PWC_ug_m3,color='#438bba',label='LDV power')
    axes[1].bar(x,total.freight_power_PWC_ug_m3,bottom=total.ldv_power_PWC_ug_m3,color='#e28a3c',label='Freight power')
    axes[1].set(title='Power increment (expanded scale)',ylabel='µg/m³');axes[1].legend()
    axes[2].bar(x-.18,total.increase_over_road_percent,width=.36,color='#6998b7',label='Power / on-road')
    axes[2].bar(x+.18,total.power_share_of_combined_percent,width=.36,color='#e28a3c',label='Power / combined')
    axes[2].set(title='Ratio of population-weighted means',ylabel='%');axes[2].legend()
    for i, row in enumerate(total.itertuples()):
        axes[0].text(i,row.combined_PWC_ug_m3+.006,f'{row.combined_PWC_ug_m3:.4f}',ha='center',fontsize=10)
        axes[1].text(i,row.power_PWC_ug_m3*1.02,f'{row.power_PWC_ug_m3:.6f}',ha='center',fontsize=10)
        axes[2].text(i+.18,row.power_share_of_combined_percent*1.02,f'{row.power_share_of_combined_percent:.3f}%',ha='center',fontsize=10)
    for ax in axes:
        ax.set_xticks(x,SCENARIOS);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.18)
    fig.tight_layout(rect=(0,.1,1,1));fig.text(.02,.035,NOTE+'\nPopulation weights unchanged. Power includes only SFBAY plants; no NO2/BC power concentration is available.',fontsize=10)
    plots.finish(fig,OUT/'population_weighted_power_impact.png',180)
    from PIL import Image
    files=list(OUT.rglob('*.png'))
    for p in files:
        with Image.open(p) as im: im.verify()
    save(OUT/'run_status.json',dict(status='complete',scenario_count=3,figure_count=len(files),
        pollutant='PM2.5',context=context,population_file=str(popfile),
        common_map_limits=dict(road_combined_P99_ug_m3=roadmax,power_max_ug_m3=powermax,power_share_max_percent=sharemax),
        numerical_checks='Population, original PWC, complete ISRM-ID mapping, component additivity passed',
        limitations=[NOTE,'Road fixed 2018 emission factors; original 4.5km removal/full ASRV support',
            'Power Cambium2040/eGRID2023 proxy; Layer0; SFBAY-located plants only',
            'Power resampled by native inmap_cell_id join, not new 100m dispersion',
            'No total-background pollution or avoided road emissions; not net electrification benefit',
            'Do not interpret absent NO2/BC power fields as zero contribution']))


if __name__ == '__main__':
    main()
