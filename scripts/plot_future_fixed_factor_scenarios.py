"""Original IMPACTS map style for the three frozen-factor BF/HF scenarios."""
import argparse
import gc
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from audit_future_scenario_inputs import DEFAULT_OUTPUT, SCENARIOS
import plot_gateway_products as plots


def run(root=DEFAULT_OUTPUT,dpi=150,only=None,available=False):
    figures=root/'figures'
    scenarios=[only] if only else list(SCENARIOS)
    if available:
        scenarios=[s for s in scenarios if (root/s/'Total/run_status.json').exists()
            and json.loads((root/s/'Total/run_status.json').read_text()).get('status')=='complete']
        if not scenarios:raise ValueError('No completed scenarios to plot')
    complete=set(scenarios)==set(SCENARIOS)
    sources={s:root/s/'Total/exposure/beam_concentration_distribution.parquet' for s in scenarios}
    reports={s:json.loads((root/s/'Total/run_status.json').read_text()) for s in scenarios}
    if any(r['status']!='complete' for r in reports.values()):
        raise ValueError('A scenario has not passed concentration validation')
    signatures={s:(p.stat().st_size,p.stat().st_mtime_ns) for s,p in sources.items()}
    plots.NOTE='Fixed 2018 factors; traffic ×10; no buses/power generation; partial ISRM coverage'
    first=scenarios[0]
    mask,ids,layout,net,counties,context=plots.load_context(root,sources[first],
        manifest_path=root/first/'Total/preprocess_manifest.yaml')
    summary={}
    rows=[]
    for s in scenarios:
        print('RENDER',s,flush=True)
        data=plots.load_values(sources[s],mask,ids)
        limits={p:plots.positive_limit(data,[p]) for p in plots.POLLUTANTS}
        vmax=plots.positive_limit(data,[c for c,_,_,_ in plots.native._PM25_COMPONENT_LAYERS])
        summary[s]=dict(name=s,source=str(sources[s]),limits=limits,component_shared_vmax=vmax,
                        true_sfbay_maxima={p:float(data[p].max()) for p in plots.POLLUTANTS})
        for p,title in plots.POLLUTANTS.items():
            plots.single_map(data,p,title,s,layout,net,counties,limits[p],figures/s/f'{p.lower()}.png',dpi)
        plots.component_comparison(data,s,layout,net,counties,figures/s/'primary_secondary_pm25.png',dpi)
        for col,title,filename,_ in plots.native._PM25_COMPONENT_LAYERS:
            plots.single_map(data,col,title,s,layout,net,counties,vmax,figures/s/filename,dpi)
        rows.append(dict(group=s,**{p+'_PWC_ug_m3':reports[s]['population_weighted_concentrations'][p] for p in plots.POLLUTANTS}))
        del data
        gc.collect()
    scales={}
    if not only:
        scales=plots.comparison_figures(scenarios,summary,layout,net,counties,
            lambda s:plots.load_values(sources[s],mask,ids),figures/'comparison',dpi,
            comparison_title='BF / HF scenarios · fixed emission factors')
        plots.pwc_chart(rows,'Scenario comparison — population-weighted annual concentration',figures/'comparison/scenario_pwc.png',dpi)
    if signatures!={s:(p.stat().st_size,p.stat().st_mtime_ns) for s,p in sources.items()}:
        raise ValueError('Source concentrations changed during plotting')
    verified=[]
    for path in figures.rglob('*.png'):
        with Image.open(path) as im:
            pixels=im.size
            im.verify()
        verified.append(dict(file=str(path.relative_to(figures)),pixels=pixels))
    expected=8*len(scenarios)+4
    if not only and len(verified)!=expected:
        raise ValueError(f'Expected {expected} maps/charts; found {len(verified)}')
    figures.mkdir(exist_ok=True)
    (figures/('figure_manifest.json' if not only else f'{only}_figure_manifest.json')).write_text(json.dumps(dict(
        status='complete',context=context,scenarios=summary,shared_scales=scales,files=verified,
        concentration_units='ug/m3',dpi=dpi,source_products_unchanged=True,
        no2_label_basis='Existing user-confirmed ug/m3 convention; no new numerical conversion',
        emission_factor_year=2018,fleet_scale_already_applied=10,power_generation=False),indent=2),encoding='utf-8')
    if not only:
        pd.DataFrame(rows).to_csv(root/'concentration_comparison.csv',index=False)
        (root/'results_summary.json').write_text(json.dumps(dict(status='complete' if complete else 'partial_scenarios_complete',scenarios=reports,
            pending_scenarios=sorted(set(SCENARIOS)-set(scenarios)),
            figure_count=len(verified),scenario_products=summary,concentration_units='ug/m3',
            limitations=['Frozen 2018 emission factors, not projected 2050 emission factors',
                'BE/H2FC non-exhaust is same-body-class baseline proxy, no regenerative braking adjustment',
                'PHEV mixed trips conservatively engine-on; see per-scenario fuel-mode audits',
                'Previously approved missing ISRM sources omitted; AERMOD sources unchanged',
                'No buses/power generation; representative 0-24h activity window']),indent=2),encoding='utf-8')
        (root/'README.txt').write_text(
            'BF/HF fixed-factor traffic air-quality comparison\n\n'
            + 'Completed concentration scenarios: '+', '.join(scenarios)+'\n'
            + 'Pending scenarios: '+(', '.join(sorted(set(SCENARIOS)-set(scenarios))) or 'None')+'\n\n'
            +
            'Three scenarios: BF-06EP, BF-1EP, HF-1EP. local_run is supporting data, not a fourth events scenario.\n'
            'Each scenario/Total/exposure/beam_concentration_distribution.parquet: final 100-m concentrations.\n'
            'Each scenario/Total/concentrations/beam_inmap_concentrations.parquet: native InMAP grid components.\n'
            'Each scenario/Total/concentrations/beam_aermod_concentrations.parquet: AERMOD local contributions.\n'
            'Each scenario/Total/emissions: spatially allocated annual traffic emissions.\n'
            'Each scenario/annual_emissions_by_category.csv: Passenger, PortRelated and NonPort annual emissions.\n'
            'Category subfolders contain prepared emissions; separate category dispersion is not part of this total-scenario run.\n'
            'figures/<scenario>: eight original-style concentration/component PNGs.\n'
            'figures/comparison: three shared-scale maps and one population-weighted concentration chart.\n'
            'concentration_comparison.csv and results_summary.json: validated concentration summaries.\n\n'
            'Actual new events, sample fraction 0.1, expansion factor 10 applied exactly once before dispersion.\n'
            'Emissions: short tons/year. Concentrations: ug/m3 (existing NO2 convention retained).\n'
            '2018 emission factors and cohort priors are held fixed; these are NOT 2050-factor forecasts.\n'
            'Population weighting uses the unchanged baseline population surface; population is not expanded 10x or projected to 2050.\n'
            'BE/H2FC trucks: same-class baseline brake/tire/dust proxy, zero tailpipe, no regenerative-braking discount.\n'
            'PHEV trucks: secondaryFuel>0 identifies engine use; mixed trips conservatively entirely engine-on.\n'
            'Pure-electric PHEV trips retain non-exhaust and applicable gasoline DIURN.\n'
            'Short slow synthetic connector roads use a documented Local Urban dust proxy where class is missing.\n'
            'Passenger PHEVs retain the original baseline EMFAC PHEV factors, not the new freight mode split.\n'
            'No buses, no power generation. Representative day 0-24 h, baseline annual operating days.\n'
            'Unobserved cold starts/overnight parking are not inferred.\n'
            'HF-1EP: ten user-approved negative-weight trip rows excluded from derived input; original events and excluded rows retained under _work/inputs/derived_events.\n'
            'HF-1EP: additional missing ISRM sources 1926, 1927, 2078 excluded with explicit user approval; their AERMOD contributions retained.\n'
            'Hybrid primary PM2.5/BC/NO2: nonlocal InMAP + AERMOD directly; secondary unchanged.\n'
            'InMAP-only retains local contributions. Previously approved unavailable ISRM source rows omitted, AERMOD intact.\n'
            'Maps use original 100-m cells within the nine-county SF Bay region, green-to-red colors, no plant markers.\n'
            'Positive-cell P99 controls color saturation, NOT the true maximum; both recorded in figure_manifest.json.\n',encoding='utf-8')
    print('PLOTS COMPLETE',len(verified),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--dpi',type=int,default=150)
    p.add_argument('--only',choices=SCENARIOS)
    p.add_argument('--available',action='store_true',help='Render validated scenarios only; label the overall result partial if any scenario is blocked')
    a=p.parse_args()
    run(a.root,a.dpi,a.only,a.available)
