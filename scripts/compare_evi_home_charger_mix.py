"""Compare three EVI residential charger mixes without altering charging code."""
from pathlib import Path
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import match_baseline_evi_profiles as evi
from compare_charging_0224_evi20c import sha
from impacts.pipeline.workflow.power_generation import allocate_charging_event

BASE=evi.BASE
MODEL=BASE/'bev-phev-home80-evi20c-20261012'
REFERENCE=BASE/'home-priority-0224-evi20c-20261012'
OUT=BASE/'evi-home-l1-l2-sensitivity-20261012'
MIXES={'MostL2':('L1 20% / L2 80%','#d95f02','--'),
       'Equal':('L1 50% / L2 50%','#00876c','-.'),
       'MostL1':('L1 80% / L2 20%','#7b3294',':')}


def metrics(name,p,code):
    p=np.asarray(p);code=np.asarray(code)
    return dict(scenario=name,daily_MWh=float(p.sum()),peak_MW=float(p.max()),peak_hour=int(p.argmax()),
        daily_kWh_per_vehicle=float(p.sum()*1000/15440),power14_MW=float(p[14]),power18_MW=float(p[18]),
        ramp14_18_MW=float(p[18]-p[14]),energy14_18_MWh=float(p[14:18].sum()),
        evening18_24_share=float(p[18:].sum()/p.sum()),
        RMSE_MW_vs_code=float(np.sqrt(np.mean((p-code)**2))),
        RMSE14_18_MW_vs_code=float(np.sqrt(np.mean((p[14:18]-code[14:18])**2))),
        shape_RMSE_percentage_points=float(np.sqrt(np.mean((100*p/p.sum()-100*code/code.sum())**2))),
        EVI_minus_code_daily_pct=float((p.sum()/code.sum()-1)*100))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fetch-only',action='store_true');args=parser.parse_args()
    if (OUT/'run_summary.json').exists():raise FileExistsError(f'Preserve completed results: {OUT}')
    OUT.mkdir(exist_ok=True)
    protected={p:sha(p) for folder in [MODEL,REFERENCE] for p in folder.rglob('*') if p.is_file()}
    inputs=json.loads((REFERENCE/'run_summary.json').read_text())['EVI_inputs']
    assert inputs['temp_c']==20 and inputs['mean_dvmt']==35 and inputs['home_power_dist']=='MostL2'
    assert inputs['work_power_dist']=='MostL2' and inputs['pref_dist']=='Home80'
    targets=json.loads((BASE/'evi-fleet-matched-20261012/matching_summary.json').read_text())['range_proxy_targets']
    assert sum(targets.values())==15440
    # Keep a local copy of the official contract (not generated profile data).
    spec=OUT/'official_api_spec.yml'
    if not spec.exists():
        response=requests.get(evi.SPEC,timeout=30);response.raise_for_status()
        spec.write_text(response.text,encoding='utf-8')
    assert 'MostL1' in spec.read_text() and 'home_power_dist' in spec.read_text()
    evi.OUT=OUT
    # Fetch actual new scenario outputs, never rescale L1/L2 curves manually.
    def fetch_mix(mix):
        scenario={**inputs,'home_power_dist':mix}
        if mix=='MostL2':
            result=[json.loads((REFERENCE/f'evi20c_{kind}.json').read_text()) for kind in ['breakdown','aggregate']]
            assert all(r['inputs']==scenario and r['http_status']==200 for r in result)
            return mix,result
        return mix,[evi.fetch(evi.API+endpoint,scenario,f'{mix}_{kind}.json') for kind,endpoint in
                    [('breakdown','/daily-load-profile/breakdown.json'),('aggregate','/daily-load-profile.json')]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses=dict(pool.map(fetch_mix,MIXES))
    print('EVI API responses verified for all three home mixes',flush=True)
    if args.fetch_only:return
    model=pd.read_csv(MODEL/'comparison_hourly.csv')
    code=model.Both_Home80_MW.to_numpy()
    hourly=pd.DataFrame({'hour_start':range(24),'Charging_code_MW':code,'Charging_code_kW_per1000':code*1e6/15440})
    quarter=pd.DataFrame({'hour_start':np.arange(96)/4})
    aggregate_metrics=[metrics('Charging_code',code,code)];part_metrics=[];allparts={};checks={}
    for mix,(breakdown,aggregate) in responses.items():
        parts=evi.unpack(breakdown['response'])
        full=aggregate['response']['results']['weekday_load_profile']
        errors={}
        for charger in evi.CHARGERS:
            np.testing.assert_allclose(sum(parts[charger].values()),full[charger],atol=.046,rtol=0)
            errors[charger]=float(np.max(np.abs(sum(parts[charger].values())-full[charger])))
        checks[mix]=errors
        matched=evi.reweight(parts,targets)
        p=evi.hourly(evi.total(matched))/1000
        if mix=='MostL2':np.testing.assert_allclose(p,model.EVI20C_MW,atol=1e-10,rtol=0)
        hourly[mix+'_MW']=p;hourly[mix+'_kW_per1000']=p*1e6/15440
        hourly[mix+'_energy_share_pct']=100*p/p.sum()
        aggregate_metrics.append(metrics(mix,p,code))
        for charger,subtypes in matched.items():
            for subtype,x in subtypes.items():quarter[mix+'_'+charger+'_'+subtype+'_kW']=x
        components={charger:evi.hourly(sum(data.values()))/1000 for charger,data in matched.items()}
        components.update(home=components['home_l1']+components['home_l2'],
                          work=components['work_l1']+components['work_l2'],
                          public=components['public_l2']+components['public_l3'])
        for kind in ['BEV','PHEV']:components[kind]=evi.hourly(evi.total(matched,kind.lower()))/1000
        allparts[mix]=components
        for component,x in components.items():
            hourly[mix+'_'+component+'_MW']=x
            part_metrics.append(dict(scenario=mix,component=component,daily_MWh=float(x.sum()),
                peak_MW=float(x.max()),peak_hour=int(x.argmax()),energy14_18_MWh=float(x[14:18].sum())))
        print(json.dumps(aggregate_metrics[-1]),flush=True)
    # Independently reconstruct unchanged model components from charging sessions.
    cp={k:np.zeros(24) for k in ['home_l1','home_l2','work','other_nonhome']}
    for kind in ['bev','phev']:
        ss=pd.read_parquet(MODEL/(kind+'_home80')/'sessions.parquet')
        if 'included_in_profile' in ss:ss=ss.loc[ss.included_in_profile]
        for r in ss.itertuples():
            group=('home_l1' if r.grid_power_kW==1.9 else 'home_l2') if r.is_home else (
                'work' if r.activity=='Work/Education' else 'other_nonhome')
            duration=getattr(r,'timeCharging_s',getattr(r,'charging_time_s',None))
            allocate_charging_event(cp[group],start_hour=r.arrivalTime/3600,
                duration_hours=duration/3600,energy_kwh=r.grid_energy_kWh)
    cp={k:p*.01 for k,p in cp.items()}
    np.testing.assert_allclose(sum(cp.values()),code,atol=1e-10,rtol=0)
    for group,p in cp.items():hourly['Charging_code_'+group+'_MW']=p
    for kind in ['BEV','PHEV']:hourly['Charging_code_'+kind+'_MW']=model[kind+'_Home80_MW']
    hourly.to_csv(OUT/'comparison_hourly.csv',index=False)
    quarter.to_csv(OUT/'evi_15min_components.csv',index=False)
    stats=pd.DataFrame(aggregate_metrics);stats.to_csv(OUT/'comparison_metrics.csv',index=False)
    pd.DataFrame(part_metrics).to_csv(OUT/'component_metrics.csv',index=False)
    ranking=stats.loc[stats.scenario.ne('Charging_code')].copy()
    for key in ['RMSE_MW_vs_code','shape_RMSE_percentage_points','RMSE14_18_MW_vs_code']:
        ranking[key+'_rank']=ranking[key].rank(method='min').astype(int)
    ranking.to_csv(OUT/'fit_ranking.csv',index=False)

    plt.rcParams.update({'font.size':11})
    fig,axes=plt.subplots(3,1,figsize=(13,12),layout='constrained')
    series=[('Charging code: both Home80',code,'#2166ac','-')]+[(MIXES[m][0],hourly[m+'_MW'].to_numpy(),MIXES[m][1],MIXES[m][2]) for m in MIXES]
    for label,p,color,style in series:
        for ax,y in zip(axes,[p,p*1e6/15440,100*p/p.sum()]):
            ax.plot(range(24),y,color=color,ls=style,lw=2.4,label=label)
    for ax,title,unit in zip(axes,['Total: 15,440 input PEVs','Per 1,000 input PEVs','Shape: each daily total = 100%'],
        ['Hourly-average grid power (MW)','Grid kW / 1,000 PEVs','Share of daily energy (%)']):
        ax.set(title=title,xlabel='Hour (interval start; no time shift)',ylabel=unit,xlim=(0,23),xticks=range(24),ylim=(0,None))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=10)
    fig.suptitle('EVI residential L1/L2 mix sensitivity vs unchanged charging code',fontsize=16)
    fig.supxlabel('Weekday; 20 C, 35 mi/day, Home80 preference, HA100 access; workplace mix remains L1 20% / L2 80%.\n'
        'EVI subtype profiles reweighted to 6,070 BEVs + 9,370 PHEVs; source API fleet = 30,000.\n'
        'Charging code unchanged: 0.202 battery kWh/km, 90% efficiency, both Home80, 10x; infeasible BEVs audited.\n'
        'Fit to another model is not validation of the actual SF Bay charger mix.',fontsize=9)
    fig.savefig(OUT/'evi_home_mix_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for ax,component in zip(axes.flat,['home_l1','home_l2','work','public']):
        codekey=component if component!='public' else 'other_nonhome'
        label='Code: other nonhome (proxy)' if component=='public' else 'Charging code'
        ax.plot(range(24),cp[codekey],color='#2166ac',lw=2.3,label=label)
        for mix,(label,color,style) in MIXES.items():
            ax.plot(range(24),allparts[mix][component],color=color,ls=style,lw=2,label=label)
        ax.set(title=component.replace('_',' ').title(),xlabel='Hour (interval start)',ylabel='Hourly-average MW',
               xlim=(0,23),xticks=range(0,24,2),ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Charging location / charger components: only EVI home power mix varies',fontsize=15)
    fig.supxlabel('Code Work includes Work/Education labels; other nonhome is not verified public charging infrastructure.',fontsize=10)
    fig.savefig(OUT/'evi_home_mix_components.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,kind,n in zip(axes,['BEV','PHEV'],[6070,9370]):
        ax.plot(range(24),model[kind+'_Home80_MW']*1e6/n,color='#2166ac',lw=2.3,label='Charging code')
        for mix,(label,color,style) in MIXES.items():
            ax.plot(range(24),allparts[mix][kind]*1e6/n,color=color,ls=style,lw=2,label=label)
        ax.set(title=kind,xlabel='Hour (interval start)',ylabel='Grid kW / 1,000 '+kind+'s',xlim=(0,23),
               xticks=range(0,24,2),ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('BEV / PHEV comparison across EVI residential charger mixes',fontsize=15)
    fig.savefig(OUT/'evi_home_mix_bev_phev.png',dpi=180);plt.close(fig)
    for p,h in protected.items():assert sha(p)==h,f'Protected file changed: {p}'
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:im.verify()
    result=dict(status='complete',only_changed_EVI_parameter='home_power_dist',base_EVI_inputs=inputs,
        scenarios={m:{**inputs,'home_power_dist':m} for m in MIXES},range_proxy_targets=targets,
        hourly_aggregation='Mean of four 15-minute kW observations; hour denotes interval start',
        daily_energy='sum of hourly-average MW times 1 hour',
        source_reference=str(REFERENCE),model_reference=str(MODEL),source_contract=evi.SPEC,
        MostL2_cached_reference_exact_match=True,model_session_reconstruction_exact_match=True,
        aggregate_breakdown_max_errors_kW=checks,metrics=aggregate_metrics,
        best_absolute_fit=ranking.sort_values('RMSE_MW_vs_code').scenario.iloc[0],
        best_shape_fit=ranking.sort_values('shape_RMSE_percentage_points').scenario.iloc[0],
        best_afternoon_fit=ranking.sort_values('RMSE14_18_MW_vs_code').scenario.iloc[0],
        caveats=['Model-to-model fit cannot identify the real local charger mix.',
                 'Only EVI is varied; code remains the L1 20% / L2 80% assignment scenario.',
                 'EVI range proxy and 80/20 sedan/SUV mixture retained, not locally validated.',
                 'Code nonhome activity categories are not measured charger sites.'],
        prior_results_preserved=True,source_hashes={str(p):h for p,h in protected.items()},
        script_sha256=sha(Path(__file__)),charging_code_rerun=False,power_generation_run=False,air_quality_run=False)
    (OUT/'run_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(stats.to_string(index=False));print('OUTPUT:',OUT)


if __name__=='__main__':main()
