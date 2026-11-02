"""Add native EVI 25% BEV preset; preserve the events-matched EVI reference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import match_baseline_evi_profiles as evi
from compare_charging_0224_evi20c import sha
from compare_both_home80_evi20c import metric

BASE=evi.BASE
PREVIOUS=BASE/'bev80-phev80-90-evi-equal-20261012'
OUT=PREVIOUS/'evi-bev25-reference'
FLEET=15440
SHARES={'phev20':.25,'phev50':.50,'bev100':.10,'bev250':.15}


def main():
    if (OUT/'run_summary.json').exists():raise FileExistsError(f'Preserve completed results: {OUT}')
    protected={p:sha(p) for p in PREVIOUS.rglob('*') if p.is_file() and OUT not in p.parents}
    previous_manifest=json.loads((PREVIOUS/'run_summary.json').read_text())
    inputs={**previous_manifest['EVI_inputs'],'pev_dist':'PHEV'}
    assert inputs['home_power_dist']=='Equal' and inputs['pref_dist']=='Home80'
    assert inputs['fleet_size']==30000 and inputs['temp_c']==20 and inputs['mean_dvmt']==35
    OUT.mkdir(exist_ok=True)
    evi.OUT=OUT
    breakdown=evi.fetch(evi.API+'/daily-load-profile/breakdown.json',inputs,'evi_bev25_breakdown.json')
    aggregate_path=OUT/'evi_bev25_aggregate.json'
    try:
        aggregate=evi.fetch(evi.API+'/daily-load-profile.json',inputs,aggregate_path.name)
    except ValueError:
        # A complete, validated breakdown is sufficient to construct the load.
        # Preserve rate-limit evidence; do not retry, rotate keys or evade limits.
        aggregate=json.loads(aggregate_path.read_text()) if aggregate_path.exists() else {}
        if aggregate.get('http_status')!=429:raise
        print('Aggregate crosscheck unavailable (HTTP 429); using complete HTTP 200 breakdown.',flush=True)
    parts=evi.unpack(breakdown['response'])
    full=aggregate['response']['results']['weekday_load_profile'] if aggregate['http_status']==200 else None
    errors={}
    if full is not None:
        for charger in evi.CHARGERS:
            np.testing.assert_allclose(sum(parts[charger].values()),full[charger],atol=.046,rtol=0)
            errors[charger]=float(np.max(np.abs(sum(parts[charger].values())-full[charger])))
    # Preserve the requested native PHEV-dominant preset; scale fleet size ONLY.
    factor=FLEET/inputs['fleet_size']
    scaled={c:{k:x*factor for k,x in group.items()} for c,group in parts.items()}
    refs={kind:evi.hourly(evi.total(scaled,None if kind=='total' else kind.lower()))/1000 for kind in ['total','BEV','PHEV']}
    np.testing.assert_allclose(refs['total'],refs['BEV']+refs['PHEV'],atol=1e-12)
    quarter={'hour_start':np.arange(96)/4}
    for charger,group in scaled.items():
        for subtype,p in group.items():quarter[charger+'_'+subtype+'_kW']=p
    pd.DataFrame(quarter).to_csv(OUT/'evi_bev25_15min_components.csv',index=False)
    curves=pd.read_csv(PREVIOUS/'comparison_hourly.csv')
    old_columns=list(curves.columns)
    for kind,n in [('total',FLEET),('BEV',3860),('PHEV',11580)]:
        curves['EVI_bev25_'+kind+'_MW']=refs[kind]
        curves['EVI_bev25_'+kind+'_kW_per1000']=refs[kind]*1e6/n
    curves.to_csv(OUT/'comparison_hourly.csv',index=False)
    pd.testing.assert_frame_equal(curves[old_columns],pd.read_csv(PREVIOUS/'comparison_hourly.csv'))
    rows=[]
    for label,column in [('BEV80_PHEV80','BEV80_PHEV_Home80_MW'),('BEV80_PHEV90','BEV80_PHEV_Home90_MW'),
                          ('EVI_events_matched','EVI_equal_MW'),('EVI_25pct_BEV','EVI_bev25_total_MW')]:
        x=curves[column].to_numpy()
        row=metric(label,x)
        row['RMSE_MW_vs_EVI_bev25']=float(np.sqrt(np.mean((x-refs['total'])**2)))
        row['daily_gap_pct_vs_EVI_bev25']=float((x.sum()/refs['total'].sum()-1)*100)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'comparison_metrics.csv',index=False)
    plt.rcParams.update({'font.size':11})
    fig,axes=plt.subplots(3,1,figsize=(13,12),layout='constrained')
    lines=[('BEV Home80 + PHEV Home80','BEV80_PHEV_Home80_MW','#2166ac','-'),
           ('BEV Home80 + PHEV Home90','BEV80_PHEV_Home90_MW','#7b3294','--'),
           ('EVI: events-matched (39.3% BEV)','EVI_equal_MW','#00876c','-.'),
           ('EVI: 25% all-electric / 75% PHEV (new)','EVI_bev25_total_MW','#d95f02',':')]
    for label,column,color,style in lines:
        p=curves[column].to_numpy()
        for ax,y in zip(axes,[p,p*1e6/FLEET,p/p.sum()*100]):
            ax.plot(range(24),y,label=label,color=color,ls=style,lw=2.4)
    ensemble=pd.read_csv(PREVIOUS/'hourly_all_seeds.csv')
    for name,color in [('PHEV_Home80','#2166ac'),('PHEV_Home90','#7b3294')]:
        g=ensemble.loc[ensemble.scenario.eq(name)].groupby('hour').total_MW
        axes[0].fill_between(range(24),g.min(),g.max(),color=color,alpha=.1)
    for ax,title,unit in zip(axes,['Total: same 15,440 PEVs for all curves','Per 1,000 input PEVs','Hourly energy share'],
        ['Hourly-average grid power (MW)','Grid kW / 1,000 PEVs','Share of daily energy (%)']):
        ax.set(title=title,xlabel='Hour (interval start; no time shift)',ylabel=unit,xlim=(0,23),xticks=range(24),ylim=(0,None));ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle('Charging code vs EVI: adding the 25% all-electric vehicle scenario',fontsize=15)
    fig.supxlabel('Both EVI curves: weekday, 20 C, 35 mi/day, Home80, residential L1/L2 50:50, workplace L1/L2 20:80.\n'
        'New native preset: PHEV20/PHEV50/BEV100/BEV250 = 25/50/10/15%; fleet-size scaling only.\n'
        'Existing EVI reference retains events-matched powertrain/range weights; the new curve changes both.\n'
        'Charging code and events unchanged (code L1/L2 20:80); no additional x2, time shift or AQ rerun.',fontsize=9)
    fig.savefig(OUT/'charging_with_evi_bev25.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,kind,old_n,new_n in zip(axes,['BEV','PHEV'],[6070,9370],[3860,11580]):
        if kind=='BEV':ax.plot(range(24),curves.BEV_Home80_MW*1e6/old_n,color='#2166ac',lw=2.3,label='BEV Home80 (unchanged)')
        else:
            for name,color,style in [('PHEV_Home80','#2166ac','-'),('PHEV_Home90','#7b3294','--')]:
                ax.plot(range(24),curves[name+'_MW']*1e6/old_n,color=color,ls=style,lw=2.3,label=name.replace('_',' '))
        ax.plot(range(24),curves['EVI_equal_'+kind+'_MW']*1e6/old_n,'-.',color='#00876c',lw=2.3,label='EVI: events-matched')
        ax.plot(range(24),refs[kind]*1e6/new_n,':',color='#d95f02',lw=2.5,label='EVI: 25% BEV preset (new)')
        ax.set(title=kind,xlabel='Hour (interval start)',ylabel='Grid kW / 1,000 '+kind+'s',xlim=(0,23),xticks=range(0,24,2),ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Separate BEV / PHEV: each curve uses its own powertrain denominator',fontsize=14)
    fig.supxlabel('Code and matched EVI: 6,070 BEVs / 9,370 PHEVs. New EVI preset: 3,860 BEVs / 11,580 PHEVs.\n'
        'The new preset also changes range-class shares; these are not the same underlying fleets.',fontsize=10)
    fig.savefig(OUT/'bev_phev_with_evi_bev25.png',dpi=180);plt.close(fig)
    for p,h in protected.items():assert sha(p)==h,f'Previous result changed: {p}'
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:im.verify()
    result=dict(status='complete',EVI_inputs=inputs,source_contract=evi.SPEC,
        source_fleet=inputs['fleet_size'],comparison_fleet=FLEET,fleet_size_scale_only=factor,
        powertrain_counts={'BEV':3860,'PHEV':11580},native_subtype_shares=SHARES,
        native_subtype_counts={k:int(FLEET*v) for k,v in SHARES.items()},
        retained_previous_curves_exactly=True,aggregate_breakdown_errors_kW=errors,
        breakdown_http_status=breakdown['http_status'],aggregate_http_status=aggregate['http_status'],
        independent_aggregate_crosscheck_completed=full is not None,
        previous_reference=str(PREVIOUS),new_EVI_raw_reference=str(OUT/'evi_bev25_breakdown.json'),
        metrics=rows,source_hashes={str(p):h for p,h in protected.items()},script_sha256=sha(Path(__file__)),
        limitations=['EVI 25% preset changes range-class mix as well as BEV/PHEV proportions.',
            'New preset deliberately not reweighted to the events fleet composition.',
            'Code L1/L2 remains 20:80; EVI residential L1/L2 is 50:50.',
            'EVI curves are model references, not observations.',
            'Independent aggregate API crosscheck unavailable if HTTP 429; complete breakdown remains validated.'],
        charging_code_rerun=False,power_generation_run=False,air_quality_run=False)
    (OUT/'run_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False));print('OUTPUT:',OUT)


if __name__=='__main__':main()
