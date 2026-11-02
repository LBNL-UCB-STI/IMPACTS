"""Hold BEV Home80 fixed; compare PHEV Home80/90 to cached EVI home L1/L2 50:50."""
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
from compare_unified_ev_energy_evi20c import save_details,SEEDS
from impacts.pipeline.workflow.phev_home_charging import build_phev_preference_schedules,run_phev_periodic

BASE=evi.BASE
OLD=BASE/'bev-phev-home80-evi20c-20261012'
REFERENCE=BASE/'evi-home-l1-l2-sensitivity-20261012'
OUT=BASE/'bev80-phev80-90-evi-equal-20261012'


def comparison_metric(name,curve,reference,n=15440,seed=42):
    x=np.asarray(curve);ref=np.asarray(reference)
    return dict(**metric(name,x,n,seed),RMSE_MW_vs_EVI=float(np.sqrt(np.mean((x-ref)**2))),
        shape_RMSE_percentage_points=float(np.sqrt(np.mean((x/x.sum()*100-ref/ref.sum()*100)**2))),
        energy14_18_MWh=float(x[14:18].sum()),daily_gap_pct_vs_EVI=float((x.sum()/ref.sum()-1)*100))


def main():
    if OUT.exists():raise FileExistsError(f'Preserve results: {OUT}')
    inputs=[BASE/'existing_ev_trips.parquet',BASE/'existing_ev_inventory.parquet',
            BASE/'phev20-sensitivity-20261012/phev20/ev_stop_charging_audit.parquet']
    protected={p:sha(p) for folder in [OLD,REFERENCE] for p in folder.rglob('*') if p.is_file()}
    protected.update({p:sha(p) for p in inputs})
    trips,inventory,audit=[pd.read_parquet(p) for p in inputs]
    assert len(inventory)==1544
    prior=pd.read_csv(OLD/'hourly_all_seeds.csv')
    rec=json.loads((REFERENCE/'Equal_breakdown.json').read_text())
    assert rec['http_status']==200 and rec['inputs']['home_power_dist']=='Equal'
    assert rec['inputs']['work_power_dist']=='MostL2' and rec['inputs']['temp_c']==20 and rec['inputs']['mean_dvmt']==35
    targets=json.loads((REFERENCE/'run_summary.json').read_text())['range_proxy_targets']
    matched=evi.reweight(evi.unpack(rec['response']),targets)
    refs={kind:evi.hourly(evi.total(matched,None if kind=='total' else kind.lower()))/1000 for kind in ['total','BEV','PHEV']}
    saved_ref=pd.read_csv(REFERENCE/'comparison_hourly.csv')
    for kind in refs:
        np.testing.assert_allclose(refs[kind],saved_ref['Equal_MW' if kind=='total' else 'Equal_'+kind+'_MW'],atol=1e-10)
    OUT.mkdir()
    curves=pd.DataFrame({'hour_start':range(24),'EVI_equal_MW':refs['total'],
                         'EVI_equal_BEV_MW':refs['BEV'],'EVI_equal_PHEV_MW':refs['PHEV']})
    records=[];stats=[];hourly=[]
    for seed in SEEDS:
        old=prior.loc[prior.seed.eq(seed)].sort_values('hour')
        b=old.BEV_MW.to_numpy()
        old_schedules=build_phev_preference_schedules(trips,audit,.8,seed)
        old_by_id={s.vehicle:s for s in old_schedules}
        for share in [.8,.9]:
            name=f'PHEV_Home{int(share*100)}'
            schedules=old_schedules if share==.8 else build_phev_preference_schedules(trips,audit,share,seed)
            for s in schedules:
                previous=old_by_id[s.vehicle]
                assert s.capacity==previous.capacity and s.initial==previous.initial and s.ac_power==previous.ac_power
                np.testing.assert_array_equal(s.energy,previous.energy)
                np.testing.assert_array_equal(s.arrival,previous.arrival)
                assert not previous.home_preferred or s.home_preferred
            r=run_phev_periodic(schedules,charging_policy='primary_preference')
            p=r['profiles']['steady_grid']*.01
            v=r['vehicles'];ss=r['sessions']
            assert v.converged.all() and len(v)==936
            np.testing.assert_allclose(v.steady_grid_charge_kWh*.9,v.steady_battery_traction_kWh,atol=1e-7,rtol=0)
            assert (ss.charging_time_s<=ss.parking_time_s+1e-8).all()
            eligible=sum(bool(s.home[-1] and s.dwell[-1]>900) for s in schedules)
            assert v.home_preferred.sum()==int(np.floor(share*eligible))
            home_kwh=float(ss.loc[ss.is_home].grid_energy_kWh.sum())
            row=dict(seed=seed,scenario=name,eligible_home_vehicles=eligible,
                assigned_home_preference=int(v.home_preferred.sum()),primary_home_vehicles=int(v.primary_home.sum()),
                primary_nonhome_vehicles=int(((v.primary_index>=0)&~v.primary_home).sum()),
                home_grid_energy_share=home_kwh/ss.grid_energy_kWh.sum(),PHEV_daily_MWh=float(p.sum()),
                PHEV_home_MWh=home_kwh*.01,PHEV_nonhome_MWh=float(p.sum()-home_kwh*.01),
                electric_distance_share=float(v.steady_electric_distance_km.sum()/v.steady_total_distance_km.sum()),
                max_iterations=int(v.iterations.max()))
            records.append(row)
            stats.extend([comparison_metric('BEV80_'+name,b+p,refs['total'],seed=seed),
                          comparison_metric(name,p,refs['PHEV'],9370,seed)])
            for h in range(24):hourly.append(dict(seed=seed,scenario=name,hour=h,BEV_MW=float(b[h]),PHEV_MW=float(p[h]),total_MW=float(b[h]+p[h])))
            if share==.8:
                np.testing.assert_allclose(p,old.PHEV_MW,atol=1e-10,rtol=0)
                if seed==42:
                    for key in ['vehicles','sessions','trips']:
                        pd.testing.assert_frame_equal(r[key],pd.read_parquet(OLD/'phev_home80'/(key+'.parquet')))
            if seed==42:
                save_details(r,OUT/name.lower())
                curves[name+'_MW']=p;curves['BEV80_'+name+'_MW']=b+p
                curves['BEV_Home80_MW']=b
            print(json.dumps(row),flush=True)
    data=pd.DataFrame(stats);ensemble=pd.DataFrame(hourly)
    data.to_csv(OUT/'metrics_all_seeds.csv',index=False)
    ensemble.to_csv(OUT/'hourly_all_seeds.csv',index=False)
    pd.DataFrame(records).to_csv(OUT/'phev_location_audit.csv',index=False)
    for col,n in [('BEV80_PHEV_Home80_MW',15440),('BEV80_PHEV_Home90_MW',15440),('EVI_equal_MW',15440),
                  ('PHEV_Home80_MW',9370),('PHEV_Home90_MW',9370),('EVI_equal_PHEV_MW',9370),
                  ('BEV_Home80_MW',6070),('EVI_equal_BEV_MW',6070)]:
        curves[col.replace('_MW','_kW_per1000')]=curves[col]*1e6/n
    curves.to_csv(OUT/'comparison_hourly.csv',index=False)
    selected=data.loc[data.seed.eq(42)].to_dict('records')
    selected.append(comparison_metric('EVI_equal',refs['total'],refs['total']))
    pd.DataFrame(selected).to_csv(OUT/'comparison_metrics.csv',index=False)
    plt.rcParams.update({'font.size':11})
    fig,axes=plt.subplots(3,1,figsize=(13,12),layout='constrained')
    for label,col,color,style in [('BEV Home80 + PHEV Home80 (previous)','BEV80_PHEV_Home80_MW','#2166ac','-'),
            ('BEV Home80 + PHEV Home90 (new)','BEV80_PHEV_Home90_MW','#7b3294','--'),
            ('EVI: residential L1/L2 50:50','EVI_equal_MW','#00876c','-.')]:
        x=curves[col].to_numpy()
        for ax,y in zip(axes,[x,x*1e6/15440,x/x.sum()*100]):
            ax.plot(range(24),y,color=color,ls=style,lw=2.3,label=label)
    for name,color in [('PHEV_Home80','#2166ac'),('PHEV_Home90','#7b3294')]:
        g=ensemble.loc[ensemble.scenario.eq(name)].groupby('hour').total_MW
        axes[0].fill_between(range(24),g.min(),g.max(),color=color,alpha=.1)
    for ax,title,unit in zip(axes,['Total: 15,440 input PEVs (10x)','Per 1,000 input PEVs','Hourly energy share'],
                            ['Hourly-average grid power (MW)','Grid kW / 1,000 PEVs','Share of daily energy (%)']):
        ax.set(title=title,xlabel='Hour (interval start; no time shift)',ylabel=unit,xlim=(0,23),xticks=range(24),ylim=(0,None));ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle('PHEV Home80 vs Home90, with BEV Home80 fixed',fontsize=16)
    fig.supxlabel('Code: 0.202 battery kWh/km, 90% efficiency; L1/L2 remains 20:80; SOC-driven PHEV fuel fallback.\n'
        'EVI only: residential L1/L2 = 50:50; weekday, 20 C, 35 mi/day, Home80, workplace L1/L2 = 20:80.\n'
        'Home preference is a vehicle assignment, not a required energy share; both code scenarios use one primary PHEV stop.\n'
        'Seed 42 lines; shaded five-seed range. EVI is a model reference, not observed charging; no AQ rerun.',fontsize=9)
    fig.savefig(OUT/'phev_home80_90_vs_evi_equal.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,kind,n in zip(axes,['BEV','PHEV'],[6070,9370]):
        if kind=='BEV':ax.plot(range(24),curves.BEV_Home80_MW*1e6/n,color='#2166ac',lw=2.3,label='BEV Home80 (unchanged)')
        else:
            for name,color,style in [('PHEV_Home80','#2166ac','-'),('PHEV_Home90','#7b3294','--')]:
                ax.plot(range(24),curves[name+'_MW']*1e6/n,color=color,ls=style,lw=2.3,label=name.replace('_',' '))
        ax.plot(range(24),refs[kind]*1e6/n,'-.',color='#00876c',lw=2.3,label='EVI home L1/L2 50:50')
        ax.set(title=kind,xlabel='Hour (interval start)',ylabel='Grid kW / 1,000 '+kind+'s',xlim=(0,23),xticks=range(0,24,2),ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Separate BEV / PHEV comparison: home preference sensitivity',fontsize=15)
    fig.savefig(OUT/'bev_phev_home80_90_vs_evi_equal.png',dpi=180);plt.close(fig)
    for p,h in protected.items():assert sha(p)==h,f'Prior artifact changed: {p}'
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:im.verify()
    result=dict(status='complete',EVI_inputs=rec['inputs'],EVI_reference=str(REFERENCE),
        model_reference=str(OLD),BEV_reused_unchanged=True,PHEV_Home80_exact_prior_parity=True,
        code_changed_parameter_only='PHEV home preference 0.8 vs 0.9',
        code_L1_L2=[.2,.8],EVI_home_L1_L2=[.5,.5],battery_kWh_per_km=.202,charging_efficiency=.9,
        input_fleet=1544,expanded_fleet=15440,modeled_PHEVs=936,missing_PHEV_itinerary=1,
        seed42_excluded_BEVs=12,BEV_results_reused_for_all_seeds=True,
        random_seeds=SEEDS,seed42_audit=records[:2],metrics=selected,
        caveats=['Home90 means floor(90% of eligible terminal-home vehicles), with original fallback selection.',
            'PHEV fuel fallback distance is not fuel emissions.',
            'EVI and charging-code charger mixtures intentionally differ; comparison is sensitivity, not identical assumptions.',
            'No travel time shift, no post-hoc energy scaling, no EVI curve fitting.'],
        previous_results_preserved=True,source_hashes={str(p):h for p,h in protected.items()},
        code_hashes={str(p):sha(p) for p in [Path(__file__),Path('src/impacts/pipeline/workflow/phev_home_charging.py'),
            Path('src/impacts/pipeline/workflow/home_priority_charging.py')]},
        power_generation_run=False,air_quality_run=False)
    (OUT/'run_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(pd.DataFrame(selected).to_string(index=False));print('OUTPUT:',OUT)


if __name__=='__main__':main()
