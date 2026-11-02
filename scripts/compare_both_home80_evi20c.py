"""Existing BEV + PHEV Home80 preference, with full SOC-based energy accounting."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from compare_charging_0224_evi20c import sha
from compare_unified_ev_energy_evi20c import BASE, REFERENCE, SEEDS, save_details
from impacts.pipeline.workflow.home_priority_charging import build_schedules,run_periodic
from impacts.pipeline.workflow.phev_home_charging import (
    build_phev_preference_schedules,build_phev_home_schedules,run_phev_periodic,
    BATTERY_KWH_PER_KM,CHARGING_EFFICIENCY)

OLD=BASE/'unified-ev-energy-evi20c-20261012'
OUT=BASE/'bev-phev-home80-evi20c-20261012'


def metric(name,p,n=15440,seed=42,phase='steady'):
    p=np.asarray(p)
    return dict(scenario=name,seed=seed,phase=phase,normalization_input_fleet=n,
        daily_MWh=float(p.sum()),peak_MW=float(p.max()),peak_hour=int(p.argmax()),
        daily_grid_kWh_per_input_vehicle=float(p.sum()*1000/n),
        evening18_24_share=float(p[18:].sum()/p.sum()))


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserve results: {OUT}')
    inputs=[BASE/'existing_ev_trips.parquet',BASE/'existing_ev_inventory.parquet',
            BASE/'phev20-sensitivity-20261012/phev20/ev_stop_charging_audit.parquet']
    protected={p:sha(p) for folder in [OLD,REFERENCE] for p in folder.rglob('*') if p.is_file()}
    protected.update({p:sha(p) for p in inputs})
    trips,inventory,audit=[pd.read_parquet(p) for p in inputs]
    assert len(inventory)==1544
    old=pd.read_csv(OLD/'comparison_hourly.csv')
    old_seeds=pd.read_csv(OLD/'hourly_all_seeds.csv')
    manifest=json.loads((OLD/'run_summary.json').read_text())
    assert manifest['EVI_inputs']['temp_c']==20
    OUT.mkdir()
    records=[];metrics=[];hourly=[]
    for seed in SEEDS:
        schedules=build_phev_preference_schedules(trips,audit,.8,seed)
        raw_km=trips.loc[trips.arrivalTime.lt(29*3600)].groupby('vehicle').length.sum()/1000
        for s in schedules:
            np.testing.assert_allclose(s.energy.sum(),raw_km[s.vehicle]*BATTERY_KWH_PER_KM,atol=1e-8)
        phev=run_phev_periodic(schedules,charging_policy='primary_preference')
        bev=run_periodic([s for s in build_schedules(trips,audit,.8,seed,
            base_kWh_per_km=BATTERY_KWH_PER_KM) if s.kind=='BEV'],efficiency=CHARGING_EFFICIENCY)
        p=phev['profiles']['steady_grid']*.01
        b=bev['profiles']['steady_BEV']*.01
        prior=old_seeds.loc[(old_seeds.seed==seed)&old_seeds.scenario.eq('Home80')].sort_values('hour')
        np.testing.assert_allclose(b,prior.BEV_MW,atol=1e-10,rtol=0)
        v=phev['vehicles'];ss=phev['sessions'];tt=phev['trips']
        assert len(v)==936 and v.converged.all() and bev['vehicles'].converged.all()
        primary={s.vehicle:s.primary for s in schedules}
        assert ss.trip_index.eq(ss.vehicle.map(primary)).all()
        assert ss.grid_power_kW.isin([1.9,7.2]).all()
        assert (ss.charging_time_s<=ss.parking_time_s+1e-8).all()
        np.testing.assert_allclose(v.steady_grid_charge_kWh*.9,v.steady_battery_traction_kWh,atol=1e-7,rtol=0)
        np.testing.assert_allclose(tt.distance_km,tt.electric_distance_km+tt.fuel_distance_km,atol=1e-8)
        eligible=sum(bool(s.home[-1] and s.dwell[-1]>900) for s in schedules)
        preferred=sum(s.home_preferred for s in schedules)
        assert preferred==int(np.floor(.8*eligible))
        home_energy=float(ss.loc[ss.is_home].grid_energy_kWh.sum())
        row=dict(seed=seed,PHEV_modeled=len(v),PHEV_home_preference_eligible=eligible,
            PHEV_assigned_home_preference=preferred,PHEV_primary_home=int(v.primary_home.sum()),
            PHEV_primary_nonhome=int(((v.primary_index>=0)&~v.primary_home).sum()),
            PHEV_no_primary=int((v.primary_index<0).sum()),PHEV_charging_vehicles=int(ss.vehicle.nunique()),
            PHEV_home_grid_energy_share=home_energy/float(ss.grid_energy_kWh.sum()),
            PHEV_home_MWh=home_energy*.01,PHEV_nonhome_MWh=float(ss.loc[~ss.is_home].grid_energy_kWh.sum()*.01),
            PHEV_grid_MWh=float(p.sum()),PHEV_battery_MWh=float(p.sum()*.9),
            PHEV_electric_distance_share=float(v.steady_electric_distance_km.sum()/v.steady_total_distance_km.sum()),
            PHEV_fuel_distance_km_10x=float(v.steady_fuel_distance_km.sum()*10),
            PHEV_max_iterations=int(v.iterations.max()),BEV_excluded=int((~bev['vehicles'].steady_included).sum()))
        records.append(row)
        metrics.extend([metric('Both_Home80',b+p,seed=seed),metric('BEV_Home80',b,6070,seed),
                        metric('PHEV_Home80',p,9370,seed),
                        metric('Both_Home80',bev['profiles']['first_feasible']*.01+phev['profiles']['first_grid']*.01,
                               seed=seed,phase='first')])
        for h in range(24):
            hourly.append(dict(seed=seed,hour=h,BEV_MW=float(b[h]),PHEV_MW=float(p[h]),total_MW=float(b[h]+p[h])))
        if seed==42:
            for key in ['vehicles','sessions','trips']:
                pd.testing.assert_frame_equal(bev[key],pd.read_parquet(OLD/'bev_home80'/(key+'.parquet')))
            # Ensure the new optional policy has not altered home-only behavior.
            legacy=run_phev_periodic(build_phev_home_schedules(trips,audit,seed))
            for key in ['vehicles','sessions','trips']:
                pd.testing.assert_frame_equal(legacy[key],pd.read_parquet(OLD/'phev_home_only'/(key+'.parquet')))
            save_details(bev,OUT/'bev_home80',bev=True)
            save_details(phev,OUT/'phev_home80')
            v.loc[v.primary_index<0].to_csv(OUT/'phev_home80/no_primary.csv',index=False)
            tt.loc[tt.fuel_distance_km>0].to_csv(OUT/'phev_home80/fuel_fallback_trips.csv',index=False)
            ss.groupby(['is_home','activity'],dropna=False).agg(
                sessions=('vehicle','size'),charging_vehicles=('vehicle','nunique'),
                grid_energy_kWh_sample=('grid_energy_kWh','sum')).to_csv(OUT/'phev_charging_by_location.csv')
            curves=pd.DataFrame({'hour':range(24),'BEV_Home80_MW':b,'PHEV_Home80_MW':p,'Both_Home80_MW':b+p,
                'Previous_PHEV_home_only_MW':old.PHEV_MW,'Previous_total_MW':old.Home80_total_MW,
                'EVI20C_MW':old.EVI20C_MW,'EVI20C_BEV_MW':old.BEV_EVI20C_MW,'EVI20C_PHEV_MW':old.PHEV_EVI20C_MW})
            for col,n in [('Both_Home80_MW',15440),('Previous_total_MW',15440),('EVI20C_MW',15440),
                          ('BEV_Home80_MW',6070),('PHEV_Home80_MW',9370),('EVI20C_BEV_MW',6070),('EVI20C_PHEV_MW',9370)]:
                curves[col.replace('_MW','_kW_per1000')]=curves[col]*1e6/n
            curves.to_csv(OUT/'comparison_hourly.csv',index=False)
        print(json.dumps(row),flush=True)
    all_metrics=pd.DataFrame(metrics);all_metrics.to_csv(OUT/'metrics_all_seeds.csv',index=False)
    ensemble=pd.DataFrame(hourly);ensemble.to_csv(OUT/'hourly_all_seeds.csv',index=False)
    pd.DataFrame(records).to_csv(OUT/'scenario_audit.csv',index=False)
    selected=all_metrics.loc[(all_metrics.seed==42)&all_metrics.phase.eq('steady')].to_dict('records')
    for name,col,n in [('Previous_total','Previous_total_MW',15440),('Previous_PHEV_home_only','Previous_PHEV_home_only_MW',9370),
                       ('EVI20C','EVI20C_MW',15440),('EVI20C_BEV','EVI20C_BEV_MW',6070),('EVI20C_PHEV','EVI20C_PHEV_MW',9370)]:
        selected.append(metric(name,curves[col],n,phase='reference'))
    pd.DataFrame(selected).to_csv(OUT/'comparison_metrics.csv',index=False)

    plt.rcParams.update({'font.size':11})
    fig,axes=plt.subplots(3,1,figsize=(13,12),layout='constrained')
    for label,col,color,style in [('BEV Home80 + PHEV Home80','Both_Home80_MW','#2166ac','-'),
            ('Previous: BEV Home80 + PHEV home-only','Previous_total_MW','#7b3294','--'),
            ('EVI 20 C, 35 mi/day, Home80 reference','EVI20C_MW','#d95f02','-.')]:
        p=curves[col].to_numpy()
        for ax,y in zip(axes,[p,p*1e6/15440,p/p.sum()*100]):
            ax.plot(range(24),y,color=color,ls=style,lw=2.3,label=label)
    group=ensemble.groupby('hour').total_MW
    axes[0].fill_between(range(24),group.min(),group.max(),color='#2166ac',alpha=.12)
    for ax,title,unit in zip(axes,['Stable-day total: 15,440 input PEVs (10x)','Per 1,000 input PEVs','Hourly energy share'],
            ['Hourly-average grid power (MW)','Grid kW / 1,000 PEVs','Share of daily energy (%)']):
        ax.set(title=title,ylabel=unit,xlabel='Hour (events clock; no time shift)',xlim=(0,23),xticks=range(24),ylim=(0,None))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9,loc='upper left')
    fig.suptitle('Existing-EV charging: Home80 preference for both BEV and PHEV',fontsize=16)
    fig.supxlabel('Both: 0.202 battery kWh/km, 90% charging efficiency; 10x expansion, no additional x2.\n'
        'PHEV: AC at one preferred stop; SOC-based fuel fallback; no fixed electric-use factor or DC topups.\n'
        'Home80 = modeled vehicle preference among eligible terminal-home vehicles, not an imposed energy share.\n'
        'Seed 42 lines; shaded five-seed range. EVI is a scenario reference, not observed charging or identical algorithms.',fontsize=9)
    fig.savefig(OUT/'both_home80_vs_evi20c.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,kind,n in zip(axes,['BEV','PHEV'],[6070,9370]):
        ax.plot(range(24),curves[kind+'_Home80_MW']*1e6/n,color='#2166ac',lw=2.3,label=kind+' Home80')
        if kind=='PHEV':
            ax.plot(range(24),curves.Previous_PHEV_home_only_MW*1e6/n,'--',color='#7b3294',lw=2,label='Previous: PHEV home-only')
        ax.plot(range(24),curves['EVI20C_'+kind+'_MW']*1e6/n,'-.',color='#d95f02',lw=2.3,label='EVI 20 C reference')
        ax.set(title=kind,ylabel='Grid kW / 1,000 input '+kind+'s',xlabel='Hour',xlim=(0,23),xticks=range(0,24,2),ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('BEV / PHEV: Home80 preference and unified energy parameters',fontsize=15)
    fig.savefig(OUT/'bev_phev_home80_vs_evi20c.png',dpi=180);plt.close(fig)
    for path,digest in protected.items():
        assert sha(path)==digest,f'Protected artifact changed: {path}'
    for path in OUT.glob('*.png'):
        with Image.open(path) as im: im.verify()
    result=dict(status='complete',battery_kWh_per_km=.202,charging_efficiency=.9,reserve_kWh=2.,
        scale=10,input_BEVs=607,input_PHEVs=937,modeled_PHEVs=936,missing_PHEV_itinerary=1,
        BEV_exact_prior_parity=True,legacy_PHEV_home_only_exact_prior_parity=True,
        policy='80% (floor) of eligible terminal-home vehicles prefer final Home; others choose longest Work/Education, then other nonhome, then Home fallback',
        PHEV_charging='Only selected primary stop >15min, AC 1.9/7.2kW, no DC/topups, SOC-driven fuel fallback',
        difference_from_home_only='One preferred stop instead of every Home stop; nonhome preference enabled',
        access_assumption='Selected nonhome stop is assumed to have an AC charger; not verified charger availability',
        BEV_policy='Unchanged Home80 plus necessary topups, infeasible vehicles excluded and audited',
        SOC='Original synthetic initial SOC, repeated-day equilibrium, first-day transient saved separately',
        EVI_inputs=manifest['EVI_inputs'],EVI_reference=str(REFERENCE),
        EVI_caveat='Cached reference and subtype weights unchanged; similar preference label does not imply identical EVI algorithms',
        seed42_metrics=selected,seed42_audit=records[0],random_seeds=SEEDS,
        previous_results_preserved=True,source_hashes={str(p):h for p,h in protected.items()},
        code_hashes={str(p):sha(p) for p in [Path(__file__),Path('src/impacts/pipeline/workflow/phev_home_charging.py'),
            Path('src/impacts/pipeline/workflow/home_priority_charging.py')]},
        power_generation_run=False,downstream_AQ_run=False)
    (OUT/'run_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(pd.DataFrame(selected).to_string(index=False))
    print('OUTPUT:',OUT)


if __name__=='__main__':
    main()
