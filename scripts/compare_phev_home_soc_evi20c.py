"""PHEV home-only SOC scenario; preserve previous BEV and EVI references."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from compare_charging_0224_evi20c import sha
from impacts.pipeline.workflow.charging.ldv_home_sensitivity import (
    build_schedules,
    run_periodic,
    build_phev_home_schedules,
    run_phev_periodic,
)

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'examples/pilates/impacts/impacts_output/events-fc07-existing-ev-charging-code10x-20260928-v2'
OLD=BASE/'home-priority-0224-evi20c-20261012'
OUT=BASE/'phev-home-only-soc-evi20c-20261012'
SEEDS=[42,7,21,84,126]


def metric(name,p,phase='steady',seed=42):
    p=np.asarray(p)
    return dict(scenario=name,phase=phase,seed=seed,daily_MWh=float(p.sum()),peak_MW=float(p.max()),
        peak_hour=int(p.argmax()),evening18_24_share=float(p[18:].sum()/p.sum()),
        daily_kWh_per_input_vehicle=float(p.sum()*1000/15440))


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserve previous results: {OUT}')
    input_paths=[BASE/'existing_ev_trips.parquet',BASE/'existing_ev_inventory.parquet',
        BASE/'phev20-sensitivity-20261012/phev20/ev_stop_charging_audit.parquet']
    protected={p:sha(p) for p in OLD.rglob('*') if p.is_file()}
    protected.update({p:sha(p) for p in input_paths})
    OUT.mkdir()
    trips=pd.read_parquet(input_paths[0]);inventory=pd.read_parquet(input_paths[1])
    audit=pd.read_parquet(input_paths[2])
    old=pd.read_csv(OLD/'comparison_hourly.csv')
    old_parts=pd.read_csv(OLD/'powertrain_comparison_hourly.csv')
    old_manifest=json.loads((OLD/'run_summary.json').read_text())
    assert old_manifest['EVI_inputs']['temp_c']==20
    assert len(inventory)==1544
    rows=[];ensembles=[];phev_records=[];stored={};old_bev={}
    for seed in SEEDS:
        schedules=build_phev_home_schedules(trips,audit,seed)
        # Validate full trip distances against source events, not an energy-scaled load curve.
        source_km=trips[trips.arrivalTime.lt(29*3600)].groupby('vehicle').length.sum()/1000
        for s in schedules:
            np.testing.assert_allclose(s.energy.sum()/.202,source_km[s.vehicle],atol=1e-8)
        phev=run_phev_periodic(schedules)
        v=phev['vehicles'];sessions=phev['sessions'];t=phev['trips']
        assert v.converged.all() and len(v)==936
        assert sessions.is_home.all()
        assert (sessions.charging_time_s<=sessions.parking_time_s+1e-8).all()
        np.testing.assert_allclose(t.distance_km,t.electric_distance_km+t.fuel_distance_km,atol=1e-8)
        np.testing.assert_allclose(v.steady_grid_charge_kWh*.9,v.steady_battery_traction_kWh,atol=1e-7)
        p=phev['profiles']['steady_grid']*.01
        ps=dict(seed=seed,modeled_PHEVs=len(v),charging_PHEVs=int(sessions.vehicle.nunique()),
            no_eligible_home=int((~v.has_eligible_home).sum()),daily_grid_MWh=float(p.sum()),
            daily_battery_MWh=float(v.steady_battery_charge_kWh.sum()*.01),
            charging_loss_MWh=float(v.steady_charging_loss_kWh.sum()*.01),
            electric_distance_km_10x=float(v.steady_electric_distance_km.sum()*10),
            fuel_distance_km_10x=float(v.steady_fuel_distance_km.sum()*10),
            electric_distance_share=float(v.steady_electric_distance_km.sum()/v.steady_total_distance_km.sum()),
            daily_grid_kWh_per_input_PHEV=float(p.sum()*1000/9370),
            peak_MW=float(p.max()),peak_hour=int(p.argmax()),max_iterations=int(v.iterations.max()))
        phev_records.append(ps)
        for share in [.8,1.]:
            name=f'BEV_Home{int(share*100)}_PHEV_home_only'
            source=build_schedules(trips,audit,share,seed,base_kWh_per_km=.224)
            bev=run_periodic([s for s in source if s.kind=='BEV'])
            assert bev['vehicles'].converged.all()
            combined=bev['profiles']['steady_BEV']*.01+p
            for phase in ['first','steady']:
                total=bev['profiles'][phase+'_feasible']*.01+phev['profiles'][phase+'_grid']*.01
                rows.append(metric(name,total,phase,seed))
            for hour in range(24):
                ensembles.append(dict(seed=seed,scenario=name,hour=hour,total_MW=float(combined[hour]),
                    BEV_MW=float(bev['profiles']['steady_BEV'][hour]*.01),PHEV_MW=float(p[hour])))
            if seed==42:
                folder=OUT/f'bev_home{int(share*100)}';folder.mkdir()
                prior=pd.read_csv(OLD/f'home{int(share*100)}'/'hourly_profiles.csv')
                np.testing.assert_allclose(bev['profiles']['steady_BEV']*.01,prior.steady_BEV_MW_10x,atol=1e-10)
                oldv=pd.read_parquet(OLD/f'home{int(share*100)}'/'vehicles.parquet')
                previous=oldv[oldv.powertrain.eq('BEV')].sort_values('vehicle').reset_index(drop=True)
                pd.testing.assert_frame_equal(previous,bev['vehicles'].sort_values('vehicle').reset_index(drop=True))
                for key in ['vehicles','sessions','trips']:
                    bev[key].to_parquet(folder/(key+'.parquet'),index=False)
                stored[share]=combined
                old_bev[share]=bev['profiles']['steady_BEV']*.01
            print(f'Completed {name}, seed {seed}: {combined.sum():.6f} MWh/day',flush=True)
        if seed==42:
            folder=OUT/'phev_home_only';folder.mkdir()
            for key in ['vehicles','sessions','trips','first_sessions','first_trips','history']:
                phev[key].to_parquet(folder/(key+'.parquet'),index=False)
            v.to_csv(folder/'vehicle_audit.csv',index=False)
            v.loc[~v.has_eligible_home].to_csv(folder/'no_eligible_home.csv',index=False)
            t.loc[t.fuel_distance_km>0].to_csv(folder/'fuel_fallback_trips.csv',index=False)
            pd.DataFrame({'hour':range(24),**{k+'_MW_10x':a*.01 for k,a in phev['profiles'].items()}}).to_csv(
                folder/'hourly_profiles.csv',index=False)
            main_phev=p.copy()
        print(json.dumps(ps),flush=True)
    stats=pd.DataFrame(rows);stats.to_csv(OUT/'scenario_metrics_all_seeds.csv',index=False)
    pd.DataFrame(phev_records).to_csv(OUT/'phev_metrics_all_seeds.csv',index=False)
    ensemble=pd.DataFrame(ensembles);ensemble.to_csv(OUT/'hourly_all_seeds.csv',index=False)
    curves=pd.DataFrame({'hour':range(24),'EVI20C_MW':old.EVI_20C_MW,'PHEV_home_only_MW':main_phev,
                         'PHEV_EVI20C_MW':old_parts.PHEV_EVI20C_MW})
    for share in [.8,1.]:
        name=f'Home{int(share*100)}'
        curves[name+'_new_total_MW']=stored[share]
        curves[name+'_BEV_unchanged_MW']=old_bev[share]
        curves[name+'_old_total_MW']=old[name+'_new0224_steady_MW']
    curves.to_csv(OUT/'comparison_hourly.csv',index=False)
    selected=stats[stats.seed.eq(42)].to_dict('records')
    selected.append(metric('EVI20C',old.EVI_20C_MW,'reference'))
    for share in [.8,1.]:
        name=f'Home{int(share*100)}'
        selected.append(metric(name+'_old_PHEV20proxy',curves[name+'_old_total_MW']))
    pd.DataFrame(selected).to_csv(OUT/'comparison_metrics.csv',index=False)

    plt.rcParams.update({'font.size':11})
    fig,axes=plt.subplots(3,1,figsize=(13,12),layout='constrained')
    lines=[('EVI 20 C: existing reference',curves.EVI20C_MW,'#d95f02','-.')]
    for name,color in [('Home80','#2166ac'),('Home100','#00876c')]:
        lines.extend([(f'BEV {name} + old PHEV 20% proxy',curves[name+'_old_total_MW'],color,'--'),
                      (f'BEV {name} + PHEV home-only SOC',curves[name+'_new_total_MW'],color,'-')])
    for label,p,color,style in lines:
        p=np.asarray(p)
        for ax,y in zip(axes,[p,p*1e6/15440,p/p.sum()*100]):
            ax.plot(range(24),y,label=label,color=color,ls=style,lw=2.2)
    for share,color in [(.8,'#2166ac'),(1.,'#00876c')]:
        g=ensemble[ensemble.scenario.eq(f'BEV_Home{int(share*100)}_PHEV_home_only')].groupby('hour').total_MW
        axes[0].fill_between(range(24),g.min(),g.max(),color=color,alpha=.12)
    for ax,title,unit in zip(axes,['Stable-day total: same 15,440 input PEVs','Per 1,000 input PEVs','Hourly energy share'],
        ['Hourly-average power (MW)','kW / 1,000 PEVs','Share of daily energy (%)']):
        ax.set(title=title,ylabel=unit,xlabel='Hour (events clock; no time shift)',xlim=(0,23),xticks=range(24),ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(ncol=1,fontsize=8,loc='upper left')
    fig.suptitle('PHEV home-only charging with SOC-driven electric / fuel use',fontsize=16)
    fig.supxlabel('PHEV: 0.202 battery kWh/km, 90% charging efficiency, 2 kWh reserve; no fixed 20% electric-use factor.\n'
        'BEV unchanged (previous 0.224 single-ledger sensitivity). Home80/Home100 now refer ONLY to BEVs.\n'
        'EVI unchanged: 20 C, 35 mi/day, Home80, matched subtype weights; not a matched home-only PHEV scenario.\n'
        'Seed 42 lines; shading = five-seed range. 10x sample expansion, no extra x2; no power-generation/AQ rerun.',fontsize=9)
    fig.savefig(OUT/'phev_home_soc_vs_evi20c.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,kind,n in zip(axes,['BEV','PHEV'],[6070,9370]):
        ax.plot(range(24),old_parts[kind+'_EVI20C_MW']*1e6/n,'-.',color='#d95f02',lw=2.4,label='EVI 20 C')
        if kind=='BEV':
            for share,color in [(.8,'#2166ac'),(1.,'#00876c')]:
                ax.plot(range(24),old_bev[share]*1e6/n,color=color,lw=2,label=f'BEV Home{int(share*100)} (unchanged)')
        else:
            ax.plot(range(24),main_phev*1e6/n,color='#7b3294',lw=2.4,label='New: home-only, SOC-driven')
            for name,color in [('Home80','#2166ac'),('Home100','#00876c')]:
                ax.plot(range(24),old_parts['PHEV_'+name+'_MW']*1e6/n,'--',color=color,label=name+' old 20% proxy')
        ax.set(title=kind,ylabel='kW / 1,000 input '+kind+'s',xlabel='Hour',xlim=(0,23),xticks=range(0,24,2),ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Separate BEV / PHEV response: only the PHEV algorithm changed',fontsize=15)
    fig.savefig(OUT/'bev_phev_home_soc_vs_evi20c.png',dpi=180);plt.close(fig)
    for p,h in protected.items():
        assert sha(p)==h, f'Prior result changed: {p}'
    for p in OUT.glob('*.png'):
        with Image.open(p) as im: im.verify()
    known=set(audit.vehicle.astype(str))
    missing=inventory[~inventory.vehicle.astype(str).isin(known)].to_dict('records')
    result=dict(status='complete',PHEV=dict(battery_kWh_per_km=.202,charging_efficiency=.9,
        reserve_kWh=2.,electric_share='Determined by chronological SOC, not a fixed multiplier',
        charging='Only every identified Home parking >15min, immediate to capacity or departure',
        home_access='Assumed at identified Home stops; no fallback to Work/public when Home is absent',
        power='Unchanged vehicle AC draw: 20% 1.9kW / 80% 7.2kW, now grid-side power',
        initial_SOC='Preserved original synthetic initial SOC; repeated days until daily SOC equilibrium',
        fuel='Residual trip distance driven on fuel; no fuel liters or combustion emissions computed',
        home_classification='Preserved baseline activity-derived Home labels, including prior proxy labels'),
        BEV='Unchanged previous 0.224 single-ledger sensitivity and Home80/Home100 strategies; seed42 exact parity verified',
        EVI_inputs=old_manifest['EVI_inputs'],EVI_reference=str(OLD),
        EVI_caveat='Fixed Home80 reference, not reconfigured to restrict PHEVs to home-only; range proxies retained',
        expanded_input_fleet=15440,modeled_sample_PHEVs=936,input_sample_PHEVs=937,
        missing_input_vehicles=missing,seed42_PHEV=phev_records[0],metrics=selected,
        random_seeds=SEEDS,all_previous_results_preserved=True,source_hashes={str(p):h for p,h in protected.items()},
        ready_for_power_generation=False,downstream_AQ_run=False,
        implementation='src/impacts/pipeline/workflow/charging/ldv_home_sensitivity.py')
    (OUT/'run_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(pd.DataFrame(selected).to_string(index=False))
    print('OUTPUT:',OUT)


if __name__=='__main__':
    main()
