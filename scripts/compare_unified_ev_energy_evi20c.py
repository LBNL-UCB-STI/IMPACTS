"""Recompute existing-EV charging with common battery consumption and efficiency.

No power generation or air-quality execution. Prior artifacts are read-only.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from compare_charging_0224_evi20c import sha
from compare_phev_home_soc_evi20c import BASE, SEEDS, metric
from impacts.pipeline.workflow.charging.ldv_home_sensitivity import (
    build_schedules,
    run_periodic,
    build_phev_home_schedules,
    run_phev_periodic,
    BATTERY_KWH_PER_KM,
    CHARGING_EFFICIENCY,
)

OLD=BASE/'phev-home-only-soc-evi20c-20261012'
REFERENCE=BASE/'home-priority-0224-evi20c-20261012'
OUT=BASE/'unified-ev-energy-evi20c-20261012'
SCALE=10


def save_details(result, folder, bev=False):
    folder.mkdir()
    for key in ['vehicles','sessions','trips','first_sessions','first_trips','history']:
        result[key].to_parquet(folder/(key+'.parquet'),index=False)
    result['vehicles'].to_csv(folder/'vehicle_audit.csv',index=False)
    if bev:
        result['vehicles'].loc[~result['vehicles'].steady_included].to_csv(
            folder/'excluded_vehicles.csv',index=False)
        hourly={'hour':range(24)}
        for key, values in result['profiles'].items():
            hourly[key+'_grid_MW_10x']=values*SCALE/1000
            hourly[key+'_battery_MW_10x']=values*SCALE/1000*CHARGING_EFFICIENCY
    else:
        hourly={'hour':range(24),**{k+'_MW_10x':v*SCALE/1000 for k,v in result['profiles'].items()}}
    pd.DataFrame(hourly).to_csv(folder/'hourly_profiles.csv',index=False)


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserve previous results: {OUT}')
    inputs=[BASE/'existing_ev_trips.parquet',BASE/'existing_ev_inventory.parquet',
            BASE/'phev20-sensitivity-20261012/phev20/ev_stop_charging_audit.parquet']
    protected={p:sha(p) for folder in [OLD,REFERENCE] for p in folder.rglob('*') if p.is_file()}
    protected.update({p:sha(p) for p in inputs})
    trips,inventory,audit=[pd.read_parquet(p) for p in inputs]
    assert len(inventory)==1544
    previous=pd.read_csv(OLD/'comparison_hourly.csv')
    previous_seeds=pd.read_csv(OLD/'hourly_all_seeds.csv')
    evi_parts=pd.read_csv(REFERENCE/'powertrain_comparison_hourly.csv')
    evi_manifest=json.loads((REFERENCE/'run_summary.json').read_text())
    assert evi_manifest['EVI_inputs']['temp_c']==20
    OUT.mkdir()
    allmetrics=[];hourly=[];audits=[];curves=pd.DataFrame({'hour':range(24),'EVI20C_MW':previous.EVI20C_MW})
    eta=CHARGING_EFFICIENCY
    for seed in SEEDS:
        phev=run_phev_periodic(build_phev_home_schedules(trips,audit,seed))
        p=phev['profiles']['steady_grid']*SCALE/1000
        prior=previous_seeds.loc[(previous_seeds.seed==seed)&
            previous_seeds.scenario.eq('BEV_Home100_PHEV_home_only')].sort_values('hour')
        np.testing.assert_allclose(p,prior.PHEV_MW,atol=1e-10,rtol=0)
        assert phev['sessions'].is_home.all()
        if seed==42:
            # Exact parity: no PHEV algorithm/parameters changed in this run.
            for key in ['vehicles','sessions','trips']:
                pd.testing.assert_frame_equal(phev[key],pd.read_parquet(OLD/'phev_home_only'/(key+'.parquet')))
            save_details(phev,OUT/'phev_home_only')
            curves['PHEV_MW']=p
            curves['PHEV_EVI20C_MW']=evi_parts.PHEV_EVI20C_MW
            curves['BEV_EVI20C_MW']=evi_parts.BEV_EVI20C_MW
        for share in [.8,1.]:
            name=f'Home{int(share*100)}'
            schedules=[s for s in build_schedules(trips,audit,share,seed,
                base_kWh_per_km=BATTERY_KWH_PER_KM) if s.kind=='BEV']
            bev=run_periodic(schedules,efficiency=eta)
            v=bev['vehicles'];sessions=bev['sessions']
            assert v.converged.all() and len(v)==607
            np.testing.assert_allclose(v.steady_grid_charge_kWh*eta,
                                      v.steady_traction_kWh+v.steady_SOC_gain_kWh,atol=1e-8,rtol=0)
            np.testing.assert_allclose(sessions.grid_energy_kWh*eta,sessions.battery_energy_kWh,atol=1e-8)
            np.testing.assert_allclose(sessions.soc_after_kWh-sessions.soc_before_kWh,
                                      sessions.battery_energy_kWh,atol=1e-8)
            assert (sessions.timeCharging_s<=sessions.parkingTime_s+1e-8).all()
            included=v.loc[v.steady_included]
            np.testing.assert_allclose(bev['profiles']['steady_BEV'].sum(),included.steady_grid_charge_kWh.sum(),atol=1e-8)
            b=bev['profiles']['steady_BEV']*SCALE/1000
            total=b+p
            for phase in ['first','steady']:
                profile=bev['profiles'][phase+'_feasible']*SCALE/1000+phev['profiles'][phase+'_grid']*SCALE/1000
                allmetrics.append(metric(name,profile,phase,seed))
            allmetrics.append(metric(name+'_BEV',b,'steady',seed))
            record=dict(seed=seed,scenario=name,BEV_input=607,BEV_included=int(v.steady_included.sum()),
                BEV_excluded=int((~v.steady_included).sum()),BEV_grid_MWh=float(b.sum()),
                BEV_battery_MWh=float(b.sum()*eta),BEV_loss_MWh=float(b.sum()*(1-eta)),
                BEV_peak_hour=int(b.argmax()),BEV_peak_MW=float(b.max()),
                BEV_excluded_unmet_battery_kWh_sample=float(v.loc[~v.steady_included].steady_shortfall_kWh.sum()),
                BEV_max_iterations=int(v.iterations.max()),PHEV_grid_MWh=float(p.sum()),
                total_MWh=float(total.sum()))
            audits.append(record)
            for h in range(24):
                hourly.append(dict(seed=seed,scenario=name,hour=h,total_MW=float(total[h]),BEV_MW=float(b[h]),PHEV_MW=float(p[h])))
            if seed==42:
                # Legacy default must still reproduce saved BEV results exactly.
                legacy=run_periodic([s for s in build_schedules(trips,audit,share,seed,
                    base_kWh_per_km=.224) if s.kind=='BEV'])
                for key in ['vehicles','sessions','trips']:
                    pd.testing.assert_frame_equal(legacy[key],pd.read_parquet(OLD/f'bev_{name.lower()}'/(key+'.parquet')))
                save_details(bev,OUT/f'bev_{name.lower()}',bev=True)
                curves[name+'_total_MW']=total
                curves[name+'_BEV_MW']=b
                curves[name+'_previous_total_MW']=previous[name+'_new_total_MW']
                curves[name+'_previous_BEV_MW']=previous[name+'_BEV_unchanged_MW']
                # Distinguish changed accounting from changed feasible-vehicle membership.
                common=set(included.vehicle)&set(legacy['vehicles'].loc[legacy['vehicles'].steady_included,'vehicle'])
                record['common_cohort_BEVs']=len(common)
                record['common_old_grid_MWh']=float(legacy['vehicles'].loc[legacy['vehicles'].vehicle.isin(common)].steady_charge_kWh.sum()*.01)
                record['common_new_grid_MWh']=float(v.loc[v.vehicle.isin(common)].steady_grid_charge_kWh.sum()*.01)
            print(json.dumps(record),flush=True)
        allmetrics.append(metric('PHEV_home_only',p,'steady',seed))

    allmetrics=pd.DataFrame(allmetrics);ensemble=pd.DataFrame(hourly)
    allmetrics.to_csv(OUT/'scenario_metrics_all_seeds.csv',index=False)
    ensemble.to_csv(OUT/'hourly_all_seeds.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'energy_audit.csv',index=False)
    curves.to_csv(OUT/'comparison_hourly.csv',index=False)
    selected=allmetrics.loc[(allmetrics.seed==42)&allmetrics.phase.eq('steady')].to_dict('records')
    for label,column in [('EVI20C','EVI20C_MW'),('EVI20C_BEV','BEV_EVI20C_MW'),('EVI20C_PHEV','PHEV_EVI20C_MW'),
                          ('Previous_Home80','Home80_previous_total_MW'),('Previous_Home100','Home100_previous_total_MW')]:
        selected.append(metric(label,curves[column],'reference'))
    pd.DataFrame(selected).to_csv(OUT/'comparison_metrics.csv',index=False)

    plt.rcParams.update({'font.size':11})
    colors={'Home80':'#2166ac','Home100':'#00876c'}
    fig,axes=plt.subplots(3,1,figsize=(13,12),layout='constrained')
    lines=[('EVI 20 C: unchanged reference',curves.EVI20C_MW,'#d95f02','-.')]
    for name,color in colors.items():
        lines.extend([(f'{name}: previous BEV accounting',curves[name+'_previous_total_MW'],color,'--'),
                      (f'{name}: unified BEV/PHEV accounting',curves[name+'_total_MW'],color,'-')])
    for label,p,color,style in lines:
        p=np.asarray(p)
        for ax,y in zip(axes,[p,p*1e6/15440,p/p.sum()*100]):
            ax.plot(range(24),y,label=label,color=color,ls=style,lw=2.2)
    for name,color in colors.items():
        group=ensemble.loc[ensemble.scenario.eq(name)].groupby('hour').total_MW
        axes[0].fill_between(range(24),group.min(),group.max(),color=color,alpha=.12)
    for ax,title,unit in zip(axes,['Stable-day total: 15,440 input PEVs (10x)','Per 1,000 input PEVs','Hourly energy share'],
        ['Hourly-average grid power (MW)','kW / 1,000 PEVs','Share of daily energy (%)']):
        ax.set(title=title,ylabel=unit,xlabel='Hour (events clock; no time shift)',xlim=(0,23),xticks=range(24),ylim=(0,None))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Existing-EV charging: unified battery consumption and charging efficiency',fontsize=15)
    fig.supxlabel('Both: 0.202 battery kWh/km, 90% charging efficiency; grid equivalent 0.22444 kWh/electric-km.\n'
        'PHEV: home-only, SOC-driven fuel fallback. Home80/Home100 apply to BEV preference only.\n'
        'EVI: 20 C, 35 mi/day, Home80, matched subtype weights; not matched to PHEV home-only restrictions.\n'
        'Seed 42; shaded five-seed range. Infeasible BEVs excluded and audited; no AQ/power-generation rerun.',fontsize=9)
    fig.savefig(OUT/'unified_ev_vs_evi20c.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,kind,n in zip(axes,['BEV','PHEV'],[6070,9370]):
        ax.plot(range(24),curves[kind+'_EVI20C_MW']*1e6/n,'-.',color='#d95f02',lw=2.4,label='EVI 20 C reference')
        if kind=='BEV':
            for name,color in colors.items():
                ax.plot(range(24),curves[name+'_BEV_MW']*1e6/n,color=color,lw=2.2,label=name+': unified')
                ax.plot(range(24),curves[name+'_previous_BEV_MW']*1e6/n,'--',color=color,lw=1.4,label=name+': previous')
        else:
            ax.plot(range(24),curves.PHEV_MW*1e6/n,color='#7b3294',lw=2.4,label='Home-only SOC (unchanged)')
        ax.set(title=kind,ylabel='Grid kW / 1,000 input '+kind+'s',xlabel='Hour',xlim=(0,23),xticks=range(0,24,2),ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('BEV / PHEV: same 0.202 battery kWh/km and 90% charging efficiency',fontsize=14)
    fig.savefig(OUT/'bev_phev_unified_vs_evi20c.png',dpi=180);plt.close(fig)

    for path,digest in protected.items():
        assert sha(path)==digest, f'Protected file changed: {path}'
    for path in OUT.glob('*.png'):
        with Image.open(path) as im: im.verify()
    summary=dict(status='complete',battery_kWh_per_km=BATTERY_KWH_PER_KM,charging_efficiency=eta,
        equivalent_grid_kWh_per_electric_km=BATTERY_KWH_PER_KM/eta,reserve_kWh=2.,sample_scale=SCALE,
        input_PEVs=1544,expanded_input_PEVs=15440,modeled_BEVs=607,modeled_PHEVs=936,
        missing_PHEV_itinerary=1,PHEV_exact_prior_parity=True,legacy_default_exact_prior_parity=True,
        BEV_policy='Unchanged Home80/Home100 primary preference and necessary AC/home or 50kW/nonhome topups',
        PHEV_policy='Unchanged: only identified Home stops >15min, SOC-driven fuel fallback, no fixed proxy',
        initial_SOC='Same synthetic initial SOC; repeat representative day to equilibrium, do not reset daily',
        main_profile='Steady-day grid power; infeasible BEVs excluded as whole vehicles, see audits',
        EVI_inputs=evi_manifest['EVI_inputs'],EVI_reference=str(REFERENCE),
        caveats=['EVI is an unchanged reference, not observations or a matched home-only PHEV scenario',
            'EVI range-proxy weights retained to isolate energy-accounting change',
            'No time shift or post-hoc magnitude/profile calibration',
            'Home labels and charging access assumptions retained, not measured charger availability',
            'Fuel fallback distances are not modeled fuel emissions'],
        seed42_metrics=selected,BEV_audits=audits,random_seeds=SEEDS,
        all_previous_results_preserved=True,source_hashes={str(p):h for p,h in protected.items()},
        code_hashes={str(p):sha(p) for p in [Path(__file__),Path('src/impacts/pipeline/workflow/charging/ldv_home_sensitivity.py'),
                                           Path('src/impacts/pipeline/workflow/charging/ldv_home_sensitivity.py')]},
        downstream_AQ_run=False,power_generation_run=False)
    (OUT/'run_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(pd.DataFrame(selected).to_string(index=False))
    print('OUTPUT:',OUT)


if __name__=='__main__':
    main()
