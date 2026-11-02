"""Isolated home-priority scenarios, periodic SOC audit and unchanged EVI comparison."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from impacts.pipeline.workflow.charging.ldv_home_sensitivity import build_schedules,run_periodic

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'examples/pilates/impacts/impacts_output/events-fc07-existing-ev-charging-code10x-20260928-v2'
CONTROL=BASE/'phev20-sensitivity-20261012/phev20'
OUT=BASE/'home-priority-tests-20261012'
SEEDS=[42,7,21,84,126]


def metrics(result,share,seed,phase):
    p=result['profiles'][phase+'_feasible'];hp=result['profiles'][phase+'_home_feasible']
    v=result['vehicles'];valid=v[phase+'_included']
    rows=result['first_sessions' if phase=='first' else 'sessions']
    rows=rows[rows.included_in_profile]
    return dict(scenario=f'Home{int(share*100)}',seed=seed,phase=phase,
        daily_MWh=float(p.sum()*.01),peak_MW=float(p.max()*.01),peak_hour=int(p.argmax()),
        evening18_24_MWh=float(p[18:].sum()*.01),evening18_24_share=float(p[18:].sum()/p.sum()),
        home_energy_share=float(hp.sum()/p.sum()),charging_vehicles_sample=int(rows.vehicle.nunique()),
        included_vehicles_sample=int(valid.sum()),excluded_BEVs=int((~valid & v.powertrain.eq('BEV')).sum()),
        unmet_BEV_MWh=float(v.loc[v.powertrain.eq('BEV'),phase+'_shortfall_kWh'].sum()*.01),
        PHEV_fuel_substitution_electric_equiv_MWh=float(v.loc[v.powertrain.eq('PHEV'),phase+'_shortfall_kWh'].sum()*.01),
        SOC_gain_included_MWh=float(v.loc[valid,phase+'_SOC_gain_kWh'].sum()*.01),
        all_profiles_diagnostic_MWh=float(result['profiles'][phase+'_all'].sum()*.01),
        home_preferred_vehicles_sample=int(v.home_preferred.sum()),
        converged_vehicles=int(v.converged.sum()),max_iterations=int(v.iterations.max()))


def main():
    if OUT.exists() and any(OUT.iterdir()):raise FileExistsError(f'Preserve previous run: {OUT}')
    protected={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.rglob('*') if p.is_file()}
    OUT.mkdir(exist_ok=True)
    trips=pd.read_parquet(BASE/'existing_ev_trips.parquet')
    audit=pd.read_parquet(CONTROL/'ev_stop_charging_audit.parquet')
    control=pd.read_csv(CONTROL/'existing_ev_hourly_charging_curve_10x.csv')
    control_vehicles=pd.read_parquet(CONTROL/'ev_vehicle_soc_audit.parquet')
    control_sessions=pd.read_parquet(CONTROL/'modeled_ev_charging_sessions.parquet')
    evi=pd.read_csv(BASE/'evi-fleet-matched-20261012/comparison_hourly.csv')
    evi_mw=evi.evi_powertrain_range_proxy_matched_MW.to_numpy()
    records=[];results={};seed_profiles=[]
    for share in [.8,1.]:
        for seed in SEEDS:
            schedules=build_schedules(trips,audit,share,seed)
            result=run_periodic(schedules)
            if not result['vehicles'].converged.all():raise ValueError('SOC did not converge')
            for phase in ['first','steady']:records.append(metrics(result,share,seed,phase))
            for hour,p in enumerate(result['profiles']['steady_feasible']):
                seed_profiles.append(dict(home_share=share,seed=seed,hour=hour,steady_MW=p*.01))
            if seed==42:
                results[share]=result
                folder=OUT/f'home{int(share*100)}';folder.mkdir()
                for name in ['vehicles','sessions','trips','first_sessions','first_trips','history']:
                    result[name].to_parquet(folder/(name+'.parquet'),index=False)
                result['vehicles'].to_csv(folder/'vehicle_audit.csv',index=False)
                pd.DataFrame({'hour':range(24),**{k+'_MW_10x':v*.01 for k,v in result['profiles'].items()}}).to_csv(
                    folder/'hourly_profiles.csv',index=False)
                failed=result['vehicles'][~result['vehicles'].steady_included]
                failed.to_csv(folder/'excluded_infeasible_vehicles.csv',index=False)
                result['trips'].query('BEV_unmet_kWh > 0 or PHEV_fuel_substitution_electric_equiv_kWh > 0').to_csv(
                    folder/'unmet_and_fuel_substitution_trips.csv',index=False)
            print(f'Completed Home{int(share*100)} seed {seed}: '+json.dumps(records[-1]),flush=True)
    stats=pd.DataFrame(records);stats.to_csv(OUT/'scenario_metrics_all_seeds.csv',index=False)
    ensemble=pd.DataFrame(seed_profiles);ensemble.to_csv(OUT/'steady_hourly_all_seeds.csv',index=False)
    seedrange=stats[stats.phase.eq('steady')].groupby('scenario').agg(
        daily_MWh_min=('daily_MWh','min'),daily_MWh_max=('daily_MWh','max'),
        peak_MW_min=('peak_MW','min'),peak_MW_max=('peak_MW','max'),
        evening_share_min=('evening18_24_share','min'),evening_share_max=('evening18_24_share','max'),
        home_share_min=('home_energy_share','min'),home_share_max=('home_energy_share','max'))
    seedrange.to_csv(OUT/'seed_sensitivity_range.csv')
    # Check like-for-like population, in addition to the main full input denominator.
    common=set(control_vehicles.loc[control_vehicles.retained,'vehicle'])
    for result in results.values():common &= set(result['vehicles'].loc[result['vehicles'].steady_included,'vehicle'])
    common_rows=[dict(scenario='Original20_single_day',vehicles=len(common),
                     daily_MWh=control_sessions.loc[control_sessions.vehicle.isin(common),'energyCharged_kWh'].sum()*.01)]
    for share,result in results.items():
        common_rows.append(dict(scenario=f'Home{int(share*100)}_steady',vehicles=len(common),
            daily_MWh=result['sessions'].loc[result['sessions'].vehicle.isin(common),'energyCharged_kWh'].sum()*.01))
    pd.DataFrame(common_rows).to_csv(OUT/'common_cohort_energy_comparison.csv',index=False)
    selected=stats[(stats.seed==42)].copy()
    for scenario,p,home,drift in [('Original20_single_day',control.total_MW_10x.to_numpy(),
        control_sessions.loc[control_sessions.is_home,'energyCharged_kWh'].sum()/control_sessions.energyCharged_kWh.sum(),
        float((control_vehicles.loc[control_vehicles.retained,'final_soc_kwh']-
               control_vehicles.loc[control_vehicles.retained,'initial_soc_kwh']).sum()*.01)),
        ('EVI_range_proxy_reference',evi_mw,np.nan,np.nan)]:
        selected=pd.concat([selected,pd.DataFrame([dict(scenario=scenario,phase='reference',seed=42,
            daily_MWh=p.sum(),peak_MW=p.max(),peak_hour=int(p.argmax()),
            evening18_24_MWh=p[18:].sum(),evening18_24_share=p[18:].sum()/p.sum(),
            home_energy_share=home,SOC_gain_included_MWh=drift)])],ignore_index=True)
    selected.to_csv(OUT/'comparison_metrics.csv',index=False)
    curve=pd.DataFrame({'hour':range(24),'original20_single_day_MW':control.total_MW_10x,'EVI_matched_MW':evi_mw})
    for share,r in results.items():
        for phase in ['first','steady']:curve[f'home{int(share*100)}_{phase}_MW']=r['profiles'][phase+'_feasible']*.01
    curve.to_csv(OUT/'comparison_hourly.csv',index=False)
    plt.rcParams.update({'font.size':11})
    colors={.8:'#2166ac',1.:'#00876c'}
    fig,axes=plt.subplots(3,1,figsize=(13,12),layout='constrained')
    for i,ax in enumerate(axes):
        ref=control.total_MW_10x.to_numpy();reference=evi_mw
        if i==1:ref=ref/ref.sum()*100;reference=reference/reference.sum()*100
        ax.plot(range(24),ref,'--',color='#777777',label='Original PHEV20: single day, SOC drift',lw=1.8)
        ax.plot(range(24),reference,':',color='#d95f02',label='EVI fleet/range-proxy matched',lw=2.5)
        for share,r in results.items():
            phase='first' if i==2 else 'steady';p=r['profiles'][phase+'_feasible']*.01
            if i==1:p=p/p.sum()*100
            ax.plot(range(24),p,color=colors[share],lw=2.3,label=f'Home priority {int(share*100)}% ({phase} day)')
            if i==0:
                g=ensemble[ensemble.home_share.eq(share)].groupby('hour').steady_MW
                ax.fill_between(range(24),g.min(),g.max(),color=colors[share],alpha=.12)
        ax.set(xlim=(0,23),xticks=range(24),ylim=(0,None),xlabel='Hour (events clock; no time shift)')
        ax.grid(alpha=.2);ax.legend(fontsize=9,ncol=2)
    axes[0].set(title='Stable daily load after SOC convergence (shading: five-seed range)',ylabel='Hourly-average MW')
    axes[1].set(title='Stable-day shape: each curve normalized by its own daily energy',ylabel='Share of daily energy (%)')
    axes[2].set(title='First day: same initial SOC as original control',ylabel='Hourly-average MW')
    fig.suptitle('Home-priority charging tests: existing BEV + PHEV fleet',fontsize=17)
    fig.supxlabel('All curves represent 15,440 input PEVs (10x sample); unmodelled/infeasible vehicles remain in denominator.\n'
        'PHEV electricity factor 20%; AC 1.9/7.2 kW; PHEV DC disabled; BEV necessary top-ups retained.\n'
        'Scenario tests, not local validation. Infeasible BEV schedules excluded and audited; PHEV fuel substitution reported.',fontsize=9)
    fig.savefig(OUT/'home_priority_vs_evi.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    evi_parts=pd.read_csv(BASE/'evi-fleet-matched-20261012/powertrain_comparison_hourly.csv')
    for ax,kind,n in zip(axes,['BEV','PHEV'],[6070,9370]):
        ax.plot(range(24),control[kind+'_MW_10x']*1e6/n,'--',color='#777777',label='Original20: single day')
        ax.plot(range(24),evi_parts[kind+'_EVI_kW_per1000'],':',color='#d95f02',lw=2.4,label='EVI matched')
        for share,r in results.items():
            ax.plot(range(24),r['profiles']['steady_'+kind]*.01*1e6/n,
                    color=colors[share],lw=2,label=f'Home{int(share*100)} stable')
        ax.set(title=kind,ylabel=f'kW / 1,000 input {kind}s',xlabel='Hour',xlim=(0,23),xticks=range(0,24,2),ylim=(0,None))
        ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.suptitle('Powertrain-specific home-priority response',fontsize=15)
    fig.savefig(OUT/'home_priority_bev_phev.png',dpi=180);plt.close(fig)
    assumptions=dict(PHEV_energy_factor=.2,BEV_kWh_per_km=.17,PHEV_kWh_per_km=.034,
        home_access='Assumed for vehicles with identified terminal-home stop; other vehicles reported separately',
        preference='Exactly floor(share*N) in each powertrain home-eligible group; nested hash-ranked cohorts',
        primary_home='Terminal trip arrival at identified Home with >15min parking until next-day first departure',
        alternative_primary='Longest Work/Education stop, otherwise longest non-home, otherwise eligible Home',
        home_target_SOC_fraction=1.,reserve_kWh=2.,PHEV_DC_allowed=False,
        BEV_emergency='At eligible stops, minimum top-up toward next primary charging opportunity; capped by capacity/dwell',
        PHEV_shortfall='Additional fuel-substitution electric-energy equivalent reported; no gasoline emissions calculation',
        AC_power='Fixed per vehicle hash draw: 20% L1 1.9kW / 80% L2 7.2kW at primary/home stops',
        nonhome_emergency_BEV_kW=50.,charger_losses='No extra losses, matching charging code baseline',
        start_SOC='Same original seed42 initial SOC for every scenario/assignment seed; carried between repeated days',
        daily_cycle='Original <29h event window; repeat each vehicle first-departure-to-next-first-departure itinerary',
        convergence='Max absolute per-vehicle daily SOC change <1e-7 kWh, up to 2000 numerical repeated days; not a calendar forecast',
        failed_BEV='Never retroactively repaired by arrival charging; entire failed vehicle curve excluded with audit',
        random_seeds=SEEDS,input_fleet=1544,modeled_fleet=1543,traffic_expansion=10,
        calibration='No optimization/fitting against EVI; EVI used only after computing charging',
        caution='Bundled algorithm changes, not a single-parameter causal experiment; actual local charger access unknown')
    manifest=dict(status='complete',assumptions=assumptions,
        seed42_metrics=selected.astype(object).where(pd.notna(selected),None).to_dict('records'),
        seed_ranges=seedrange.reset_index().to_dict('records'),common_cohort=common_rows,
        incomplete_input_vehicles=sorted(set(pd.read_parquet(BASE/'existing_ev_inventory.parquet').vehicle)-set(audit.vehicle)),
        ready_for_power_generation=False,downstream='Not run: revised PHEV fuel use, incomplete fleet and proxy assumptions need review',
        algorithm_file=str(ROOT/'src/impacts/pipeline/workflow/charging/ldv_home_sensitivity.py'))
    for p,sha in protected.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,f'Old result modified: {p}'
    manifest['all_prior_results_preserved']=True
    (OUT/'run_summary.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:im.verify()
    print(selected.to_string(index=False))
    print(seedrange.to_string())


if __name__=='__main__':main()
