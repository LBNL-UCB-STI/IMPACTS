"""Re-simulate home-priority demand with 0.224 kWh/km; fetch 20 C EVI.

This deliberately tests a direct coefficient substitution in the existing
single-ledger model. It does not add an extra charging-efficiency correction.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import match_baseline_evi_profiles as evi
from test_home_priority_scenarios import metrics
from impacts.pipeline.workflow.home_priority_charging import build_schedules, run_periodic

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'examples/pilates/impacts/impacts_output/events-fc07-existing-ev-charging-code10x-20260928-v2'
PREVIOUS = BASE/'home-priority-tests-20261012'
MATCHED = BASE/'evi-fleet-matched-20261012'
OUT = BASE/'home-priority-0224-evi20c-20261012'
SEEDS = [42, 7, 21, 84, 126]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fetch-only', action='store_true')
    args = parser.parse_args()
    if (OUT/'run_summary.json').exists() or (OUT/'home80').exists():
        raise FileExistsError(f'Preserve previous simulation: {OUT}')
    OUT.mkdir(exist_ok=True)
    old_manifest = json.loads((MATCHED/'matching_summary.json').read_text())
    inputs = {**old_manifest['EVI_inputs'], 'temp_c': 20}
    evi.OUT = OUT
    breakdown = evi.fetch(evi.API+'/daily-load-profile/breakdown.json', inputs, 'evi20c_breakdown.json')
    aggregate = evi.fetch(evi.API+'/daily-load-profile.json', inputs, 'evi20c_aggregate.json')
    parts = evi.unpack(breakdown['response'])
    totals = aggregate['response']['results']['weekday_load_profile']
    errors = {}
    for c in evi.CHARGERS:
        np.testing.assert_allclose(sum(parts[c].values()), totals[c], atol=.046, rtol=0)
        errors[c] = float(np.max(np.abs(sum(parts[c].values())-totals[c])))
    targets = old_manifest['range_proxy_targets']
    matched = evi.reweight(parts, targets)
    evi20 = evi.hourly(evi.total(matched))/1000
    print(f'EVI 20 C verified: {evi20.sum():.6f} MWh/day; {evi20.max():.6f} MW peak', flush=True)
    if args.fetch_only:
        return
    protected = {p: sha(p) for folder in [PREVIOUS, MATCHED, BASE/'phev20-sensitivity-20261012']
                 for p in folder.rglob('*') if p.is_file()}
    input_files = [BASE/'existing_ev_trips.parquet', BASE/'existing_ev_inventory.parquet',
                   BASE/'phev20-sensitivity-20261012/phev20/ev_stop_charging_audit.parquet']
    protected.update({p: sha(p) for p in input_files})
    trips = pd.read_parquet(input_files[0])
    inventory = pd.read_parquet(input_files[1])
    audit = pd.read_parquet(input_files[2])
    fleet = len(inventory)*10
    assert fleet == 15440 and sum(targets.values()) == fleet
    counts = inventory.powertrain.value_counts()*10
    old_curves = pd.read_csv(PREVIOUS/'comparison_hourly.csv')
    records, profiles, results = [], [], {}
    for share in [.8, 1.]:
        for seed in SEEDS:
            schedules = build_schedules(trips, audit, share, seed, base_kWh_per_km=.224)
            result = run_periodic(schedules)
            if not result['vehicles'].converged.all():
                raise ValueError(f'Failed SOC convergence: Home{share}, seed{seed}')
            for phase in ['first', 'steady']:
                records.append(metrics(result, share, seed, phase))
            for hour, power in enumerate(result['profiles']['steady_feasible']):
                profiles.append(dict(home_share=share, seed=seed, hour=hour, steady_MW=power*.01))
            if seed == 42:
                results[share] = result
                folder = OUT/f'home{int(share*100)}'
                folder.mkdir()
                for key in ['vehicles', 'sessions', 'trips', 'first_sessions', 'first_trips', 'history']:
                    result[key].to_parquet(folder/(key+'.parquet'), index=False)
                result['vehicles'].to_csv(folder/'vehicle_audit.csv', index=False)
                result['vehicles'].loc[~result['vehicles'].steady_included].to_csv(
                    folder/'excluded_infeasible_vehicles.csv', index=False)
                result['trips'].query('BEV_unmet_kWh > 0 or PHEV_fuel_substitution_electric_equiv_kWh > 0').to_csv(
                    folder/'unmet_and_fuel_substitution_trips.csv', index=False)
                pd.DataFrame({'hour':range(24), **{k+'_MW_10x': v*.01 for k,v in result['profiles'].items()}}).to_csv(
                    folder/'hourly_profiles.csv', index=False)
                np.testing.assert_allclose(result['vehicles'].loc[result['vehicles'].steady_included,'steady_SOC_gain_kWh'],
                                           0, atol=1e-7)
            print(json.dumps(records[-1]), flush=True)
    stats = pd.DataFrame(records)
    stats.to_csv(OUT/'scenario_metrics_all_seeds.csv', index=False)
    ensemble = pd.DataFrame(profiles)
    ensemble.to_csv(OUT/'steady_hourly_all_seeds.csv', index=False)
    selected = stats[stats.seed.eq(42)].copy()
    curves = pd.DataFrame({'hour':range(24), 'EVI_10C_MW':old_curves.EVI_matched_MW, 'EVI_20C_MW':evi20})
    for share, result in results.items():
        name = f'Home{int(share*100)}'
        curves[name+'_old017_MW'] = old_curves[f'{name.lower()}_steady_MW']
        for phase in ['first','steady']:
            curves[name+'_new0224_'+phase+'_MW'] = result['profiles'][phase+'_feasible']*.01
    curves.to_csv(OUT/'comparison_hourly.csv', index=False)
    for name, p in [('EVI_10C',curves.EVI_10C_MW),('EVI_20C',evi20)]:
        selected = pd.concat([selected, pd.DataFrame([dict(scenario=name,phase='reference',seed=42,
            daily_MWh=float(np.sum(p)),peak_MW=float(np.max(p)),peak_hour=int(np.argmax(p)),
            evening18_24_MWh=float(np.sum(p[18:])),evening18_24_share=float(np.sum(p[18:])/np.sum(p)))])],ignore_index=True)
    old_stats = pd.read_csv(PREVIOUS/'comparison_metrics.csv')
    old_stats = old_stats[(old_stats.phase.eq('steady'))&old_stats.scenario.isin(['Home80','Home100'])].copy()
    old_stats['scenario'] += '_old017'
    selected = pd.concat([selected,old_stats],ignore_index=True)
    selected.to_csv(OUT/'comparison_metrics.csv',index=False)
    quarter = pd.DataFrame({'hour_start':np.arange(96)/4})
    for charger, subtype in matched.items():
        for kind, power in subtype.items():
            quarter[charger+'_'+kind+'_kW'] = power
    quarter.to_csv(OUT/'evi20c_matched_15min_components.csv',index=False)
    powertrain = pd.DataFrame({'hour':range(24)})
    ptr_records = []
    for kind in ['BEV','PHEV']:
        ref = evi.hourly(evi.total(matched,kind.lower()))/1000
        powertrain[kind+'_EVI20C_MW'] = ref
        for share,result in results.items():
            name = f'Home{int(share*100)}'
            p = result['profiles']['steady_'+kind]*.01
            powertrain[kind+'_'+name+'_MW'] = p
            ptr_records.append(dict(scenario=name,powertrain=kind,input_fleet=int(counts[kind]),
                daily_MWh=float(p.sum()),EVI20C_daily_MWh=float(ref.sum()),
                model_kWh_per_vehicle=float(p.sum()*1000/counts[kind]),
                EVI20C_kWh_per_vehicle=float(ref.sum()*1000/counts[kind])))
    powertrain.to_csv(OUT/'powertrain_comparison_hourly.csv',index=False)
    pd.DataFrame(ptr_records).to_csv(OUT/'powertrain_metrics.csv',index=False)
    # A common feasible cohort separates energy-coefficient effects from exclusions.
    common = set(audit.vehicle.astype(str))
    for share,r in results.items():
        common &= set(r['vehicles'].loc[r['vehicles'].steady_included,'vehicle'].astype(str))
        old = pd.read_parquet(PREVIOUS/f'home{int(share*100)}'/'vehicles.parquet')
        common &= set(old.loc[old.steady_included,'vehicle'].astype(str))
    cohort = []
    for share,r in results.items():
        name = f'Home{int(share*100)}'
        old = pd.read_parquet(PREVIOUS/name.lower()/'vehicles.parquet')
        for label,v in [('old017',old),('new0224',r['vehicles'])]:
            cohort.append(dict(scenario=name,energy=label,vehicles=len(common),
                daily_MWh=float(v.loc[v.vehicle.astype(str).isin(common),'steady_charge_kWh'].sum()*.01)))
    pd.DataFrame(cohort).to_csv(OUT/'common_cohort_comparison.csv',index=False)

    plt.rcParams.update({'font.size':11})
    fig, axes = plt.subplots(3,1,figsize=(13,12),layout='constrained')
    series = [('EVI 10 C (previous)',curves.EVI_10C_MW,'#999999',':'),
              ('EVI 20 C (new)',evi20,'#d95f02','-.')]
    for name,color in [('Home80','#2166ac'),('Home100','#00876c')]:
        series.extend([(name+' old 0.170',curves[name+'_old017_MW'],color,'--'),
                       (name+' new 0.224',curves[name+'_new0224_steady_MW'],color,'-')])
    for label,p,color,style in series:
        p=np.asarray(p)
        for ax,y in zip(axes,[p,p*1e6/fleet,p/p.sum()*100]):
            ax.plot(range(24),y,label=label,color=color,ls=style,lw=2)
    for share,color in [(.8,'#2166ac'),(1.,'#00876c')]:
        g=ensemble[ensemble.home_share.eq(share)].groupby('hour').steady_MW
        axes[0].fill_between(range(24),g.min(),g.max(),color=color,alpha=.12)
    for ax,title,unit in zip(axes,['Stable-day total load: same 15,440 input PEVs',
        'Load per 1,000 input PEVs','Hourly shape normalized by each daily total'],
        ['Hourly-average power (MW)','kW / 1,000 PEVs','Share of daily energy (%)']):
        ax.set(title=title,ylabel=unit,xlabel='Hour (events clock; no time shift)',xlim=(0,23),xticks=range(24),ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=9)
    fig.suptitle('Charging-code energy 0.224 kWh/km vs EVI-Pro Lite at 20 C',fontsize=16)
    fig.supxlabel('Direct single-ledger coefficient sensitivity, no additional charging-loss correction. PHEV factor remains 20%.\n'
        'Weekday EVI: 35 mi/day, Home80, original subtype weights retained. Actual sample expansion 10x; no extra x2.\n'
        'Seed 42 lines; shading = five-seed range. Infeasible BEVs excluded and audited; no power-generation/AQ rerun.',fontsize=9)
    fig.savefig(OUT/'charging_0224_vs_evi20c.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,kind in zip(axes,['BEV','PHEV']):
        n=counts[kind]
        ax.plot(range(24),powertrain[kind+'_EVI20C_MW']*1e6/n,'-.',color='#d95f02',lw=2.4,label='EVI 20 C')
        for name,color in [('Home80','#2166ac'),('Home100','#00876c')]:
            old=pd.read_csv(PREVIOUS/name.lower()/'hourly_profiles.csv')
            ax.plot(range(24),old['steady_'+kind+'_MW_10x']*1e6/n,'--',color=color,label=name+' old 0.170')
            ax.plot(range(24),powertrain[kind+'_'+name+'_MW']*1e6/n,color=color,lw=2,label=name+' new 0.224')
        ax.set(title=kind,ylabel='kW / 1,000 input '+kind+'s',xlabel='Hour',xlim=(0,23),xticks=range(0,24,2),ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('BEV / PHEV comparison: 0.224 base rate; PHEV factor remains 20%',fontsize=15)
    fig.savefig(OUT/'bev_phev_0224_vs_evi20c.png',dpi=180);plt.close(fig)
    for p,digest in protected.items():
        assert sha(p)==digest, f'Prior file changed: {p}'
    for p in OUT.glob('*.png'):
        with Image.open(p) as im: im.verify()
    manifest=dict(status='complete',base_kWh_per_km=.224,PHEV_energy_factor=.2,PHEV_effective_kWh_per_km=.0448,
        energy_accounting='Direct single-ledger replacement: 0.224 used for SOC depletion and charging, eta=1; not separated grid/battery physics',
        caveat='0.224 was motivated by grid-side energy; treating it as SOC depletion is a sensitivity test, not a validated efficiency implementation',
        EVI_inputs=inputs,EVI_range_targets_unchanged=targets,expanded_fleet=fleet,random_seeds=SEEDS,
        no_extra_x2=True,all_previous_results_preserved=True,source_hashes={str(p):h for p,h in protected.items()},
        max_EVI_component_sum_errors_kW=errors,ready_for_power_generation=False,downstream_AQ_run=False,
        seed42_metrics=selected.astype(object).where(pd.notna(selected),None).to_dict('records'),
        powertrain_metrics=ptr_records,common_cohort=cohort)
    (OUT/'run_summary.json').write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    print(selected[['scenario','phase','daily_MWh','peak_MW','peak_hour','excluded_BEVs']].to_string(index=False))
    print(pd.DataFrame(ptr_records).to_string(index=False))
    print('OUTPUT:',OUT)


if __name__=='__main__':
    main()
