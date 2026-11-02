"""Reproduce baseline, rerun PHEV20, compare against unchanged EVI reference."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'examples/pilates/impacts/impacts_output/events-fc07-existing-ev-charging-code10x-20260928-v2'
OUT = BASE/'phev20-sensitivity-20261012'


def metrics(label, frame, summary):
    p = frame.total_MW_10x.to_numpy()
    ledger = summary['powertrain_energy_soc']
    sessions = pd.read_parquet(Path(summary['_folder'])/'modeled_ev_charging_sessions.parquet')
    return dict(scenario=label, daily_charging_MWh=p.sum(), peak_MW=p.max(), peak_hour=int(p.argmax()),
        charging_vehicles_sample=summary['actual_charging_vehicles'],
        retained_vehicles_sample=summary['retained_vehicles'],
        daily_traction_MWh=sum(x['daily_traction_MWh_10x'] for x in ledger),
        SOC_gain_MWh=sum(x['SOC_gain_MWh_10x'] for x in ledger),
        home_energy_share=sessions.loc[sessions.is_home,'energyCharged_kWh'].sum()/sessions.energyCharged_kWh.sum(),
        evening18_24_energy_share=p[18:].sum()/p.sum(),
        BEV_charging_MWh=frame.BEV_MW_10x.sum(), PHEV_charging_MWh=frame.PHEV_MW_10x.sum(),
        ready_for_power_generation=summary['ready_for_power_generation'])


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserve existing sensitivity results: {OUT}')
    protected = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.iterdir() if p.is_file()}
    original = json.loads((BASE/'charging_code_summary.json').read_text())
    OUT.mkdir()
    # Each full run reads original events and reconstructs parking, charging choices,
    # SOC and durations. No scalar multiplication of an existing charging curve.
    for name,factor in [('control-phev10',.1),('phev20',.2)]:
        command = [sys.executable,'-X','utf8',str(ROOT/'scripts/run_existing_ev_charging_code.py'),
                   '--events',original['input_events'],'--vehicle-types',original['assumptions']['battery_source'],
                   '--phev-electricity-factor',str(factor),'--output',str(OUT/name)]
        with (OUT/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'Completed {name}',flush=True)
        if name=='control-phev10':
            for filename in ['existing_ev_hourly_charging_curve_10x.csv','powertrain_energy_soc_summary.csv']:
                pd.testing.assert_frame_equal(pd.read_csv(BASE/filename),pd.read_csv(OUT/name/filename),
                                              check_exact=False,atol=1e-10,rtol=1e-12)
            for filename in ['prepared_ev_stops.parquet','ev_stop_charging_audit.parquet']:
                pd.testing.assert_frame_equal(pd.read_parquet(BASE/filename),pd.read_parquet(OUT/name/filename),
                                              check_exact=False,atol=1e-10,rtol=1e-12)
            print('Full PHEV10 baseline reproduction passed',flush=True)
    old = pd.read_csv(BASE/'existing_ev_hourly_charging_curve_10x.csv')
    new = pd.read_csv(OUT/'phev20/existing_ev_hourly_charging_curve_10x.csv')
    s20 = json.loads((OUT/'phev20/charging_code_summary.json').read_text())
    old_stops = pd.read_parquet(BASE/'prepared_ev_stops.parquet')
    new_stops = pd.read_parquet(OUT/'phev20/prepared_ev_stops.parquet')
    pd.testing.assert_frame_equal(old_stops.drop(columns='energyConsumption_kWh'),
                                  new_stops.drop(columns='energyConsumption_kWh'))
    np.testing.assert_allclose(new_stops.energyConsumption_kWh,
        old_stops.energyConsumption_kWh*np.where(old_stops.powertrain.eq('PHEV'),2,1))
    old_audit = pd.read_parquet(BASE/'ev_stop_charging_audit.parquet')
    new_audit = pd.read_parquet(OUT/'phev20/ev_stop_charging_audit.parquet')
    np.testing.assert_allclose(old_audit.initialsoc_kWh,new_audit.initialsoc_kWh)
    assert original['EVI_sha256'] == s20['EVI_sha256']
    compare = pd.read_csv(BASE/'evi_weekday_comparison.csv')
    evi_per1000 = compare.EVI_weekday_kW_per_1000_PEV.to_numpy()
    fleet = original['expanded_fleet']
    evi_mw = evi_per1000*fleet/1e6
    original['_folder']=str(BASE);s20['_folder']=str(OUT/'phev20')
    records = [metrics('PHEV10',old,original),metrics('PHEV20',new,s20),
        dict(scenario='EVI_weekday_same_15440_fleet',daily_charging_MWh=evi_mw.sum(),
             peak_MW=evi_mw.max(),peak_hour=int(evi_mw.argmax()),
             evening18_24_energy_share=evi_mw[18:].sum()/evi_mw.sum())]
    pd.DataFrame(records).to_csv(OUT/'comparison_metrics.csv',index=False)
    hourly = pd.DataFrame({'hour':range(24),
        'PHEV10_total_MW_15440':old.total_MW_10x,'PHEV20_total_MW_15440':new.total_MW_10x,
        'EVI_total_MW_15440':evi_mw,'EVI_original_MW_30000':evi_per1000*.03,
        'PHEV10_kW_per1000':old.total_kW_per_1000_PEV,'PHEV20_kW_per1000':new.total_kW_per_1000_PEV,
        'EVI_kW_per1000':evi_per1000,
        'PHEV10_BEV_MW':old.BEV_MW_10x,'PHEV20_BEV_MW':new.BEV_MW_10x,
        'PHEV10_PHEV_MW':old.PHEV_MW_10x,'PHEV20_PHEV_MW':new.PHEV_MW_10x})
    hourly.to_csv(OUT/'comparison_hourly.csv',index=False)
    plt.rcParams.update({'font.size':11})
    fig,axes = plt.subplots(3,1,figsize=(12,12),layout='constrained')
    series = [(old.total_MW_10x.to_numpy(),'PHEV factor 10% (original)','#2166ac','--'),
              (new.total_MW_10x.to_numpy(),'PHEV factor 20% (rerun)','#00876c','-'),
              (evi_mw,'EVI weekday (unchanged)','#d95f02',':')]
    for p,label,color,style in series:
        axes[0].plot(range(24),p*1e6/fleet,label=label,color=color,ls=style,lw=2.5)
        axes[1].plot(range(24),p,label=label,color=color,ls=style,lw=2.5)
        axes[2].plot(range(24),p/p.sum()*100,label=label,color=color,ls=style,lw=2.5)
    axes[0].set(title='Same-fleet intensity',ylabel='kW / 1,000 PEVs')
    axes[1].set(title='Total load: 15,440 PEVs for all three curves',ylabel='Hourly-average power (MW)')
    axes[2].set(title='Shape only: daily energy normalized to 100%',ylabel='Daily energy share (%)')
    for ax in axes:
        ax.set(xlim=(0,23),xticks=range(24),ylim=(0,None),xlabel='Hour (native clock; no time shift)')
        ax.grid(alpha=.2);ax.legend(fontsize=10)
    fig.suptitle('Existing-EV charging: PHEV electricity-factor sensitivity',fontsize=16)
    fig.supxlabel('607 BEVs + 937 PHEVs in the sample; 10x expansion. Only PHEV energy factor changes (0.10 -> 0.20).\n'
                  'Same seed 42; all charging choices recomputed. EVI: 35 mi/day, 10 C, Home80, weekday.\n'
                  'Scenario test, not validated local calibration; SOC drift and powertrain-mixture differences remain.',fontsize=9)
    fig.savefig(OUT/'phev10_vs_phev20_vs_evi.png',dpi=180);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,kind in zip(axes,['BEV','PHEV']):
        ax.plot(range(24),old[f'{kind}_MW_10x'],color='#2166ac',ls='--',lw=2.4,label='Original (PHEV factor 10%)')
        ax.plot(range(24),new[f'{kind}_MW_10x'],color='#00876c',lw=2.4,label='Rerun (PHEV factor 20%)')
        ax.set(title=kind,xlabel='Hour',ylabel='Hourly-average power (MW)',xlim=(0,23),xticks=range(0,24,2),ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Powertrain breakdown: unchanged BEV energy rate; charging choices rerun',fontsize=14)
    fig.supxlabel('Shared weighted fleet sampling can change BEV session assignments despite the same seed and BEV energy rate.',fontsize=9)
    fig.savefig(OUT/'bev_phev_breakdown.png',dpi=180);plt.close(fig)
    for p,sha in protected.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest()==sha, f'Baseline overwritten: {p}'
    for p in OUT.rglob('*.png'):
        with Image.open(p) as im: im.verify()
    result = dict(status='charging_sensitivity_complete',baseline=str(BASE),output=str(OUT),
        baseline_reproduction_passed=True,baseline_files_unchanged=True,stops_and_initial_soc_unchanged=True,
        change='Only PHEV electricity factor 0.1 -> 0.2 in both projected SOC and traction integration',
        default_factor_preserved=.1,seed=42,all_sessions_recomputed=True,
        notes=['PHEV20 is user-requested sensitivity, not established local calibration.',
               'Original global weighted random sampling allows BEV session changes despite unchanged BEV traction.',
               'EVI profile unchanged and normalized to same fleet; it is not demand input.',
               'SOC drift and excluded vehicles remain; no downstream dispatch, emissions or concentrations run.'],
        metrics=records)
    (OUT/'sensitivity_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
