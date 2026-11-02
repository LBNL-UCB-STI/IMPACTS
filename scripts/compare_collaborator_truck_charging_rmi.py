"""Read-only input comparison of collaborator events and RMI 2050 daily profiles."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'OUTPUT/collaborator-truck-charging-curves-20261026-final'
RMI = ROOT / 'OUTPUT/external-data/rmi-lbnl-bay-area-forecast-20261026'
OUT = BASE / 'rmi-lbnl-comparison-2050'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)  # Re-render only this script's named comparison products.
    inputs = [BASE/'hourly_charging_diagnostics.csv', BASE/'hourly_charging_by_class.csv']
    hashes = {str(p): sha(p) for p in inputs}
    hourly = pd.read_csv(inputs[0])
    classes = pd.read_csv(inputs[1])
    manifest = json.loads((RMI/'manifest_20261026T183549Z.json').read_text())
    expected = {x['path']: x['sha256'] for x in manifest['entries']}
    curves, metrics, rows = {}, [], []
    scenarios = list(hourly.scenario.unique())
    colors = dict(zip(scenarios, ['#0072B2', '#D55E00', '#009E73']))

    def add(source, scenario, scope, kw, tail=0):
        kw = np.asarray(kw, dtype=float)
        assert kw.shape == (24,) and np.isfinite(kw).all() and (kw >= 0).all()
        energy = kw.sum()  # 1-hour bins: kW * 1 h = kWh
        share = kw / energy * 100
        assert np.isclose(share.sum(), 100)
        curves[source, scenario, scope] = (kw, share)
        metrics.append(dict(source=source, scenario=scenario, scope=scope,
            first24_energy_MWh=energy/1000, peak_MW=kw.max()/1000,
            peak_hour_index=int(kw.argmax()), trough_hour_index=int(kw.argmin()),
            share_06_18_percent=share[6:18].sum(),
            share_18_06_percent=share[:6].sum()+share[18:].sum(),
            peak_to_mean=kw.max()/kw.mean(), after24_energy_MWh=tail/1000,
            after24_percent_of_record_energy=tail/(energy+tail)*100))
        for h in range(24):
            rows.append(dict(source=source, scenario=scenario, scope=scope,
                hour_index=h, power_kW=kw[h], energy_share_percent=share[h]))

    for s in scenarios:
        d = hourly.loc[(hourly.scenario == s) & (hourly.powertrain == 'BE')]
        d = d.sort_values('elapsed_hour_start')
        assert np.array_equal(d.loc[d.elapsed_hour_start.lt(24), 'elapsed_hour_start'], np.arange(24))
        add('Collaborator', s, 'All BE classes',
            d.loc[d.elapsed_hour_start.lt(24), 'energy_kWh'],
            d.loc[d.elapsed_hour_start.ge(24), 'energy_kWh'].sum())
        c = classes.loc[(classes.scenario == s) & (classes.powertrain == 'BE')]
        all_c = c.groupby('elapsed_hour_start').energy_kWh.sum()
        assert np.allclose(all_c.values, d.energy_kWh.values)
        c = c.loc[c.vehicle_class.isin(['MDV', 'HDV', 'HDT'])]
        c = c.groupby('elapsed_hour_start').energy_kWh.sum().reindex(range(56), fill_value=0)
        add('Collaborator', s, 'MDV+HDV+HDT', c.iloc[:24], c.iloc[24:].sum())

    differences = {}
    duties = ['medium', 'heavy', 'drayage', 'long_haul', 'all']
    for s in ['aate3', '100_zev']:
        for duty in duties:
            p = RMI / f'raw/{s}/2050/{duty}.json'
            hashes[str(p)] = sha(p)
            assert hashes[str(p)] == expected[p.relative_to(RMI).as_posix()]
            payload = json.loads(p.read_text())
            locations = np.asarray(payload['locations'], dtype=float)
            assert locations.shape[0] == 24 and (locations >= 0).all()
            add('RMI-LBNL', s, duty, locations.sum(axis=1))
        total = curves['RMI-LBNL', s, 'all'][0]
        components = sum(curves['RMI-LBNL', s, d][0] for d in duties[:-1])
        differences[s] = dict(max_hourly_difference_kW=float(abs(total-components).max()),
            daily_difference_kWh=float((total-components).sum()))

    pd.DataFrame(rows).to_csv(OUT/'comparison_hourly.csv', index=False)
    pd.DataFrame(metrics).to_csv(OUT/'comparison_metrics.csv', index=False)
    distances = []
    for s in scenarios:
        for scope in ['All BE classes', 'MDV+HDV+HDT']:
            a = curves['Collaborator', s, scope][1]
            for r in ['aate3', '100_zev']:
                b = curves['RMI-LBNL', r, 'all'][1]
                distances.append(dict(scenario=s, scope=scope, rmi_scenario=r,
                    hourly_share_RMSE_percentage_points=float(np.sqrt(np.mean((a-b)**2))),
                    hourly_shape_correlation=float(np.corrcoef(a,b)[0,1])))
    pd.DataFrame(distances).to_csv(OUT/'shape_comparison_metrics.csv', index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11})
    rstyles = {'aate3': ('#242424', '--', 'RMI Scenario 1: AATE3'),
               '100_zev': ('#888888', ':', 'RMI Scenario 2: 100% ZEV')}

    def style(ax, ylabel):
        ax.set_xlim(0,23); ax.set_xticks(range(0,24,2))
        ax.set_xlabel('Reported hour index (no timezone shift)')
        ax.set_ylabel(ylabel); ax.grid(alpha=.22); ax.set_axisbelow(True)

    def finish(fig, name, footer):
        fig.tight_layout(rect=(0,.09,1,.94))
        fig.text(.5,.025,footer,ha='center',va='bottom',fontsize=9)
        fig.savefig(OUT/(name+'.png'),dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(2,1,figsize=(12,10),sharey=True)
    for ax,scope in zip(axes,['All BE classes','MDV+HDV+HDT']):
        for s in scenarios:
            ax.plot(range(24),curves['Collaborator',s,scope][1],lw=2.1,color=colors[s],label=s)
        for r,(color,ls,label) in rstyles.items():
            ax.plot(range(24),curves['RMI-LBNL',r,'all'][1],color=color,ls=ls,lw=2.4,label=label)
        ax.set_title('Collaborator: '+scope+' | RMI: all medium/heavy duty cycles')
        style(ax,'Share of first-24-hour energy (%)');ax.legend(ncol=2,fontsize=9)
    normalized_max = max(v[1].max() for key,v in curves.items()
                         if key[0] == 'Collaborator' or key[2] == 'all')
    axes[0].set_ylim(0, normalized_max * 1.12)
    fig.suptitle('2050 truck charging: normalized hourly pattern comparison',fontsize=16)
    finish(fig,'normalized_hourly_comparison',
        'Each curve sums to 100%; no calibration or magnitude fitting. Collaborator: BE only, first 24 elapsed hours; tail excluded.\n'
        'RMI: full archived dashboard footprint, modeled daily profiles. Vehicle class and duty-cycle scopes are not equivalent.')

    fig,axes=plt.subplots(2,1,figsize=(12,9))
    for s in scenarios:
        axes[0].plot(range(24),curves['Collaborator',s,'All BE classes'][0]/1000,lw=2,color=colors[s],label=s)
    for r,(color,ls,label) in rstyles.items():
        axes[1].plot(range(24),curves['RMI-LBNL',r,'all'][0]/1000,color=color,ls=ls,lw=2.4,label=label)
    for ax in axes:
        style(ax,'Equivalent / reported power (MW)');ax.set_ylim(bottom=0);ax.legend()
    axes[0].set_title('Collaborator: recorded BE events, all classes, scale = 1')
    axes[1].set_title('RMI-LBNL: full dashboard footprint, 2050 (different y-axis scale)')
    fig.suptitle('Absolute load: different fleet sizes and geographic coverage',fontsize=16)
    finish(fig,'absolute_load_separate_scales',
        'Magnitude differences are NOT a model error estimate. No fleet-count or geographic harmonization is possible from these files.\n'
        'No H2FC records, loss adjustment, calibration, 10x scaling, midnight folding, or fabricated hour-24 point.')

    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True)
    labels=['Medium return-to-depot','Heavy return-to-depot','Port drayage','Long haul']
    for ax,r in zip(axes,['aate3','100_zev']):
        for duty,label in zip(duties[:-1],labels):
            ax.plot(range(24),curves['RMI-LBNL',r,duty][1],lw=2,label=label)
        style(ax,'Share of daily energy (%)');ax.set_ylim(bottom=0);ax.legend(fontsize=9)
        ax.set_title(rstyles[r][2])
    fig.suptitle('RMI-LBNL 2050: duty-cycle profiles',fontsize=16)
    finish(fig,'rmi_duty_cycle_patterns',
        'Duty cycles are not direct matches to LD1 / LD3 / MDV / HDV / HDT vehicle classes.\n'
        'Scenario 1 and Scenario 2 are related modeled projections, not independent observed validation datasets.')

    assumptions = [
        '2050 compared with 2050. Input files unchanged; figures use straight lines between hourly values.',
        'Collaborator hourly points represent mean power over [h,h+1); time interpreted as session start by upstream reconstruction.',
        'First 24 elapsed hours only; later event energy retained in tail statistics, not folded into a daily profile.',
        'Only BE included. H2FC is not grid electricity. No scale-up or energy-loss correction.',
        'RMI locations summed once; feeders not added. All aggregate not added to its duty components.',
        'RMI hourly values interpreted as power over 1-hour intervals for energy diagnostics; exact interval/timezone/DST conventions unconfirmed.',
        'No location-index geometry or RMI fleet count: geography and per-vehicle magnitude cannot be harmonized.',
        'MDV+HDV+HDT removes light freight, but does not match RMI duty-cycle definitions.',
        'Collaborator successful externalZone1 event sample may omit depot or unsuccessful sessions; SOC and overlap flags retained upstream.',
        'Shape agreement/disagreement is a model-to-model diagnostic, not empirical validation.',
        'RMI report PDF pages 32-33: >=4-hour stops; charging after daily activity; MD 1.0, HD 2.2 kWh/mile; effective charger output 70% nameplate.',
    ]
    for p,h in hashes.items():
        assert sha(Path(p)) == h, p
    (OUT/'comparison_manifest.json').write_text(json.dumps(dict(inputs_sha256=hashes,
        assumptions=assumptions, rmi_aggregate_rounding_differences=differences,
        script_sha256=sha(Path(__file__))),indent=2),encoding='utf-8')
    (OUT/'comparison_notes.txt').write_text('\n'.join(assumptions),encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False))
    print('OUTPUT:',OUT)


if __name__ == '__main__':
    main()
