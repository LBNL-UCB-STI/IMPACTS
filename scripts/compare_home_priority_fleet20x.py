"""Fleet-expansion sensitivity only; no behavior calibration or downstream AQ run."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'examples/pilates/impacts/impacts_output/events-fc07-existing-ev-charging-code10x-20260928-v2/home-priority-tests-20261012'
OUT = SOURCE.parent / 'home-priority-fleet20x-20261012'


def summarize(name, power, fleet, phase):
    power = np.asarray(power, dtype=float)
    return dict(scenario=name, phase=phase, input_fleet=fleet,
                daily_MWh=float(power.sum()), peak_MW=float(power.max()),
                peak_hour=int(power.argmax()),
                evening18_24_MWh=float(power[18:].sum()),
                evening18_24_share=float(power[18:].sum()/power.sum()),
                daily_kWh_per_input_vehicle=float(power.sum()*1000/fleet),
                peak_kW_per1000=float(power.max()*1e6/fleet))


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserve existing output: {OUT}')
    paths = [SOURCE / x for x in ['comparison_hourly.csv', 'steady_hourly_all_seeds.csv', 'run_summary.json']]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    original = pd.read_csv(paths[0])
    ensemble = pd.read_csv(paths[1])
    assert original.hour.tolist() == list(range(24))
    curves = pd.DataFrame({'hour': original.hour})
    records = []
    names = {'original20_single_day_MW': 'Original_PHEVfactor20_single_day',
             'home80_steady_MW': 'Home80_steady', 'home100_steady_MW': 'Home100_steady',
             'home80_first_MW': 'Home80_first', 'home100_first_MW': 'Home100_first',
             'EVI_matched_MW': 'EVI_range_proxy_reference'}
    for column, name in names.items():
        p = original[column].to_numpy()
        assert np.isfinite(p).all() and (p >= 0).all()
        phase = 'steady' if 'steady' in name else 'first' if 'first' in name else 'reference'
        for factor in [10, 20]:
            fleet = 1544*factor
            scaled = p*(factor/10)
            curves[f'{name}_{factor}x_MW'] = scaled
            curves[f'{name}_{factor}x_kW_per1000'] = scaled*1e6/fleet
            records.append(summarize(f'{name}_{factor}x', scaled, fleet, phase))
        np.testing.assert_allclose(curves[f'{name}_20x_MW'], curves[f'{name}_10x_MW']*2)
        np.testing.assert_allclose(curves[f'{name}_20x_kW_per1000'], curves[f'{name}_10x_kW_per1000'])
        np.testing.assert_allclose(p/p.sum(), (2*p)/(2*p).sum())
    OUT.mkdir()
    curves.to_csv(OUT / 'comparison_hourly.csv', index=False)
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT / 'comparison_metrics.csv', index=False)
    ensemble['steady_MW_10x'] = ensemble.pop('steady_MW')
    ensemble['steady_MW_20x'] = ensemble.steady_MW_10x*2
    ensemble.to_csv(OUT / 'steady_hourly_all_seeds.csv', index=False)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), layout='constrained')
    colors = {'Home80': '#2166ac', 'Home100': '#00876c'}
    h = original.hour
    for name, share in [('Home80', .8), ('Home100', 1.)]:
        color = colors[name]
        p = original[f'{name.lower()}_steady_MW']
        axes[0].plot(h, p, '--', color=color, lw=1.6, label=f'{name} 10x: 15,440 input EVs')
        axes[0].plot(h, p*2, color=color, lw=2.5, label=f'{name} 20x: 30,880 input EVs')
        seeds = ensemble[ensemble.home_share.eq(share)].groupby('hour').steady_MW_20x
        axes[0].fill_between(h, seeds.min(), seeds.max(), color=color, alpha=.1)
        axes[1].plot(h, p*1e6/15440, color=color, lw=2.5, label=f'{name}: 10x and 20x coincide')
    evi = original.EVI_matched_MW
    axes[0].plot(h, evi, ':', color='#d95f02', lw=2, label='EVI reference: 15,440 EVs (different fleet from 20x)')
    axes[0].plot(h, evi*2, '-.', color='#d95f02', lw=2.5, label='EVI same-fleet reference: 30,880 EVs')
    axes[1].plot(h, evi*1e6/15440, '-.', color='#d95f02', lw=2.5, label='EVI fleet/range-proxy matched')
    axes[0].set(title='Total load: 20x sample expansion = twice the previous 10x load', ylabel='Hourly-average charging power (MW)')
    axes[1].set(title='Per-vehicle comparison: fleet expansion does not change the gap', ylabel='kW / 1,000 input EVs')
    for ax in axes:
        ax.set(xlim=(0, 23), ylim=(0, None), xticks=range(24), xlabel='Hour (events clock; no time shift)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9, ncol=2)
    fig.suptitle('20x fleet-expansion sensitivity: stable-day Home80 / Home100', fontsize=17)
    fig.supxlabel('PHEV electric-energy proxy remains 20%; charging behavior and individual energy use unchanged.\n'
                  'Seed 42 lines; shading = five-seed range, not confidence intervals. Infeasible vehicles excluded as before.\n'
                  '20x is a hypothetical 5% sample interpretation, not a replacement for the established 10% sampling rate.\n'
                  'EVI curves are linearly reweighted references, not new EVI simulations. No power-generation or AQ rerun.', fontsize=9)
    fig.savefig(OUT / 'home_priority_10x_vs_20x_vs_evi.png', dpi=180)
    plt.close(fig)
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)]
    summary = dict(status='complete', scope='Fleet expansion sensitivity, not energy-factor calibration',
                   sample_input_EV_count=1544, previous_expansion=10, test_expansion=20,
                   previous_input_fleet=15440, test_input_fleet=30880,
                   implied_test_sampling_fraction=.05, established_sampling_fraction=.1,
                   unchanged=['trip schedules', 'charging decisions', 'SOC balance', 'PHEV factor 0.2',
                              'per-vehicle consumption', 'normalized hourly shape', 'excluded vehicles'],
                   seed42_excluded_BEVs_sample={'Home80': 8, 'Home100': 2},
                   EVI='Existing fleet/range-proxy reference shown at both fleet sizes; no new API simulation',
                   method='Exact linear aggregation of completed simulations; no individual simulation rerun needed',
                   downstream_AQ_run=False, source_hashes=hashes, source_files_unchanged=True,
                   checks=['hourly 20x = 2 * 10x', 'per-1000 invariance', 'normalized shape invariance', 'source preservation'],
                   metrics=records)
    (OUT / 'run_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(metrics.to_string(index=False))
    print(f'OUTPUT: {OUT}')


if __name__ == '__main__':
    main()
