"""Close the supplied LDV time-series figures using saved results only."""
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_latest_home_charging_starts import BASE, PARENT
from plot_charging_load_closed_day import close_day


def main():
    sources = {
        'starts': PARENT / 'original-vs-current-home-starts/comparison_audit.json',
        'loads': PARENT / 'comparison_hourly.csv',
        'ensemble': PARENT / 'hourly_all_seeds.csv',
        'soc': BASE / 'bev-phev-home80-evi20c-20261012/soc-distribution-20261019/soc_distribution_4times.png',
    }
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in sources.items()}
    counts = json.loads(sources['starts'].read_text())['hourly']
    curves = pd.read_csv(sources['loads']).sort_values('hour_start')
    ensemble = pd.read_csv(sources['ensemble'])
    np.testing.assert_array_equal(curves.hour_start, np.arange(24))
    out = PARENT / 'ldv-presentation-24h-20261026'
    out.mkdir(exist_ok=False)
    hours = np.arange(25)
    checked = {}

    def closed(key, values):
        raw = np.asarray(values, dtype=float)
        result = close_day(raw)
        np.testing.assert_array_equal(result[:24], raw)
        assert result[-1] == result[0]
        checked[key] = dict(original_24_bin_sum=float(raw.sum()),
                            endpoint=float(result[-1]), original_values_preserved=True)
        return result

    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14})
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=.38, wspace=.18)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    for ax, kind, title in zip(axes, ['total', 'BEV', 'PHEV'], ['BEV + PHEV', 'BEV', 'PHEV']):
        for version, color, style, marker, label in [
            ('original', '#e08214', '--', 's', 'Original code (Home80 setting)'),
            ('current', '#2166ac', '-', 'o', 'Current Home80'),
        ]:
            key = version + '_' + kind
            raw = np.asarray(counts[key])
            ax.plot(hours, closed(key, raw), color=color, ls=style, marker=marker,
                    lw=2.3, ms=4, label=f'{label}: {raw.sum():,} starts')
            peak = int(raw.argmax())
            ax.annotate(str(raw[peak]), (peak, raw[peak]), xytext=(6, 7), textcoords='offset points')
        ax.set(title=title, xlim=(0, 24), ylim=(0, 225 if kind == 'total' else 140),
               xticks=np.arange(0, 25, 1 if kind == 'total' else 2),
               xlabel='Hour of day (start of hourly interval)', ylabel='Home charging starts / hour')
        ax.grid(alpha=.22)
        ax.legend(loc='upper left', fontsize=10)
    fig.suptitle('Home charging starts: original vs current Home80 / Home80\n'
                 'Same input EV fleet; unscaled sample counts', fontsize=17, y=.975)
    fig.subplots_adjust(top=.86, bottom=.17, left=.075, right=.975)
    fig.text(.5, .035,
             'Original: saved single-day notebook result. Current: steady representative day; BEV80 + PHEV80.\n'
             'Energy assumptions, PHEV treatment, charge selection and SOC handling also differ; not a Home-share-only experiment.\n'
             'Positive-energy home sessions; each version retains its own validity filter. No time shift or 10x scaling.\n'
             '24:00 repeats 00:00 for display only; session totals use the original 24 hourly bins.', ha='center', fontsize=10)
    fig.savefig(out / 'original_vs_current_home_starts_0_24h.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    for label, col, color, style in [
        ('BEV Home80 + PHEV Home80 (previous)', 'BEV80_PHEV_Home80_MW', '#2166ac', '-'),
        ('BEV Home80 + PHEV Home90 (new)', 'BEV80_PHEV_Home90_MW', '#7b3294', '--'),
        ('EVI: residential L1/L2 50:50', 'EVI_equal_MW', '#00876c', '-.'),
    ]:
        ax.plot(hours, closed(col, curves[col]), color=color, ls=style, lw=2.5, label=label)
    for name, color in [('PHEV_Home80', '#2166ac'), ('PHEV_Home90', '#7b3294')]:
        g = ensemble.loc[ensemble.scenario.eq(name)].groupby('hour').total_MW
        low, high = g.min().reindex(range(24)), g.max().reindex(range(24))
        ax.fill_between(hours, closed(name + '_min', low), closed(name + '_max', high), color=color, alpha=.1)
    ax.set(title='Total: 15,440 input PEVs (10x)', xlabel='Hour of day',
           ylabel='Hourly-average grid power (MW)', xlim=(0, 24), xticks=hours, ylim=(0, None))
    ax.grid(alpha=.2)
    ax.legend(loc='upper left', fontsize=12)
    fig.suptitle('PHEV Home80 vs Home90, with BEV Home80 fixed', fontsize=19)
    fig.text(.5, .025, '24:00 repeats 00:00 for display only; daily energy uses 24 bins. Seed 42 lines; five-seed range.\n'
             'Same EVI reference as the supplied figure (events-matched); no reference replacement or model rerun.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .09, 1, .94))
    fig.savefig(out / 'phev_home80_90_load_0_24h.png', dpi=180)
    plt.close(fig)
    # A SOC histogram has a SOC axis, not an hourly time axis. Preserve it exactly.
    shutil.copy2(sources['soc'], out / 'soc_distribution_4times_unchanged.png')
    for key, path in sources.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[key]
    report = dict(sources={key: str(path) for key, path in sources.items()}, source_sha256=hashes,
                  endpoint='24:00 = 00:00; display-only, excluded from totals', checks=checked,
                  soc_histogram_unchanged=True, charging_rerun=False, air_quality_rerun=False)
    (out / 'plot_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(out)
    print('Verified unchanged source hashes and all 0/24 endpoints.')


if __name__ == '__main__':
    main()
