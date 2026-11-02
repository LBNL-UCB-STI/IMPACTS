"""Compare saved original home starts with current BEV80/PHEV80 sessions."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_latest_home_charging_starts import BASE, PARENT


def main():
    sources = {
        'original': BASE / 'modeled_ev_charging_sessions.parquet',
        'current_BEV': BASE / 'bev-phev-home80-evi20c-20261012/bev_home80/sessions.parquet',
        'current_PHEV': PARENT / 'phev_home80/sessions.parquet',
        'original_hourly_check': BASE / 'arrival_audit_hourly.csv',
        'current_hourly_check': PARENT / 'home-charging-start-audit/home_charging_starts_audit.json',
    }
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sources.items()}
    original = pd.read_parquet(sources['original'])
    counts, details = {}, {}
    for version in ['original', 'current']:
        for kind in ['BEV', 'PHEV']:
            d = original.loc[original.powertrain.eq(kind)] if version == 'original' else pd.read_parquet(sources['current_' + kind])
            energy = 'energyCharged_kWh' if version == 'original' else 'grid_energy_kWh'
            keep = d.is_home & d[energy].gt(0)
            if version == 'original':
                keep &= d.retained
            elif 'included_in_profile' in d:
                keep &= d.included_in_profile
            d = d.loc[keep]
            assert np.isfinite(d.arrivalTime).all()
            hours = np.floor(d.arrivalTime.to_numpy() % 86400 / 3600).astype(int)
            values = np.bincount(hours, minlength=24)
            key = version + '_' + kind
            counts[key] = values
            details[key] = dict(sessions=len(d), unique_vehicles=int(d.vehicle.nunique()),
                                peak_hour=int(values.argmax()), peak_starts=int(values.max()))
        counts[version + '_total'] = counts[version + '_BEV'] + counts[version + '_PHEV']
    check = pd.read_csv(sources['original_hourly_check']).sort_values('hour')
    np.testing.assert_array_equal(counts['original_total'], check.home_charging_starts)
    check_new = json.loads(sources['current_hourly_check'].read_text())['hourly']
    np.testing.assert_array_equal(counts['current_BEV'], check_new['BEV Home80'])
    np.testing.assert_array_equal(counts['current_PHEV'], check_new['PHEV Home80'])
    assert counts['original_total'].sum() == 524
    assert counts['current_total'].sum() == 1239

    out = PARENT / 'original-vs-current-home-starts'
    if out.exists():
        raise FileExistsError(f'Preserve existing results: {out}')
    out.mkdir()
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14})
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=.38, wspace=.18)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    h = np.arange(24)
    for ax, kind, title in zip(axes, ['total', 'BEV', 'PHEV'], ['BEV + PHEV', 'BEV', 'PHEV']):
        for version, color, style, marker, label in [
            ('original', '#e08214', '--', 's', 'Original code (Home80 setting)'),
            ('current', '#2166ac', '-', 'o', 'Current Home80'),
        ]:
            y = counts[version + '_' + kind]
            ax.plot(h, y, color=color, ls=style, marker=marker, lw=2.3, ms=4,
                    label=f'{label}: {y.sum():,} starts')
            peak = int(y.argmax())
            ax.annotate(str(y[peak]), (peak, y[peak]), xytext=(6, 7), textcoords='offset points')
        ax.set_title(title)
        ax.set_xlim(-.3, 23.3)
        ax.set_ylim(0, 225 if kind == 'total' else 140)
        ax.set_xticks(np.arange(0, 24, 1 if kind == 'total' else 2))
        ax.set_xlabel('Hour of day (start of hourly interval)')
        ax.set_ylabel('Home charging starts / hour')
        ax.grid(alpha=.22)
        ax.legend(loc='upper left', fontsize=10)
    fig.suptitle('Home charging starts: original vs current Home80 / Home80\n'
                 'Same input EV fleet; unscaled sample counts', fontsize=17, y=.975)
    fig.subplots_adjust(top=.86, bottom=.16, left=.075, right=.975)
    fig.text(.5, .045,
             'Original: saved single-day notebook result. Current: steady representative day; BEV80 + PHEV80.\n'
             'Energy assumptions, PHEV treatment, charge selection and SOC handling also differ; not a Home-share-only experiment.\n'
             'Positive-energy home sessions; each version retains its own validity filter. Hours folded modulo 24; no time shift or ×10 scaling.',
             ha='center', fontsize=10)
    fig.savefig(out / 'original_vs_current_home_starts.png', dpi=180)
    plt.close(fig)
    report = dict(sources={k: str(p) for k, p in sources.items()}, sha256=hashes,
                  scale=1, statistics=details, hourly={k: v.tolist() for k, v in counts.items()},
                  caveat='Version comparison, not an isolated Home-share sensitivity; original single day vs current equilibrium day.')
    (out / 'comparison_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for k, p in sources.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[k]
    print(json.dumps({'output': str(out), 'statistics': details}, indent=2))


if __name__ == '__main__':
    main()
