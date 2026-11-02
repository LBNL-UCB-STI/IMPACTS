"""Plot actual modeled home charging starts from saved steady-day sessions."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'examples/pilates/impacts/impacts_output/events-fc07-existing-ev-charging-code10x-20260928-v2'
PARENT = BASE / 'bev80-phev80-90-evi-equal-20261012'
OUT = PARENT / 'home-charging-start-audit'


def main():
    sources = {
        'BEV Home80': BASE / 'bev-phev-home80-evi20c-20261012/bev_home80/sessions.parquet',
        'PHEV Home80': PARENT / 'phev_home80/sessions.parquet',
        'PHEV Home90': PARENT / 'phev_home90/sessions.parquet',
    }
    counts, metadata = {}, {}
    for name, path in sources.items():
        data = pd.read_parquet(path)
        keep = data.is_home & data.grid_energy_kWh.gt(0)
        if 'included_in_profile' in data:
            keep &= data.included_in_profile
        selected = data.loc[keep].copy()
        assert selected.arrivalTime.notna().all()
        # These models initiate charging immediately on arrival at chosen stops.
        hours = np.floor((selected.arrivalTime.to_numpy() % 86400) / 3600).astype(int)
        values = np.bincount(hours, minlength=24)
        assert values.sum() == len(selected)
        counts[name] = values
        metadata[name] = dict(
            source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            sessions=len(selected), unique_vehicles=int(selected.vehicle.nunique()),
            starts_after_24h=int(selected.arrivalTime.ge(86400).sum()),
            peak_hour=int(values.argmax()), peak_starts=int(values.max()),
        )
    counts['BEV80 + PHEV80'] = counts['BEV Home80'] + counts['PHEV Home80']
    counts['BEV80 + PHEV90'] = counts['BEV Home80'] + counts['PHEV Home90']
    assert [int(counts[k].sum()) for k in sources] == [493, 746, 837]
    if OUT.exists():
        raise FileExistsError(f'Preserve existing output: {OUT}')
    OUT.mkdir()
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'legend.fontsize': 11})
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    h = np.arange(24)
    styles = [('BEV Home80', '#2166ac', 'o', '-'),
              ('PHEV Home80', '#e08214', 's', '-'),
              ('PHEV Home90', '#7b3294', '^', '--')]
    for name, color, marker, ls in styles:
        axes[0].plot(h, counts[name], color=color, marker=marker, ls=ls,
                     lw=2.2, ms=4, label=f'{name}: {counts[name].sum():,} starts/day')
    axes[0].set_title('By powertrain and home-preference scenario')
    axes[0].set_ylim(0, 155)
    for name, color, marker, ls in [
        ('BEV80 + PHEV80', '#2166ac', 'o', '-'),
        ('BEV80 + PHEV90', '#7b3294', '^', '--'),
    ]:
        values = counts[name]
        axes[1].plot(h, values, color=color, marker=marker, ls=ls,
                     lw=2.4, ms=4, label=f'{name}: {values.sum():,} starts/day')
        peak = int(values.argmax())
        axes[1].annotate(str(values[peak]), (peak, values[peak]),
                         xytext=(-24 if 'PHEV80' in name else 10, 4),
                         textcoords='offset points')
    axes[1].set_title('Combined BEV + PHEV: peak starts at 18:00–19:00')
    axes[1].set_ylim(0, 245)
    for ax in axes:
        ax.set_ylabel('Home charging starts / hour')
        ax.set_xlim(-.3, 23.3)
        ax.set_xticks(h)
        ax.grid(alpha=.22)
        ax.legend(loc='upper left', framealpha=.95)
    axes[1].set_xlabel('Hour of day (18 denotes 18:00–19:00)')
    fig.suptitle('Modeled home charging starts — latest existing-EV scenarios\n'
                 'Steady representative day; seed 42; unscaled sample counts', fontsize=16)
    fig.text(.5, .018, 'Positive-energy home sessions only; BEV infeasible vehicles excluded. '
             'Starts after midnight folded into 0–24 h.\n'
             'Counts are charging sessions, not power (kW), and are not scaled ×10.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .93))
    fig.savefig(OUT / 'home_charging_starts_hourly.png', dpi=180)
    plt.close(fig)
    report = dict(seed=42, scale=1, time_bins='[h, h+1), modulo 24 h',
                  series=metadata, hourly={k: v.tolist() for k, v in counts.items()})
    (OUT / 'home_charging_starts_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for name, path in sources.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata[name]['sha256']
    print(json.dumps({'output': str(OUT), 'series': metadata}, indent=2))


if __name__ == '__main__':
    main()
