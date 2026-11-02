"""Plot saved Home80/Home80 loads with a display-only periodic 24:00 point."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_latest_home_charging_starts import PARENT


def close_day(values):
    """Return an independent 25-point display array; never use it for energy sums."""
    x = np.asarray(values, dtype=float)
    if x.shape != (24,) or not np.isfinite(x).all():
        raise ValueError('Expected exactly 24 finite hourly values')
    return np.concatenate([x, x[:1]])


def main():
    source = PARENT / 'evi-bev25-reference/comparison_hourly.csv'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = pd.read_csv(source).sort_values('hour_start')
    np.testing.assert_array_equal(data.hour_start, np.arange(24))
    out = PARENT / 'charging-load-closed-24h'
    if out.exists():
        raise FileExistsError(f'Preserve existing results: {out}')
    out.mkdir()
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14})
    hours = np.arange(25)
    lines = [
        ('Charging code: BEV Home80 + PHEV Home80', 'BEV80_PHEV_Home80_MW', '#2166ac', '-'),
        ('EVI: events-matched (39.3% BEV)', 'EVI_equal_MW', '#00876c', '--'),
        ('EVI: 25% BEV / 75% PHEV', 'EVI_bev25_total_MW', '#d95f02', ':'),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), layout='constrained', sharex=True)
    metrics = {}
    for label, column, color, style in lines:
        raw = data[column].to_numpy()
        display = close_day(raw)
        np.testing.assert_array_equal(display[:24], raw)
        assert display[24] == raw[0]
        for ax, factor in zip(axes, [1, 1e6 / 15440]):
            ax.plot(hours, display * factor, color=color, ls=style, lw=2.5, label=label)
        metrics[column] = dict(daily_MWh=float(raw.sum()), peak_MW=float(raw.max()),
                               peak_hour=int(raw.argmax()), display_24h_MW=float(display[24]))
    for ax, title, unit in zip(axes,
        ['Total: same 15,440 PEVs', 'Per 1,000 input PEVs'],
        ['Hourly-average grid power (MW)', 'Hourly-average grid power (kW / 1,000 PEVs)']):
        ax.set(title=title, ylabel=unit, xlim=(0, 24), ylim=(0, None), xticks=hours)
        ax.grid(alpha=.22)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[1].set_xlabel('Hour of day (24:00 repeats 00:00 for display)')
    fig.suptitle('Charging load: BEV Home80 + PHEV Home80 vs EVI', fontsize=17)
    fig.supxlabel('EVI: weekday, 20 C, 35 mi/day, Home80; home L1/L2 50:50. Charging code: L1/L2 20:80.\n'
                  '24:00 is a periodic plotting endpoint, not an extra hour; daily energy uses the original 24 bins.', fontsize=10)
    fig.savefig(out / 'charging_load_home80_evi_0_24h.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for ax, kind, count, evi25_count in zip(axes, ['BEV', 'PHEV'], [6070, 9370], [3860, 11580]):
        for label, column, color, style, denominator in [
            (f'Charging code: {kind} Home80', f'{kind}_Home80_MW', '#2166ac', '-', count),
            ('EVI: events-matched', f'EVI_equal_{kind}_MW', '#00876c', '--', count),
            ('EVI: 25% BEV preset', f'EVI_bev25_{kind}_MW', '#d95f02', ':', evi25_count),
        ]:
            ax.plot(hours, close_day(data[column]) * 1e6 / denominator,
                    label=label, color=color, ls=style, lw=2.4)
        ax.set(title=kind, xlabel='Hour (24:00 repeats 00:00)', ylabel=f'Grid kW / 1,000 {kind}s',
               xlim=(0, 24), ylim=(0, None), xticks=np.arange(0, 25, 2))
        ax.grid(alpha=.22)
        ax.legend(fontsize=10)
    fig.suptitle('BEV and PHEV charging loads — Home80 / Home80', fontsize=16)
    fig.supxlabel('Each curve uses its own powertrain denominator. The EVI 25% preset also changes range-class shares.\n'
                  '24:00 is display-only; no extra energy, rescaling, time shift or model rerun.', fontsize=10)
    fig.savefig(out / 'bev_phev_load_home80_evi_0_24h.png', dpi=180)
    plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    (out / 'plot_audit.json').write_text(json.dumps(dict(source=str(source), source_sha256=digest,
        display_point_24='Same as hour 0; excluded from daily energy calculations',
        metrics_original_24_bins=metrics, charging_rerun=False, air_quality_rerun=False), indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'metrics': metrics}, indent=2))


if __name__ == '__main__':
    main()
