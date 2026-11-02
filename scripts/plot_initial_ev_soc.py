"""Plot saved synthetic initial SOC once per vehicle, before periodic warm-up."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / 'examples/pilates/impacts/impacts_output/events-fc07-existing-ev-charging-code10x-20260928-v2'
    source = base / 'phev20-sensitivity-20261012/phev20/ev_stop_charging_audit.parquet'
    out = base / 'initial-soc-distribution-20261026'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    columns = ['vehicle', 'powertrain', 'battery_kWh', 'initialsoc_kWh']
    rows = pd.read_parquet(source, columns=columns)
    assert rows[columns].notna().all().all()
    assert (rows.groupby('vehicle')[columns[1:]].nunique() == 1).all().all()
    vehicles = rows.drop_duplicates('vehicle').copy()
    assert vehicles.battery_kWh.gt(0).all()
    vehicles['initial_SOC_pct'] = 100 * vehicles.initialsoc_kWh / vehicles.battery_kWh
    assert np.isfinite(vehicles.initial_SOC_pct).all()
    assert vehicles.initial_SOC_pct.between(0, 100).all()
    assert vehicles.groupby('powertrain').size().to_dict() == {'BEV': 607, 'PHEV': 936}
    out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    bins = np.arange(0, 105, 5)
    summaries = {}
    for ax, kind, color in zip(axes, ['BEV', 'PHEV'], ['#2166ac', '#d9781c']):
        values = vehicles.loc[vehicles.powertrain.eq(kind), 'initial_SOC_pct'].to_numpy()
        counts, _ = np.histogram(values, bins=bins)
        assert counts.sum() == len(values)
        summary = dict(vehicles=len(values), mean_pct=float(values.mean()),
                       median_pct=float(np.median(values)), min_pct=float(values.min()), max_pct=float(values.max()))
        summaries[kind] = summary
        ax.bar(bins[:-1] + 2.5, counts / len(values) * 100, width=4.6, color=color, alpha=.85)
        ax.axvline(values.mean(), color='#222222', ls='--', lw=1.7, label='Mean initial SOC')
        ax.text(.03, .96, f'n = {len(values):,}\nMean: {values.mean():.1f}%\nMedian: {np.median(values):.1f}%',
                transform=ax.transAxes, va='top', bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
        ax.set(title=kind, xlabel='Initial battery SOC (%)', xlim=(0, 100),
               xticks=np.arange(0, 101, 20), ylim=(0, 30))
        ax.grid(axis='y', alpha=.18)
        ax.legend(loc='upper right', fontsize=10)
    axes[0].set_ylabel('Share of vehicles (%)')
    fig.suptitle('Initial SOC distributions before charging simulation\nExisting BEVs and PHEVs | Original synthetic initialization', fontsize=17)
    fig.text(.5, .035, 'Saved Beta-distribution draws; one observation per vehicle; 5-percentage-point bins.\n'
             '607 BEVs + 936 PHEVs with activity records; before steady-day exclusions. No fleet scaling.\n'
             'Model initialization, not measured SOC or the steady-state midnight distribution.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .16, 1, .86), w_pad=2)
    fig.savefig(out / 'initial_soc_distribution_bev_phev.png', dpi=200)
    plt.close(fig)
    vehicles.to_parquet(out / 'initial_soc_per_vehicle.parquet', index=False)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    report = dict(source=str(source), source_sha256=digest, statistics=summaries,
                  bins_pct=bins.tolist(), unique_vehicle_weighting=True,
                  includes_12_BEVs_excluded_from_steady_results=True, missing_PHEV_itinerary=1,
                  model_rerun=False, original_source_unchanged=True)
    (out / 'initial_soc_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'statistics': summaries}, indent=2))


if __name__ == '__main__':
    main()
