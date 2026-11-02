"""Plot existing-EV baseline by vehicle class; EVI is a benchmark, not an input."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_sf_evi_410k import hourly_kw


def build_curves(source, summary, evi):
    """MWh in one-hour bins equals mean MW; normalize by fleet, not chargers."""
    if source.hour_of_day.tolist() != list(range(24)):
        raise ValueError('Expected 24 ordered hourly bins')
    counts = summary['electric_road_vehicle_counts']
    if any(not name.startswith(('ev-', 'phev-')) for name in counts):
        raise ValueError('New electric vehicle class requires explicit classification')
    if summary['truck_charging_sessions'] != 0 or summary['phev_charging_sessions'] != 0:
        raise ValueError('This BEV-only benchmark requires review for new truck/PHEV charging')
    expected = {f"charging_demand_{name.replace('-', '_')}_MWh"
                for name in counts if name.startswith('ev-')}
    components = set(source.columns) - {'hour_of_day', 'charging_demand_MWh'}
    if not components or not components.issubset(expected):
        raise ValueError('Unexpected charging components')
    values = source[sorted(components)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid charging energy')
    ldv = values.sum(axis=1)
    np.testing.assert_allclose(ldv, source.charging_demand_MWh)
    np.testing.assert_allclose(ldv.sum(), summary['daily_energy_MWh'])
    sample_bevs = sum(n for name, n in counts.items() if name.startswith('ev-'))
    expanded_bevs = sample_bevs * summary['fleet_scale_factor']
    if expanded_bevs <= 0:
        raise ValueError('No BEVs for normalization')
    result = pd.DataFrame(dict(hour_of_day=np.arange(24), LDV_native_MW=ldv,
        LDT_native_MW=np.zeros(24), MDT_native_MW=np.zeros(24), HDT_native_MW=np.zeros(24),
        LDV_native_kW_per_1000_BEV=ldv * 1e6 / expanded_bevs))
    fleet = float(evi['inputs']['fleet_size'])
    if evi['inputs']['pev_dist'] != 'BEV' or fleet <= 0:
        raise ValueError('Expected all-BEV EVI fleet')
    for day, kw in hourly_kw(evi['response']).items():
        result[f'EVI_{day}_kW_per_1000_BEV'] = kw / fleet * 1000
        result[f'EVI_{day}_MW_at_baseline_BEV_count'] = kw / fleet * expanded_bevs / 1000
    return result, sample_bevs, expanded_bevs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Baseline charging_demand directory')
    parser.add_argument('--evi-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source / 'native_ev_hourly_curve_10x.csv'
    summary_path = args.source / 'beam_charging_summary.json'
    evi_path = args.evi_reference / 'evi_sf_oakland_410000_response.json'
    manifest_path = args.evi_reference / 'comparison_manifest.json'
    inputs = [source_path, summary_path, evi_path, manifest_path]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    summary = json.loads(summary_path.read_text())
    evi = json.loads(evi_path.read_text())
    reference = json.loads(manifest_path.read_text())
    if evi['inputs'] != reference['evi_inputs']:
        raise ValueError('EVI response and manifest disagree')
    curves, sample_bevs, expanded_bevs = build_curves(pd.read_csv(source_path), summary, evi)
    # Independently confirm aggregation against the earlier saved comparison.
    previous = pd.read_csv(args.evi_reference / 'hourly_comparison.csv')
    for day in ('weekday', 'weekend'):
        np.testing.assert_allclose(curves[f'EVI_{day}_kW_per_1000_BEV'],
            previous[f'EVI SF-Oakland {day} kW_per_1000'])
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    curves.to_csv(out / 'ldv_truck_evi_hourly_comparison.csv', index=False)
    series = [('LDV native BEAM charging', 'LDV_native_kW_per_1000_BEV', '#2166ac', '-'),
              ('EVI weekday benchmark', 'EVI_weekday_kW_per_1000_BEV', '#d95f02', '--'),
              ('EVI weekend benchmark', 'EVI_weekend_kW_per_1000_BEV', '#238443', ':')]
    metrics = []
    for label, col, _, _ in series:
        v = curves[col].to_numpy()
        metrics.append(dict(series=label, daily_kWh_per_BEV=float(v.sum()/1000),
            peak_kW_per_1000_BEV=float(v.max()), peak_hour=int(v.argmax())))
    pd.DataFrame(metrics).to_csv(out / 'comparison_metrics.csv', index=False)
    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for label, col, color, style in series:
        v = curves[col].to_numpy()
        axes[0].plot(curves.hour_of_day, v, color=color, ls=style, lw=2.5, label=label)
        axes[1].plot(curves.hour_of_day, v/v.sum()*100, color=color, ls=style, lw=2.5)
    params = evi['inputs']
    axes[0].set_title('LDV existing-EV baseline vs EVI-Pro Lite\n'
        f"EVI CA / San Francisco-Oakland: {params['mean_dvmt']} mi/day, {params['temp_c']} C, 100% BEV")
    axes[0].set_ylabel('Hourly-average power (kW / 1,000 BEVs)')
    axes[0].legend(loc='upper left')
    axes[1].set_title('Daily-energy-normalized shape; no calibration applied')
    axes[1].set_ylabel('Share of daily charging energy (%)')
    axes[1].set_xlabel('Hour of day (BEAM simulation-local / EVI local)')
    for ax in axes:
        ax.set(xlim=(0, 23), xticks=range(24), ylim=(0, None))
        ax.grid(alpha=.25)
    fig.text(.5, .012,
        f'Native: {sample_bevs:,} BEVs x {summary["fleet_scale_factor"]:g} = {expanded_bevs:,.0f} BEVs; '
        f'{summary["daily_energy_MWh"]:.2f} MWh/day. EVI: saved 410,000-BEV scenario, normalized.\n'
        'Recorded 0-24h supplied charging only; not equilibrium charging needs. PHEVs have no recorded charging.',
        ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(out / 'ldv_native_vs_evi.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, color, marker) in enumerate(zip(('LDT', 'MDT', 'HDT'),
            ('#d95f02', '#238443', '#756bb1'), ('o', 's', '^'))):
        ax.plot(curves.hour_of_day, curves[f'{name}_native_MW'], color=color,
            lw=1.8, marker=marker, markevery=(i, 3), ms=7,
            label=f'{name}: 0 MW (no native electric-truck charging)')
    ax.set(title='Truck existing-EV baseline charging: LDT / MDT / HDT',
        xlabel='Hour of day (BEAM simulation-local)', ylabel='Hourly-average charging power (MW)',
        xlim=(0, 23), xticks=range(24), ylim=(-.04, .3), yticks=[0, .1, .2, .3])
    ax.legend(loc='upper left')
    ax.grid(alpha=.25)
    ax.text(.5, .36, 'All three curves coincide at zero.\n'
        'EVI-Pro Lite passenger profiles are not a truck benchmark.\n'
        'No hypothetical ICE-to-electric truck conversion is included.',
        transform=ax.transAxes, ha='center', va='center')
    fig.tight_layout()
    fig.savefig(out / 'truck_native_charging.png', dpi=180)
    plt.close(fig)
    for p in inputs:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.resolve())]
    metadata = dict(input_sha256=hashes, sample_BEVs=sample_bevs, expanded_BEVs=expanded_bevs,
        fleet_scale_factor=summary['fleet_scale_factor'], evi_inputs=params,
        evi_geographic_scope=reference['geographic_scope'],
        evi_retrieved_utc=evi['retrieved_utc'], evi_source=evi['endpoint'],
        EVI_cached_response_reused=True, calibration_applied=False, concentrations_modified=False,
        source_curve_modified=False, truck_native_charging_MWh=0,
        normalization='Native MW * 1e6 / expanded BEV fleet; EVI kW * 1000 / 410000',
        caveat='BEAM native delivered energy proxy and EVI modeled charging demand are not matched observations; '
               '35 mi/day, charger access, SOC, and electricity boundaries are not calibrated.')
    (out / 'comparison_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out / 'README.md').write_text(
        '# Existing-EV LDV / Truck charging comparison\n\n'
        'Uses the latest native-existing-EV baseline, not the older all-electric charging-code scenario.\n'
        f'LDV: {sample_bevs} sample BEVs; {expanded_bevs:g} BEVs after the existing 10x expansion. '
        'The curve is already expanded: no additional 10x factor is applied. '
        'Normalize by all BEVs, not only the 601 vehicles that charged. '
        '937 sample PHEVs have no recorded charging and are excluded from the BEV-only denominator.\n\n'
        'Truck: LDT, MDT, HDT native charging is zero; no native electric trucks. '
        'Do not reuse the older hypothetical truck-electrification demand here. '
        'EVI-Pro Lite passenger reference is not applied to trucks.\n\n'
        'EVI: reuse the saved official 410,000-BEV, 35 mile/day, 10 C response for the '
        'California / San Francisco-Oakland comparison. Six power components are summed and '
        'four 15-minute power bins averaged per hour, then divided by 410 to obtain kW/1,000 BEVs. '
        'This is a scenario benchmark, not measured charging, not a nine-county-specific simulation.\n\n'
        'CSV includes the actual expanded native MW and optional EVI MW at the native BEV fleet size '
        '(display-only linear normalization, not a new EVI simulation). '
        'Native daily charging is the recorded 0-24h supplied energy only; no inferred overnight '
        'top-up, no SOC balancing, and no UTC offset or peak shifting. '
        'No calibration, dispatch rerun, or concentration changes.\n', encoding='utf-8')
    print(json.dumps(dict(output=str(out), metrics=metrics), indent=2))


if __name__ == '__main__':
    main()
