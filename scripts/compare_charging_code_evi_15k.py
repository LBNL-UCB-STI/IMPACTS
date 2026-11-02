"""Compare modeled charging-code load (NOT native refuel events) with EVI weekday."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_native_evi_15k_weekday import weekday_hourly, normalized_curves


def prepare(source, summary, evi):
    if source.hour_of_day.tolist() != list(range(24)):
        raise ValueError('Expected 24 sorted hourly bins')
    if not summary['original_core_functions_reused'] or summary['calibrated_to_evi']:
        raise ValueError('Expected original-core, uncalibrated charging-code result')
    if summary['fleet_scale_factor'] != 1:
        raise ValueError('Review modeled fleet denominator for expanded source')
    np.testing.assert_allclose(source.passenger_MW.sum(), summary['passenger_daily_supplied_MWh'])
    fleet = int(summary['modeled_vehicles'])
    params = evi['inputs']
    if params['pref_dist'] != 'Home80' or params['mean_dvmt'] != 35 or params['temp_c'] != 10:
        raise ValueError('Unexpected EVI scenario')
    kw, _ = weekday_hourly(evi['response'])
    result = normalized_curves(source.passenger_MW, fleet, kw, params['fleet_size'])
    result.columns = [c.replace('events_original_expanded_MW', 'charging_code_original_sample_MW')
                      .replace('events_', 'charging_code_').replace('_PEV', '_vehicles') for c in result.columns]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--evi-response', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    curve_file = args.source / 'hourly_charging_curve.csv'
    summary_file = args.source / 'charging_reuse_summary.json'
    inputs = (curve_file, summary_file, args.evi_response)
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    summary = json.loads(summary_file.read_text())
    evi = json.loads(args.evi_response.read_text())
    curves = prepare(pd.read_csv(curve_file), summary, evi)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    curves.to_csv(out / 'weekday_hourly_comparison.csv', index=False)
    metrics = []
    for name, prefix in [('Charging code (uncalibrated)', 'charging_code'), ('EVI weekday Home80', 'EVI_weekday')]:
        v = curves[f'{prefix}_MW_at_15000_vehicles'].to_numpy()
        per = curves[f'{prefix}_kW_per_1000_vehicles'].to_numpy()
        np.testing.assert_allclose(per * 15 / 1000, v)
        metrics.append(dict(series=name, daily_MWh_at_15000=float(v.sum()),
            daily_kWh_per_vehicle=float(v.sum()/15), peak_MW_at_15000=float(v.max()),
            peak_kW_per_1000=float(per.max()), peak_hour=int(v.argmax())))
    pd.DataFrame(metrics).to_csv(out / 'comparison_metrics.csv', index=False)
    plt.rcParams.update({'font.size': 11})
    def draw(ax, suffix, title, ylabel):
        ax.plot(curves.hour_of_day, curves[f'charging_code_{suffix}'], lw=2.7,
            color='#2166ac', label='Charging code: modeled demand (uncalibrated)')
        ax.plot(curves.hour_of_day, curves[f'EVI_weekday_{suffix}'], lw=2.7, ls='--',
            color='#d95f02', label='EVI weekday: Home80')
        ax.set(title=title, ylabel=ylabel, xlabel='Hour of day (local)',
            xlim=(0, 23), xticks=range(24), ylim=(0, None))
        ax.grid(alpha=.25)
        ax.legend(loc='upper left', fontsize=10)
    foot = ('Charging code: all-passenger electrification scenario; 413,067 modeled vehicles, normalized to 15,000.\n'
        'EVI: 30,000-PEV profile x 0.5; 75% BEV / 25% PHEV. No native charging records used.\n'
        'Fleet mixes differ; charging-code daily SOC is not balanced. No calibration or concentration changes.')
    panels = [('kW_per_1000_vehicles', 'Charging demand per 1,000 fleet vehicles',
               'Hourly-average power (kW / 1,000 vehicles)', 'weekday_per1000.png'),
              ('MW_at_15000_vehicles', 'Total charging load at 15,000 fleet vehicles',
               'Hourly-average total power (MW)', 'weekday_total_15000.png')]
    fig, axes = plt.subplots(2, 1, figsize=(12, 9.5), layout='constrained')
    for ax, (suffix, title, ylabel, _) in zip(axes, panels):
        draw(ax, suffix, title, ylabel)
    fig.suptitle('Charging code vs EVI-Pro Lite: weekday only\n'
        'EVI SF-Oakland reference | 35 mi/day | 10 C | Home80', fontsize=15)
    fig.supxlabel(foot, fontsize=9)
    fig.savefig(out / 'charging_code_vs_evi_weekday.png', dpi=180)
    plt.close(fig)
    for suffix, title, ylabel, name in panels:
        fig, ax = plt.subplots(figsize=(12, 6), layout='constrained')
        draw(ax, suffix, title+'\nEVI: 35 mi/day, 10 C, Home80; weekday only', ylabel)
        fig.supxlabel(foot, fontsize=9)
        fig.savefig(out / name, dpi=180)
        plt.close(fig)
    metadata = dict(source_kind='Charging-code modeled demand, NOT native BEAM recorded charging',
        input_sha256=hashes, modeled_passenger_fleet=summary['modeled_vehicles'], target_fleet=15000,
        code_assumptions=summary['assumptions'], evi_inputs=evi['inputs'],
        normalization='Code MW / 413067 * 1e6 = kW/1000; EVI kW / 30000 * 1000; total MW = per1000 * 15/1000',
        reused_saved_charging_code_output=True, reran_charging_model=False,
        code_daily_SOC_gain_MWh=summary['retained_soc_change_MWh'],
        calibration_applied=False, native_charging_records_used=False, concentration_modified=False,
        metrics=metrics, limitations=summary['limitations'])
    (out / 'comparison_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (out / 'README.md').write_text(
        '# Corrected comparison: charging CODE versus EVI, not native refuel records\n\n'
        'Reuses the original-core charging-code diagnostic output from the same 3.events.parquet travel input. '
        'This is not a new simulation restricted to the 607 native BEVs and 937 PHEVs. '
        'The code modeled 413,067 passenger vehicles as electric using 75 kWh batteries and 0.17 kWh/km. '
        'The denominator is all modeled vehicles, not only vehicles that charged. '
        'Per-1,000 power is sample MW * 1,000,000 / 413067; 15,000-vehicle power is sample MW * 15000 / 413067. '
        'No extra 10x expansion is applied for this normalized comparison.\n\n'
        'EVI reuses the saved successful 30,000-vehicle API response: 35 mi/day, 10 C, Home80, HA100, '
        'MostL2 home/work, immediate charging, BEV-dominant mix (75% BEV, 25% PHEV), Sedan-dominant (80/20). '
        '15,000 is unsupported directly, so total EVI power is multiplied by 0.5. '
        'Home80 is a location preference, not an 80% home energy constraint. '
        'Six 15-minute component power curves are summed, then averaged hourly. No weekend plotted.\n\n'
        'This supersedes the native-event plot for the purpose of validating CHARGING CODE. '
        'It does not supersede baseline concentration inputs, rerun dispatch, or change any prior results. '
        'It is a scenario-level comparison: different vehicle mixtures and positive charging-code SOC drift '
        '(2696.07 MWh for the modeled sample) prevent treating the difference as a calibrated model error. '
        'The prior native-record comparison is retained solely for provenance.\n', encoding='utf-8')
    for p in inputs:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.resolve())]
    print(json.dumps(dict(output=str(out), metrics=metrics), indent=2))


if __name__ == '__main__':
    main()
