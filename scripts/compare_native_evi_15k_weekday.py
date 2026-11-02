"""Native baseline versus EVI weekday: PEV denominator, Home80, 15k display fleet."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

API = 'https://developer.nlr.gov/api/evi-pro-lite/v1/daily-load-profile.json'


def weekday_hourly(response):
    parts = response['results']['weekday_load_profile']
    expected = {'home_l1', 'home_l2', 'work_l1', 'work_l2', 'public_l2', 'public_l3'}
    if set(parts) != expected:
        raise ValueError('Unexpected EVI weekday components')
    data = {k: np.asarray(v, dtype=float) for k, v in parts.items()}
    if any(v.shape != (96,) or not np.isfinite(v).all() or (v < 0).any() for v in data.values()):
        raise ValueError('Invalid EVI power bins; null components are not zero')
    hourly = {k: v.reshape(24, 4).mean(axis=1) for k, v in data.items()}
    total = sum(hourly.values())
    np.testing.assert_allclose(total.sum(), sum(v.sum() for v in data.values()) / 4)
    return total, hourly


def normalized_curves(native_mw, expanded_pevs, evi_kw, evi_fleet, target=15000):
    if min(expanded_pevs, evi_fleet, target) <= 0:
        raise ValueError('Fleet counts must be positive')
    native = np.asarray(native_mw, dtype=float)
    evi = np.asarray(evi_kw, dtype=float)
    if native.shape != (24,) or evi.shape != (24,):
        raise ValueError('Expected 24 hours')
    if not np.isfinite([native, evi]).all() or (native < 0).any() or (evi < 0).any():
        raise ValueError('Invalid hourly power')
    return pd.DataFrame(dict(hour_of_day=np.arange(24),
        events_original_expanded_MW=native,
        events_kW_per_1000_PEV=native * 1e6 / expanded_pevs,
        EVI_weekday_kW_per_1000_PEV=evi * 1000 / evi_fleet,
        events_MW_at_15000_PEV=native * target / expanded_pevs,
        EVI_weekday_MW_at_15000_PEV=evi * target / evi_fleet / 1000))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = args.source / 'native_ev_hourly_curve_10x.csv'
    summary_path = args.source / 'beam_charging_summary.json'
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, summary_path)}
    summary = json.loads(summary_path.read_text())
    frame = pd.read_csv(source)
    if frame.hour_of_day.tolist() != list(range(24)):
        raise ValueError('Unexpected source hours')
    counts = summary['electric_road_vehicle_counts']
    if any(not k.startswith(('ev-', 'phev-')) for k in counts):
        raise ValueError('Unknown electric vehicle class')
    sample_pevs = sum(counts.values())
    assert sample_pevs == summary['electric_road_vehicles']
    expanded_pevs = sample_pevs * summary['fleet_scale_factor']
    native = frame.charging_demand_MWh.to_numpy()
    np.testing.assert_allclose(native.sum(), summary['daily_energy_MWh'])
    params = dict(fleet_size=15000, mean_dvmt=35, temp_c=10, pev_dist='BEV',
        class_dist='Sedan', home_access_dist='HA100', home_power_dist='MostL2',
        work_power_dist='MostL2', pref_dist='Home80', res_charging='min_delay', work_charging='min_delay')

    def query(inputs, filename):
        response = requests.get(API, params={**inputs, 'api_key': 'DEMO_KEY'}, timeout=45)
        try:
            payload = response.json()
        except ValueError:
            payload = {'raw_text': response.text}
        record = dict(endpoint=API, inputs=inputs, retrieved_utc=datetime.now(timezone.utc).isoformat(),
            http_status=response.status_code, response=payload)
        (out / filename).write_text(json.dumps(record, indent=2), encoding='utf-8')
        return response, payload

    response, payload = query(params, 'evi_request_15000.json')
    fallback = False
    if response.status_code in (400, 422) and any(
            'Invalid fleet_size' in str(error) for error in payload.get('errors', [])):
        fallback = True
        params = {**params, 'fleet_size': 30000}
        response, payload = query(params, 'evi_response_30000.json')
    response.raise_for_status()
    evi, parts = weekday_hourly(payload)
    if int(payload['metadata']['inputs']['fleet_size']) != params['fleet_size']:
        raise ValueError('EVI accepted fleet size differs from request')
    curves = normalized_curves(native, expanded_pevs, evi, params['fleet_size'])
    for key, values in parts.items():
        curves[f'EVI_{key}_MW_at_15000_PEV'] = values * 15000 / params['fleet_size'] / 1000
    curves.to_csv(out / 'weekday_hourly_comparison.csv', index=False)
    metrics = []
    for label, col in [('Events recorded charging', 'events'), ('EVI weekday Home80', 'EVI_weekday')]:
        v = curves[f'{col}_MW_at_15000_PEV'].to_numpy()
        per = curves[f'{col}_kW_per_1000_PEV'].to_numpy()
        metrics.append(dict(series=label, fleet_PEV=15000, daily_MWh=float(v.sum()),
            daily_kWh_per_PEV=float(v.sum()*1000/15000), peak_MW=float(v.max()),
            peak_hour=int(v.argmax()), peak_kW_per_1000_PEV=float(per.max())))
        np.testing.assert_allclose(per * 15 / 1000, v)
    home_share = float((parts['home_l1'].sum()+parts['home_l2'].sum()) / evi.sum())
    pd.DataFrame(metrics).to_csv(out / 'comparison_metrics.csv', index=False)

    plt.rcParams.update({'font.size': 11})
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    def draw(ax, suffix, title, ylabel):
        ax.plot(curves.hour_of_day, curves[f'events_{suffix}'], lw=2.6,
            color='#2166ac', label='Events: recorded BEV + PHEV fleet charging')
        ax.plot(curves.hour_of_day, curves[f'EVI_weekday_{suffix}'], lw=2.6, ls='--',
            color='#d95f02', label='EVI weekday: Home80, BEV-dominant mix')
        ax.set(title=title, ylabel=ylabel, xlim=(0, 23), xticks=range(24), ylim=(0, None))
        ax.grid(alpha=.25)
        ax.legend(loc='upper left', fontsize=10)
    draw(axes[0], 'kW_per_1000_PEV', 'Per 1,000 plug-in vehicles (BEV + PHEV)',
        'Hourly-average power (kW / 1,000 PEVs)')
    draw(axes[1], 'MW_at_15000_PEV', 'Total charging load at a common fleet of 15,000 PEVs',
        'Hourly-average total charging power (MW)')
    axes[1].set_xlabel('Hour of day (BEAM simulation-local / EVI local)')
    fig.suptitle('Existing-EV baseline vs EVI-Pro Lite: weekday comparison\n'
        'SF-Oakland reference | 35 mi/day | 10 C | 80% prefer home charging', fontsize=15)
    foot = (f'Events: {expanded_pevs:,.0f} PEVs after 10x expansion, displayed at 15,000; '
        'PHEVs have no recorded charging.\n'
        f'EVI: {params["fleet_size"]:,}-PEV API profile' + (' scaled to 15,000' if fallback else '') +
        '; 75% BEV / 25% PHEV. Home80 is preference, not energy share. No calibration.')
    fig.text(.5, .012, foot, ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .067, 1, .935))
    fig.savefig(out / 'weekday_per1000_and_total.png', dpi=180)
    plt.close(fig)
    for suffix, title, ylabel, name in [
        ('kW_per_1000_PEV', 'Weekday charging comparison per 1,000 PEVs', 'Hourly-average power (kW / 1,000 PEVs)', 'weekday_per1000.png'),
        ('MW_at_15000_PEV', 'Weekday total charging comparison: 15,000 PEVs', 'Hourly-average charging power (MW)', 'weekday_total_15000.png')]:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        draw(ax, suffix, title+'\nEVI: SF-Oakland reference, 35 mi/day, 10 C, Home80', ylabel)
        ax.set_xlabel('Hour of day')
        fig.text(.5, .014, foot, ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .10, 1, 1))
        fig.savefig(out / name, dpi=180)
        plt.close(fig)
    for p in (source, summary_path):
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.resolve())]
    manifest = dict(input_sha256=hashes, input_events=summary['events_file'],
        target_fleet=15000, sample_PEVs=sample_pevs, expanded_PEVs=expanded_pevs,
        native_BEVs=sum(v for k,v in counts.items() if k.startswith('ev-')),
        native_PHEVs=sum(v for k,v in counts.items() if k.startswith('phev-')),
        evi_inputs=params, evi_direct_15000_unavailable=fallback,
        evi_phev_bev_mix_pct=dict(PHEV20=10, PHEV50=15, BEV100=25, BEV250=50),
        evi_sedan_suv_mix_pct=dict(Sedan=80,SUV=20),
        evi_home_daily_energy_share=home_share,
        native_daily_MWh_original=float(native.sum()), metrics=metrics,
        native_display_multiplier=15000/expanded_pevs,
        calibration_applied=False, concentration_modified=False, weekend_comparison=False,
        warning='Recorded supplied charging, not equilibrium charging needs; fleets differ in BEV/PHEV mix. '
        'Home80 is an EVI preference, not 80% home-energy constraint. No native event locations/times changed.')
    (out / 'comparison_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (out / 'README.md').write_text(
        '# Weekday native events versus EVI: 15,000 PEVs, Home80\n\n'
        'Both denominators include BEV and PHEV, not only vehicles that charged. '
        f'Native sample has {sample_pevs} PEVs; existing 10x expansion gives {expanded_pevs:g}. '
        'The original hourly demand remains unchanged; the comparison displays a common 15,000-PEV fleet. '
        'Both total-power and per-1,000 curves are exported. No weekend comparison.\n\n'
        f'EVI API request fleet: {params["fleet_size"]}. 15,000 unsupported fallback used: {fallback}. '
        'When a 30,000 profile is used it is multiplied by 0.5 for display, not described as a direct 15,000 simulation. '
        'Full requests and responses, including any rejection, are saved here.\n\n'
        'EVI inputs: 35 mi/day, 10 C, BEV-dominant mix (10/15/25/50% PHEV20/PHEV50/BEV100/BEV250), '
        '80/20% Sedan/SUV, 100% home access, 80/20% L2/L1 at both home and work, '
        'Home80 preference, immediate home/work charging. No unsupported pev_type override. '
        'Regional reference is CA / San Francisco-Oakland urban area; this is not a geographic API argument.\n\n'
        f'Actual EVI home share of daily charging energy: {home_share:.2%}; Home80 does not fix that share to 80%. '
        'Native PHEVs have no recorded charging; native and EVI powertrain mixtures are different. '
        'This is a scenario comparison, not matched validation or calibration. '
        'Original events, charging inputs, dispatch and concentration outputs are untouched.\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
