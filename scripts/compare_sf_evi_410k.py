"""SF-Oakland EVI comparison: 410,000 EVs, normalized by total fleet, no fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

API = 'https://developer.nlr.gov/api/evi-pro-lite/v1/'


def fetch(name, params, out, filename, cache=None):
    cached = cache/filename if cache else None
    if cached and cached.exists():
        saved=json.loads(cached.read_text())
        if saved['inputs']!=params or saved['endpoint']!=API+name:
            raise ValueError('Cached response does not match requested parameters')
        data=saved['response']
    else:
        response = requests.get(API + name, params={**params, 'api_key': 'DEMO_KEY'}, timeout=45)
        response.raise_for_status()
        data = response.json()
    (out/filename).write_text(json.dumps(dict(endpoint=API+name, inputs=params,
        retrieved_utc=saved['retrieved_utc'] if cached and cached.exists() else datetime.now(timezone.utc).isoformat(), response=data), indent=2), encoding='utf-8')
    if 'results' not in data:
        raise ValueError(f'No results in {filename}')
    return data


def hourly_kw(data):
    output = {}
    for key in ['weekday_load_profile', 'weekend_load_profile']:
        arrays = [np.asarray(v, dtype=float) for v in data['results'][key].values()]
        if len(arrays) != 6 or any(a.shape != (96,) or not np.isfinite(a).all() or (a < 0).any() for a in arrays):
            raise ValueError('Expected six nonnegative 96-bin charging-power components')
        total = np.sum(arrays, axis=0)
        output[key.split('_')[0]] = total.reshape(24, 4).mean(axis=1)
        np.testing.assert_allclose(output[key.split('_')[0]].sum(), total.sum()/4)
    return output


def describe(name, curve, target=410000):
    values = np.asarray(curve)
    return dict(series=name, daily_kWh_per_1000=values.sum(),
                daily_kWh_per_vehicle=values.sum()/1000, peak_kW_per_1000=values.max(),
                peak_hour=int(values.argmax()), trough_kW_per_1000=values.min(),
                trough_hour=int(values.argmin()), peak_to_trough=values.max()/values.min(),
                daily_MWh_410000=values.sum()*target/1e6, peak_MW_410000=values.max()*target/1e6)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--api-cache', type=Path)
    p.add_argument('--mean-dvmt', type=int, choices=[25, 35, 45], default=None,
                   help='Explicit mileage scenario override; otherwise use official urban default.')
    args=p.parse_args()
    out=args.output.resolve();out.mkdir(parents=True, exist_ok=False)
    summary=json.loads((args.source/'charging_reuse_summary.json').read_text())
    curve_file=args.source/'hourly_charging_curve.csv'
    source=pd.read_csv(curve_file)
    if source.hour_of_day.tolist()!=list(range(24)):
        raise ValueError('Expected 24 sorted hourly bins')
    sample_fleet=int(summary['modeled_vehicles'])
    # Denominator is ALL modeled passenger vehicles, not retained/charging vehicles.
    ours=source.passenger_MW.to_numpy()*1e6/sample_fleet
    stats=fetch('stats-by-urban-area.json', {'state':'CA','urban_area':'San Francisco - Oakland'}, out, 'sf_oakland_official_stats.json',args.api_cache)
    city=stats['results']
    if city['urban_area']!='San Francisco - Oakland':raise ValueError('Wrong urban area')
    # Reproduce setUrbanInfo in official widget: closest allowed Fahrenheit value;
    # ties retain ascending positive object-key order (50F before 68F).
    temp_lookup={14:-10,32:0,50:10,68:20,86:30,104:40,-4:-20}
    selected_f=min(temp_lookup, key=lambda f:abs(city['mean_temp_f']-f))
    temp=temp_lookup[selected_f]
    mean_dvmt=city['mean_dvmt'] if args.mean_dvmt is None else args.mean_dvmt
    params=dict(fleet_size=410000, mean_dvmt=mean_dvmt, temp_c=temp,
                pev_type='BEV250', pev_dist='BEV', class_dist='Sedan', home_access_dist='HA100',
                home_power_dist='MostL2', work_power_dist='MostL2', pref_dist='Home80',
                res_charging='min_delay', work_charging='min_delay')
    if params['fleet_size']>=city['ldv_count']:raise ValueError('Fleet exceeds urban LDV count')
    raw=fetch('daily-load-profile.json',params,out,'evi_sf_oakland_410000_response.json',args.api_cache)
    if int(raw['metadata']['inputs']['fleet_size'])!=410000:raise ValueError('API did not accept requested fleet')
    evi_full=hourly_kw(raw)
    # A separate 1,000-vehicle request is a sensitivity check, not assumed to be
    # the same finite-fleet profile. Null components are NOT imputed as zero.
    check=fetch('daily-load-profile.json',{**params,'fleet_size':1000},out,'evi_sf_oakland_1000_check_response.json',args.api_cache)
    common_errors={}; missing_components={}
    for day,parts in raw['results'].items():
        comparison=check['results'][day]
        missing_components[day]=[key for key,value in comparison.items() if value is None]
        common_errors[day]={key:float(np.max(np.abs(np.asarray(value)/410-np.asarray(comparison[key]))))
                           for key,value in parts.items() if comparison[key] is not None}
    curves={'Charging code (uncalibrated)':ours}
    errors={}
    for day,full in evi_full.items():
        per1000=full/410
        errors[day]=float(abs(per1000*410-full).max())
        np.testing.assert_allclose(per1000*410,full,rtol=0,atol=1e-8)
        curves[f'EVI SF-Oakland {day}']=per1000
    hourly=pd.DataFrame({'hour_of_day':np.arange(24)})
    metrics=[]
    for name,values in curves.items():
        hourly[name+' kW_per_1000']=values
        hourly[name+' MW_410000']=values*410/1000
        item=describe(name,values)
        if name.startswith('EVI'):
            item['charging_code_energy_difference_pct']=(ours.sum()/values.sum()-1)*100
            item['hourly_correlation']=float(np.corrcoef(ours,values)[0,1])
            item['hourly_rmse_kW_per_1000']=float(np.sqrt(np.mean((ours-values)**2)))
            item['shape_total_variation']=float(np.abs(ours/ours.sum()-values/values.sum()).sum()/2)
            item['charging_code_peak_lag_hours']=int(ours.argmax()-values.argmax())
        metrics.append(item)
    hourly.to_csv(out/'hourly_comparison.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'comparison_metrics.csv',index=False)
    manifest=dict(target_fleet=410000, sample_fleet=sample_fleet,
        source=str(curve_file.resolve()), source_sha256=hashlib.sha256(curve_file.read_bytes()).hexdigest(),
        regional_stats=city, web_selected_temp_c=temp, mean_city_temp_c=(city['mean_temp_f']-32)*5/9,
        evi_inputs=params, normalization='kW / total fleet * 1000; not charging-vehicle count',
        requested_mean_dvmt_override=args.mean_dvmt,
        calibration_applied=False, source_curve_modified=False,
        matched_410k_ours='Presentation only: original sample curve * 410000 / modeled_vehicles',
        normalization_round_trip_max_kW_error=errors,
        api_1000_missing_components=missing_components,
        api_1000_common_component_difference_kW_per_1000=common_errors,
        geographic_scope='California / San Francisco - Oakland urban area, not nine-county SFBAY',
        regional_effect='Official web application maps city to mean_dvmt and temperature bin; city is not a daily-load-profile API argument.',
        nondefault_controls='100% BEV250 Sedan, HA100, MostL2, Home80, immediate charging retained from prior comparison; not unchanged web base-case.',
        sample_miles_per_vehicle=summary['modeled_daily_distance_km']/sample_fleet/1.609344,
        source_soc_drift_MWh=summary['retained_soc_change_MWh'],
        warning='Single-run passenger SOC drift remains. This is not a validated steady-state annual electricity demand.')
    (out/'comparison_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(2,1,figsize=(11,9),sharex=True)
    for i,(name,values) in enumerate(curves.items()):
        style=dict(lw=2.5 if i==0 else 2.,ls='-' if i==0 else '--',label=name)
        axes[0].plot(np.arange(24),values,**style)
        axes[1].plot(np.arange(24),values/values.sum()*100,**style)
    axes[0].set(title=f'410,000-EV scenario: passenger load per 1,000 fleet vehicles\nEVI California / San Francisco-Oakland: {mean_dvmt} mi/day, {temp}C, 100% BEV',ylabel='Hourly-average power (kW / 1,000 vehicles)')
    axes[1].set(title='Shape comparison only; no calibration applied',ylabel='Share of daily charging energy (%)',xlabel='Hour of day',xticks=range(0,24,2))
    axes[0].legend()
    for ax in axes:ax.grid(alpha=.25);ax.set_ylim(bottom=0)
    fig.tight_layout();fig.savefig(out/'sf_oakland_410000_per1000_comparison.png',dpi=180);plt.close(fig)
    (out/'README.md').write_text(f'''# 410,000 EVs: SF-Oakland EVI comparison, no calibration

The official urban statistics endpoint was queried for CA / San Francisco - Oakland.
The official web application's regional parameter logic selects {city['mean_dvmt']} miles/day and
{temp}C (city annual mean 59F/15C; 10C wins the nearest-bin tie with 20C).
This run uses {mean_dvmt} miles/day. Explicit user mileage override: {args.mean_dvmt}.
Other controls retain the earlier all-BEV comparison: BEV250 Sedan, HA100, MostL2,
Home80, immediate home/work charging. This is NOT the unmodified web base case.

The 410,000-vehicle profile was requested directly from the official daily-load-profile
API. A second direct 1,000-vehicle request is saved as a sensitivity check; any null
component arrays are NOT interpreted as zeros. It is not used as the
reference. Normalization is verified by round-trip recovery of the 410,000 curve.
Six 15-minute kW components are summed, then averaged into hourly kW. Daily energy
is the sum of hourly kW times one hour. Raw API responses and inputs are preserved.

Charging-code denominator is all 413,067 modeled passenger vehicles, not 203,330
charging vehicles. Comparison per 1,000 is P_sample * 1000 / 413067. For display at
410,000 vehicles only, P_sample is multiplied by 410000/413067. No trip simulation,
charging probabilities, SOC, timing, truck curves or air-quality results were changed.

The sample averages 36.70 miles/vehicle; this EVI scenario uses {mean_dvmt}. Hence absolute
energy differences are not solely algorithm errors. Original SOC drift also persists.
The share-of-daily-energy panel is descriptive normalization, not profile calibration.
San Francisco-Oakland urban area is not the entire nine-county Bay Area.
EVI is a modeled reference scenario, not measured charging ground truth. No raw
BEAM events are altered and no fitted parameters or calibrated curves are produced.
''',encoding='utf-8')
    print(json.dumps(manifest,indent=2),flush=True)
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


if __name__=='__main__':main()
