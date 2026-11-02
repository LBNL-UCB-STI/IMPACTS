"""Read-only baseline diagnosis; save new arrival audit products, never calibrate."""
from pathlib import Path
import hashlib
import json

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from impacts.pipeline.workflow.charging.ldv_charging import (
    load_notebook_functions,
    prepare_existing_ev_stops,
)
from impacts.pipeline.workflow.power_generation import allocate_charging_event
from compare_native_evi_15k_weekday import weekday_hourly

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'examples/pilates/impacts/impacts_output/events-fc07-existing-ev-charging-code10x-20260928-v2'


def counts(frame):
    return (frame.arrivalTime // 3600 % 24).astype(int).value_counts().reindex(range(24), fill_value=0).sort_index().to_numpy()


def power(frame):
    result = np.zeros(24)
    for r in frame.itertuples():
        allocate_charging_event(result, start_hour=r.arrivalTime/3600,
                                duration_hours=r.timeCharging_s/3600, energy_kwh=r.energyCharged_kWh)
    return result


def profile_stats(values):
    x = np.asarray(values)
    return dict(total=float(x.sum()), peak_hour=int(x.argmax()), peak=float(x.max()),
                fraction_16_to_24=float(x[16:].sum()/x.sum()), fraction_18_to_24=float(x[18:].sum()/x.sum()))


def main():
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir()
                 if p.is_file() and not p.name.startswith('arrival_audit_')}
    summary = json.loads((OUT/'charging_code_summary.json').read_text())
    trips = pd.read_parquet(OUT/'existing_ev_trips.parquet')
    stops = pd.read_parquet(OUT/'prepared_ev_stops.parquet')
    audit = pd.read_parquet(OUT/'ev_stop_charging_audit.parquet')
    sessions = pd.read_parquet(OUT/'modeled_ev_charging_sessions.parquet')
    con = duckdb.connect()
    con.execute('SET threads=4')
    fresh = con.execute("""SELECT vehicle,vehicleType,driver,
        departureTime::DOUBLE departureTime,arrivalTime::DOUBLE arrivalTime,
        length::DOUBLE length,primaryFuelLevel::DOUBLE primaryFuelLevel
        FROM read_parquet(?) WHERE type='PathTraversal'
        AND (vehicleType LIKE 'ev-%' OR vehicleType LIKE 'phev-%')
        AND mode IN ('car','car_hov2','car_hov3','ride_hail','ride_hail_pooled')""",
        [summary['input_events']]).df()
    keys = trips.columns.tolist()
    pd.testing.assert_frame_equal(trips.sort_values(keys).reset_index(drop=True),
                                  fresh.sort_values(keys).reset_index(drop=True), check_dtype=False)
    drivers = pd.DataFrame({'driver': fresh.driver.dropna().unique()})
    con.register('drivers', drivers)
    activities = con.execute("""SELECT person,time::DOUBLE AS time,actType FROM read_parquet(?)
        WHERE type='actstart' AND person IN (SELECT driver FROM drivers)""", [summary['input_events']]).df()
    con.close()
    original, provenance = load_notebook_functions(ROOT/'chargingcode')
    assert provenance == summary['notebook_provenance']
    metadata = pd.read_csv(summary['assumptions']['battery_source'], low_memory=False)
    rebuilt, _ = prepare_existing_ev_stops(fresh, metadata, activities, original)
    pd.testing.assert_frame_equal(stops.sort_values('PTID').reset_index(drop=True),
                                  rebuilt.sort_values('PTID').reset_index(drop=True), check_dtype=False)
    selected = audit.loc[audit.retained & audit.energyCharged_kWh.gt(0)]
    np.testing.assert_array_equal(selected.PTID, sessions.PTID)
    comparison = pd.read_csv(OUT/'evi_weekday_comparison.csv')
    sample_kw = power(sessions)
    n = summary['input_total_vehicles']
    np.testing.assert_allclose(sample_kw*1000/n, comparison.charging_code_kW_per_1000_PEV)
    evi = json.loads(Path(summary['EVI_reference']).read_text())
    evi_kw, evi_parts = weekday_hourly(evi['response'])
    np.testing.assert_allclose(evi_kw/30, comparison.EVI_weekday_kW_per_1000_PEV)
    home = stops[stops.is_home]
    observed_home = home[home.activity_source.eq('events_driver_actstart')]
    last_home = home.sort_values('arrivalTime').groupby('vehicle').tail(1)
    home_sessions = sessions[sessions.is_home]
    hourly = pd.DataFrame({'hour': range(24)})
    sets = {'raw_arrivals_first24h':trips[trips.arrivalTime.lt(86400)],
            'raw_arrivals_all_folded24':trips, 'eligible_parking_arrivals':stops,
            'home_arrivals':home, 'observed_home_arrivals':observed_home,
            'last_home_arrival_per_vehicle':last_home, 'charging_starts':sessions,
            'home_charging_starts':home_sessions}
    for name, frame in sets.items():
        hourly[name] = counts(frame)
    hourly['code_kW_per1000'] = sample_kw*1000/n
    hourly['evi_kW_per1000'] = evi_kw/30
    hourly['code_energy_share_pct'] = sample_kw/sample_kw.sum()*100
    hourly['evi_energy_share_pct'] = evi_kw/evi_kw.sum()*100
    home_kw = power(home_sessions)
    hourly['code_home_kW_per1000'] = home_kw*1000/n
    hourly['code_nonhome_kW_per1000'] = (sample_kw-home_kw)*1000/n
    evi_home = evi_parts['home_l1']+evi_parts['home_l2']
    hourly['evi_home_kW_per1000'] = evi_home/30
    hourly['evi_nonhome_kW_per1000'] = (evi_kw-evi_home)/30
    hourly.to_csv(OUT/'arrival_audit_hourly.csv', index=False)
    by_activity = sessions.groupby(['powertrain','groupedActivityType','ChosenChargerType']).agg(
        sessions=('vehicle','size'), vehicles=('vehicle','nunique'), sample_kWh=('energyCharged_kWh','sum'),
        mean_duration_s=('timeCharging_s','mean'))
    by_activity.to_csv(OUT/'arrival_audit_charging_by_activity.csv')
    detail = audit.copy()
    detail['arrival_hour_absolute'] = detail.arrivalTime/3600
    detail['arrival_hour_folded'] = detail.arrivalTime//3600%24
    detail['charging_start_s'] = np.where(detail.retained & detail.energyCharged_kWh.gt(0), detail.arrivalTime, np.nan)
    detail.to_parquet(OUT/'arrival_audit_stop_details.parquet', index=False)
    weights = original['build_ptid_weights'](audit.copy(), l1_share=.2, l2_share=.8)
    weights['total_selection_weight'] = weights[['w_L1','w_L2','w_F']].sum(axis=1)
    weights[['vehicle','PTID','arrivalTime','parkingTime_s','powertrain','is_home','soc_kWh',
             'battery_kWh','w_L1','w_L2','w_F','total_selection_weight','attemptCharge']].to_csv(
                 OUT/'arrival_audit_selection_weights.csv', index=False)
    stats = {name: profile_stats(hourly[name]) for name in sets}
    eligible_retained_home = audit[audit.retained & audit.is_home]
    h18 = eligible_retained_home[eligible_retained_home.arrivalTime.between(18*3600,19*3600,inclusive='left')]
    metrics = dict(source_events=summary['input_events'], raw_events_exact_match=True,
        rebuilt_stops_exact_match=True, charging_curve_reconstructed_exactly=True,
        raw_trip_count=len(trips), fleet=n, scale=10,
        raw_trips_after24h=int(trips.arrivalTime.ge(86400).sum()),
        stops_after24h=int(stops.arrivalTime.ge(86400).sum()),
        home_proxies=int(home.activity_source.ne('events_driver_actstart').sum()),
        counts_and_peaks=stats,
        home18_retained_stops=len(h18), home18_charging_stops=int(h18.energyCharged_kWh.gt(0).sum()),
        code_home_energy_share=float(home_kw.sum()/sample_kw.sum()),
        evi_home_energy_share=float(evi_home.sum()/evi_kw.sum()),
        code=profile_stats(sample_kw*1000/n), evi=profile_stats(evi_kw/30),
        same_fleet_daily_energy_ratio=float(sample_kw.sum()*1000/n/(evi_kw.sum()/30)),
        same_fleet_peak_ratio=float(sample_kw.max()*1000/n/(evi_kw.max()/30)),
        code_daily_kwh_per_input_vehicle=float(sample_kw.sum()/n),
        evi_daily_kwh_per_input_vehicle=float(evi_kw.sum()/30000),
        code_peak_fraction_of_daily_energy=float(sample_kw.max()/sample_kw.sum()),
        evi_peak_fraction_of_daily_energy=float(evi_kw.max()/evi_kw.sum()),
        home_weights_below_1e_10=int((weights.is_home & weights.total_selection_weight.lt(1e-10)).sum()),
        selection_weight_caveat='Original exp(-(parking_hours*power/SOC_deficit - 1)^2) score can give long '
            'parking intervals very small weights. These are relative within-vehicle weights; a small absolute '
            'weight alone does NOT prove suppression or explain the aggregate discrepancy.',
        clocks='Native simulation seconds, no UTC conversion or DST shift. EVI clock alignment not independently established.',
        method='Arrival histograms count segments/stops, NOT unique fleet vehicles; last-home series is one per vehicle. '
               'Charging starts equal selected parking arrivals. 29h stop window folded modulo24; raw first24h also shown. '
               'Energy shares are diagnostic normalization only; no calibration or charging regeneration.',
        conclusion='Evening home-arrival and charging-start modes are 18h, while code power peaks at 19h. '
                   'Absence of evening EV activity does not explain mismatch. Charge selection, energy assumptions, '
                   'powertrain mix and home/nonhome allocation also differ. No causal attribution to one parameter is established.',
        caveats=['EVI raw vehicle arrival records are not available for direct travel-distribution comparison.',
                 'Home labels are driver activities, not geographically validated home locations.',
                 'Original SOC drift and excluded vehicles persist; this audit does not validate annual demand.'])
    (OUT/'arrival_audit_summary.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.size':11})
    fig, axes = plt.subplots(2,2,figsize=(15,10),layout='constrained')
    h = hourly.hour
    for col,label,style in [('raw_arrivals_first24h','Raw trip-segment arrivals (0-24h)','-'),
                            ('eligible_parking_arrivals','Parking arrivals >15 min (folded)','--')]:
        axes[0,0].plot(h,hourly[col],style,lw=2,label=label)
    axes[0,0].set(title='A. EV activity already has an evening arrival peak',ylabel='Arrivals / hour (sample)')
    for col,label,style in [('observed_home_arrivals','Observed home arrivals','-'),
                            ('last_home_arrival_per_vehicle','Last home arrival per vehicle','--'),
                            ('home_charging_starts','Home charging starts',':')]:
        axes[0,1].plot(h,hourly[col],style,lw=2,label=label)
    axes[0,1].set(title='B. Arriving home does not always trigger charging',ylabel='Arrivals or starts / hour (sample)')
    for col,label,style in [('code_kW_per1000','Charging code','-'),('evi_kW_per1000','EVI weekday','--')]:
        axes[1,0].plot(h,hourly[col],style,lw=2.3,label=label)
    axes[1,0].set(title='C. Same fleet denominator; no calibration',ylabel='Hourly-average kW / 1,000 PEVs')
    for col,label,style in [('code_energy_share_pct','Charging code','-'),('evi_energy_share_pct','EVI weekday','--')]:
        axes[1,1].plot(h,hourly[col],style,lw=2.3,label=label)
    axes[1,1].set(title='D. Shape only: each daily energy total = 100%',ylabel='Share of daily energy (%)')
    for ax in axes.flat:
        ax.set(xlim=(0,23),xticks=range(0,24,2),xlabel='Hour (native clock; no time shift)',ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Baseline existing EVs: arrival timing vs modeled charging and EVI',fontsize=17)
    fig.supxlabel('1,544 input PEVs (607 BEVs + 937 PHEVs); arrival counts are unscaled.\n'
                  'EVI: weekday, 35 mi/day, 10 C, Home80. SOC drift / fleet-mixture differences remain.',fontsize=10)
    fig.savefig(OUT/'arrival_audit_ev_activity_vs_evi.png',dpi=170)
    plt.close(fig)
    report = '''Baseline EV arrival audit

Scope: existing BEV/PHEV trips from original 3.events.parquet; NOT all vehicles electrified.
Original events extraction and activity-derived parking stops exactly match saved baseline.
Original charging curve is exactly reconstructed from saved charging sessions.
No calibration, time shift, emission/concentration calculation, or original result modification.

Findings (sample counts; arrival counts are NOT unique vehicle counts):
  Input: 1,544 PEVs, 12,401 PathTraversal segments, 5,660 eligible parking stops.
  First-24-hour raw arrival mode: 18:00-19:00, 1,028 segments.
  Observed home-arrival mode: 18:00-19:00, 273 stops.
  Last home arrival per vehicle mode: 18:00-19:00, 243 vehicles.
  Charging-start mode: 18:00-19:00, 81 sessions (71 at home).
  71 / 273 = 26.0% of retained home stops at 18:00-19:00 actually charge.
  Charging-code power peak: 19:00-20:00; EVI hourly-average peak: 18:00-19:00.
  Code home energy share: 61.1%; EVI: 74.9%.
  Code 18:00-24:00 energy share: 33.3%; EVI: 46.9%.
  Same-fleet code daily energy: 42.4% of EVI; peak: 34.8% of EVI.
  After independently normalizing daily energy to 100%, peak-hour shares are 8.07% vs 9.84%.

Interpretation:
There is a clear evening home-arrival peak in the events, so absent evening activity does
not explain the charging mismatch. The power peak is not the same statistic as the
arrival peak: code charging starts immediately at selected arrivals and lasts until full
or departure, causing overlap across hours. No explicit one-hour delay was applied.
Energy magnitude and allocation differ in addition to timing. Check PHEV 0.017 kWh/km
proxy, BEV/PHEV mix, initial/final SOC drift, selected charging fraction, home preference,
and assumed non-home fast-charger availability before calibrating arrival times.

Original stop-selection scores can favor particular dwell times relative to SOC deficit;
476 home stops have total weights below 1e-10. Weights are relative within each vehicle,
so this is a sensitivity-test candidate, NOT proof of aggregate causal suppression.

Limitations:
EVI raw arrival records were not supplied; direct travel-distribution matching is unavailable.
Native simulation-clock hours are used with no DST/UTC shift; EVI clock convention is not
independently established here. Home labels are matched driver activities (25 proxy stops),
not a home-location geospatial validation. Original SOC drift and exclusions remain.
'''
    (OUT/'arrival_audit_report.txt').write_text(report, encoding='utf-8')
    for p, sha in protected.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == sha, f'Existing result changed: {p}'
    print(json.dumps(metrics,indent=2))


if __name__ == '__main__':
    main()
