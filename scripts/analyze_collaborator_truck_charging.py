"""Diagnostic reconstruction only; preserve source CSVs and all flagged events.

Reported energy is not independently verified grid-side energy. H2FC is kept
separate and must not be used as an electric-grid forecast. No fleet expansion,
24-hour folding, loss adjustment, dispatch or concentration calculation.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#0072B2', '#D55E00', '#009E73']
CLASSES = ['LD1', 'LD3', 'MDV', 'HDV', 'HDT']


def allocate(start_h, energy_kwh, power_kw, n_hours):
    """Exact interval overlap; output kWh per one-hour interval [h,h+1)."""
    if not all(np.isfinite([start_h, energy_kwh, power_kw])):
        raise ValueError('Nonfinite charging input')
    if start_h < 0 or energy_kwh < 0 or power_kw <= 0:
        raise ValueError('Invalid charging input')
    result = np.zeros(n_hours)
    end_h = start_h + energy_kwh / power_kw
    if end_h > n_hours + 1e-10:
        raise ValueError('Output horizon truncates event energy')
    for h in range(math.floor(start_h), min(n_hours, math.ceil(end_h))):
        result[h] = power_kw * max(0., min(end_h, h+1) - max(start_h, h))
    np.testing.assert_allclose(result.sum(), energy_kwh, rtol=1e-10, atol=1e-8)
    return result


def self_test():
    np.testing.assert_allclose(allocate(.5, 150, 100, 3), [50, 100, 0])
    a = allocate(23.5, 100, 100, 26)
    np.testing.assert_allclose(a[23:26], [50, 50, 0])
    np.testing.assert_allclose(allocate(1., 100, 100, 3), [0, 100, 0])
    assert allocate(0, 0, 100, 3).sum() == 0
    try:
        allocate(23.5, 100, 100, 24)
    except ValueError:
        pass
    else:
        raise AssertionError('Truncation must fail')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def savefig(fig, out, name, note):
    fig.text(.5, .015, note, ha='center', va='bottom', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .95))
    fig.savefig(out / (name+'.png'), dpi=180)
    plt.close(fig)


def axis_style(ax, xmax=24):
    ax.set_xlim(0, xmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Simulation elapsed hour (not timezone-converted)')
    ax.set_xticks(np.arange(0, xmax+1, 3 if xmax==24 else 6))
    ax.grid(axis='y', alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'chargingcode/ElectricTruck_event')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    self_test()
    paths = sorted(args.input.glob('charging_events_*.csv'))
    if len(paths) != 3:
        raise ValueError(f'Expected three input scenarios; found {len(paths)}')
    out = args.output or ROOT/'OUTPUT'/('collaborator-truck-charging-curves-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True, exist_ok=False)
    data, source_hashes = {}, {}
    for path in paths:
        source_hashes[str(path.resolve())] = sha(path)
        d = pd.read_csv(path)
        if d.scenario.nunique() != 1 or not set(d.powerTrain).issubset({'BE','H2FC'}):
            raise ValueError('Unexpected scenario or powertrain')
        for col in ['time','energy_needed_kwh','actual_charge_speed_kw','charge_time_minutes']:
            if not np.isfinite(d[col]).all():
                raise ValueError(f'Invalid {col}')
        np.testing.assert_allclose(d.energy_needed_kwh, d.recharge_amt*d.battery_capacity_kwh)
        np.testing.assert_allclose(d.charge_time_minutes, d.energy_needed_kwh/d.actual_charge_speed_kw*60+5)
        d['source_csv_row_1based'] = np.arange(2, len(d)+2)
        d['assumed_start_h'] = d.time/3600
        d['active_duration_h'] = d.energy_needed_kwh/d.actual_charge_speed_kw
        d['assumed_end_h'] = d.assumed_start_h+d.active_duration_h
        d['negative_end_soc_flag'] = d.end_SoC.lt(0)
        d['overlaps_prior_active_session_flag'] = False
        for _, group in d.groupby('vehicle'):
            s = group.sort_values('assumed_start_h')
            prior_max_end = s.assumed_end_h.cummax().shift()
            d.loc[s.index, 'overlaps_prior_active_session_flag'] = s.assumed_start_h.lt(prior_max_end)
        data[d.scenario.iloc[0].replace('SFBAY-2050-','')] = d
    horizon = max(24, math.ceil(max(d.assumed_end_h.max()+5/60 for d in data.values())))
    profiles, metrics, audits, class_rows, flags = {}, [], [], [], []
    for scenario, d in data.items():
        for pt, group in d.groupby('powerTrain'):
            base = np.zeros(horizon)
            delayed = np.zeros(horizon)
            classes = {k: np.zeros(horizon) for k in CLASSES}
            starts = np.zeros(horizon, dtype=int)
            active = [set() for _ in range(horizon)]
            for r in group.itertuples():
                e = allocate(r.assumed_start_h, r.energy_needed_kwh, r.actual_charge_speed_kw, horizon)
                base += e
                delayed += allocate(r.assumed_start_h+5/60, r.energy_needed_kwh, r.actual_charge_speed_kw, horizon)
                classes[r.Class] += e
                starts[math.floor(r.assumed_start_h)] += 1
                for h in np.flatnonzero(e > 0):
                    active[h].add(r.vehicle)
            np.testing.assert_allclose(base.sum(), group.energy_needed_kwh.sum(), atol=1e-7)
            np.testing.assert_allclose(sum(classes.values()), base, atol=1e-7)
            profiles[(scenario,pt)] = dict(base=base, delayed=delayed, classes=classes,
                starts=starts, active=np.array([len(v) for v in active]))
            for h in range(horizon):
                for k, energy in classes.items():
                    class_rows.append(dict(scenario=scenario,powertrain=pt,vehicle_class=k,
                        elapsed_hour_start=h,elapsed_hour_end=h+1,energy_kWh=energy[h],mean_power_MW=energy[h]/1000))
            peak = int(base.argmax())
            m = dict(scenario=scenario,powertrain=pt,events=len(group),vehicles=group.vehicle.nunique(),
                stations=group.Station_Id.nunique(),full_record_energy_MWh=base.sum()/1000,
                first24_energy_MWh=base[:24].sum()/1000,after24_energy_MWh=base[24:].sum()/1000,
                peak_hour_start=peak,peak_hour_end=peak+1,peak_hourly_mean_MW=base[peak]/1000,
                first24_peak_hour_start=int(base[:24].argmax()),
                first24_peak_hourly_mean_MW=base[:24].max()/1000,
                first24_06_18_energy_share_pct=100*base[6:18].sum()/base[:24].sum(),
                delayed5min_peak_MW=delayed.max()/1000,delayed5min_peak_hour_start=int(delayed.argmax()),
                max_hourly_change_due_to_5min_MW=abs(delayed-base).max()/1000,
                energy_balance_error_kWh=base.sum()-group.energy_needed_kwh.sum(),
                negative_end_soc_events=int(group.negative_end_soc_flag.sum()),
                overlapping_active_events=int(group.overlaps_prior_active_session_flag.sum()),
                starts_after24=int(group.assumed_start_h.ge(24).sum()))
            metrics.append(m)
        audits.append(dict(scenario=scenario,rows=len(d),vehicles=int(d.vehicle.nunique()),
            duplicate_vehicle_ptid=int(d.duplicated(['vehicle','pt_id']).sum()),
            rounded_hour_matches=int(np.isclose(d.hour,np.round(d.time/3600)).sum()),
            floor_hour_matches=int(np.isclose(d.hour,np.floor(d.time/3600)).sum()),
            status_counts=d.type.value_counts().to_dict(),external_zone_counts=d.externalZone.value_counts().to_dict()))
        f = d.loc[d.negative_end_soc_flag | d.overlaps_prior_active_session_flag].copy()
        f.insert(0,'audit_scenario',scenario)
        flags.append(f)
    hourly = []
    for (s,pt), p in profiles.items():
        for h in range(horizon):
            hourly.append(dict(scenario=s,powertrain=pt,elapsed_hour_start=h,elapsed_hour_end=h+1,
                energy_kWh=p['base'][h],mean_power_MW=p['base'][h]/1000,
                setup5min_energy_kWh=p['delayed'][h],setup5min_mean_power_MW=p['delayed'][h]/1000,
                event_starts=p['starts'][h],unique_active_vehicles=p['active'][h]))
    # Machine-readable analysis tables, not an authored spreadsheet workbook.
    pd.DataFrame(hourly).to_csv(out/'hourly_charging_diagnostics.csv',index=False)
    pd.DataFrame(class_rows).to_csv(out/'hourly_charging_by_class.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'scenario_summary.csv',index=False)
    pd.concat(flags,ignore_index=True).to_csv(out/'flagged_events_preserved.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    note = ('Diagnostic only: BE records; scale = 1; no charging-loss correction; flagged events retained.\n'
            'Assume time = charging start, constant power, and 5 extra minutes draw no energy. No 24-hour wrapping.')
    fig, ax = plt.subplots(figsize=(12,6.5))
    for color,(s,d) in zip(COLORS,data.items()):
        p=profiles[(s,'BE')]
        ax.stairs(p['base'][:24]/1000,np.arange(25),color=color,lw=2.3,
            label=f'{s} ({d.loc[d.powerTrain.eq("BE"),"vehicle"].nunique():,} BE vehicles in file)',baseline=None)
    ax.set_ylabel('Hourly-average equivalent charging power (MW)')
    axis_style(ax); ax.legend(loc='upper left')
    fig.suptitle('Collaborator truck charging: first 24 simulation hours\nBE only; unscaled input records',fontsize=16)
    savefig(fig,out,'BE_hourly_first24_comparison',note)
    fig, axes=plt.subplots(2,1,figsize=(12,8.5),sharex=True,sharey=True)
    for ax,pt in zip(axes,['BE','H2FC']):
        for color,s in zip(COLORS,data):
            ax.stairs(profiles[(s,pt)]['base']/1000,np.arange(horizon+1),lw=2,color=color,label=s,baseline=None)
        for x in [24,48]:
            if x<horizon: ax.axvline(x,color='.4',ls=':',lw=1)
        ax.set_title('BE: battery-electric records' if pt=='BE' else 'H2FC: reported-energy equivalent only; NOT electric-grid demand')
        ax.set_ylabel('Hourly-average equivalent power (MW)')
        axis_style(ax,horizon); ax.legend()
    fig.suptitle('Full elapsed-time charging reconstruction',fontsize=16)
    savefig(fig,out,'full_timeline_BE_and_H2FC',
        'Raw event horizon retained; no fleet expansion or losses applied. No events beyond the file are inferred.\n'
        'H2FC is isolated, not added to BE. Constant-power reconstruction is provisional; event overlaps remain.')
    fig,axes=plt.subplots(3,1,figsize=(12,11),sharex=True,sharey=True)
    for ax,s in zip(axes,data):
        p=profiles[(s,'BE')]
        for k,color in zip(CLASSES,['#0072B2','#56B4E9','#009E73','#D55E00','#CC79A7']):
            ax.stairs(p['classes'][k][:24]/1000,np.arange(25),color=color,lw=1.8,label=k,baseline=None)
        ax.stairs(p['base'][:24]/1000,np.arange(25),lw=2.5,color='black',label='BE total',baseline=None)
        ax.set_title(s); ax.set_ylabel('Hourly-average power (MW)');axis_style(ax);ax.legend(ncol=6,fontsize=9)
    axes[0].set_ylim(0, max(profiles[(s,'BE')]['base'][:24].max() for s in data)/1000*1.12)
    fig.suptitle('BE charging by vehicle class: first 24 hours',fontsize=16)
    savefig(fig,out,'BE_hourly_by_vehicle_class',note)
    fig,axes=plt.subplots(2,1,figsize=(12,8.5),sharex=True)
    for color,s in zip(COLORS,data):
        p=profiles[(s,'BE')]
        axes[0].stairs(p['starts'][:24],np.arange(25),color=color,lw=2,label=s,baseline=None)
        axes[1].stairs(p['active'][:24],np.arange(25),color=color,lw=2,label=s,baseline=None)
    axes[0].set_ylabel('Charging-event starts per hour')
    axes[1].set_ylabel('Unique vehicles charging during hour')
    for ax in axes:axis_style(ax);ax.legend()
    fig.suptitle('BE charging activity: starts and participating vehicles',fontsize=16)
    savefig(fig,out,'BE_hourly_event_counts',
        'Unique active vehicles are counted once per hour, even if their reported sessions overlap.\n'
        'Counts are not simultaneous charger occupancy or total fleet size. Unscaled; assumed event-start times.')
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,sharey=True)
    for ax,s in zip(axes,data):
        p=profiles[(s,'BE')]
        ax.stairs(p['base'][:24]/1000,np.arange(25),lw=2,color='#0072B2',label='Start at time',baseline=None)
        ax.stairs(p['delayed'][:24]/1000,np.arange(25),lw=1.8,ls='--',color='#D55E00',label='Start after 5-min setup',baseline=None)
        ax.set_title(s);ax.set_ylabel('Hourly-average power (MW)');axis_style(ax);ax.legend()
    axes[0].set_ylim(0, max(max(profiles[(s,'BE')][k][:24]) for s in data for k in ['base','delayed'])/1000*1.12)
    fig.suptitle('Timing sensitivity: placement of the extra 5 minutes',fontsize=16)
    savefig(fig,out,'BE_five_minute_timing_sensitivity',
        'Both alternatives preserve reported event energy; the 5 minutes never add charging energy.\n'
        'This tests timing only, not SOC validity, overlapping sessions, missing charging or fleet expansion.')
    manifest=dict(status='diagnostic_complete_not_dispatch_ready',sources=source_hashes,
        output=str(out.resolve()),horizon_hours=horizon,scale_factor=1,energy_loss_correction=None,
        assumptions=['time assumed charging start; simulation seconds / 3600; timezone unknown',
            'constant reported power over energy/power hours; extra 5 minutes noncharging',
            '5-minute start delay is a separate sensitivity, not an applied correction',
            'BE main results; H2FC isolated as reported-energy equivalent, not grid demand',
            'full time retained; no folding, annualization, calendar-day averaging or extrapolation',
            'flagged SOC and overlaps retained; no queue, charger-capacity or itinerary repair',
            'zero hourly values mean no reconstructed activity in supplied file, not proven absence of demand',
            'reported-energy boundary unverified; no losses or vehicle scaling added'],
        audits=audits,metrics=metrics,self_tests='interval boundaries, cross-midnight, zero energy, truncation rejection',
        source_files_unchanged=all(sha(Path(p))==h for p,h in source_hashes.items()),
        downstream_run=False)
    assert manifest['source_files_unchanged']
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,default=lambda x:x.item()),encoding='utf-8')
    print(json.dumps(dict(output=str(out.resolve()),metrics=metrics),indent=2,default=lambda x:x.item()))


if __name__ == '__main__':
    main()
