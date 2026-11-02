"""Read-only analysis of saved Home80 charging results, following the supplied log.

Does not rerun charging, change parameters, or run power generation / air quality.
All count tables distinguish input, modeled, included, excluded, and missing vehicles.
"""
from research_paths import ISRM_PATH, BREATHE_ROOT, BEAM_DATA_ROOT, CHARGING_REFERENCE
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from impacts.pipeline.workflow.power_generation import allocate_charging_event

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'examples/pilates/impacts/impacts_output/events-fc07-existing-ev-charging-code10x-20260928-v2'
RUN = BASE / 'bev-phev-home80-evi20c-20261012'
EVI = BASE / 'bev80-phev80-90-evi-equal-20261012/native-evi-l1-l2-50-50-closed-24h/comparison_audit.json'
REFERENCE = CHARGING_REFERENCE
COLORS = {'BEV': '#2166ac', 'PHEV': '#d9781c', 'Total': '#222222',
          'Home': '#347e66', 'Work/Education': '#6789b4', 'Other/unknown': '#b7a7c8',
          'L1': '#73a5cf', 'L2': '#e6a23c', 'DC': '#ab5579'}
SCALE = 10


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(sessions, column='grid_energy_kWh'):
    values = np.zeros(24)
    for row in sessions.itertuples(index=False):
        energy = row.duration_h if column == 'duration_h' else getattr(row, column)
        allocate_charging_event(values, start_hour=row.arrivalTime / 3600,
                                duration_hours=row.duration_h, energy_kwh=energy)
    np.testing.assert_allclose(values.sum(), sessions[column].sum(), atol=1e-8)
    return values


def close(values):
    return np.r_[values, values[0]]


def clock_axes(ax, ylabel):
    ax.set(xlim=(0, 24), xticks=np.arange(0, 25, 2), xlabel='Hour of day', ylabel=ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.18)


def save(fig, out, name, note):
    fig.text(.5, .015, note, ha='center', va='bottom', fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .065, 1, .95), h_pad=2.5, w_pad=2.5)
    fig.savefig(out / name, dpi=180, facecolor='white')
    plt.close(fig)


def instantaneous_peak(sessions):
    events = []
    for r in sessions.itertuples(index=False):
        start = r.arrivalTime / 3600 % 24
        assert 0 < r.duration_h <= 24 + 1e-8
        for shift in (-24, 0, 24):
            left, right = max(0., start + shift), min(24., start + shift + r.duration_h)
            if right > left:
                events.extend([(left, r.grid_power_kW), (right, -r.grid_power_kW)])
    trace = pd.DataFrame(events, columns=['hour', 'change_kW']).groupby('hour').change_kW.sum().sort_index().cumsum()
    hour = float(trace.idxmax())
    return dict(peak_MW=float(trace.max() * SCALE / 1000), hour=hour,
                description='Exact event-boundary peak under constant-power session assumption; not measured')


def main(out):
    if out.exists():
        raise FileExistsError(f'Preserve existing analysis: {out}')
    sources = [BASE / 'existing_ev_inventory.parquet', BASE / 'existing_ev_trips.parquet',
               RUN / 'run_summary.json', EVI, REFERENCE]
    sources += [RUN / folder / f'{name}.parquet' for folder in ('bev_home80', 'phev_home80')
                for name in ('vehicles', 'sessions', 'trips', 'first_trips')]
    hashes = {str(p): sha(p) for p in sources}
    inventory = pd.read_parquet(sources[0])
    raw_trips = pd.read_parquet(sources[1])
    evi = json.loads(EVI.read_text())
    manifest = json.loads((RUN / 'run_summary.json').read_text())
    assert manifest['battery_kWh_per_km'] == .202 and manifest['charging_efficiency'] == .9
    assert manifest['scale'] == SCALE and evi['comparison_fleet'] == len(inventory) * SCALE
    assert not evi['subtype_reweighting']
    sessions, vehicle_tables, trip_tables, soc_rows = [], {}, {}, []
    summary = {}
    for kind, folder in [('BEV', 'bev_home80'), ('PHEV', 'phev_home80')]:
        v = pd.read_parquet(RUN / folder / 'vehicles.parquet')
        s = pd.read_parquet(RUN / folder / 'sessions.parquet')
        t = pd.read_parquet(RUN / folder / 'trips.parquet').sort_values(['vehicle', 'trip_index'])
        included = v.steady_included if kind == 'BEV' else pd.Series(True, index=v.index)
        kept = set(v.loc[included, 'vehicle'])
        s = s.loc[s.vehicle.isin(kept) & s.grid_energy_kWh.gt(0)].copy()
        if kind == 'BEV':
            assert s.included_in_profile.all()
        s['duration_h'] = s['timeCharging_s' if kind == 'BEV' else 'charging_time_s'] / 3600
        s['parking_h'] = s['parkingTime_s' if kind == 'BEV' else 'parking_time_s'] / 3600
        s['powertrain'] = kind
        s['location'] = np.where(s.is_home, 'Home', np.where(s.activity.eq('Work/Education'), 'Work/Education', 'Other/unknown'))
        s['charger'] = s.grid_power_kW.map({1.9: 'L1', 7.2: 'L2', 50.: 'DC'})
        assert s.charger.notna().all() and (s.duration_h <= s.parking_h + 1e-8).all()
        assert not s.duplicated(['vehicle', 'trip_index']).any()
        np.testing.assert_allclose(s.grid_energy_kWh, s.grid_power_kW * s.duration_h, atol=1e-8)
        np.testing.assert_allclose(s.battery_energy_kWh, s.grid_energy_kWh * .9, atol=1e-8)
        p = profile(s) * SCALE / 1000
        np.testing.assert_allclose(p, evi['hourly_MW']['code'][kind], atol=1e-9)
        nv = int(inventory.powertrain.eq(kind).sum())
        counts = s.groupby('vehicle').size().reindex(v.loc[included, 'vehicle'], fill_value=0)
        dist = {str(int(k)): int(n) for k, n in counts.value_counts().sort_index().items()}
        assert sum(dist.values()) == len(kept)
        assert sum(int(k) * n for k, n in dist.items()) == len(s)
        home = s.loc[s.is_home]
        summary[kind] = dict(input_vehicles_sample=nv, modeled_vehicles_sample=len(v),
            included_vehicles_sample=len(kept), excluded_vehicles_sample=int((~included).sum()),
            missing_itinerary_vehicles_sample=nv-len(v), charging_vehicles_sample=int(s.vehicle.nunique()),
            sessions_sample=len(s), sessions_by_vehicle_included_sample=dist,
            home_preferred_vehicles_included_sample=int(v.loc[included, 'home_preferred'].sum()),
            primary_home_vehicles_included_sample=int(v.loc[included, 'primary_home'].sum()),
            home_charging_vehicles_sample=int(home.vehicle.nunique()), home_sessions_sample=len(home),
            home_energy_share=float(home.grid_energy_kWh.sum()/s.grid_energy_kWh.sum()),
            home_session_share=len(home)/len(s), grid_MWh_10x=float(p.sum()),
            battery_MWh_10x=float(s.battery_energy_kWh.sum()*SCALE/1000),
            loss_MWh_10x=float(s.charging_loss_kWh.sum()*SCALE/1000),
            charging_vehicle_hours_sample=float(s.duration_h.sum()),
            mean_hours_per_charging_vehicle=float(s.duration_h.sum()/s.vehicle.nunique()),
            mean_grid_kWh_per_charging_vehicle=float(s.grid_energy_kWh.sum()/s.vehicle.nunique()),
            mean_grid_kWh_per_input_vehicle=float(s.grid_energy_kWh.sum()/nv),
            mean_session_hours=float(s.duration_h.mean()), median_session_hours=float(s.duration_h.median()),
            hourly_peak_MW_10x=float(p.max()), hourly_peak_start=int(p.argmax()),
            overnight_00_06_energy_share=float(p[:6].sum()/p.sum()),
            evening_18_24_energy_share=float(p[18:].sum()/p.sum()))
        tv = t.loc[t.vehicle.isin(kept)]
        for phase, tt in [('steady', tv), ('first', pd.read_parquet(RUN / folder / 'first_trips.parquet'))]:
            tt = tt.loc[tt.vehicle.isin(kept)].sort_values(['vehicle', 'trip_index'])
            first, last = tt.groupby('vehicle').first(), tt.groupby('vehicle').last()
            cap = v.set_index('vehicle').capacity_kWh
            start = first.soc_arrival_kWh + first.served_electric_kWh if kind == 'BEV' else first.soc_departure_kWh
            end = last.soc_departure_next_kWh if kind == 'BEV' else last.soc_next_departure_kWh
            for vehicle in start.index:
                soc_rows.append(dict(vehicle=vehicle, powertrain=kind, phase=phase,
                    initial_kWh=float(start[vehicle]), final_kWh=float(end[vehicle]),
                    capacity_kWh=float(cap[vehicle]), initial_pct=float(start[vehicle]/cap[vehicle]*100),
                    final_pct=float(end[vehicle]/cap[vehicle]*100)))
        traction = tv.served_electric_kWh.sum() if kind == 'BEV' else tv.battery_traction_kWh.sum()
        gain = v.loc[included, 'steady_SOC_gain_kWh'].sum()
        np.testing.assert_allclose(s.battery_energy_kWh.sum()-traction, gain, atol=1e-6)
        summary[kind]['traction_battery_MWh_10x'] = float(traction * SCALE / 1000)
        if kind == 'PHEV':
            summary[kind]['electric_distance_share'] = float(tv.electric_distance_km.sum()/tv.distance_km.sum())
            summary[kind]['fuel_distance_km_10x'] = float(tv.fuel_distance_km.sum()*SCALE)
        else:
            summary[kind]['excluded_unmet_battery_kWh_sample'] = float(v.loc[~included, 'steady_shortfall_kWh'].sum())
            summary[kind]['excluded_supplied_grid_kWh_sample'] = float(v.loc[~included, 'steady_grid_charge_kWh'].sum())
        selected_trips = raw_trips.loc[raw_trips.vehicle.isin(kept) & raw_trips.arrivalTime.lt(29*3600)].sort_values(['vehicle','departureTime','arrivalTime'])
        parking = []
        for _, g in selected_trips.groupby('vehicle'):
            dwell = np.r_[g.departureTime.to_numpy()[1:],g.departureTime.iloc[0]+86400] - g.arrivalTime.to_numpy()
            assert (dwell >= 0).all()
            parking.extend(dwell.tolist())
        summary[kind]['included_travel_km_sample'] = float(selected_trips.length.sum()/1000)
        summary[kind]['included_parking_hours_sample'] = float(np.sum(parking)/3600)
        summary[kind]['eligible_parking_stops_included_sample'] = int((np.asarray(parking)>900).sum())
        sessions.append(s)
        vehicle_tables[kind], trip_tables[kind] = v.loc[included], tv
    ss = pd.concat(sessions, ignore_index=True)
    assert not ss.vehicle.isin(inventory.loc[~inventory.powertrain.isin(['BEV','PHEV']), 'vehicle']).any()
    curves = {k: profile(ss.loc[ss.powertrain.eq(k)])*SCALE/1000 for k in ('BEV','PHEV')}
    curves['Total'] = curves['BEV'] + curves['PHEV']
    total = curves['Total']
    ref = np.asarray(evi['hourly_MW']['EVI']['total'])
    starts, concurrency = {}, {}
    for k in ('BEV', 'PHEV', 'Total'):
        part = ss if k == 'Total' else ss.loc[ss.powertrain.eq(k)]
        starts[k] = np.bincount(np.floor(part.arrivalTime.to_numpy()/3600).astype(int)%24, minlength=24)
        concurrency[k] = profile(part, 'duration_h')
        assert starts[k].sum() == len(part)
    summary['Total'] = dict(input_vehicles_sample=len(inventory),
        modeled_vehicles_sample=sum(summary[k]['modeled_vehicles_sample'] for k in ('BEV','PHEV')),
        included_vehicles_sample=sum(summary[k]['included_vehicles_sample'] for k in ('BEV','PHEV')),
        charging_vehicles_sample=int(ss.vehicle.nunique()), sessions_sample=len(ss),
        home_sessions_sample=int(ss.is_home.sum()), home_charging_vehicles_sample=int(ss.loc[ss.is_home].vehicle.nunique()),
        home_energy_share=float(ss.loc[ss.is_home].grid_energy_kWh.sum()/ss.grid_energy_kWh.sum()),
        grid_MWh_10x=float(total.sum()), battery_MWh_10x=float(ss.battery_energy_kWh.sum()*SCALE/1000),
        loss_MWh_10x=float(ss.charging_loss_kWh.sum()*SCALE/1000),
        charging_vehicle_hours_sample=float(ss.duration_h.sum()),
        mean_hours_per_charging_vehicle=float(ss.duration_h.sum()/ss.vehicle.nunique()),
        mean_grid_kWh_per_charging_vehicle=float(ss.grid_energy_kWh.sum()/ss.vehicle.nunique()),
        mean_grid_kWh_per_input_vehicle=float(ss.grid_energy_kWh.sum()/len(inventory)),
        hourly_peak_MW_10x=float(total.max()), hourly_peak_start=int(total.argmax()),
        hourly_min_MW_10x=float(total.min()), hourly_min_start=int(total.argmin()),
        peak_all_session_starts_hour=int(starts['Total'].argmax()),
        peak_all_session_starts_sample=int(starts['Total'].max()),
        evening_18_24_energy_share=float(total[18:].sum()/total.sum()),
        instantaneous_model_peak_10x=instantaneous_peak(ss))
    locations = ss.groupby(['powertrain','location'], observed=True).agg(
        sessions_sample=('vehicle','size'), charging_vehicles_sample=('vehicle','nunique'),
        grid_kWh_sample=('grid_energy_kWh','sum'), charging_hours_sample=('duration_h','sum')).reset_index()
    chargers = ss.groupby(['powertrain','charger'], observed=True).agg(
        sessions_sample=('vehicle','size'), charging_vehicles_sample=('vehicle','nunique'),
        grid_kWh_sample=('grid_energy_kWh','sum'), charging_hours_sample=('duration_h','sum')).reset_index()
    reasons = ss.assign(reason=ss.reason.fillna('selected_primary_stop')).groupby(['powertrain','reason']).agg(
        sessions_sample=('vehicle','size'), grid_kWh_sample=('grid_energy_kWh','sum')).reset_index()
    soc = pd.DataFrame(soc_rows)
    steady = soc.loc[soc.phase.eq('steady')]
    assert (steady.final_kWh-steady.initial_kWh).abs().max() < 1e-6
    metrics = dict(grid_daily_difference_pct=float((total.sum()/ref.sum()-1)*100),
        peak_difference_pct=float((total.max()/ref.max()-1)*100),
        EVI_daily_MWh=float(ref.sum()), EVI_peak_MW=float(ref.max()), EVI_peak_hour=int(ref.argmax()),
        hourly_RMSE_MW=float(np.sqrt(np.mean((total-ref)**2))),
        shape_correlation=float(np.corrcoef(total/total.sum(),ref/ref.sum())[0,1]))
    report = dict(scenario='Existing EVs: BEV Home80 + PHEV Home80; steady representative day; seed 42',
        reference_document=str(REFERENCE), reference_used='Charging Assignment Log and output structure; not numerical targets',
        sources_sha256=hashes, scale=SCALE, fleet_expanded=len(inventory)*SCALE,
        assumptions=dict(battery_kWh_per_km=.202, efficiency=.9, minimum_battery_kWh=2.,
                         model_L1_L2='20:80', EVI_L1_L2='50:50', model_BEV_PHEV='607:937',
                         EVI_BEV_PHEV='75:25', EVI_inputs=evi['EVI_inputs']),
        charging_model_rerun=False, power_generation_run=False, air_quality_run=False,
        notes=['All count metrics are unscaled sample unless explicitly marked 10x.',
               'All energy/load totals marked 10x are expanded; per-vehicle means are not multiplied.',
               '12 excluded BEVs and 1 missing-itinerary PHEV are separate from zero-session vehicles.',
               'Original P1/P2 planned counts and 10-kWh critical preassignment are not used in this model.',
               'BEV necessary topups are not equivalent to original critical-charge assignments.',
               'Location is Home/Work/Other activity, not validated physical charging-station access.',
               'Spatial TAZ map and K-means are outside this four-part analysis; not inferred from missing coordinates.',
               'Hourly average power differs from event-boundary instantaneous modeled power.',
               'SOC endpoints are itinerary-cycle boundaries, not necessarily calendar midnight.',
               'EVI is a scenario-model comparator, not measured ground truth; compositions and L1/L2 differ.',
               'Plot point 24 repeats hour 0 for display only. No timezone shift or calibration applied.'],
        summary=summary, EVI_comparison=metrics, location=locations.to_dict('records'),
        chargers=chargers.to_dict('records'), assignment_reasons=reasons.to_dict('records'),
        SOC=soc.groupby(['powertrain','phase'])[['initial_pct','final_pct']].mean().reset_index().to_dict('records'))
    out.mkdir(parents=True)
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False})
    h = np.arange(25)
    fig, axes = plt.subplots(3,1,figsize=(12,12))
    for k in ('BEV','PHEV','Total'):
        axes[0].plot(h,close(curves[k]),label=k,lw=2.4,color=COLORS[k])
    axes[0].plot(h,close(ref),'--',label='EVI native total',color='#8b4da0',lw=2.2)
    axes[0].set_title('Expanded fleet load: 15,440 input PEVs')
    clock_axes(axes[0],'Hourly-average power (MW)')
    axes[0].legend(ncol=4)
    for k,values,color in [('Charging code',total,COLORS['BEV']),('EVI native',ref,'#8b4da0')]:
        axes[1].plot(h,close(values)*1e6/(len(inventory)*SCALE),label=k,color=color,lw=2.4)
        axes[2].plot(h,close(values/values.sum()*100),label=k,color=color,lw=2.4)
    axes[1].set_title('Fleet-normalized magnitude (all input EVs in denominator)')
    clock_axes(axes[1],'kW / 1,000 input EVs')
    axes[1].legend(ncol=2)
    axes[2].set_title('Timing / shape comparison (each curve sums to 100% over 24 bins)')
    clock_axes(axes[2],'Daily grid energy share (%)')
    axes[2].legend(ncol=2)
    fig.suptitle('1 | Charging curves: existing EVs vs EVI-Pro Lite',fontsize=17)
    save(fig,out,'01_charging_curves.png','BEV/PHEV Home80; 0.202 battery kWh/km; 90% efficiency; seed 42. No fitted scaling or time shift.\nEVI: weekday, 35 mi/day, 20 C, home L1/L2 50:50; model remains 20:80. Hour 24 is display-only.')
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    x=np.arange(2)
    for offset,(field,label,color) in enumerate([
        ('home_preferred_vehicles_included_sample','Home preference','#a0cbb6'),
        ('primary_home_vehicles_included_sample','Home primary stop','#347e66'),
        ('home_charging_vehicles_sample','Actually charged at Home','#203b32')]):
        vals=[summary[k][field] for k in ('BEV','PHEV')]
        bars=axes[0,0].bar(x+(offset-1)*.24,vals,width=.23,label=label,color=color)
        axes[0,0].bar_label(bars,padding=3,fontsize=9)
    axes[0,0].set(xticks=x,xticklabels=['BEV','PHEV'],ylabel='Included vehicles (sample)',title='Preference, primary location and realized Home charging')
    axes[0,0].legend(fontsize=8,loc='upper left')
    axes[0,0].set_ylim(0,1100)
    for ax,group,labels,value,title in [
        (axes[0,1],'location',['Home','Work/Education','Other/unknown'],'count','Actual sessions by activity'),
        (axes[1,0],'location',['Home','Work/Education','Other/unknown'],'energy','Grid energy by activity'),
        (axes[1,1],'charger',['L1','L2','DC'],'count','Actual sessions by charger type')]:
        bottom=np.zeros(2)
        for label in labels:
            vals=np.array([len(ss.loc[ss.powertrain.eq(k)&ss[group].eq(label)]) if value=='count'
                else ss.loc[ss.powertrain.eq(k)&ss[group].eq(label),'grid_energy_kWh'].sum()*SCALE/1000 for k in ('BEV','PHEV')])
            ax.bar(x,vals,bottom=bottom,label=label,color=COLORS[label],width=.6)
            for i,val in enumerate(vals):
                if val>0: ax.text(i,bottom[i]+val/2,f'{val:.1f}' if value=='energy' else str(int(val)),ha='center',va='center',fontsize=10)
            bottom+=vals
        ax.set(xticks=x,xticklabels=['BEV','PHEV'],title=title,ylabel='Grid energy (MWh/day, 10x)' if value=='energy' else 'Charging sessions (sample)')
        ax.set_ylim(0,max(bottom)*1.35)
        ax.legend(fontsize=9,loc='upper left')
    fig.suptitle('2 | Charging assignment and realized allocation',fontsize=17)
    save(fig,out,'02_charging_assignment.png','Included fleet only: 595 BEVs and 936 PHEVs. Counts are sample counts; energy is expanded 10x.\nHome preference is not a fixed Home energy share. Original P1/P2 preassignment is not used.')
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,group,labels,title in [(axes[0,0],'location',['Home','Work/Education','Other/unknown'],'Load by activity'),
                                (axes[0,1],'charger',['L1','L2','DC'],'Load by charger power')]:
        parts=[profile(ss.loc[ss[group].eq(k)])*SCALE/1000 for k in labels]
        np.testing.assert_allclose(np.sum(parts,axis=0),total,atol=1e-9)
        ax.stackplot(h,*[close(p) for p in parts],labels=labels,colors=[COLORS[k] for k in labels],alpha=.85)
        clock_axes(ax,'Hourly-average power (MW)')
        ax.set_title(title)
        ax.legend(fontsize=9,loc='upper left')
    for k in ('BEV','PHEV','Total'):
        axes[1,0].plot(h,np.r_[0,np.cumsum(curves[k])],color=COLORS[k],label=k,lw=2)
    axes[1,0].plot(h,np.r_[0,np.cumsum(ref)],color='#8b4da0',ls='--',label='EVI native')
    clock_axes(axes[1,0],'Cumulative grid energy (MWh)')
    axes[1,0].set_title('Energy delivered since 00:00 (not display closure)')
    axes[1,0].legend(fontsize=9)
    battery=[summary[k]['battery_MWh_10x'] for k in ('BEV','PHEV')]
    loss=[summary[k]['loss_MWh_10x'] for k in ('BEV','PHEV')]
    axes[1,1].bar(x,battery,label='Into battery',color='#347e66')
    axes[1,1].bar(x,loss,bottom=battery,label='Charging loss',color='#c7c7c7')
    for i,(a,b) in enumerate(zip(battery,loss)):
        axes[1,1].text(i,a+b+.8,f'{a+b:.2f}',ha='center')
    axes[1,1].set(xticks=x,xticklabels=['BEV','PHEV'],ylabel='Energy (MWh/day, 10x)',title='Grid energy = battery energy + charging losses',ylim=(0,max(np.array(battery)+loss)*1.3))
    axes[1,1].legend(fontsize=9)
    fig.suptitle('3 | Charging load and energy accounting',fontsize=17)
    save(fig,out,'03_load_and_energy.png','All load and energy totals expanded 10x. L1 = 1.9 kW; L2 = 7.2 kW; DC = 50 kW.\nSupplied charging only: 12 infeasible BEVs excluded; PHEV fuel travel does not contribute to electric load.')
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    hour=np.arange(24)
    for k in ('BEV','PHEV','Total'):
        axes[0,0].plot(hour,starts[k],color=COLORS[k],label=k,lw=2,marker='o',ms=3)
        axes[1,0].plot(hour,concurrency[k],color=COLORS[k],label=k,lw=2)
    home_starts=np.bincount(np.floor(ss.loc[ss.is_home,'arrivalTime'].to_numpy()/3600).astype(int)%24,minlength=24)
    axes[0,0].plot(hour,home_starts,'--',color=COLORS['Home'],label='Home starts (all EVs)',lw=2)
    for ax,ylabel,title in [(axes[0,0],'Sessions starting / hour (sample)','Charging starts, not arrivals at all stops'),
                            (axes[1,0],'Mean concurrent charging vehicles (sample)','Charging occupancy: session overlap / 1 hour')]:
        clock_axes(ax,ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
    cats=['0','1','2','3+','Excluded','Missing']
    for i,k in enumerate(('BEV','PHEV')):
        d=summary[k]['sessions_by_vehicle_included_sample']
        vals=[d.get('0',0),d.get('1',0),d.get('2',0),sum(v for n,v in d.items() if int(n)>=3),
              summary[k]['excluded_vehicles_sample'],summary[k]['missing_itinerary_vehicles_sample']]
        assert sum(vals)==summary[k]['input_vehicles_sample']
        bars=axes[0,1].bar(np.arange(6)+(i-.5)*.35,vals,width=.34,label=k,color=COLORS[k])
        axes[0,1].bar_label(bars,padding=3,fontsize=9)
    axes[0,1].set(xticks=np.arange(6),xticklabels=cats,ylabel='Vehicles (sample)',title='Sessions per included vehicle + separate exclusions',ylim=(0,1120))
    axes[0,1].legend()
    for k in ('BEV','PHEV'):
        vals=np.sort(ss.loc[ss.powertrain.eq(k),'duration_h'].to_numpy())
        axes[1,1].step(vals,np.arange(1,len(vals)+1)/len(vals)*100,where='post',label=k,color=COLORS[k],lw=2)
    axes[1,1].set(xlabel='Charging duration per session (hours)',ylabel='Cumulative share of sessions (%)',
                  title='Session duration distribution',ylim=(0,102),xlim=(0,None))
    axes[1,1].grid(alpha=.18)
    axes[1,1].legend()
    fig.suptitle('4 | Number of charges, timing and duration',fontsize=17)
    save(fig,out,'04_number_of_charges.png','All counts are unscaled sample counts; multiply by 10 for expanded equivalents, not additional observations.\nCharging starts differ from concurrent charging vehicles. Missing/excluded vehicles are NOT zero-session vehicles.')
    hourly=pd.DataFrame({'hour_start':np.arange(24),'total_MW_10x':total,'BEV_MW_10x':curves['BEV'],
                         'PHEV_MW_10x':curves['PHEV'],'EVI_MW_same_fleet':ref})
    for k in ('BEV','PHEV','Total'):
        hourly[k+'_starts_sample']=starts[k]
        hourly[k+'_mean_concurrent_sample']=concurrency[k]
    hourly.to_parquet(out/'hourly_analysis.parquet',index=False)
    locations.to_parquet(out/'location_summary.parquet',index=False)
    chargers.to_parquet(out/'charger_summary.parquet',index=False)
    soc.to_parquet(out/'SOC_endpoints.parquet',index=False)
    (out/'analysis_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['CHARGING ASSIGNMENT LOG - EXISTING EVs',report['scenario'],
           'Sample=10%; expansion=10x; no model rerun; no power-generation/AQ run.',
           'Reference: supplied Charging Profiles & SOC Estimation.pdf, pp. 9-10.',
           'Planned P1/P2 sessions and original critical-charge assignments: NOT APPLICABLE.',
           'Counts below are SAMPLE; energy/load marked 10x are EXPANDED.', '']
    for k,stats in summary.items():
        lines.append(k)
        lines.extend(f'  {key}: {value}' for key,value in stats.items())
        lines.append('')
    lines.extend(['LOCATION ALLOCATION',locations.to_string(index=False),'','CHARGER ALLOCATION',chargers.to_string(index=False),
                  '', 'ASSIGNMENT REASONS',reasons.to_string(index=False),'','SOC (vehicle-mean percentages; same included cohort)',
                  pd.DataFrame(report['SOC']).to_string(index=False),'','EVI COMPARISON',json.dumps(metrics,indent=2),
                  '', 'INTERPRETATION NOTES',*report['notes']])
    (out/'charging_assignment_log.txt').write_text('\n'.join(lines),encoding='utf-8')
    for p in sources:
        assert sha(p)==hashes[str(p)], f'Input changed: {p}'
    print(json.dumps(dict(output=str(out),summary=summary,EVI_comparison=metrics,
                         location=report['location'],chargers=report['chargers'],SOC=report['SOC']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=RUN/'results-analysis-20261019')
    main(parser.parse_args().output)
