"""Reconstruct clock-time SOC snapshots from saved steady-day endpoints/sessions.

No charging-model rerun. Within-trip SOC is not observed: interpolate uniform
traction demand over travel time, electric-first for PHEVs. Preserve a separate
arrival-debit sensitivity matching the original discrete energy bookkeeping.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'examples/pilates/impacts/impacts_output/events-fc07-existing-ev-charging-code10x-20260928-v2/bev-phev-home80-evi20c-20261012'
HOURS = (0, 8, 12, 18)
COLORS = {'BEV': '#2166ac', 'PHEV': '#d9781c', 'All EVs': '#555555'}


def snapshot(clock_seconds, departure, arrival, requested, used, start_soc,
             charge_kwh, charge_duration, capacity, *, arrival_debit=False):
    """Right-continuous SOC within a periodic itinerary; energy units are kWh.

    Charging starts at the arrival timestamp. Nonzero driving energy in a
    zero-duration traversal is an instantaneous debit in itinerary order.
    """
    arrays = [np.asarray(a, dtype=float) for a in
              (departure, arrival, requested, used, charge_kwh, charge_duration)]
    dep, arr, req, consumption, added, durations = arrays
    if not len(dep) or any(len(a) != len(dep) for a in arrays):
        raise ValueError('Empty or incompatible itinerary arrays')
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite itinerary')
    if (arr < dep).any() or (dep[1:] < arr[:-1]).any() or arr[-1] > dep[0]+86400:
        raise ValueError('Overlapping or nonperiodic itinerary')
    if (consumption < -1e-9).any() or (consumption > req+1e-8).any():
        raise ValueError('Invalid electric traction')
    if (added < 0).any() or (durations < 0).any() or ((added > 0) & (durations <= 0)).any():
        raise ValueError('Invalid charging energy/duration')
    next_dep = np.r_[dep[1:], dep[0]+86400]
    if (arr+durations > next_dep+1e-6).any():
        raise ValueError('Charging exceeds parking interval')
    if abs(added.sum()-consumption.sum()) > 1e-6:
        raise ValueError('Not a steady periodic energy balance')
    t = dep[0] + (float(clock_seconds)-dep[0]) % 86400
    soc = float(start_soc)
    for i in range(len(dep)):
        if dep[i] <= t < arr[i]:
            fraction = (t-dep[i])/(arr[i]-dep[i])
            loss = 0. if arrival_debit else min(req[i]*fraction, consumption[i])
            state = 'driving' if loss < consumption[i]-1e-8 else 'driving_after_electric_energy_exhausted'
            value = soc-loss
            break
        soc -= consumption[i]
        if t < next_dep[i]:
            elapsed = max(0.,t-arr[i])
            fraction = min(elapsed/durations[i],1.) if durations[i] > 0 else 0.
            value = soc + added[i]*fraction
            state = 'charging' if added[i] > 0 and elapsed < durations[i] else 'parked_not_charging'
            break
        soc += added[i]
    else:
        raise AssertionError('Clock time not covered by cycle')
    if not -1e-7 <= value <= capacity+1e-7:
        raise AssertionError(f'SOC outside battery bounds: {value}, {capacity}')
    return float(value), state


def main(out):
    if out.exists():
        raise FileExistsError(f'Preserve existing output: {out}')
    source_paths = [RUN/folder/(table+'.parquet') for folder in ('bev_home80','phev_home80')
                    for table in ('vehicles','trips','sessions')]
    source_paths += [RUN.parent/'existing_ev_inventory.parquet']
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    records, counts, zero_trips = [], {}, 0
    endpoint_error = 0.
    for kind,folder in [('BEV','bev_home80'),('PHEV','phev_home80')]:
        vehicles = pd.read_parquet(RUN/folder/'vehicles.parquet')
        trips = pd.read_parquet(RUN/folder/'trips.parquet')
        sessions = pd.read_parquet(RUN/folder/'sessions.parquet')
        keep = vehicles.steady_included if kind == 'BEV' else pd.Series(True,index=vehicles.index)
        counts[kind] = dict(modeled=len(vehicles),included=int(keep.sum()),excluded=int((~keep).sum()))
        selected = vehicles.loc[keep].set_index('vehicle')
        trips = trips.loc[trips.vehicle.isin(selected.index)].sort_values(['vehicle','trip_index'])
        sessions = sessions.loc[sessions.vehicle.isin(selected.index)]
        if kind == 'BEV':
            assert trips.included_in_profile.all() and sessions.included_in_profile.all()
        assert not sessions.duplicated(['vehicle','trip_index']).any()
        sg = {v:g.set_index('trip_index') for v,g in sessions.groupby('vehicle')}
        for vehicle,g in trips.groupby('vehicle',sort=True):
            meta = selected.loc[vehicle]
            dep,arr = g.departureTime.to_numpy(float),g.arrivalTime.to_numpy(float)
            used = g.served_electric_kWh.to_numpy(float) if kind == 'BEV' else g.battery_traction_kWh.to_numpy(float)
            req = g.requested_electric_kWh.to_numpy(float) if kind == 'BEV' else g.distance_km.to_numpy(float)*.202
            start = float(g.soc_arrival_kWh.iloc[0]+used[0]) if kind == 'BEV' else float(g.soc_departure_kWh.iloc[0])
            s = sg.get(vehicle,pd.DataFrame()).reindex(g.trip_index)
            assert len(s) == len(g)
            added = s.battery_energy_kWh.fillna(0).to_numpy(float)
            duration = s['timeCharging_s' if kind == 'BEV' else 'charging_time_s'].fillna(0).to_numpy(float)
            zero_trips += int(((arr==dep)&(used>0)).sum())
            # Independently check every saved endpoint and charging-session boundary.
            current = start
            for i,r in enumerate(g.itertuples(index=False)):
                if kind == 'PHEV':
                    endpoint_error=max(endpoint_error,abs(current-r.soc_departure_kWh))
                before=current-used[i]
                endpoint_error=max(endpoint_error,abs(before-r.soc_arrival_kWh))
                after=before+added[i]
                expected=r.soc_departure_next_kWh if kind=='BEV' else r.soc_next_departure_kWh
                endpoint_error=max(endpoint_error,abs(after-expected))
                if added[i]>0:
                    sr=s.iloc[i]
                    endpoint_error=max(endpoint_error,abs(before-sr.soc_before_kWh),abs(after-sr.soc_after_kWh))
                    np.testing.assert_allclose(added[i],sr.grid_power_kW*.9*duration[i]/3600,atol=1e-8)
                current=after
            np.testing.assert_allclose(current,start,atol=1e-6,rtol=0)
            args=(dep,arr,req,used,start,added,duration,float(meta.capacity_kWh))
            for hour in HOURS:
                value,state=snapshot(hour*3600,*args)
                step,_=snapshot(hour*3600,*args,arrival_debit=True)
                next_day,_=snapshot((hour+24)*3600,*args)
                np.testing.assert_allclose(value,next_day,atol=1e-9)
                records.append(dict(vehicle=vehicle,powertrain=kind,hour=hour,clock=f'{hour:02d}:00',
                    SOC_kWh=value,capacity_kWh=float(meta.capacity_kWh),SOC_pct=value/meta.capacity_kWh*100,
                    arrival_debit_SOC_pct=step/meta.capacity_kWh*100,state=state,
                    difference_from_arrival_debit_pp=(value-step)/meta.capacity_kWh*100))
    assert endpoint_error<1e-6
    data=pd.DataFrame(records)
    assert len(data)==1531*4 and not data.duplicated(['vehicle','hour']).any()
    assert data.SOC_kWh.min()>=2-1e-6 and data.SOC_pct.max()<=100+1e-6
    summary=[]
    for kind in ('BEV','PHEV','All EVs'):
        subset=data if kind=='All EVs' else data.loc[data.powertrain.eq(kind)]
        for hour,g in subset.groupby('hour',sort=True):
            pct=g.SOC_pct.clip(0,100)
            summary.append(dict(powertrain=kind,hour=int(hour),clock=f'{hour:02d}:00',vehicles=len(g),
                mean_SOC_pct=float(pct.mean()),median_SOC_pct=float(pct.median()),
                p10_SOC_pct=float(pct.quantile(.1)),p25_SOC_pct=float(pct.quantile(.25)),
                p75_SOC_pct=float(pct.quantile(.75)),p90_SOC_pct=float(pct.quantile(.9)),
                minimum_SOC_pct=float(pct.min()),maximum_SOC_pct=float(pct.max()),
                below20_pct=float(pct.lt(20).mean()*100),atleast90_pct=float(pct.ge(90).mean()*100),
                full_pct=float(np.isclose(pct,100,atol=1e-6,rtol=0).mean()*100),
                driving_vehicles=int(g.state.str.startswith('driving').sum()),
                charging_vehicles=int(g.state.eq('charging').sum()),
                mean_arrival_debit_SOC_pct=float(g.arrival_debit_SOC_pct.mean()),
                interpolation_affected_vehicles=int(g.difference_from_arrival_debit_pp.abs().gt(1e-7).sum()),
                mean_interpolation_difference_pp=float(g.difference_from_arrival_debit_pp.mean()),
                max_absolute_interpolation_difference_pp=float(g.difference_from_arrival_debit_pp.abs().max())))
    summary=pd.DataFrame(summary)
    histogram=[]
    bins=np.arange(0,101,5)
    for kind in ('BEV','PHEV'):
        for hour in HOURS:
            values=data.loc[data.powertrain.eq(kind)&data.hour.eq(hour),'SOC_pct'].clip(0,100)
            n,_=np.histogram(values,bins=bins)
            assert n.sum()==len(values)
            histogram.extend(dict(powertrain=kind,hour=hour,bin_left_pct=int(bins[i]),bin_right_pct=int(bins[i+1]),
                                  count_sample=int(n[i]),share_pct=float(n[i]/len(values)*100)) for i in range(len(n)))
    out.mkdir(parents=True)
    plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,4,figsize=(17,8),sharex=True,sharey=True)
    hist=pd.DataFrame(histogram)
    for row,kind in enumerate(('BEV','PHEV')):
        for col,hour in enumerate(HOURS):
            ax=axes[row,col]
            g=hist.loc[hist.powertrain.eq(kind)&hist.hour.eq(hour)]
            st=summary.loc[summary.powertrain.eq(kind)&summary.hour.eq(hour)].iloc[0]
            ax.bar(g.bin_left_pct+2.5,g.share_pct,width=4.6,color=COLORS[kind],alpha=.85)
            ax.axvline(st.mean_SOC_pct,color='#222222',ls='--',lw=1.6,label='Mean')
            ax.set_title(f'{kind} | {hour:02d}:00')
            ax.text(.03,.97,f'n = {int(st.vehicles):,}\nMean: {st.mean_SOC_pct:.1f}%\nMedian: {st.median_SOC_pct:.1f}%\nFull: {st.full_pct:.1f}%',
                    transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',alpha=.88,edgecolor='none'))
            ax.set(xlim=(0,100),xticks=np.arange(0,101,20),ylim=(0,100))
            ax.grid(axis='y',alpha=.15)
            if col==0:ax.set_ylabel('Share of vehicles (%)')
            if row==1:ax.set_xlabel('Battery SOC (%)')
    fig.suptitle('SOC distributions at 00:00, 08:00, 12:00 and 18:00\nExisting BEVs / PHEVs | Home80 steady representative day',fontsize=17)
    fig.text(.5,.025,'5-percentage-point bins; last bin includes 100%. Separate powertrain denominators; no fleet scaling needed for percentages.\n'
             'Clock-time reconstruction: constant-power charging; uniform traction demand during each trip, electric-first for PHEVs.\n'
             '595 BEVs + 936 PHEVs included; 12 infeasible BEVs and 1 missing-itinerary PHEV excluded. Not measured SOC.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.11,1,.9),w_pad=1.2,h_pad=2)
    fig.savefig(out/'soc_distribution_4times.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(17,5),sharex=True,sharey=True)
    for ax,hour in zip(axes,HOURS):
        for kind in ('BEV','PHEV'):
            values=np.sort(data.loc[data.powertrain.eq(kind)&data.hour.eq(hour),'SOC_pct'].clip(0,100).to_numpy())
            xx=np.r_[0,values,100]; yy=np.r_[0,np.arange(1,len(values)+1)/len(values)*100,100]
            ax.step(xx,yy,where='post',color=COLORS[kind],lw=2,label=kind)
        ax.set(title=f'{hour:02d}:00',xlabel='Battery SOC (%)',xlim=(0,100),ylim=(0,102))
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Cumulative share of vehicles (%)')
    axes[0].legend()
    fig.suptitle('SOC cumulative distributions | BEV vs PHEV',fontsize=17)
    fig.text(.5,.025,'Same included fleet and reconstruction assumptions as the histogram. Large jumps at 100% indicate fully charged vehicles.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.09,1,.9))
    fig.savefig(out/'soc_cumulative_distribution_4times.png',dpi=180)
    plt.close(fig)
    data.to_parquet(out/'vehicle_SOC_snapshots.parquet',index=False)
    summary.to_parquet(out/'SOC_summary.parquet',index=False)
    hist.to_parquet(out/'SOC_histogram_bins.parquet',index=False)
    report=dict(hours=list(HOURS),scenario='Home80 BEV + Home80 PHEV; steady day; seed 42',
        normalization='Percent of vehicles within each powertrain; unscaled sample counts. Uniform 10x expansion leaves distributions unchanged.',
        counts=counts,input_BEV=607,input_PHEV=937,missing_PHEV=1,
        source_hashes=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        algorithm='Trip-uniform total traction demand, electric-first until available PHEV electric energy is exhausted; constant-power parking charging; cyclic 24h boundary.',
        arrival_debit_sensitivity='Alternate within-trip SOC retains departure SOC until arrival; uses same continuous charging.',
        clock='BEAM simulation clock seconds modulo 86400; no timezone or daylight-saving correction.',
        boundary_convention='Right-continuous; zero-duration trips debit instantaneously in itinerary order.',
        max_endpoint_error_kWh=endpoint_error,positive_energy_zero_duration_trips=zero_trips,
        charging_model_rerun=False,air_quality_rerun=False,summary=summary.to_dict('records'))
    (out/'SOC_analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'SOC_summary.txt').write_text('CLOCK-TIME SOC DISTRIBUTION\n'+report['scenario']+'\n'+report['normalization']+
        '\nInterpolation is a reporting assumption, not observed minute-resolution SOC.\n\n'+summary.to_string(index=False),encoding='utf-8')
    for p in source_paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)]
    print(json.dumps(dict(output=str(out),counts=counts,summary=summary.to_dict('records'),max_endpoint_error_kWh=endpoint_error),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=RUN/'soc-distribution-20261019')
    main(parser.parse_args().output)
