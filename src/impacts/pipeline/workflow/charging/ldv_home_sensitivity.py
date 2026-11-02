"""Experimental LDV home-charging preferences and PHEV sensitivities.

This is a separate research model, not the Cristian-compatible algorithm.
Shared schedules support repeat-day SOC convergence, Home80/Home90 scenarios,
and explicit PHEV electric-first/fuel-equivalent accounting. Fuel equivalents
are not measured fuel use or calculated on-road emissions. Existing parameters
and numerical calculations are preserved for historical result reproduction.
"""

from dataclasses import dataclass, replace
import hashlib

import numpy as np
import pandas as pd

from ..power_generation import allocate_charging_event


@dataclass
class Schedule:
    vehicle: str
    kind: str
    capacity: float
    initial: float
    departure: np.ndarray
    arrival: np.ndarray
    dwell: np.ndarray
    energy: np.ndarray
    home: np.ndarray
    activity: np.ndarray
    primary: int
    home_preferred: bool
    ac_power: float


def uniform(vehicle, seed, purpose):
    digest=hashlib.sha256(f'{seed}|{purpose}|{vehicle}'.encode()).digest()
    return int.from_bytes(digest[:8],'big')/2**64


def build_schedules(trips, stop_audit, home_share, seed=42, *, base_kWh_per_km=.17):
    # A direct coefficient sensitivity in the existing single-energy-ledger
    # model, not a separate battery-side/grid-side efficiency implementation.
    if not np.isfinite(base_kWh_per_km) or base_kWh_per_km <= 0:
        raise ValueError('base_kWh_per_km must be finite and positive')
    if not 0 <= home_share <= 1:raise ValueError('Invalid home share')
    t=trips.loc[trips.vehicle.isin(stop_audit.vehicle) & trips.arrivalTime.lt(29*3600)].copy()
    t=t.sort_values(['vehicle','departureTime','arrivalTime'],kind='stable')
    if stop_audit.duplicated(['vehicle','departureTime','arrivalTime']).any():
        raise ValueError('Ambiguous stop key')
    labels=stop_audit[['vehicle','departureTime','arrivalTime','is_home','groupedActivityType']]
    t=t.merge(labels,on=['vehicle','departureTime','arrivalTime'],how='left',validate='many_to_one')
    t['is_home']=t.is_home.fillna(False).astype(bool)
    t['groupedActivityType']=t.groupedActivityType.fillna('Unlabelled short stop')
    meta=stop_audit.groupby('vehicle',sort=True).first()
    groups={v:g.reset_index(drop=True) for v,g in t.groupby('vehicle',sort=True)}
    # Exactly floor(share*N) among identified terminal-home vehicles, stratified
    # by powertrain. Hash ranks make 80% membership a subset of 100% membership.
    home_candidates={k:[] for k in ('BEV','PHEV')}
    for v,g in groups.items():
        final_dwell=g.departureTime.iloc[0]+86400-g.arrivalTime.iloc[-1]
        if bool(g.is_home.iloc[-1]) and final_dwell>900:
            home_candidates[meta.loc[v,'powertrain']].append(v)
    preferred=set()
    for kind,ids in home_candidates.items():
        ordered=sorted(ids,key=lambda v:uniform(v,seed,'preference'))
        preferred.update(ordered[:int(np.floor(home_share*len(ordered)))])
    schedules=[]
    for v,g in groups.items():
        dep=g.departureTime.to_numpy(float);arr=g.arrivalTime.to_numpy(float)
        dwell=np.r_[dep[1:],dep[0]+86400]-arr
        if (arr<dep).any() or (dwell<0).any():raise ValueError(f'Invalid itinerary: {v}')
        kind=meta.loc[v,'powertrain'];home=g.is_home.to_numpy(bool)
        eligible=dwell>900
        if v in preferred:
            primary=len(g)-1
        else:
            work=np.flatnonzero(eligible & g.groupedActivityType.eq('Work/Education').to_numpy())
            other=np.flatnonzero(eligible & ~home)
            fallback=np.flatnonzero(eligible & home)
            candidates=work if len(work) else other if len(other) else fallback
            primary=int(candidates[np.argmax(dwell[candidates])]) if len(candidates) else -1
        energy=g.length.to_numpy(float)*base_kWh_per_km/1000*(.2 if kind=='PHEV' else 1.)
        schedules.append(Schedule(str(v),kind,float(meta.loc[v,'battery_kWh']),
            float(meta.loc[v,'initialsoc_kWh']),dep,arr,dwell,energy,home,
            g.groupedActivityType.to_numpy(),primary,v in preferred,
            1.9 if uniform(v,seed,'ac_power')<.2 else 7.2))
    expected=stop_audit.groupby('vehicle').energyConsumption_kWh.sum()
    for s in schedules:
        # The supplied stop audit is the preserved 0.17 / PHEV-factor-0.2 control.
        np.testing.assert_allclose(s.energy.sum(),expected[s.vehicle]*base_kWh_per_km/.17,atol=1e-9)
    return schedules


def simulate_day(s, initial, reserve=2., capture=False, *, efficiency=1.):
    """Trip/SOC energy is battery-side; profiles and charger power are grid-side.

    Efficiency defaults to 1 to preserve the historical single-ledger scenarios.
    ``charge`` remains battery energy for backward-compatible SOC accounting.
    """
    if not np.isfinite(efficiency) or not 0 < efficiency <= 1:
        raise ValueError('Invalid charging efficiency')
    if not 0<=initial<=s.capacity or not 0<=reserve<s.capacity:
        raise ValueError('Invalid SOC or reserve')
    soc=float(initial);charge=served=shortfall=0.;min_soc=soc
    sessions=[];trip_rows=[];profile=np.zeros(24);home_profile=np.zeros(24)
    for i,demand in enumerate(s.energy):
        available=max(soc-reserve,0.)
        used=min(demand,available);deficit=max(demand-used,0.)
        soc-=used;served+=used;shortfall+=deficit;min_soc=min(min_soc,soc)
        before=soc;power=0.;requested=0.;reason='none'
        if s.dwell[i]>900:
            if i==s.primary:
                requested=s.capacity-soc;power=s.ac_power;reason='primary_home' if s.home[i] else 'primary_nonhome'
            elif s.kind=='BEV':
                # Look ahead to the next preferred charging opportunity, including
                # trips across the daily boundary. No charging after a deficit can
                # retroactively make the failed trip feasible.
                n=len(s.energy)
                k=(s.primary-i)%n if s.primary>=0 else n
                need=sum(s.energy[(i+j)%n] for j in range(1,k+1))+reserve
                requested=max(min(need,s.capacity)-soc,0.)
                power=s.ac_power if s.home[i] else 50.
                reason='necessary_home_topup' if s.home[i] else 'necessary_nonhome_topup'
        given=max(min(requested,power*s.dwell[i]/3600*efficiency,s.capacity-soc),0.)
        if given>1e-10:
            grid_given=given/efficiency
            duration=grid_given/power*3600;soc+=given;charge+=given
            p=np.zeros(24)
            allocate_charging_event(p,start_hour=s.arrival[i]/3600,duration_hours=duration/3600,energy_kwh=grid_given)
            profile+=p
            if s.home[i]:home_profile+=p
            if capture:
                sessions.append(dict(vehicle=s.vehicle,powertrain=s.kind,trip_index=i,arrivalTime=s.arrival[i],
                    parkingTime_s=s.dwell[i],activity=s.activity[i],is_home=bool(s.home[i]),
                    reason=reason,power_kW=power,timeCharging_s=duration,energyCharged_kWh=given,
                    soc_before_kWh=before,soc_after_kWh=soc))
                if efficiency != 1.:
                    sessions[-1].update(grid_power_kW=power,battery_power_kW=power*efficiency,
                        grid_energy_kWh=grid_given,battery_energy_kWh=given,
                        charging_loss_kWh=grid_given-given)
        if capture:
            trip_rows.append(dict(vehicle=s.vehicle,powertrain=s.kind,trip_index=i,departureTime=s.departure[i],
                arrivalTime=s.arrival[i],requested_electric_kWh=demand,served_electric_kWh=used,
                BEV_unmet_kWh=deficit if s.kind=='BEV' else 0.,
                PHEV_fuel_substitution_electric_equiv_kWh=deficit if s.kind=='PHEV' else 0.,
                soc_arrival_kWh=before,soc_departure_next_kWh=soc))
    residual=initial+charge-served-soc
    if abs(residual)>1e-8 or soc>s.capacity+1e-8:raise ValueError('SOC accounting error')
    np.testing.assert_allclose(profile.sum(),charge/efficiency,atol=1e-8)
    return dict(initial=initial,final=soc,charge=charge,traction=served,requested=float(s.energy.sum()),
        shortfall=shortfall,min_soc=min_soc,profile=profile,home_profile=home_profile,
        sessions=sessions,trips=trip_rows,residual=residual,
        grid_charge=charge/efficiency,battery_profile=profile*efficiency)


def run_periodic(schedules, max_days=2000, tolerance=1e-7, reserve=2., *, efficiency=1.):
    if max_days < 1:
        raise ValueError('max_days must be positive')
    vehicles=[];sessions=[];trips=[];first_sessions=[];first_trips=[];history=[]
    profiles={key:np.zeros(24) for key in ['first_all','first_feasible','steady_all','steady_feasible',
        'first_home_feasible','steady_home_feasible','steady_BEV','steady_PHEV']}
    for s in schedules:
        first=simulate_day(s,s.initial,reserve,capture=True,efficiency=efficiency)
        start=s.initial
        converged=False
        for day in range(1,max_days+1):
            r=simulate_day(s,start,reserve,efficiency=efficiency)
            history.append(dict(vehicle=s.vehicle,day=day,initial_kWh=start,final_kWh=r['final'],
                charge_kWh=r['charge'],shortfall_kWh=r['shortfall']))
            if abs(r['final']-start)<tolerance:
                converged=True
                break
            start=r['final']
        steady=simulate_day(s,r['initial'],reserve,capture=True,efficiency=efficiency)
        first_valid=s.kind=='PHEV' or first['shortfall']<1e-8
        valid=converged and (s.kind=='PHEV' or steady['shortfall']<1e-8)
        profiles['first_all']+=first['profile'];profiles['steady_all']+=steady['profile']
        if first_valid:
            profiles['first_feasible']+=first['profile'];profiles['first_home_feasible']+=first['home_profile']
        if valid:
            profiles['steady_feasible']+=steady['profile'];profiles['steady_home_feasible']+=steady['home_profile']
            profiles['steady_'+s.kind]+=steady['profile']
        for rows,field in [(first['sessions'],'first_sessions'),(first['trips'],'first_trips'),
                           (steady['sessions'],'sessions'),(steady['trips'],'trips')]:
            for row in rows:row['included_in_profile']=first_valid if field.startswith('first') else valid
            {'first_sessions':first_sessions,'first_trips':first_trips,'sessions':sessions,'trips':trips}[field].extend(rows)
        vehicles.append(dict(vehicle=s.vehicle,powertrain=s.kind,capacity_kWh=s.capacity,
            home_preferred=s.home_preferred,primary_home=bool(s.primary>=0 and s.home[s.primary]),
            ac_power_kW=s.ac_power,iterations=day,converged=converged,
            first_included=first_valid,steady_included=valid,
            first_charge_kWh=first['charge'],first_SOC_gain_kWh=first['final']-first['initial'],
            first_shortfall_kWh=first['shortfall'],steady_charge_kWh=steady['charge'],
            steady_requested_kWh=steady['requested'],steady_traction_kWh=steady['traction'],
            steady_shortfall_kWh=steady['shortfall'],steady_SOC_gain_kWh=steady['final']-steady['initial'],
            steady_min_SOC_kWh=steady['min_soc']))
        if efficiency != 1.:
            for phase,result in [('first',first),('steady',steady)]:
                vehicles[-1].update({phase+'_grid_charge_kWh':result['grid_charge'],
                    phase+'_battery_charge_kWh':result['charge'],
                    phase+'_charging_loss_kWh':result['grid_charge']-result['charge']})
    return dict(vehicles=pd.DataFrame(vehicles),sessions=pd.DataFrame(sessions),trips=pd.DataFrame(trips),
        first_sessions=pd.DataFrame(first_sessions),first_trips=pd.DataFrame(first_trips),
        history=pd.DataFrame(history),profiles=profiles)


BATTERY_KWH_PER_KM = .202
CHARGING_EFFICIENCY = .90


def build_phev_home_schedules(trips, stop_audit, seed=42):
    # Reuse established itinerary, Home classification, initial SOC and AC draws.
    source = build_schedules(trips, stop_audit, 1., seed, base_kWh_per_km=.224)
    return [replace(s, energy=s.energy/(.224*.2)*BATTERY_KWH_PER_KM)
            for s in source if s.kind == 'PHEV']


def build_phev_preference_schedules(trips, stop_audit, home_share=.8, seed=42):
    """Reuse BEV-style primary-location preference, but full PHEV electric demand.

    The inherited 20% proxy is undone BEFORE chronological SOC simulation.
    Home share is assigned among eligible terminal-home vehicles, not energy.
    Nonpreferred vehicles choose Work/Education, then other long stops, then Home.
    """
    source=build_schedules(trips,stop_audit,home_share,seed,
                           base_kWh_per_km=BATTERY_KWH_PER_KM)
    return [replace(s,energy=s.energy/.2) for s in source if s.kind=='PHEV']


def simulate_phev_day(s, initial, *, reserve=2., efficiency=CHARGING_EFFICIENCY,
                      rate=BATTERY_KWH_PER_KM, capture=False, charging_policy='home_only'):
    if charging_policy not in ('home_only','primary_preference'):
        raise ValueError('Invalid charging policy')
    if s.kind != 'PHEV':
        raise ValueError('PHEV schedules only')
    if not np.isfinite([initial,reserve,efficiency,rate,s.ac_power]).all():
        raise ValueError('Non-finite parameters')
    if not (0<=initial<=s.capacity and 0<=reserve<s.capacity and 0<efficiency<=1 and rate>0 and s.ac_power>0):
        raise ValueError('Invalid SOC, efficiency, consumption or power')
    if not np.isfinite(s.energy).all() or (s.energy<0).any():
        raise ValueError('Invalid trip energy')
    soc=float(initial)
    battery_charge=grid_charge=electric=fuel_equivalent=0.
    grid_profile=np.zeros(24)
    battery_profile=np.zeros(24)
    sessions=[]; trip_rows=[]
    for i,demand in enumerate(s.energy):
        start=soc
        used=min(float(demand),max(soc-reserve,0.))
        remainder=max(float(demand)-used,0.)
        soc-=used; electric+=used; fuel_equivalent+=remainder
        arrival_soc=soc
        # Preserve old home-only policy exactly. Preference scenarios instead
        # charge at the selected primary stop, with no emergency/DC topups.
        eligible=bool(s.home[i]) if charging_policy=='home_only' else i==s.primary
        if eligible and s.dwell[i]>900:
            battery_added=min(s.capacity-soc,s.ac_power*s.dwell[i]/3600*efficiency)
            if battery_added>1e-10:
                grid_added=battery_added/efficiency
                duration=grid_added/s.ac_power*3600
                soc+=battery_added; battery_charge+=battery_added; grid_charge+=grid_added
                grid=np.zeros(24)
                allocate_charging_event(grid,start_hour=s.arrival[i]/3600,
                    duration_hours=duration/3600,energy_kwh=grid_added)
                grid_profile+=grid; battery_profile+=grid*efficiency
                if capture:
                    sessions.append(dict(vehicle=s.vehicle,powertrain='PHEV',trip_index=i,
                        arrivalTime=float(s.arrival[i]),activity=str(s.activity[i]),is_home=bool(s.home[i]),
                        grid_power_kW=s.ac_power,battery_power_kW=s.ac_power*efficiency,
                        grid_energy_kWh=grid_added,battery_energy_kWh=battery_added,
                        charging_loss_kWh=grid_added-battery_added,
                        charging_time_s=duration,parking_time_s=float(s.dwell[i]),
                        soc_before_kWh=arrival_soc,soc_after_kWh=soc))
        if capture:
            trip_rows.append(dict(vehicle=s.vehicle,trip_index=i,
                departureTime=float(s.departure[i]),arrivalTime=float(s.arrival[i]),
                distance_km=float(demand)/rate,electric_distance_km=used/rate,
                fuel_distance_km=remainder/rate,battery_traction_kWh=used,
                fuel_distance_electric_equivalent_kWh=remainder,
                soc_departure_kWh=start,soc_arrival_kWh=arrival_soc,soc_next_departure_kWh=soc,
                destination_is_home=bool(s.home[i])))
    np.testing.assert_allclose(initial+battery_charge-electric,soc,atol=1e-8,rtol=0)
    np.testing.assert_allclose(grid_charge*efficiency,battery_charge,atol=1e-8,rtol=0)
    np.testing.assert_allclose(grid_profile.sum(),grid_charge,atol=1e-8,rtol=0)
    np.testing.assert_allclose(battery_profile.sum(),battery_charge,atol=1e-8,rtol=0)
    np.testing.assert_allclose(electric+fuel_equivalent,s.energy.sum(),atol=1e-8,rtol=0)
    return dict(initial_kWh=initial,final_kWh=soc,SOC_gain_kWh=soc-initial,
        grid_charge_kWh=grid_charge,battery_charge_kWh=battery_charge,
        charging_loss_kWh=grid_charge-battery_charge,battery_traction_kWh=electric,
        electric_distance_km=electric/rate,fuel_distance_km=fuel_equivalent/rate,
        total_distance_km=float(s.energy.sum())/rate,grid_profile=grid_profile,
        battery_profile=battery_profile,sessions=sessions,trips=trip_rows)


def run_phev_periodic(schedules, max_days=2000, tolerance=1e-7, *, charging_policy='home_only'):
    if max_days<1:
        raise ValueError('max_days must be positive')
    vehicles=[]; history=[]
    tables={k:[] for k in ['sessions','trips','first_sessions','first_trips']}
    profiles={p+'_'+side:np.zeros(24) for p in ['first','steady'] for side in ['grid','battery']}
    for s in schedules:
        first=simulate_phev_day(s,s.initial,capture=True,charging_policy=charging_policy)
        start=s.initial
        for day in range(1,max_days+1):
            r=simulate_phev_day(s,start,charging_policy=charging_policy)
            history.append(dict(vehicle=s.vehicle,day=day,initial_kWh=start,final_kWh=r['final_kWh'],
                grid_charge_kWh=r['grid_charge_kWh'],fuel_distance_km=r['fuel_distance_km']))
            if abs(r['SOC_gain_kWh'])<tolerance:
                break
            start=r['final_kWh']
        else:
            raise ValueError(f'PHEV SOC failed to converge: {s.vehicle}')
        steady=simulate_phev_day(s,r['initial_kWh'],capture=True,charging_policy=charging_policy)
        row=dict(vehicle=s.vehicle,powertrain='PHEV',capacity_kWh=s.capacity,
            grid_ac_power_kW=s.ac_power,has_eligible_home=bool(np.any(s.home & (s.dwell>900))),
            iterations=day,converged=True)
        if charging_policy=='primary_preference':
            row.update(charging_policy=charging_policy,home_preferred=s.home_preferred,
                       primary_index=s.primary,primary_home=bool(s.primary>=0 and s.home[s.primary]))
        for phase,result in [('first',first),('steady',steady)]:
            for key,val in result.items():
                if isinstance(val,(float,int,np.floating)):
                    row[phase+'_'+key]=float(val)
            for side in ['grid','battery']:
                profiles[phase+'_'+side]+=result[side+'_profile']
            for key in ['sessions','trips']:
                tables[('first_' if phase=='first' else '')+key].extend(result[key])
        vehicles.append(row)
    return dict(vehicles=pd.DataFrame(vehicles),history=pd.DataFrame(history),
                profiles=profiles,**{k:pd.DataFrame(v) for k,v in tables.items()})
