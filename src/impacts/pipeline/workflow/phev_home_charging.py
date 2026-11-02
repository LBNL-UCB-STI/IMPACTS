"""Home-only PHEV sensitivity with battery/grid ledgers and SOC-driven fuel use.

Fuel distance is a modeled fallback, not measured fuel consumption or emissions.
This module does not modify the original notebook or the BEV implementation.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .home_priority_charging import build_schedules
from .power_generation import allocate_charging_event

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
