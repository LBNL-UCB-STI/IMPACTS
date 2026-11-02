"""Auditable class-specific truck energy and parking-constrained charging.

Electricity rates are GRID-side. Battery SOC is tracked separately, avoiding a
second application of the losses already included in EMFAC/EPA rates.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .power_generation import allocate_charging_event

EMFAC_URL = 'https://ww2.arb.ca.gov/sites/default/files/2021-08/emfac2021_technical_documentation_april2021.pdf'
EPA_URL = 'https://www.energystar.gov/products/ev_chargers/electric-vehicles-and-hybrids/details/45318'
FORD_URL = 'https://media.ford.com/content/dam/fordmedia/North%20America/US/product/2022/f-150-lightning/pdf/F-150_Lightning_Tech_Specs_08102022.pdf'
TRANSIT_URL = 'https://www.fromtheroad.ford.com/us/en/articles/2024/ford-pro-ships-e-transit-with-enhanced-range'
FREIGHTLINER_URL = 'https://www.freightliner.com/blog-and-newsletters/electric-truck-faqs/'
SPEED_MPH = np.arange(5, 66, 5, dtype=float)
SPEED_FACTORS = np.array([2.85, 1.35, 1.13, 1., .92, .74, .75, .79, .84, .84, .87, .86, .91])
PARAMETERS = {
    'ld1': dict(beam_class='Class 1-2a vocational', energy_model='EPA combined pickup proxy',
                grid_kwh_per_mile=.49, speed_corrected=False, usable_battery_kwh=98.,
                battery_proxy='2022 F-150 Lightning standard range', grid_charger_kw=50.,
                energy_source=EPA_URL, battery_source=FORD_URL),
    'ld3': dict(beam_class='Class 2b-3 vocational', energy_model='EMFAC2021 LHD',
                grid_kwh_per_mile=.66, speed_corrected=True, usable_battery_kwh=89.,
                battery_proxy='2024 Ford E-Transit enhanced range', grid_charger_kw=50.,
                energy_source=EMFAC_URL, battery_source=TRANSIT_URL),
    'mdv': dict(beam_class='Class 4-6 vocational', energy_model='EMFAC2021 MHD proxy',
                grid_kwh_per_mile=1.45, speed_corrected=True, usable_battery_kwh=194.,
                battery_proxy='Freightliner eM2 Class 6', grid_charger_kw=100.,
                energy_source=EMFAC_URL, battery_source=FREIGHTLINER_URL),
    'hdv': dict(beam_class='Class 7-8 vocational', energy_model='EMFAC2021 HHD proxy',
                grid_kwh_per_mile=2.07, speed_corrected=True, usable_battery_kwh=291.,
                battery_proxy='Freightliner eM2 Class 7', grid_charger_kw=150.,
                energy_source=EMFAC_URL, battery_source=FREIGHTLINER_URL),
    'hdt': dict(beam_class='Class 7-8 tractor', energy_model='EMFAC2021 HHD proxy',
                grid_kwh_per_mile=2.07, speed_corrected=True, usable_battery_kwh=438.,
                battery_proxy='Freightliner eCascadia Class 8', grid_charger_kw=150.,
                energy_source=EMFAC_URL, battery_source=FREIGHTLINER_URL),
}
ASSUMPTIONS = {
    'parameter_version': 'truck-proxies-v1-20260923',
    'emfac_reference': 'EMFAC2021 Volume III Tables 3.2.2-1 and 3.2.2-2',
    'energy_boundary': 'Grid-side kWh: EMFAC includes 15% grid-to-battery loss; EPA includes charging losses.',
    'battery_efficiency': .85,
    'battery_efficiency_note': '0.85 converts grid energy to SOC; applied to EPA pickup as an explicit assumption too.',
    'soc_min': .10, 'soc_max': .90, 'min_parking_minutes': 15.,
    'speed_rule': 'Trip-average distance/time; linearly interpolate EMFAC factors; clamp to 5-65 mph; EPA LD1 has no speed correction.',
    'zero_duration_rule': 'Nonzero distance with zero duration uses 20 mph; flagged in trip audit; not a measured speed.',
    'charger_rule': 'Constant grid-side 50/100/150 kW scenario powers, below representative maximum ratings; not measured charge curves.',
    'parking_rule': 'Opportunity charging at every >=15 minute stop; no infrastructure or queue constraint; no charging while driving.',
    'soc_rule': 'Greedy charging to 90% during parking, cyclic SOC equilibrium; unmet traction energy is retained and separately reported.',
    'time_rule': 'Each vehicle repeats its itinerary over the smallest whole-day cycle covering it. Clock-hour demand is averaged over cycle days (staggered phases), then repeated 365 days.',
    'scope': 'Representative model proxies, not measured electric fleet specifications; payload is NOT interpreted as GVWR; no extra payload/grade/HVAC calibration.',
}


def trip_energy(trips: pd.DataFrame) -> pd.DataFrame:
    result = trips.copy()
    required = ['vehicle', 'vehicleType', 'departureTime', 'arrivalTime', 'length']
    if result[required].isna().any().any():
        raise ValueError('Missing truck trip fields.')
    result['truck_class'] = result.vehicleType.str.extract(r'^ft-(ld1|ld3|mdv|hdv|hdt)-(?:Dsl|Gas)$')[0]
    if result.truck_class.isna().any():
        raise ValueError('Unmapped truck vehicleType.')
    values = result[['departureTime', 'arrivalTime', 'length']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid truck time or distance.')
    seconds = result.arrivalTime - result.departureTime
    if (seconds < 0).any():
        raise ValueError('Truck arrival precedes departure.')
    result['distance_km'] = result.length / 1000.
    result['speed_fallback'] = seconds.eq(0) & result.length.gt(0)
    result['speed_mph'] = result.distance_km / 1.609344 / (seconds.replace(0, np.nan) / 3600.)
    result['speed_mph'] = result.speed_mph.fillna(20.)
    result['speed_clamped'] = result.speed_mph.lt(5) | result.speed_mph.gt(65)
    corrected = result.truck_class.map({k: p['speed_corrected'] for k, p in PARAMETERS.items()})
    result['speed_factor'] = np.where(corrected, np.interp(result.speed_mph, SPEED_MPH, SPEED_FACTORS), 1.)
    result['grid_kwh_per_mile'] = result.truck_class.map({k: p['grid_kwh_per_mile'] for k, p in PARAMETERS.items()}) * result.speed_factor
    result['required_grid_kwh'] = result.distance_km / 1.609344 * result.grid_kwh_per_mile
    result['traction_battery_kwh'] = result.required_grid_kwh * ASSUMPTIONS['battery_efficiency']
    return result.sort_values(['vehicle', 'departureTime', 'arrivalTime']).reset_index(drop=True)


def schedule_vehicle(trips, *, battery_kwh, grid_kw, efficiency=.85, soc_min=.10,
                     soc_max=.90, min_parking_minutes=15.):
    """Periodic greedy upper-opportunity schedule; never discard infeasible vehicles.

    Unmet energy is a feasibility diagnostic, NOT secretly supplied by a virtual
    charger. It must prevent claiming a complete all-electric scenario.
    """
    departure = trips.departureTime.to_numpy(dtype=float)
    arrival = trips.arrivalTime.to_numpy(dtype=float)
    energy = trips.traction_battery_kwh.to_numpy(dtype=float)
    if not (0 <= soc_min < soc_max <= 1 and battery_kwh > 0 and grid_kw > 0 and 0 < efficiency <= 1):
        raise ValueError('Invalid battery/charger/SOC parameters.')
    if np.any(departure[1:] < arrival[:-1]):
        raise ValueError('Overlapping truck trips.')
    days = max(1, math.ceil((arrival[-1] - departure[0]) / 86400.))
    parking = (np.r_[departure[1:], departure[0] + days * 86400.] - arrival) / 3600.
    if (parking < 0).any():
        raise ValueError('Negative parking interval.')
    capacity = battery_kwh * (soc_max - soc_min)
    possible = np.where(parking * 60 >= min_parking_minutes, parking * grid_kw * efficiency, 0.)

    def cycle(initial):
        available = initial
        charges, missing, after_trip = [], [], []
        for need, allowed in zip(energy, possible):
            missing.append(max(0., need - available))
            available = max(0., available - need)
            after_trip.append(available)
            charge = min(capacity - available, allowed)
            available += charge
            charges.append(charge)
        return available, np.array(charges), np.array(missing), np.array(after_trip)

    initial = capacity
    for _ in range(1000):
        final, charged, unmet, after = cycle(initial)
        if abs(final - initial) < 1e-8:
            break
        initial = final
    else:
        raise ValueError('Cyclic SOC failed to converge.')
    np.testing.assert_allclose(charged.sum() + unmet.sum(), energy.sum(), atol=1e-6, rtol=1e-9)
    return dict(days=days, parking_hours=parking, grid_kwh=charged / efficiency,
                unmet_battery_kwh=unmet, battery_after_trip_kwh=after + battery_kwh * soc_min,
                initial_battery_kwh=initial + battery_kwh * soc_min,
                final_battery_kwh=final + battery_kwh * soc_min)


def build_truck_profiles(trips):
    energy = trip_energy(trips)
    profiles, sessions, vehicles = {}, [], []
    for vehicle, group in energy.groupby('vehicle', sort=True):
        if group.vehicleType.nunique() != 1:
            raise ValueError(f'Vehicle changes truck type: {vehicle}')
        kind, vehicle_type = group.truck_class.iloc[0], group.vehicleType.iloc[0]
        params = PARAMETERS[kind]
        schedule = schedule_vehicle(group, battery_kwh=params['usable_battery_kwh'], grid_kw=params['grid_charger_kw'])
        days = schedule['days']
        profile = profiles.setdefault(vehicle_type, np.zeros(24))
        for i, row in enumerate(group.itertuples()):
            supplied = schedule['grid_kwh'][i]
            if supplied > 1e-9:
                duration = supplied / params['grid_charger_kw']
                allocate_charging_event(profile, start_hour=row.arrivalTime / 3600.,
                                        duration_hours=duration, energy_kwh=supplied / days)
                sessions.append(dict(vehicle=vehicle, vehicleType=vehicle_type, truck_class=kind,
                                     time=row.arrivalTime, charge_time_minutes=duration * 60,
                                     energy_needed_kwh=supplied, representative_daily_kwh=supplied / days,
                                     cycle_days=days, charger_type='truck_opportunity',
                                     grid_charger_kw=params['grid_charger_kw'], is_home_inferred=False))
        vehicles.append(dict(vehicle=vehicle, vehicleType=vehicle_type, truck_class=kind,
                             cycle_days=days, distance_km=group.distance_km.sum(),
                             required_grid_kwh=group.required_grid_kwh.sum(),
                             supplied_grid_kwh=schedule['grid_kwh'].sum(),
                             unmet_grid_kwh=schedule['unmet_battery_kwh'].sum() / .85,
                             battery_kwh=params['usable_battery_kwh'],
                             initial_battery_kwh=schedule['initial_battery_kwh'],
                             final_battery_kwh=schedule['final_battery_kwh'],
                             trips_with_unmet_energy=int((schedule['unmet_battery_kwh'] > 1e-6).sum())))
        energy.loc[group.index, 'unmet_battery_kwh'] = schedule['unmet_battery_kwh']
        energy.loc[group.index, 'battery_after_trip_kwh'] = schedule['battery_after_trip_kwh']
    vehicles = pd.DataFrame(vehicles)
    for field in ['distance_km', 'required_grid_kwh', 'supplied_grid_kwh', 'unmet_grid_kwh']:
        vehicles['daily_' + field] = vehicles[field] / vehicles.cycle_days
    np.testing.assert_allclose(sum(v.sum() for v in profiles.values()), vehicles.daily_supplied_grid_kwh.sum(), atol=1e-6)
    return profiles, pd.DataFrame(sessions), vehicles, energy
