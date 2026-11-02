"""Existing-EV input adapter for the original charging notebook core.

No native charging sessions or native SOC are used to generate demand. Battery
metadata and driver activities replace the all-passenger adapter's placeholders.
Missing charger inventories and grid load remain explicitly labelled assumptions.
"""
import numpy as np
import pandas as pd

from .notebook_charging import prepare_passenger_stops


def prepare_existing_ev_stops(trips, metadata, activities, original, *, phev_electricity_factor=.1):
    if not np.isfinite(phev_electricity_factor) or not 0 <= phev_electricity_factor <= 1:
        raise ValueError('PHEV electricity factor must be finite and between 0 and 1')
    if not trips.vehicleType.str.startswith(('ev-', 'phev-')).all():
        raise ValueError('Only native BEV/PHEV trips may enter this adapter')
    types = metadata.set_index('vehicleTypeId')
    if types.index.duplicated().any():
        raise ValueError('Vehicle type metadata is not unique')
    missing = set(trips.vehicleType) - set(types.index)
    if missing:
        raise ValueError(f'Missing battery metadata: {missing}')
    matched = types.loc[sorted(set(trips.vehicleType))]
    if not matched.primaryFuelType.str.lower().eq('electricity').all():
        raise ValueError('Expected electricity as primary fuel for selected EVs')
    capacities = types.primaryFuelCapacityInJoule / 3_600_000
    if not np.isfinite(capacities.loc[matched.index]).all() or capacities.loc[matched.index].le(0).any():
        raise ValueError('Invalid electric battery capacity')
    stops, audit = prepare_passenger_stops(trips, original)
    stops['battery_kWh'] = stops.vehicleType.map(capacities)
    stops['powertrain'] = np.where(stops.vehicleType.str.startswith('phev-'), 'PHEV', 'BEV')
    # The original notebook identifies hybrids by capacity <30 kWh. Explicitly
    # verify that this heuristic agrees with the actual powertrain for this run.
    if not np.array_equal(stops.battery_kWh.lt(30), stops.powertrain.eq('PHEV')):
        raise ValueError('Notebook hybrid heuristic conflicts with actual powertrain')
    stops['energyConsumption_kWh'] = stops.length * .17 / 1000 * np.where(stops.powertrain.eq('PHEV'), phev_electricity_factor, 1.)

    stops['driver'] = stops.driver.astype(str)
    stops = stops.drop(columns=['is_home', 'groupedActivityType'])
    if len(activities):
        acts = activities.rename(columns={'person': 'driver', 'time': 'activity_time'}).copy()
        acts['driver'] = acts.driver.astype(str)
        acts['activity_time'] = acts.activity_time.astype(float)
        acts = acts.dropna(subset=['actType', 'activity_time']).sort_values(['activity_time', 'driver'], kind='stable')
        stops['arrivalTime'] = stops.arrivalTime.astype(float)
        stops = pd.merge_asof(stops.sort_values('arrivalTime', kind='stable'), acts,
            left_on='arrivalTime', right_on='activity_time', by='driver',
            direction='forward', tolerance=1800.)
        valid = stops.activity_time.lt(stops.arrivalTime + stops.parkingTime_s)
        stops.loc[~valid, 'actType'] = None
    else:
        stops['activity_time'] = np.nan
        stops['actType'] = None
    labels = stops.actType.fillna('').str.lower()
    stops['is_home'] = labels.eq('home')
    stops['groupedActivityType'] = np.select(
        [stops.is_home, labels.str.contains('work|school|education', regex=True), labels.ne('')],
        ['Home', 'Work/Education', 'Other'], default='Unknown')
    stops['activity_source'] = np.where(labels.ne(''), 'events_driver_actstart', 'unmatched')
    # Only infer home for a vehicle without ANY matched Home; never overwrite
    # an explicitly observed non-home activity with the longest-stop heuristic.
    no_home = set(stops.vehicle) - set(stops.loc[stops.is_home, 'vehicle'])
    candidates = stops.loc[stops.vehicle.isin(no_home) & stops.actType.isna()]
    fallback = candidates.groupby('vehicle').parkingTime_s.idxmax()
    stops.loc[fallback, 'is_home'] = True
    stops.loc[fallback, 'groupedActivityType'] = 'Home'
    stops.loc[fallback, 'activity_source'] = 'longest_unmatched_stop_home_proxy'
    stops = stops.sort_values('PTID').reset_index(drop=True)
    audit.update(activity_source_stops=stops.activity_source.value_counts().to_dict(),
        activity_groups=stops.groupedActivityType.value_counts().to_dict(),
        vehicles_with_home=int(stops.loc[stops.is_home, 'vehicle'].nunique()),
        represented_traction_kwh=float(stops.energyConsumption_kWh.sum()))
    trips_factor = np.where(trips.vehicleType.str.startswith('phev-'), phev_electricity_factor, 1.)
    audit['input_traction_proxy_kwh'] = float((trips.length * .17 / 1000 * trips_factor).sum())
    audit['unrepresented_input_traction_proxy_kwh'] = audit['input_traction_proxy_kwh'] - audit['represented_traction_kwh']
    audit.pop('unrepresented_tail_traction_kwh', None)
    return stops, audit
