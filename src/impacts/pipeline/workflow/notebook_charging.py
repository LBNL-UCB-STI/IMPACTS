"""Uncalibrated charging-code reuse audit, not a complete notebook-input replacement.

Load selected original function definitions (never notebook I/O cells). Missing
plans/inventory/load inputs remain explicit proxies. Indexed assignment avoids
the original O(vehicles * stops) scans; regression tests compare its actual choices
with the notebook. No EVI data are accepted by any demand-generation function.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from .power_generation import allocate_charging_event


def load_notebook_functions(folder: Path) -> tuple[dict, dict]:
    namespace = dict(np=np, pd=pd, beta=beta, tqdm=lambda x, **kw: x,
                     BETA_A0=1.99, BETA_A1=48.05, BETA_B0=2.77, BETA_B1=9.30,
                     AVG_E_CONS_kWh_KM=.17, CORR_DAILY_RANGE=1,
                     THRESHOLD_HYBRID_KWH=30, HYBRID_ELECTRICITY_CONSUMPTION_FACTOR=.1,
                     POWER_BY_TYPE={'L1': 1.9, 'L2': 7.2, '_F': 50.},
                     l1_share=.2, l2_share=.8, P1=.45, P2=.04, SH=.8)
    wanted = {
        'GenerateVehicleActivity.ipynb': {'parkingTimecol', 'travelLengthCol'},
        'AssignChargingEvents.ipynb': {'_p', 'add_initial_soc', 'add_daily_soc',
                                     'decide_daily_charges', 'build_ptid_weights',
                                     'pre_assign_charging', 'mark_attemptCharges', 'fill_time_bins'},
    }
    provenance = {}
    for filename, names in wanted.items():
        path = folder / filename
        notebook = json.loads(path.read_text(encoding='utf-8'))
        found = {}
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in names:
                    found[node.name] = node
        if set(found) != names:
            raise ValueError(f'Missing notebook functions: {names - set(found)}')
        exec(compile(ast.Module(body=list(found.values()), type_ignores=[]), str(path), 'exec'), namespace)
        provenance[filename] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'functions': sorted(found)}
    return namespace, provenance


def prepare_passenger_stops(trips: pd.DataFrame, original: dict) -> tuple[pd.DataFrame, dict]:
    """Use original parking and distance helpers; preserve zero dwell as zero.

    Match the notebook's arrival < 29h analysis window. Invalid/overlapping and
    >24h vehicle itineraries are excluded and counted, not wrapped into parking.
    All retained passenger types are hypothetical BEVs with a 75 kWh proxy.
    """
    trips = trips.copy()
    audit = {'input_vehicles': int(trips.vehicle.nunique()), 'input_trips': len(trips)}
    valid = np.isfinite(trips[['departureTime', 'arrivalTime', 'length']]).all(axis=1)
    valid &= trips.length.ge(0) & trips.departureTime.ge(0) & trips.arrivalTime.ge(trips.departureTime)
    audit['invalid_rows'] = int((~valid).sum())
    trips = trips.loc[valid].copy()
    audit['trips_outside_notebook_29h_window'] = int(trips.arrivalTime.ge(104400).sum())
    trips = trips.loc[trips.arrivalTime.lt(104400)].copy()
    trips['vehicle'] = trips.vehicle.astype(str)
    # Resolve equal departure timestamps chronologically before the notebook's
    # vehicle/departure sort (zero-duration BEAM segments are common).
    trips = trips.sort_values(['vehicle', 'departureTime', 'arrivalTime'], kind='stable')
    trips = original['parkingTimecol'](trips)
    bad_vehicles = trips.loc[trips.parkingTime_s.lt(0), 'vehicle'].unique()
    audit['excluded_overlap_or_over_24h_vehicles'] = len(bad_vehicles)
    trips = trips.loc[~trips.vehicle.isin(bad_vehicles)].copy()
    trips = original['travelLengthCol'](trips)
    audit['zero_dwell_segments'] = int(trips.parkingTime_s.eq(0).sum())
    audit['window_vehicles'] = int(trips.vehicle.nunique())
    audit['window_distance_km'] = float(trips.length.sum() / 1000)
    stops = trips.loc[trips.parkingTime_s.gt(900) & trips.dailyVehicleLength_m.gt(0)].copy()
    # Original notebook carries distance across discarded short/zero dwell stops.
    stops['length'] = stops.groupby('vehicle').cumulativeTravelLength_m.diff().fillna(stops.cumulativeTravelLength_m)
    stops = stops.reset_index(drop=True)
    stops['PTID'] = np.arange(len(stops), dtype=np.int64)
    stops['battery_kWh'] = 75.
    stops['is_home'] = False
    stops.loc[stops.groupby('vehicle').parkingTime_s.idxmax(), 'is_home'] = True
    stops['groupedActivityType'] = np.where(stops.is_home, 'Home', 'Other')
    stops['netEnergyLoadGrid_MW'] = 0.
    for field in ['lev1Chargers_cnt', 'lev2Chargers_cnt', 'fastChargers_cnt']:
        stops[field] = 1.
    stops['energyConsumption_kWh'] = stops.length * .17 / 1000
    stops['binLabel_s'] = (stops.arrivalTime // 3600 % 24).astype(int) * 3600
    audit['modeled_vehicles'] = int(stops.vehicle.nunique())
    audit['eligible_stops'] = len(stops)
    audit['modeled_daily_distance_km'] = float(stops.groupby('vehicle').dailyVehicleLength_m.first().sum()/1000)
    audit['represented_traction_kwh'] = float(stops.energyConsumption_kWh.sum())
    audit['unrepresented_tail_traction_kwh'] = float(audit['window_distance_km']*.17 - stops.energyConsumption_kWh.sum())
    return stops, audit


def assign_stops_indexed(stops: pd.DataFrame, original: dict, seed: int = 42) -> pd.DataFrame:
    """Match original preassignment and two-stage PTID/type draws on stop rows."""
    df = original['add_initial_soc'](stops.copy(), seed)
    df = original['add_daily_soc'](df)
    critical = {}
    for vehicle, sub in df.groupby('vehicle', sort=True):
        need = max(10. - float(sub.soc_kWh.min()), 0.)
        if need <= 0:
            continue
        power = np.where(sub.is_home, 7.2, np.where(sub.fastChargers_cnt.gt(0), 50.,
                         np.where(sub.lev2Chargers_cnt.gt(0), 7.2, 1.9)))
        cap = sub.parkingTime_s.to_numpy() * power / 3600
        eligible = np.flatnonzero((cap > 0) & (sub.soc_kWh.to_numpy() > 2))
        order = eligible[np.argsort(-cap[eligible], kind='quicksort')]
        take = int(np.searchsorted(np.cumsum(cap[order]), need)) + 1
        for i in order[:take]:
            critical[int(sub.PTID.iloc[i])] = '_F' if power[i] == 50 else 'L2' if power[i] == 7.2 else 'L1'
    vmap = original['decide_daily_charges'](df, p1=.45, p2=.04, seed=seed)
    grp = original['build_ptid_weights'](df.sort_values('PTID').copy(), l1_share=.2, l2_share=.8)
    np.random.seed(seed)
    home_pick = set(np.random.choice(df.vehicle.unique(), size=int(.8*df.vehicle.nunique()), replace=False))
    crit_vehicles = set(grp.loc[grp.PTID.isin(critical), 'vehicle'])
    choices = dict(critical)
    # Each group retains original melt order: PTIDs ascending, types L1/L2/_F.
    grouped = grp.groupby('vehicle', sort=True).indices
    all_weights = grp[['w_L1', 'w_L2', 'w_F']].to_numpy()
    all_ptids = grp.PTID.to_numpy()
    all_home = grp.is_home.to_numpy()
    for count, (vehicle, positions) in enumerate(grouped.items()):
        if count % 25000 == 0:
            print(f'Assign stops: {count:,}/{len(grouped):,}', flush=True)
        need = 0 if vehicle in crit_vehicles else vmap[vehicle]
        if not need:
            continue
        pos = positions
        if need < 2:
            pos = pos[all_home[pos] == (vehicle in home_pick)]
        candidates = []
        for i in pos:
            weights = all_weights[i]
            kinds = np.flatnonzero(weights > 0)
            if not len(kinds):
                continue
            kind = int(np.random.choice(kinds, p=weights[kinds]/weights[kinds].sum()))
            candidates.append((int(all_ptids[i]), kind, weights[kind]))
        if candidates:
            weights = np.array([c[2] for c in candidates])
            sampled = np.random.choice(len(candidates), min(need, len(candidates)), replace=False,
                                       p=weights/weights.sum())
            for i in sampled:
                ptid, kind, _ = candidates[i]
                choices[ptid] = ('L1', 'L2', '_F')[kind]
    df['ChosenChargerType'] = df.PTID.map(choices)
    df['attemptCharge'] = df.ChosenChargerType.notna().astype(int)
    df['chargerPower_kW'] = df.ChosenChargerType.map(original['POWER_BY_TYPE']).fillna(0)
    return df


def fill_stops(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Original constant-power SOC integration, with before-charge diagnostics.

    Stop-level integration equals bin integration for constant charger power and
    static availability. Physical minimum-before-charge is retained separately;
    the notebook's filter is based only on post-charge SOC.
    """
    result = df.sort_values(['vehicle', 'departureTime']).reset_index(drop=True).copy()
    n = len(result)
    given = np.zeros(n)
    durations = np.zeros(n)
    after = np.zeros(n)
    before = np.zeros(n)
    first_bin_after = np.zeros(n)
    energy = result.energyConsumption_kWh.to_numpy()
    power = result.chargerPower_kW.to_numpy()
    dwell = result.parkingTime_s.to_numpy()
    initial = result.initialsoc_kWh.to_numpy()
    battery = result.battery_kWh.to_numpy()
    arrival = result.arrivalTime.to_numpy()
    vehicles = []
    for vehicle, pos in result.groupby('vehicle', sort=True).indices.items():
        soc = initial[pos[0]]
        for i in pos:
            soc -= energy[i]
            before[i] = soc
            if power[i] > 0 and soc < battery[i]:
                given[i] = min(power[i]*dwell[i]/3600, battery[i]-soc)
                durations[i] = given[i]/power[i]*3600
                soc += given[i]
            after[i] = soc
            first_bin_seconds = min(dwell[i], 3600. - arrival[i] % 3600.)
            first_bin_after[i] = before[i] + min(given[i], power[i]*first_bin_seconds/3600.)
        vehicles.append(dict(vehicle=vehicle, initial_soc_kwh=initial[pos[0]], final_soc_kwh=soc,
                             traction_kwh=float(energy[pos].sum()), charging_kwh=float(given[pos].sum()),
                             min_before_charge_kwh=float(before[pos].min()),
                             min_after_charge_kwh=float(after[pos].min()),
                             min_after_hour_bin_kwh=float(first_bin_after[pos].min()),
                             retained=bool(first_bin_after[pos].min() >= 2.)))
    result['energyCharged_kWh'] = given
    result['timeCharging_s'] = durations
    result['socBeforeCharging_kWh'] = before
    result['socAfterCharging_kWh'] = after
    result['socAfterFirstHourBin_kWh'] = first_bin_after
    audits = pd.DataFrame(vehicles)
    residual = audits.initial_soc_kwh + audits.charging_kwh - audits.traction_kwh - audits.final_soc_kwh
    if not np.allclose(residual, 0., atol=1e-8):
        raise ValueError('SOC accounting failed')
    result['retained'] = result.vehicle.isin(audits.loc[audits.retained, 'vehicle'])
    profile = np.zeros(24)
    for row in result.loc[result.retained & result.energyCharged_kWh.gt(0)].itertuples():
        if row.timeCharging_s > row.parkingTime_s + 1e-7 or row.parkingTime_s <= 900:
            raise ValueError('Charging outside an eligible parking interval')
        allocate_charging_event(profile, start_hour=row.arrivalTime/3600,
                                duration_hours=row.timeCharging_s/3600, energy_kwh=row.energyCharged_kWh)
    return result, audits, profile
