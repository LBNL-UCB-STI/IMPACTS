"""Cristian-compatible LDV charging and prepared-activity adapters.

run_ldv_charging executes the supplied original notebook functions and
literal settings; it is the preferred compatible model. Only definitions and
settings are loaded, never notebook I/O or plotting cells. The input-preparation
and optimized single-stop helpers retain their explicit historical assumptions
and parity tests. No EVI fitting or periodic/electric-first model is added.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

from ..power_generation import allocate_charging_event


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


PARAMETERS = (
    'BETA_A0', 'BETA_A1', 'BETA_B0', 'BETA_B1', 'AVG_E_CONS_kWh_KM',
    'POWER_BY_TYPE', 'CORR_DAILY_RANGE', 'criticalSOC_kWh', 'minimumAllowed_kWh',
    'l1_share', 'l2_share', 'P1', 'P2', 'SH',
    'HYBRID_ELECTRICITY_CONSUMPTION_FACTOR', 'THRESHOLD_HYBRID_KWH',
)


def load_original(folder: Path):
    original, provenance = load_notebook_functions(folder)
    notebook = json.loads((folder / 'AssignChargingEvents.ipynb').read_text(encoding='utf-8'))
    settings = {}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell['source']))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                names = ([target.id] if isinstance(target, ast.Name) else
                         [n.id for n in target.elts if isinstance(n, ast.Name)]
                         if isinstance(target, (ast.Tuple, ast.List)) else [])
                if not set(names).intersection(PARAMETERS):
                    continue
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    raise ValueError('Charging settings must be explicit notebook literals: ' + str(names))
                values = [value] if isinstance(target, ast.Name) else list(value)
                for name, value in zip(names, values, strict=True):
                    if name in PARAMETERS:
                        settings[name] = value
    missing = set(PARAMETERS) - settings.keys()
    if missing:
        raise ValueError(f'Missing original notebook settings: {sorted(missing)}')
    original.update(settings)
    return original, settings, provenance


@contextmanager
def pandas2_sampling():
    """Retain pre-pandas-3 weighted sampling used by the original notebook.

    This is process-local and intended for serial CLI execution, not concurrent
    notebook calls in threads. No original function source is rewritten.
    """
    previous = pd.DataFrame.sample
    random_state = np.random.get_state()

    def sample(frame, n=1, weights=None, replace=False):
        w = np.asarray(frame[weights] if isinstance(weights, str) else weights, dtype=float)
        if w.ndim != 1 or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError('Invalid original notebook sampling weights')
        return frame.iloc[np.random.choice(len(frame), n, replace=replace, p=w / w.sum())]

    pd.DataFrame.sample = sample
    try:
        yield
    finally:
        pd.DataFrame.sample = previous
        np.random.set_state(random_state)


def expand_stop_bins(stops: pd.DataFrame) -> pd.DataFrame:
    """Adapt one-row-per-stop input to the notebook's hourly parking rows.

    Driving distance is charged to the first bin only; cumulative distance is
    retained in subsequent parking bins. No home, battery, charger or load
    attributes are inferred here.
    """
    if stops.PTID.duplicated().any():
        raise ValueError('Stop input requires globally unique PTIDs')
    rows = []
    for row in stops.to_dict('records'):
        left = float(row['arrivalTime'])
        duration = float(row['parkingTime_s'])
        if not np.isfinite([left, duration]).all() or left < 0 or duration <= 0:
            raise ValueError('Invalid parking interval')
        end = left + duration
        first = True
        while left < end:
            right = min(end, (np.floor(left / 3600) + 1) * 3600)
            rows.append(dict(row, parkingTime_s=right-left,
                             length=row['length'] if first else 0.,
                             binLabel_s=np.floor(left / 3600) * 3600))
            first = False
            left = right
    return pd.DataFrame(rows)


def run_ldv_charging(activity: pd.DataFrame, *, notebooks: Path,
                         seed: int = 42, input_layout: str = 'hourly_bins'):
    """Original single-day charging, including its post-charge SOC filter."""
    original, settings, provenance = load_original(notebooks)
    required = {'vehicle', 'PTID', 'battery_kWh', 'dailyVehicleLength_m',
                'cumulativeTravelLength_m', 'length', 'departureTime', 'arrivalTime',
                'parkingTime_s', 'groupedActivityType', 'lev1Chargers_cnt',
                'lev2Chargers_cnt', 'fastChargers_cnt', 'netEnergyLoadGrid_MW'}
    missing = required - set(activity)
    if missing or activity.empty:
        raise ValueError(f'Prepared activity is empty or missing fields: {sorted(missing)}')
    if input_layout not in ('hourly_bins', 'stops'):
        raise ValueError('input_layout must be hourly_bins or stops')
    data = activity.copy()
    numeric = list(required - {'vehicle', 'PTID', 'groupedActivityType'})
    if not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Prepared activity has non-finite numeric inputs')
    if data.battery_kWh.le(0).any() or data.length.lt(0).any() or data.parkingTime_s.le(0).any():
        raise ValueError('Invalid battery, distance or parking duration')
    if data.groupby('PTID').vehicle.nunique().gt(1).any():
        raise ValueError('A PTID cannot belong to multiple vehicles')
    for col in ('battery_kWh', 'dailyVehicleLength_m'):
        if data.groupby('vehicle')[col].nunique().gt(1).any():
            raise ValueError(f'Inconsistent vehicle-level field: {col}')
    if input_layout == 'stops':
        data = expand_stop_bins(data)
    if 'binLabel_s' not in data or not np.isfinite(data.binLabel_s).all():
        raise ValueError('Hourly activity requires finite binLabel_s values')
    if (data.binLabel_s % 3600 != 0).any() or data.parkingTime_s.gt(3600 + 1e-7).any():
        raise ValueError('Use hourly bins or explicitly select input_layout=stops')
    # Same two preprocessing assignments as the original notebook's input cell.
    data['is_home'] = data.groupedActivityType.eq('Home')
    hybrid = data.battery_kWh.lt(settings['THRESHOLD_HYBRID_KWH'])
    data['energyConsumption_kWh'] = data.length / 1000 * settings['AVG_E_CONS_kWh_KM']
    data.loc[hybrid, 'energyConsumption_kWh'] *= settings['HYBRID_ELECTRICITY_CONSUMPTION_FACTOR']
    with pandas2_sampling():
        data = original['add_initial_soc'](data, seed)
        data = original['add_daily_soc'](data)
        critical = original['pre_assign_charging'](data,
            safety_margin_kWh=settings['criticalSOC_kWh'],
            minimumAllowed_kWh=settings['minimumAllowed_kWh'])
        counts = original['decide_daily_charges'](data, p1=settings['P1'], p2=settings['P2'], seed=seed)
        data = original['mark_attemptCharges'](data, counts, critical, share_home=settings['SH'], seed=seed)
        data['chargerPower_kW'] = data.ChosenChargerType.map(settings['POWER_BY_TYPE']).fillna(0)
        data = original['fill_time_bins'](data, counts)
    data['retained'] = data.groupby('vehicle').socAfterCharging_kWh.transform(
        lambda values: values.ge(settings['minimumAllowed_kWh']).all())
    data['powertrain_proxy'] = np.where(data.battery_kWh.lt(settings['THRESHOLD_HYBRID_KWH']), 'PHEV', 'BEV')
    vehicles = data.groupby('vehicle', sort=True).agg(
        powertrain_proxy=('powertrain_proxy', 'first'), battery_kWh=('battery_kWh', 'first'),
        initial_soc_kWh=('initialsoc_kWh', 'first'), final_soc_kWh=('socAfterCharging_kWh', 'last'),
        traction_kWh=('energyConsumption_kWh', 'sum'), charging_kWh=('energyCharged_kWh', 'sum'),
        minimum_postcharge_soc_kWh=('socAfterCharging_kWh', 'min'), retained=('retained', 'first'))
    np.testing.assert_allclose(vehicles.initial_soc_kWh + vehicles.charging_kWh - vehicles.traction_kWh,
                               vehicles.final_soc_kWh, atol=1e-8, rtol=1e-12)
    kept = data.loc[data.retained].copy()
    kept['hour_of_day'] = (kept.binLabel_s // 3600 % 24).astype(int)
    hourly = kept.groupby(['hour_of_day', 'powertrain_proxy']).energyCharged_kWh.sum().unstack(fill_value=0)
    hourly = hourly.reindex(index=range(24), columns=['BEV', 'PHEV'], fill_value=0).fillna(0)
    hourly.columns = ['BEV_sample_kWh', 'PHEV_sample_kWh']
    hourly['total_sample_kWh'] = hourly.sum(axis=1)
    np.testing.assert_allclose(hourly.total_sample_kWh.sum(), kept.energyCharged_kWh.sum(), atol=1e-8)
    return data, vehicles.reset_index(), hourly.reset_index(), settings, provenance
