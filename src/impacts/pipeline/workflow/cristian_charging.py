"""Execute the original collaborator charging functions without model changes.

Only function definitions and literal settings are loaded; notebook I/O and
plotting cells are not executed. Pandas sampling compatibility is scoped to
this serial call and restores the caller's method and NumPy random state.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .notebook_charging import load_notebook_functions

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


def run_cristian_charging(activity: pd.DataFrame, *, notebooks: Path,
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
