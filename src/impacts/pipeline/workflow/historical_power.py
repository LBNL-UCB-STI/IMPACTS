"""Allocate hourly charging using a historical eGRID annual generation mix.

This is average generation accounting, not a reconstruction of hourly dispatch.
All positive-generation plants in the selected ReEDS region contribute to the
denominator, including non-emitting plants. Imports and transmission constraints
are not modeled. Bay Area source filtering happens after allocation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

LB_TO_KG = 0.45359237
TON_TO_KG = 907.18474
MASS_FIELDS = {
    'NOx': ('PLNOXAN', TON_TO_KG, 'SRNOXRTA'),
    'SOx': ('PLSO2AN', TON_TO_KG, 'SRSO2RTA'),
    'CO2': ('PLCO2AN', TON_TO_KG, 'SRCO2RTA'),
    'CH4': ('PLCH4AN', LB_TO_KG, 'SRCH4RTA'),
    'N2O': ('PLN2OAN', LB_TO_KG, 'SRN2ORTA'),
    'CO2e': ('PLCO2EQA', TON_TO_KG, 'SRC2ERTA'),
}
SUPPLEMENTAL = {
    'PM2_5': ('PM', 'PLPM25RTA', 'Subregion Rates', 'SRPM25RTA'),
    'NH3': ('NH3', 'PLNH3RTA', 'NH3 Subregion-level Data', 'SRNH3RTA'),
    'VOC': ('VOC', 'PLVOCRTA', 'VOC Subregion-level Data', 'SRVOCRTA'),
}
NONCOMBUSTION_FUELS = {'SUN', 'WND', 'WAT', 'NUC'}


def read_year_sheet(path, sheet, year):
    data = pd.read_excel(path, sheet_name=sheet, header=1)
    years = pd.to_numeric(data.YEAR, errors='coerce').dropna().unique()
    if len(years) != 1 or years[0] != year:
        raise ValueError(f'{path}: {sheet} does not contain only year {year}.')
    return data


def complete_rate(plants, rate, *, pollutant, year, fallback, policy):
    """Return kg/MWh and a per-plant provenance label; never fill unknowns with 0."""
    rate = pd.to_numeric(rate, errors='raise').copy()
    basis = pd.Series(f'eGRID_{year}_plant', index=plants.index)
    structural_zero = rate.isna() & plants.PLPRMFL.isin(NONCOMBUSTION_FUELS)
    rate.loc[structural_zero] = 0.0
    basis.loc[structural_zero] = 'noncombustion_primary_fuel_assumed_zero_direct_emissions'
    missing = rate.isna()
    if missing.any() and policy == 'subregion':
        rate.loc[missing] = plants.loc[missing, 'SUBRGN'].map(fallback) * LB_TO_KG
        basis.loc[missing] = f'eGRID_{year}_subregion_proxy'
    if rate.isna().any() or not np.isfinite(rate).all() or (rate < 0).any():
        bad = plants.loc[rate.isna() | ~np.isfinite(rate) | rate.lt(0), 'ORISPL'].tolist()
        raise ValueError(f'Missing or invalid {pollutant} factors for plants {bad}. '
                         'A same-year subregion proxy requires missing_rate_policy=subregion.')
    return rate, basis


def prepare_plants(*, egrid_file, supplemental_files, year, reeds_file,
                   region, loss_region, missing_rate_policy):
    import geopandas as gpd

    suffix = str(year)[-2:]
    plants = read_year_sheet(egrid_file, f'PLNT{suffix}', year)
    if plants.ORISPL.duplicated().any():
        raise ValueError('eGRID plant identifiers must be unique.')
    boundaries = gpd.read_file(reeds_file).to_crs('EPSG:4326')
    boundary = boundaries.loc[boundaries.rb.eq(region)]
    if boundary.empty:
        raise ValueError(f'Unknown ReEDS allocation region {region}.')
    # Missing coordinates cannot be silently assigned to a nearby region.
    active = plants.PLNGENAN.gt(0)
    if not np.isfinite(plants.loc[active, ['LAT', 'LON']].to_numpy(dtype=float)).all():
        raise ValueError('Positive-generation plants require valid coordinates.')
    points = gpd.GeoSeries(gpd.points_from_xy(plants.LON, plants.LAT), crs='EPSG:4326')
    inside = points.intersects(boundary.geometry.union_all())
    excluded = plants.loc[inside & ~active, ['ORISPL', 'PNAME', 'PLNGENAN']].copy()
    plants = plants.loc[inside & active].copy().reset_index(drop=True)
    if plants.empty:
        raise ValueError(f'No positive-generation plants in {region}.')
    plants['allocation_weight'] = plants.PLNGENAN / plants.PLNGENAN.sum()
    regional = read_year_sheet(egrid_file, f'SRL{suffix}', year).set_index('SUBRGN')
    for pollutant, (mass, conversion, regional_rate) in MASS_FIELDS.items():
        rate, basis = complete_rate(plants, plants[mass] * conversion / plants.PLNGENAN,
            pollutant=pollutant, year=year, fallback=regional[regional_rate], policy=missing_rate_policy)
        plants[f'{pollutant}_rate_kg_per_MWh'] = rate
        plants[f'{pollutant}_factor_basis'] = basis
    for pollutant, (label, rate_col, region_sheet, region_col) in SUPPLEMENTAL.items():
        path = supplemental_files[pollutant]
        data = read_year_sheet(path, f'{year} {label} Plant-level Data', year)
        if data.ORISPL.duplicated().any():
            raise ValueError(f'Duplicate plant identifiers in {path}.')
        regional = read_year_sheet(path, f'{year} {region_sheet}', year).set_index('SUBRGN')
        rate, basis = complete_rate(plants, plants.ORISPL.map(data.set_index('ORISPL')[rate_col]) * LB_TO_KG,
            pollutant=pollutant, year=year, fallback=regional[region_col], policy=missing_rate_policy)
        plants[f'{pollutant}_rate_kg_per_MWh'] = rate
        plants[f'{pollutant}_factor_basis'] = basis
    losses = read_year_sheet(egrid_file, f'GGL{suffix}', year)
    loss = losses.loc[losses.REGION.eq(loss_region), 'GGRSLOSS']
    if len(loss) != 1 or not 0 <= float(loss.iloc[0]) < 1:
        raise ValueError(f'Invalid grid gross loss for region {loss_region}.')
    return plants, excluded, float(loss.iloc[0])


def allocate_hourly(demand, plants, loss):
    """Conserve end-use energy after gross-grid-loss adjustment, for every hour."""
    from .hourly_power import POLLUTANTS
    if not np.isfinite(loss) or not 0 <= loss < 1:
        raise ValueError('Grid gross loss must be a fraction in [0, 1).')
    weights = plants.allocation_weight.to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Plant allocation weights must be nonnegative and sum to one.')
    generation = demand.charging_demand_MWh.to_numpy()[:, None] / (1 - loss) * weights[None, :]
    hourly = pd.DataFrame({
        'hour_of_year': np.repeat(demand.hour_of_year.to_numpy(), len(plants)),
        'ORISPL': np.tile(plants.ORISPL.to_numpy(), len(demand)),
        'added_generation_MWh': generation.ravel(),
    })
    for pollutant in POLLUTANTS:
        hourly[f'{pollutant}_kg'] = (generation * plants[f'{pollutant}_rate_kg_per_MWh'].to_numpy()[None, :]).ravel()
    annual = hourly.groupby('ORISPL', sort=False)[['added_generation_MWh'] + [f'{p}_kg' for p in POLLUTANTS]].sum()
    annual = plants.merge(annual, on='ORISPL', validate='one_to_one')
    totals = hourly.groupby('hour_of_year').added_generation_MWh.sum().reindex(demand.hour_of_year)
    np.testing.assert_allclose(totals, demand.charging_demand_MWh / (1-loss), rtol=1e-12, atol=1e-9)
    return hourly, annual


def dispatch_historical(output_dir, *, hourly_demand_file, charging_summary_file,
                        breathe_root, electricity_year, balancing_area,
                        grid_loss_region, missing_rate_policy):
    from .hourly_power import load_hourly_demand, save_json, POLLUTANTS

    root, output = Path(breathe_root), Path(output_dir)
    year = electricity_year
    # Workbook names identify releases; individual sheets identify data years.
    egrid = root / 'input_Data' / f'egrid{year}_data_v2.xlsx'
    supplemental = dict(zip(SUPPLEMENTAL, [root / 'input_Data' / f for f in (
        'egrid-draft-pm-emissions.xlsx', 'egrid2021-draft-nh3-emissions.xlsx',
        'egrid2021-draft-voc-emissions.xlsx')]))
    reeds = root / 'ExternalData/ReEDS_BA/US_PCA/US_PCA.shp'
    demand = load_hourly_demand(hourly_demand_file)
    plants, excluded, loss = prepare_plants(egrid_file=egrid, supplemental_files=supplemental,
        year=year, reeds_file=reeds, region=balancing_area, loss_region=grid_loss_region,
        missing_rate_policy=missing_rate_policy)
    hourly, annual = allocate_hourly(demand, plants, loss)
    allocation = output / 'plant_allocation'
    allocation.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(output / 'sfbay_hourly_charging_demand.parquet', index=False)
    shutil.copy2(charging_summary_file, output / 'beam_charging_summary.json')
    hourly.to_parquet(allocation / 'plant_hourly_generation_emissions.parquet', index=False)
    annual.to_csv(allocation / 'plant_emissions_approx.csv', index=False)
    plants.to_csv(allocation / 'plant_allocation_factors.csv', index=False)
    excluded.to_csv(allocation / 'excluded_nonpositive_or_missing_generation.csv', index=False)
    hourly_summary = hourly.groupby('hour_of_year')[[f'{p}_kg' for p in POLLUTANTS] + ['added_generation_MWh']].sum()
    hourly_summary['enduse_MWh'] = demand.set_index('hour_of_year').charging_demand_MWh
    hourly_summary.to_csv(allocation / 'hourly_dispatch_summary.csv')
    files = [egrid, *supplemental.values(), *sorted(reeds.parent.glob(reeds.stem + '.*')),
             Path(hourly_demand_file), Path(charging_summary_file)]
    provenance = [{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    proxies = {p: int(plants[f'{p}_factor_basis'].str.endswith('_proxy').sum()) for p in POLLUTANTS}
    qa = {
        'allocation_method': 'historical_generation_share', 'electricity_year': year,
        'egrid_year': year, 'pollutant_factor_years': {p: year for p in POLLUTANTS},
        'allocation_region': balancing_area, 'allocation_region_definition': 'ReEDS polygon; not a historical balancing-authority dispatch footprint',
        'cambium_used': False, 'charging_source': str(hourly_demand_file), 'charging_scale_factor': 1.0,
        'grid_loss_region': grid_loss_region, 'grid_gross_loss_fraction': loss,
        'grid_loss_formula': 'generation_MWh = charging_meter_MWh / (1 - grid_gross_loss_fraction)',
        'missing_rate_policy': missing_rate_policy, 'subregion_proxy_plant_counts': proxies,
        'subregion_proxy_generation_share': {p: float(plants.loc[plants[f'{p}_factor_basis'].str.endswith('_proxy'), 'allocation_weight'].sum()) for p in POLLUTANTS},
        'allocation_plant_count': len(plants), 'reference_generation_MWh': float(plants.PLNGENAN.sum()),
        'annual_charging_MWh': float(demand.charging_demand_MWh.sum()),
        'annual_added_generation_MWh': float(annual.added_generation_MWh.sum()),
        'annual_emissions_kg': {p: float(annual[f'{p}_kg'].sum()) for p in POLLUTANTS},
        'hourly_annual_totals_match': True, 'input_files': provenance,
        'power_isrm_completed': False,
        'supplemental_method': 'EPA draft estimates for the selected inventory year; not all factors are directly measured in that year',
        'time_alignment': 'Input hour_of_year preserved; no Cambium time shift or daylight-saving correction',
        'limitations': 'Fixed annual shares and emission rates, including renewable generation at all hours. Not historical hourly dispatch or marginal emissions. Imports and network constraints omitted.',
    }
    save_json(allocation / 'qa_summary.json', qa)
    save_json(output / 'power_generation_summary.json', qa)
