"""Hourly charging -> approximate plant dispatch -> annual emissions -> ISRM.

Shared implementation for the native YAML pipeline and command-line tools.
Fleet expansion is already represented in the input curve; no annual-energy
ratio or saved-concentration scaling is performed here.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
POLLUTANTS = ['NOx', 'SOx', 'CO2', 'CH4', 'N2O', 'CO2e', 'PM2_5', 'NH3', 'VOC']
BAY_COUNTY_FIPS = {'001', '013', '041', '055', '075', '081', '085', '095', '097'}

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')

def load_hourly_demand(path: str | Path) -> pd.DataFrame:
    demand = pd.read_parquet(path)
    required = {'hour_of_year', 'charging_demand_MWh'}
    if not required.issubset(demand.columns):
        raise ValueError(f'Hourly demand requires columns {sorted(required)}')
    demand = demand.sort_values('hour_of_year').reset_index(drop=True)
    hours = pd.to_numeric(demand.hour_of_year, errors='raise').to_numpy()
    load = pd.to_numeric(demand.charging_demand_MWh, errors='raise').to_numpy()
    if len(hours) != 8760 or not np.array_equal(hours, np.arange(8760)):
        raise ValueError('Hourly demand must contain each hour_of_year 0..8759 exactly once.')
    if not np.isfinite(load).all() or (load < 0).any():
        raise ValueError('Hourly charging demand must be finite and nonnegative.')
    demand['charging_demand_MWh'] = load
    return demand

def run_hourly_power(*, config: dict, isrm_url: str, inmap_grid_path: str,
                     output_dir: Path, compute_concentrations: bool = True) -> Path:
    from ...config.settings import PowerGeneration
    settings = PowerGeneration.from_dict(config)
    if not settings.enabled:
        raise ValueError('Power generation is disabled.')
    summary = json.loads(Path(settings.charging_summary_file).read_text(encoding='utf-8'))
    if (summary.get('ready_for_full_electrification_pipeline') is False
            and not settings.allow_incomplete_charging_scenario):
        raise ValueError('Unmet charging energy: resolve feasibility or explicitly allow a supplied-load-only scenario.')
    # Validate all declared inputs before starting the external allocator.
    for value in (settings.hourly_demand_file, settings.county_boundaries_file, inmap_grid_path):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    from dataclasses import asdict
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_json(Path(output_dir) / 'power_generation_config.json', dict(
        power_generation=asdict(settings), isrm_url=isrm_url, inmap_grid_path=inmap_grid_path,
        compute_concentrations=compute_concentrations))
    kwargs = dict(hourly_demand_file=Path(settings.hourly_demand_file),
                  charging_summary_file=Path(settings.charging_summary_file),
                  breathe_root=Path(settings.breathe_root),
                  electricity_year=settings.electricity_year, balancing_area=settings.balancing_area)
    if settings.allocation_method == 'historical_generation_share':
        from .historical_power import dispatch_historical
        dispatch_historical(output_dir, **kwargs, grid_loss_region=settings.grid_loss_region,
                            missing_rate_policy=settings.missing_rate_policy)
    else:
        if not (Path(settings.breathe_root) / 'scripts/allocate_power_plants_approx.py').is_file():
            raise FileNotFoundError('External hourly allocator is missing under breathe_root.')
        dispatch_hourly(output_dir, **kwargs, cambium_scenario=settings.cambium_scenario)
    if not compute_concentrations:
        summary_path = Path(output_dir) / 'power_generation_summary.json'
        summary = json.loads(summary_path.read_text())
        summary.update(power_isrm_completed=False, concentration_status='not_requested',
                       power_source_layer_index=settings.source_layer_index)
        save_json(summary_path, summary)
        return Path(output_dir) / 'plant_allocation/plant_emissions_approx.csv'
    return compute_hourly_power_isrm(output_dir, inmap_grid_path=inmap_grid_path,
        counties_path=settings.county_boundaries_file, isrm_store=isrm_url,
        source_layer_index=settings.source_layer_index)

def select_sfbay_plants(plants, counties):
    """Filter actual plant locations, without reallocating outside generation."""
    import geopandas as gpd
    counties = counties.loc[
        counties.STATEFP.astype(str).str.zfill(2).eq('06')
        & counties.COUNTYFP.astype(str).str.zfill(3).isin(BAY_COUNTY_FIPS)
    ].to_crs('EPSG:4326')
    if set(counties.COUNTYFP.astype(str).str.zfill(3)) != BAY_COUNTY_FIPS:
        raise ValueError('All nine Bay Area county boundaries are required.')
    coordinates = plants[['LAT', 'LON']].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(coordinates.to_numpy()).all():
        raise ValueError('Plant coordinates must be finite.')
    points = gpd.GeoSeries(gpd.points_from_xy(coordinates.LON, coordinates.LAT),
                          index=plants.index, crs='EPSG:4326')
    inside = points.intersects(counties.geometry.union_all())
    return plants.loc[inside].copy()

def dispatch_hourly(
    output_dir: Path, *, hourly_demand_file: Path, charging_summary_file: Path,
    breathe_root: Path, electricity_year: int, balancing_area: str, cambium_scenario: str,
) -> None:
    power = Path(output_dir)
    demand = load_hourly_demand(hourly_demand_file)
    power.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(power / 'sfbay_hourly_charging_demand.parquet', index=False)
    shutil.copy2(charging_summary_file, power / 'beam_charging_summary.json')
    demand = pd.read_parquet(power / 'sfbay_hourly_charging_demand.parquet')
    assert len(demand) == 8760 and demand.hour_of_year.is_unique
    demand = demand.sort_values('hour_of_year')
    load = demand.charging_demand_MWh.to_numpy()
    assert np.isfinite(load).all() and (load >= 0).all()
    load_path = power / f'grid_yearly_beam_{balancing_area}.csv'
    pd.DataFrame([np.r_[load, int(balancing_area[1:])]]).to_csv(load_path, header=False, index=False)
    cambium_dir = Path(breathe_root) / 'tmp/cambium_check/hourly_balancingArea'
    selected_cambium = cambium_dir / f'Cambium22_{cambium_scenario}_hourly_{balancing_area}_{electricity_year}.csv'
    if not selected_cambium.is_file():
        raise FileNotFoundError(selected_cambium)
    selected_dir = power / 'cambium_input'
    selected_dir.mkdir(parents=True, exist_ok=True)
    # The external allocator uses a fixed filename prefix. This alias contains
    # exactly the requested scenario bytes; record the original path below.
    shutil.copy2(selected_cambium, selected_dir / f'Cambium22_LowRECost_hourly_{balancing_area}_{electricity_year}.csv')
    allocation = power / 'plant_allocation'
    command = [sys.executable, str(Path(breathe_root) / 'scripts/allocate_power_plants_approx.py'),
               '--load-csv', str(load_path), '--cambium-dir', str(selected_dir),
               '--egrid-xlsx', str(Path(breathe_root) / 'input_Data/egrid2023_data_rev2.xlsx'),
               '--pm25-xlsx', str(Path(breathe_root) / 'input_Data/egrid-draft-pm-emissions.xlsx'),
               '--nh3-xlsx', str(Path(breathe_root) / 'input_Data/egrid2021-draft-nh3-emissions.xlsx'),
               '--voc-xlsx', str(Path(breathe_root) / 'input_Data/egrid2021-draft-voc-emissions.xlsx'),
               '--reeds-shapefile', str(Path(breathe_root) / 'ExternalData/ReEDS_BA/US_PCA/US_PCA.shp'),
               '--year', str(electricity_year), '--output-dir', str(allocation)]
    subprocess.run(command, check=True)
    cambium = pd.read_csv(cambium_dir / f'Cambium22_{cambium_scenario}_hourly_{balancing_area}_{electricity_year}.csv', skiprows=5)
    assert pd.to_datetime(cambium.timestamp_local).dt.hour.tolist() == (demand.hour_of_year % 24).tolist()
    hourly = pd.DataFrame({'hour_of_year': demand.hour_of_year.to_numpy(),
                           'timestamp_local': cambium.timestamp_local,
                           'source_BA': balancing_area, 'technology': cambium.marg_gen_tech.fillna('unknown'),
                           'enduse_MWh': load,
                           'grid_MWh': load * (1 + cambium.distloss_rate_marg.fillna(0).to_numpy())})
    generators = pd.read_csv(allocation / 'generator_allocation_detail.csv')
    frames = []
    for tech, group in generators.groupby('marginal_technology'):
        weights = group.groupby('ORISPL').allocation_weight.sum()
        assert np.isclose(weights.sum(), 1)
        rates = group.groupby('ORISPL')[[f'{p}_rate_kg_per_MWh' for p in POLLUTANTS]].first().fillna(0)
        hours = hourly.loc[hourly.technology.eq(tech)]
        generation = hours.grid_MWh.to_numpy()[:, None] * weights.to_numpy()[None, :]
        frame = pd.DataFrame({'hour_of_year': np.repeat(hours.hour_of_year.to_numpy(), len(weights)),
                              'ORISPL': np.tile(weights.index.to_numpy(), len(hours)),
                              'marginal_technology': tech,
                              'added_generation_MWh': generation.ravel()})
        for pollutant in POLLUTANTS:
            frame[f'{pollutant}_kg'] = (generation * rates.loc[weights.index, f'{pollutant}_rate_kg_per_MWh'].to_numpy()[None, :]).ravel()
        frames.append(frame)
    plant_hourly = pd.concat(frames, ignore_index=True)
    plant_hourly.to_parquet(allocation / 'plant_hourly_generation_emissions.parquet', index=False)
    totals = plant_hourly.groupby('hour_of_year').sum(numeric_only=True)
    hourly['allocated_MWh'] = hourly.hour_of_year.map(totals.added_generation_MWh).fillna(0)
    hourly['unallocated_MWh'] = hourly.grid_MWh - hourly.allocated_MWh
    for pollutant in POLLUTANTS:
        hourly[f'{pollutant}_kg'] = hourly.hour_of_year.map(totals[f'{pollutant}_kg']).fillna(0)
    hourly.to_csv(allocation / 'hourly_dispatch_summary.csv', index=False)
    annual = pd.read_csv(allocation / 'plant_emissions_approx.csv').set_index('ORISPL')
    annual_check = plant_hourly.groupby('ORISPL').sum(numeric_only=True)
    for col in ['added_generation_MWh'] + [f'{p}_kg' for p in POLLUTANTS]:
        np.testing.assert_allclose(annual_check.loc[annual.index, col], annual[col], rtol=1e-10, atol=1e-7)
    qa = json.loads((allocation / 'qa_summary.json').read_text())
    qa.update({'allocation_method': 'cambium_marginal_technology', 'electricity_year': electricity_year,
               'cambium_input_file': str(selected_cambium),
               'egrid_year': 2023, 'supplemental_factor_year': 2021, 'cambium_scenario': cambium_scenario,
               'allocation_region': balancing_area, 'hourly_annual_totals_match': True, 'annual_emissions_kg': {p: float(annual[f'{p}_kg'].sum()) for p in POLLUTANTS},
               'charging_source': str(hourly_demand_file), 'charging_scale_factor': 1.0,
               'time_alignment': 'BEAM hour-of-day matched to Cambium timestamp_local; representative day repeated 365 times',
               'power_source_layer_index': None})
    save_json(power / 'power_generation_summary.json', qa)
    logger.info('Hourly plant dispatch complete: %s', power)

def compute_hourly_power_isrm(
    output_dir: Path, *, inmap_grid_path: str, counties_path: str,
    isrm_store: str, source_layer_index: int,
) -> Path:
    import geopandas as gpd
    import zarr
    from impacts.pipeline.workflow.power_generation import (
        POWER_COMPONENTS, _map_plants_to_source_cells,
        compute_power_generation_isrm, plot_power_generation_concentration,
    )
    folder = Path(output_dir)
    grid = (gpd.read_parquet(inmap_grid_path) if str(inmap_grid_path).lower().endswith('.parquet') else gpd.read_file(inmap_grid_path))
    all_plants = pd.read_csv(folder / 'plant_allocation/plant_emissions_approx.csv')
    counties = gpd.read_file(counties_path)
    plants = select_sfbay_plants(all_plants, counties)
    plants = plants.loc[plants[list(POWER_COMPONENTS)].gt(0).any(axis=1)].copy()
    if plants.empty:
        raise ValueError('No emitting plants in the nine-county Bay Area.')
    plants.to_csv(folder / 'plant_emissions_used_for_concentration.csv', index=False)
    # This continuation uses the explicitly selected local compact matrix.
    sr = zarr.open_group(isrm_store, mode='r')
    mapped = _map_plants_to_source_cells(plants, sr)
    source_ids = np.asarray(sr['source_ids'][:]) if 'source_ids' in sr else np.arange(sr['PrimaryPM25'].shape[1])
    mapped['matrix_source_available'] = mapped.isrm_source_cell.isin(source_ids)
    mapped.to_csv(folder / 'selected_plant_source_coverage.csv', index=False)
    if not mapped.matrix_source_available.all():
        missing = mapped.loc[~mapped.matrix_source_available]
        summary = json.loads((folder / 'power_generation_summary.json').read_text())
        summary.update(power_isrm_completed=False, power_source_layer_index=source_layer_index,
            all_selected_sources_covered=False, missing_source_cells=sorted(missing.isrm_source_cell.unique().astype(int).tolist()),
            missing_plant_ids=missing.ORISPL.astype(int).tolist(), missing_plant_count=len(missing),
            missing_annual_emissions_kg={p: float(missing[f'{p}_kg'].sum()) for p in POLLUTANTS})
        save_json(folder / 'power_generation_summary.json', summary)
        raise ValueError(f"Selected SFBAY plants lack ISRM source cells {summary['missing_source_cells']}; "
                         f"plant IDs {summary['missing_plant_ids']}. See selected_plant_source_coverage.csv.")
    receptors = np.sort(grid.inmap_cell_id.unique().astype(np.int64))
    summary = json.loads((folder / 'power_generation_summary.json').read_text())
    charging_qa_path = folder / 'beam_charging_summary.json'
    charging_qa = json.loads(charging_qa_path.read_text(encoding='utf-8')) if charging_qa_path.exists() else {}
    incomplete = charging_qa.get('ready_for_full_electrification_pipeline') is False
    summary.update({'charging_scenario_scope': charging_qa.get('curve_scope', 'hourly charging scenario'),
                    'incomplete_charging_scenario': incomplete,
                    'unmet_truck_energy_excluded': bool(charging_qa.get('daily_truck_unmet_MWh', 0) > 0),
                    'daily_truck_unmet_MWh': charging_qa.get('daily_truck_unmet_MWh', 0.),
                    'infeasible_truck_vehicles': charging_qa.get('infeasible_truck_vehicles', 0)})
    result = compute_power_generation_isrm(plants, isrm_url=isrm_store,
                                           receptor_cells=receptors, source_layer_index=source_layer_index)
    summary.update({
        'concentration_method': summary.get('allocation_method', 'hourly_allocation') + '_annual_emissions_times_ISRM',
        'new_hourly_dispatch_used_for_concentration': summary.get('allocation_method') != 'historical_generation_share',
        'concentration_source_scope': 'Emitting plants located inside the nine SFBAY counties only',
        'concentration_source_counties': str(counties_path),
        'outside_generation_reallocated': False,
        'selected_emitting_plants': len(plants),
        'selected_source_cells': int(mapped.isrm_source_cell.nunique()),
        'all_selected_sources_covered': True,
        'selected_annual_generation_MWh': float(plants.added_generation_MWh.sum()),
        'selected_annual_emissions_kg': {p: float(plants[f'{p}_kg'].sum()) for p in POLLUTANTS},
        'excluded_annual_emissions_kg': {p: float(all_plants[f'{p}_kg'].sum() - plants[f'{p}_kg'].sum()) for p in POLLUTANTS},
        'concentration_limitation': ('Layer 0 is a ground-release approximation. ' if source_layer_index == 0 else '') + 'Outside-Bay plants and unallocated generation are not represented.',
    })
    geometry = grid[['inmap_cell_id', 'geometry']].drop_duplicates('inmap_cell_id')
    result = gpd.GeoDataFrame(geometry.merge(result, on='inmap_cell_id', validate='one_to_one'),
                              geometry='geometry', crs=grid.crs)
    result.to_parquet(folder / 'power_generation_isrm_concentrations.parquet', index=False)
    result.to_file(folder / 'power_generation_isrm_concentrations.gpkg', driver='GPKG')
    plot_power_generation_concentration(result, plants, folder / 'power_generation_isrm_total_pm25.png',
                                       source_layer_index=source_layer_index,
                                       method_note=('SFBAY plants; supplied charging load only (unmet energy excluded)' if incomplete
                                                    else f"SFBAY plants; {summary.get('electricity_year', 'unspecified')} electricity; {summary.get('allocation_method', 'hourly allocation')}"))
    summary.update({'power_isrm_source': isrm_store, 'power_isrm_completed': True,
                    'power_source_layer_index': source_layer_index,
                    'power_local_local_removal_applied': False})
    save_json(folder / 'power_generation_summary.json', summary)
    return folder / 'power_generation_isrm_concentrations.parquet'
