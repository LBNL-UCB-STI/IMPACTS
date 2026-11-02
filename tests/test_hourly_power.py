"""Contracts for the single supported hourly power workflow."""
import json

import numpy as np
import numpy.testing  # Initialize before mocking the allocator subprocess.
import pandas as pd
import pytest

from impacts.config.settings import PowerGeneration
from impacts.pipeline.workflow import hourly_power as hp


@pytest.mark.parametrize('key', ['charging_events_file', 'precomputed_power_isrm_file',
                                 'baseline_fleet_size', 'plant_emissions_file'])
def test_annual_scaling_configuration_is_rejected(key):
    with pytest.raises(ValueError, match='annual-scaling power method has been removed'):
        PowerGeneration.from_dict({'enabled': False, key: 'old-input'})


def test_enabled_power_requires_explicit_inputs_and_layer():
    with pytest.raises(ValueError, match='source_layer_index'):
        PowerGeneration.from_dict({'enabled': True})


@pytest.mark.parametrize('problem', ['duplicate', 'missing', 'negative', 'nan'])
def test_invalid_hourly_demand_is_rejected(tmp_path, problem):
    df = pd.DataFrame({'hour_of_year': np.arange(8760), 'charging_demand_MWh': 1.0})
    if problem == 'duplicate':
        df.loc[1, 'hour_of_year'] = 0
    elif problem == 'missing':
        df = df.iloc[:-1]
    else:
        df.loc[0, 'charging_demand_MWh'] = -1 if problem == 'negative' else np.nan
    path = tmp_path / 'demand.parquet'
    df.to_parquet(path)
    with pytest.raises(ValueError):
        hp.load_hourly_demand(path)


def test_native_orchestrator_uses_hourly_input_without_scaling(tmp_path, monkeypatch):
    demand = tmp_path / 'demand.parquet'
    df = pd.DataFrame({'hour_of_year': np.arange(8760), 'charging_demand_MWh': 2.0})
    df.to_parquet(demand)
    summary = tmp_path / 'summary.json'
    summary.write_text('{}')
    grid = tmp_path / 'grid.parquet'
    counties = tmp_path / 'counties.gpkg'
    grid.touch()
    counties.touch()
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/allocate_power_plants_approx.py').touch()
    config = dict(enabled=True, hourly_demand_file=str(demand), charging_summary_file=str(summary),
                  allocation_method='cambium_marginal_technology', electricity_year=2040, egrid_year=2023,
                  breathe_root=str(tmp_path), county_boundaries_file=str(counties), source_layer_index=0)
    calls = []
    def dispatch(output, **kwargs):
        pd.testing.assert_frame_equal(hp.load_hourly_demand(kwargs['hourly_demand_file']), df)
        calls.append('dispatch')
    def isrm(output, **kwargs):
        assert kwargs['source_layer_index'] == 0
        assert kwargs['isrm_store'] == 'selected.zarr'
        assert 'emissions_scale_factor' not in kwargs
        calls.append('isrm')
        return output / 'result.parquet'
    monkeypatch.setattr(hp, 'dispatch_hourly', dispatch)
    monkeypatch.setattr(hp, 'compute_hourly_power_isrm', isrm)
    result = hp.run_hourly_power(config=config, isrm_url='selected.zarr',
                                inmap_grid_path=str(grid), output_dir=tmp_path / 'out')
    assert result.name == 'result.parquet'
    assert calls == ['dispatch', 'isrm']
    summary.write_text(json.dumps({'ready_for_full_electrification_pipeline': False}))
    with pytest.raises(ValueError, match='Unmet charging energy'):
        hp.run_hourly_power(config=config, isrm_url='selected.zarr',
                            inmap_grid_path=str(grid), output_dir=tmp_path / 'out')
    assert calls == ['dispatch', 'isrm']


def test_dispatch_depends_on_hourly_shape_not_annual_energy(tmp_path, monkeypatch):
    """Moving equal energy from coal to gas hours must change annual emissions."""
    cambium_dir = tmp_path / 'tmp/cambium_check/hourly_balancingArea'
    cambium_dir.mkdir(parents=True)
    tech = np.where(np.arange(8760) % 2 == 0, 'coal', 'gas')
    cambium = pd.DataFrame(dict(timestamp_local=pd.date_range('2040-01-01', periods=8760, freq='h'),
                               marg_gen_tech=tech, distloss_rate_marg=0.0))
    path = cambium_dir / 'Cambium22_LowRECost_hourly_p9_2040.csv'
    path.write_text('\n' * 5 + cambium.to_csv(index=False))
    summary = tmp_path / 'summary.json'
    summary.write_text('{}')

    def allocator(command, check):
        # Fixture: one coal plant (2 kg/MWh), one gas plant (1 kg/MWh).
        load = pd.read_csv(command[command.index('--load-csv') + 1], header=None).iloc[0, :8760].to_numpy()
        output = hp.Path(command[command.index('--output-dir') + 1])
        output.mkdir(parents=True)
        generators, annual = [], []
        for plant, technology, factor in [(1, 'coal', 2.0), (2, 'gas', 1.0)]:
            energy = float(load[tech == technology].sum())
            generators.append(dict(ORISPL=plant, marginal_technology=technology, allocation_weight=1.0,
                **{f'{p}_rate_kg_per_MWh': factor for p in hp.POLLUTANTS}))
            annual.append(dict(ORISPL=plant, added_generation_MWh=energy,
                **{f'{p}_kg': energy * factor for p in hp.POLLUTANTS}))
        pd.DataFrame(generators).to_csv(output / 'generator_allocation_detail.csv', index=False)
        pd.DataFrame(annual).to_csv(output / 'plant_emissions_approx.csv', index=False)
        (output / 'qa_summary.json').write_text('{}')
    monkeypatch.setattr(hp.subprocess, 'run', allocator)
    emissions = []
    for hour in (0, 1):
        load = np.zeros(8760)
        load[hour] = 10
        demand = tmp_path / f'demand{hour}.parquet'
        pd.DataFrame(dict(hour_of_year=np.arange(8760), charging_demand_MWh=load)).to_parquet(demand)
        output = tmp_path / f'result{hour}'
        hp.dispatch_hourly(output, hourly_demand_file=demand, charging_summary_file=summary, breathe_root=tmp_path,
                           electricity_year=2040, balancing_area='p9', cambium_scenario='LowRECost')
        annual = pd.read_csv(output / 'plant_allocation/plant_emissions_approx.csv')
        assert annual.added_generation_MWh.sum() == 10
        emissions.append(annual.NOx_kg.sum())
    assert emissions == [20.0, 10.0]
