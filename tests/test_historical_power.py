"""Energy accounting and year isolation for the historical electricity option."""
import numpy as np
import pandas as pd
import pytest

from impacts.config.settings import PowerGeneration
from impacts.pipeline.workflow import historical_power as hp
from impacts.pipeline.workflow.hourly_power import POLLUTANTS


def test_average_mix_retains_nonemitting_generation_and_hourly_shape():
    plants = pd.DataFrame({'ORISPL': [1, 2], 'allocation_weight': [.25, .75],
        **{f'{p}_rate_kg_per_MWh': [2., 0.] for p in POLLUTANTS}})
    demand = pd.DataFrame({'hour_of_year': [0, 1], 'charging_demand_MWh': [9., 18.]})
    hourly, annual = hp.allocate_hourly(demand, plants, .1)
    np.testing.assert_allclose(hourly.groupby('hour_of_year').added_generation_MWh.sum(), [10, 20])
    np.testing.assert_allclose(annual.added_generation_MWh, [7.5, 22.5])
    assert annual.NOx_kg.sum() == 15
    assert hourly.loc[hourly.ORISPL.eq(2), 'NOx_kg'].eq(0).all()
    # Under fixed historical shares, moving energy in time does not change annual emissions.
    _, shifted = hp.allocate_hourly(demand.assign(charging_demand_MWh=[18., 9.]), plants, .1)
    np.testing.assert_allclose(shifted.NOx_kg, annual.NOx_kg)


def test_unknown_emissions_are_not_silently_zeroed():
    plants = pd.DataFrame({'ORISPL': [1, 2], 'PLPRMFL': ['SUN', 'NG'], 'SUBRGN': ['X', 'X']})
    kwargs = dict(pollutant='NOx', year=2018, fallback=pd.Series({'X': 2.}))
    with pytest.raises(ValueError, match='Missing or invalid NOx'):
        hp.complete_rate(plants, pd.Series([np.nan, np.nan]), policy='error', **kwargs)
    rate, basis = hp.complete_rate(plants, pd.Series([np.nan, np.nan]), policy='subregion', **kwargs)
    np.testing.assert_allclose(rate, [0., 2 * hp.LB_TO_KG])
    assert basis.iloc[1] == 'eGRID_2018_subregion_proxy'


def test_workbook_release_name_cannot_override_inventory_year(monkeypatch):
    monkeypatch.setattr(pd, 'read_excel', lambda *a, **k: pd.DataFrame({'YEAR': [2021]}))
    with pytest.raises(ValueError, match='only year 2018'):
        hp.read_year_sheet('egrid2021-draft-nh3-emissions.xlsx', '2018 NH3 Plant-level Data', 2018)


def test_historical_settings_reject_mismatched_egrid():
    with pytest.raises(ValueError, match='years to agree'):
        PowerGeneration.from_dict(dict(electricity_year=2018, egrid_year=2023))


def test_cambium_cannot_be_relabelled_as_2018():
    with pytest.raises(ValueError, match='does not supply'):
        PowerGeneration.from_dict(dict(enabled=True, allocation_method='cambium_marginal_technology',
                                       electricity_year=2018, egrid_year=2023))


def test_native_default_uses_historical_2018():
    settings = PowerGeneration.from_dict({})
    assert settings.allocation_method == 'historical_generation_share'
    assert settings.electricity_year == 2018
    assert settings.missing_rate_policy == 'error'


def test_orchestrator_routes_historical_inputs_to_isrm(tmp_path, monkeypatch):
    from impacts.pipeline.workflow import hourly_power
    paths = {name: tmp_path / name for name in ('demand', 'summary', 'grid', 'counties')}
    for p in paths.values():
        p.write_text('{}')
    calls = []
    def allocate(output, **kwargs):
        assert kwargs['electricity_year'] == 2018
        assert kwargs['balancing_area'] == 'p9'
        calls.append('historical')
    monkeypatch.setattr(hp, 'dispatch_historical', allocate)
    def concentrate(output, **kwargs):
        assert kwargs['source_layer_index'] == 0
        calls.append('isrm')
        return output / 'result.parquet'
    monkeypatch.setattr(hourly_power, 'compute_hourly_power_isrm', concentrate)
    hourly_power.run_hourly_power(config=dict(enabled=True, hourly_demand_file=str(paths['demand']),
        charging_summary_file=str(paths['summary']), breathe_root=str(tmp_path),
        county_boundaries_file=str(paths['counties']), source_layer_index=0),
        isrm_url='local.zarr', inmap_grid_path=str(paths['grid']), output_dir=tmp_path / 'out')
    assert calls == ['historical', 'isrm']


def test_emissions_only_does_not_call_isrm(tmp_path, monkeypatch):
    from impacts.pipeline.workflow import hourly_power
    for name in ['demand', 'summary', 'grid', 'counties']:
        (tmp_path / name).write_text('{}')
    def allocate(output, **kwargs):
        (output / 'power_generation_summary.json').write_text('{}')
    monkeypatch.setattr(hp, 'dispatch_historical', allocate)
    def forbidden(*args, **kwargs):
        pytest.fail('ISRM must not run in emissions-only mode')
    monkeypatch.setattr(hourly_power, 'compute_hourly_power_isrm', forbidden)
    result = hourly_power.run_hourly_power(config=dict(enabled=True,
        hourly_demand_file=str(tmp_path/'demand'), charging_summary_file=str(tmp_path/'summary'),
        breathe_root=str(tmp_path), county_boundaries_file=str(tmp_path/'counties'), source_layer_index=0),
        isrm_url='local.zarr', inmap_grid_path=str(tmp_path/'grid'), output_dir=tmp_path/'out',
        compute_concentrations=False)
    assert result.name == 'plant_emissions_approx.csv'
