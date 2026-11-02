"""Unit/time and fleet normalization checks, without network requests."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'compare_sf_evi', Path(__file__).parents[1]/'scripts/compare_sf_evi_410k.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def payload():
    return {'results': {day+'_load_profile': {str(i): list(np.arange(96)+i) for i in range(6)}
                        for day in ['weekday', 'weekend']}}


def test_quarter_hour_power_is_averaged_and_energy_conserved():
    data = payload()
    hourly = comparison.hourly_kw(data)
    quarter_hour_power = sum(np.array(v) for v in data['results']['weekday_load_profile'].values())
    assert len(hourly['weekday']) == 24
    assert hourly['weekday'][0] == np.mean(quarter_hour_power[:4])
    assert hourly['weekday'].sum() == quarter_hour_power.sum()*.25


def test_missing_component_is_not_treated_as_zero():
    data = payload()
    data['results']['weekday_load_profile']['0'] = None
    with pytest.raises(ValueError, match='six nonnegative'):
        comparison.hourly_kw(data)


def test_per_thousand_and_410000_units():
    per_thousand_kw = np.full(24, 1000.)
    result = comparison.describe('test', per_thousand_kw)
    assert result['daily_kWh_per_vehicle'] == 24.
    assert result['peak_MW_410000'] == 410.
    assert result['daily_MWh_410000'] == 9840.
