import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from compare_native_evi_15k_weekday import normalized_curves, weekday_hourly


def test_pev_normalization_common_fleet():
    result = normalized_curves(np.ones(24), 15440, np.ones(24)*30000, 30000)
    np.testing.assert_allclose(result.events_MW_at_15000_PEV, 15000/15440)
    np.testing.assert_allclose(result.events_kW_per_1000_PEV, 1e6/15440)
    np.testing.assert_allclose(result.EVI_weekday_kW_per_1000_PEV, 1000)
    np.testing.assert_allclose(result.EVI_weekday_MW_at_15000_PEV, 15)


def test_average_quarter_hour_power_not_sum():
    parts = {k:list(range(96)) for k in ('home_l1','home_l2','work_l1','work_l2','public_l2','public_l3')}
    total, _ = weekday_hourly({'results': {'weekday_load_profile': parts}})
    assert total[0] == 9
    assert total.sum() == sum(range(96))*6/4
    parts['home_l1'] = None
    with pytest.raises(ValueError):
        weekday_hourly({'results': {'weekday_load_profile': parts}})
