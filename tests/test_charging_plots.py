import numpy as np
import pandas as pd
import pytest

from impacts.pipeline.workflow.charging_plots import prepare_separate_curves, group_passenger_and_truck_curves


def demand():
    df = pd.DataFrame({'hour_of_year': np.arange(48)})
    for i in range(12):
        df[f'charging_demand_type{i}_MWh'] = (np.arange(48) % 24) * (i + 1) / 100.
    df['charging_demand_MWh'] = df.filter(like='charging_demand_').sum(axis=1)
    return df


def test_all_types_retained_and_other_and_total_exact():
    data = demand()
    original = data.copy(deep=True)
    daily, ranked, displayed = prepare_separate_curves(data)
    assert len(ranked) == 12 and len(displayed) == 10
    assert ranked[0] == 'charging_demand_type11_MWh'
    np.testing.assert_allclose(daily.Other, daily[ranked[8:]].sum(axis=1))
    np.testing.assert_allclose(daily.charging_demand_MWh, daily[ranked[:8]].sum(axis=1) + daily.Other)
    pd.testing.assert_frame_equal(data, original)


def test_incorrect_total_rejected():
    data = demand()
    data.loc[0, 'charging_demand_MWh'] = 10
    with pytest.raises(AssertionError):
        prepare_separate_curves(data)


def test_incomplete_day_rejected():
    with pytest.raises(ValueError, match='24'):
        prepare_separate_curves(demand().iloc[:20])


def test_four_groups_combine_all_powertrains_and_exclude_other_modes():
    types = ['cv_L1_a', 'ev_L1_a', 'hev_L1_a', 'phev_L1_a', 'ft_ld1_Gas',
             'ft_ld3_Dsl', 'ft_mdv_Gas', 'ft_mdv_Dsl', 'ft_hdv_Dsl', 'ft_hdt_Dsl',
             'BUS_DEFAULT', 'BIKE_DEFAULT', 'FERRY_DEFAULT']
    data = pd.DataFrame({'hour_of_year': np.arange(24)})
    for token in types:
        data[f'charging_demand_{token}_MWh'] = 1.
    data['charging_demand_MWh'] = len(types)
    original = data.copy(deep=True)
    grouped, mapping, excluded = group_passenger_and_truck_curves(data)
    assert list(grouped) == ['Passenger vehicles', 'LDT', 'MDT', 'HDT']
    np.testing.assert_array_equal(grouped.iloc[0], [4., 2., 2., 2.])
    assert len(excluded) == 3
    assert len(mapping['Passenger vehicles']) == 4
    pd.testing.assert_frame_equal(data, original)
