from __future__ import annotations

import pandas as pd
import pytest

from impacts.emfac.activities.step1_prepare_emissions_and_activities_tables import _complete_sparse_inventory_counties


def _base_complete_county_frame() -> list[dict[str, object]]:
    return [
        {"county": "Alameda", "vehicleCategory": "LDA", "fuel": "Gas", "modelYear": 2018},
        {"county": "Marin", "vehicleCategory": "LDA", "fuel": "Gas", "modelYear": 2018},
        {"county": "Sonoma", "vehicleCategory": "LDA", "fuel": "Gas", "modelYear": 2018},
    ]


@pytest.mark.parametrize(
    ("source_type", "extra_keys", "value_columns", "expected_values"),
    [
        ("population-inventory", {}, ["population"], {"population": 12.0}),
        ("trips-inventory", {}, ["trips"], {"trips": 12.0}),
        ("vmt-inventory", {"speed": 30}, ["total_vmt", "cvmt", "evmt"], {"total_vmt": 12.0, "cvmt": 6.0, "evmt": 2.0}),
        (
            "emission-inventory",
            {"speed": 30, "process": "RUNEX", "pollutant": "NOx"},
            ["emission"],
            {"emission": 12.0},
        ),
        (
            "ghg-inventory",
            {"speed": 30, "process": "RUNEX", "pollutant": "CO2e"},
            ["emission"],
            {"emission": 12.0},
        ),
    ],
)
def test_complete_sparse_inventory_counties_imputes_missing_sf_counties(source_type: str, extra_keys: dict[str, object], value_columns: list[str], expected_values: dict[str, float]) -> None:
    frame = pd.DataFrame(
        [
            {
                "county": "Marin",
                "vehicleCategory": "LDT1",
                "fuel": "Dsl",
                "modelYear": 2018,
                **extra_keys,
                **dict(zip(value_columns, [10.0, 5.0, 1.0])),
            },
            {
                "county": "Sonoma",
                "vehicleCategory": "LDT1",
                "fuel": "Dsl",
                "modelYear": 2018,
                **extra_keys,
                **dict(zip(value_columns, [14.0, 7.0, 3.0])),
            },
            *[
                {
                    **row,
                    **extra_keys,
                    **dict(zip(value_columns, [20.0, 10.0, 4.0])),
                }
                for row in _base_complete_county_frame()
            ],
        ]
    )

    result = _complete_sparse_inventory_counties(frame, source_type)
    sparse_slice = result.loc[
        (result["vehicleCategory"] == "LDT1")
        & (result["fuel"] == "Dsl")
        & (result["modelYear"] == 2018)
    ].sort_values("county")
    for key, value in extra_keys.items():
        sparse_slice = sparse_slice.loc[sparse_slice[key] == value]

    assert sparse_slice["county"].tolist() == ["Alameda", "Marin", "Sonoma"]
    for column, expected in expected_values.items():
        assert sparse_slice[column].tolist() == [expected, sparse_slice.iloc[1][column], sparse_slice.iloc[2][column]]
    assert sparse_slice["is_imputed_county"].tolist() == [True, False, False]
