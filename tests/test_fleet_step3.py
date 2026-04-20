from __future__ import annotations

import pandas as pd

from impacts.emfac.fleet.step3_map_emfac_atlas import _prepare_mapped_passenger_vehicles_output


def test_prepare_mapped_passenger_vehicles_output_writes_required_columns() -> None:
    vehicles = pd.DataFrame(
        [
            {
                "household_id": "353",
                "vehicle_id": "2",
                "vehicleTypeId": "type-1",
                "stateOfCharge": "0.75",
                "bodytype": "Sedan",
            }
        ]
    )

    result = _prepare_mapped_passenger_vehicles_output(vehicles)

    assert list(result.columns) == [
        "household_id",
        "vehicle_id",
        "householdId",
        "vehicleId",
        "vehicleTypeId",
        "initialSoc",
    ]
    assert result.loc[0, "household_id"] == "353"
    assert result.loc[0, "vehicle_id"] == "2"
    assert result.loc[0, "householdId"] == "353"
    assert result.loc[0, "vehicleId"] == "353-2"
    assert result.loc[0, "vehicleTypeId"] == "type-1"
    assert pd.isna(result.loc[0, "initialSoc"])


def test_prepare_mapped_passenger_vehicles_output_normalizes_beam_alias_ids() -> None:
    vehicles = pd.DataFrame(
        [
            {
                "household_id": "4227970.0",
                "vehicle_id": "2.0",
                "vehicleTypeId": "type-2",
            },
            {
                "household_id": "1e+05",
                "vehicle_id": "3",
                "vehicleTypeId": "type-3",
            },
        ]
    )

    result = _prepare_mapped_passenger_vehicles_output(vehicles)

    assert result.loc[0, "household_id"] == "4227970.0"
    assert result.loc[0, "vehicle_id"] == "2.0"
    assert result.loc[0, "householdId"] == "4227970"
    assert result.loc[0, "vehicleId"] == "4227970-2"
    assert result.loc[1, "householdId"] == "100000"
    assert result.loc[1, "vehicleId"] == "100000-3"
