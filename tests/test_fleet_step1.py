from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.pipeline.emfac._common import read_atlas_vehicles_input


def test_read_atlas_vehicles_input_accepts_raw_snake_case_schema(tmp_path: Path) -> None:
    vehicles_file = tmp_path / "vehicles.parquet"
    pd.DataFrame(
        [
            {
                "household_id": 10,
                "vehicle_id": 2,
                "bodytype": "Car",
                "modelyear": 2019,
                "adopt_fuel": "gasoline",
            }
        ]
    ).to_parquet(vehicles_file, index=False)

    loaded = read_atlas_vehicles_input(str(vehicles_file))

    assert loaded.loc[0, "household_id"] == 10
    assert loaded.loc[0, "vehicle_id"] == 2
    assert loaded.loc[0, "bodytype"] == "Car"
    assert loaded.loc[0, "modelyear"] == 2019
    assert loaded.loc[0, "adopt_fuel"] == "gasoline"


def test_read_atlas_vehicles_input_accepts_processed_camel_case_schema(tmp_path: Path) -> None:
    vehicles_file = tmp_path / "vehicles.parquet"
    pd.DataFrame(
        [
            {
                "householdId": 10,
                "vehicleId": "10-2",
                "sourceVehicleId": 2,
                "bodytype": "Car",
                "modelyear": 2019,
                "adopt_fuel": "gasoline",
                "vehicleTypeId": "paxcar-abc123",
            }
        ]
    ).to_parquet(vehicles_file, index=False)

    loaded = read_atlas_vehicles_input(str(vehicles_file))

    assert loaded.loc[0, "household_id"] == 10
    assert loaded.loc[0, "vehicle_id"] == 2
    assert loaded.loc[0, "bodytype"] == "Car"
    assert loaded.loc[0, "modelyear"] == 2019
    assert loaded.loc[0, "adopt_fuel"] == "gasoline"
    assert loaded.loc[0, "vehicleTypeId"] == "paxcar-abc123"
