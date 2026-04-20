from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.pipeline.workflow.step1_process_emissions import _derive_county_correction_factors
from impacts.pipeline.workflow.step1_process_emissions import _derive_inventory_activity_targets_for_assignment


def test_inventory_targets_per_assignment_dedup_trips(tmp_path: Path) -> None:
    passenger_inventory = pd.DataFrame(
        [
            {
                "county": "Alameda",
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": 2018,
                "speed": 10,
                "total_vmt_vehicle_miles_per_year": 100.0,
                "trips_per_year": 20.0,
            },
            {
                "county": "Alameda",
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": 2018,
                "speed": 20,
                "total_vmt_vehicle_miles_per_year": 50.0,
                "trips_per_year": 20.0,
            },
        ]
    )
    passenger_inventory_path = tmp_path / "passenger_inventory.parquet"
    passenger_inventory.to_parquet(passenger_inventory_path, index=False)

    freight_inventory = pd.DataFrame(
        [
            {
                "county": "Alameda",
                "vehicleCategory": "T7 Tractor Class 8",
                "fuel": "Dsl",
                "modelYear": 2018,
                "speed": 55,
                "total_vmt_vehicle_miles_per_year": 40.0,
                "trips_per_year": 4.0,
            },
        ]
    )
    freight_inventory_path = tmp_path / "freight_inventory.parquet"
    freight_inventory.to_parquet(freight_inventory_path, index=False)

    passenger_targets = _derive_inventory_activity_targets_for_assignment(
        inventory_path=str(passenger_inventory_path),
        county_name_lookup={"Alameda": "001"},
        assignment_group="passenger",
    )
    freight_targets = _derive_inventory_activity_targets_for_assignment(
        inventory_path=str(freight_inventory_path),
        county_name_lookup={"Alameda": "001"},
        assignment_group="freight",
    )
    targets = (
        pd.concat([passenger_targets, freight_targets], ignore_index=True)
        .sort_values(["countyfp", "assignment_group"])
        .reset_index(drop=True)
    )

    assert targets.to_dict("records") == [
        {"countyfp": "001", "assignment_group": "freight", "totVMT": 40.0, "totTrips": 4.0},
        {"countyfp": "001", "assignment_group": "passenger", "totVMT": 150.0, "totTrips": 20.0},
    ]

    beam_group_totals = pd.DataFrame(
        [
            {"countyfp": "001", "assignment_group": "passenger", "totVMT": 90.0, "totTrips": 9.0},
            {"countyfp": "001", "assignment_group": "freight", "totVMT": 30.0, "totTrips": 3.0},
        ]
    )
    factors = _derive_county_correction_factors(beam_group_totals, targets).sort_values(
        ["countyfp", "assignment_group"]
    ).reset_index(drop=True)

    assert factors[["countyfp", "assignment_group", "factor_totVMT", "factor_totTrips"]].to_dict("records") == [
        {
            "countyfp": "001",
            "assignment_group": "freight",
            "factor_totVMT": 40.0 / 30.0,
            "factor_totTrips": 4.0 / 3.0,
        },
        {
            "countyfp": "001",
            "assignment_group": "passenger",
            "factor_totVMT": 150.0 / 90.0,
            "factor_totTrips": 20.0 / 9.0,
        },
    ]
