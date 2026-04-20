from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.pipeline.workflow.step1_process_emissions import _derive_county_correction_factors
from impacts.pipeline.workflow.step1_process_emissions import _derive_inventory_activity_targets


def test_inventory_targets_split_shared_categories_and_dedup_trips(tmp_path: Path) -> None:
    inventory = pd.DataFrame(
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
    inventory_path = tmp_path / "inventory.parquet"
    inventory.to_parquet(inventory_path, index=False)

    beam_activity_details = pd.DataFrame(
        [
            {
                "countyfp": "001",
                "assignment_group": "passenger",
                "emfacVehicleCategory": "LDA",
                "totVMT": 30.0,
                "totTrips": 9.0,
            },
            {
                "countyfp": "001",
                "assignment_group": "freight",
                "emfacVehicleCategory": "LDA",
                "totVMT": 10.0,
                "totTrips": 1.0,
            },
            {
                "countyfp": "001",
                "assignment_group": "freight",
                "emfacVehicleCategory": "T7 Tractor Class 8",
                "totVMT": 20.0,
                "totTrips": 2.0,
            },
        ]
    )

    targets = _derive_inventory_activity_targets(
        inventory_path=str(inventory_path),
        county_name_lookup={"Alameda": "001"},
        beam_activity_details=beam_activity_details,
    ).sort_values(["countyfp", "assignment_group"]).reset_index(drop=True)

    assert targets.to_dict("records") == [
        {"countyfp": "001", "assignment_group": "freight", "totVMT": 77.5, "totTrips": 6.0},
        {"countyfp": "001", "assignment_group": "passenger", "totVMT": 112.5, "totTrips": 18.0},
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
            "factor_totVMT": 77.5 / 30.0,
            "factor_totTrips": 2.0,
        },
        {
            "countyfp": "001",
            "assignment_group": "passenger",
            "factor_totVMT": 112.5 / 90.0,
            "factor_totTrips": 2.0,
        },
    ]
