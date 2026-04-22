from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.pipeline.workflow.step1_process_emissions import _derive_county_correction_factors
from impacts.pipeline.workflow.step1_process_emissions import _derive_inventory_activity_targets_for_assignment
from impacts.pipeline.workflow.step1_process_emissions import apply_county_corrections


def test_inventory_targets_per_assignment_grouped_by_model_year_and_process(tmp_path: Path) -> None:
    passenger_inventory = pd.DataFrame(
        [
            {
                "county": "Alameda",
                "modelYear": "post2014",
                "process": "RUNEX",
                "total_vmt_vehicle_miles_per_year": 100.0,
                "trips_per_year": 20.0,
            },
            {
                "county": "Alameda",
                "modelYear": "post2014",
                "process": "RUNEX",
                "total_vmt_vehicle_miles_per_year": 50.0,
                "trips_per_year": 5.0,
            },
            {
                "county": "Alameda",
                "modelYear": "pre2004",
                "process": "STREX",
                "total_vmt_vehicle_miles_per_year": 0.0,
                "trips_per_year": 7.0,
            },
        ]
    )
    passenger_inventory_path = tmp_path / "passenger_inventory.parquet"
    passenger_inventory.to_parquet(passenger_inventory_path, index=False)

    freight_inventory = pd.DataFrame(
        [
            {
                "county": "Alameda",
                "modelYear": "2007to2009",
                "process": "PRDUST",
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
        .sort_values(["assignment_group", "modelYear", "process"])
        .reset_index(drop=True)
    )

    assert targets.to_dict("records") == [
        {
            "countyfp": "001",
            "assignment_group": "freight",
            "modelYear": "2007to2009",
            "process": "PRDUST",
            "totVMT": 40.0,
            "totTrips": 4.0,
        },
        {
            "countyfp": "001",
            "assignment_group": "passenger",
            "modelYear": "post2014",
            "process": "RUNEX",
            "totVMT": 150.0,
            "totTrips": 25.0,
        },
        {
            "countyfp": "001",
            "assignment_group": "passenger",
            "modelYear": "pre2004",
            "process": "STREX",
            "totVMT": 0.0,
            "totTrips": 7.0,
        },
    ]

    beam_group_totals = pd.DataFrame(
        [
            {
                "countyfp": "001",
                "assignment_group": "passenger",
                "modelYear": "post2014",
                "process": "RUNEX",
                "totVMT": 90.0,
                "totTrips": 9.0,
            },
            {
                "countyfp": "001",
                "assignment_group": "passenger",
                "modelYear": "pre2004",
                "process": "STREX",
                "totVMT": 0.0,
                "totTrips": 3.5,
            },
            {
                "countyfp": "001",
                "assignment_group": "freight",
                "modelYear": "2007to2009",
                "process": "PRDUST",
                "totVMT": 30.0,
                "totTrips": 3.0,
            },
        ]
    )
    factors = _derive_county_correction_factors(beam_group_totals, targets).sort_values(
        ["assignment_group", "modelYear", "process"]
    ).reset_index(drop=True)

    assert factors[
        ["countyfp", "assignment_group", "modelYear", "process", "factor_totVMT", "factor_totTrips"]
    ].to_dict("records") == [
        {
            "countyfp": "001",
            "assignment_group": "freight",
            "modelYear": "2007to2009",
            "process": "PRDUST",
            "factor_totVMT": 40.0 / 30.0,
            "factor_totTrips": 4.0 / 3.0,
        },
        {
            "countyfp": "001",
            "assignment_group": "passenger",
            "modelYear": "post2014",
            "process": "RUNEX",
            "factor_totVMT": 150.0 / 90.0,
            "factor_totTrips": 25.0 / 9.0,
        },
        {
            "countyfp": "001",
            "assignment_group": "passenger",
            "modelYear": "pre2004",
            "process": "STREX",
            "factor_totVMT": 1.0,
            "factor_totTrips": 2.0,
        },
    ]


def test_apply_county_corrections_leaves_transit_rows_neutral(tmp_path: Path) -> None:
    passenger_vehicle_types = pd.DataFrame(
        [
            {"vehicleTypeId": "pax-car", "vehicleCategory": "Car", "emfacId": "post2014LDAGas"},
            {"vehicleTypeId": "BUS-DEFAULT", "vehicleCategory": "MediumDutyPassenger", "emfacVehicleCategory": "UBUS", "emfacId": "2013to2015UBUSDsl"},
            {"vehicleTypeId": "RAIL-DEFAULT", "vehicleCategory": "Rail-Default", "emfacId": ""},
        ]
    )
    freight_vehicle_types = pd.DataFrame(
        [
            {"vehicleTypeId": "ft-md", "vehicleCategory": "Class456Vocational", "vehicleClass": "truck", "vehicleUse": "freight", "emfacId": "2007to2009Class4Dsl"},
        ]
    )
    passenger_vehicle_types_path = tmp_path / "vehicleTypes--atlas.csv"
    freight_vehicle_types_path = tmp_path / "vehicleTypes--frism.csv"
    passenger_vehicle_types.to_csv(passenger_vehicle_types_path, index=False)
    freight_vehicle_types.to_csv(freight_vehicle_types_path, index=False)

    allocated = pd.DataFrame(
        [
            {"vehicleTypeId": "pax-car", "countyfp": "001", "process": "RUNEX", "tons_per_year_NOx_county_allocated": 10.0},
            {"vehicleTypeId": "BUS-DEFAULT", "countyfp": "001", "process": "RUNEX", "tons_per_year_NOx_county_allocated": 20.0},
            {"vehicleTypeId": "RAIL-DEFAULT", "countyfp": "001", "process": "RUNEX", "tons_per_year_NOx_county_allocated": 30.0},
        ]
    )
    factors = pd.DataFrame(
        [
            {
                "countyfp": "001",
                "assignment_group": "passenger",
                "modelYear": "post2014",
                "process": "RUNEX",
                "factor_totVMT": 2.0,
                "factor_totTrips": 3.0,
            },
        ]
    )

    corrected = apply_county_corrections(
        allocated,
        factors,
        county_col="countyfp",
        passenger_vehicle_types_path=str(passenger_vehicle_types_path),
        freight_vehicle_types_path=str(freight_vehicle_types_path),
    )

    assert corrected["tons_per_year_NOx_county_allocated"].tolist() == [20.0, 20.0, 30.0]
