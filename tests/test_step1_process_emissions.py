from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from impacts.pipeline.workflow.step1_process_emissions import _derive_county_correction_factors
from impacts.pipeline.workflow.step1_process_emissions import _derive_inventory_activity_targets_for_assignment
from impacts.pipeline.workflow.step1_process_emissions import _build_county_corrected_table
from impacts.pipeline.workflow.step1_process_emissions import apply_county_corrections


def test_inventory_targets_per_assignment_grouped_by_model_year_and_process(tmp_path: Path) -> None:
    passenger_inventory = pd.DataFrame(
        [
            {
                "county": "Alameda",
                "modelYear": ">=2015",
                "process": "RUNEX",
                "total_vmt_vehicle_miles_per_year": 100.0,
                "trips_per_year": 20.0,
            },
            {
                "county": "Alameda",
                "modelYear": ">=2015",
                "process": "RUNEX",
                "total_vmt_vehicle_miles_per_year": 50.0,
                "trips_per_year": 5.0,
            },
            {
                "county": "Alameda",
                "modelYear": "<=2003",
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
                "modelYear": "2007-2009",
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
            "modelYear": "2007-2009",
            "process": "PRDUST",
            "totVMT": 40.0,
            "totTrips": 4.0,
        },
        {
            "countyfp": "001",
            "assignment_group": "passenger",
            "modelYear": "<=2003",
            "process": "STREX",
            "totVMT": 0.0,
            "totTrips": 7.0,
        },
        {
            "countyfp": "001",
            "assignment_group": "passenger",
            "modelYear": ">=2015",
            "process": "RUNEX",
            "totVMT": 150.0,
            "totTrips": 25.0,
        },
    ]

    beam_group_totals = pd.DataFrame(
        [
            {
                "countyfp": "001",
                "assignment_group": "passenger",
                "modelYear": ">=2015",
                "process": "RUNEX",
                "totVMT": 90.0,
                "totTrips": 9.0,
            },
            {
                "countyfp": "001",
                "assignment_group": "passenger",
                "modelYear": "<=2003",
                "process": "STREX",
                "totVMT": 0.0,
                "totTrips": 3.5,
            },
            {
                "countyfp": "001",
                "assignment_group": "freight",
                "modelYear": "2007-2009",
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
            "modelYear": "2007-2009",
            "process": "PRDUST",
            "factor_totVMT": 40.0 / 30.0,
            "factor_totTrips": 4.0 / 3.0,
        },
        {
            "countyfp": "001",
            "assignment_group": "passenger",
            "modelYear": "<=2003",
            "process": "STREX",
            "factor_totVMT": 1.0,
            "factor_totTrips": 2.0,
        },
        {
            "countyfp": "001",
            "assignment_group": "passenger",
            "modelYear": ">=2015",
            "process": "RUNEX",
            "factor_totVMT": 150.0 / 90.0,
            "factor_totTrips": 25.0 / 9.0,
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
        scratch_dir=tmp_path,
        passenger_vehicle_types_path=str(passenger_vehicle_types_path),
        freight_vehicle_types_path=str(freight_vehicle_types_path),
    )

    assert corrected["tons_per_year_NOx_county_allocated"].tolist() == [20.0, 20.0, 30.0]


def test_build_county_corrected_table_skips_when_inventory_corrections_disabled(monkeypatch, tmp_path: Path) -> None:
    county_allocated = pd.DataFrame(
        [
            {"linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "countyfp": "001", "tons_per_year_NOx_county_allocated": 10.0},
        ]
    )
    county_grouped = pd.DataFrame(
        [
            {"linkId": 101, "countyfp": "001", "county_proportion": 1.0},
        ]
    )
    skims = pd.DataFrame(
        [
            {"linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "totVMT": 1.0, "totTrips": 1.0},
        ]
    )
    pipeline = SimpleNamespace(
        enable_passenger_inventory_activity_correction=False,
        enable_freight_inventory_activity_correction=False,
        passenger_inventory_file="passenger.parquet",
        freight_inventory_file="freight.parquet",
    )

    def _fail(*args, **kwargs):
        raise AssertionError("inventory correction helpers should not run when both correction flags are false")

    monkeypatch.setattr(
        "impacts.pipeline.workflow.step1_process_emissions.resolve_required_manifest_input",
        _fail,
    )
    monkeypatch.setattr(
        "impacts.pipeline.workflow.step1_process_emissions._build_beam_activity_details",
        _fail,
    )

    corrected, beam_activity_totals, county_correction_factors = _build_county_corrected_table(
        county_allocated_df=county_allocated,
        county_grouped_df=county_grouped,
        skims_df=skims,
        pipeline=pipeline,
        manifest_inputs={},
        scratch_dir=tmp_path,
    )

    assert beam_activity_totals is None
    assert county_correction_factors is None
    pd.testing.assert_frame_equal(corrected, county_allocated)
