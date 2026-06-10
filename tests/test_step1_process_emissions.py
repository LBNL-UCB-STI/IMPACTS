from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from impacts.pipeline.workflow.step1_process_emissions import _derive_county_correction_factors
from impacts.pipeline.workflow.step1_process_emissions import _derive_inventory_activity_targets_for_assignment
from impacts.pipeline.workflow.step1_process_emissions import _build_county_corrected_table
from impacts.pipeline.workflow.step1_process_emissions import _build_corrected_source_totals
from impacts.pipeline.workflow.step1_process_emissions import _aggregate_aermod_emissions_for_export
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
            {
                "vehicleTypeId": "pax-car",
                "vehicleCategory": "Car",
                "emfacResolvedModelYear": "post2014",
            },
            {
                "vehicleTypeId": "BUS-DEFAULT",
                "vehicleCategory": "MediumDutyPassenger",
                "emfacVehicleCategory": "UBUS",
                "emfacResolvedModelYear": pd.NA,
            },
            {
                "vehicleTypeId": "RAIL-DEFAULT",
                "vehicleCategory": "Rail-Default",
                "emfacResolvedModelYear": pd.NA,
            },
        ]
    )
    freight_vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": "ft-md",
                "vehicleCategory": "Class456Vocational",
                "vehicleClass": "truck",
                "vehicleUse": "freight",
                "emfacResolvedModelYear": "2007-2009",
            },
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


def test_aggregate_aermod_emissions_for_export_preserves_source_classes(tmp_path: Path) -> None:
    aermod_allocated = pd.DataFrame(
        [
            {
                "aermod_cell_id": 101,
                "source_temporal_class": "CITYSTREET",
                "source_release_height": 1.0,
                "source_urban_class": 1000,
                "tons_per_year_NOx_aermod_allocated": 1.25,
                "tons_per_year_PM25_aermod_allocated": 0.25,
            },
            {
                "aermod_cell_id": 101,
                "source_temporal_class": "FREEWAY",
                "source_release_height": 3.5,
                "source_urban_class": 1000,
                "tons_per_year_NOx_aermod_allocated": 1.25,
                "tons_per_year_PM25_aermod_allocated": 0.25,
            },
            {
                "aermod_cell_id": 101,
                "source_temporal_class": "FREEWAY",
                "source_release_height": 3.5,
                "source_urban_class": 1000,
                "tons_per_year_NOx_aermod_allocated": 2.75,
                "tons_per_year_PM25_aermod_allocated": 0.75,
            },
            {
                "aermod_cell_id": 202,
                "source_temporal_class": "CITYSTREET",
                "source_release_height": 1.0,
                "source_urban_class": 0,
                "tons_per_year_NOx_aermod_allocated": 5.0,
                "tons_per_year_PM25_aermod_allocated": 1.5,
            },
        ]
    )

    aggregated = _aggregate_aermod_emissions_for_export(
        aermod_allocated,
        scratch_dir=tmp_path,
    ).sort_values(["aermod_cell_id", "source_release_height"]).reset_index(drop=True)

    assert aggregated.to_dict("records") == [
        {
            "aermod_cell_id": 101,
            "source_temporal_class": "CITYSTREET",
            "source_release_height": 1.0,
            "source_urban_class": 1000,
            "tons_per_year_NOx_aermod_allocated": 1.25,
            "tons_per_year_PM25_aermod_allocated": 0.25,
        },
        {
            "aermod_cell_id": 101,
            "source_temporal_class": "FREEWAY",
            "source_release_height": 3.5,
            "source_urban_class": 1000,
            "tons_per_year_NOx_aermod_allocated": 4.0,
            "tons_per_year_PM25_aermod_allocated": 1.0,
        },
        {
            "aermod_cell_id": 202,
            "source_temporal_class": "CITYSTREET",
            "source_release_height": 1.0,
            "source_urban_class": 0,
            "tons_per_year_NOx_aermod_allocated": 5.0,
            "tons_per_year_PM25_aermod_allocated": 1.5,
        },
    ]


def test_aggregate_aermod_emissions_for_export_requires_canonical_source_columns(tmp_path: Path) -> None:
    aermod_allocated = pd.DataFrame(
        [
            {
                "aermod_cell_id": 101,
                "tons_per_year_NOx_aermod_allocated": 1.25,
            },
        ]
    )

    try:
        _aggregate_aermod_emissions_for_export(
            aermod_allocated,
            scratch_dir=tmp_path,
        )
    except ValueError as exc:
        assert str(exc) == (
            "AERMOD allocated emissions table is missing required columns: "
            "['source_temporal_class', 'source_release_height', 'source_urban_class']"
        )
    else:
        raise AssertionError("expected canonical AERMOD export columns to be required")


def test_build_corrected_source_totals_preserves_aermod_source_attributes(tmp_path: Path) -> None:
    county_corrected = pd.DataFrame(
        [
            {
                "linkId": 101,
                "vehicleTypeId": "pax-car",
                "process": "RUNEX",
                "roadCategory": "motorway",
                "source_release_height": 3.0,
                "totVMT_county_allocated": 10.0,
                "totTrips_county_allocated": 2.0,
                "tons_per_year_NOx_county_allocated": 1.25,
            },
            {
                "linkId": 101,
                "vehicleTypeId": "pax-car",
                "process": "RUNEX",
                "roadCategory": "motorway",
                "source_release_height": 3.0,
                "totVMT_county_allocated": 5.0,
                "totTrips_county_allocated": 1.0,
                "tons_per_year_NOx_county_allocated": 0.75,
            },
        ]
    )

    corrected = _build_corrected_source_totals(
        county_corrected,
        scratch_dir=tmp_path,
    )

    assert corrected.to_dict("records") == [
        {
            "linkId": 101,
            "vehicleTypeId": "pax-car",
            "process": "RUNEX",
            "roadCategory": "motorway",
            "source_release_height": 3.0,
            "totVMT": 15.0,
            "totTrips": 3.0,
            "tons_per_year_NOx": 2.0,
        },
    ]


def test_build_corrected_source_totals_keeps_distinct_release_heights(tmp_path: Path) -> None:
    county_corrected = pd.DataFrame(
        [
            {
                "linkId": 101,
                "vehicleTypeId": "mixed-source",
                "process": "RUNEX",
                "roadCategory": "motorway",
                "source_release_height": 1.0,
                "tons_per_year_NOx_county_allocated": 1.25,
            },
            {
                "linkId": 101,
                "vehicleTypeId": "mixed-source",
                "process": "RUNEX",
                "roadCategory": "motorway",
                "source_release_height": 3.5,
                "tons_per_year_NOx_county_allocated": 0.75,
            },
        ]
    )

    corrected = _build_corrected_source_totals(
        county_corrected,
        scratch_dir=tmp_path,
    ).sort_values("source_release_height").reset_index(drop=True)

    assert corrected.to_dict("records") == [
        {
            "linkId": 101,
            "vehicleTypeId": "mixed-source",
            "process": "RUNEX",
            "roadCategory": "motorway",
            "source_release_height": 1.0,
            "tons_per_year_NOx": 1.25,
        },
        {
            "linkId": 101,
            "vehicleTypeId": "mixed-source",
            "process": "RUNEX",
            "roadCategory": "motorway",
            "source_release_height": 3.5,
            "tons_per_year_NOx": 0.75,
        },
    ]
