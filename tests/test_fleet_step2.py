from __future__ import annotations

import os
import pandas as pd

from impacts.emfac.common import attach_idle_time_fraction
from impacts.emfac.fleet.step2_map_emfac_bus_bike import _matched_emfac_fuels
from impacts.emfac.fleet.step2_map_emfac_bus_bike import _read_step2_vehicle_types
from impacts.emfac.fleet.step2_map_emfac_bus_bike import run_step2


def _write_model_file(path) -> None:
    path.write_text(
        "\n".join(
            [
                "fleet_assignment:",
                "  models:",
                "    freight_bayesian_dag:",
                "      scoring:",
                "        likelihood_floor: 0.01",
                "        weights:",
                "          fleet_vmt_prior: 1.0",
                "          naics_sector: 1.0",
                "          payload_mass: 1.0",
                "          port_location: 1.0",
                "      evidence: {}",
                "    passenger_bayesian_dag:",
                "      scoring:",
                "        likelihood_floor: 0.001",
                "        weights:",
                "          fleet_vmt_prior: 1.0",
                "          income: 1.0",
                "      evidence:",
                "        income:",
                "          center_ratio: 0.30",
                "          sigma_ratio: 0.10",
                "  mappings:",
                "    freight:",
                "      vehicle_categories:",
                "        Class12aVocational: [LDA]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "      naics_sector: []",
                "      port_location: []",
                "    passenger:",
                "      body_types:",
                "        car: [LDA]",
                "      fuel_types:",
                "        gasoline: [Gas]",
                "        diesel: [Dsl]",
                "        electricity: [Elec]",
                "        naturalgas: [Gas]",
                "      vehicle_categories:",
                "        Bike: [MCY]",
                "        MediumDutyPassenger: [UBUS]",
                ]
        ),
        encoding="utf-8",
    )


def test_matched_emfac_fuels_uses_direct_beam_fuel_key() -> None:
    category_fuel_map = pd.DataFrame(
        [
            {
                "emfac_vehicle_category": "UBUS",
                "emfac_fuel": "Gas",
                "normalized_adopt_fuel": "naturalgas",
            }
        ]
    )

    result = _matched_emfac_fuels(
        emfac_vehicle_category="UBUS",
        adopt_fuel="naturalgas",
        category_fuel_map=category_fuel_map,
    )

    assert result == ["Gas"]


def test_run_step2_assigns_bus_and_bike_and_leaves_other_empty(tmp_path) -> None:
    model_file = tmp_path / "fleet_assignment.yaml"
    _write_model_file(model_file)
    metadata_file = tmp_path / "vehicle_category_attributes.csv"
    output_root = tmp_path / "fleet_output"
    activities_output_root = tmp_path / "activities_output"
    rates_store_root = activities_output_root / "emissions" / "2018-Baseline" / "dataset"
    pd.DataFrame(
        [
            {"emfac_vehicle_category": "UBUS", "idle_time_fraction": 0.25},
            {"emfac_vehicle_category": "MCY", "idle_time_fraction": 0.05},
        ]
    ).to_csv(metadata_file, index=False)
    for emfac_id in ["2019UBUSGas", "2019MCYGas"]:
        partition_dir = rates_store_root / f"emfacId={emfac_id}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"county": "001", "process": "RUNEX"}]).to_parquet(
            partition_dir / f"{emfac_id}.parquet",
            index=False,
        )

    rates = pd.DataFrame(
        [
            {"vehicleCategory": "UBUS", "fuel": "Gas", "modelYear": "2019"},
            {"vehicleCategory": "MCY", "fuel": "Gas", "modelYear": "2019"},
        ]
    )
    activity = pd.DataFrame(
        [
            {
                "vehicleCategory": "UBUS",
                "fuel": "Gas",
                "modelYear": "2019",
                "population_vehicles": 15,
                "total_vmt_vehicle_miles_per_year": 1500,
            },
            {
                "vehicleCategory": "MCY",
                "fuel": "Gas",
                "modelYear": "2019",
                "population_vehicles": 5,
                "total_vmt_vehicle_miles_per_year": 500,
            },
        ]
    )
    fleet = pd.DataFrame(
        [
            {"vehicleCategory": "UBUS", "fuel": "Gas", "modelYear": "2019"},
            {"vehicleCategory": "MCY", "fuel": "Gas", "modelYear": "2019"},
        ]
    )
    rates_file = tmp_path / "passenger_rates.csv"
    activity_file = tmp_path / "passenger_activity.csv"
    fleet_file = tmp_path / "passenger_fleet.csv"
    rates.to_csv(rates_file, index=False)
    activity.to_csv(activity_file, index=False)
    fleet.to_csv(fleet_file, index=False)

    bus_file = tmp_path / "bus_vehicle_types.csv"
    bike_file = tmp_path / "bike_vehicle_types.csv"
    other_file = tmp_path / "other_vehicle_types.csv"
    pd.DataFrame(
        [
            {
                "vehicleTypeId": "bus-type-1",
                "vehicleCategory": "MediumDutyPassenger",
                "adopt_fuel": "gasoline",
                "primaryFuelType": "diesel",
                "seatingCapacity": 30,
            }
        ]
    ).to_csv(bus_file, index=False)
    pd.DataFrame(
        [
            {
                "vehicleTypeId": "bike-type-1",
                "vehicleCategory": "Bike",
                "adopt_fuel": "gasoline",
                "primaryFuelType": "human",
            }
        ]
    ).to_csv(bike_file, index=False)
    pd.DataFrame(
        [
            {
                "vehicleTypeId": "other-type-1",
                "vehicleCategory": "Other",
                "adopt_fuel": "gasoline",
                "primaryFuelType": "other",
            }
        ]
    ).to_csv(other_file, index=False)

    workflow = {
        "config": {
            "activities": {
                "passenger_rates_file": str(rates_file),
                "passenger_activity_file": str(activity_file),
                "passenger_fleet_file": str(fleet_file),
                "outputs": str(activities_output_root),
            },
            "vehicle_type_assignment": {"model_file": str(model_file)},
            "vehicle_category_attributes_file": str(metadata_file),
            "frism": {"year": 2018},
            "output": str(output_root),
        },
        "scenario": "Baseline",
        "built_passenger_bus_vehicle_types_file": str(bus_file),
        "built_passenger_bike_vehicle_types_file": str(bike_file),
        "built_passenger_other_vehicle_types_file": str(other_file),
    }

    result = run_step2(workflow)

    assert result["built_passenger_bus_vehicle_types"].loc[0, "emfacId"] == "2019UBUSGas"
    assert result["built_passenger_bus_vehicle_types"].loc[0, "emfacVehicleCategory"] == "UBUS"
    assert result["built_passenger_bus_vehicle_types"].loc[0, "primaryFuelType"] == "diesel"
    assert result["built_passenger_bus_vehicle_types"].loc[0, "seatingCapacity"] == 30
    assert result["built_passenger_bus_vehicle_types"].loc[0, "idleTimeFraction"] == 0.25
    assert result["built_passenger_bus_vehicle_types"].loc[0, "emissionsRatesFile"] == os.path.relpath(
        rates_store_root / "emfacId=2019UBUSGas" / "2019UBUSGas.parquet",
        output_root,
    )
    assert result["built_passenger_bike_vehicle_types"].loc[0, "emfacId"] == "2019MCYGas"
    assert result["built_passenger_bike_vehicle_types"].loc[0, "emfacVehicleCategory"] == "MCY"
    assert result["built_passenger_bike_vehicle_types"].loc[0, "idleTimeFraction"] == 0.05
    assert result["built_passenger_bike_vehicle_types"].loc[0, "emissionsRatesFile"] == os.path.relpath(
        rates_store_root / "emfacId=2019MCYGas" / "2019MCYGas.parquet",
        output_root,
    )
    assert result["built_passenger_other_vehicle_types"].loc[0, "emfacId"] == ""
    assert result["built_passenger_other_vehicle_types"].loc[0, "emissionsRatesFile"] == ""
    assert pd.isna(result["built_passenger_other_vehicle_types"].loc[0, "idleTimeFraction"])

    written_bus = pd.read_csv(result["built_passenger_bus_vehicle_types_file"], dtype=str).fillna("")
    written_bike = pd.read_csv(result["built_passenger_bike_vehicle_types_file"], dtype=str).fillna("")
    written_other = pd.read_csv(result["built_passenger_other_vehicle_types_file"], dtype=str).fillna("")

    assert written_bus.loc[0, "emfacId"] == "2019UBUSGas"
    assert written_bus.loc[0, "primaryFuelType"] == "diesel"
    assert written_bus.loc[0, "seatingCapacity"] == "30"
    assert written_bus.loc[0, "idleTimeFraction"] == "0.25"
    assert written_bus.loc[0, "emissionsRatesFile"] == os.path.relpath(
        rates_store_root / "emfacId=2019UBUSGas" / "2019UBUSGas.parquet",
        output_root,
    )
    assert written_bike.loc[0, "emfacId"] == "2019MCYGas"
    assert written_bike.loc[0, "idleTimeFraction"] == "0.05"
    assert written_bike.loc[0, "emissionsRatesFile"] == os.path.relpath(
        rates_store_root / "emfacId=2019MCYGas" / "2019MCYGas.parquet",
        output_root,
    )
    assert written_other.loc[0, "emfacId"] == ""
    assert written_other.loc[0, "idleTimeFraction"] == ""
    assert written_other.loc[0, "emissionsRatesFile"] == ""


def test_read_step2_vehicle_types_preserves_full_vehicle_type_columns(tmp_path) -> None:
    path = tmp_path / "vehicle_types.csv"
    pd.DataFrame(
        [
            {
                "vehicleTypeId": "bus-type-1",
                "vehicleCategory": "MediumDutyPassenger",
                "adopt_fuel": "gasoline",
                "primaryFuelType": "diesel",
                "primaryFuelConsumptionInJoulePerMeter": 12.5,
                "seatingCapacity": 30,
                "automationLevel": 3,
            }
        ]
    ).to_csv(path, index=False)

    result = _read_step2_vehicle_types(str(path))

    assert result.loc[0, "vehicleCategory"] == "MediumDutyPassenger"
    assert result.loc[0, "primaryFuelType"] == "diesel"
    assert result.loc[0, "primaryFuelConsumptionInJoulePerMeter"] == 12.5
    assert result.loc[0, "seatingCapacity"] == 30
    assert result.loc[0, "automationLevel"] == 3.0


def test_attach_idle_time_fraction_can_fallback_to_emfac_id_for_mcy_and_ubus() -> None:
    vehicle_types = pd.DataFrame(
        [
            {"vehicleTypeId": "bus-1", "emfacId": "2019UBUSGas"},
            {"vehicleTypeId": "bike-1", "emfacId": "2019MCYGas"},
        ]
    )

    result = attach_idle_time_fraction(
        vehicle_types,
        idle_time_fraction_lookup={"UBUS": 0.25, "MCY": 0.05},
    )

    assert result["idleTimeFraction"].tolist() == [0.25, 0.05]
