from __future__ import annotations

import hashlib
from pathlib import Path
import pandas as pd
import pytest

from impacts.pipeline.emfac.common import read_atlas_vehicles_input
from impacts.pipeline.emfac.fleet.step1_build_vehicle_types import _build_atlas_vehicle_type_targets
from impacts.pipeline.emfac.fleet.step1_build_vehicle_types import _build_passenger_vehicle_types_from_atlas_targets
from impacts.pipeline.emfac.fleet.step3_map_emfac_atlas import _assign_passenger_fuel_consumption_fields
from impacts.pipeline.emfac.fleet.step3_map_emfac_atlas import _build_passenger_emfac_candidates
from impacts.pipeline.emfac.fleet.step3_map_emfac_atlas import _combine_passenger_vehicle_types_for_output
from impacts.pipeline.emfac.fleet.step3_map_emfac_atlas import _finalize_passenger_vehicle_type_probabilities
from impacts.pipeline.emfac.fleet.step3_map_emfac_atlas import _prepare_mapped_passenger_vehicles_output
from impacts.pipeline.emfac.fleet.step3_map_emfac_atlas import _read_step3_passenger_vehicle_types
from impacts.pipeline.emfac.fleet.step3_map_emfac_atlas import _sample_passenger_vehicle_type_ids_for_vehicles


def test_build_atlas_vehicle_type_targets_preserves_exact_model_year_combinations() -> None:
    vehicles = pd.DataFrame(
        [
            {"bodytype": "car", "modelyear": 2001, "adopt_fuel": "conv"},
            {"bodytype": "car", "modelyear": 2003, "adopt_fuel": "conv"},
            {"bodytype": "car", "modelyear": 2004, "adopt_fuel": "conv"},
            {"bodytype": "car", "modelyear": 2019, "adopt_fuel": "conv"},
        ]
    )

    result = _build_atlas_vehicle_type_targets(
        vehicles,
        config={"atlas": {"fuel_map": {"gasoline": ["conv"]}}},
    )

    assert set(result["atlasVehicleTypeId"]) == {
        "CarConv2001",
        "CarConv2003",
        "CarConv2004",
        "CarConv2019",
    }
    assert len(result) == 4
    assert result["atlasVehicleTypeId"].is_unique
    assert result.loc[result["modelyear"] == 2001, "vehicleCount"].iloc[0] == 1
    assert result.loc[result["modelyear"] == 2003, "vehicleCount"].iloc[0] == 1
    assert result.loc[result["modelyear"] == 2004, "vehicleCount"].iloc[0] == 1
    assert result.loc[result["modelyear"] == 2019, "vehicleCount"].iloc[0] == 1


def test_build_passenger_vehicle_types_from_atlas_targets_uses_beam_fuel_specific_car_defaults() -> None:
    source_car_vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": "electric-template",
                "vehicleCategory": "Car",
                "primaryFuelType": "electricity",
                "secondaryFuelType": "",
                "seatingCapacity": 4,
                "sampleProbabilityWithinCategory": "0.1",
                "sampleProbabilityString": "income | 0-50:0.1",
                "primaryVehicleEnergyFile": "fuel/electric.csv",
                "secondaryVehicleEnergyFile": "",
            },
            {
                "vehicleTypeId": "phev-template",
                "vehicleCategory": "Car",
                "primaryFuelType": "electricity",
                "secondaryFuelType": "gasoline",
                "seatingCapacity": 7,
                "sampleProbabilityWithinCategory": "0.2",
                "sampleProbabilityString": "income | 0-50:0.2",
                "primaryVehicleEnergyFile": "fuel/phev_primary.csv",
                "secondaryVehicleEnergyFile": "fuel/phev_secondary.csv",
            },
        ]
    )
    atlas_vehicle_type_targets = pd.DataFrame(
        [
            {
                "atlasVehicleTypeId": "CarEv2019",
                "bodytype": "car",
                "passenger_bodytype_norm": "car",
                "modelyear": 2019,
                "adopt_fuel": "ev",
                "beamFuel": "electricity",
                "fleetShare": 0.4,
                "incomeBin": "50-100",
                "incomeProbability": 1.0,
            },
            {
                "atlasVehicleTypeId": "CarPhev2019",
                "bodytype": "car",
                "passenger_bodytype_norm": "car",
                "modelyear": 2019,
                "adopt_fuel": "phev",
                "beamFuel": "electricity+gasoline",
                "fleetShare": 0.6,
                "incomeBin": "50-100",
                "incomeProbability": 1.0,
            },
        ]
    )

    result = _build_passenger_vehicle_types_from_atlas_targets(
        config={
            "passenger_mapping": {
                "fuel_types": {
                    "electricity": ["Elec"],
                    "gasoline": ["Gas"],
                    "electricity+gasoline": ["Phe"],
                }
            }
        },
        source_car_vehicle_types=source_car_vehicle_types,
        atlas_vehicle_type_targets=atlas_vehicle_type_targets,
    )

    electric_row = result.loc[result["beamFuel"] == "electricity"].iloc[0]
    phev_row = result.loc[result["beamFuel"] == "electricity+gasoline"].iloc[0]

    assert electric_row["seatingCapacity"] == 4
    assert electric_row["primaryFuelType"] == "electricity"
    assert electric_row["secondaryFuelType"] == ""
    assert electric_row["adopt_fuel"] == "electricity"
    assert electric_row["primaryVehicleEnergyFile"] == ""
    assert electric_row["secondaryVehicleEnergyFile"] == ""

    assert phev_row["seatingCapacity"] == 7
    assert phev_row["primaryFuelType"] == "electricity"
    assert phev_row["secondaryFuelType"] == "gasoline"
    assert phev_row["adopt_fuel"] == "electricity+gasoline"
    assert phev_row["primaryVehicleEnergyFile"] == ""
    assert phev_row["secondaryVehicleEnergyFile"] == ""


def test_build_passenger_emfac_candidates_matches_grouped_model_year_exactly() -> None:
    candidates = pd.DataFrame(
        [
            {
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": "<=2003",
                "fleetVmtPrior": 0.1,
                "fleetPopulationPrior": 0.1,
                "total_vmt_vehicle_miles_per_year": 10.0,
                "population_vehicles": 5.0,
            },
            {
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": "2004-2014",
                "fleetVmtPrior": 0.7,
                "fleetPopulationPrior": 0.7,
                "total_vmt_vehicle_miles_per_year": 70.0,
                "population_vehicles": 35.0,
            },
            {
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": ">=2015",
                "fleetVmtPrior": 0.2,
                "fleetPopulationPrior": 0.2,
                "total_vmt_vehicle_miles_per_year": 20.0,
                "population_vehicles": 10.0,
            },
        ]
    )
    body_type_mapping = pd.DataFrame(
        [{"body_type": "car", "vehicleCategory": "LDA"}]
    )
    fuel_mapping = pd.DataFrame(
        [{"emfac_vehicle_category": "LDA", "emfac_fuel": "Gas", "adopt_fuel": "gasoline"}]
    )

    result = _build_passenger_emfac_candidates(
        vehicle_type_id="CarConv<=2003",
        bodytype="car",
        model_year_group="<=2003",
        adopt_fuel="gasoline",
        emfac_candidates=candidates,
        body_type_mapping=body_type_mapping,
        fuel_mapping=fuel_mapping,
        passenger_mapping={},
    )

    assert result["modelYear"].tolist() == ["<=2003"]
    assert result["fleetVmtPrior"].iloc[0] == 1.0
    assert result["fleetPopulationPrior"].iloc[0] == 1.0


def test_build_passenger_emfac_candidates_uses_configured_fuel_fallback_when_exact_year_match_is_missing() -> None:
    candidates = pd.DataFrame(
        [
            {
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": "<=2003",
                "fleetVmtPrior": 0.8,
                "fleetPopulationPrior": 0.8,
                "total_vmt_vehicle_miles_per_year": 80.0,
                "population_vehicles": 40.0,
            },
            {
                "vehicleCategory": "LDA",
                "fuel": "Phe",
                "modelYear": "2004-2014",
                "fleetVmtPrior": 0.4,
                "fleetPopulationPrior": 0.4,
                "total_vmt_vehicle_miles_per_year": 40.0,
                "population_vehicles": 20.0,
            },
            {
                "vehicleCategory": "LDA",
                "fuel": "Phe",
                "modelYear": ">=2015",
                "fleetVmtPrior": 0.6,
                "fleetPopulationPrior": 0.6,
                "total_vmt_vehicle_miles_per_year": 60.0,
                "population_vehicles": 30.0,
            },
        ]
    )
    body_type_mapping = pd.DataFrame(
        [{"body_type": "car", "vehicleCategory": "LDA"}]
    )
    fuel_mapping = pd.DataFrame(
        [{"emfac_vehicle_category": "LDA", "emfac_fuel": "Phe", "adopt_fuel": "electricity+gasoline"}]
    )

    result = _build_passenger_emfac_candidates(
        vehicle_type_id="CarPhev<=2003",
        bodytype="car",
        model_year_group="<=2003",
        adopt_fuel="electricity+gasoline",
        emfac_candidates=candidates,
        body_type_mapping=body_type_mapping,
        fuel_mapping=fuel_mapping,
        passenger_mapping={
            "fuel_fallbacks": [
                {
                    "source_fuel": "electricity+gasoline",
                    "if_model_year": "<=2003",
                    "fallback_emfac_fuels": ["Gas"],
                }
            ]
        },
    )

    assert result["fuel"].tolist() == ["Gas"]
    assert result["modelYear"].tolist() == ["<=2003"]
    assert result["fleetVmtPrior"].iloc[0] == 1.0
    assert result["fleetPopulationPrior"].iloc[0] == 1.0


def test_build_passenger_emfac_candidates_maps_hybrid_to_gasoline_emfac_candidates() -> None:
    candidates = pd.DataFrame(
        [
            {
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": "2004-2014",
                "fleetVmtPrior": 0.7,
                "fleetPopulationPrior": 0.7,
                "total_vmt_vehicle_miles_per_year": 70.0,
                "population_vehicles": 35.0,
            },
            {
                "vehicleCategory": "LDA",
                "fuel": "Phe",
                "modelYear": "2004-2014",
                "fleetVmtPrior": 0.3,
                "fleetPopulationPrior": 0.3,
                "total_vmt_vehicle_miles_per_year": 30.0,
                "population_vehicles": 15.0,
            },
        ]
    )
    body_type_mapping = pd.DataFrame(
        [{"body_type": "car", "vehicleCategory": "LDA"}]
    )
    fuel_mapping = pd.DataFrame(
        [
            {"emfac_vehicle_category": "LDA", "emfac_fuel": "Gas", "adopt_fuel": "hybrid"},
            {"emfac_vehicle_category": "LDA", "emfac_fuel": "Phe", "adopt_fuel": "electricity+gasoline"},
        ]
    )

    result = _build_passenger_emfac_candidates(
        vehicle_type_id="CarHybrid2004-2014",
        bodytype="car",
        model_year_group="2004-2014",
        adopt_fuel="hybrid",
        emfac_candidates=candidates,
        body_type_mapping=body_type_mapping,
        fuel_mapping=fuel_mapping,
        passenger_mapping={},
    )

    assert result["fuel"].tolist() == ["Gas"]
    assert result["modelYear"].tolist() == ["2004-2014"]


def test_combine_passenger_vehicle_types_for_output_keeps_all_sections_and_blanks_other_emfac_fields() -> None:
    passenger_car = pd.DataFrame(
        [
            {
                "vehicleTypeId": "car-1",
                "vehicleCategory": "Car",
                "sampleProbabilityWithinCategory": "0.9",
                "sampleProbabilityString": "income | 0-999999:0.900000",
                "adopt_fuel": "gasoline",
                "bodytype": "car",
                "modelyear": 2019,
                "emfacId": "post2014LDAGas",
                "emissionsRatesFile": "emfacId=post2014LDAGas/post2014LDAGas.parquet",
                "idleTimeFraction": 0.12,
            }
        ]
    )
    passenger_bus = pd.DataFrame(
        [
            {
                "vehicleTypeId": "bus-1",
                "vehicleCategory": "Bus",
                "sampleProbabilityWithinCategory": "1.0",
                "sampleProbabilityString": "income | 0-999999:1.000000",
                "adopt_fuel": "diesel",
                "emfacId": "post2014MDBusDsl",
                "emissionsRatesFile": "emfacId=post2014MDBusDsl/post2014MDBusDsl.parquet",
                "idleTimeFraction": 0.25,
            }
        ]
    )
    passenger_bike = pd.DataFrame(
        [
            {
                "vehicleTypeId": "bike-1",
                "vehicleCategory": "Bike",
                "sampleProbabilityWithinCategory": "1.0",
                "sampleProbabilityString": "income | 0-999999:1.000000",
                "adopt_fuel": "human",
                "emfacId": "bikeHuman",
                "emissionsRatesFile": "emfacId=bikeHuman/bikeHuman.parquet",
                "idleTimeFraction": 0.05,
            }
        ]
    )
    passenger_other = pd.DataFrame(
        [
            {
                "vehicleTypeId": "other-1",
                "vehicleCategory": "Body",
                "sampleProbabilityWithinCategory": "1.0",
                "sampleProbabilityString": "income | 0-999999:1.000000",
                "adopt_fuel": "other",
                "emfacId": "should-be-cleared",
                "emissionsRatesFile": "should-be-cleared.parquet",
                "idleTimeFraction": 0.77,
            }
        ]
    )

    result = _combine_passenger_vehicle_types_for_output(
        passenger_car_with_emfac=passenger_car,
        passenger_bus_with_emfac=passenger_bus,
        passenger_bike_with_emfac=passenger_bike,
        passenger_other_with_emfac=passenger_other,
    )

    assert result["vehicleTypeId"].tolist() == ["car-1", "bus-1", "bike-1", "other-1"]

    car_row = result.loc[result["vehicleTypeId"] == "car-1"].iloc[0]
    assert car_row["emfacId"] == "post2014LDAGas"
    assert car_row["emissionsRatesFile"] == "emfacId=post2014LDAGas/post2014LDAGas.parquet"
    assert car_row["idleTimeFraction"] == 0.12

    bus_row = result.loc[result["vehicleTypeId"] == "bus-1"].iloc[0]
    assert bus_row["emfacId"] == "post2014MDBusDsl"
    assert bus_row["emissionsRatesFile"] == "emfacId=post2014MDBusDsl/post2014MDBusDsl.parquet"
    assert bus_row["idleTimeFraction"] == 0.25

    bike_row = result.loc[result["vehicleTypeId"] == "bike-1"].iloc[0]
    assert bike_row["emfacId"] == "bikeHuman"
    assert bike_row["emissionsRatesFile"] == "emfacId=bikeHuman/bikeHuman.parquet"
    assert bike_row["idleTimeFraction"] == 0.05

    other_row = result.loc[result["vehicleTypeId"] == "other-1"].iloc[0]
    assert other_row["emfacId"] == ""
    assert other_row["emissionsRatesFile"] == ""
    assert other_row["idleTimeFraction"] == ""


def test_read_step3_passenger_vehicle_types_preserves_full_vehicle_type_columns(tmp_path: Path) -> None:
    path = tmp_path / "passenger_vehicle_types.csv"
    pd.DataFrame(
        [
            {
                "vehicleTypeId": "car-1",
                "vehicleCategory": "Car",
                "bodytype": "car",
                "adopt_fuel": "gasoline",
                "modelyear": 2019,
                "sampleProbabilityWithinCategory": "1.0",
                "sampleProbabilityString": "income | 0-999999:1.000000",
                "primaryFuelType": "gasoline",
                "primaryFuelConsumptionInJoulePerMeter": 10.5,
                "seatingCapacity": 5,
                "mappingVehicleTypeId": "source-car-1",
            }
        ]
    ).to_csv(path, index=False)

    result = _read_step3_passenger_vehicle_types(str(path))

    assert result.loc[0, "primaryFuelType"] == "gasoline"
    assert result.loc[0, "primaryFuelConsumptionInJoulePerMeter"] == 10.5
    assert result.loc[0, "seatingCapacity"] == 5
    assert result.loc[0, "mappingVehicleTypeId"] == "source-car-1"


def test_build_passenger_emfac_candidates_splits_by_fleet_share() -> None:
    candidates = pd.DataFrame(
        [
            {
                "emfacId": "post2014LDAGas",
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": "2004-2014",
                "fleetVmtPrior": 0.9,
                "fleetPopulationPrior": 0.1,
                "total_vmt_vehicle_miles_per_year": 90.0,
                "population_vehicles": 10.0,
            },
            {
                "emfacId": "pre2004LDAGas",
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": "2004-2014",
                "fleetVmtPrior": 0.1,
                "fleetPopulationPrior": 0.9,
                "total_vmt_vehicle_miles_per_year": 10.0,
                "population_vehicles": 90.0,
            },
        ]
    )
    body_type_mapping = pd.DataFrame(
        [{"body_type": "car", "vehicleCategory": "LDA"}]
    )
    fuel_mapping = pd.DataFrame(
        [{"emfac_vehicle_category": "LDA", "emfac_fuel": "Gas", "adopt_fuel": "gasoline"}]
    )

    result = _build_passenger_emfac_candidates(
        vehicle_type_id="CarConv2004-2014",
        bodytype="car",
        model_year_group="2004-2014",
        adopt_fuel="gasoline",
        emfac_candidates=candidates,
        body_type_mapping=body_type_mapping,
        fuel_mapping=fuel_mapping,
        passenger_mapping={},
    )

    fleet_vmt_prior = dict(zip(result["emfacId"], result["fleetVmtPrior"]))
    fleet_population_prior = dict(zip(result["emfacId"], result["fleetPopulationPrior"]))
    assert fleet_vmt_prior["post2014LDAGas"] > fleet_vmt_prior["pre2004LDAGas"]
    assert fleet_population_prior["post2014LDAGas"] < fleet_population_prior["pre2004LDAGas"]


def test_build_passenger_emfac_candidates_errors_when_exact_year_match_is_missing_without_fallback() -> None:
    candidates = pd.DataFrame(
        [
            {
                "vehicleCategory": "MDV",
                "fuel": "Gas",
                "modelYear": "2003-2006",
                "fleetVmtPrior": 0.2,
                "fleetPopulationPrior": 0.2,
                "total_vmt_vehicle_miles_per_year": 20.0,
                "population_vehicles": 10.0,
            },
            {
                "vehicleCategory": "MDV",
                "fuel": "Gas",
                "modelYear": "2007-2009",
                "fleetVmtPrior": 0.3,
                "fleetPopulationPrior": 0.3,
                "total_vmt_vehicle_miles_per_year": 30.0,
                "population_vehicles": 15.0,
            },
            {
                "vehicleCategory": "MDV",
                "fuel": "Gas",
                "modelYear": "2010-2012",
                "fleetVmtPrior": 0.4,
                "fleetPopulationPrior": 0.4,
                "total_vmt_vehicle_miles_per_year": 40.0,
                "population_vehicles": 20.0,
            },
            {
                "vehicleCategory": "MDV",
                "fuel": "Gas",
                "modelYear": "2013-2015",
                "fleetVmtPrior": 0.5,
                "fleetPopulationPrior": 0.5,
                "total_vmt_vehicle_miles_per_year": 50.0,
                "population_vehicles": 25.0,
            },
            {
                "vehicleCategory": "MDV",
                "fuel": "Gas",
                "modelYear": "<=2002",
                "fleetVmtPrior": 0.1,
                "fleetPopulationPrior": 0.1,
                "total_vmt_vehicle_miles_per_year": 10.0,
                "population_vehicles": 5.0,
            },
            {
                "vehicleCategory": "MDV",
                "fuel": "Gas",
                "modelYear": ">=2016",
                "fleetVmtPrior": 0.6,
                "fleetPopulationPrior": 0.6,
                "total_vmt_vehicle_miles_per_year": 60.0,
                "population_vehicles": 30.0,
            },
        ]
    )
    body_type_mapping = pd.DataFrame(
        [{"body_type": "car", "vehicleCategory": "MDV"}]
    )
    fuel_mapping = pd.DataFrame(
        [{"emfac_vehicle_category": "MDV", "emfac_fuel": "Gas", "adopt_fuel": "gasoline"}]
    )

    try:
        _build_passenger_emfac_candidates(
            vehicle_type_id="CarConv2004-2014",
            bodytype="car",
            model_year_group="2004-2014",
            adopt_fuel="gasoline",
            emfac_candidates=candidates,
            body_type_mapping=body_type_mapping,
            fuel_mapping=fuel_mapping,
            passenger_mapping={},
        )
    except ValueError as error:
        assert "No passenger EMFAC candidates matched the configured modelYear group" in str(error)
    else:
        raise AssertionError("Expected ValueError when no exact passenger EMFAC modelYear match exists")


def test_prepare_mapped_passenger_vehicles_output_preserves_existing_columns() -> None:
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
        "vehicleTypeId",
        "stateOfCharge",
        "bodytype",
        "householdId",
        "vehicleId",
        "initialSoc",
    ]
    assert result.loc[0, "household_id"] == "353"
    assert result.loc[0, "vehicle_id"] == "2"
    assert result.loc[0, "stateOfCharge"] == "0.75"
    assert result.loc[0, "bodytype"] == "Sedan"
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


def test_read_atlas_vehicles_input_preserves_full_schema(tmp_path: Path) -> None:
    vehicles_file = tmp_path / "vehicles.csv"
    pd.DataFrame(
        [
            {
                "vehicle_id": 2,
                "household_id": 353,
                "bodytype": "Car",
                "modelyear": 2019,
                "adopt_fuel": "gasoline",
                "extraFlag": "keep-me",
                "weight": 12.5,
            }
        ]
    ).to_csv(vehicles_file, index=False)

    loaded = read_atlas_vehicles_input(str(vehicles_file))

    assert list(loaded.columns) == ["vehicle_id", "household_id", "bodytype", "modelyear", "adopt_fuel", "extraFlag", "weight"]
    assert loaded.loc[0, "extraFlag"] == "keep-me"
    assert loaded.loc[0, "weight"] == 12.5


def _write_step3_test_model_file(model_file: Path) -> None:
    model_file.write_text(
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
                "          port_location: 1.0",
                "      evidence: {}",
                "    passenger_bayesian_dag:",
                "      scoring:",
                "        likelihood_floor: 0.001",
                "        weights:",
                "          fleet_vmt_prior: 1.0",
                "          fleet_population_prior: 1.0",
                "          income: 1.0",
                "      evidence:",
                "        income:",
                "          center_ratio: 0.30",
                "          sigma_ratio: 0.10",
                "  mappings:",
                "    fuel_consumption:",
                "      - fastsim_id: 2015_gasoline_Chrysler_200",
                "        vehicle_categories: [LDA]",
                "        fuel_types: [Gas]",
                "    freight:",
                "      vehicle_categories:",
                "        Class12aVocational: [LDA]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "      naics_sector:",
                "        - naics_code_2: ['11']",
                "          vehicle_category: [T7 Tractor Class 8]",
                "      port_location:",
                "        - zone_codes: ['060019819001']",
                "          vehicle_category: [T7 POAK Class 8]",
                "    passenger:",
                "      body_types:",
                "        car: [LDA]",
                "      fuel_types:",
                "        gasoline: [Gas]",
                "      vehicle_categories:",
                "        Car: [LDA]",
            ]
        ),
        encoding="utf-8",
    )


def _write_step3_test_catalog(catalog_file: Path) -> None:
    catalog_file.write_text(
        "\n".join(
            [
                "fastsim_id,model_year,fuel,charge_behavior,model_trim,msrp_usd,fastsim_relative_path",
                "2015_gasoline_Chrysler_200,2015,gasoline,,Base,24000,test.csv",
            ]
        ),
        encoding="utf-8",
    )


def _write_step3_test_source_vehicle_types(source_file: Path) -> None:
    pd.DataFrame(
        [
            {
                "vehicleTypeId": "2015_gasoline_Chrysler_200",
                "primaryVehicleEnergyFile": "source/2015_gasoline_Chrysler_200_lookup_table.csv.gz",
                "secondaryVehicleEnergyFile": "",
                "primaryFuelType": "gasoline",
                "secondaryFuelType": "",
            }
        ]
    ).to_csv(source_file, index=False)


def test_assign_passenger_fuel_consumption_fields_attach_fastsim_id_and_rewrite_vehicle_type_id(tmp_path: Path) -> None:
    model_file = tmp_path / "fleet_assignment.yaml"
    catalog_file = tmp_path / "fuel_catalog.csv"
    source_vehicle_types_file = tmp_path / "passenger_vehicle_types.csv"
    _write_step3_test_model_file(model_file)
    _write_step3_test_catalog(catalog_file)
    _write_step3_test_source_vehicle_types(source_vehicle_types_file)
    passenger_vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": "2004to2014LDAGas--CarConv2008",
                "atlasVehicleTypeId": "CarConv2008",
                "emfacId": "2004to2014LDAGas",
                "emfacVehicleCategory": "LDA",
                "emfacFuel": "Gas",
                "emfacResolvedModelYear": "2004-2014",
                "sampleProbabilityWithinCategory": "1.000000",
                "sampleProbabilityString": "income | 100-200:1.000000",
            }
        ]
    )

    result = _assign_passenger_fuel_consumption_fields(
        passenger_car_vehicle_types=passenger_vehicle_types,
        config={
            "assignment_model": str(model_file),
            "fuel_consumption_catalog": str(catalog_file),
            "passenger_vehicle_types_file": str(source_vehicle_types_file),
            "fuel_consumption_mapping": [
                {
                    "fastsim_id": "2015_gasoline_Chrysler_200",
                    "vehicle_categories": ["LDA"],
                    "fuel_types": ["Gas"],
                }
            ]
        },
        seed=0,
    )

    assert result.loc[0, "fuelConsumptionId"] == "2015_gasoline_Chrysler_200"
    assert result.loc[0, "msrp_usd"] == 24000.0
    assert result.loc[0, "primaryVehicleEnergyFile"] == "test.csv"
    assert result.loc[0, "secondaryVehicleEnergyFile"] == ""
    expected_mapping_id = "2015gasolineChrysler200--2004to2014LDAGas--CarConv2008"
    expected_vehicle_type_id = f"paxcar-{hashlib.sha256(expected_mapping_id.encode('utf-8')).hexdigest()[:6]}"
    assert result.loc[0, "mappingVehicleTypeId"] == expected_mapping_id
    assert result.loc[0, "vehicleTypeId"] == expected_vehicle_type_id


def test_assign_passenger_fuel_consumption_fields_raise_when_no_match_exists(tmp_path: Path) -> None:
    model_file = tmp_path / "fleet_assignment.yaml"
    catalog_file = tmp_path / "fuel_catalog.csv"
    source_vehicle_types_file = tmp_path / "passenger_vehicle_types.csv"
    _write_step3_test_model_file(model_file)
    _write_step3_test_catalog(catalog_file)
    _write_step3_test_source_vehicle_types(source_vehicle_types_file)
    passenger_vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": "2004to2014LDAPhe--CarPhev2008",
                "atlasVehicleTypeId": "CarPhev2008",
                "emfacId": "2004to2014LDAPhe",
                "emfacVehicleCategory": "LDA",
                "emfacFuel": "Phe",
                "emfacResolvedModelYear": "2004-2014",
                "sampleProbabilityWithinCategory": "1.000000",
                "sampleProbabilityString": "income | 100-200:1.000000",
            }
        ]
    )

    with pytest.raises(ValueError, match="No fuel-consumption mapping matched"):
        _assign_passenger_fuel_consumption_fields(
            passenger_car_vehicle_types=passenger_vehicle_types,
            config={
                "assignment_model": str(model_file),
                "fuel_consumption_catalog": str(catalog_file),
                "passenger_vehicle_types_file": str(source_vehicle_types_file),
                "fuel_consumption_mapping": [
                    {
                        "fastsim_id": "2015_gasoline_Chrysler_200",
                        "vehicle_categories": ["LDA"],
                        "fuel_types": ["Gas"],
                    }
                ]
            },
            seed=0,
        )


def test_assign_passenger_fuel_consumption_fields_keep_baseline_values_when_mapping_missing(
    tmp_path: Path, capsys
) -> None:
    model_file = tmp_path / "fleet_assignment.yaml"
    catalog_file = tmp_path / "fuel_catalog.csv"
    source_vehicle_types_file = tmp_path / "passenger_vehicle_types.csv"
    _write_step3_test_model_file(model_file)
    _write_step3_test_catalog(catalog_file)
    _write_step3_test_source_vehicle_types(source_vehicle_types_file)
    passenger_vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": "2004to2014LDAPhe--CarPhev2008",
                "atlasVehicleTypeId": "CarPhev2008",
                "emfacId": "2004to2014LDAPhe",
                "emfacVehicleCategory": "LDA",
                "emfacFuel": "Phe",
                "emfacResolvedModelYear": "2004-2014",
                "sampleProbabilityWithinCategory": "1.000000",
                "sampleProbabilityString": "income | 100-200:1.000000",
                "primaryFuelType": "electricity",
                "primaryFuelConsumptionInJoulePerMeter": 1.0,
                "primaryFuelCapacityInJoule": 4.0e10,
                "msrp_usd": 30000.0,
                "primaryVehicleEnergyFile": "",
                "secondaryVehicleEnergyFile": "",
            }
        ]
    )

    result = _assign_passenger_fuel_consumption_fields(
        passenger_car_vehicle_types=passenger_vehicle_types,
        config={
            "assignment_model": str(model_file),
            "fuel_consumption_catalog": str(catalog_file),
            "passenger_vehicle_types_file": str(source_vehicle_types_file),
            "fuel_consumption_mapping": [
                {
                    "fastsim_id": "2015_gasoline_Chrysler_200",
                    "vehicle_categories": ["LDA"],
                    "fuel_types": ["Gas"],
                }
            ],
        },
        seed=0,
    )

    assert result.loc[0, "fuelConsumptionId"] == ""
    assert result.loc[0, "msrp_usd"] == 30000.0
    assert result.loc[0, "primaryVehicleEnergyFile"] == ""
    assert result.loc[0, "secondaryVehicleEnergyFile"] == ""
    expected_mapping_id = "unmapped--2004to2014LDAPhe--CarPhev2008"
    expected_vehicle_type_id = f"paxcar-{hashlib.sha256(expected_mapping_id.encode('utf-8')).hexdigest()[:6]}"
    assert result.loc[0, "mappingVehicleTypeId"] == expected_mapping_id
    assert result.loc[0, "vehicleTypeId"] == expected_vehicle_type_id
    assert "WARNING: Passenger Step 3.3 leaving fuel-consumption template fields empty" in capsys.readouterr().out


def test_finalize_passenger_vehicle_type_probabilities_normalizes_within_vehicle_category() -> None:
    passenger_vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": "car-a",
                "vehicleCategory": "Car",
                "sampleProbabilityWithinCategory": "0.000000",
                "sampleProbabilityString": "",
            },
            {
                "vehicleTypeId": "car-b",
                "vehicleCategory": "Car",
                "sampleProbabilityWithinCategory": "0.000000",
                "sampleProbabilityString": "",
            },
            {
                "vehicleTypeId": "suv-a",
                "vehicleCategory": "SUV",
                "sampleProbabilityWithinCategory": "0.000000",
                "sampleProbabilityString": "",
            },
            {
                "vehicleTypeId": "suv-b",
                "vehicleCategory": "SUV",
                "sampleProbabilityWithinCategory": "0.000000",
                "sampleProbabilityString": "",
            },
        ]
    )
    sampled_vehicles = pd.DataFrame(
        [
            {"vehicleTypeId": "car-a", "income_in_thousands": 50.0},
            {"vehicleTypeId": "car-a", "income_in_thousands": 50.0},
            {"vehicleTypeId": "car-b", "income_in_thousands": 50.0},
            {"vehicleTypeId": "suv-a", "income_in_thousands": 50.0},
            {"vehicleTypeId": "suv-b", "income_in_thousands": 50.0},
            {"vehicleTypeId": "suv-b", "income_in_thousands": 50.0},
        ]
    )

    result = _finalize_passenger_vehicle_type_probabilities(
        passenger_car_vehicle_types=passenger_vehicle_types,
        sampled_vehicles=sampled_vehicles,
        config={"income_bins": [0, 100]},
    )

    probabilities = result.set_index("vehicleTypeId")["sampleProbabilityWithinCategory"].to_dict()
    assert probabilities == {
        "car-a": "0.666667",
        "car-b": "0.333333",
        "suv-a": "0.333333",
        "suv-b": "0.666667",
    }

    probability_strings = result.set_index("vehicleTypeId")["sampleProbabilityString"].to_dict()
    assert probability_strings == {
        "car-a": "income | all:0.666667",
        "car-b": "income | all:0.333333",
        "suv-a": "income | all:0.333333",
        "suv-b": "income | all:0.666667",
    }


def test_sample_passenger_vehicle_type_ids_for_vehicles_uses_canonical_atlas_vehicle_type_id() -> None:
    vehicles = pd.DataFrame(
        [
            {"household_id": 1, "bodytype": "car", "modelyear": 2008, "adopt_fuel": "conv"},
        ]
    )
    households = pd.DataFrame(
        [
            {"household_id": 1, "income_in_thousands": 120},
        ]
    )
    expected_mapping_id = "2015gasolineChrysler200--2004to2014LDAGas--CarConv2008"
    expected_vehicle_type_id = f"paxcar-{hashlib.sha256(expected_mapping_id.encode('utf-8')).hexdigest()[:12]}"
    passenger_car_vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": expected_vehicle_type_id,
                "atlasVehicleTypeId": "CarConv2008",
                "fleetVmtPrior": "1.000000",
                "fleetPopulationPrior": "1.000000",
                "msrp_usd": 24000.0,
            }
        ]
    )

    result = _sample_passenger_vehicle_type_ids_for_vehicles(
        vehicles=vehicles,
        passenger_car_vehicle_types=passenger_car_vehicle_types,
        households=households,
        config={
            "passenger_bayesian_dag": {
                "likelihood_floor": 0.001,
                "fleet_vmt_prior_weight": 1.0,
                "fleet_population_prior_weight": 1.0,
                "income_weight": 1.0,
                "income_enabled": True,
                "income_center_ratio": 0.30,
                "income_sigma_ratio": 0.10,
            }
        },
        seed=0,
    )

    assert result.loc[0, "vehicleTypeId"] == expected_vehicle_type_id


def test_sample_passenger_vehicle_type_ids_for_vehicles_skips_income_when_disabled() -> None:
    vehicles = pd.DataFrame(
        [
            {"household_id": 1, "bodytype": "car", "modelyear": 2008, "adopt_fuel": "conv"},
        ]
    )
    households = pd.DataFrame(
        [
            {"household_id": 1, "income_in_thousands": 120},
        ]
    )
    passenger_car_vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": "vt-1",
                "atlasVehicleTypeId": "CarConv2008",
                "fleetVmtPrior": "1.000000",
                "fleetPopulationPrior": "1.000000",
            }
        ]
    )

    result = _sample_passenger_vehicle_type_ids_for_vehicles(
        vehicles=vehicles,
        passenger_car_vehicle_types=passenger_car_vehicle_types,
        households=households,
        config={
            "passenger_bayesian_dag": {
                "likelihood_floor": 0.001,
                "fleet_vmt_prior_weight": 1.0,
                "fleet_population_prior_weight": 1.0,
                "income_weight": 1.0,
                "income_enabled": False,
            }
        },
        seed=0,
    )

    assert result.loc[0, "vehicleTypeId"] == "vt-1"
