from __future__ import annotations

import pandas as pd

from impacts.emfac.fleet.step1_build_vehicle_types import _build_atlas_vehicle_type_targets
from impacts.emfac.fleet.step3_map_emfac_atlas import _attach_passenger_fastsim_templates
from impacts.emfac.fleet.step3_map_emfac_atlas import _build_passenger_emfac_candidates
from impacts.emfac.fleet.step3_map_emfac_atlas import _prepare_mapped_passenger_vehicles_output
from impacts.emfac.fleet.step3_map_emfac_atlas import _sample_passenger_vehicle_type_ids_for_vehicles


def test_build_atlas_vehicle_type_targets_groups_model_years_before_mapping() -> None:
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
        model_year_groups={
            "light_duty": [
                {"max_year": 2003},
                {"min_year": 2004, "max_year": 2014},
                {"min_year": 2015},
            ],
            "medium_heavy_duty": [{"max_year": 2002}, {"min_year": 2003}],
        },
    )

    assert set(result["emfacModelYearGroup"]) == {"pre2004", "2004to2014", "post2014"}
    assert set(result["atlasVehicleTypeId"]) == {
        "Car_Conv_pre2004",
        "Car_Conv_2004to2014",
        "Car_Conv_post2014",
    }
    assert len(result) == 3
    assert result["atlasVehicleTypeId"].is_unique
    assert result.loc[result["emfacModelYearGroup"] == "pre2004", "vehicleCount"].iloc[0] == 2
    assert result.loc[result["emfacModelYearGroup"] == "2004to2014", "vehicleCount"].iloc[0] == 1
    assert result.loc[result["emfacModelYearGroup"] == "post2014", "vehicleCount"].iloc[0] == 1
    assert result.loc[result["emfacModelYearGroup"] == "pre2004", "modelyear"].iloc[0] == 2002
    assert result.loc[result["emfacModelYearGroup"] == "2004to2014", "modelyear"].iloc[0] == 2004
    assert result.loc[result["emfacModelYearGroup"] == "post2014", "modelyear"].iloc[0] == 2019


def test_build_passenger_emfac_candidates_matches_grouped_model_year_exactly() -> None:
    candidates = pd.DataFrame(
        [
            {
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": "pre2004",
                "fleetShare": 0.1,
                "total_vmt_vehicle_miles_per_year": 10.0,
                "population_vehicles": 5.0,
            },
            {
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": "2004to2014",
                "fleetShare": 0.7,
                "total_vmt_vehicle_miles_per_year": 70.0,
                "population_vehicles": 35.0,
            },
            {
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": "post2014",
                "fleetShare": 0.2,
                "total_vmt_vehicle_miles_per_year": 20.0,
                "population_vehicles": 10.0,
            },
        ]
    )
    vehicle_category_weights = pd.DataFrame(
        [{"body_type": "car", "vehicleCategory": "LDA", "bodytypeWeight": 1.0}]
    )
    fuel_mapping = pd.DataFrame(
        [{"emfac_vehicle_category": "LDA", "emfac_fuel": "Gas", "adopt_fuel": "conv"}]
    )

    result = _build_passenger_emfac_candidates(
        vehicle_type_id="CarConvpre2004",
        bodytype="car",
        model_year_group="pre2004",
        adopt_fuel="gasoline",
        emfac_candidates=candidates,
        vehicle_category_weights=vehicle_category_weights,
        fuel_mapping=fuel_mapping,
    )

    assert result["modelYear"].tolist() == ["pre2004"]
    assert result["score"].iloc[0] == 1.0


def test_build_passenger_emfac_candidates_uses_nearest_valid_group_for_impossible_phev_slice() -> None:
    candidates = pd.DataFrame(
        [
            {
                "vehicleCategory": "LDA",
                "fuel": "Phe",
                "modelYear": "2004to2014",
                "fleetShare": 0.4,
                "total_vmt_vehicle_miles_per_year": 40.0,
                "population_vehicles": 20.0,
            },
            {
                "vehicleCategory": "LDA",
                "fuel": "Phe",
                "modelYear": "post2014",
                "fleetShare": 0.6,
                "total_vmt_vehicle_miles_per_year": 60.0,
                "population_vehicles": 30.0,
            },
        ]
    )
    vehicle_category_weights = pd.DataFrame(
        [{"body_type": "car", "vehicleCategory": "LDA", "bodytypeWeight": 1.0}]
    )
    fuel_mapping = pd.DataFrame(
        [{"emfac_vehicle_category": "LDA", "emfac_fuel": "Phe", "adopt_fuel": "phev"}]
    )

    result = _build_passenger_emfac_candidates(
        vehicle_type_id="CarPhevpre2004",
        bodytype="car",
        model_year_group="pre2004",
        adopt_fuel="electricity+gasoline",
        emfac_candidates=candidates,
        vehicle_category_weights=vehicle_category_weights,
        fuel_mapping=fuel_mapping,
    )

    assert result["modelYear"].tolist() == ["2004to2014"]
    assert result["score"].iloc[0] == 1.0


def test_build_passenger_emfac_candidates_maps_hybrid_to_gasoline_emfac_candidates() -> None:
    candidates = pd.DataFrame(
        [
            {
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": "2004to2014",
                "fleetShare": 0.7,
                "total_vmt_vehicle_miles_per_year": 70.0,
                "population_vehicles": 35.0,
            },
            {
                "vehicleCategory": "LDA",
                "fuel": "Phe",
                "modelYear": "2004to2014",
                "fleetShare": 0.3,
                "total_vmt_vehicle_miles_per_year": 30.0,
                "population_vehicles": 15.0,
            },
        ]
    )
    vehicle_category_weights = pd.DataFrame(
        [{"body_type": "car", "vehicleCategory": "LDA", "bodytypeWeight": 1.0}]
    )
    fuel_mapping = pd.DataFrame(
        [
            {"emfac_vehicle_category": "LDA", "emfac_fuel": "Gas", "adopt_fuel": "conv"},
            {"emfac_vehicle_category": "LDA", "emfac_fuel": "Phe", "adopt_fuel": "phev"},
        ]
    )

    result = _build_passenger_emfac_candidates(
        vehicle_type_id="Car_Hybrid_2004to2014",
        bodytype="car",
        model_year_group="2004to2014",
        adopt_fuel="hybrid",
        emfac_candidates=candidates,
        vehicle_category_weights=vehicle_category_weights,
        fuel_mapping=fuel_mapping,
    )

    assert result["fuel"].tolist() == ["Gas"]
    assert result["modelYear"].tolist() == ["2004to2014"]


def test_build_passenger_emfac_candidates_supports_population_and_vmt_bias_knobs() -> None:
    candidates = pd.DataFrame(
        [
            {
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": "2004to2014",
                "fleetShare": 0.5,
                "total_vmt_vehicle_miles_per_year": 90.0,
                "population_vehicles": 10.0,
            },
            {
                "vehicleCategory": "LDA",
                "fuel": "Gas",
                "modelYear": "2004to2014",
                "fleetShare": 0.5,
                "total_vmt_vehicle_miles_per_year": 10.0,
                "population_vehicles": 90.0,
            },
        ]
    )
    vehicle_category_weights = pd.DataFrame(
        [{"body_type": "car", "vehicleCategory": "LDA", "bodytypeWeight": 1.0}]
    )
    fuel_mapping = pd.DataFrame(
        [{"emfac_vehicle_category": "LDA", "emfac_fuel": "Gas", "adopt_fuel": "conv"}]
    )

    population_biased = _build_passenger_emfac_candidates(
        vehicle_type_id="Car_Conv_2004to2014",
        bodytype="car",
        model_year_group="2004to2014",
        adopt_fuel="gasoline",
        emfac_candidates=candidates,
        vehicle_category_weights=vehicle_category_weights,
        fuel_mapping=fuel_mapping,
        emfac_population_bias=1.0,
        emfac_vmt_bias=0.0,
    )
    vmt_biased = _build_passenger_emfac_candidates(
        vehicle_type_id="Car_Conv_2004to2014",
        bodytype="car",
        model_year_group="2004to2014",
        adopt_fuel="gasoline",
        emfac_candidates=candidates,
        vehicle_category_weights=vehicle_category_weights,
        fuel_mapping=fuel_mapping,
        emfac_population_bias=0.0,
        emfac_vmt_bias=1.0,
    )

    assert population_biased.iloc[0]["population_vehicles"] == 90.0
    assert vmt_biased.iloc[0]["total_vmt_vehicle_miles_per_year"] == 90.0


def test_build_passenger_emfac_candidates_matches_overlapping_detailed_year_bins() -> None:
    candidates = pd.DataFrame(
        [
            {
                "vehicleCategory": "MDV",
                "fuel": "Gas",
                "modelYear": "2003to2006",
                "fleetShare": 0.2,
                "total_vmt_vehicle_miles_per_year": 20.0,
                "population_vehicles": 10.0,
            },
            {
                "vehicleCategory": "MDV",
                "fuel": "Gas",
                "modelYear": "2007to2009",
                "fleetShare": 0.3,
                "total_vmt_vehicle_miles_per_year": 30.0,
                "population_vehicles": 15.0,
            },
            {
                "vehicleCategory": "MDV",
                "fuel": "Gas",
                "modelYear": "2010to2012",
                "fleetShare": 0.4,
                "total_vmt_vehicle_miles_per_year": 40.0,
                "population_vehicles": 20.0,
            },
            {
                "vehicleCategory": "MDV",
                "fuel": "Gas",
                "modelYear": "2013to2015",
                "fleetShare": 0.5,
                "total_vmt_vehicle_miles_per_year": 50.0,
                "population_vehicles": 25.0,
            },
            {
                "vehicleCategory": "MDV",
                "fuel": "Gas",
                "modelYear": "pre2003",
                "fleetShare": 0.1,
                "total_vmt_vehicle_miles_per_year": 10.0,
                "population_vehicles": 5.0,
            },
            {
                "vehicleCategory": "MDV",
                "fuel": "Gas",
                "modelYear": "post2015",
                "fleetShare": 0.6,
                "total_vmt_vehicle_miles_per_year": 60.0,
                "population_vehicles": 30.0,
            },
        ]
    )
    vehicle_category_weights = pd.DataFrame(
        [{"body_type": "car", "vehicleCategory": "MDV", "bodytypeWeight": 1.0}]
    )
    fuel_mapping = pd.DataFrame(
        [{"emfac_vehicle_category": "MDV", "emfac_fuel": "Gas", "adopt_fuel": "conv"}]
    )

    result = _build_passenger_emfac_candidates(
        vehicle_type_id="CarConv2004to2014",
        bodytype="car",
        model_year_group="2004to2014",
        adopt_fuel="gasoline",
        emfac_candidates=candidates,
        vehicle_category_weights=vehicle_category_weights,
        fuel_mapping=fuel_mapping,
    )

    assert result["modelYear"].tolist() == ["2013to2015", "2010to2012", "2007to2009", "2003to2006"]


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


def test_attach_passenger_fastsim_templates_keeps_blank_emfac_rows() -> None:
    passenger_rows = pd.DataFrame(
        [
            {
                "vehicleTypeId": "Car_Hybrid_2004to2014",
                "oldVehicleTypeId": "Car_Hybrid_2004to2014",
                "sampleProbabilityWithinCategory": "0.100000",
                "sampleProbabilityString": "income | 0-50:0.100000",
                "adopt_fuel": "hybrid",
                "emfacId": "",
                "emfacVehicleCategory": "",
                "emfacFuel": "",
                "emfacResolvedModelYear": "",
                "bodytype": "car",
                "emfacModelYearGroup": "2004to2014",
                "modelyear": 2010,
                "primaryFuelType": "Electricity",
                "secondaryFuelType": "Gasoline",
                "primaryVehicleEnergyFile": "fuel/sample.csv",
                "secondaryVehicleEnergyFile": "",
            }
        ]
    )
    source_vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": "2019Template",
                "primaryFuelType": "Electricity",
                "secondaryFuelType": "Gasoline",
                "primaryVehicleEnergyFile": "fuel/template.csv",
                "secondaryVehicleEnergyFile": "",
            }
        ]
    )
    vehicle_type_mapping = pd.DataFrame(
        [
            {
                "vehicleTypeId": "2019Template",
                "body_type": "car",
                "modelyear": 2019,
                "primaryFuelType": "Electricity",
                "secondaryFuelType": "Gasoline",
            }
        ]
    )

    result = _attach_passenger_fastsim_templates(
        passenger_car_vehicle_types=passenger_rows,
        source_vehicle_types=source_vehicle_types,
        vehicle_type_mapping=vehicle_type_mapping,
    )

    assert result.loc[0, "vehicleTypeId"] == "Car_Hybrid_2004to2014"
    assert result.loc[0, "emfacFuel"] == ""
    assert result.loc[0, "primaryVehicleEnergyFile"] == "fuel/sample.csv"


def test_attach_passenger_fastsim_templates_uses_mapped_passenger_bodytype() -> None:
    passenger_rows = pd.DataFrame(
        [
            {
                "vehicleTypeId": "2004to2014LDT2Gas--Pickup_Conv_2004to2014",
                "oldVehicleTypeId": "Pickup_Conv_2004to2014",
                "sampleProbabilityWithinCategory": "0.100000",
                "sampleProbabilityString": "income | 0-50:0.100000",
                "adopt_fuel": "conv",
                "emfacId": "2004to2014LDT2Gas",
                "emfacVehicleCategory": "LDT2",
                "emfacFuel": "Gas",
                "emfacResolvedModelYear": "2004to2014",
                "bodytype": "pickup",
                "passenger_bodytype_norm": "car",
                "emfacModelYearGroup": "2004to2014",
                "modelyear": 2010,
                "primaryFuelType": "gasoline",
                "secondaryFuelType": "",
                "primaryVehicleEnergyFile": "fuel/original.csv",
                "secondaryVehicleEnergyFile": "",
            }
        ]
    )
    source_vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": "2015Template",
                "primaryFuelType": "gasoline",
                "secondaryFuelType": "",
                "primaryVehicleEnergyFile": "fuel/template.csv",
                "secondaryVehicleEnergyFile": "",
            }
        ]
    )
    vehicle_type_mapping = pd.DataFrame(
        [
            {
                "vehicleTypeId": "2015Template",
                "body_type": "car",
                "modelyear": 2015,
                "primaryFuelType": "gasoline",
                "secondaryFuelType": "",
            }
        ]
    )

    result = _attach_passenger_fastsim_templates(
        passenger_car_vehicle_types=passenger_rows,
        source_vehicle_types=source_vehicle_types,
        vehicle_type_mapping=vehicle_type_mapping,
    )

    assert result.loc[0, "vehicleTypeId"] == "2004to2014LDT2Gas--Pickup_Conv_2004to2014"
    assert result.loc[0, "primaryVehicleEnergyFile"] == "fuel/template.csv"


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
    passenger_car_vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": "2004to2014LDAGas--Car_Conv_2004to2014",
                "oldVehicleTypeId": "Car_Conv_2004to2014",
                "sampleProbabilityString": "income | 100-200:1.000000",
            }
        ]
    )

    result = _sample_passenger_vehicle_type_ids_for_vehicles(
        vehicles=vehicles,
        passenger_car_vehicle_types=passenger_car_vehicle_types,
        households=households,
        income_bins=[0, 30, 60, 100, 200, 9999],
        model_year_groups={
            "light_duty": [
                {"max_year": 2003},
                {"min_year": 2004, "max_year": 2014},
                {"min_year": 2015},
            ],
            "medium_heavy_duty": [{"max_year": 2002}, {"min_year": 2003}],
        },
        seed=0,
    )

    assert result.loc[0, "vehicleTypeId"] == "2004to2014LDAGas--Car_Conv_2004to2014"
