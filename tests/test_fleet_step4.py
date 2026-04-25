from __future__ import annotations

import math
import csv
from pathlib import Path

import pandas as pd

from impacts.emfac.fleet.step4_map_emfac_frism import _build_freight_bayesian_log_score
from impacts.emfac.fleet.step4_map_emfac_frism import _build_freight_naics_sector_weight_lookup
from impacts.emfac.fleet.step4_map_emfac_frism import _build_payload_mass_thresholds
from impacts.emfac.fleet.step4_map_emfac_frism import _filter_required_port_classes
from impacts.emfac.fleet.step4_map_emfac_frism import _build_tour_port_weight_lookup
from impacts.emfac.fleet.step4_map_emfac_frism import _attach_freight_fuel_consumption_templates
from impacts.emfac.fleet.step4_map_emfac_frism import _payload_mass_gvwr_likelihood
from impacts.emfac.fleet.step4_map_emfac_frism import _load_configured_port_classes
from impacts.emfac.fleet.step4_map_emfac_frism import _load_vehicle_type_assignment_table
from impacts.emfac.fleet.step4_map_emfac_frism import _load_port_zone_mapping
from impacts.emfac.fleet.step4_map_emfac_frism import _port_category_weight
from impacts.emfac.fleet.step4_map_emfac_frism import _normalize_zone_id


def _write_model_file(tmp_path: Path, *, port_rows: list[str] | None = None) -> dict[str, dict[str, str]]:
    model_file = tmp_path / "fleet_assignment.yaml"
    port_entries = []
    for line in port_rows or ["zone,port_name,emfac_vehicle_category"]:
        if line == "zone,port_name,emfac_vehicle_category" or line == "zone,emfac_vehicle_category,port_name":
            continue
        parts = next(csv.reader([line]))
        if len(parts) == 3:
            parts = [part.strip() for part in parts]
            if parts[1].startswith("T7 "):
                zone, emfac_vehicle_category, port_name = parts
            else:
                zone, port_name, emfac_vehicle_category = parts
            port_entries.append(
                f"    - vehicle_category:\n        - \"{emfac_vehicle_category}\"\n      zone_codes:\n        - '{zone}'\n      label: \"{port_name}\""
            )
    port_rows_yaml = "\n".join(port_entries) if port_entries else "    []"
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
                "          payload_mass: 1.0",
                "          port_location: 1.0",
                "      evidence:",
                "        payload_mass:",
                "          source: gvwr_lbs",
                "          unit: lbs",
                "          overload_penalty_power: 2.0",
                "    passenger_bayesian_dag:",
                "      scoring:",
                "        weights:",
                "          bodytype: 1.0",
                "          fuel: 1.0",
                "          emfac_vmt: 0.0",
                "      evidence: {}",
                "  mappings:",
                "    freight:",
                "      vehicle_categories:",
                "        Class12aVocational: [LDA, LDT1, LDT2]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "      naics_sector:",
                "        - naics_code_2: '11'",
                "          vehicle_category:",
                "            - T7 Tractor Class 8",
                "      port_location:",
                port_rows_yaml.replace("    - ", "      - ").replace("      zone_codes:", "        zone_codes:").replace("      label:", "        label:").replace("      vehicle_category:", "        vehicle_category:").replace("        - ", "          - "),
                "    passenger:",
                "      body_types:",
                "        car: [LDA]",
                "      fuel_types:",
                "        conv: [Gas]",
                "      vehicle_categories:",
                "        Car: [LDA]",
            ]
        ),
        encoding="utf-8",
    )
    return {"vehicle_type_assignment": {"model_file": str(model_file)}}


def test_build_freight_bayesian_log_score_uses_grouped_geometric_means() -> None:
    matched = pd.DataFrame(
        [
            {
                "fleetShare": 0.25,
                "naicsSectorLikelihood": 0.25,
                "payloadMassLikelihood": 1.0,
                "portLikelihood": 0.25,
            }
        ]
    )

    score = _build_freight_bayesian_log_score(
        matched=matched,
        branch_weights={"fleet_vmt_prior": 1.0, "naics_sector": 2.0, "payload_mass": 2.0, "port_location": 3.0},
    )

    expected = (
        (1.0 / 8.0) * math.log(0.25)
        + (2.0 / 8.0) * math.log(0.25)
        + (2.0 / 8.0) * math.log(1.0)
        + (3.0 / 8.0) * math.log(0.25)
    )
    assert score.iloc[0] == expected


def test_normalize_zone_id_preserves_string_ids_and_strips_decimal_suffix() -> None:
    assert _normalize_zone_id("060014017001") == "060014017001"
    assert _normalize_zone_id("60014014003") == "60014014003"
    assert _normalize_zone_id("60014014003.0") == "60014014003"
    assert _normalize_zone_id(60014014003.0) == "60014014003"
    assert _normalize_zone_id("") == ""


def test_load_vehicle_type_assignment_table_expands_naics_code_lists(tmp_path: Path) -> None:
    model_file = tmp_path / "fleet_assignment.yaml"
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
                "          payload_mass: 1.0",
                "          port_location: 1.0",
                "      evidence: {}",
                "    passenger_bayesian_dag:",
                "      scoring: {}",
                "      evidence: {}",
                "  mappings:",
                "    freight:",
                "      vehicle_categories:",
                "        Class12aVocational: [LDA, LDT1, LDT2]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "      naics_sector:",
                "        - naics_code_2:",
                "            - '31'",
                "            - '32'",
                "          vehicle_category:",
                "            - T7 Tractor Class 8",
                "      port_location: []",
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
    config = {"vehicle_type_assignment": {"model_file": str(model_file)}}

    frame = _load_vehicle_type_assignment_table(config, "naics_sector")

    records = frame[["naics_code_2", "vehicleCategory"]].to_dict("records")
    assert records == [
        {"naics_code_2": "31", "vehicleCategory": "T7 Tractor Class 8"},
        {"naics_code_2": "32", "vehicleCategory": "T7 Tractor Class 8"},
    ]


def test_build_tour_port_weight_lookup_is_tour_specific() -> None:
    payload_profiles = pd.DataFrame(
        [
            {"tourId": "t1", "frismVehicleTypeId": "HdtDsl", "locationZone": "060014017001"},
            {"tourId": "t2", "frismVehicleTypeId": "HdtDsl", "locationZone": "060750101001"},
            {"tourId": "t3", "frismVehicleTypeId": "HdtDsl", "locationZone": "999999999999"},
        ]
    )
    port_zone_mapping = pd.DataFrame(
        [
            {"zone": "060014017001", "emfac_vehicle_category": "T7 POAK Class 8", "port_name": "Port of Oakland"},
            {"zone": "060750101001", "emfac_vehicle_category": "T7 Other Port Class 8", "port_name": "Port of San Francisco"},
        ]
    )

    lookup = _build_tour_port_weight_lookup(
        payload_profiles=payload_profiles,
        port_zone_mapping=port_zone_mapping,
    )

    assert lookup["t1"] == {"T7 POAK Class 8": 1.0}
    assert lookup["t2"] == {"T7 Other Port Class 8": 1.0}
    assert "t3" not in lookup


def test_build_freight_naics_sector_weight_lookup_uses_naics_sector_only() -> None:
    payload_profiles = pd.DataFrame(
        [
            {
                "tourId": "t1",
                "frismVehicleTypeId": "HdtDsl",
                "sequenceRank": 1,
                "activityType": "loading",
                "sellerNAICS": "111100",
                "buyerNAICS": "111200",
                "payloadType": "bulk",
                "weightInKg": 1000.0,
                "locationZone": "",
            },
            {
                "tourId": "t2",
                "frismVehicleTypeId": "HdtDsl",
                "sequenceRank": 1,
                "activityType": "loading",
                "sellerNAICS": "111300",
                "buyerNAICS": "111400",
                "payloadType": "bulk",
                "weightInKg": 1500.0,
                "locationZone": "",
            },
            {
                "tourId": "t3",
                "frismVehicleTypeId": "HdtDsl",
                "sequenceRank": 1,
                "activityType": "loading",
                "sellerNAICS": "311100",
                "buyerNAICS": "311200",
                "payloadType": "mfr_goods",
                "weightInKg": 800.0,
                "locationZone": "",
            },
        ]
    )
    naics_sector_mapping = pd.DataFrame(
        [
            {"naics_source": "all", "naics_code_2": "11", "vehicleCategory": "T7 Tractor Class 8"},
            {"naics_source": "all", "naics_code_2": "11", "vehicleCategory": "T7 Single Dump Class 8"},
            {"naics_source": "all", "naics_code_2": "31", "vehicleCategory": "T7 Single Other Class 8"},
        ]
    )

    lookup = _build_freight_naics_sector_weight_lookup(
        payload_profiles=payload_profiles,
        sector_mapping=naics_sector_mapping,
    )

    assert "HdtDsl" in lookup
    assert set(lookup["HdtDsl"].keys()) == {
        "T7 Tractor Class 8",
        "T7 Single Dump Class 8",
        "T7 Single Other Class 8",
    }
    assert lookup["HdtDsl"]["T7 Tractor Class 8"] == lookup["HdtDsl"]["T7 Single Other Class 8"]


def test_build_payload_mass_thresholds_uses_peak_cumulative_onboard_payload() -> None:
    payload_profiles = pd.DataFrame(
        [
            {"tourId": "t1", "frismVehicleTypeId": "HdtDsl", "sequenceRank": 1, "activityType": "loading", "weightInKg": 2000.0},
            {"tourId": "t1", "frismVehicleTypeId": "HdtDsl", "sequenceRank": 2, "activityType": "loading", "weightInKg": 3000.0},
            {"tourId": "t1", "frismVehicleTypeId": "HdtDsl", "sequenceRank": 3, "activityType": "unloading", "weightInKg": 1000.0},
            {"tourId": "t2", "frismVehicleTypeId": "HdtDsl", "sequenceRank": 1, "activityType": "unloading", "weightInKg": 500.0},
            {"tourId": "t2", "frismVehicleTypeId": "HdtDsl", "sequenceRank": 2, "activityType": "loading", "weightInKg": 2500.0},
            {"tourId": "t2", "frismVehicleTypeId": "HdtDsl", "sequenceRank": 3, "activityType": "loading", "weightInKg": 500.0},
        ]
    )

    thresholds = _build_payload_mass_thresholds(payload_profiles)

    # Tour peaks are 5000 kg for t1 and 3000 kg for t2 after shifting the
    # second tour upward so onboard mass never goes negative.
    assert thresholds["HdtDsl"] == (4000.0, 4800.0)


def test_payload_mass_gvwr_likelihood_prefers_closer_supported_gvwr() -> None:
    gvwr_kg_lookup = {
        "MDV": 4000.0,
        "T6 Instate Delivery Class 7": 8000.0,
        "T7 Tractor Class 8": 16000.0,
    }

    mdv = _payload_mass_gvwr_likelihood(
        vehicle_category="MDV",
        observed_peak_payload_kg=6000.0,
        gvwr_kg_lookup=gvwr_kg_lookup,
        likelihood_floor=0.01,
        overload_penalty_power=2.0,
    )
    t6 = _payload_mass_gvwr_likelihood(
        vehicle_category="T6 Instate Delivery Class 7",
        observed_peak_payload_kg=6000.0,
        gvwr_kg_lookup=gvwr_kg_lookup,
        likelihood_floor=0.01,
        overload_penalty_power=2.0,
    )
    t7 = _payload_mass_gvwr_likelihood(
        vehicle_category="T7 Tractor Class 8",
        observed_peak_payload_kg=6000.0,
        gvwr_kg_lookup=gvwr_kg_lookup,
        likelihood_floor=0.01,
        overload_penalty_power=2.0,
    )

    assert mdv < t6
    assert t7 < t6


def test_payload_mass_gvwr_likelihood_uses_configured_penalty_power() -> None:
    low_penalty = _payload_mass_gvwr_likelihood(
        vehicle_category="MDV",
        observed_peak_payload_kg=6000.0,
        gvwr_kg_lookup={"MDV": 4000.0},
        likelihood_floor=0.01,
        overload_penalty_power=1.0,
    )
    high_penalty = _payload_mass_gvwr_likelihood(
        vehicle_category="MDV",
        observed_peak_payload_kg=6000.0,
        gvwr_kg_lookup={"MDV": 4000.0},
        likelihood_floor=0.01,
        overload_penalty_power=3.0,
    )

    assert high_penalty < low_penalty


def test_load_port_zone_mapping_reads_direct_emfac_mapping(tmp_path) -> None:
    config = _write_model_file(
        tmp_path,
        port_rows=[
            "zone,port_name,emfac_vehicle_category",
            '060019819001,"Oakland, CA","T7 POAK Class 8"',
            '060759809001,"San Francisco, CA","T7 Other Port Class 8"',
        ],
    )

    mapping = _load_port_zone_mapping(config)

    records = mapping.to_dict("records")
    assert {"zone": "060019819001", "emfac_vehicle_category": "T7 POAK Class 8", "port_name": ""} in records
    assert {"zone": "60019819001", "emfac_vehicle_category": "T7 POAK Class 8", "port_name": ""} in records
    assert {
        "zone": "060759809001",
        "emfac_vehicle_category": "T7 Other Port Class 8",
        "port_name": "",
    } in records
    assert {
        "zone": "60759809001",
        "emfac_vehicle_category": "T7 Other Port Class 8",
        "port_name": "",
    } in records


def test_port_category_weight_prefers_matching_port_for_tractors() -> None:
    configured_port_classes = {"T7 POAK Class 8", "T7 Other Port Class 8"}
    poak_weights = {"T7 POAK Class 8": 1.0}
    other_port_weights = {"T7 Other Port Class 8": 1.0}

    poak_match = _port_category_weight(
        vehicle_category="T7 POAK Class 8",
        port_weights=poak_weights,
        configured_port_classes=configured_port_classes,
        likelihood_floor=0.01,
    )
    poak_base = _port_category_weight(
        vehicle_category="T7 Tractor Class 8",
        port_weights=poak_weights,
        configured_port_classes=configured_port_classes,
        likelihood_floor=0.01,
    )
    other_port_match = _port_category_weight(
        vehicle_category="T7 Other Port Class 8",
        port_weights=other_port_weights,
        configured_port_classes=configured_port_classes,
        likelihood_floor=0.01,
    )
    other_port_base = _port_category_weight(
        vehicle_category="T7 Tractor Class 8",
        port_weights=other_port_weights,
        configured_port_classes=configured_port_classes,
        likelihood_floor=0.01,
    )
    assert poak_match == 1.0
    assert other_port_match == 1.0
    assert 0.0 <= poak_base <= 1.0
    assert 0.0 <= other_port_base <= 1.0
    assert poak_match > poak_base
    assert other_port_match > other_port_base
    assert poak_base == 0.01
    assert other_port_base == 0.01
    assert _port_category_weight(
        vehicle_category="T7 POAK Class 8",
        port_weights=poak_weights,
        configured_port_classes=configured_port_classes,
        likelihood_floor=0.01,
    ) == 1.0


def test_load_configured_port_classes_reads_mapping_file(tmp_path) -> None:
    config = _write_model_file(
        tmp_path,
        port_rows=[
            "zone,emfac_vehicle_category,port_name",
            '060019819001,"T7 POAK Class 8","Oakland, CA"',
            '060759809001,"T7 Other Port Class 8","San Francisco, CA"',
        ],
    )

    loaded = _load_configured_port_classes(config)

    assert loaded == {"T7 POAK Class 8", "T7 Other Port Class 8"}


def test_filter_required_port_classes_applies_port_class_gating_when_configured() -> None:
    matched = pd.DataFrame(
        [
            {"vehicleCategory": "T7 POAK Class 8"},
            {"vehicleCategory": "T7 Other Port Class 8"},
            {"vehicleCategory": "T7 Tractor Class 8"},
        ]
    )

    no_port = _filter_required_port_classes(
        matched=matched,
        port_weights={},
        configured_port_classes={"T7 POAK Class 8", "T7 Other Port Class 8"},
    )
    assert no_port["vehicleCategory"].tolist() == ["T7 Tractor Class 8"]

    poak = _filter_required_port_classes(
        matched=matched,
        port_weights={"T7 POAK Class 8": 1.0},
        configured_port_classes={"T7 POAK Class 8", "T7 Other Port Class 8"},
    )
    assert poak["vehicleCategory"].tolist() == ["T7 POAK Class 8", "T7 Tractor Class 8"]


def test_attach_freight_fuel_consumption_templates_keeps_baseline_values_when_mapping_missing(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "impacts.emfac.fleet.step4_map_emfac_frism.build_fuel_consumption_emfac_assignment_catalog",
        lambda model_file, breakdown_path: pd.DataFrame(
            [{"emfac_vehicle_category": "T7 Tractor Class 8", "emfac_fuel": "Dsl", "fastsim_relative_path": "foo.csv"}]
        ),
    )
    mapped = pd.DataFrame(
        [
            {
                "vehicleTypeId": "2004to2014LDADsl--Ld1Dsl",
                "frismVehicleTypeId": "Ld1Dsl",
                "emfacId": "2004to2014LDADsl",
                "emfacVehicleCategory": "LDA",
                "emfacFuel": "Dsl",
                "emfacResolvedModelYear": "2004-2014",
                "vehicleCategory": "Class12aVocational",
                "primaryFuelType": "diesel",
                "primaryFuelConsumptionInJoulePerMeter": 6007.565107,
                "primaryFuelCapacityInJoule": 1.2e16,
                "primaryVehicleEnergyFile": "",
                "secondaryVehicleEnergyFile": "",
                "sampleProbabilityWithinCategory": 1.0,
                "sampleProbabilityString": "",
            }
        ]
    )
    source = pd.DataFrame(
        [
            {
                "vehicleTypeId": "Ld1Dsl",
                "primaryVehicleEnergyFile": "",
                "secondaryVehicleEnergyFile": "",
            }
        ]
    )
    config = {
        "vehicle_type_assignment": {"model_file": "fleet_assignment.yaml"},
        "beam": {"fuel_consumption_catalog": "fuel_catalog.csv"},
    }

    result = _attach_freight_fuel_consumption_templates(
        mapped_freight_vehicle_types=mapped,
        source_vehicle_types=source,
        config=config,
    )

    assert result.loc[0, "vehicleTypeId"] == "unmapped--2004to2014LDADsl--Ld1Dsl"
    assert result.loc[0, "fuelConsumptionId"] == ""
    assert result.loc[0, "primaryFuelType"] == "diesel"
    assert result.loc[0, "primaryFuelConsumptionInJoulePerMeter"] == 6007.565107
    assert result.loc[0, "primaryFuelCapacityInJoule"] == 1.2e16
    assert result.loc[0, "primaryVehicleEnergyFile"] == ""
    assert result.loc[0, "secondaryVehicleEnergyFile"] == ""
    assert "WARNING: Freight Step 4.3 leaving fuel-consumption template fields empty" in capsys.readouterr().out


def test_attach_freight_fuel_consumption_templates_rebuilds_vehicle_type_id_with_fuel_consumption_id() -> None:
    mapped = pd.DataFrame(
        [
            {
                "vehicleTypeId": "2004to2014LDADsl--Ld1Dsl",
                "frismVehicleTypeId": "Ld1Dsl",
                "emfacId": "2004to2014LDADsl",
                "emfacVehicleCategory": "LDA",
                "emfacFuel": "Dsl",
                "emfacResolvedModelYear": "2004-2014",
                "sampleProbabilityWithinCategory": 1.0,
                "sampleProbabilityString": "",
            }
        ]
    )
    source = pd.DataFrame(
        [
            {
                "vehicleTypeId": "2015_gasoline_Chrysler_200",
                "primaryVehicleEnergyFile": "test.csv",
                "secondaryVehicleEnergyFile": "",
            }
        ]
    )
    config = {
        "vehicle_type_assignment": {"model_file": "fleet_assignment.yaml"},
        "beam": {"fuel_consumption_catalog": "fuel_catalog.csv"},
    }

    original_catalog = _attach_freight_fuel_consumption_templates.__globals__["build_fuel_consumption_emfac_assignment_catalog"]
    try:
        _attach_freight_fuel_consumption_templates.__globals__["build_fuel_consumption_emfac_assignment_catalog"] = (
            lambda model_file, breakdown_path: pd.DataFrame(
                [
                    {
                        "emfac_vehicle_category": "LDA",
                        "emfac_fuel": "Dsl",
                        "fastsim_relative_path": "test.csv",
                    }
                ]
            )
        )
        result = _attach_freight_fuel_consumption_templates(
            mapped_freight_vehicle_types=mapped,
            source_vehicle_types=source,
            config=config,
        )
    finally:
        _attach_freight_fuel_consumption_templates.__globals__["build_fuel_consumption_emfac_assignment_catalog"] = original_catalog

    assert result.loc[0, "fuelConsumptionId"] == "2015_gasoline_Chrysler_200"
    assert result.loc[0, "vehicleTypeId"] == "2015gasolineChrysler200--2004to2014LDADsl--Ld1Dsl"


def test_attach_freight_fuel_consumption_templates_still_errors_without_baseline_values(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "impacts.emfac.fleet.step4_map_emfac_frism.build_fuel_consumption_emfac_assignment_catalog",
        lambda model_file, breakdown_path: pd.DataFrame(columns=["emfac_vehicle_category", "emfac_fuel", "fastsim_relative_path"]),
    )
    mapped = pd.DataFrame(
        [
            {
                "vehicleTypeId": "2004to2014LDADsl--Ld1Dsl",
                "frismVehicleTypeId": "Ld1Dsl",
                "emfacId": "2004to2014LDADsl",
                "emfacVehicleCategory": "LDA",
                "emfacFuel": "Dsl",
                "emfacResolvedModelYear": "2004-2014",
                "vehicleCategory": "Class12aVocational",
                "primaryFuelType": "",
                "primaryFuelConsumptionInJoulePerMeter": None,
                "primaryFuelCapacityInJoule": None,
                "primaryVehicleEnergyFile": "",
                "secondaryVehicleEnergyFile": "",
                "sampleProbabilityWithinCategory": 1.0,
                "sampleProbabilityString": "",
            }
        ]
    )
    source = pd.DataFrame([{"vehicleTypeId": "Ld1Dsl"}])
    config = {
        "vehicle_type_assignment": {"model_file": "fleet_assignment.yaml"},
        "beam": {"fuel_consumption_catalog": "fuel_catalog.csv"},
    }

    try:
        _attach_freight_fuel_consumption_templates(
            mapped_freight_vehicle_types=mapped,
            source_vehicle_types=source,
            config=config,
        )
    except ValueError as error:
        assert "No fuel-consumption freight assignment matched EMFAC-assigned class/fuel" in str(error)
    else:
        raise AssertionError("Expected ValueError when no freight baseline fuel-consumption values are present")
