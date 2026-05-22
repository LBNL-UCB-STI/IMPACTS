from __future__ import annotations

from pathlib import Path

import pytest

from impacts.config.settings import _ingest_fleet_sources
from impacts.config.settings import _normalize_activities_inputs
from impacts.config.settings import _normalize_alias_mapping
from impacts.config.settings import _normalize_model_spec_path


def _write_required_model_file(model_file: Path) -> None:
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
                "          fleet_population_prior: 1.0",
                "          naics_sector: 1.0",
                "          payload_mass: 1.0",
                "          port_location: 1.0",
                "      evidence:",
                "        port_location:",
                  "          - zone_codes:",
                  "              - '060019819001'",
                  "            vehicle_category:",
                  "              - T7 POAK Class 8",
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
                "        Class12aVocational: [LDA, LDT1, LDT2]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "    passenger:",
                "      body_types:",
                "        car: [LDA]",
                "      fuel_types:",
                "        gasoline: [Gas]",
                "        electricity+gasoline: [Phe]",
                "      vehicle_categories:",
                "        Car: [LDA]",
            ]
        ),
        encoding="utf-8",
    )


def test_normalize_model_spec_path_requires_expected_model_file(tmp_path: Path) -> None:
    model_file = tmp_path / "fleet_assignment.yaml"

    with pytest.raises(FileNotFoundError, match="fleet_assignment.yaml"):
        _normalize_model_spec_path(str(model_file), path_label="vehicle_type_assignment.model_file")


def test_normalize_activities_inputs_preserves_residential_link_road_category_map(tmp_path: Path) -> None:
    dust_dir = tmp_path / "dust"
    emissions_dir = tmp_path / "emissions"
    dust_dir.mkdir()
    emissions_dir.mkdir()
    for name in ("rainy_days.csv", "silt_loading.csv"):
        (dust_dir / name).write_text("stub\n", encoding="utf-8")
    for name in ("emission.csv",):
        (emissions_dir / name).write_text("stub\n", encoding="utf-8")

    normalized = _normalize_activities_inputs(
        {
            "project_analysis": {
                "paved_road_dust": {
                    "folder": str(dust_dir),
                    "road_category_map": {
                        "residential": "Local Urban",
                        "residential_link": "Local Urban",
                    },
                }
            },
            "emissions_inventory": {
                "inventory_folder": str(emissions_dir),
            },
        }
    )

    assert normalized["road_category_map"]["residential"] == "Local Urban"
    assert normalized["road_category_map"]["residential_link"] == "Local Urban"


def test_normalize_model_spec_path_requires_expected_evidence_sources(tmp_path: Path) -> None:
    model_file = tmp_path / "fleet_assignment.yaml"
    _write_required_model_file(model_file)
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
                "          fleet_population_prior: 1.0",
                "          naics_sector: 1.0",
                "          payload_mass: 1.0",
                "          port_location: 1.0",
                "      evidence:",
                "        port_location:",
                "          - zone_codes:",
                "              - '060019819001'",
                "            vehicle_category:",
                "              - T7 POAK Class 8",
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
                "        Class12aVocational: [LDA, LDT1, LDT2]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "    passenger:",
                "      body_types:",
                "        car: [LDA]",
                "      fuel_types:",
                "        gasoline: [Gas]",
                "        electricity+gasoline: [Phe]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models.freight_bayesian_dag.evidence.naics_sector"):
        _normalize_model_spec_path(str(model_file), path_label="vehicle_type_assignment.model_file")


def test_normalize_model_spec_path_requires_embedded_rows_to_exist(tmp_path: Path) -> None:
    model_file = tmp_path / "fleet_assignment.yaml"
    _write_required_model_file(model_file)
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
                "          fleet_population_prior: 1.0",
                "          naics_sector: 1.0",
                "          payload_mass: 1.0",
                "          port_location: 1.0",
                "      evidence:",
                "        naics_sector:",
                "          - naics_code_2: ['11']",
                "            vehicle_category:",
                "              - T7 Tractor Class 8",
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
                "        Class12aVocational: [LDA, LDT1, LDT2]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "    passenger:",
                "      body_types:",
                "        car: [LDA]",
                "      fuel_types:",
                "        gasoline: [Gas]",
                "        electricity+gasoline: [Phe]",
                "      vehicle_categories:",
                "        Car: [LDA]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models.freight_bayesian_dag.evidence.port_location"):
        _normalize_model_spec_path(str(model_file), path_label="vehicle_type_assignment.model_file")


def test_ingest_fleet_sources_requires_likelihood_floor_in_model_scoring(tmp_path: Path) -> None:
    model_file = tmp_path / "fleet_assignment.yaml"
    model_file.write_text(
        "\n".join(
            [
                "fleet_assignment:",
                "  models:",
                "    freight_bayesian_dag:",
                "      scoring:",
                "        weights:",
                "          fleet_vmt_prior: 1.0",
                "          fleet_population_prior: 1.0",
                "          naics_sector: 1.0",
                "          payload_mass: 1.0",
                "          port_location: 1.0",
                "      evidence:",
                "        naics_sector:",
                "          - naics_code_2: ['11']",
                "            vehicle_category:",
                "              - T7 Tractor Class 8",
                "        port_location:",
                "          - zone_codes:",
                "              - '060019819001'",
                "            vehicle_category:",
                "              - T7 POAK Class 8",
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
                "        Class12aVocational: [LDA, LDT1, LDT2]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "      naics_sector:",
                "        - naics_code_2: ['11']",
                "          vehicle_category:",
                "            - T7 Tractor Class 8",
                "      port_location:",
                "        - zone_codes:",
                "            - '060019819001'",
                "          vehicle_category:",
                "            - T7 POAK Class 8",
                "    passenger:",
                "      body_types:",
                "        car: [LDA]",
                "      fuel_types:",
                "        gasoline: [Gas]",
                "        electricity+gasoline: [Phe]",
                "      vehicle_categories:",
                "        Car: [LDA]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="freight_bayesian_dag.scoring.likelihood_floor"):
        _ingest_fleet_sources(
            {
                "vehicle_type_assignment_model_settings": str(model_file),
                "activities": {
                    "region_label": "SF",
                    "calendar_year": 2018,
                    "outputs": str(tmp_path / "out"),
                    "model_year_groups": {
                        "light_duty": [],
                        "medium_heavy_duty": [],
                    },
                    "project_analysis": {},
                    "emissions_inventory": {},
                },
            }
        )


def test_ingest_fleet_sources_requires_scoring_weights_in_model_scoring(tmp_path: Path) -> None:
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
                "          fleet_population_prior: 1.0",
                "      evidence:",
                "        naics_sector:",
                "          - naics_code_2: ['11']",
                "            vehicle_category:",
                "              - T7 Tractor Class 8",
                "        port_location:",
                "          - zone_codes:",
                "              - '060019819001'",
                "            vehicle_category:",
                "              - T7 POAK Class 8",
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
                "        Class12aVocational: [LDA, LDT1, LDT2]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "      naics_sector:",
                "        - naics_code_2: ['11']",
                "          vehicle_category:",
                "            - T7 Tractor Class 8",
                "      port_location:",
                "        - zone_codes:",
                "            - '060019819001'",
                "          vehicle_category:",
                "            - T7 POAK Class 8",
                "    passenger:",
                "      body_types:",
                "        car: [LDA]",
                "      fuel_types:",
                "        gasoline: [Gas]",
                "        electricity+gasoline: [Phe]",
                "      vehicle_categories:",
                "        Car: [LDA]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="freight_bayesian_dag.scoring.weights"):
        _ingest_fleet_sources(
            {
                "vehicle_type_assignment_model_settings": str(model_file),
                "activities": {
                    "region_label": "SF",
                    "calendar_year": 2018,
                    "outputs": str(tmp_path / "out"),
                    "model_year_groups": {
                        "light_duty": [],
                        "medium_heavy_duty": [],
                    },
                    "project_analysis": {},
                    "emissions_inventory": {},
                },
            }
        )


def test_ingest_fleet_sources_disables_payload_mass_when_freight_evidence_is_absent(tmp_path: Path) -> None:
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
                "          fleet_population_prior: 1.0",
                "          naics_sector: 1.0",
                "          payload_mass: 1.0",
                "          port_location: 1.0",
                "      evidence:",
                "        naics_sector:",
                "          - naics_code_2: ['11']",
                "            vehicle_category:",
                "              - T7 Tractor Class 8",
                "        port_location:",
                "          - zone_codes:",
                "              - '060019819001'",
                "            vehicle_category:",
                "              - T7 POAK Class 8",
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
                "        Class12aVocational: [LDA, LDT1, LDT2]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "      naics_sector:",
                "        - naics_code_2: ['11']",
                "          vehicle_category:",
                "            - T7 Tractor Class 8",
                "      port_location:",
                "        - zone_codes:",
                "            - '060019819001'",
                "          vehicle_category:",
                "            - T7 POAK Class 8",
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

    config = _ingest_fleet_sources(
        {
            "vehicle_type_assignment_model_settings": str(model_file),
            "activities": {
                "region_label": "SF",
                "calendar_year": 2018,
                "outputs": str(tmp_path / "out"),
                "model_year_groups": {
                    "light_duty": [],
                    "medium_heavy_duty": [],
                },
                "project_analysis": {},
                "emissions_inventory": {},
            },
        }
    )

    assert config["freight_bayesian_dag"]["payload_mass_enabled"] is False


def test_ingest_fleet_sources_disables_income_when_passenger_evidence_is_absent(tmp_path: Path) -> None:
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
                "          fleet_population_prior: 1.0",
                "          naics_sector: 1.0",
                "          payload_mass: 1.0",
                "          port_location: 1.0",
                "      evidence:",
                "        naics_sector:",
                "          - naics_code_2: ['11']",
                "            vehicle_category:",
                "              - T7 Tractor Class 8",
                "        port_location:",
                "          - zone_codes:",
                "              - '060019819001'",
                "            vehicle_category:",
                "              - T7 POAK Class 8",
                "    passenger_bayesian_dag:",
                "      scoring:",
                "        likelihood_floor: 0.001",
                "        weights:",
                "          fleet_vmt_prior: 1.0",
                "          fleet_population_prior: 1.0",
                "          income: 1.0",
                "      evidence: {}",
                "  mappings:",
                "    fuel_consumption:",
                "      - fastsim_id: 2015_gasoline_Chrysler_200",
                "        vehicle_categories: [LDA]",
                "        fuel_types: [Gas]",
                "    freight:",
                "      vehicle_categories:",
                "        Class12aVocational: [LDA, LDT1, LDT2]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "      naics_sector:",
                "        - naics_code_2: ['11']",
                "          vehicle_category:",
                "            - T7 Tractor Class 8",
                "      port_location:",
                "        - zone_codes:",
                "            - '060019819001'",
                "          vehicle_category:",
                "            - T7 POAK Class 8",
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

    config = _ingest_fleet_sources(
        {
            "vehicle_type_assignment_model_settings": str(model_file),
            "activities": {
                "region_label": "SF",
                "calendar_year": 2018,
                "outputs": str(tmp_path / "out"),
                "model_year_groups": {
                    "light_duty": [],
                    "medium_heavy_duty": [],
                },
                "project_analysis": {},
                "emissions_inventory": {},
            },
        }
    )

    assert config["passenger_bayesian_dag"]["income_enabled"] is False


def test_normalize_alias_mapping_supports_quoted_list_emissions_inventory_fuel_map() -> None:
    assert _normalize_alias_mapping(
        {
            "Dsl": ["Diesel"],
            "Elec": ["Electricity"],
            "Gas": ["Gasoline"],
            "NG": ["Natural Gas"],
            "Phe": ["Plug-in Hybrid"],
        }
    ) == {
        "Diesel": "Dsl",
        "Electricity": "Elec",
        "Gasoline": "Gas",
        "Natural Gas": "NG",
        "Plug-in Hybrid": "Phe",
    }
