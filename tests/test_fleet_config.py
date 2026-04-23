from __future__ import annotations

from pathlib import Path

import pytest

from impacts.emfac.config import _ingest_fleet_sources
from impacts.emfac.config import _normalize_model_spec_path


def _write_required_model_file(model_file: Path) -> None:
    model_file.write_text(
        "\n".join(
            [
                "models:",
                "  freight_bayesian_dag:",
                "    scoring:",
                "      likelihood_floor: 0.01",
                "      weights:",
                "        prior_vmt_share: 1.0",
                "        naics_sector: 1.0",
                "        port_location: 1.0",
                "    evidence:",
                "      vehicle_categories:",
                "        Class12aVocational: [LDA, LDT1, LDT2]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "      naics_sector: []",
                "      port_location: []",
                "  passenger_bayesian_dag:",
                "    scoring:",
                "      weights:",
                "        bodytype: 1.0",
                "        fuel: 1.0",
                "        emfac_vmt: 0.0",
                "    evidence:",
                "      atlas_vehicle_categories:",
                "        car: [LDA]",
                "      atlas_fuel_types:",
                "        conv: [Gas]",
                "      beam_fuel_types:",
                "        gasoline: [Gas]",
                "      beam_vehicle_categories:",
                "        Car: [LDA]",
                "  fastsim_assignment_model:",
                "    scoring:",
                "      weights:",
                "        year_distance: 1.0",
                "        affordability: 1.0",
                "    evidence:",
                "      assignments:",
                "      - fastsim_id: 2015_gasoline_Chrysler_200",
                "        vehicle_categories: [LDA]",
                "        fuel_types: [Gas]",
            ]
        ),
        encoding="utf-8",
    )


def test_normalize_model_spec_path_requires_expected_model_file(tmp_path: Path) -> None:
    model_file = tmp_path / "vehicle_type_assignment_model.yaml"

    with pytest.raises(FileNotFoundError, match="vehicle_type_assignment_model.yaml"):
        _normalize_model_spec_path(str(model_file), path_label="vehicle_type_assignment.model_file")


def test_normalize_model_spec_path_requires_expected_evidence_sources(tmp_path: Path) -> None:
    model_file = tmp_path / "vehicle_type_assignment_model.yaml"
    _write_required_model_file(model_file)
    model_file.write_text(
        "\n".join(
            [
                "models:",
                "  freight_bayesian_dag:",
                "    scoring:",
                "      likelihood_floor: 0.01",
                "      weights:",
                "        prior_vmt_share: 1.0",
                "    evidence:",
                "      vehicle_categories:",
                "        Class12aVocational: [LDA, LDT1, LDT2]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "      naics_sector: []",
                "      port_location:",
                "        - vehicle_category:",
                "            - T7 POAK Class 8",
                "          zone_codes:",
                "            - '060019819001'",
                "  passenger_bayesian_dag:",
                "    scoring:",
                "      weights:",
                "        bodytype: 1.0",
                "        fuel: 1.0",
                "        emfac_vmt: 0.0",
                "    evidence:",
                "      atlas_vehicle_categories:",
                "        car: [LDA]",
                "      atlas_fuel_types:",
                "        conv: [Gas]",
                "      beam_fuel_types:",
                "        gasoline: [Gas]",
                "  fastsim_assignment_model:",
                "    scoring:",
                "      weights:",
                "        year_distance: 1.0",
                "        affordability: 1.0",
                "    evidence:",
                "      assignments:",
                "      - fastsim_id: 2015_gasoline_Chrysler_200",
                "        vehicle_categories: [LDA]",
                "        fuel_types: [Gas]",
                "      beam_vehicle_categories:",
                "        Car: [LDA]",
                "  fastsim_assignment_model:",
                "    scoring:",
                "      weights:",
                "        year_distance: 1.0",
                "        affordability: 1.0",
                "    evidence:",
                "      assignments:",
                "      - fastsim_id: 2015_gasoline_Chrysler_200",
                "        vehicle_categories: [LDA]",
                "        fuel_types: [Gas]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="freight_bayesian_dag.evidence.naics_sector"):
        _normalize_model_spec_path(str(model_file), path_label="vehicle_type_assignment.model_file")


def test_normalize_model_spec_path_requires_embedded_rows_to_exist(tmp_path: Path) -> None:
    model_file = tmp_path / "vehicle_type_assignment_model.yaml"
    _write_required_model_file(model_file)
    model_file.write_text(
        "\n".join(
            [
                "models:",
                "  freight_bayesian_dag:",
                "    scoring:",
                "      likelihood_floor: 0.01",
                "      weights:",
                "        prior_vmt_share: 1.0",
                "    evidence:",
                "      vehicle_categories:",
                "        Class12aVocational: [LDA, LDT1, LDT2]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "      naics_sector:",
                "        - naics_code_2: ['11']",
                "          vehicle_category:",
                "            - T7 Tractor Class 8",
                "      port_location: []",
                "  passenger_bayesian_dag:",
                "    scoring:",
                "      weights:",
                "        bodytype: 1.0",
                "        fuel: 1.0",
                "        emfac_vmt: 0.0",
                "    evidence:",
                "      atlas_vehicle_categories:",
                "        car: [LDA]",
                "      atlas_fuel_types:",
                "        conv: [Gas]",
                "      beam_fuel_types:",
                "        gasoline: [Gas]",
                "      beam_vehicle_categories:",
                "        Car: [LDA]",
                "  fastsim_assignment_model:",
                "    scoring:",
                "      weights:",
                "        year_distance: 1.0",
                "        affordability: 1.0",
                "    evidence:",
                "      assignments:",
                "      - fastsim_id: 2015_gasoline_Chrysler_200",
                "        vehicle_categories: [LDA]",
                "        fuel_types: [Gas]",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="freight_bayesian_dag.evidence.port_location"):
        _normalize_model_spec_path(str(model_file), path_label="vehicle_type_assignment.model_file")


def test_ingest_fleet_sources_requires_likelihood_floor_in_model_scoring(tmp_path: Path) -> None:
    model_file = tmp_path / "vehicle_type_assignment_model.yaml"
    model_file.write_text(
        "\n".join(
            [
                "models:",
                "  freight_bayesian_dag:",
                "    scoring:",
                "      weights:",
                "        prior_vmt_share: 1.0",
                "        naics_sector: 1.0",
                "        port_location: 1.0",
                "    evidence:",
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
                "  passenger_bayesian_dag:",
                "    scoring:",
                "      weights:",
                "        bodytype: 1.0",
                "        fuel: 1.0",
                "        emfac_vmt: 0.0",
                "    evidence:",
                "      atlas_vehicle_categories:",
                "        car: [LDA]",
                "      atlas_fuel_types:",
                "        conv: [Gas]",
                "      beam_fuel_types:",
                "        gasoline: [Gas]",
                "      beam_vehicle_categories:",
                "        Car: [LDA]",
                "  fastsim_assignment_model:",
                "    scoring:",
                "      weights:",
                "        year_distance: 1.0",
                "        affordability: 1.0",
                "    evidence:",
                "      assignments:",
                "      - fastsim_id: 2015_gasoline_Chrysler_200",
                "        vehicle_categories: [LDA]",
                "        fuel_types: [Gas]",
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
    model_file = tmp_path / "vehicle_type_assignment_model.yaml"
    model_file.write_text(
        "\n".join(
            [
                "models:",
                "  freight_bayesian_dag:",
                "    scoring:",
                "      likelihood_floor: 0.01",
                "      weights:",
                "        prior_vmt_share: 1.0",
                "    evidence:",
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
                "  passenger_bayesian_dag:",
                "    scoring:",
                "      weights:",
                "        bodytype: 1.0",
                "        fuel: 1.0",
                "        emfac_vmt: 0.0",
                "    evidence:",
                "      atlas_vehicle_categories:",
                "        car: [LDA]",
                "      atlas_fuel_types:",
                "        conv: [Gas]",
                "      beam_fuel_types:",
                "        gasoline: [Gas]",
                "      beam_vehicle_categories:",
                "        Car: [LDA]",
                "  fastsim_assignment_model:",
                "    scoring:",
                "      weights:",
                "        year_distance: 1.0",
                "        affordability: 1.0",
                "    evidence:",
                "      assignments:",
                "      - fastsim_id: 2015_gasoline_Chrysler_200",
                "        vehicle_categories: [LDA]",
                "        fuel_types: [Gas]",
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
