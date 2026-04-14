from __future__ import annotations

from pathlib import Path

import pytest

from impacts.fleet.config import _normalize_model_spec_path


def _write_required_model_file(model_file: Path) -> None:
    model_file.write_text(
        "\n".join(
            [
                "model:",
                "  scoring:",
                "    likelihood_floor: 0.01",
                "    weights:",
                "      prior_vmt_share: 1.0",
                "      naics_sector: 1.0",
                "      port_location: 1.0",
                "  evidence:",
                "    naics_sector: []",
                "    port_location: []",
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
                "model:",
                "  scoring:",
                "    likelihood_floor: 0.01",
                "    weights:",
                "      prior_vmt_share: 1.0",
                "  evidence:",
                "    naics_sector: []",
                "    port_location:",
                "      - vehicle_category:",
                "          - T7 POAK Class 8",
                "        zone_codes:",
                "          - '060019819001'",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evidence.naics_sector"):
        _normalize_model_spec_path(str(model_file), path_label="vehicle_type_assignment.model_file")


def test_normalize_model_spec_path_requires_embedded_rows_to_exist(tmp_path: Path) -> None:
    model_file = tmp_path / "vehicle_type_assignment_model.yaml"
    _write_required_model_file(model_file)
    model_file.write_text(
        "\n".join(
            [
                "model:",
                "  scoring:",
                "    likelihood_floor: 0.01",
                "    weights:",
                "      prior_vmt_share: 1.0",
                "  evidence:",
                "    naics_sector:",
                "      - naics_code_2: '11'",
                "        vehicle_category:",
                "          - T7 Tractor Class 8",
                "    port_location: []",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evidence.port_location"):
        _normalize_model_spec_path(str(model_file), path_label="vehicle_type_assignment.model_file")
