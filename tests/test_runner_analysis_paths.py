from __future__ import annotations

from pathlib import Path

import pytest

from impacts.config.path_registry import PathRegistry
from impacts.postprocessor import _resolve_inventory_emfacid_activity_path
from impacts.postprocessor import _resolve_vehicle_category_metadata_path


def test_resolve_inventory_emfacid_activity_path_uses_colocated_emfacid_file_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("impacts:\n  local_output_folder: impacts_output\n")
    missing_path = tmp_path / "passenger-activity-by-emfacid.parquet"

    monkeypatch.setattr(
        "impacts.postprocessor._load_context",
        lambda *_args, **_kwargs: (
            None,
            None,
            None,
            {
                "passenger_inventory_emfacid_file": {
                    "kind": "local",
                    "source_path": str(missing_path),
                    "staged_path": str(missing_path),
                    "optional": False,
                    "exists": False,
                }
            },
        ),
    )

    with pytest.raises(FileNotFoundError) as error:
        _resolve_inventory_emfacid_activity_path(
            settings_path,
            manifest_key="passenger_inventory_emfacid_file",
            registry=PathRegistry([tmp_path]),
        )

    assert str(missing_path.resolve()) in str(error.value)


def test_resolve_vehicle_category_metadata_path_prefers_manifest_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("impacts:\n  local_output_folder: impacts_output\n")
    metadata_path = tmp_path / "emissions_vehicle_categories.csv"
    metadata_path.write_text("emfac_vehicle_category,generic_vehicle_category,operation_days_per_year,idle_time_fraction\n")
    monkeypatch.setattr(
        "impacts.postprocessor._load_context",
        lambda *_args, **_kwargs: (
            None,
            None,
            None,
            {
                "vehicle_category_metadata_file_input": {
                    "kind": "local",
                    "source_path": str(metadata_path),
                    "staged_path": str(metadata_path),
                    "optional": False,
                    "exists": True,
                }
            },
        ),
    )

    resolved = _resolve_vehicle_category_metadata_path(settings_path, registry=PathRegistry([tmp_path]))

    assert resolved == metadata_path.resolve()
