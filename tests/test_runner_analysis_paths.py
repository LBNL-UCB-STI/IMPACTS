from __future__ import annotations

from pathlib import Path

from impacts.runner import _resolve_analysis_inventory_emfacid_activity_path
from impacts.runner import _resolve_analysis_vehicle_category_metadata_path


def test_resolve_analysis_inventory_emfacid_activity_path_falls_back_to_emfac_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("impacts:\n  local_output_folder: impacts_output\n")
    repo_root = tmp_path / "repo"
    production_dir = repo_root / "examples" / "pipeline" / "pilates" / "beam" / "production" / "sfbay" / "vehicle-tech" / "emissions"
    production_dir.mkdir(parents=True, exist_ok=True)
    source = production_dir / "sf-emfac-2018-inventory-final-passenger-activity.parquet"
    source.write_text("")

    fallback_dir = repo_root / "examples" / "emfac" / "output" / "activities"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    fallback = fallback_dir / "sf-emfac-2018-inventory-final-passenger-activity-by-emfacid.parquet"
    fallback.write_text("")

    monkeypatch.setattr("impacts.runner._REPO_ROOT", repo_root)
    monkeypatch.setattr("impacts.runner._load_analysis_context", lambda _: (None, None, None, {}))

    resolved = _resolve_analysis_inventory_emfacid_activity_path(
        settings_path,
        source=source,
        manifest_key="passenger_inventory_emfacid_file",
    )

    assert resolved == fallback.resolve()


def test_resolve_analysis_vehicle_category_metadata_path_prefers_manifest_input(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("impacts:\n  local_output_folder: impacts_output\n")
    metadata_path = tmp_path / "emfac_vehicle_category_attributes.csv"
    metadata_path.write_text("emfac_vehicle_category,generic_vehicle_category,operation_days_per_year,idle_time_fraction\n")
    monkeypatch.setattr(
        "impacts.runner._load_analysis_context",
        lambda _: (
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

    resolved = _resolve_analysis_vehicle_category_metadata_path(settings_path)

    assert resolved == metadata_path.resolve()
