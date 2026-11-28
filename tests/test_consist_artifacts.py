from __future__ import annotations

from pathlib import Path
import sys
import types

from impacts.common import register_managed_input


def test_register_managed_input_tracks_source_without_copy_when_consist_unavailable(monkeypatch, tmp_path: Path):
    monkeypatch.delitem(sys.modules, "consist", raising=False)

    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    input_root = tmp_path / "staged"
    manifest_inputs = {}

    staged_path = register_managed_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="settings",
        source_path=str(source),
        relative_target="config/settings.yaml",
        optional=False,
        prefer_reference=True,
    )

    assert staged_path == str(source.resolve())
    assert Path(staged_path).exists()
    assert manifest_inputs["settings"]["kind"] == "local"
    assert manifest_inputs["settings"]["staged_path"] == staged_path


def test_register_managed_input_logs_reference_when_consist_enabled(monkeypatch, tmp_path: Path):
    class _Artifact:
        id = "artifact-1"
        key = "emissions_skims_input"

    fake_consist = types.SimpleNamespace(
        __version__="0.test",
        log_input=lambda *, path, key, metadata: _Artifact(),
    )
    monkeypatch.setitem(sys.modules, "consist", fake_consist)

    source = tmp_path / "0.skimsEmissions.parquet"
    source.write_text("placeholder", encoding="utf-8")
    input_root = tmp_path / "staged"
    manifest_inputs = {}

    staged_path = register_managed_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="emissions_skims_input",
        source_path=str(source),
        relative_target="skims/0.skimsEmissions.parquet",
        optional=True,
        prefer_reference=True,
        metadata={"artifact_family": "emissions_skims_input"},
    )

    assert staged_path == str(source.resolve())
    assert manifest_inputs["emissions_skims_input"]["kind"] == "consist"
    assert manifest_inputs["emissions_skims_input"]["storage_mode"] == "reference"
    assert manifest_inputs["emissions_skims_input"]["staged_path"] == str(source.resolve())
    assert manifest_inputs["emissions_skims_input"]["consist"]["artifact"]["id"] == "artifact-1"


def test_register_managed_input_uses_local_entry_when_consist_has_no_active_run(monkeypatch, tmp_path: Path):
    def _raise_no_active_run(*, path, key, metadata):
        raise RuntimeError("No active Consist run found. Ensure you are within a start_run block.")

    fake_consist = types.SimpleNamespace(
        __version__="0.test",
        log_input=_raise_no_active_run,
    )
    monkeypatch.setitem(sys.modules, "consist", fake_consist)

    source = tmp_path / "network.csv.gz"
    source.write_text("placeholder", encoding="utf-8")
    input_root = tmp_path / "staged"
    manifest_inputs = {}

    staged_path = register_managed_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="network",
        source_path=str(source),
        relative_target="network.csv.gz",
        optional=False,
        prefer_reference=True,
    )

    assert staged_path == str(source.resolve())
    assert manifest_inputs["network"]["kind"] == "local"
    assert manifest_inputs["network"]["staged_path"] == str(source.resolve())
