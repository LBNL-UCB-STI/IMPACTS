from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from impacts.common import register_managed_input
from impacts.config.path_registry import PathRegistry
from impacts.config.settings import DEFAULT_CONFIG_PATH
from impacts.config.settings_builder import derive_settings_from_pilates, load_settings_from_yaml
from impacts.manifest.file_ops import load_structured_file, write_structured_file
from impacts.pipeline.preprocessing import step1_collect_inputs
from impacts.consist_artifacts import find_beam_r5_osm_reference
from impacts.consist_artifacts import find_latest_beam_events_reference


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


def test_find_latest_beam_events_reference_prefers_exact_key(monkeypatch, tmp_path: Path):
    exact = tmp_path / "0.events.parquet"
    exact.write_text("exact", encoding="utf-8")
    sub1 = tmp_path / "0.events.sub1.parquet"
    sub1.write_text("sub1", encoding="utf-8")
    sub2 = tmp_path / "0.events.sub2.parquet"
    sub2.write_text("sub2", encoding="utf-8")

    fake_consist = types.SimpleNamespace(
        get_run_outputs=lambda: {
            "events_parquet_2018_0_sub1": str(sub1),
            "events_parquet_2018_0_sub2": str(sub2),
            "events_parquet_2018_0": str(exact),
        }
    )
    monkeypatch.setitem(sys.modules, "consist", fake_consist)

    entry = find_latest_beam_events_reference(optional=True)

    assert entry is not None
    assert entry["staged_path"] == str(exact.resolve())
    assert entry["consist"]["artifact_key"] == "events_parquet_2018_0"


def test_find_beam_r5_osm_reference_reads_canonical_key(monkeypatch, tmp_path: Path):
    osm = tmp_path / "network.osm.pbf"
    osm.write_text("osm", encoding="utf-8")

    fake_consist = types.SimpleNamespace(
        find_artifact=lambda *, key, metadata=None: str(osm) if key == "beam_r5_osm_file" else None,
    )
    monkeypatch.setitem(sys.modules, "consist", fake_consist)

    entry = find_beam_r5_osm_reference(optional=True)

    assert entry is not None
    assert entry["staged_path"] == str(osm.resolve())
    assert entry["consist"]["artifact_key"] == "beam_r5_osm_file"


def test_find_beam_r5_osm_reference_uses_local_fallback_without_active_run(monkeypatch):
    def _raise_no_active_run():
        raise RuntimeError("No active Consist run found. Ensure you are within a start_run block.")

    def _unexpected_lookup(**_kwargs):
        raise AssertionError("Artifact lookup should be skipped without an active run")

    fake_consist = types.SimpleNamespace(
        get_run_outputs=_raise_no_active_run,
        get_artifact=_unexpected_lookup,
    )
    monkeypatch.setitem(sys.modules, "consist", fake_consist)

    assert find_beam_r5_osm_reference(optional=True) is None


def test_find_beam_r5_osm_reference_keeps_warning_for_active_run_lookup_error(
    monkeypatch, caplog
):
    def _legacy_get_artifact(run_id, key):
        raise AssertionError("The incompatible legacy lookup should not be invoked")

    fake_consist = types.SimpleNamespace(
        get_run_outputs=lambda: {},
        get_artifact=_legacy_get_artifact,
    )
    monkeypatch.setitem(sys.modules, "consist", fake_consist)

    assert find_beam_r5_osm_reference(optional=True) is None
    assert "Consist get_artifact lookup failed" in caplog.text


@pytest.mark.parametrize("explicit", [False, True])
def test_osm_input_preserves_explicit_file_or_standalone_folder(monkeypatch, tmp_path, explicit):
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    folder_osm = fallback / "fallback.osm.pbf"
    folder_osm.touch()
    selected = tmp_path / "selected.osm.pbf"
    selected.touch()
    emissions = {"osm_network_folder": str(fallback)}
    if explicit:
        emissions.update(osm_network_file=str(selected), osm_network_artifact_id="artifact-123")

    def find_osm(*, optional):
        assert not explicit, "Explicit OSM selection must bypass Consist"
        return None

    monkeypatch.setattr(step1_collect_inputs, "find_beam_r5_osm_reference", find_osm)

    default = load_structured_file(DEFAULT_CONFIG_PATH)
    bridge = {key: default[key] for key in ("run", "shared", "beam")}
    bridge["impacts"] = {"emissions": emissions}
    bridge_path, derived = tmp_path / "bridge.yaml", tmp_path / "settings.yaml"
    write_structured_file(bridge_path, bridge)
    derive_settings_from_pilates(pilates_settings_path=bridge_path, output_path=derived)

    manifest = {}
    settings = load_settings_from_yaml(derived)
    resolved = step1_collect_inputs._resolve_osm_input(
        manifest, emissions=settings.impacts.emissions,
        registry=PathRegistry([tmp_path]), input_root=tmp_path / "staged",
    )
    assert resolved == str((selected if explicit else folder_osm).resolve())
    assert manifest["osm_network"]["source_path"] == resolved
    assert manifest["osm_network"]["staged_path"] == resolved
    assert manifest["osm_network"].get("artifact_id") == ("artifact-123" if explicit else None)

    if explicit:
        selected.unlink()
        with pytest.raises(FileNotFoundError, match="osm_network_file"):
            step1_collect_inputs._resolve_osm_input(
                {}, emissions=settings.impacts.emissions,
                registry=PathRegistry([tmp_path]), input_root=tmp_path / "staged",
            )
