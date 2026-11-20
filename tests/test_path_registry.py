from pathlib import Path
from types import SimpleNamespace

from impacts.config.path_registry import build_registry


def test_build_registry_does_not_search_impacts_local_input_folder(tmp_path: Path):
    beam_input = tmp_path / "beam-input"
    beam_output = tmp_path / "beam-output"
    impacts_input = tmp_path / "impacts-input"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    impacts_input.mkdir()
    stale = impacts_input / "stale-input.csv"
    stale.write_text("should not resolve implicitly")
    settings = SimpleNamespace(
        beam=SimpleNamespace(
            local_input_folder=str(beam_input),
            local_output_folder=str(beam_output),
        ),
        impacts=SimpleNamespace(
            local_input_folder=str(impacts_input),
            local_output_folder=str(tmp_path / "impacts-output"),
            population=SimpleNamespace(vehicle_folder="vehicle-tech"),
        ),
        run=SimpleNamespace(region="sfbay"),
    )

    registry = build_registry(settings, config_dir / "settings.yaml")

    assert impacts_input.resolve() not in registry.roots
    assert registry.locate("stale-input.csv") is None
