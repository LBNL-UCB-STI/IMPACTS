from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import impacts.__main__ as cli


FIXTURE = Path(__file__).parent / "fixtures" / "pilates-bridge.yaml"


def _run_smoke(tmp_path: Path):
    settings_path = tmp_path / "settings.yaml"
    assert cli.main([
        "derive_settings_from_pilates", "--pilates-settings", str(FIXTURE),
        "--output", str(settings_path),
    ]) == 0
    settings = cli.load_settings_from_yaml(settings_path)
    assert settings.run.region == "fixture"
    assert settings.beam.local_input_folder == "/app/beam_input"
    assert settings.impacts.dispersions.inmap.isrm_zarr == "/app/inmap/isrm_v1.2.1.zarr"

    # Redirect fixture mount paths into the temporary smoke-test workspace.
    import yaml

    payload = settings.to_dict()
    payload["impacts"]["local_output_folder"] = str(tmp_path / "impacts_output")
    settings_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert cli.main(["presim", "--config", str(settings_path)]) == 0

    manifest = tmp_path / "pipeline_manifest.yaml"
    manifest.write_text("output_dir: fixture\n", encoding="utf-8")
    with (
        patch.object(cli, "_resolve_pipeline_manifest_path", lambda *_: manifest),
        patch(
            "impacts.preprocessor.preprocess_workflow",
            lambda **_: {"pipeline_manifest_path": str(manifest)},
        ),
        patch("impacts.postprocessor.postprocess_from_pipeline_manifest", lambda **_: None),
        patch(
            "impacts.config.path_registry.build_registry",
            lambda *_args, **_kwargs: SimpleNamespace(roots=()),
        ),
        patch.dict("os.environ", {"IMPACTS_POSTSIM_OUTPUT_DIR": str(tmp_path / "postsim")}),
    ):
        assert cli.main(["postsim", "--config", str(settings_path)]) == 0


def test_pilates_commands_smoke(tmp_path: Path):
    _run_smoke(tmp_path)


if __name__ == "__main__":
    with TemporaryDirectory() as smoke_dir:
        _run_smoke(Path(smoke_dir))
