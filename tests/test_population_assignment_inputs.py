from pathlib import Path

from impacts import postprocessor


def test_manifest_population_assignments_are_resolved_independently(monkeypatch, tmp_path: Path) -> None:
    passenger_path = tmp_path / "vehicles.parquet"
    monkeypatch.setattr(
        postprocessor,
        "_load_context",
        lambda *_args, **_kwargs: (tmp_path / "run.yaml", {}, {}, {"passenger_vehicles_input": {}}),
    )
    monkeypatch.setattr(
        postprocessor,
        "_resolve_input_path",
        lambda *_args, **_kwargs: passenger_path,
    )

    passenger, freight = postprocessor._resolve_optional_population_assignment_paths(
        tmp_path / "settings.yaml", registry=object()
    )

    assert passenger == passenger_path
    assert freight is None
