from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.error import HTTPError

import pytest

from impacts import reference_scenario


def test_reference_is_pinned_once_per_run_across_iterations(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"reference from iteration one")
    run_root = tmp_path / "run"
    reference_root = run_root / "input" / "reference"

    first = reference_scenario.pin_reference_scenario(str(source), run_root, reference_root=reference_root)
    source.write_bytes(b"a later iteration sees a changed source")
    second = reference_scenario.pin_reference_scenario(str(source), run_root, reference_root=reference_root)

    pinned = run_root / first["saved_path"]
    assert first["reused"] is False
    assert second["reused"] is True
    assert first["saved_path"] == "input/reference/beam_concentration_distribution.parquet"
    assert pinned.read_bytes() == b"reference from iteration one"
    assert first["sha256"] == second["sha256"]


def test_reference_directory_requires_one_concentration_file(tmp_path: Path) -> None:
    source = tmp_path / "archive"
    (source / "one" / "exposure").mkdir(parents=True)
    (source / "two" / "exposure").mkdir(parents=True)
    for directory in (source / "one" / "exposure", source / "two" / "exposure"):
        (directory / reference_scenario.REFERENCE_FILENAME).write_bytes(b"parquet")

    with pytest.raises(ValueError, match="exactly one"):
        reference_scenario.pin_reference_scenario(str(source), tmp_path / "run")


def test_reference_expected_checksum_is_enforced(tmp_path: Path) -> None:
    source = tmp_path / "reference.parquet"
    source.write_bytes(b"parquet bytes")
    wrong = hashlib.sha256(b"different bytes").hexdigest()

    with pytest.raises(ValueError, match="checksum mismatch"):
        reference_scenario.pin_reference_scenario(str(source), tmp_path / "run", expected_sha256=wrong)


def test_published_http_manifest_resolves_relative_artifact(monkeypatch, tmp_path: Path) -> None:
    artifact = b"published parquet bytes"
    checksum = hashlib.sha256(artifact).hexdigest()
    manifest_url = "https://example.test/archive/postsim/postsim_manifest.yaml"
    artifact_url = "https://example.test/archive/postsim/exposure/beam_concentration_distribution.parquet"

    def fake_http_read(url: str) -> bytes:
        if url == "https://example.test/archive/postsim_manifest.yaml":
            raise HTTPError(url, 404, "not found", {}, None)
        if url == manifest_url:
            return (
                "outputs:\n  beam_concentration_distribution:\n"
                "    path: exposure/beam_concentration_distribution.parquet\n"
                f"    sha256: {checksum}\n"
            ).encode()
        if url == artifact_url:
            return artifact
        raise AssertionError(url)

    monkeypatch.setattr(reference_scenario, "_http_read", fake_http_read)
    pinned = reference_scenario.pin_reference_scenario(
        "https://example.test/archive",
        tmp_path / "run",
        reference_root=tmp_path / "run" / "input" / "reference",
    )
    assert pinned["resolved_source"] == artifact_url
    assert pinned["sha256"] == checksum
    assert (tmp_path / "run" / pinned["saved_path"]).read_bytes() == artifact


def test_reference_grid_requires_same_cells_and_pollutants(tmp_path: Path) -> None:
    duckdb = pytest.importorskip("duckdb")
    current = tmp_path / "current.parquet"
    reference = tmp_path / "reference.parquet"
    other_grid = tmp_path / "other-grid.parquet"
    required = """
        SELECT 1::BIGINT AS aermod_cell_id, 10::BIGINT AS inmap_cell_id,
               1.0 AS TotalPM25, 0.2 AS PrimaryPM25, 0.8 AS SecondaryPM25,
               0.1 AS BC, 2.0 AS NO2
        UNION ALL
        SELECT 2::BIGINT, 10::BIGINT, 2.0, 0.3, 1.7, 0.2, 3.0
    """
    connection = duckdb.connect()
    try:
        connection.execute(f"COPY ({required}) TO '{current}' (FORMAT PARQUET)")
        connection.execute(f"COPY ({required}) TO '{reference}' (FORMAT PARQUET)")
        connection.execute(
            f"COPY (SELECT * REPLACE (20::BIGINT AS inmap_cell_id) FROM read_parquet('{current}')) "
            f"TO '{other_grid}' (FORMAT PARQUET)"
        )
    finally:
        connection.close()

    reference_scenario.validate_reference_grid(current, reference)
    with pytest.raises(ValueError, match="different AERMOD/InMAP grid"):
        reference_scenario.validate_reference_grid(current, other_grid)
