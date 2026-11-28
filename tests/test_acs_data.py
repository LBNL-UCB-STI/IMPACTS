from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import pandas as pd

from impacts import acs_data
from impacts.pipeline.preprocessing.step6_calibrate_acs_household_population import run as calibrate_acs_households
from impacts.pipeline.preprocessing.step7_assign_acs_household_population_to_grid import run as assign_acs_households
from impacts.manifest.file_ops import file_entry


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self):
        return None

    def iter_content(self, *, chunk_size: int):
        assert chunk_size == 1024 * 1024
        yield self.body


def test_acs_table_urls_follow_baseline_release_year() -> None:
    assert acs_data._table_url(2018, "1yr", "B11001") == (
        "https://www2.census.gov/programs-surveys/acs/summary_file/2018/prototype/"
        "1YRData/acsdt1y2018-b11001.dat"
    )
    assert acs_data._table_url(2023, "5yr", "B26001") == (
        "https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/"
        "5YRData/acsdt5y2023-b26001.dat"
    )
    with pytest.raises(ValueError, match="2018 or later"):
        acs_data._table_url(2017, "5yr", "B11001")


def test_acquire_acs_tables_downloads_and_reuses_checked_cache(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_get(url, **_kwargs):
        calls.append(url)
        table = url.rsplit("-", 1)[-1].split(".", 1)[0].upper()
        return _Response(f"#GEO_ID|{table}_001E|{table}_001M\n06001|10|1\n".encode())

    monkeypatch.setattr(acs_data.requests, "get", fake_get)
    first = acs_data.acquire_acs_tables(2018, tmp_path)
    expected_files = sum(len(tables) for tables in acs_data.ACS_TABLES.values())
    assert len(calls) == expected_files
    assert set(first) == {"1yr", "5yr"}
    assert all(Path(path).is_file() for tables in first.values() for path in tables.values())

    second = acs_data.acquire_acs_tables(2018, tmp_path)
    assert second == first
    assert len(calls) == expected_files
    manifest = (tmp_path / "acs" / "2018" / "manifest.json").read_text()
    assert hashlib.sha256(Path(first["1yr"]["B11001"]).read_bytes()).hexdigest() in manifest


def test_acquire_acs_tables_rejects_non_table_response(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(acs_data.requests, "get", lambda *_args, **_kwargs: _Response(b"not a table"))
    with pytest.raises(ValueError, match="not the expected ACS"):
        acs_data.acquire_acs_tables(2018, tmp_path)


def test_read_acs_estimates_filters_geographies_and_keeps_moe(tmp_path: Path) -> None:
    table = tmp_path / "b11001.dat"
    table.write_text(
        "#GEO_ID|B11001_001E|B11001_001M\n"
        "0500000US06001|100|5\n"
        "1400000US06001000100|20|4\n"
        "0500000US06013|200|6\n",
        encoding="utf-8",
    )

    counties = acs_data.read_acs_estimates(
        table,
        table="B11001",
        geography="county",
        geoid_prefixes=("06001",),
    )

    assert counties.to_dict("records") == [{"geoid": "06001", "estimate": 100, "moe": 5}]


def test_acs_household_calibration_matches_both_controls(tmp_path: Path) -> None:
    households = tmp_path / "households.parquet"
    pd.DataFrame(
        {
            "household_id": [11, 12],
            "county_geoid": ["06001", "06001"],
            "household_size": [1, 3],
        }
    ).to_parquet(households, index=False)
    b11001 = tmp_path / "b11001.dat"
    b11002 = tmp_path / "b11002.dat"
    b11001.write_text(
        "#GEO_ID|B11001_001E|B11001_001M\n0500000US06001|2|1\n",
        encoding="utf-8",
    )
    b11002.write_text(
        "#GEO_ID|B11002_001E|B11002_001M\n0500000US06001|4|1\n",
        encoding="utf-8",
    )

    weights_path, summary_path, _ = calibrate_acs_households(
        household_metadata_path=households,
        acs_tables={"1yr": {"B11001": str(b11001), "B11002": str(b11002)}},
        state_fips="06",
        county_fips_codes=["001"],
        year=2018,
        output_root=tmp_path / "step6",
    )

    weights = pd.read_parquet(weights_path)
    assert weights["acs_household_weight"].sum() == pytest.approx(2)
    assert (weights["acs_household_weight"] * weights["household_size"]).sum() == pytest.approx(4)
    assert '"group_quarters_included": false' in Path(summary_path).read_text()


def test_acs_household_weights_apply_to_each_assigned_resident(tmp_path: Path) -> None:
    staged = tmp_path / "staged_population.parquet"
    weights = tmp_path / "acs_household_weights.parquet"
    pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "household_id": [10, 10, 20],
            "aermod_cell_id": [101, 101, 202],
        }
    ).to_parquet(staged, index=False)
    pd.DataFrame(
        {"household_id": [10, 20], "acs_household_weight": [1.5, 0.5]}
    ).to_parquet(weights, index=False)

    output_path, summary = assign_acs_households(
        staged_population_entry=file_entry(kind="local", path=staged, staged_path=staged),
        household_weights_entry=file_entry(kind="local", path=weights, staged_path=weights),
        output_root=tmp_path / "step7",
    )

    result = pd.read_parquet(output_path).sort_values("aermod_cell_id")
    assert result.to_dict("records") == [
        {"aermod_cell_id": 101, "acs_household_population": 3.0},
        {"aermod_cell_id": 202, "acs_household_population": 0.5},
    ]
    assert summary["residents_assigned_to_grid"] == 3
    assert summary["acs_household_population_assigned_to_grid"] == pytest.approx(3.5)


def test_acs_household_assignment_rejects_unweighted_persons(tmp_path: Path) -> None:
    staged = tmp_path / "staged_population.parquet"
    weights = tmp_path / "acs_household_weights.parquet"
    pd.DataFrame(
        {"person_id": [1], "household_id": [10], "aermod_cell_id": [101]}
    ).to_parquet(staged, index=False)
    pd.DataFrame({"household_id": [20], "acs_household_weight": [1.0]}).to_parquet(
        weights, index=False
    )

    with pytest.raises(ValueError, match="weights are missing for 1 assigned residents"):
        assign_acs_households(
            staged_population_entry=file_entry(kind="local", path=staged, staged_path=staged),
            household_weights_entry=file_entry(kind="local", path=weights, staged_path=weights),
            output_root=tmp_path / "step7",
        )
