from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from impacts import acs_data
from impacts.pipeline.preprocessing.step6_calibrate_acs_household_population import run as calibrate_acs_households
from impacts.pipeline.preprocessing.step7_assign_acs_household_population_to_grid import run as assign_acs_households
from impacts.manifest.file_ops import file_entry


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

    weights_path, summary_path, diagnostics = calibrate_acs_households(
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
    assert diagnostics["06001"]["household_weight_min"] <= diagnostics["06001"]["household_weight_median"]
    assert diagnostics["06001"]["household_weight_median"] <= diagnostics["06001"]["household_weight_max"]


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
        {
            "household_id": [10, 20, 30],
            "household_size": [2, 1, 2],
            "acs_household_weight": [1.5, 0.5, 0.25],
        }
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
    assert summary["acs_household_population_calibrated_total"] == pytest.approx(4.0)
    assert summary["acs_household_population_assigned_to_grid"] == pytest.approx(3.5)
    assert summary["acs_household_population_not_assigned_to_grid"] == pytest.approx(0.5)


def test_acs_household_assignment_rejects_unweighted_persons(tmp_path: Path) -> None:
    staged = tmp_path / "staged_population.parquet"
    weights = tmp_path / "acs_household_weights.parquet"
    pd.DataFrame(
        {"person_id": [1], "household_id": [10], "aermod_cell_id": [101]}
    ).to_parquet(staged, index=False)
    pd.DataFrame(
        {"household_id": [20], "household_size": [1], "acs_household_weight": [1.0]}
    ).to_parquet(weights, index=False)

    with pytest.raises(ValueError, match="weights are missing for 1 assigned residents"):
        assign_acs_households(
            staged_population_entry=file_entry(kind="local", path=staged, staged_path=staged),
            household_weights_entry=file_entry(kind="local", path=weights, staged_path=weights),
            output_root=tmp_path / "step7",
        )


def test_acs_household_assignment_checks_resident_rows_against_household_size(tmp_path: Path) -> None:
    staged = tmp_path / "staged_population.parquet"
    weights = tmp_path / "acs_household_weights.parquet"
    pd.DataFrame(
        {"person_id": [1], "household_id": [10], "aermod_cell_id": [101]}
    ).to_parquet(staged, index=False)
    pd.DataFrame(
        {"household_id": [10], "household_size": [2], "acs_household_weight": [1.0]}
    ).to_parquet(weights, index=False)

    with pytest.raises(ValueError, match="do not match their UrbanSim household sizes"):
        assign_acs_households(
            staged_population_entry=file_entry(kind="local", path=staged, staged_path=staged),
            household_weights_entry=file_entry(kind="local", path=weights, staged_path=weights),
            output_root=tmp_path / "step7",
        )
