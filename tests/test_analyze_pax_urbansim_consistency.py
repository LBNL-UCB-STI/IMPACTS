from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.tools.beam.analyze_pax_urbansim_consistency import _aggregate_plan_auto_usage
from impacts.tools.beam.analyze_pax_urbansim_consistency import _build_household_report
from impacts.tools.beam.analyze_pax_urbansim_consistency import _build_summary
from impacts.tools.beam.analyze_pax_urbansim_consistency import _load_households
from impacts.tools.beam.analyze_pax_urbansim_consistency import _load_person_households
from impacts.tools.beam.analyze_pax_urbansim_consistency import _load_vehicle_counts


def test_analyze_pax_urbansim_consistency_builds_expected_household_comparisons(tmp_path: Path) -> None:
    households_path = tmp_path / "households.parquet"
    persons_path = tmp_path / "persons.parquet"
    vehicles_path = tmp_path / "vehicles.csv.gz"
    plans_path = tmp_path / "plans.parquet"

    pd.DataFrame(
        [{"cars": 2}, {"cars": 0}],
        index=pd.Index([101, 202], name="household_id"),
    ).to_parquet(households_path)

    pd.DataFrame(
        [{"household_id": 101}, {"household_id": 202}],
        index=pd.Index([1001, 2001], name="person_id"),
    ).to_parquet(persons_path)

    pd.DataFrame(
        [
            {"household_id": "101.0", "vehicle_id": "1"},
            {"household_id": "101.0", "vehicle_id": "2"},
            {"household_id": "202.0", "vehicle_id": "1"},
        ]
    ).to_csv(vehicles_path, index=False, compression="gzip")

    pd.DataFrame(
        [
            {"person_id": 1001, "tour_id": "t1", "trip_id": "r1", "tour_mode": "DRIVEALONEPAY", "trip_mode": "DRIVEALONEPAY"},
            {"person_id": 1001, "tour_id": "t1", "trip_id": "r2", "tour_mode": "DRIVEALONEPAY", "trip_mode": "WALK"},
            {"person_id": 2001, "tour_id": "t2", "trip_id": "r3", "tour_mode": "WALK", "trip_mode": "WALK"},
            {"person_id": 2001, "tour_id": "t3", "trip_id": "r4", "tour_mode": "SHARED2PAY", "trip_mode": "SHARED2PAY"},
        ]
    ).to_parquet(plans_path, index=False)

    households = _load_households(households_path)
    person_households = _load_person_households(persons_path)
    vehicle_counts = _load_vehicle_counts(vehicles_path)
    plan_counts = _aggregate_plan_auto_usage(plans_path, person_households, batch_size=2)
    report = _build_household_report(households, vehicle_counts, plan_counts)
    summary = _build_summary(report)

    assert report.loc["101", "cars"] == 2
    assert report.loc["101", "vehicle_rows"] == 2
    assert report.loc["101", "auto_trip_ids"] == 1
    assert report.loc["101", "auto_tour_ids"] == 1
    assert report.loc["101", "vehicles_match_cars"]

    assert report.loc["202", "cars"] == 0
    assert report.loc["202", "vehicle_rows"] == 1
    assert report.loc["202", "auto_trip_ids"] == 1
    assert report.loc["202", "auto_tour_ids"] == 1
    assert not report.loc["202", "vehicles_match_cars"]
    assert report.loc["202", "auto_tours_gt_cars"]
    assert not report.loc["202", "auto_tours_gt_vehicles"]

    assert summary["households"] == 2
    assert summary["total_household_cars"] == 2
    assert summary["total_vehicle_rows"] == 3
    assert summary["households_vehicle_rows_match_cars"] == 1
    assert summary["households_vehicle_rows_do_not_match_cars"] == 1
    assert summary["households_auto_tours_gt_cars"] == 1
