from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from impacts.sampling import sample_events_by_vehicle
from impacts.sampling import sample_skims_by_fraction


def test_sample_events_by_vehicle_keeps_full_vehicle_traces(tmp_path: Path):
    source = tmp_path / "events.csv.gz"
    pd.DataFrame(
        [
            {"vehicle": 1, "type": "PathTraversal", "time": 1},
            {"vehicle": 1, "type": "ParkingEvent", "time": 2},
            {"vehicle": 2, "type": "PathTraversal", "time": 3},
            {"vehicle": 3, "type": "PathTraversal", "time": 4},
            {"vehicle": 3, "type": "RefuelSessionEvent", "time": 5},
        ]
    ).to_csv(source, index=False, compression="gzip")

    result = sample_events_by_vehicle(
        input_path=source,
        output_path=tmp_path / "events_sample.csv.gz",
        fraction=0.34,
        seed=1,
    )

    sampled = pd.read_csv(tmp_path / "events_sample.csv.gz", compression="gzip")
    sampled_vehicles = set(sampled["vehicle"].unique().tolist())
    assert result["selected_vehicles"] == len(sampled_vehicles)
    for vehicle in sampled_vehicles:
        original_rows = pd.read_csv(source, compression="gzip")
        original_count = len(original_rows[original_rows["vehicle"] == vehicle])
        sampled_count = len(sampled[sampled["vehicle"] == vehicle])
        assert sampled_count == original_count


def test_sample_skims_by_fraction_returns_subset(tmp_path: Path):
    source = tmp_path / "skims.csv.gz"
    pd.DataFrame(
        [
            {"hour": 0, "linkId": idx, "vehicleTypeId": "car", "emissionsProcess": "RUNEX", "observations": 1}
            for idx in range(100)
        ]
    ).to_csv(source, index=False, compression="gzip")

    result = sample_skims_by_fraction(
        input_path=source,
        output_path=tmp_path / "skims_sample.csv.gz",
        fraction=0.05,
        seed=7,
    )

    sampled = pd.read_csv(tmp_path / "skims_sample.csv.gz", compression="gzip")
    assert 0 < len(sampled) < 100
    assert result["total_rows"] == 100
    assert result["kept_rows"] == len(sampled)


def test_sample_skims_compacts_explicit_pollutant_schema(tmp_path: Path):
    source = tmp_path / "skims_explicit.csv.gz"
    pd.DataFrame(
        [
            {
                "hour": 13,
                "linkId": 1184,
                "tazId": 395,
                "vehicleTypeId": "truck",
                "emissionsProcess": "PMTW",
                "speedInMps": 25.5,
                "energyInJoule": 123.0,
                "observations": 1,
                "iterations": 1,
                "CH4": 0.0,
                "CO": 0.0,
                "CO2": 0.0,
                "HC": 0.0,
                "NH3": 0.0,
                "NOx": 0.0,
                "PM": 0.00055,
                "PM10": 0.00055,
                "PM2_5": 0.00013,
                "ROG": 0.0,
                "SOx": 0.0,
                "TOG": 0.0,
            }
        ]
    ).to_csv(source, index=False, compression="gzip")

    sample_skims_by_fraction(
        input_path=source,
        output_path=tmp_path / "skims_sample.csv.gz",
        fraction=1.0,
        seed=7,
    )

    sampled = pd.read_csv(tmp_path / "skims_sample.csv.gz", compression="gzip")
    assert "process" in sampled.columns
    assert "emissionsProcess" not in sampled.columns
    assert list(sampled.columns) == ["linkId", "vehicleTypeId", "process", "CH4", "CO", "CO2", "HC", "NH3", "NOx", "PM", "PM10", "PM2_5", "ROG", "SOx", "TOG"]
    assert sampled.loc[0, "PM"] == pytest.approx(0.00055)
    assert sampled.loc[0, "PM2_5"] == pytest.approx(0.00013)


def test_sample_skims_compact_schema_converts_to_totals_and_scales(tmp_path: Path):
    source = tmp_path / "skims_compact.csv.gz"
    pd.DataFrame(
        [
            {
                "linkId": 1,
                "vehicleTypeId": "car",
                "process": "RUNEX",
                "emissions": "NOx:1.5;PM2_5:0.2",
                "travelTimeInSecond": 10.0,
                "parkingDurationInSecond": 0.0,
                "observations": 2,
                "iterations": 1,
            }
        ]
    ).to_csv(source, index=False, compression="gzip")

    sample_skims_by_fraction(
        input_path=source,
        output_path=tmp_path / "skims_sample.csv.gz",
        fraction=1.0,
        seed=7,
        population_sample=0.1,
    )

    sampled = pd.read_csv(tmp_path / "skims_sample.csv.gz", compression="gzip")
    assert list(sampled.columns) == [
        "linkId",
        "vehicleTypeId",
        "process",
        "NOx",
        "PM2_5",
    ]
    assert sampled.loc[0, "NOx"] == pytest.approx(30.0)
    assert sampled.loc[0, "PM2_5"] == pytest.approx(4.0)


def test_sample_skims_totals_schema_is_sampled_as_is(tmp_path: Path):
    source = tmp_path / "skims_totals.csv.gz"
    pd.DataFrame(
        [
            {"linkId": 1, "vehicleTypeId": "car", "process": "RUNEX", "NOx": 10.0, "PM2_5": 2.0},
            {"linkId": 2, "vehicleTypeId": "car", "process": "RUNEX", "NOx": 4.0, "PM2_5": 1.0},
        ]
    ).to_csv(source, index=False, compression="gzip")

    sample_skims_by_fraction(
        input_path=source,
        output_path=tmp_path / "skims_totals_sample.csv.gz",
        fraction=1.0,
        seed=7,
        population_sample=0.2,
    )

    sampled = pd.read_csv(tmp_path / "skims_totals_sample.csv.gz", compression="gzip")
    assert list(sampled.columns) == ["linkId", "vehicleTypeId", "process", "NOx", "PM2_5"]
    assert sampled["NOx"].tolist() == [10.0, 4.0]
    assert sampled["PM2_5"].tolist() == [2.0, 1.0]
