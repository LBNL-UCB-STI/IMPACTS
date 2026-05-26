from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.tools.beam.sample_beam_output import sample_events_by_vehicle
from impacts.tools.beam.sample_beam_output import sample_skims_by_fraction


def test_sample_events_by_vehicle_keeps_full_vehicle_traces(tmp_path: Path):
    source = tmp_path / "events.parquet"
    pd.DataFrame(
        [
            {"vehicle": 1, "type": "PathTraversal", "time": 1},
            {"vehicle": 1, "type": "ParkingEvent", "time": 2},
            {"vehicle": 2, "type": "PathTraversal", "time": 3},
            {"vehicle": 3, "type": "PathTraversal", "time": 4},
            {"vehicle": 3, "type": "RefuelSessionEvent", "time": 5},
        ]
    ).to_parquet(source, index=False)

    result = sample_events_by_vehicle(
        input_path=source,
        output_path=tmp_path / "events_sample.parquet",
        fraction=0.34,
        seed=1,
    )

    sampled = pd.read_parquet(tmp_path / "events_sample.parquet")
    original = pd.read_parquet(source)
    sampled_vehicles = set(sampled["vehicle"].astype(str).unique().tolist())
    assert result["selected_vehicles"] == len(sampled_vehicles)
    for vehicle in sampled_vehicles:
        original_count = len(original[original["vehicle"].astype(str) == vehicle])
        sampled_count = len(sampled[sampled["vehicle"].astype(str) == vehicle])
        assert sampled_count == original_count


def test_sample_skims_by_fraction_returns_subset(tmp_path: Path):
    source = tmp_path / "skims.parquet"
    pd.DataFrame(
        [
            {"hour": 0, "linkId": idx, "vehicleTypeId": "car", "emissionsProcess": "RUNEX", "observations": 1}
            for idx in range(100)
        ]
    ).to_parquet(source, index=False)

    result = sample_skims_by_fraction(
        input_path=source,
        output_path=tmp_path / "skims_sample.parquet",
        fraction=0.05,
        seed=7,
    )

    sampled = pd.read_parquet(tmp_path / "skims_sample.parquet")
    assert 0 < len(sampled) < 100
    assert result["total_rows"] == 100
    assert result["kept_rows"] == len(sampled)


def test_sample_skims_by_fraction_streams_parquet_without_pandas_read_parquet(tmp_path: Path, monkeypatch):
    source = tmp_path / "skims.parquet"
    pd.DataFrame(
        [
            {"hour": 0, "linkId": idx, "vehicleTypeId": "car", "emissionsProcess": "RUNEX", "observations": 1}
            for idx in range(20)
        ]
    ).to_parquet(source, index=False)

    def _forbid_read_parquet(*args, **kwargs):
        raise AssertionError("sample_skims_by_fraction should stream parquet instead of calling pandas.read_parquet")

    monkeypatch.setattr(pd, "read_parquet", _forbid_read_parquet)

    result = sample_skims_by_fraction(
        input_path=source,
        output_path=tmp_path / "skims_sample.parquet",
        fraction=0.25,
        seed=7,
    )

    assert result["total_rows"] == 20
    assert (tmp_path / "skims_sample.parquet").exists()


def test_sample_skims_compacts_explicit_pollutant_schema(tmp_path: Path):
    source = tmp_path / "skims_explicit.parquet"
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
    ).to_parquet(source, index=False)

    sample_skims_by_fraction(
        input_path=source,
        output_path=tmp_path / "skims_sample.parquet",
        fraction=1.0,
        seed=7,
    )

    sampled = pd.read_parquet(tmp_path / "skims_sample.parquet")
    assert list(sampled.columns) == [
        "hour",
        "linkId",
        "vehicleTypeId",
        "process",
        "emissions",
        "travelTimeInSecond",
        "parkingDurationInSecond",
        "observations",
        "iterations",
    ]
    assert sampled.loc[0, "process"] == "PMTW"
    assert sampled.loc[0, "emissions"] == "PM:0.00055;PM10:0.00055;PM2_5:0.00013"


def test_sample_skims_compact_schema_is_preserved(tmp_path: Path):
    source = tmp_path / "skims_compact.parquet"
    pd.DataFrame(
        [
            {
                "hour": 0,
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
    ).to_parquet(source, index=False)

    sample_skims_by_fraction(
        input_path=source,
        output_path=tmp_path / "skims_sample.parquet",
        fraction=1.0,
        seed=7,
    )

    sampled = pd.read_parquet(tmp_path / "skims_sample.parquet")
    assert list(sampled.columns) == [
        "hour",
        "linkId",
        "vehicleTypeId",
        "process",
        "emissions",
        "travelTimeInSecond",
        "parkingDurationInSecond",
        "observations",
        "iterations",
    ]
    assert sampled.loc[0, "emissions"] == "NOx:1.5;PM2_5:0.2"


def test_sample_skims_totals_schema_is_sampled_as_is(tmp_path: Path):
    source = tmp_path / "skims_totals.parquet"
    pd.DataFrame(
        [
            {"linkId": 1, "vehicleTypeId": "car", "process": "RUNEX", "NOx": 10.0, "PM2_5": 2.0},
            {"linkId": 2, "vehicleTypeId": "car", "process": "RUNEX", "NOx": 4.0, "PM2_5": 1.0},
        ]
    ).to_parquet(source, index=False)

    sample_skims_by_fraction(
        input_path=source,
        output_path=tmp_path / "skims_totals_sample.csv.gz",
        fraction=1.0,
        seed=7,
    )

    sampled = pd.read_csv(tmp_path / "skims_totals_sample.csv.gz", compression="gzip")
    assert list(sampled.columns) == ["linkId", "vehicleTypeId", "process", "NOx", "PM2_5"]
    assert sampled["NOx"].tolist() == [10.0, 4.0]
    assert sampled["PM2_5"].tolist() == [2.0, 1.0]
