from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import impacts.common as common
from impacts.pipeline.workflow.annualization import _build_skims_scale_factors
from impacts.pipeline.workflow.prepare_emissions_from_skims import _filter_prepared_skims_by_assignment


def test_filter_prepared_skims_by_assignment_uses_vehicle_types_contract(tmp_path: Path) -> None:
    passenger_vehicle_types = pd.DataFrame(
        [
            {"vehicleTypeId": "pax-car", "vehicleCategory": "Car"},
            {"vehicleTypeId": "pax-bus", "vehicleCategory": "Bus"},
        ]
    )
    freight_vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": "ft-md",
                "vehicleCategory": "Class456Vocational",
                "vehicleClass": "truck",
                "vehicleUse": "freight",
            },
        ]
    )
    passenger_vehicle_types_path = tmp_path / "vehicleTypes--atlas.csv"
    freight_vehicle_types_path = tmp_path / "vehicleTypes--frism.csv"
    passenger_vehicle_types.to_csv(passenger_vehicle_types_path, index=False)
    freight_vehicle_types.to_csv(freight_vehicle_types_path, index=False)

    prepared = pd.DataFrame(
        [
            {"linkId": 1, "vehicleTypeId": "pax-car", "process": "RUNEX", "NOx": 1.0, "observations": 2.0},
            {"linkId": 1, "vehicleTypeId": "pax-bus", "process": "RUNEX", "NOx": 1.0, "observations": 2.0},
            {"linkId": 1, "vehicleTypeId": "ft-md", "process": "RUNEX", "NOx": 1.0, "observations": 2.0},
        ]
    )

    passenger_only = _filter_prepared_skims_by_assignment(
        prepared,
        passenger_vehicle_types_path=str(passenger_vehicle_types_path),
        freight_vehicle_types_path=str(freight_vehicle_types_path),
        include_passenger=True,
        include_freight=False,
    )
    assert passenger_only["vehicleTypeId"].tolist() == ["pax-car", "pax-bus"]

    freight_only = _filter_prepared_skims_by_assignment(
        prepared,
        passenger_vehicle_types_path=str(passenger_vehicle_types_path),
        freight_vehicle_types_path=str(freight_vehicle_types_path),
        include_passenger=False,
        include_freight=True,
    )
    assert freight_only["vehicleTypeId"].tolist() == ["ft-md"]


def test_filter_prepared_skims_by_assignment_keeps_transit_with_passenger(tmp_path: Path) -> None:
    passenger_vehicle_types = pd.DataFrame(
        [
            {"vehicleTypeId": "pax-car", "vehicleCategory": "Car"},
            {"vehicleTypeId": "BUS-DEFAULT", "vehicleCategory": "MediumDutyPassenger", "emfacVehicleCategory": "UBUS"},
            {"vehicleTypeId": "RAIL-DEFAULT", "vehicleCategory": "Rail-Default"},
        ]
    )
    freight_vehicle_types = pd.DataFrame(
        [
            {"vehicleTypeId": "ft-md", "vehicleCategory": "Class456Vocational", "vehicleClass": "truck", "vehicleUse": "freight"},
        ]
    )
    passenger_vehicle_types_path = tmp_path / "vehicleTypes--atlas.csv"
    freight_vehicle_types_path = tmp_path / "vehicleTypes--frism.csv"
    passenger_vehicle_types.to_csv(passenger_vehicle_types_path, index=False)
    freight_vehicle_types.to_csv(freight_vehicle_types_path, index=False)

    prepared = pd.DataFrame(
        [
            {"linkId": 1, "vehicleTypeId": "pax-car", "process": "RUNEX", "NOx": 1.0, "observations": 2.0},
            {"linkId": 1, "vehicleTypeId": "BUS-DEFAULT", "process": "RUNEX", "NOx": 1.0, "observations": 2.0},
            {"linkId": 1, "vehicleTypeId": "RAIL-DEFAULT", "process": "RUNEX", "NOx": 1.0, "observations": 2.0},
            {"linkId": 1, "vehicleTypeId": "ft-md", "process": "RUNEX", "NOx": 1.0, "observations": 2.0},
        ]
    )

    passenger_only = _filter_prepared_skims_by_assignment(
        prepared,
        passenger_vehicle_types_path=str(passenger_vehicle_types_path),
        freight_vehicle_types_path=str(freight_vehicle_types_path),
        include_passenger=True,
        include_freight=False,
    )

    assert passenger_only["vehicleTypeId"].tolist() == ["pax-car", "BUS-DEFAULT", "RAIL-DEFAULT"]


def test_build_skims_scale_factors_uses_transit_sample_for_non_bus_transit() -> None:
    prepared = pd.DataFrame(
        [
            {"vehicleTypeId": "pax-car"},
            {"vehicleTypeId": "BUS-DEFAULT"},
            {"vehicleTypeId": "RAIL-DEFAULT"},
            {"vehicleTypeId": "TRAM-SF"},
        ]
    )

    factors = _build_skims_scale_factors(
        prepared,
        population_sample=0.1,
        transit_sample=0.25,
    )

    assert factors.tolist() == [10.0, 4.0, 4.0, 4.0]


def test_prepare_skims_for_grid_allocation_streams_parquet_and_aggregates_across_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_skims = pd.DataFrame(
        [
            {"hour": 7, "linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "NOx": 1.25, "PM2_5": 0.10, "observations": 2.0},
            {"hour": 8, "linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "NOx": 0.75, "PM2_5": 0.20, "observations": 3.0},
            {"hour": 9, "linkId": 202, "vehicleTypeId": "ft-md", "process": "PMBW", "NOx": 4.00, "PM2_5": 0.50, "observations": 1.0},
        ]
    )
    raw_path = tmp_path / "skimsEmissions.parquet"
    pq.write_table(pa.Table.from_pandas(raw_skims, preserve_index=False), raw_path, row_group_size=1)
    output_path = tmp_path / "prepared_skims.parquet"

    original_read_parquet = pd.read_parquet

    def _guarded_read_parquet(path_like, *args, **kwargs):
        if Path(path_like) == raw_path:
            raise AssertionError("prepare_skims_for_grid_allocation should stream parquet batches, not eager-read the raw skims file")
        return original_read_parquet(path_like, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _guarded_read_parquet)
    monkeypatch.setattr(common, "default_chunk_size", 1)

    aggregated = common.prepare_skims_for_grid_allocation(
        str(raw_path),
        str(output_path),
        group_cols=["linkId", "vehicleTypeId", "process"],
        required_pollutants=["NOx", "PM2_5"],
    ).sort_values(["linkId", "vehicleTypeId", "process"]).reset_index(drop=True)

    assert aggregated.to_dict("records") == [
        {
            "linkId": 101,
            "vehicleTypeId": "pax-car",
            "process": "RUNEX",
            "observations": 5.0,
            "NOx": 2.0,
            "PM2_5": 0.30000000000000004,
        },
        {
            "linkId": 202,
            "vehicleTypeId": "ft-md",
            "process": "PMBW",
            "observations": 1.0,
            "NOx": 4.0,
            "PM2_5": 0.5,
        },
    ]

    written = original_read_parquet(output_path).sort_values(["linkId", "vehicleTypeId", "process"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(aggregated, written)
