from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import impacts.common as common
import impacts.pipeline.workflow.prepare_emissions.annualization as annualization_module
import impacts.pipeline.workflow.prepare_emissions.from_skims as prepare_skims_module
from impacts.pipeline.workflow.prepare_emissions.annualization import _build_skims_scale_factors
from impacts.pipeline.workflow.prepare_emissions.annualization import annualize_prepared_skims_for_grid_allocation
from impacts.pipeline.workflow.prepare_emissions.from_skims import _build_zone_allocated_table
from impacts.pipeline.workflow.prepare_emissions.from_skims import _compute_aermod_source_attributes_parquet


def test_configure_duckdb_progress_bar_toggles_settings() -> None:
    class _FakeConnection:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def execute(self, sql: str):
            self.commands.append(sql)
            return self

    con = _FakeConnection()
    common._configure_duckdb_progress_bar(con, enabled=True)
    common._configure_duckdb_progress_bar(con, enabled=False)

    assert con.commands == [
        "SET enable_progress_bar = true",
        "SET progress_bar_time = 0",
        "SET enable_progress_bar = false",
    ]


def test_configure_duckdb_execution_settings_uses_bounded_threads(monkeypatch) -> None:
    class _FakeConnection:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def execute(self, sql: str):
            self.commands.append(sql)
            return self

    monkeypatch.setattr(common.os, "cpu_count", lambda: 12)
    monkeypatch.delenv("IMPACTS_DUCKDB_MEMORY_LIMIT", raising=False)

    con = _FakeConnection()
    common._configure_duckdb_execution_settings(con)

    assert con.commands == [
        "SET preserve_insertion_order = false",
        "SET threads = 4",
    ]


def test_resolve_duckdb_temp_directory_stays_under_impacts_output_tmp_root(tmp_path: Path) -> None:
    output_path = tmp_path / "impacts_output" / "inputs" / "skims" / "prepared_skims.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved = common.resolve_duckdb_temp_directory(output_path)

    assert resolved == tmp_path / "impacts_output" / "_tmp" / "duckdb"
    assert resolved.exists()


def test_build_skims_scale_factors_keeps_transit_unscaled() -> None:
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

    assert factors.tolist() == [10.0, 1.0, 1.0, 1.0]


def test_build_skims_scale_factors_uses_freight_sample_for_freight_vehicle_types() -> None:
    prepared = pd.DataFrame(
        [
            {"vehicleTypeId": "pax-car"},
            {"vehicleTypeId": "ft-md"},
            {"vehicleTypeId": "BUS-DEFAULT"},
        ]
    )

    factors = _build_skims_scale_factors(
        prepared,
        population_sample=0.1,
        transit_sample=1.0,
        freight_sample=1.0 / 101.0,
        assignment_group_by_vehicle_type={
            "pax-car": "passenger",
            "ft-md": "freight",
            "BUS-DEFAULT": "transit",
        },
    )

    assert factors.tolist() == [10.0, 101.0, 1.0]


def test_compute_aermod_source_attributes_preserves_row_release_heights(tmp_path: Path) -> None:
    allocated = pd.DataFrame(
        [
            {
                "linkId": 1,
                "vehicleTypeId": "pax-car",
                "process": "RUNEX",
                "aermod_cell_id": 101,
                "roadCategory": "residential",
                "source_release_height": 1.0,
                "tons_per_year_PM25_aermod_allocated": 0.25,
            },
            {
                "linkId": 2,
                "vehicleTypeId": "freight-truck",
                "process": "RUNEX",
                "aermod_cell_id": 101,
                "roadCategory": "motorway",
                "source_release_height": 3.5,
                "tons_per_year_PM25_aermod_allocated": 0.75,
            },
        ]
    )
    input_path = tmp_path / "aermod_raw.parquet"
    output_path = tmp_path / "aermod_attrs.parquet"
    allocated.to_parquet(input_path, index=False)

    _compute_aermod_source_attributes_parquet(
        str(input_path),
        str(output_path),
        scratch_dir=tmp_path,
        freeway_road_categories=frozenset({"motorway"}),
        cell_population_df=pd.DataFrame(
            {"aermod_cell_id": [101], "source_urban_class": [1000]}
        ),
    )

    result = pd.read_parquet(output_path).sort_values("source_release_height").reset_index(drop=True)

    assert result["vehicleTypeId"].tolist() == ["pax-car", "freight-truck"]
    assert result["source_temporal_class"].tolist() == ["CITYSTREET", "FREEWAY"]
    assert result["source_release_height"].tolist() == [1.0, 3.5]
    assert result["source_urban_class"].tolist() == [1000, 1000]


def test_compute_aermod_source_attributes_defaults_unpopulated_cells_to_rural(tmp_path: Path) -> None:
    allocated = pd.DataFrame(
        [
            {
                "linkId": 1,
                "vehicleTypeId": "pax-car",
                "process": "RUNEX",
                "aermod_cell_id": 101,
                "roadCategory": "residential",
                "source_release_height": 1.0,
                "tons_per_year_PM25_aermod_allocated": 0.25,
            },
            {
                "linkId": 2,
                "vehicleTypeId": "freight-truck",
                "process": "RUNEX",
                "aermod_cell_id": 202,
                "roadCategory": "motorway",
                "source_release_height": 3.5,
                "tons_per_year_PM25_aermod_allocated": 0.75,
            },
        ]
    )
    input_path = tmp_path / "aermod_raw.parquet"
    output_path = tmp_path / "aermod_attrs.parquet"
    allocated.to_parquet(input_path, index=False)

    _compute_aermod_source_attributes_parquet(
        str(input_path),
        str(output_path),
        scratch_dir=tmp_path,
        freeway_road_categories=frozenset({"motorway"}),
        cell_population_df=pd.DataFrame(
            {"aermod_cell_id": [101], "source_urban_class": [1000]}
        ),
    )

    result = pd.read_parquet(output_path).sort_values("aermod_cell_id").reset_index(drop=True)

    assert result["aermod_cell_id"].tolist() == [101, 202]
    assert result["source_temporal_class"].tolist() == ["CITYSTREET", "FREEWAY"]
    assert result["source_release_height"].tolist() == [1.0, 3.5]
    assert result["source_urban_class"].tolist() == [1000, 0]


def test_prepare_skims_for_grid_allocation_aggregates_parquet_without_eager_raw_read(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_skims = pd.DataFrame(
        [
            {"hour": 7, "linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "NOx": 1.25, "PM25": 0.10, "observations": 2.0},
            {"hour": 8, "linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "NOx": 0.75, "PM25": 0.20, "observations": 3.0},
            {"hour": 9, "linkId": 202, "vehicleTypeId": "ft-md", "process": "PMBW", "NOx": 4.00, "PM25": 0.50, "observations": 1.0},
        ]
    )
    raw_path = tmp_path / "skimsEmissions.parquet"
    pq.write_table(pa.Table.from_pandas(raw_skims, preserve_index=False), raw_path, row_group_size=1)
    output_path = tmp_path / "prepared_skims.parquet"

    original_read_parquet = pd.read_parquet

    def _guarded_read_parquet(path_like, *args, **kwargs):
        if Path(path_like) == raw_path:
            raise AssertionError("prepare_skims_for_grid_allocation should not eager-read the raw skims file")
        return original_read_parquet(path_like, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _guarded_read_parquet)

    aggregated = common.prepare_skims_for_grid_allocation(
        str(raw_path),
        str(output_path),
        group_cols=["linkId", "vehicleTypeId", "process"],
        required_pollutants=["NOx", "PM25"],
    ).sort_values(["linkId", "vehicleTypeId", "process"]).reset_index(drop=True)

    assert aggregated.to_dict("records") == [
        {
            "linkId": 101,
            "vehicleTypeId": "pax-car",
            "process": "RUNEX",
            "observations": 5.0,
            "NOx": 2.0,
            "PM25": 0.30000000000000004,
        },
        {
            "linkId": 202,
            "vehicleTypeId": "ft-md",
            "process": "PMBW",
            "observations": 1.0,
            "NOx": 4.0,
            "PM25": 0.5,
        },
    ]

    written = original_read_parquet(output_path).sort_values(["linkId", "vehicleTypeId", "process"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(aggregated, written)


def test_prepare_skims_for_grid_allocation_filters_vehicle_types_during_aggregation(tmp_path: Path) -> None:
    raw_skims = pd.DataFrame(
        [
            {"hour": 7, "linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "NOx": 1.25, "observations": 2.0},
            {"hour": 8, "linkId": 202, "vehicleTypeId": "ft-md", "process": "RUNEX", "NOx": 9.99, "observations": 4.0},
        ]
    )
    raw_path = tmp_path / "skimsEmissions.parquet"
    pq.write_table(pa.Table.from_pandas(raw_skims, preserve_index=False), raw_path, row_group_size=1)
    output_path = tmp_path / "prepared_skims.parquet"

    aggregated = common.prepare_skims_for_grid_allocation(
        str(raw_path),
        str(output_path),
        group_cols=["linkId", "vehicleTypeId", "process"],
        required_pollutants=["NOx"],
        allowed_vehicle_type_ids={"pax-car"},
        known_vehicle_type_ids={"pax-car", "ft-md"},
    ).reset_index(drop=True)

    assert aggregated.to_dict("records") == [
        {
            "linkId": 101,
            "vehicleTypeId": "pax-car",
            "process": "RUNEX",
            "observations": 2.0,
            "NOx": 1.25,
        }
    ]


def test_prepare_skims_for_grid_allocation_parses_compact_emissions_schema(tmp_path: Path) -> None:
    raw_skims = pd.DataFrame(
        [
            {
                "hour": 7,
                "linkId": 101,
                "vehicleTypeId": "pax-car",
                "process": "RUNEX",
                "emissions": "NOx:1.25;PM25:0.10;SOx:0.01",
                "observations": 2.0,
            },
            {
                "hour": 8,
                "linkId": 101,
                "vehicleTypeId": "pax-car",
                "process": "RUNEX",
                "emissions": "NOx:0.75;PM25:0.20",
                "observations": 3.0,
            },
            {
                "hour": 9,
                "linkId": 202,
                "vehicleTypeId": "ft-md",
                "process": "PMBW",
                "emissions": "NOx:4.00;PM25:0.50",
                "observations": 1.0,
            },
        ]
    )
    raw_path = tmp_path / "skimsEmissions.parquet"
    pq.write_table(pa.Table.from_pandas(raw_skims, preserve_index=False), raw_path, row_group_size=1)
    output_path = tmp_path / "prepared_skims.parquet"

    aggregated = common.prepare_skims_for_grid_allocation(
        str(raw_path),
        str(output_path),
        group_cols=["linkId", "vehicleTypeId", "process"],
        required_pollutants=["NOx", "PM25"],
    ).sort_values(["linkId", "vehicleTypeId", "process"]).reset_index(drop=True)

    assert aggregated.to_dict("records") == [
        {
            "linkId": 101,
            "vehicleTypeId": "pax-car",
            "process": "RUNEX",
            "observations": 5.0,
            "NOx": 4.75,
            "PM25": 0.8,
        },
        {
            "linkId": 202,
            "vehicleTypeId": "ft-md",
            "process": "PMBW",
            "observations": 1.0,
            "NOx": 4.0,
            "PM25": 0.5,
        },
    ]


def test_prepare_skims_for_grid_allocation_parses_compact_emissions_with_pm25_and_rog(tmp_path: Path) -> None:
    raw_skims = pd.DataFrame(
        [
            {
                "hour": 7,
                "linkId": 101,
                "vehicleTypeId": "pax-car",
                "process": "RUNEX",
                "emissions": "PM25:0.50;ROG:0.25",
                "observations": 4.0,
            }
        ]
    )
    raw_path = tmp_path / "skimsEmissions.parquet"
    pq.write_table(pa.Table.from_pandas(raw_skims, preserve_index=False), raw_path, row_group_size=1)
    output_path = tmp_path / "prepared_skims.parquet"

    aggregated = common.prepare_skims_for_grid_allocation(
        str(raw_path),
        str(output_path),
        group_cols=["linkId", "vehicleTypeId", "process"],
        required_pollutants=["PM25", "ROG"],
    ).reset_index(drop=True)

    assert aggregated.to_dict("records") == [
        {
            "linkId": 101,
            "vehicleTypeId": "pax-car",
            "process": "RUNEX",
            "observations": 4.0,
            "PM25": 2.0,
            "ROG": 1.0,
        }
    ]


def test_prepare_skims_for_grid_allocation_creates_output_parent_directory(tmp_path: Path) -> None:
    raw_skims = pd.DataFrame(
        [
            {
                "hour": 7,
                "linkId": 101,
                "vehicleTypeId": "pax-car",
                "process": "RUNEX",
                "emissions": "NOx:1.25",
                "observations": 2.0,
            }
        ]
    )
    raw_path = tmp_path / "skimsEmissions.parquet"
    pq.write_table(pa.Table.from_pandas(raw_skims, preserve_index=False), raw_path, row_group_size=1)
    output_path = tmp_path / "nested" / "skims" / "prepared_skims.parquet"

    aggregated = common.prepare_skims_for_grid_allocation(
        str(raw_path),
        str(output_path),
        group_cols=["linkId", "vehicleTypeId", "process"],
        required_pollutants=["NOx"],
    ).reset_index(drop=True)

    assert output_path.exists()
    assert aggregated.to_dict("records") == [
        {
            "linkId": 101,
            "vehicleTypeId": "pax-car",
            "process": "RUNEX",
            "observations": 2.0,
            "NOx": 2.5,
        }
    ]


def test_build_zone_allocated_table_uses_duckdb_and_preserves_allocated_values(tmp_path: Path) -> None:
    grouped = pd.DataFrame(
        [
            {
                "linkId": 101,
                "county_COUNTYFP": "001",
                "county_proportion": 0.25,
                "county_link_length_m": 25.0,
            },
            {
                "linkId": 101,
                "county_COUNTYFP": "013",
                "county_proportion": 0.75,
                "county_link_length_m": 75.0,
            },
        ]
    )
    skims = pd.DataFrame(
        [
            {
                "linkId": 101,
                "vehicleTypeId": " pax-car ",
                "process": " RUNEX ",
                "totVMT": 20.0,
                "totTrips": 4.0,
                "tons_per_year_NOx": 8.0,
            }
        ]
    )

    allocated = (
        _build_zone_allocated_table(
            grouped_df=grouped,
            skims_df=skims,
            zone_label="county",
            scratch_dir=tmp_path,
        )
        .sort_values("county_COUNTYFP")
        .reset_index(drop=True)
    )

    assert allocated.to_dict("records") == [
        {
            "linkId": 101,
            "vehicleTypeId": "pax-car",
            "process": "RUNEX",
            "county_COUNTYFP": "001",
            "county_proportion": 0.25,
            "county_link_length_m": 25.0,
            "totVMT_county_allocated": 5.0,
            "totTrips_county_allocated": 1.0,
            "tons_per_year_NOx_county_allocated": 2.0,
        },
        {
            "linkId": 101,
            "vehicleTypeId": "pax-car",
            "process": "RUNEX",
            "county_COUNTYFP": "013",
            "county_proportion": 0.75,
            "county_link_length_m": 75.0,
            "totVMT_county_allocated": 15.0,
            "totTrips_county_allocated": 3.0,
            "tons_per_year_NOx_county_allocated": 6.0,
        },
    ]


def test_build_zone_allocated_table_uses_supplied_step_label(caplog, tmp_path: Path) -> None:
    grouped = pd.DataFrame(
        [
            {
                "linkId": 101,
                "inmap_cell_id": 9,
                "inmap_proportion": 0.5,
                "inmap_link_length_m": 50.0,
            }
        ]
    )
    skims = pd.DataFrame(
        [
            {
                "linkId": 101,
                "vehicleTypeId": "pax-car",
                "process": "RUNEX",
                "totVMT": 20.0,
                "totTrips": 4.0,
                "tons_per_year_NOx": 8.0,
            }
        ]
    )

    with caplog.at_level("INFO", logger="impacts.pipeline.workflow.prepare_emissions.from_skims"):
        allocated = _build_zone_allocated_table(
            grouped_df=grouped,
            skims_df=skims,
            zone_label="inmap",
            scratch_dir=tmp_path,
            step_id="1.4",
        )

    assert allocated is not None
    assert "Step 1.4" in caplog.text
    assert "allocated across inmap rows=1" in caplog.text


def test_prepare_staged_skims_for_processing_reuses_grouped_intermediate(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "inputs"
    grouped_path = common.prepared_table_target(input_root, "prepared_skims_grouped_for_grid_allocation")
    grouped_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "observations": 5.0, "NOx": 2.0},
        ]
    ).to_parquet(grouped_path, index=False)

    def _fail_prepare(*args, **kwargs):
        raise AssertionError("prepare_skims_for_grid_allocation should not rerun when grouped prepared skims already exist")

    def _stub_annualize(*, prepared_skims_path: str, output_path: str, **kwargs):
        assert Path(prepared_skims_path) == grouped_path
        pd.DataFrame(
            [
                {"linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "totTrips": 10.0, "totVMT": 3.0, "tons_per_year_NOx": 0.1},
            ]
        ).to_parquet(output_path, index=False)

    monkeypatch.setattr(prepare_skims_module, "prepare_skims_for_grid_allocation", _fail_prepare)
    monkeypatch.setattr(prepare_skims_module, "annualize_prepared_skims_for_grid_allocation", _stub_annualize)

    result = prepare_skims_module.prepare_staged_skims_for_processing(
        input_root=input_root,
        skims_input_source=str(tmp_path / "raw.skims.parquet"),
        network_path=str(tmp_path / "network.parquet"),
        passenger_vehicle_types_path=None,
        freight_vehicle_types_path=None,
        beam_length_col="length",
        prepared_skims_group_cols=["linkId", "vehicleTypeId", "process"],
        pollutants=["NOx"],
        vehicle_category_metadata_file=str(tmp_path / "vehicle_categories.csv"),
        annualization_days={"light_duty": 327.0, "medium_heavy_duty": 312.0},
        population_sample=1.0,
        transit_sample=1.0,
        freight_sample=None,
        include_passenger=True,
        include_freight=True,
    )

    assert result.to_dict("records") == [
        {"linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "totTrips": 10.0, "totVMT": 3.0, "tons_per_year_NOx": 0.1},
    ]


def test_load_or_prepare_skims_df_rejects_stale_prepared_cache_for_aermod(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    prepared_path = common.prepared_table_target(input_root, "prepared_skims_for_grid_allocation")
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "totTrips": 10.0, "totVMT": 3.0, "tons_per_year_NOx": 0.1},
        ]
    ).to_parquet(prepared_path, index=False)

    with pytest.raises(ValueError, match="Prepared skims cache is stale for AERMOD processing and must be rebuilt"):
        prepare_skims_module.load_or_prepare_skims_df(
            input_root=input_root,
            intersection_path="",
            beam_length_col="length",
            prepared_skims_group_cols=["linkId", "vehicleTypeId", "process"],
            pollutants=["NOx"],
            vehicle_category_metadata_file=str(tmp_path / "vehicle_categories.csv"),
            annualization_days={"light_duty": 327.0},
            population_sample=1.0,
            transit_sample=1.0,
            freight_sample=None,
            include_passenger=True,
            include_freight=False,
            manifest_inputs=None,
            require_aermod_support=True,
        )


def test_prepare_staged_skims_for_processing_rebuilds_invalid_grouped_intermediate(tmp_path: Path, monkeypatch) -> None:
    input_root = tmp_path / "inputs"
    grouped_path = common.prepared_table_target(input_root, "prepared_skims_grouped_for_grid_allocation")
    grouped_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_path.write_bytes(b"")
    called = {"prepare": 0}

    def _stub_prepare(*args, **kwargs):
        called["prepare"] += 1
        pd.DataFrame(
            [
                {"linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "observations": 5.0, "NOx": 2.0},
            ]
        ).to_parquet(grouped_path, index=False)
        return pd.read_parquet(grouped_path)

    def _stub_annualize(*, prepared_skims_path: str, output_path: str, **kwargs):
        assert Path(prepared_skims_path) == grouped_path
        pd.DataFrame(
            [
                {"linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "totTrips": 10.0, "totVMT": 3.0, "tons_per_year_NOx": 0.1},
            ]
        ).to_parquet(output_path, index=False)

    monkeypatch.setattr(prepare_skims_module, "prepare_skims_for_grid_allocation", _stub_prepare)
    monkeypatch.setattr(prepare_skims_module, "annualize_prepared_skims_for_grid_allocation", _stub_annualize)

    result = prepare_skims_module.prepare_staged_skims_for_processing(
        input_root=input_root,
        skims_input_source=str(tmp_path / "raw.skims.parquet"),
        network_path=str(tmp_path / "network.parquet"),
        passenger_vehicle_types_path=None,
        freight_vehicle_types_path=None,
        beam_length_col="length",
        prepared_skims_group_cols=["linkId", "vehicleTypeId", "process"],
        pollutants=["NOx"],
        vehicle_category_metadata_file=str(tmp_path / "vehicle_categories.csv"),
        annualization_days={"light_duty": 327.0, "medium_heavy_duty": 312.0},
        population_sample=1.0,
        transit_sample=1.0,
        freight_sample=None,
        include_passenger=True,
        include_freight=True,
    )

    assert called["prepare"] == 1
    assert result.to_dict("records") == [
        {"linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "totTrips": 10.0, "totVMT": 3.0, "tons_per_year_NOx": 0.1},
    ]


def test_annualize_prepared_skims_uses_grouped_input_without_eager_read(tmp_path: Path, monkeypatch) -> None:
    grouped = pd.DataFrame(
        [
            {"linkId": 101, "vehicleTypeId": "pax-car", "process": "RUNEX", "observations": 2.0, "NOx": 10.0},
            {"linkId": 102, "vehicleTypeId": "ft-md", "process": "RUNEX", "observations": 3.0, "NOx": 30.0},
        ]
    )
    grouped_path = tmp_path / "prepared_grouped.parquet"
    pq.write_table(pa.Table.from_pandas(grouped, preserve_index=False), grouped_path, row_group_size=1)

    network = pd.DataFrame(
        [
            {"linkId": 101, "length": 1609.344, "attributeOrigType": "local"},
            {"linkId": 102, "length": 3218.688, "attributeOrigType": "arterial"},
        ]
    )
    network_path = tmp_path / "network.parquet"
    network.to_parquet(network_path, index=False)

    passenger_vehicle_types = pd.DataFrame(
        [{"vehicleTypeId": "pax-car", "vehicleCategory": "Car", "emfacVehicleCategory": "LDA"}]
    )
    freight_vehicle_types = pd.DataFrame(
        [{"vehicleTypeId": "ft-md", "vehicleCategory": "Class456Vocational", "vehicleClass": "truck", "vehicleUse": "freight", "emfacVehicleCategory": "Class 4-6 Vocational"}]
    )
    passenger_path = tmp_path / "vehicleTypes--atlas.csv"
    freight_path = tmp_path / "vehicleTypes--frism.csv"
    passenger_vehicle_types.to_csv(passenger_path, index=False)
    freight_vehicle_types.to_csv(freight_path, index=False)

    metadata = pd.DataFrame(
        [
            {"emfac_vehicle_category": "LDA", "operation_days_per_year": 327.0, "tailpipe_height_meters": 1.0},
            {"emfac_vehicle_category": "Class 4-6 Vocational", "operation_days_per_year": 312.0, "tailpipe_height_meters": 3.5},
        ]
    )
    metadata_path = tmp_path / "vehicle_categories.csv"
    metadata.to_csv(metadata_path, index=False)
    output_path = tmp_path / "annualized.parquet"

    original_read_parquet = pd.read_parquet

    def _guarded_read_parquet(path_like, *args, **kwargs):
        if Path(path_like) == grouped_path:
            raise AssertionError("annualize_prepared_skims_for_grid_allocation should not eager-read the grouped skims input")
        return original_read_parquet(path_like, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _guarded_read_parquet)
    result = annualize_prepared_skims_for_grid_allocation(
        prepared_skims_path=str(grouped_path),
        output_path=str(output_path),
        network_path=str(network_path),
        beam_length_col="length",
        group_cols=["linkId", "vehicleTypeId", "process"],
        required_pollutants=["NOx"],
        vehicle_category_metadata_file=str(metadata_path),
        annualization_days={"light_duty": 327.0, "medium_heavy_duty": 312.0},
        passenger_vehicle_types_path=str(passenger_path),
        freight_vehicle_types_path=str(freight_path),
        population_sample=1.0,
        transit_sample=1.0,
        freight_sample=0.5,
    )

    assert list(result.columns) == [
        "linkId",
        "vehicleTypeId",
        "process",
        "roadCategory",
        "source_release_height",
        "totTrips",
        "totVMT",
        "tons_per_year_NOx",
    ]

    written = original_read_parquet(output_path).sort_values(["linkId"]).reset_index(drop=True)
    assert written["linkId"].tolist() == [101, 102]
    assert written["vehicleTypeId"].tolist() == ["pax-car", "ft-md"]
    assert written["process"].tolist() == ["RUNEX", "RUNEX"]
    assert written["roadCategory"].tolist() == ["local", "arterial"]
    assert written["source_release_height"].tolist() == [1.0, 3.5]
    assert written["totTrips"].tolist() == [654.0, 1872.0]
    assert written["totVMT"].tolist() == pytest.approx([654.0, 3744.0])
    assert written["tons_per_year_NOx"].tolist() == pytest.approx(
        [10.0 * 327.0 / 907184.74, 30.0 * 2.0 * 312.0 / 907184.74]
    )


def test_tailpipe_height_lookup_requires_metadata_height_column(tmp_path: Path) -> None:
    passenger_path = tmp_path / "vehicleTypes--atlas.csv"
    freight_path = tmp_path / "vehicleTypes--frism.csv"
    pd.DataFrame(
        [{"vehicleTypeId": "pax-car", "emfacVehicleCategory": "LDA"}]
    ).to_csv(passenger_path, index=False)
    pd.DataFrame(
        [{"vehicleTypeId": "ft-md", "emfacVehicleCategory": "T7 Tractor"}]
    ).to_csv(freight_path, index=False)
    metadata_path = tmp_path / "vehicle_categories.csv"
    pd.DataFrame(
        [{"emfac_vehicle_category": "LDA", "operation_days_per_year": 327.0}]
    ).to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="tailpipe_height_meters"):
        annualization_module._resolve_vehicle_type_tailpipe_height_lookup(
            vehicle_category_metadata_file=str(metadata_path),
            passenger_vehicle_types_path=str(passenger_path),
            freight_vehicle_types_path=str(freight_path),
        )


def test_tailpipe_height_lookup_rejects_conflicting_vehicle_type_categories(tmp_path: Path) -> None:
    passenger_path = tmp_path / "vehicleTypes--atlas.csv"
    freight_path = tmp_path / "vehicleTypes--frism.csv"
    pd.DataFrame(
        [{"vehicleTypeId": "dup-type", "emfacVehicleCategory": "LDA"}]
    ).to_csv(passenger_path, index=False)
    pd.DataFrame(
        [{"vehicleTypeId": "dup-type", "emfacVehicleCategory": "T7 Tractor"}]
    ).to_csv(freight_path, index=False)
    metadata_path = tmp_path / "vehicle_categories.csv"
    pd.DataFrame(
        [
            {"emfac_vehicle_category": "LDA", "tailpipe_height_meters": 1.0},
            {"emfac_vehicle_category": "T7 Tractor", "tailpipe_height_meters": 3.5},
        ]
    ).to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="conflicting emfacVehicleCategory"):
        annualization_module._resolve_vehicle_type_tailpipe_height_lookup(
            vehicle_category_metadata_file=str(metadata_path),
            passenger_vehicle_types_path=str(passenger_path),
            freight_vehicle_types_path=str(freight_path),
        )
