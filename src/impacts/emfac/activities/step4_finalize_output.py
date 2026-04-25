from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from impacts.emfac.config import _apply_table_schema
from impacts.emfac.activities.step3_fill_project_analysis_rates import ACTIVITY_COLUMN
from impacts.emfac.activities.step3_fill_project_analysis_rates import POLLUTANT_COLUMNS
from impacts.emfac.common import frame_summary
from impacts.emfac.common import write_trace
from impacts.emfac.common import assign_model_year_groups
from impacts.emfac.common import model_year_group_id_component

ACTIVITY_JOIN_COLUMNS = ["county", "vehicleCategory", "fuel", "modelYear"]
PTO_PROCESS_NAME = "PTOEX"
FINAL_RATE_GROUP_COLUMNS = [
    "county",
    "vehicleCategory",
    "fuel",
    "modelYear",
    "process",
    ACTIVITY_COLUMN,
    "roadCategory",
]
VMT_WEIGHTED_PROCESSES = {"RUNEX", "PMBW", "PMTW", PTO_PROCESS_NAME}
ACTIVITY_COLUMNS = [
    "total_vmt_vehicle_miles_per_year",
    "cvmt_vehicle_miles_per_year",
    "evmt_vehicle_miles_per_year",
    "population_vehicles",
    "trips_per_year",
    "pto_total_vmt_vehicle_miles_per_year",
]
_RATES_STORE_NUMERIC_DIMENSION_COLUMNS = {
    "modelYear",
    "speedMph_timeMin",
    "speed_mph_float_bins",
    "time_minutes_float_bins",
}
_RATES_SCHEMA_STRING_COLUMNS = [
    "county",
    "vehicleCategory",
    "fuel",
    "modelYear",
    "process",
    ACTIVITY_COLUMN,
    "roadCategory",
    "speedMph_timeMin",
    "emfacId",
]
_SURFACE_BASE_SCHEMA = {
    "county": "string",
    "vehicleCategory": "string",
    "fuel": "string",
    "modelYear": "Int64",
    "process": "string",
}
_ACTIVITY_WEIGHTS_BASE_SCHEMA = {
    "county": "string",
    "vehicleCategory": "string",
    "fuel": "string",
    "modelYear": "Int64",
}


def _enforce_surface_schema(frame: pd.DataFrame) -> pd.DataFrame:
    schema = dict(_SURFACE_BASE_SCHEMA)
    if ACTIVITY_COLUMN in frame.columns:
        schema[ACTIVITY_COLUMN] = "string"
    if "roadCategory" in frame.columns:
        schema["roadCategory"] = "string"
    if "speed" in frame.columns:
        schema["speed"] = "Float64"
    numeric_columns = [column for column in POLLUTANT_COLUMNS if column in frame.columns]
    for column in numeric_columns:
        schema[column] = "Float64"
    return _apply_table_schema(frame, schema, frame_name="EMFAC finalization surface")


def _enforce_activity_weights_schema(frame: pd.DataFrame) -> pd.DataFrame:
    schema = dict(_ACTIVITY_WEIGHTS_BASE_SCHEMA)
    if "speed" in frame.columns:
        schema["speed"] = "Float64"
    for column in [column for column in ACTIVITY_COLUMNS if column in frame.columns]:
        schema[column] = "Float64"
    return _apply_table_schema(frame, schema, frame_name="EMFAC activity weights")


def _enforce_rates_schema(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in [column for column in _RATES_SCHEMA_STRING_COLUMNS if column in result.columns]:
        result[column] = result[column].astype("string")
    for column in [column for column in POLLUTANT_COLUMNS if column in result.columns]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result

def _assign_model_year_groups(frame: pd.DataFrame, model_year_groups: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    return assign_model_year_groups(frame, model_year_groups)


def _load_surface(path: str) -> pd.DataFrame:
    return _enforce_surface_schema(pd.read_parquet(Path(path).expanduser().resolve()))


def _build_activity_weights(emissions_inventory_path: str) -> pd.DataFrame:
    source = pd.read_parquet(Path(emissions_inventory_path).expanduser().resolve())
    available_columns = [column for column in ["county", "vehicleCategory", "fuel", "modelYear", "speed", *ACTIVITY_COLUMNS] if column in source.columns]
    weights = _enforce_activity_weights_schema(source[available_columns].copy())
    if "speed" in weights.columns:
        weights = weights.drop_duplicates(["county", "vehicleCategory", "fuel", "modelYear", "speed"])
    else:
        weights = weights.drop_duplicates(ACTIVITY_JOIN_COLUMNS)
    grouped = (
        weights.groupby(ACTIVITY_JOIN_COLUMNS, dropna=False)[[column for column in ACTIVITY_COLUMNS if column in weights.columns]]
        .sum(min_count=1)
        .reset_index()
    )
    for column in ACTIVITY_COLUMNS:
        if column not in grouped.columns:
            grouped[column] = pd.NA
    return grouped[ACTIVITY_JOIN_COLUMNS + ACTIVITY_COLUMNS]


def _filter_activity_weights_to_surface(activity_weights: pd.DataFrame, surface: pd.DataFrame) -> pd.DataFrame:
    keys = surface[ACTIVITY_JOIN_COLUMNS].drop_duplicates().copy()
    return activity_weights.merge(keys, on=ACTIVITY_JOIN_COLUMNS, how="inner")


def _build_study_area_wide_fleet(activity_weights: pd.DataFrame, model_year_groups: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    fleet = _assign_model_year_groups(
        activity_weights[["vehicleCategory", "fuel", "modelYear", "total_vmt_vehicle_miles_per_year"]].copy(),
        model_year_groups,
    )
    fleet = (
        fleet.groupby(["vehicleCategory", "fuel", "modelYear"], dropna=False)["total_vmt_vehicle_miles_per_year"]
        .sum(min_count=1)
        .reset_index()
    )
    total_vmt = fleet["total_vmt_vehicle_miles_per_year"].sum(min_count=1)
    fleet["vmtShare"] = (
        fleet["total_vmt_vehicle_miles_per_year"] / total_vmt
        if pd.notna(total_vmt) and total_vmt > 0 else pd.NA
    )
    if pd.notna(total_vmt) and total_vmt > 0 and not fleet.empty:
        fleet["vmtShare"] = fleet["vmtShare"].round(12)
        rounded_total = fleet["vmtShare"].sum(min_count=1)
        if pd.notna(rounded_total):
            remainder = round(1.0 - float(rounded_total), 12)
            if remainder != 0.0:
                max_idx = fleet["vmtShare"].idxmax()
                fleet.loc[max_idx, "vmtShare"] = round(float(fleet.loc[max_idx, "vmtShare"]) + remainder, 12)
    return fleet.drop(columns=["total_vmt_vehicle_miles_per_year"])


def _prepare_surface_keys(surface: pd.DataFrame, model_year_groups: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    keys = surface[["county", "vehicleCategory", "fuel", "modelYear", "process"]].drop_duplicates().copy()
    return _assign_model_year_groups(keys, model_year_groups).drop_duplicates()


def _build_matching_activity_table(
    activity_weights: pd.DataFrame,
    model_year_groups: dict[str, list[dict[str, object]]],
) -> pd.DataFrame:
    grouped = _assign_model_year_groups(activity_weights.copy(), model_year_groups)
    aggregated = (
        grouped.groupby(["vehicleCategory", "fuel", "modelYear"], dropna=False)[ACTIVITY_COLUMNS]
        .sum(min_count=1)
        .reset_index()
    )
    return aggregated[["vehicleCategory", "fuel", "modelYear", *ACTIVITY_COLUMNS]]


def _build_aggregated_activity_table(
    activity_weights: pd.DataFrame,
    surface: pd.DataFrame,
    model_year_groups: dict[str, list[dict[str, object]]],
) -> pd.DataFrame:
    grouped = _assign_model_year_groups(activity_weights.copy(), model_year_groups)
    aggregated = (
        grouped.groupby(["county", "modelYear"], dropna=False)[ACTIVITY_COLUMNS]
        .sum(min_count=1)
        .reset_index()
    )
    process_keys = _prepare_surface_keys(surface, model_year_groups)[["county", "modelYear", "process"]].drop_duplicates()
    return aggregated.merge(process_keys, on=["county", "modelYear"], how="inner")[
        ["county", "modelYear", "process", *ACTIVITY_COLUMNS]
    ].drop_duplicates()


def _build_activity_by_emfac_id_table(
    activity_weights: pd.DataFrame,
    surface: pd.DataFrame,
    model_year_groups: dict[str, list[dict[str, object]]],
) -> pd.DataFrame:
    grouped = _assign_model_year_groups(activity_weights.copy(), model_year_groups)
    process_keys = _prepare_surface_keys(surface, model_year_groups)[
        ["county", "vehicleCategory", "fuel", "modelYear", "process"]
    ].drop_duplicates()
    prepared = grouped.merge(
        process_keys,
        on=["county", "vehicleCategory", "fuel", "modelYear"],
        how="inner",
    )
    prepared["emfacId"] = prepared.apply(
        lambda row: _build_emfac_id(
            vehicle_category=row["vehicleCategory"],
            fuel=row["fuel"],
            model_year=row["modelYear"],
        ),
        axis=1,
    )
    aggregated = (
        prepared.groupby(["county", "emfacId", "process"], dropna=False)[ACTIVITY_COLUMNS]
        .sum(min_count=1)
        .reset_index()
    )
    return aggregated[["county", "emfacId", "process", *ACTIVITY_COLUMNS]]


def _build_inventory_final_fleet_table(
    activity_weights: pd.DataFrame,
    model_year_groups: dict[str, list[dict[str, object]]],
) -> pd.DataFrame:
    return _build_study_area_wide_fleet(activity_weights, model_year_groups)[
        ["vehicleCategory", "fuel", "modelYear", "vmtShare"]
    ].drop_duplicates()


def _aggregation_weight_for_process(process: pd.Series, frame: pd.DataFrame) -> pd.Series:
    weights = pd.Series(np.nan, index=frame.index, dtype="float64")
    process_values = process.astype(str)
    vmt_mask = process_values.isin(VMT_WEIGHTED_PROCESSES)
    prdust_mask = process_values == "PRDUST"
    weights.loc[vmt_mask] = pd.to_numeric(
        frame.loc[vmt_mask, "total_vmt_vehicle_miles_per_year"], errors="coerce"
    ).to_numpy(dtype="float64", na_value=np.nan)
    weights.loc[prdust_mask] = pd.to_numeric(
        frame.loc[prdust_mask, "population_vehicles"], errors="coerce"
    ).to_numpy(dtype="float64", na_value=np.nan)
    other_mask = ~vmt_mask & ~prdust_mask
    weights.loc[other_mask] = pd.to_numeric(
        frame.loc[other_mask, "trips_per_year"], errors="coerce"
    ).to_numpy(dtype="float64", na_value=np.nan)
    return weights


def _aggregate_pollutant_column(frame: pd.DataFrame, pollutant_column: str) -> pd.DataFrame:
    values = pd.to_numeric(frame[pollutant_column], errors="coerce")
    valid_weight = frame["aggregation_weight"].notna() & (frame["aggregation_weight"] > 0) & values.notna()

    weighted = pd.DataFrame(columns=FINAL_RATE_GROUP_COLUMNS + [pollutant_column])
    if valid_weight.any():
        weighted = frame.loc[valid_weight, FINAL_RATE_GROUP_COLUMNS].copy()
        weighted["weighted_rate"] = values.loc[valid_weight] * frame.loc[valid_weight, "aggregation_weight"]
        weighted["aggregation_weight"] = frame.loc[valid_weight, "aggregation_weight"]
        weighted = (
            weighted.groupby(FINAL_RATE_GROUP_COLUMNS, dropna=False)[["weighted_rate", "aggregation_weight"]]
            .sum()
            .reset_index()
        )
        weighted[pollutant_column] = weighted["weighted_rate"] / weighted["aggregation_weight"]
        weighted = weighted[FINAL_RATE_GROUP_COLUMNS + [pollutant_column]]

    fallback = pd.DataFrame(columns=FINAL_RATE_GROUP_COLUMNS + [pollutant_column])
    fallback_mask = ~valid_weight & values.notna()
    if fallback_mask.any():
        fallback = (
            frame.loc[fallback_mask, FINAL_RATE_GROUP_COLUMNS + [pollutant_column]]
            .groupby(FINAL_RATE_GROUP_COLUMNS, dropna=False)[pollutant_column]
            .mean()
            .reset_index()
        )
        if not weighted.empty:
            fallback = fallback.merge(
                weighted[FINAL_RATE_GROUP_COLUMNS].drop_duplicates(),
                on=FINAL_RATE_GROUP_COLUMNS,
                how="left",
                indicator=True,
            )
            fallback = fallback.loc[fallback["_merge"] == "left_only"].drop(columns="_merge")

    return pd.concat([weighted, fallback], ignore_index=True)


def _build_final_rate_table(
    surface: pd.DataFrame,
    activity_weights: pd.DataFrame,
    *,
    model_year_groups: dict[str, list[dict[str, object]]],
) -> pd.DataFrame:
    frame = surface.copy()
    frame = frame.merge(activity_weights, on=ACTIVITY_JOIN_COLUMNS, how="left")
    frame = _assign_model_year_groups(frame, model_year_groups)
    frame["aggregation_weight"] = _aggregation_weight_for_process(frame["process"], frame)

    aggregated_frames: list[pd.DataFrame] = []
    for pollutant_column in [column for column in POLLUTANT_COLUMNS if column in frame.columns]:
        aggregated = _aggregate_pollutant_column(frame, pollutant_column)
        if not aggregated.empty:
            aggregated_frames.append(aggregated)

    if not aggregated_frames:
        return pd.DataFrame(columns=FINAL_RATE_GROUP_COLUMNS + POLLUTANT_COLUMNS)

    result = aggregated_frames[0]
    for aggregated in aggregated_frames[1:]:
        result = result.merge(aggregated, on=FINAL_RATE_GROUP_COLUMNS, how="outer")
    value_columns = [column for column in POLLUTANT_COLUMNS if column in result.columns]
    result = result.loc[result[value_columns].notna().any(axis=1)].reset_index(drop=True)
    return _enforce_rates_schema(result)


def _write_parquet(frame: pd.DataFrame, path: str) -> str:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported output format for {target}. Expected .parquet")
    frame.to_parquet(target, index=False)
    return str(target)


def _missing_rate_summary(frame: pd.DataFrame) -> dict[str, object]:
    available_columns = [column for column in POLLUTANT_COLUMNS if column in frame.columns]
    if not available_columns:
        return {
            "row_count": int(len(frame)),
            "pollutant_column_count": 0,
            "expected_rate_cells": 0,
            "filled_rate_cells": 0,
            "missing_rate_cells": 0,
            "rows_with_any_rate": 0,
            "rows_with_no_rates": int(len(frame)),
        }

    non_null = frame[available_columns].notna()
    rows_with_any_rate = non_null.any(axis=1)
    return {
        "row_count": int(len(frame)),
        "pollutant_column_count": len(available_columns),
        "expected_rate_cells": int(len(frame) * len(available_columns)),
        "filled_rate_cells": int(non_null.sum().sum()),
        "missing_rate_cells": int((~non_null).sum().sum()),
        "rows_with_any_rate": int(rows_with_any_rate.sum()),
        "rows_with_no_rates": int((~rows_with_any_rate).sum()),
    }


def _print_model_year_group_stats(
    final_rates: pd.DataFrame,
    aggregated_activity: pd.DataFrame,
    fleet: pd.DataFrame,
) -> None:
    print("    4.5 Model year group stats")
    if "modelYear" in final_rates.columns:
        print("      Final rates rows by modelYear:")
        for model_year, count in final_rates["modelYear"].value_counts(dropna=False).sort_index().items():
            print(f"        {model_year}: {int(count):,}")
    if "modelYear" in aggregated_activity.columns:
        print("      Activity rows by modelYear:")
        for model_year, count in aggregated_activity["modelYear"].value_counts(dropna=False).sort_index().items():
            print(f"        {model_year}: {int(count):,}")
    if "modelYear" in fleet.columns:
        print("      Fleet rows by modelYear:")
        for model_year, count in fleet["modelYear"].value_counts(dropna=False).sort_index().items():
            print(f"        {model_year}: {int(count):,}")


def _build_emfac_id(*, vehicle_category: object, fuel: object, model_year: object) -> str:
    def _sanitize_emfac_component(value: object) -> str:
        return "".join(ch for ch in str("" if pd.isna(value) else value).strip() if ch.isalnum())

    return (
        f"{_sanitize_emfac_component(model_year_group_id_component(model_year))}"
        f"{_sanitize_emfac_component(vehicle_category)}"
        f"{_sanitize_emfac_component(fuel)}"
    )


def _complete_sparse_counties_for_rates_store(frame: pd.DataFrame, *, expected_counties: list[str]) -> pd.DataFrame:
    if "county" not in frame.columns or len(expected_counties) <= 1:
        return frame

    result = frame.copy()
    result["county"] = result["county"].astype(str).str.strip()
    value_columns = [
        column
        for column in result.columns
        if column not in {"county", "emfacId"}
        and column not in _RATES_STORE_NUMERIC_DIMENSION_COLUMNS
        and pd.api.types.is_numeric_dtype(result[column])
    ]
    if not value_columns:
        return result
    group_columns = [column for column in result.columns if column not in {"county", *value_columns}]
    slice_means = result.groupby(group_columns, dropna=False)[value_columns].mean(numeric_only=True).reset_index()
    observed = result[group_columns + ["county"]].drop_duplicates()
    county_frame = pd.DataFrame({"county": expected_counties, "_join_key": 1})
    missing = (
        slice_means.assign(_join_key=1)
        .merge(county_frame, on="_join_key", how="inner")
        .drop(columns="_join_key")
        .merge(observed.assign(_observed=True), on=group_columns + ["county"], how="left")
    )
    missing = missing.loc[missing["_observed"].isna()].drop(columns="_observed")
    if missing.empty:
        return result
    return pd.concat([result, missing[result.columns]], ignore_index=True)


def _column_exists(con: duckdb.DuckDBPyConnection, table_name: str, column_name: str) -> bool:
    rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return any(str(row[1]) == column_name for row in rows)


def _summarize_rates_store_county_coverage(frame: pd.DataFrame) -> dict[str, object]:
    if "county" not in frame.columns or "emfacId" not in frame.columns:
        return {
            "partition_count": int(frame["emfacId"].nunique(dropna=True)) if "emfacId" in frame.columns else 0,
            "full_county_partition_count": 0,
            "partial_county_partition_count": 0,
            "sample_partial_partitions": [],
        }
    working = frame.copy()
    working["county"] = working["county"].astype(str).str.strip()
    coverage = (
        working.groupby("emfacId", dropna=False)["county"]
        .nunique(dropna=True)
        .sort_values()
    )
    expected_count = int(working["county"].nunique(dropna=True))
    partial = coverage[coverage < expected_count]
    return {
        "expected_county_count": expected_count,
        "partition_count": int(len(coverage)),
        "full_county_partition_count": int((coverage == expected_count).sum()),
        "partial_county_partition_count": int(len(partial)),
        "sample_partial_partitions": [
            {"emfacId": str(emfac_id), "county_count": int(count)}
            for emfac_id, count in partial.head(20).items()
        ],
    }


def _build_rates_store_duckdb(*, parquet_root: Path, duckdb_path: Path) -> Path:
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_glob = (parquet_root / "**" / "*.parquet").as_posix()
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute("DROP TABLE IF EXISTS emfac_rates")
        con.execute(
            """
            CREATE TABLE emfac_rates AS
            SELECT *
            FROM read_parquet(?, hive_partitioning = true, union_by_name = true)
            """,
            [parquet_glob],
        )
        con.execute("CREATE INDEX IF NOT EXISTS emfac_rates_emfac_id_idx ON emfac_rates (emfacId)")
        if _column_exists(con, "emfac_rates", "county"):
            con.execute("CREATE INDEX IF NOT EXISTS emfac_rates_county_idx ON emfac_rates (county)")
        if _column_exists(con, "emfac_rates", "process"):
            con.execute("CREATE INDEX IF NOT EXISTS emfac_rates_process_idx ON emfac_rates (process)")
    finally:
        con.close()
    return duckdb_path


def _write_rates_store_from_dataframe(
    *,
    rates: pd.DataFrame,
    output_dir: str | Path,
    compression: str = "zstd",
) -> dict[str, object]:
    output_root = Path(output_dir).resolve()
    parquet_root = output_root / "dataset"
    duckdb_path = output_root / "dataset.duckdb"
    parquet_root.mkdir(parents=True, exist_ok=True)

    working = rates.copy()
    working["emfacId"] = working["emfacId"].astype(str)
    expected_counties = (
        sorted(working["county"].dropna().astype(str).str.strip().unique().tolist())
        if "county" in working.columns else []
    )
    pre_write_summary = _summarize_rates_store_county_coverage(working)
    written: list[Path] = []
    relative_paths: dict[str, str] = {}
    completed_frames: list[pd.DataFrame] = []
    for emfac_id, frame in working.groupby("emfacId", dropna=False):
        frame = _complete_sparse_counties_for_rates_store(frame, expected_counties=expected_counties)
        completed_frames.append(frame)
        emfac_id_str = str(emfac_id)
        partition_dir = parquet_root / f"emfacId={emfac_id_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        output_path = partition_dir / f"{emfac_id_str}.parquet"
        parquet_frame = frame.drop(columns=["emfacId"], errors="ignore").reset_index(drop=True)
        table = pa.Table.from_pandas(parquet_frame, preserve_index=False)
        pq.write_table(table, output_path, compression=compression)
        written.append(output_path)
        relative_paths[emfac_id_str] = str(output_path.relative_to(output_root))

    built_duckdb = _build_rates_store_duckdb(parquet_root=parquet_root, duckdb_path=duckdb_path)
    post_write_summary = _summarize_rates_store_county_coverage(
        pd.concat(completed_frames, ignore_index=True, sort=False) if completed_frames else working.iloc[0:0].copy()
    )
    return {
        "parquet_file_count": len(written),
        "output_dir": str(output_root),
        "parquet_root": str(parquet_root),
        "duckdb_path": str(built_duckdb),
        "relative_paths": relative_paths,
        "pre_write_county_coverage": pre_write_summary,
        "post_write_county_coverage": post_write_summary,
    }


def _prepare_rates_store_frame(
    *,
    passenger_rates: pd.DataFrame,
    freight_rates: pd.DataFrame,
) -> pd.DataFrame:
    rates = pd.concat([passenger_rates, freight_rates], ignore_index=True, sort=False).copy()
    rates["emfacId"] = rates.apply(
        lambda row: _build_emfac_id(
            vehicle_category=row["vehicleCategory"],
            fuel=row["fuel"],
            model_year=row["modelYear"],
        ),
        axis=1,
    )
    return _enforce_rates_schema(rates)


def _finalize_group_outputs(
    *,
    group_name: str,
    surface: pd.DataFrame,
    activity_weights: pd.DataFrame,
    model_year_groups: dict[str, list[dict[str, object]]],
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], dict[str, object]]:
    group_activity_weights = _filter_activity_weights_to_surface(activity_weights, surface)
    matching_activity = _build_matching_activity_table(
        group_activity_weights,
        model_year_groups,
    )
    final_rates = _build_final_rate_table(
        surface,
        group_activity_weights,
        model_year_groups=model_year_groups,
    )
    aggregated_activity = _build_aggregated_activity_table(
        group_activity_weights,
        surface,
        model_year_groups,
    )
    activity_by_emfac_id = _build_activity_by_emfac_id_table(
        group_activity_weights,
        surface,
        model_year_groups,
    )
    fleet = _build_inventory_final_fleet_table(
        group_activity_weights,
        model_year_groups,
    )
    trace_payload = {
        f"{group_name}_filled_surface": frame_summary(surface, name=f"filled_project_analysis_{group_name}"),
        f"{group_name}_activity_weights": frame_summary(group_activity_weights, name=f"activity_weights_{group_name}"),
        f"{group_name}_matching_activity": frame_summary(matching_activity, name=f"inventory_matching_activity_{group_name}"),
        f"{group_name}_final_rates": frame_summary(final_rates, name=f"final_horizontal_rates_{group_name}"),
        f"{group_name}_final_rates_missing": _missing_rate_summary(final_rates),
        f"{group_name}_aggregated_activity": frame_summary(aggregated_activity, name=f"inventory_final_activity_{group_name}"),
        f"{group_name}_activity_by_emfac_id": frame_summary(activity_by_emfac_id, name=f"inventory_activity_by_emfacid_{group_name}"),
        f"{group_name}_fleet": frame_summary(fleet, name=f"inventory_final_fleet_{group_name}"),
    }
    return (final_rates, matching_activity, aggregated_activity, activity_by_emfac_id, fleet), trace_payload


def run_step4(workflow: dict[str, object]) -> dict[str, object]:
    print("  Step 4. Finalize Output")
    print("    4.1 Load filled project analysis surfaces and activity weights")
    passenger_surface = _load_surface(workflow["paths"]["project_analysis_passenger"])
    freight_surface = _load_surface(workflow["paths"]["project_analysis_freight"])
    activity_weights = _build_activity_weights(workflow["paths"]["emissions_inventory"])
    model_year_groups = workflow["run"]["model_year_groups"]
    trace_payload = {
        "passenger_filled_surface": frame_summary(passenger_surface, name="filled_project_analysis_passenger"),
        "freight_filled_surface": frame_summary(freight_surface, name="filled_project_analysis_freight"),
        "activity_weights": frame_summary(activity_weights, name="activity_weights"),
    }
    print("    4.2 Finalize passenger outputs")
    passenger_outputs, passenger_trace_payload = _finalize_group_outputs(
        group_name="passenger",
        surface=passenger_surface,
        activity_weights=activity_weights,
        model_year_groups=model_year_groups,
    )
    trace_payload.update(passenger_trace_payload)
    print("    4.3 Finalize freight outputs")
    freight_outputs, freight_trace_payload = _finalize_group_outputs(
        group_name="freight",
        surface=freight_surface,
        activity_weights=activity_weights,
        model_year_groups=model_year_groups,
    )
    trace_payload.update(freight_trace_payload)
    outputs = {
        "passenger": passenger_outputs,
        "freight": freight_outputs,
    }
    print("    4.4 Build emissions rates store dataset")
    rates_store_frame = _prepare_rates_store_frame(
        passenger_rates=outputs["passenger"][0],
        freight_rates=outputs["freight"][0],
    )
    trace_payload["rates_store_source"] = frame_summary(rates_store_frame, name="rates_store_source")
    print("    4.5 Write final outputs and derived store")
    for group_name, (final_rates, matching_activity, aggregated_activity, activity_by_emfac_id, fleet) in outputs.items():
        _write_parquet(final_rates, workflow["paths"][f"final_output_{group_name}"])
        _write_parquet(matching_activity, workflow["paths"][f"matching_activity_output_{group_name}"])
        _write_parquet(aggregated_activity, workflow["paths"][f"final_activity_output_{group_name}"])
        _write_parquet(activity_by_emfac_id, workflow["paths"][f"final_activity_emfacid_output_{group_name}"])
        _write_parquet(fleet, workflow["paths"][f"final_fleet_output_{group_name}"])
        _print_model_year_group_stats(final_rates, aggregated_activity, fleet)
    rates_store = _write_rates_store_from_dataframe(
        rates=rates_store_frame,
        output_dir=workflow["paths"]["emissions_store_root"],
        compression="zstd",
    )
    write_trace(
        workflow,
        "step4_finalize_output",
        {
            **trace_payload,
            "rates_store": {
                "store_root": rates_store["output_dir"],
                "parquet_root": rates_store["parquet_root"],
                "duckdb_path": rates_store["duckdb_path"],
                "parquet_file_count": rates_store["parquet_file_count"],
                "pre_write_county_coverage": rates_store["pre_write_county_coverage"],
                "post_write_county_coverage": rates_store["post_write_county_coverage"],
            },
        },
    )
    return workflow
