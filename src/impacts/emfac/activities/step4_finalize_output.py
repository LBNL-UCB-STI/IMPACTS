from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from impacts.emfac.activities.step3_fill_project_analysis_rates import ACTIVITY_COLUMN
from impacts.emfac.activities.step3_fill_project_analysis_rates import POLLUTANT_COLUMNS
from impacts.emfac.common import frame_summary
from impacts.emfac.common import write_trace

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
LIGHT_DUTY_VEHICLE_CATEGORIES = {"LDA", "LDT1", "LDT2"}


def _model_year_group_label(group: dict[str, object]) -> str:
    min_year = group.get("min_year")
    max_year = group.get("max_year")
    if min_year is None:
        return f"pre{int(max_year) + 1:02d}"
    if max_year is None:
        return f"post{int(min_year) - 1:02d}"
    return f"{int(min_year)}to{int(max_year)}"


def _vehicle_group(vehicle_category: object) -> str:
    vehicle_category = str(vehicle_category).strip()
    if vehicle_category in LIGHT_DUTY_VEHICLE_CATEGORIES:
        return "light_duty"
    return "medium_heavy_duty"


def _assign_model_year_groups(frame: pd.DataFrame, model_year_groups: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    result = frame.copy()
    labels = pd.Series(pd.NA, index=result.index, dtype="object")
    model_year = pd.to_numeric(result["modelYear"], errors="raise").astype(int)
    vehicle_groups = result["vehicleCategory"].map(_vehicle_group)
    for vehicle_group, groups in model_year_groups.items():
        group_mask = vehicle_groups == vehicle_group
        if not group_mask.any():
            continue
        for group in groups:
            min_year = group.get("min_year")
            max_year = group.get("max_year")
            mask = group_mask.copy()
            if min_year is not None:
                mask &= model_year >= int(min_year)
            if max_year is not None:
                mask &= model_year <= int(max_year)
            labels.loc[mask] = _model_year_group_label(group)
    if labels.isna().any():
        missing_rows = result.loc[labels.isna(), ["vehicleCategory", "modelYear"]].drop_duplicates()
        raise ValueError(
            "Some vehicleCategory/modelYear rows are not covered by the configured model_year_groups: "
            f"{missing_rows.to_dict(orient='records')[:20]}"
        )
    result["modelYear"] = labels
    return result


def _load_surface(path: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path).expanduser().resolve())


def _build_activity_weights(emissions_inventory_path: str) -> pd.DataFrame:
    source = pd.read_parquet(Path(emissions_inventory_path).expanduser().resolve())
    available_columns = [column for column in ["county", "vehicleCategory", "fuel", "modelYear", "speed", *ACTIVITY_COLUMNS] if column in source.columns]
    weights = source[available_columns].copy()
    weights["county"] = weights["county"].astype(str)
    weights["vehicleCategory"] = weights["vehicleCategory"].astype(str)
    weights["fuel"] = weights["fuel"].astype(str)
    weights["modelYear"] = pd.to_numeric(weights["modelYear"], errors="raise").astype(int)
    if "speed" in weights.columns:
        weights["speed"] = pd.to_numeric(weights["speed"], errors="coerce")
        weights = weights.drop_duplicates(["county", "vehicleCategory", "fuel", "modelYear", "speed"])
    else:
        weights = weights.drop_duplicates(ACTIVITY_JOIN_COLUMNS)
    for column in [column for column in ACTIVITY_COLUMNS if column in weights.columns]:
        weights[column] = pd.to_numeric(weights[column], errors="coerce")
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
    return fleet.drop(columns=["total_vmt_vehicle_miles_per_year"])


def _prepare_surface_keys(surface: pd.DataFrame, model_year_groups: dict[str, list[dict[str, object]]]) -> pd.DataFrame:
    keys = surface[["county", "vehicleCategory", "fuel", "modelYear", "process"]].drop_duplicates().copy()
    keys["county"] = keys["county"].astype(str)
    keys["vehicleCategory"] = keys["vehicleCategory"].astype(str)
    keys["fuel"] = keys["fuel"].astype(str)
    keys["modelYear"] = pd.to_numeric(keys["modelYear"], errors="raise").astype(int)
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
    )
    weights.loc[prdust_mask] = pd.to_numeric(frame.loc[prdust_mask, "population_vehicles"], errors="coerce")
    other_mask = ~vmt_mask & ~prdust_mask
    weights.loc[other_mask] = pd.to_numeric(frame.loc[other_mask, "trips_per_year"], errors="coerce")
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
    frame["county"] = frame["county"].astype(str)
    frame["vehicleCategory"] = frame["vehicleCategory"].astype(str)
    frame["fuel"] = frame["fuel"].astype(str)
    frame["modelYear"] = pd.to_numeric(frame["modelYear"], errors="raise").astype(int)
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
    return result.loc[result[value_columns].notna().any(axis=1)].reset_index(drop=True)


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
    print("    4.4 Model year group stats")
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


def _run_step4_substep_finalize_group(
    *,
    workflow: dict[str, object],
    group_name: str,
    activity_weights: pd.DataFrame,
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], dict[str, object]]:
    surface = _load_surface(workflow["paths"][f"project_analysis_{group_name}"])
    group_activity_weights = _filter_activity_weights_to_surface(activity_weights, surface)
    matching_activity = _build_matching_activity_table(
        group_activity_weights,
        workflow["run"]["model_year_groups"],
    )
    final_rates = _build_final_rate_table(
        surface,
        group_activity_weights,
        model_year_groups=workflow["run"]["model_year_groups"],
    )
    aggregated_activity = _build_aggregated_activity_table(
        group_activity_weights,
        surface,
        workflow["run"]["model_year_groups"],
    )
    fleet = _build_inventory_final_fleet_table(
        group_activity_weights,
        workflow["run"]["model_year_groups"],
    )
    trace_payload = {
        f"{group_name}_filled_surface": frame_summary(surface, name=f"filled_project_analysis_{group_name}"),
        f"{group_name}_activity_weights": frame_summary(group_activity_weights, name=f"activity_weights_{group_name}"),
        f"{group_name}_matching_activity": frame_summary(matching_activity, name=f"inventory_matching_activity_{group_name}"),
        f"{group_name}_final_rates": frame_summary(final_rates, name=f"final_horizontal_rates_{group_name}"),
        f"{group_name}_final_rates_missing": _missing_rate_summary(final_rates),
        f"{group_name}_aggregated_activity": frame_summary(aggregated_activity, name=f"inventory_final_activity_{group_name}"),
        f"{group_name}_fleet": frame_summary(fleet, name=f"inventory_final_fleet_{group_name}"),
    }
    return (final_rates, matching_activity, aggregated_activity, fleet), trace_payload


def run_step4(workflow: dict[str, object]) -> dict[str, object]:
    print("  Step 4. Finalize Output")
    print("    4.1 Load filled project analysis surface and activity weights")
    activity_weights = _build_activity_weights(workflow["paths"]["emissions_inventory"])
    print("    4.2 Aggregate final horizontal rates, inventory-final activity table, and inventory-final fleet shares")
    trace_payload = {
        "activity_weights": frame_summary(activity_weights, name="activity_weights"),
    }
    outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for group_name in ("passenger", "freight"):
        outputs[group_name], group_trace_payload = _run_step4_substep_finalize_group(
            workflow=workflow,
            group_name=group_name,
            activity_weights=activity_weights,
        )
        trace_payload.update(group_trace_payload)
    print("    4.3 Write final outputs")
    for group_name, (final_rates, matching_activity, aggregated_activity, fleet) in outputs.items():
        _write_parquet(final_rates, workflow["paths"][f"final_output_{group_name}"])
        _write_parquet(matching_activity, workflow["paths"][f"matching_activity_output_{group_name}"])
        _write_parquet(aggregated_activity, workflow["paths"][f"final_activity_output_{group_name}"])
        _write_parquet(fleet, workflow["paths"][f"final_fleet_output_{group_name}"])
        _print_model_year_group_stats(final_rates, aggregated_activity, fleet)
    write_trace(
        workflow,
        "step4_finalize_output",
        trace_payload,
    )
    return workflow
