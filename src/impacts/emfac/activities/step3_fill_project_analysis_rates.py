from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.emfac.common import frame_summary
from impacts.emfac.common import write_trace

ACTIVITY_COLUMN = "speedMph_timeMin"
SURFACE_KEYS = ["county", "vehicleCategory", "fuel", "modelYear", "process", ACTIVITY_COLUMN, "roadCategory"]
BASE_KEYS = ["county", "vehicleCategory", "fuel", "modelYear", "process", ACTIVITY_COLUMN]
STATEWIDE_BASE_KEYS = ["vehicleCategory", "fuel", "modelYear", "process", ACTIVITY_COLUMN]
POLLUTANT_COLUMNS = [
    "bc_gram",
    "ch4_gram",
    "co_gram",
    "co2_gram",
    "hc_gram",
    "nh3_gram",
    "n2o_gram",
    "nox_gram",
    "pm_gram",
    "pm10_gram",
    "pm25_gram",
    "rog_gram",
    "sox_gram",
    "tog_gram",
]
GROUP_RATE_KEYS = ["county", "vehicleCategory", "fuel", "modelYear", "process"]
SPEED_MPH_PROCESSES = {"RUNEX", "PMBW", "PTOEX"}
TIME_PROCESSES = {"STREX"}
OTHER_PROCESSES = {"DIURN", "HOTSOAK", "IDLEX", "PMTW", "RUNLOSS"}
INVENTORY_POLLUTANT_MAP = {
    "ch4_gram": "ch4_{process}_short_tons_per_year",
    "co_gram": "co_{process}_short_tons_per_year",
    "co2_gram": "co2_{process}_short_tons_per_year",
    "hc_gram": "hc_{process}_short_tons_per_year",
    "nh3_gram": "nh3_{process}_short_tons_per_year",
    "n2o_gram": "n2o_{process}_short_tons_per_year",
    "nox_gram": "nox_{process}_short_tons_per_year",
    "pm_gram": "pm_{process}_short_tons_per_year",
    "pm10_gram": "pm10_{process}_short_tons_per_year",
    "pm25_gram": "pm25_{process}_short_tons_per_year",
    "rog_gram": "rog_{process}_short_tons_per_year",
    "sox_gram": "sox_{process}_short_tons_per_year",
    "tog_gram": "tog_{process}_short_tons_per_year",
}

_ACTIVITIES_RATE_MAPPINGS: dict[str, object] = {}


def _set_activities_rate_mappings(mappings: dict[str, object]) -> None:
    global _ACTIVITIES_RATE_MAPPINGS
    _ACTIVITIES_RATE_MAPPINGS = {}


def _format_numeric_series(values: pd.Series) -> pd.Series:
    return values.map(lambda value: f"{float(value):g}" if pd.notna(value) else pd.NA)


def _available_pollutant_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in POLLUTANT_COLUMNS if column in frame.columns]


def _all_zero_pollutant_row_mask(frame: pd.DataFrame) -> pd.Series:
    pollutant_columns = _available_pollutant_columns(frame)
    if not pollutant_columns:
        return pd.Series(False, index=frame.index)

    numeric = frame[pollutant_columns].apply(pd.to_numeric, errors="coerce")
    has_any_value = numeric.notna().any(axis=1)
    has_any_nonzero = numeric.fillna(0.0).ne(0.0).any(axis=1)
    return has_any_value & ~has_any_nonzero


def _group_rate_keys(frame: pd.DataFrame) -> list[str]:
    return [column for column in GROUP_RATE_KEYS if column in frame.columns]


def _rows_with_any_real_rate_mask(frame: pd.DataFrame) -> pd.Series:
    pollutant_columns = _available_pollutant_columns(frame)
    if not pollutant_columns:
        return pd.Series(False, index=frame.index)

    numeric = frame[pollutant_columns].apply(pd.to_numeric, errors="coerce")
    return numeric.fillna(0.0).ne(0.0).any(axis=1)


def _missing_rate_groups(frame: pd.DataFrame) -> pd.DataFrame:
    group_keys = _group_rate_keys(frame)
    if not group_keys:
        return pd.DataFrame(columns=GROUP_RATE_KEYS + ["rows"])

    group_frame = frame[group_keys].copy()
    group_frame["_has_real_rate"] = _rows_with_any_real_rate_mask(frame).to_numpy()
    summary = (
        group_frame.groupby(group_keys, dropna=False)["_has_real_rate"]
        .agg(rows="size", has_real_rate="any")
        .reset_index()
    )
    return summary.loc[~summary["has_real_rate"]].drop(columns="has_real_rate")


def _zero_row_examples(frame: pd.DataFrame, *, limit: int = 10) -> list[dict[str, object]]:
    mask = _all_zero_pollutant_row_mask(frame)
    if not mask.any():
        return []

    columns = [
        column
        for column in ["county", "vehicleCategory", "fuel", "modelYear", "process", ACTIVITY_COLUMN, "roadCategory"]
        if column in frame.columns
    ]
    return frame.loc[mask, columns].head(limit).to_dict(orient="records")


def _replace_all_zero_pollutant_rows_with_missing(
    frame: pd.DataFrame,
    *,
    label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    pollutant_columns = _available_pollutant_columns(frame)
    if not pollutant_columns:
        return frame, {"label": label, "rows": 0, "examples": []}

    result = frame.copy()
    zero_mask = _all_zero_pollutant_row_mask(result)
    if not zero_mask.any():
        return result, {"label": label, "rows": 0, "examples": []}

    result.loc[zero_mask, pollutant_columns] = pd.NA
    summary = {
        "label": label,
        "rows": int(zero_mask.sum()),
        "examples": _zero_row_examples(frame),
    }
    return result, summary


def _assert_no_missing_rate_groups(frame: pd.DataFrame, *, label: str) -> None:
    missing_groups = _missing_rate_groups(frame)
    if missing_groups.empty:
        return

    examples = missing_groups.head(10).to_dict(orient="records")
    raise ValueError(
        f"{label} still contains {int(len(missing_groups)):,} county/vehicle/fuel/modelYear/process group(s) "
        "with no real pollutant rates. This means the fill pipeline could not recover any positive "
        "pollutant coverage anywhere within those groups. Check the source project-analysis rates and "
        f"inventory rows for these keys. Example groups: {examples}"
    )


def _summarize_missing_rate_groups(frame: pd.DataFrame) -> dict[str, object]:
    missing_groups = _missing_rate_groups(frame)
    return {
        "group_count": int(len(missing_groups)),
        "rows": int(missing_groups["rows"].sum()) if not missing_groups.empty else 0,
        "by_process": {
            str(process): int(count)
            for process, count in missing_groups["process"].value_counts(dropna=False).items()
        }
        if "process" in missing_groups.columns
        else {},
        "examples": missing_groups.head(10).to_dict(orient="records"),
    }


def _print_missing_rate_group_summary(frame: pd.DataFrame, *, label: str) -> dict[str, object]:
    summary = _summarize_missing_rate_groups(frame)
    if summary["group_count"] <= 0:
        return summary

    print(
        f"      Error: found {summary['group_count']:,} county/vehicle/fuel/modelYear/process group(s) "
        f"with no real pollutant rates after {label}."
    )
    print(
        "        Grouped across all speed bins and pollutant columns; these groups have no positive "
        "pollutant values anywhere in the current filled surface."
    )
    if summary["by_process"]:
        print("        Affected groups by process:")
        for process, count in summary["by_process"].items():
            print(f"          {process}: {count:,}")
    for example in summary["examples"][:5]:
        print(f"        Example group: {example}")
    return summary


def _normalize_activity_column(frame: pd.DataFrame, *, column: str = ACTIVITY_COLUMN) -> pd.DataFrame:
    result = frame.copy()
    if column in result.columns:
        result[column] = _format_numeric_series(pd.to_numeric(result[column], errors="coerce"))
    return result


def _load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(Path(path).expanduser().resolve())


def _missing_rate_summary(frame: pd.DataFrame) -> dict[str, object]:
    available_columns = _available_pollutant_columns(frame)
    if not available_columns:
        return {
            "row_count": int(len(frame)),
            "pollutant_column_count": 0,
            "expected_rate_cells": 0,
            "filled_rate_cells": 0,
            "missing_rate_cells": 0,
            "rows_with_any_rate": 0,
            "rows_with_no_rates": int(len(frame)),
            "rows_with_all_zero_rates": 0,
            "missing_by_pollutant": {},
            "missing_rows_by_process": {},
        }

    numeric = frame[available_columns].apply(pd.to_numeric, errors="coerce")
    non_null = numeric.notna()
    zero_rows = _all_zero_pollutant_row_mask(frame)
    rows_with_any_rate = non_null.any(axis=1) & ~zero_rows
    return {
        "row_count": int(len(frame)),
        "pollutant_column_count": len(available_columns),
        "expected_rate_cells": int(len(frame) * len(available_columns)),
        "filled_rate_cells": int(non_null.sum().sum()),
        "missing_rate_cells": int((~non_null).sum().sum()),
        "rows_with_any_rate": int(rows_with_any_rate.sum()),
        "rows_with_no_rates": int((~rows_with_any_rate).sum()),
        "rows_with_all_zero_rates": int(zero_rows.sum()),
        "missing_by_pollutant": {
            column: int(numeric[column].isna().sum())
            for column in available_columns
        },
        "missing_rows_by_process": {
            str(process): int((~group[available_columns].apply(pd.to_numeric, errors="coerce").notna().any(axis=1) | _all_zero_pollutant_row_mask(group)).sum())
            for process, group in frame.groupby("process", dropna=False)
        }
        if "process" in frame.columns
        else {},
    }


def _explain_dropped_source_coverage_rows(dropped_rows: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if dropped_rows.empty:
        return [], []

    result = dropped_rows.copy()
    process = result["process"].astype(str)
    fuel = result["fuel"].astype(str)

    explanations: list[tuple[str, str, pd.Series]] = [
        (
            "evaporative_process_on_non_gasoline",
            "Evaporative processes are not present in Bay Area source for these non-gasoline cohorts.",
            process.isin({"DIURN", "HOTSOAK", "RUNLOSS"}) & ~fuel.isin({"Gas", "Phe"}),
        ),
        (
            "strex_on_non_gasoline",
            "Soak exhaust is not present in Bay Area source for these non-gasoline cohorts.",
            (process == "STREX") & ~fuel.isin({"Gas", "Phe"}),
        ),
        (
            "runex_on_electric",
            "Running exhaust is not present in Bay Area source for these electric cohorts.",
            (process == "RUNEX") & (fuel == "Elec"),
        ),
        (
            "idlex_without_source_coverage",
            "Idle exhaust is not present in Bay Area source for these class/fuel cohorts.",
            process == "IDLEX",
        ),
        (
            "ptoex_outside_supported_pto_cohorts",
            "PTO exhaust is only sourced for the configured diesel PTO target vehicle categories.",
            process == "PTOEX",
        ),
    ]

    explained_rows = pd.Series(False, index=result.index)
    explanation_summaries: list[dict[str, object]] = []
    for code, message, mask in explanations:
        matched = mask & ~explained_rows
        if not matched.any():
            continue
        rows = result.loc[matched]
        explained_rows.loc[matched] = True
        explanation_summaries.append(
            {
                "code": code,
                "message": message,
                "rows": int(len(rows)),
                "process_breakdown": {
                    str(name): int(count)
                    for name, count in rows["process"].value_counts(dropna=False).items()
                },
            }
        )

    unexplained = (
        result.loc[~explained_rows, ["vehicleCategory", "fuel", "process"]]
        .value_counts(dropna=False)
        .head(10)
        .reset_index(name="rows")
        .to_dict(orient="records")
    )
    return explanation_summaries, unexplained


def _merge_rate_columns(surface: pd.DataFrame, rates: pd.DataFrame, *, keys: list[str]) -> pd.DataFrame:
    value_columns = [column for column in POLLUTANT_COLUMNS if column in rates.columns]
    if not value_columns:
        return surface
    merged = surface.merge(rates[keys + value_columns], on=keys, how="left", suffixes=("", "_fill"))
    for column in value_columns:
        fill_column = f"{column}_fill"
        if fill_column in merged.columns:
            merged[column] = merged[fill_column].combine_first(merged.get(column))
            merged = merged.drop(columns=fill_column)
        elif column not in surface.columns:
            merged[column] = merged[column]
    return merged


def _build_inventory_fill_frame(inventory: pd.DataFrame, *, include_county: bool) -> pd.DataFrame:
    frame = inventory.copy()
    if include_county:
        frame["county"] = frame["county"].astype(str)
    frame["vehicleCategory"] = frame["vehicleCategory"].astype(str)
    frame["fuel"] = frame["fuel"].astype(str)
    frame["modelYear"] = pd.to_numeric(frame["modelYear"], errors="raise").astype(int)
    frame["speed"] = pd.to_numeric(frame["speed"], errors="coerce")

    outputs: list[pd.DataFrame] = []
    id_columns = ["vehicleCategory", "fuel", "modelYear"]
    if include_county:
        id_columns = ["county"] + id_columns

    for process in sorted(SPEED_MPH_PROCESSES | TIME_PROCESSES | OTHER_PROCESSES):
        process_key = process.lower()
        process_frame = frame[id_columns].drop_duplicates().copy()
        process_frame["process"] = process
        if process in SPEED_MPH_PROCESSES:
            process_frame[ACTIVITY_COLUMN] = _format_numeric_series(frame["speed"])
        elif process in TIME_PROCESSES:
            process_frame[ACTIVITY_COLUMN] = _format_numeric_series(frame["speed"])
        else:
            process_frame[ACTIVITY_COLUMN] = pd.NA
        process_frame["roadCategory"] = pd.NA

        has_values = False
        for output_column, inventory_pattern in INVENTORY_POLLUTANT_MAP.items():
            inventory_column = inventory_pattern.format(process=process_key)
            if inventory_column in frame.columns:
                process_frame[output_column] = pd.to_numeric(frame[inventory_column], errors="coerce")
                has_values = True
        if has_values:
            outputs.append(process_frame)

    if not outputs:
        return pd.DataFrame(columns=SURFACE_KEYS + POLLUTANT_COLUMNS)

    combined = pd.concat(outputs, ignore_index=True, sort=False)
    combined = _normalize_activity_column(combined)
    return combined.drop_duplicates().reset_index(drop=True)


def fill_project_analysis_rates(
    comprehensive_surface: pd.DataFrame,
    *,
    project_analysis_source: pd.DataFrame,
    project_analysis_nh3_rates: pd.DataFrame,
    project_analysis_bc: pd.DataFrame,
    project_analysis_prdust: pd.DataFrame,
    emissions_inventory: pd.DataFrame,
    statewide_inventory: pd.DataFrame,
) -> pd.DataFrame:
    surface = _normalize_activity_column(comprehensive_surface)

    for rates, keys in [
        (_normalize_activity_column(project_analysis_source), BASE_KEYS),
        (_normalize_activity_column(project_analysis_nh3_rates), BASE_KEYS),
        (_normalize_activity_column(project_analysis_bc), BASE_KEYS),
        (_normalize_activity_column(project_analysis_prdust), SURFACE_KEYS),
    ]:
        surface = _merge_rate_columns(surface, rates, keys=keys)
        surface, _ = _replace_all_zero_pollutant_rows_with_missing(surface, label="rate merge")

    study_inventory_rates = _build_inventory_fill_frame(emissions_inventory, include_county=True)
    surface = _merge_rate_columns(surface, study_inventory_rates, keys=BASE_KEYS)
    surface, _ = _replace_all_zero_pollutant_rows_with_missing(surface, label="study-area inventory fill")

    statewide_inventory_rates = _build_inventory_fill_frame(statewide_inventory, include_county=False)
    statewide_surface = surface.merge(
        statewide_inventory_rates[STATEWIDE_BASE_KEYS + [column for column in POLLUTANT_COLUMNS if column in statewide_inventory_rates.columns]],
        on=STATEWIDE_BASE_KEYS,
        how="left",
        suffixes=("", "_statewide"),
    )
    for column in POLLUTANT_COLUMNS:
        fill_column = f"{column}_statewide"
        if fill_column in statewide_surface.columns:
            statewide_surface[column] = statewide_surface[fill_column].combine_first(statewide_surface.get(column))
            statewide_surface = statewide_surface.drop(columns=fill_column)
    statewide_surface, _ = _replace_all_zero_pollutant_rows_with_missing(
        statewide_surface,
        label="statewide inventory fill",
    )
    return statewide_surface


def _load_step3_inputs(
    workflow: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _load_parquet(workflow["paths"]["project_analysis_source"]),
        _load_parquet(workflow["paths"]["project_analysis_nh3_rates"]),
        _load_parquet(workflow["paths"]["project_analysis_bc"]),
        _load_parquet(workflow["paths"]["project_analysis_prdust"]),
        _load_parquet(workflow["paths"]["emissions_inventory"]),
        _load_parquet(workflow["paths"]["statewide_inventory"]),
    )


def _fill_group_rates(
    *,
    group_name: str,
    comprehensive_surface: pd.DataFrame,
    project_analysis_source: pd.DataFrame,
    project_analysis_nh3_rates: pd.DataFrame,
    project_analysis_bc: pd.DataFrame,
    project_analysis_prdust: pd.DataFrame,
    emissions_inventory: pd.DataFrame,
    statewide_inventory: pd.DataFrame,
) -> pd.DataFrame:
    return fill_project_analysis_rates(
        comprehensive_surface,
        project_analysis_source=project_analysis_source,
        project_analysis_nh3_rates=project_analysis_nh3_rates,
        project_analysis_bc=project_analysis_bc,
        project_analysis_prdust=project_analysis_prdust,
        emissions_inventory=emissions_inventory,
        statewide_inventory=statewide_inventory,
    )


def _impute_group_rates(
    *,
    workflow: dict[str, object],
    group_name: str,
    comprehensive_surface: pd.DataFrame,
    initially_filled: pd.DataFrame,
    project_analysis_source: pd.DataFrame,
    project_analysis_nh3_rates: pd.DataFrame,
    project_analysis_bc: pd.DataFrame,
    project_analysis_prdust: pd.DataFrame,
    emissions_inventory: pd.DataFrame,
    statewide_inventory: pd.DataFrame,
) -> dict[str, object]:
    filled = initially_filled.copy()
    grouped_missing_summaries: list[dict[str, object]] = []
    grouped_missing_summaries.append(
        {"label": "initial fill", **_print_missing_rate_group_summary(filled, label=f"{group_name} initial fill")}
    )
    zero_row_cleanups: list[dict[str, object]] = []
    print(f"      Apply speed fallback for unresolved {group_name} rows and refill")
    filled = _fill_with_speed_fallback(
        filled,
        reference_rates=project_analysis_source,
        project_analysis_source=project_analysis_source,
        project_analysis_nh3_rates=project_analysis_nh3_rates,
        project_analysis_bc=project_analysis_bc,
        project_analysis_prdust=project_analysis_prdust,
        emissions_inventory=emissions_inventory,
        statewide_inventory=statewide_inventory,
    )
    filled, zero_summary = _replace_all_zero_pollutant_rows_with_missing(filled, label=f"{group_name} speed fallback refill")
    zero_row_cleanups.append(zero_summary)
    grouped_missing_summaries.append(
        {"label": "speed fallback refill", **_print_missing_rate_group_summary(filled, label=f"{group_name} speed fallback refill")}
    )
    filled, source_filter_summary = _filter_unresolved_rows_without_source_coverage(
        filled,
        project_analysis_source=project_analysis_source,
    )
    _assert_no_missing_rate_groups(filled, label=f"Filled {group_name} project-analysis rates")
    _write_parquet(filled, workflow["paths"][f"project_analysis_{group_name}"])
    return {
        f"{group_name}_comprehensive_surface": frame_summary(comprehensive_surface, name=f"comprehensive_project_analysis_{group_name}"),
        f"{group_name}_filled_surface": frame_summary(filled, name=f"filled_project_analysis_{group_name}"),
        f"{group_name}_filled_surface_missing_rates": _missing_rate_summary(filled),
        f"{group_name}_zero_row_cleanups": zero_row_cleanups,
        f"{group_name}_grouped_missing_rate_summaries": grouped_missing_summaries,
        f"{group_name}_source_coverage_filter": source_filter_summary,
    }


def _filter_unresolved_rows_without_source_coverage(
    surface: pd.DataFrame,
    *,
    project_analysis_source: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    unresolved_mask = ~surface[POLLUTANT_COLUMNS].notna().any(axis=1)
    if not unresolved_mask.any():
        return surface, {
            "dropped_missing_class_fuel_rows": 0,
            "dropped_missing_class_fuel_year_rows": 0,
            "dropped_missing_class_fuel_year_process_rows": 0,
            "remaining_missing_exact_key_rows": 0,
            "dropped_by_process": {},
            "dropped_top_vehicle_fuel": [],
            "remaining_exact_key_missing_by_process": {},
            "remaining_exact_key_top_vehicle_fuel": [],
        }

    source = _normalize_activity_column(project_analysis_source)
    unresolved = surface.loc[unresolved_mask].copy()
    unresolved["_surface_index"] = unresolved.index

    class_fuel_keys = ["vehicleCategory", "fuel"]
    class_fuel_year_keys = ["vehicleCategory", "fuel", "modelYear"]
    class_fuel_year_process_keys = ["vehicleCategory", "fuel", "modelYear", "process"]
    exact_keys = class_fuel_year_process_keys + [ACTIVITY_COLUMN]

    source_class_fuel = source[class_fuel_keys].drop_duplicates().assign(_has_class_fuel=True)
    source_class_fuel_year = source[class_fuel_year_keys].drop_duplicates().assign(_has_class_fuel_year=True)
    source_class_fuel_year_process = (
        source[class_fuel_year_process_keys].drop_duplicates().assign(_has_class_fuel_year_process=True)
    )
    source_exact = source[exact_keys].drop_duplicates().assign(_has_exact_key=True)

    classified = unresolved.merge(source_class_fuel, on=class_fuel_keys, how="left")
    classified = classified.merge(source_class_fuel_year, on=class_fuel_year_keys, how="left")
    classified = classified.merge(source_class_fuel_year_process, on=class_fuel_year_process_keys, how="left")
    classified = classified.merge(source_exact, on=exact_keys, how="left")

    missing_class_fuel = classified["_has_class_fuel"].isna()
    missing_class_fuel_year = classified["_has_class_fuel"].notna() & classified["_has_class_fuel_year"].isna()
    missing_class_fuel_year_process = (
        classified["_has_class_fuel_year"].notna() & classified["_has_class_fuel_year_process"].isna()
    )
    drop_mask = missing_class_fuel | missing_class_fuel_year | missing_class_fuel_year_process

    dropped_missing_class_fuel_rows = int(missing_class_fuel.sum())
    dropped_missing_class_fuel_year_rows = int(missing_class_fuel_year.sum())
    dropped_missing_class_fuel_year_process_rows = int(missing_class_fuel_year_process.sum())
    remaining_missing_exact_key_rows = int((~drop_mask & classified["_has_exact_key"].isna()).sum())
    dropped_rows = classified.loc[drop_mask].copy()
    remaining_exact_key_rows = classified.loc[~drop_mask & classified["_has_exact_key"].isna()].copy()

    dropped_by_process = {
        str(process): int(count)
        for process, count in dropped_rows["process"].value_counts(dropna=False).items()
    }
    dropped_top_vehicle_fuel = [
        {
            "vehicleCategory": str(vehicle_category),
            "fuel": str(fuel),
            "rows": int(count),
        }
        for (vehicle_category, fuel), count in dropped_rows[["vehicleCategory", "fuel"]]
        .value_counts(dropna=False)
        .head(10)
        .items()
    ]
    remaining_exact_key_missing_by_process = {
        str(process): int(count)
        for process, count in remaining_exact_key_rows["process"].value_counts(dropna=False).items()
    }
    remaining_exact_key_top_vehicle_fuel = [
        {
            "vehicleCategory": str(vehicle_category),
            "fuel": str(fuel),
            "rows": int(count),
        }
        for (vehicle_category, fuel), count in remaining_exact_key_rows[["vehicleCategory", "fuel"]]
        .value_counts(dropna=False)
        .head(10)
        .items()
    ]
    drop_explanations, unexplained_drop_combinations = _explain_dropped_source_coverage_rows(dropped_rows)

    if drop_mask.any():
        print("      Warning: filtering unresolved rows with no Bay Area project-analysis source coverage.")
        print(
            "        These class/fuel, class/fuel/year, and class/fuel/year/process combinations are absent from Bay Area project-analysis and should be reviewed against EMFAC documentation."
        )
        print(f"        Dropped missing class+fuel rows: {dropped_missing_class_fuel_rows:,}")
        print(f"        Dropped missing class+fuel+year rows: {dropped_missing_class_fuel_year_rows:,}")
        print(
            f"        Dropped missing class+fuel+year+process rows: {dropped_missing_class_fuel_year_process_rows:,}"
        )
        if dropped_by_process:
            print("        Dropped rows by process:")
            for process, count in dropped_by_process.items():
                print(f"          {process}: {count:,}")
        if dropped_top_vehicle_fuel:
            print("        Top dropped vehicle/fuel combinations:")
            for item in dropped_top_vehicle_fuel:
                print(f"          {item['vehicleCategory']} + {item['fuel']}: {item['rows']:,}")
        if drop_explanations:
            print("        Interpretable drop reasons:")
            for item in drop_explanations:
                print(f"          {item['code']}: {item['rows']:,}")
                print(f"            {item['message']}")
                if item["process_breakdown"]:
                    process_text = ", ".join(f"{process}={count:,}" for process, count in item["process_breakdown"].items())
                    print(f"            Processes: {process_text}")
        if unexplained_drop_combinations:
            print("        Dropped without a good explanation yet:")
            for item in unexplained_drop_combinations:
                print(
                    f"          {item['vehicleCategory']} + {item['fuel']} + {item['process']}: {item['rows']:,}"
                )
        surface = surface.drop(index=classified.loc[drop_mask, "_surface_index"]).reset_index(drop=True)

    print(
        f"        Remaining unresolved rows missing only at class+fuel+year+process+{ACTIVITY_COLUMN}: {remaining_missing_exact_key_rows:,}"
    )
    if remaining_exact_key_missing_by_process:
        print("        Remaining exact-key missing rows by process:")
        for process, count in remaining_exact_key_missing_by_process.items():
            print(f"          {process}: {count:,}")
    if remaining_exact_key_top_vehicle_fuel:
        print("        Top remaining exact-key missing vehicle/fuel combinations:")
        for item in remaining_exact_key_top_vehicle_fuel:
            print(f"          {item['vehicleCategory']} + {item['fuel']}: {item['rows']:,}")
    return surface, {
        "dropped_missing_class_fuel_rows": dropped_missing_class_fuel_rows,
        "dropped_missing_class_fuel_year_rows": dropped_missing_class_fuel_year_rows,
        "dropped_missing_class_fuel_year_process_rows": dropped_missing_class_fuel_year_process_rows,
        "remaining_missing_exact_key_rows": remaining_missing_exact_key_rows,
        "dropped_by_process": dropped_by_process,
        "dropped_top_vehicle_fuel": dropped_top_vehicle_fuel,
        "remaining_exact_key_missing_by_process": remaining_exact_key_missing_by_process,
        "remaining_exact_key_top_vehicle_fuel": remaining_exact_key_top_vehicle_fuel,
    }


def _apply_speed_fallback_keys(surface: pd.DataFrame, reference_rates: pd.DataFrame) -> pd.DataFrame:
    unresolved_mask = ~surface[POLLUTANT_COLUMNS].notna().any(axis=1)
    if not unresolved_mask.any():
        return surface

    result = surface.copy()
    candidates = result.loc[
        unresolved_mask & result[ACTIVITY_COLUMN].notna(),
        ["vehicleCategory", "fuel", "process", "modelYear", ACTIVITY_COLUMN],
    ].copy()
    if candidates.empty:
        return result

    candidates["modelYear"] = pd.to_numeric(candidates["modelYear"], errors="raise").astype(int)
    candidates[ACTIVITY_COLUMN] = pd.to_numeric(candidates[ACTIVITY_COLUMN], errors="coerce")
    available = reference_rates.copy()
    available = available.loc[available[ACTIVITY_COLUMN].notna()].copy()
    if available.empty:
        return result
    available["modelYear"] = pd.to_numeric(available["modelYear"], errors="raise").astype(int)
    available[ACTIVITY_COLUMN] = pd.to_numeric(available[ACTIVITY_COLUMN], errors="coerce")
    bounds = (
        available.groupby(["vehicleCategory", "fuel", "process", "modelYear"], dropna=False)[ACTIVITY_COLUMN]
        .agg(min_value="min", max_value="max")
        .reset_index()
    )
    candidates = candidates.reset_index().merge(
        bounds,
        on=["vehicleCategory", "fuel", "process", "modelYear"],
        how="left",
    )
    high = candidates["max_value"].notna() & (candidates[ACTIVITY_COLUMN] > candidates["max_value"])
    low = candidates["min_value"].notna() & (candidates[ACTIVITY_COLUMN] < candidates["min_value"])
    if high.any():
        result.loc[candidates.loc[high, "index"], ACTIVITY_COLUMN] = _format_numeric_series(candidates.loc[high, "max_value"]).to_numpy()
    if low.any():
        result.loc[candidates.loc[low, "index"], ACTIVITY_COLUMN] = _format_numeric_series(candidates.loc[low, "min_value"]).to_numpy()
    return result


def _fill_with_speed_fallback(
    surface: pd.DataFrame,
    *,
    project_analysis_source: pd.DataFrame,
    project_analysis_nh3_rates: pd.DataFrame,
    project_analysis_bc: pd.DataFrame,
    project_analysis_prdust: pd.DataFrame,
    emissions_inventory: pd.DataFrame,
    statewide_inventory: pd.DataFrame,
    reference_rates: pd.DataFrame,
) -> pd.DataFrame:
    speed_fallback_surface = surface.copy()
    original_activity = speed_fallback_surface[ACTIVITY_COLUMN].copy()
    speed_fallback_surface = _apply_speed_fallback_keys(speed_fallback_surface, reference_rates)
    filled = fill_project_analysis_rates(
        speed_fallback_surface,
        project_analysis_source=project_analysis_source,
        project_analysis_nh3_rates=project_analysis_nh3_rates,
        project_analysis_bc=project_analysis_bc,
        project_analysis_prdust=project_analysis_prdust,
        emissions_inventory=emissions_inventory,
        statewide_inventory=statewide_inventory,
    )
    speed_mask = filled["process"].astype(str).isin(SPEED_MPH_PROCESSES)
    resolved_mask = filled[POLLUTANT_COLUMNS].notna().any(axis=1)
    restore_mask = speed_mask & resolved_mask
    filled.loc[restore_mask, ACTIVITY_COLUMN] = original_activity.loc[restore_mask].to_numpy()
    return filled


def _write_parquet(frame: pd.DataFrame, path: str) -> str:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported output format for {target}. Expected .parquet")
    frame.to_parquet(target, index=False)
    return str(target)


def run_step3(workflow: dict[str, object]) -> dict[str, object]:
    print("  Step 3. Fill Project Analysis Rates")
    _set_activities_rate_mappings(workflow.get("run", {}).get("mappings", {}) or {})
    print("    3.1 Load rate sources and fallback inventories")
    (
        project_analysis_source,
        project_analysis_nh3_rates,
        project_analysis_bc,
        project_analysis_prdust,
        emissions_inventory,
        statewide_inventory,
    ) = _load_step3_inputs(workflow)
    trace_payload: dict[str, object] = {
        "project_analysis_source": frame_summary(project_analysis_source, name="project_analysis_source"),
        "project_analysis_nh3_rates": frame_summary(project_analysis_nh3_rates, name="project_analysis_nh3_rates"),
        "project_analysis_bc": frame_summary(project_analysis_bc, name="project_analysis_bc"),
        "project_analysis_prdust": frame_summary(project_analysis_prdust, name="project_analysis_prdust"),
        "emissions_inventory": frame_summary(emissions_inventory, name="emissions_inventory"),
        "statewide_inventory": frame_summary(statewide_inventory, name="statewide_inventory"),
    }
    passenger_comprehensive_surface = _load_parquet(workflow["paths"]["project_analysis_passenger"])
    print("    3.2 Initial fill passenger rates")
    passenger_initial_fill = _fill_group_rates(
        group_name="passenger",
        comprehensive_surface=passenger_comprehensive_surface,
        project_analysis_source=project_analysis_source,
        project_analysis_nh3_rates=project_analysis_nh3_rates,
        project_analysis_bc=project_analysis_bc,
        project_analysis_prdust=project_analysis_prdust,
        emissions_inventory=emissions_inventory,
        statewide_inventory=statewide_inventory,
    )
    print("    3.3 Impute passenger rates")
    trace_payload.update(
        _impute_group_rates(
            workflow=workflow,
            group_name="passenger",
            comprehensive_surface=passenger_comprehensive_surface,
            initially_filled=passenger_initial_fill,
            project_analysis_source=project_analysis_source,
            project_analysis_nh3_rates=project_analysis_nh3_rates,
            project_analysis_bc=project_analysis_bc,
            project_analysis_prdust=project_analysis_prdust,
            emissions_inventory=emissions_inventory,
            statewide_inventory=statewide_inventory,
        )
    )
    freight_comprehensive_surface = _load_parquet(workflow["paths"]["project_analysis_freight"])
    print("    3.4 Initial fill freight rates")
    freight_initial_fill = _fill_group_rates(
        group_name="freight",
        comprehensive_surface=freight_comprehensive_surface,
        project_analysis_source=project_analysis_source,
        project_analysis_nh3_rates=project_analysis_nh3_rates,
        project_analysis_bc=project_analysis_bc,
        project_analysis_prdust=project_analysis_prdust,
        emissions_inventory=emissions_inventory,
        statewide_inventory=statewide_inventory,
    )
    print("    3.5 Impute freight rates")
    trace_payload.update(
        _impute_group_rates(
            workflow=workflow,
            group_name="freight",
            comprehensive_surface=freight_comprehensive_surface,
            initially_filled=freight_initial_fill,
            project_analysis_source=project_analysis_source,
            project_analysis_nh3_rates=project_analysis_nh3_rates,
            project_analysis_bc=project_analysis_bc,
            project_analysis_prdust=project_analysis_prdust,
            emissions_inventory=emissions_inventory,
            statewide_inventory=statewide_inventory,
        )
    )
    write_trace(
        workflow,
        "step3_fill_project_analysis_rates",
        trace_payload,
    )
    return workflow
