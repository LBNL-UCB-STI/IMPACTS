from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from impacts.emfac.common import assert_row_count
from impacts.emfac.common import frame_summary
from impacts.emfac.common import write_trace
from impacts.emfac.mappings import load_class_fuel_alternatives

GRAMS_PER_SHORT_TON = 907_184.74
RATE_COLUMN = "rateGram"
SPEED_COLUMN = "speedMps_timeMin"
DEFAULT_MODEL_YEAR_GROUPS = [
    {"max_year": 2003},
    {"min_year": 2004, "max_year": 2013},
    {"min_year": 2014, "max_year": 2016},
    {"min_year": 2017},
]
STUDY_AREA_SEARCH_KEYS = [
    "county",
    "vehicleCategory",
    "fuel",
    "modelYear",
    SPEED_COLUMN,
]
STATEWIDE_SEARCH_KEYS = [
    "vehicleCategory",
    "fuel",
    "modelYear",
    SPEED_COLUMN,
]
PROJECT_LEVEL_COLUMNS = [
    "county",
    "vehicleCategory",
    "fuel",
    "modelYear",
    "process",
    SPEED_COLUMN,
    "pollutant",
    RATE_COLUMN,
]
INVENTORY_COLUMNS = [
    "vehicleCategory",
    "modelYear",
    "speed",
    "fuel",
    "total_vmt",
    "nh3_runex",
]
STUDY_AREA_INVENTORY_COLUMNS = [
    "county",
    "vehicleCategory",
    "modelYear",
    "speed",
    "fuel",
    "total_vmt",
    "nh3_runex",
]


def _read_parquet(path: str, *, columns: list[str]) -> pd.DataFrame:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Input path does not exist: {target}")
    if target.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported input format for {target}. Expected .parquet")
    return pd.read_parquet(target, columns=columns)


def _write_parquet(frame: pd.DataFrame, path: str) -> str:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported output format for {target}. Expected .parquet")
    frame.to_parquet(target, index=False)
    return str(target)


def _normalize_project_level(project_level: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in PROJECT_LEVEL_COLUMNS if column not in project_level.columns]
    if missing:
        raise ValueError(f"Project-level file is missing required columns: {', '.join(missing)}")
    frame = project_level.copy()
    for column in ["modelYear", SPEED_COLUMN, RATE_COLUMN]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["modelYear"] = frame["modelYear"].astype(int)
    for column in ["county", "vehicleCategory", "fuel", "process", "pollutant"]:
        frame[column] = frame[column].astype(str)
    return frame


def _build_inventory_rates(
    emissions_inventory: pd.DataFrame,
    *,
    county_column: str | None,
) -> pd.DataFrame:
    frame = emissions_inventory.copy()
    frame["modelYear"] = pd.to_numeric(frame["modelYear"], errors="raise").astype(int)
    frame["speed"] = pd.to_numeric(frame["speed"], errors="raise")
    frame["total_vmt"] = pd.to_numeric(frame["total_vmt"], errors="coerce")
    frame["nh3_runex"] = pd.to_numeric(frame["nh3_runex"], errors="coerce")
    if county_column is not None:
        frame["county"] = frame[county_column].astype(str)
    frame["vehicleCategory"] = frame["vehicleCategory"].astype(str)
    frame["fuel"] = frame["fuel"].astype(str)
    frame[SPEED_COLUMN] = frame["speed"]
    frame["pollutant"] = "NH3"
    frame["process"] = "RUNEX"
    valid = frame.loc[
        frame["total_vmt"].notna() & (frame["total_vmt"] > 0) & frame["nh3_runex"].notna()
    ].copy()
    valid["emission_rate"] = (valid["nh3_runex"] / valid["total_vmt"]) * GRAMS_PER_SHORT_TON
    keys = STUDY_AREA_SEARCH_KEYS if county_column is not None else STATEWIDE_SEARCH_KEYS
    return valid[keys + ["pollutant", "process", "emission_rate"]].drop_duplicates().reset_index(drop=True)


def _build_exact_lookup(rates: pd.DataFrame, *, keys: list[str]) -> pd.DataFrame:
    return rates.set_index(keys)[["emission_rate", "pollutant", "process"]].sort_index()


def _refill_exact_rates(
    merged: pd.DataFrame,
    inventory_lookup: pd.DataFrame,
    statewide_lookup: pd.DataFrame | None,
) -> pd.DataFrame:
    if not merged[RATE_COLUMN].isna().any():
        return merged

    missing_index = merged.index[merged[RATE_COLUMN].isna()]
    refill_input = merged.loc[missing_index, STUDY_AREA_SEARCH_KEYS]
    refill_merged = inventory_lookup.reindex(pd.MultiIndex.from_frame(refill_input)).reset_index(drop=True)
    if statewide_lookup is not None and refill_merged["emission_rate"].isna().any():
        refill_missing_index = refill_merged.index[refill_merged["emission_rate"].isna()]
        statewide_input = refill_input.iloc[refill_missing_index][STATEWIDE_SEARCH_KEYS]
        statewide_merged = statewide_lookup.reindex(pd.MultiIndex.from_frame(statewide_input)).reset_index(drop=True)
        statewide_matched = statewide_merged["emission_rate"].notna()
        if statewide_matched.any():
            refill_merged.loc[
                refill_missing_index[statewide_matched.to_numpy()],
                ["emission_rate", "pollutant", "process"],
            ] = statewide_merged.loc[statewide_matched, ["emission_rate", "pollutant", "process"]].to_numpy()

    matched = refill_merged["emission_rate"].notna().to_numpy()
    if matched.any():
        merged.loc[missing_index[matched], [RATE_COLUMN, "pollutant", "process"]] = refill_merged.loc[
            matched, ["emission_rate", "pollutant", "process"]
        ].to_numpy()
    return merged


def _select_model_year_candidate(
    requested_model_year: int,
    available_model_years: pd.Series,
    model_year_groups: list[dict[str, Any]] | None = None,
) -> int | None:
    years = sorted({int(year) for year in available_model_years.dropna().tolist()})
    if not years:
        return None

    groups = model_year_groups or DEFAULT_MODEL_YEAR_GROUPS
    selected_group = None
    for group in groups:
        min_year = group.get("min_year")
        max_year = group.get("max_year")
        if (min_year is None or requested_model_year >= int(min_year)) and (
            max_year is None or requested_model_year <= int(max_year)
        ):
            selected_group = group
            break
    if selected_group is None:
        return None

    min_year = selected_group.get("min_year")
    max_year = selected_group.get("max_year")
    candidates = [
        year
        for year in years
        if (min_year is None or year >= int(min_year)) and (max_year is None or year <= int(max_year))
    ]
    if not candidates:
        return None
    if min_year is None and max_year is not None:
        candidates = [year for year in candidates if year <= requested_model_year]
        return max(candidates) if candidates else None
    if min_year is not None and max_year is None:
        candidates = [year for year in candidates if year >= requested_model_year]
        return min(candidates) if candidates else None
    return min(candidates, key=lambda year: (abs(year - requested_model_year), year))


def _apply_model_year_fallback(
    merged: pd.DataFrame,
    inventory_rates: pd.DataFrame,
    model_year_groups: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    missing_mask = merged[RATE_COLUMN].isna()
    if not missing_mask.any():
        return merged

    available_years = (
        inventory_rates[["vehicleCategory", "fuel", "modelYear"]]
        .drop_duplicates()
        .groupby(["vehicleCategory", "fuel"], dropna=False)["modelYear"]
        .apply(lambda years: tuple(sorted(int(year) for year in years.dropna())))
        .rename("available_model_years")
        .reset_index()
    )
    candidates = merged.loc[missing_mask, ["vehicleCategory", "fuel", "modelYear"]].reset_index()
    candidates = candidates.merge(available_years, on=["vehicleCategory", "fuel"], how="left")
    unique_candidates = candidates[["modelYear", "available_model_years"]].drop_duplicates().copy()
    unique_candidates["candidate_model_year"] = [
        _select_model_year_candidate(int(model_year), pd.Series(available_model_years), model_year_groups)
        if isinstance(available_model_years, tuple)
        else None
        for model_year, available_model_years in zip(
            unique_candidates["modelYear"], unique_candidates["available_model_years"]
        )
    ]
    candidates = candidates.merge(
        unique_candidates,
        on=["modelYear", "available_model_years"],
        how="left",
    )
    matched = candidates["candidate_model_year"].notna()
    if matched.any():
        merged.loc[candidates.loc[matched, "index"], "modelYear"] = (
            candidates.loc[matched, "candidate_model_year"].astype(int).to_numpy()
        )
    return merged


def _apply_speed_fallback(merged: pd.DataFrame, inventory_rates: pd.DataFrame) -> pd.DataFrame:
    missing_mask = merged[RATE_COLUMN].isna()
    if not missing_mask.any():
        return merged

    speed_bounds = (
        inventory_rates.groupby(["vehicleCategory", "fuel", "modelYear"], dropna=False)[SPEED_COLUMN]
        .agg(min_speed="min", max_speed="max")
        .reset_index()
    )
    candidates = merged.loc[missing_mask, ["vehicleCategory", "fuel", "modelYear", SPEED_COLUMN]].reset_index()
    candidates = candidates.merge(speed_bounds, on=["vehicleCategory", "fuel", "modelYear"], how="left")

    high = candidates["max_speed"].notna() & (candidates[SPEED_COLUMN] > candidates["max_speed"])
    low = candidates["min_speed"].notna() & (candidates[SPEED_COLUMN] < candidates["min_speed"])
    if high.any():
        merged.loc[candidates.loc[high, "index"], SPEED_COLUMN] = candidates.loc[high, "max_speed"].to_numpy()
    if low.any():
        merged.loc[candidates.loc[low, "index"], SPEED_COLUMN] = candidates.loc[low, "min_speed"].to_numpy()
    return merged


def _apply_class_fuel_alternatives(
    merged: pd.DataFrame,
    alternatives: dict[tuple[str, str], tuple[str, str]],
) -> pd.DataFrame:
    if not alternatives or not merged[RATE_COLUMN].isna().any():
        return merged

    candidates = merged.loc[merged[RATE_COLUMN].isna(), ["vehicleCategory", "fuel"]].reset_index()
    replacement_keys = list(zip(candidates["vehicleCategory"], candidates["fuel"]))
    replacements = pd.Series(replacement_keys).map(alternatives)
    matched = replacements.notna()
    if matched.any():
        replacement_frame = pd.DataFrame(replacements.loc[matched].tolist(), columns=["vehicleCategory", "fuel"])
        merged.loc[candidates.loc[matched, "index"], ["vehicleCategory", "fuel"]] = replacement_frame.to_numpy()
    return merged


def _build_nh3_rows(
    project_level: pd.DataFrame,
    inventory_rates: pd.DataFrame,
    statewide_fallback_rates: pd.DataFrame | None = None,
    model_year_groups: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    reference_rates = statewide_fallback_rates if statewide_fallback_rates is not None else inventory_rates
    alternatives = load_class_fuel_alternatives()
    inventory_lookup = _build_exact_lookup(inventory_rates, keys=STUDY_AREA_SEARCH_KEYS)
    statewide_lookup = (
        None
        if statewide_fallback_rates is None
        else _build_exact_lookup(statewide_fallback_rates, keys=STATEWIDE_SEARCH_KEYS)
    )
    project_base = (
        project_level.loc[project_level["process"] == "RUNEX"]
        .drop(columns=["process", "pollutant", RATE_COLUMN])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    merged = project_base.copy()
    merged["pollutant"] = pd.NA
    merged["process"] = pd.NA
    merged[RATE_COLUMN] = pd.NA
    stage_counts: dict[str, int] = {}
    merged = _refill_exact_rates(merged, inventory_lookup, statewide_lookup)
    stage_counts["after_exact_refill_1"] = int(merged[RATE_COLUMN].isna().sum())
    merged = _apply_class_fuel_alternatives(merged, alternatives)
    merged = _refill_exact_rates(merged, inventory_lookup, statewide_lookup)
    stage_counts["after_class_fuel_and_exact_refill"] = int(merged[RATE_COLUMN].isna().sum())
    merged = _apply_model_year_fallback(merged, reference_rates, model_year_groups)
    merged = _refill_exact_rates(merged, inventory_lookup, statewide_lookup)
    stage_counts["after_model_year_and_exact_refill"] = int(merged[RATE_COLUMN].isna().sum())
    merged = _apply_speed_fallback(merged, reference_rates)
    merged = _refill_exact_rates(merged, inventory_lookup, statewide_lookup)
    stage_counts["after_speed_and_exact_refill"] = int(merged[RATE_COLUMN].isna().sum())
    return merged, stage_counts


def _load_nh3_inputs(
    *,
    project_level_path: str,
    emissions_inventory_path: str,
    imputation_fallback_inventory_path: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    project_level = _normalize_project_level(_read_parquet(project_level_path, columns=PROJECT_LEVEL_COLUMNS))
    inventory_rates = _build_inventory_rates(
        _read_parquet(emissions_inventory_path, columns=STUDY_AREA_INVENTORY_COLUMNS),
        county_column="county",
    )
    statewide_fallback_rates = None
    if imputation_fallback_inventory_path is not None:
        statewide_fallback_rates = _build_inventory_rates(
            _read_parquet(imputation_fallback_inventory_path, columns=INVENTORY_COLUMNS),
            county_column=None,
        )
    return project_level, inventory_rates, statewide_fallback_rates


def _append_nh3_rows(
    project_level: pd.DataFrame,
    nh3_rows: pd.DataFrame,
    *,
    drop_existing_nh3: bool,
) -> pd.DataFrame:
    result = project_level.copy()
    if drop_existing_nh3:
        result = result.loc[result["pollutant"] != "NH3"].copy()
    return pd.concat([result, nh3_rows[PROJECT_LEVEL_COLUMNS]], ignore_index=True)


def impute_project_level_nh3(
    *,
    project_level_path: str,
    emissions_inventory_path: str,
    imputation_fallback_inventory_path: str | None,
    output_path: str,
    drop_existing_nh3: bool = True,
    model_year_groups: list[dict[str, Any]] | None = None,
) -> str:
    project_level, inventory_rates, statewide_fallback_rates = _load_nh3_inputs(
        project_level_path=project_level_path,
        emissions_inventory_path=emissions_inventory_path,
        imputation_fallback_inventory_path=imputation_fallback_inventory_path,
    )
    nh3_rows, _ = _build_nh3_rows(
        project_level,
        inventory_rates,
        statewide_fallback_rates=statewide_fallback_rates,
        model_year_groups=model_year_groups,
    )
    result = _append_nh3_rows(project_level, nh3_rows, drop_existing_nh3=drop_existing_nh3)
    return _write_parquet(result, output_path)


def run_step2(workflow: dict[str, object]) -> dict[str, object]:
    print("  Step 2. Append NH3")
    print("    2.1 Load project-analysis and NH3 inventory inputs")
    project_level, inventory_rates, statewide_fallback_rates = _load_nh3_inputs(
        project_level_path=workflow["paths"]["project_analysis_clean"],
        emissions_inventory_path=workflow["paths"]["emissions_inventory_and_activities"],
        imputation_fallback_inventory_path=workflow["paths"]["statewide_emissions_inventory"],
    )
    print("    2.2 Build and append NH3 rows")
    nh3_rows, stage_counts = _build_nh3_rows(
        project_level,
        inventory_rates,
        statewide_fallback_rates=statewide_fallback_rates,
        model_year_groups=workflow["run"]["model_year_groups"],
    )
    result = _append_nh3_rows(project_level, nh3_rows, drop_existing_nh3=True)
    non_nh3_input_rows = int((project_level["pollutant"] != "NH3").sum())
    non_nh3_output_rows = int((result["pollutant"] != "NH3").sum())
    assert_row_count(non_nh3_input_rows, non_nh3_output_rows, label="NH3 append non-NH3 preservation")
    assert_row_count(len(nh3_rows), int((result["pollutant"] == "NH3").sum()), label="NH3 row append")
    print("    2.3 Write project-analysis with NH3")
    _write_parquet(result, workflow["paths"]["project_analysis_with_nh3"])
    write_trace(
        workflow,
        "step2_append_nh3",
        {
            "inputs": {
                "project_analysis": frame_summary(project_level, name="project_analysis_clean"),
                "study_area_inventory_rates": frame_summary(inventory_rates, name="study_area_nh3_rates"),
                "statewide_inventory_rates": None
                if statewide_fallback_rates is None
                else frame_summary(statewide_fallback_rates, name="statewide_nh3_rates"),
            },
            "nh3_rows": frame_summary(nh3_rows, name="nh3_rows"),
            "result": frame_summary(result, name="project_analysis_with_nh3"),
            "stage_missing_counts": stage_counts,
        },
    )
    return workflow
