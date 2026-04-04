from __future__ import annotations
"""Impute NH3 RUNEX rows into cleaned project-analysis parquet data.

Workflow:
1. exact study-area match, then exact statewide fallback
2. class/fuel substitution
3. model-year substitution within the substituted class/fuel
4. speed substitution within the substituted class/fuel/model-year
5. exact refill after each substitution stage
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

GRAMS_PER_SHORT_TON = 907_184.74
MATCH_KEYS = [
    "calendar_year",
    "season_month",
    "sub_area",
    "vehicle_class",
    "fuel",
    "model_year",
    "speed_time",
]
PROJECT_LEVEL_COLUMNS = [
    "calendar_year",
    "season_month",
    "sub_area",
    "vehicle_class",
    "fuel",
    "model_year",
    "temperature",
    "relative_humidity",
    "process",
    "speed_time",
    "pollutant",
    "emission_rate",
]
INVENTORY_COLUMNS = [
    "region",
    "calendar_year",
    "vehicle_category",
    "model_year",
    "speed",
    "fuel",
    "total_vmt",
    "nh3_runex",
]
CLASS_FUEL_ALTERNATIVES = {
    ("T6 Instate Tractor Class 6", "NG"): ("T6 Instate Other Class 6", "NG"),
    ("T6 Utility Class 5", "NG"): ("T6 Public Class 5", "NG"),
    ("T6 Utility Class 6", "NG"): ("T6 Public Class 6", "NG"),
    ("T6 Utility Class 7", "NG"): ("T6 Public Class 7", "NG"),
    ("T7 POAK Class 8", "NG"): ("T7 POLA Class 8", "NG"),
}


def _read_table(path: str, *, columns: list[str] | None = None) -> pd.DataFrame:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Input path does not exist: {target}")
    if target.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported input format for {target}. Expected .parquet")
    return pd.read_parquet(target, columns=columns)


def _write_table(frame: pd.DataFrame, path: str) -> str:
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
    frame["calendar_year"] = pd.to_numeric(frame["calendar_year"], errors="raise").astype(int)
    frame["model_year"] = pd.to_numeric(frame["model_year"], errors="raise").astype(int)
    frame["speed_time"] = pd.to_numeric(frame["speed_time"], errors="raise")
    frame["temperature"] = pd.to_numeric(frame["temperature"], errors="raise")
    frame["relative_humidity"] = pd.to_numeric(frame["relative_humidity"], errors="raise")
    frame["emission_rate"] = pd.to_numeric(frame["emission_rate"], errors="raise")
    frame["fuel"] = frame["fuel"].astype(str)
    for column in ["season_month", "sub_area", "vehicle_class", "process", "pollutant"]:
        frame[column] = frame[column].astype(str)
    return frame


def _build_inventory_rates(emissions_inventory: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in INVENTORY_COLUMNS if column not in emissions_inventory.columns]
    if missing:
        raise ValueError(f"Emissions-inventory file is missing required columns: {', '.join(missing)}")

    frame = emissions_inventory.copy()
    frame["calendar_year"] = pd.to_numeric(frame["calendar_year"], errors="raise").astype(int)
    frame["model_year"] = pd.to_numeric(frame["model_year"], errors="raise").astype(int)
    frame["speed"] = pd.to_numeric(frame["speed"], errors="raise")
    frame["total_vmt"] = pd.to_numeric(frame["total_vmt"], errors="coerce")
    frame["nh3_runex"] = pd.to_numeric(frame["nh3_runex"], errors="coerce")
    frame["fuel"] = frame["fuel"].astype(str)
    frame["sub_area"] = frame["region"].astype(str)
    frame["vehicle_class"] = frame["vehicle_category"].astype(str)
    frame["speed_time"] = frame["speed"]
    frame["season_month"] = "Annual"
    frame["pollutant"] = "NH3"
    frame["process"] = "RUNEX"
    valid = frame.loc[
        frame["total_vmt"].notna() & (frame["total_vmt"] > 0) & frame["nh3_runex"].notna()
    ].copy()
    valid["emission_rate"] = (valid["nh3_runex"] / valid["total_vmt"]) * GRAMS_PER_SHORT_TON
    return valid[MATCH_KEYS + ["pollutant", "process", "emission_rate"]].reset_index(drop=True)


def _refill_exact_rates(
    merged: pd.DataFrame,
    inventory_rates: pd.DataFrame,
    statewide_fallback_rates: pd.DataFrame | None,
) -> pd.DataFrame:
    if not merged["emission_rate"].isna().any():
        return merged

    missing_index = merged.index[merged["emission_rate"].isna()]
    refill_input = merged.loc[missing_index, MATCH_KEYS].copy()
    refill_merged = refill_input.merge(
        inventory_rates,
        on=MATCH_KEYS,
        how="left",
        validate="many_to_one",
    )
    if statewide_fallback_rates is not None and refill_merged["emission_rate"].isna().any():
        refill_missing_index = refill_merged.index[refill_merged["emission_rate"].isna()]
        statewide_input = refill_merged.loc[refill_missing_index, MATCH_KEYS].copy()
        statewide_input["sub_area"] = "Statewide Totals"
        statewide_merged = statewide_input.merge(
            statewide_fallback_rates,
            on=MATCH_KEYS,
            how="left",
            validate="many_to_one",
        )
        statewide_matched = statewide_merged["emission_rate"].notna()
        if statewide_matched.any():
            refill_merged.loc[
                refill_missing_index[statewide_matched.to_numpy()],
                ["emission_rate", "pollutant", "process"],
            ] = statewide_merged.loc[statewide_matched, ["emission_rate", "pollutant", "process"]].to_numpy()
    matched = refill_merged["emission_rate"].notna()
    if not matched.any():
        return merged

    merged.loc[
        missing_index[matched.to_numpy()],
        ["emission_rate", "pollutant", "process"],
    ] = refill_merged.loc[matched, ["emission_rate", "pollutant", "process"]].to_numpy()
    return merged


def _select_model_year_candidate(
    requested_model_year: int,
    available_model_years: pd.Series,
) -> int | None:
    years = sorted({int(year) for year in available_model_years.dropna().tolist()})
    if not years:
        return None

    if requested_model_year <= 2003:
        candidates = [year for year in years if year <= requested_model_year]
        return max(candidates) if candidates else None

    if 2004 <= requested_model_year <= 2013:
        candidates = [year for year in years if 2004 <= year <= 2013]
        return min(candidates, key=lambda year: (abs(year - requested_model_year), year)) if candidates else None

    if 2014 <= requested_model_year <= 2016:
        candidates = [year for year in years if 2014 <= year <= 2016]
        return min(candidates, key=lambda year: (abs(year - requested_model_year), year)) if candidates else None

    candidates = [year for year in years if year >= 2017]
    return min(candidates, key=lambda year: (abs(year - requested_model_year), year)) if candidates else None


def _apply_model_year_fallback(
    merged: pd.DataFrame,
    inventory_rates: pd.DataFrame,
) -> pd.DataFrame:
    missing_mask = merged["emission_rate"].isna()
    if not missing_mask.any():
        return merged

    available_years = {
        key: group["model_year"].drop_duplicates().sort_values()
        for key, group in inventory_rates.groupby(["vehicle_class", "fuel"], dropna=False)
    }

    for row in merged.loc[missing_mask, ["vehicle_class", "fuel", "model_year"]].itertuples(index=True):
        years = available_years.get((row.vehicle_class, row.fuel))
        if years is None:
            continue
        candidate_model_year = _select_model_year_candidate(int(row.model_year), years)
        if candidate_model_year is None:
            continue
        merged.loc[row.Index, "model_year"] = candidate_model_year
    return merged


def _apply_speed_fallback(
    merged: pd.DataFrame,
    inventory_rates: pd.DataFrame,
) -> pd.DataFrame:
    missing_mask = merged["emission_rate"].isna()
    if not missing_mask.any():
        return merged

    available_speeds = {
        key: sorted(group["speed_time"].dropna().unique().tolist())
        for key, group in inventory_rates.groupby(["vehicle_class", "fuel", "model_year"], dropna=False)
    }

    for row in merged.loc[missing_mask, ["vehicle_class", "fuel", "model_year", "speed_time"]].itertuples(index=True):
        speeds = available_speeds.get((row.vehicle_class, row.fuel, row.model_year))
        if not speeds:
            continue
        if row.speed_time > speeds[-1]:
            merged.loc[row.Index, "speed_time"] = speeds[-1]
        elif row.speed_time < speeds[0]:
            merged.loc[row.Index, "speed_time"] = speeds[0]
    return merged


def _apply_class_fuel_alternatives(
    merged: pd.DataFrame,
    alternatives: dict[tuple[str, str], tuple[str, str]],
) -> pd.DataFrame:
    if not alternatives or not merged["emission_rate"].isna().any():
        return merged

    for row in merged.loc[merged["emission_rate"].isna(), ["vehicle_class", "fuel"]].itertuples(index=True):
        replacement = alternatives.get((row.vehicle_class, row.fuel))
        if replacement is None:
            continue
        merged.loc[row.Index, ["vehicle_class", "fuel"]] = replacement

    return merged


def _build_nh3_rows(
    project_level: pd.DataFrame,
    inventory_rates: pd.DataFrame,
    alternatives: pd.DataFrame,
    statewide_fallback_rates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    reference_rates = statewide_fallback_rates if statewide_fallback_rates is not None else inventory_rates
    project_base = (
        project_level.loc[project_level["process"] == "RUNEX"]
        .drop(columns=["process", "pollutant", "emission_rate"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    merged = project_base.copy()
    merged["pollutant"] = pd.NA
    merged["process"] = pd.NA
    merged["emission_rate"] = pd.NA
    merged = _refill_exact_rates(merged, inventory_rates, statewide_fallback_rates)
    merged = _apply_class_fuel_alternatives(merged, alternatives)
    merged = _refill_exact_rates(merged, inventory_rates, statewide_fallback_rates)
    merged = _apply_model_year_fallback(merged, reference_rates)
    merged = _refill_exact_rates(merged, inventory_rates, statewide_fallback_rates)
    merged = _apply_speed_fallback(merged, reference_rates)
    merged = _refill_exact_rates(merged, inventory_rates, statewide_fallback_rates)
    return merged


def impute_project_level_nh3(
    *,
    project_level_path: str,
    emissions_inventory_path: str,
    imputation_fallback_inventory_path: str | None = None,
    output_path: str,
    drop_existing_nh3: bool = True,
) -> str:
    phase_count = 6 if imputation_fallback_inventory_path is not None else 5
    with tqdm(total=phase_count, desc="Imputing NH3", unit="phase") as progress:
        project_level = _normalize_project_level(_read_table(project_level_path, columns=PROJECT_LEVEL_COLUMNS))
        progress.set_postfix_str("project-level loaded")
        progress.update(1)

        inventory_rates = _build_inventory_rates(_read_table(emissions_inventory_path, columns=INVENTORY_COLUMNS))
        progress.set_postfix_str("regional inventory loaded")
        progress.update(1)

        statewide_fallback_rates = None
        if imputation_fallback_inventory_path is not None:
            statewide_fallback_rates = _build_inventory_rates(
                _read_table(imputation_fallback_inventory_path, columns=INVENTORY_COLUMNS)
            )
            progress.set_postfix_str("statewide inventory loaded")
            progress.update(1)

        alternatives = CLASS_FUEL_ALTERNATIVES
        progress.set_postfix_str("alternatives loaded")
        progress.update(1)

        nh3_rows = _build_nh3_rows(project_level, inventory_rates, alternatives, statewide_fallback_rates)
        progress.set_postfix_str("nh3 rows built")
        progress.update(1)

        base = project_level
        if drop_existing_nh3:
            base = base.loc[~((base["process"] == "RUNEX") & (base["pollutant"] == "NH3"))]

        result = pd.concat([base, nh3_rows], ignore_index=True)
        progress.set_postfix_str("output assembled")
        progress.update(1)

    return _write_table(result, output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.tools.emfac.impute_project_level_nh3",
        description="Append imputed NH3 RUNEX rows to cleaned project-analysis parquet using cleaned emissions-inventory parquet.",
    )
    parser.add_argument(
        "--project-level",
        required=True,
        help="Cleaned project-analysis parquet, for example processed/sfbay-emfac-project-analysis.parquet",
    )
    parser.add_argument(
        "--emissions-inventory",
        required=True,
        help="Cleaned emissions-inventory parquet, for example processed/sfbay-emfac-emissions-inventory.parquet",
    )
    parser.add_argument(
        "--imputation-fallback-inventory",
        default=None,
        help="Optional fallback emissions-inventory parquet used only for missing NH3 cohorts, for example processed/statewide-emfac-emissions-inventory.parquet",
    )
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument(
        "--keep-existing-nh3",
        action="store_true",
        help="Keep existing RUNEX NH3 rows instead of replacing them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = impute_project_level_nh3(
        project_level_path=args.project_level,
        emissions_inventory_path=args.emissions_inventory,
        imputation_fallback_inventory_path=args.imputation_fallback_inventory,
        output_path=args.output,
        drop_existing_nh3=not args.keep_existing_nh3,
    )
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
