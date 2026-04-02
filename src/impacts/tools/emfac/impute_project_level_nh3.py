from __future__ import annotations
"""Impute NH3 RUNEX rows into cleaned project-analysis parquet data.

Expected inputs:
- project level parquet with columns:
  calendar_year, season_month, sub_area, vehicle_class, fuel, model_year,
  temperature, relative_humidity, process, speed_time, pollutant, emission_rate
- emissions inventory parquet with columns:
  region, calendar_year, vehicle_category, model_year, speed, fuel, total_vmt,
  nh3_runex
"""

import argparse
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

GRAMS_PER_SHORT_TON = 907_184.74
PROJECT_LEVEL_KEYS = [
    "calendar_year",
    "season_month",
    "sub_area",
    "vehicle_class",
    "fuel",
    "model_year",
    "speed_time",
]

FUEL_NORMALIZATION = {
    "gasoline": "gasoline",
    "gas": "gasoline",
    "diesel": "diesel",
    "dsl": "diesel",
    "electricity": "electricity",
    "elec": "electricity",
    "naturalgas": "naturalgas",
    "ng": "naturalgas",
    "pluginhybrid": "pluginhybrid",
    "phe": "pluginhybrid",
}


def _sanitize_token(value: object) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def _normalize_fuel_series(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda value: FUEL_NORMALIZATION.get(_sanitize_token(value), str(value)))


def _read_table(path: str | Path) -> pd.DataFrame:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Input path does not exist: {target}")
    if target.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported input format for {target}. Expected .parquet")
    return pd.read_parquet(target)


def _write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported output format for {target}. Expected .parquet")
    frame.to_parquet(target, index=False)
    return target


def _normalize_project_level(project_level: pd.DataFrame) -> pd.DataFrame:
    required = [
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
    missing = [column for column in required if column not in project_level.columns]
    if missing:
        raise ValueError(f"Project-level file is missing required columns: {', '.join(missing)}")
    frame = project_level.copy()
    frame["calendar_year"] = pd.to_numeric(frame["calendar_year"], errors="raise").astype(int)
    frame["model_year"] = pd.to_numeric(frame["model_year"], errors="raise").astype(int)
    frame["speed_time"] = pd.to_numeric(frame["speed_time"], errors="raise")
    frame["temperature"] = pd.to_numeric(frame["temperature"], errors="raise")
    frame["relative_humidity"] = pd.to_numeric(frame["relative_humidity"], errors="raise")
    frame["emission_rate"] = pd.to_numeric(frame["emission_rate"], errors="raise")
    frame["fuel"] = _normalize_fuel_series(frame["fuel"])
    for column in ["season_month", "sub_area", "vehicle_class", "process", "pollutant"]:
        frame[column] = frame[column].astype(str)
    return frame


def _build_inventory_rates(emissions_inventory: pd.DataFrame) -> pd.DataFrame:
    required = [
        "region",
        "calendar_year",
        "vehicle_category",
        "model_year",
        "speed",
        "fuel",
        "total_vmt",
        "nh3_runex",
    ]
    missing = [column for column in required if column not in emissions_inventory.columns]
    if missing:
        raise ValueError(f"Emissions-inventory file is missing required columns: {', '.join(missing)}")

    frame = emissions_inventory.copy()
    frame["calendar_year"] = pd.to_numeric(frame["calendar_year"], errors="raise").astype(int)
    frame["model_year"] = pd.to_numeric(frame["model_year"], errors="raise").astype(int)
    frame["speed"] = pd.to_numeric(frame["speed"], errors="raise")
    frame["total_vmt"] = pd.to_numeric(frame["total_vmt"], errors="coerce")
    frame["nh3_runex"] = pd.to_numeric(frame["nh3_runex"], errors="coerce")
    frame["fuel"] = _normalize_fuel_series(frame["fuel"])
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
    return valid[PROJECT_LEVEL_KEYS + ["pollutant", "process", "emission_rate"]].reset_index(drop=True)


def _fill_missing_rates_from_nearest_speed(
    missing_keys: pd.DataFrame,
    inventory_rates: pd.DataFrame,
) -> pd.DataFrame:
    group_keys = [key for key in PROJECT_LEVEL_KEYS if key != "speed_time"]
    inventory_groups = {
        group: frame.sort_values("speed_time").reset_index(drop=True)
        for group, frame in inventory_rates.groupby(group_keys, dropna=False)
    }
    filled_rows: list[dict[str, object]] = []
    for row in missing_keys.to_dict(orient="records"):
        group = tuple(row[key] for key in group_keys)
        candidates = inventory_groups.get(group)
        if candidates is None or candidates.empty:
            filled_rows.append({"pollutant": None, "process": None, "emission_rate": pd.NA})
            continue
        distance = (candidates["speed_time"] - row["speed_time"]).abs()
        nearest = candidates.loc[distance.idxmin()]
        filled_rows.append(
            {
                "pollutant": nearest["pollutant"],
                "process": nearest["process"],
                "emission_rate": nearest["emission_rate"],
            }
        )
    return pd.DataFrame(filled_rows)


def _fill_missing_rates(
    merged: pd.DataFrame,
    inventory_rates: pd.DataFrame,
    *,
    group_keys: list[str],
) -> pd.DataFrame:
    if merged["emission_rate"].isna().any():
        missing_mask = merged["emission_rate"].isna()
        fallback_rates = _fill_missing_rates_from_nearest_speed(
            merged.loc[missing_mask, PROJECT_LEVEL_KEYS],
            inventory_rates,
        )
        missing_index = merged.index[missing_mask]
        merged.loc[missing_index, "emission_rate"] = pd.to_numeric(fallback_rates["emission_rate"], errors="coerce").tolist()
        merged.loc[missing_index, "pollutant"] = fallback_rates["pollutant"].tolist()
        merged.loc[missing_index, "process"] = fallback_rates["process"].tolist()
    if merged["emission_rate"].isna().any():
        missing_mask = merged["emission_rate"].isna()
        fallback_rates = _fill_missing_rates_from_nearest_model_year_and_speed_grouped(
            merged.loc[missing_mask, PROJECT_LEVEL_KEYS],
            inventory_rates,
            group_keys=group_keys,
        )
        missing_index = merged.index[missing_mask]
        merged.loc[missing_index, "emission_rate"] = pd.to_numeric(fallback_rates["emission_rate"], errors="coerce").tolist()
        merged.loc[missing_index, "pollutant"] = fallback_rates["pollutant"].tolist()
        merged.loc[missing_index, "process"] = fallback_rates["process"].tolist()
    return merged


def _fill_missing_rates_from_nearest_model_year_and_speed_grouped(
    missing_keys: pd.DataFrame,
    inventory_rates: pd.DataFrame,
    *,
    group_keys: list[str],
) -> pd.DataFrame:
    inventory_groups = {
        group: frame.reset_index(drop=True)
        for group, frame in inventory_rates.groupby(group_keys, dropna=False)
    }
    filled_rows: list[dict[str, object]] = []
    for row in missing_keys.to_dict(orient="records"):
        group = tuple(row[key] for key in group_keys)
        candidates = inventory_groups.get(group)
        if candidates is None or candidates.empty:
            filled_rows.append({"pollutant": None, "process": None, "emission_rate": pd.NA})
            continue
        ranked = candidates.assign(
            _model_year_distance=(candidates["model_year"] - row["model_year"]).abs(),
            _speed_distance=(candidates["speed_time"] - row["speed_time"]).abs(),
        ).sort_values(["_model_year_distance", "_speed_distance", "model_year", "speed_time"], kind="stable")
        nearest = ranked.iloc[0]
        filled_rows.append(
            {
                "pollutant": nearest["pollutant"],
                "process": nearest["process"],
                "emission_rate": nearest["emission_rate"],
            }
        )
    return pd.DataFrame(filled_rows)


def _build_nh3_rows(
    project_level: pd.DataFrame,
    inventory_rates: pd.DataFrame,
    statewide_fallback_rates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    project_base = (
        project_level.loc[project_level["process"] == "RUNEX"]
        .drop(columns=["process", "pollutant", "emission_rate"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    merged = project_base.merge(
        inventory_rates,
        on=PROJECT_LEVEL_KEYS,
        how="left",
        validate="many_to_one",
    )
    merged = _fill_missing_rates(
        merged,
        inventory_rates,
        group_keys=["calendar_year", "season_month", "sub_area", "vehicle_class", "fuel"],
    )
    if merged["emission_rate"].isna().any() and statewide_fallback_rates is not None:
        fallback_input = merged.loc[merged["emission_rate"].isna(), project_base.columns].copy()
        fallback_input["sub_area"] = "Statewide Totals"
        fallback_merged = fallback_input.merge(
            statewide_fallback_rates,
            on=PROJECT_LEVEL_KEYS,
            how="left",
            validate="many_to_one",
        )
        fallback_merged = _fill_missing_rates(
            fallback_merged,
            statewide_fallback_rates,
            group_keys=["calendar_year", "season_month", "sub_area", "vehicle_class", "fuel"],
        )
        matched = fallback_merged["emission_rate"].notna()
        if matched.any():
            replacement_index = merged.index[merged["emission_rate"].isna()][matched.to_numpy()]
            merged.loc[replacement_index, "emission_rate"] = pd.to_numeric(
                fallback_merged.loc[matched, "emission_rate"], errors="coerce"
            ).tolist()
            merged.loc[replacement_index, "pollutant"] = fallback_merged.loc[matched, "pollutant"].tolist()
            merged.loc[replacement_index, "process"] = fallback_merged.loc[matched, "process"].tolist()
    if merged["emission_rate"].isna().any():
        missing = merged.loc[merged["emission_rate"].isna(), PROJECT_LEVEL_KEYS].drop_duplicates()
        print(
            f"warning: could not find NH3 inventory rates for {len(missing)} project-level key combinations; "
            f"skipping those rows. Sample missing keys: {missing.head(10).to_dict(orient='records')}",
            file=sys.stderr,
        )
        merged = merged.loc[merged["emission_rate"].notna()].copy()
    return merged


def impute_project_level_nh3(
    *,
    project_level_path: str,
    emissions_inventory_path: str,
    imputation_fallback_inventory_path: str | None = None,
    output_path: str,
    drop_existing_nh3: bool = True,
) -> Path:
    project_level = _normalize_project_level(_read_table(project_level_path))
    inventory_rates = _build_inventory_rates(_read_table(emissions_inventory_path))
    statewide_fallback_rates = (
        _build_inventory_rates(_read_table(imputation_fallback_inventory_path))
        if imputation_fallback_inventory_path is not None
        else None
    )
    nh3_rows = _build_nh3_rows(project_level, inventory_rates, statewide_fallback_rates)

    base = project_level.copy()
    if drop_existing_nh3:
        base = base.loc[~((base["process"] == "RUNEX") & (base["pollutant"] == "NH3"))].copy()

    result = pd.concat([base, nh3_rows], ignore_index=True)
    result = result.sort_values(
        [
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
        ],
        kind="stable",
    ).reset_index(drop=True)
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


def main(argv: Optional[list[str]] = None) -> int:
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
