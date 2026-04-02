from __future__ import annotations
"""Clean raw EMFAC exports into a single normalized parquet file.

Supported source types:
- emissions-inventory
- project-analysis
- population-inventory
- trips-inventory
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


INVENTORY_HEADER_PREFIX = 'Region,"Calendar Year","Vehicle Category","Model Year",Speed,Fuel'
PROJECT_ANALYSIS_HEADER_PREFIX = "calendar_year,season_month,sub_area,vehicle_class,fuel,model_year,temperature,relative_humidity,process,speed_time,pollutant,emission_rate"
POPULATION_INVENTORY_HEADER_PREFIX = "calendar_year,sub_area,vehicle_class,fuel,model_year,population"
TRIPS_INVENTORY_HEADER_PREFIX = "calendar_year,sub_area,vehicle_class,fuel,model_year,trips"
INVENTORY_COLUMN_RENAME_MAP = {
    "Region": "region",
    "Calendar Year": "calendar_year",
    "Vehicle Category": "vehicle_category",
    "Model Year": "model_year",
    "Speed": "speed",
    "Fuel": "fuel",
    "Total VMT": "total_vmt",
    "CVMT": "cvmt",
    "EVMT": "evmt",
    "NOx_RUNEX": "nox_runex",
    "PM2.5_RUNEX": "pm25_runex",
    "PM10_RUNEX": "pm10_runex",
    "CO2_RUNEX": "co2_runex",
    "CH4_RUNEX": "ch4_runex",
    "N2O_RUNEX": "n2o_runex",
    "ROG_RUNEX": "rog_runex",
    "TOG_RUNEX": "tog_runex",
    "CO_RUNEX": "co_runex",
    "SOx_RUNEX": "sox_runex",
    "NH3_RUNEX": "nh3_runex",
    "PM10_PMBW": "pm10_pmbw",
    "PM2.5_PMBW": "pm25_pmbw",
    "Fuel Consumption": "fuel_consumption",
    "Energy Consumption": "energy_consumption",
}
PROJECT_ANALYSIS_COLUMN_RENAME_MAP = {
    "calendar_year": "calendar_year",
    "season_month": "season_month",
    "sub_area": "sub_area",
    "vehicle_class": "vehicle_class",
    "fuel": "fuel",
    "model_year": "model_year",
    "temperature": "temperature",
    "relative_humidity": "relative_humidity",
    "process": "process",
    "speed_time": "speed_time",
    "pollutant": "pollutant",
    "emission_rate": "emission_rate",
}
POPULATION_INVENTORY_COLUMN_RENAME_MAP = {
    "calendar_year": "calendar_year",
    "sub_area": "sub_area",
    "vehicle_class": "vehicle_class",
    "fuel": "fuel",
    "model_year": "model_year",
    "population": "population",
}
TRIPS_INVENTORY_COLUMN_RENAME_MAP = {
    "calendar_year": "calendar_year",
    "sub_area": "sub_area",
    "vehicle_class": "vehicle_class",
    "fuel": "fuel",
    "model_year": "model_year",
    "trips": "trips",
}
INVENTORY_NUMERIC_COLUMNS = [
    "Calendar Year",
    "Model Year",
    "Speed",
    "Total VMT",
    "CVMT",
    "EVMT",
    "NOx_RUNEX",
    "PM2.5_RUNEX",
    "PM10_RUNEX",
    "CO2_RUNEX",
    "CH4_RUNEX",
    "N2O_RUNEX",
    "ROG_RUNEX",
    "TOG_RUNEX",
    "CO_RUNEX",
    "SOx_RUNEX",
    "NH3_RUNEX",
    "PM10_PMBW",
    "PM2.5_PMBW",
    "Fuel Consumption",
    "Energy Consumption",
]
PROJECT_ANALYSIS_NUMERIC_COLUMNS = [
    "calendar_year",
    "model_year",
    "temperature",
    "relative_humidity",
    "speed_time",
    "emission_rate",
]
POPULATION_INVENTORY_NUMERIC_COLUMNS = [
    "calendar_year",
    "model_year",
    "population",
]
TRIPS_INVENTORY_NUMERIC_COLUMNS = [
    "calendar_year",
    "model_year",
    "trips",
]


def _iter_input_csvs(path: str | Path) -> list[Path]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Input path does not exist: {target}")
    if target.is_file():
        if target.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file: {target}")
        return [target]
    files = sorted(p for p in target.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {target}")
    return files


def _detect_skiprows(path: Path, source_type: str) -> int:
    if source_type == "emissions-inventory":
        header_prefix = INVENTORY_HEADER_PREFIX
    elif source_type == "project-analysis":
        header_prefix = PROJECT_ANALYSIS_HEADER_PREFIX
    elif source_type == "population-inventory":
        header_prefix = POPULATION_INVENTORY_HEADER_PREFIX
    elif source_type == "trips-inventory":
        header_prefix = TRIPS_INVENTORY_HEADER_PREFIX
    else:
        raise ValueError(f"Unsupported source type: {source_type}")
    with path.open() as handle:
        for index, line in enumerate(handle):
            if line.strip().startswith(header_prefix):
                return index
    raise ValueError(f"Could not find {source_type} header row in: {path}")


def _normalize_sub_area(value: object) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(value)).strip()


def _filter_and_normalize_region_label(frame: pd.DataFrame, region_label: str | None) -> pd.DataFrame:
    if region_label is None:
        return frame
    if "sub_area" not in frame.columns:
        raise ValueError("--region-label is only supported for source types with a sub_area column.")
    pattern = rf"\({re.escape(region_label)}\)\s*$"
    matched_mask = frame["sub_area"].astype(str).str.contains(pattern, regex=True, na=False)
    if matched_mask.any():
        filtered = frame.loc[matched_mask].copy()
        filtered["region"] = region_label
        filtered["sub_area"] = filtered["sub_area"].map(_normalize_sub_area)
        return filtered
    normalized = frame.copy()
    normalized["sub_area"] = normalized["sub_area"].map(_normalize_sub_area)
    return normalized


def _clean_emissions_inventory_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=_detect_skiprows(path, "emissions-inventory"))
    for column in INVENTORY_NUMERIC_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"Missing expected column '{column}' in {path}")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.rename(columns=INVENTORY_COLUMN_RENAME_MAP)
    frame["source_file"] = str(path)
    return frame


def _clean_project_analysis_file(path: Path, region_label: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=_detect_skiprows(path, "project-analysis"))
    missing = [column for column in PROJECT_ANALYSIS_COLUMN_RENAME_MAP if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected project-analysis columns in {path}: {', '.join(missing)}")
    for column in PROJECT_ANALYSIS_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.rename(columns=PROJECT_ANALYSIS_COLUMN_RENAME_MAP)
    frame = _filter_and_normalize_region_label(frame, region_label)
    if region_label is None:
        frame["sub_area"] = frame["sub_area"].map(_normalize_sub_area)
    frame["source_file"] = str(path)
    return frame


def _clean_population_inventory_file(path: Path, region_label: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=_detect_skiprows(path, "population-inventory"))
    missing = [column for column in POPULATION_INVENTORY_COLUMN_RENAME_MAP if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected population-inventory columns in {path}: {', '.join(missing)}")
    for column in POPULATION_INVENTORY_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.rename(columns=POPULATION_INVENTORY_COLUMN_RENAME_MAP)
    frame = _filter_and_normalize_region_label(frame, region_label)
    if region_label is None:
        frame["sub_area"] = frame["sub_area"].map(_normalize_sub_area)
    frame["source_file"] = str(path)
    return frame


def _clean_trips_inventory_file(path: Path, region_label: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=_detect_skiprows(path, "trips-inventory"))
    missing = [column for column in TRIPS_INVENTORY_COLUMN_RENAME_MAP if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected trips-inventory columns in {path}: {', '.join(missing)}")
    for column in TRIPS_INVENTORY_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.rename(columns=TRIPS_INVENTORY_COLUMN_RENAME_MAP)
    frame = _filter_and_normalize_region_label(frame, region_label)
    if region_label is None:
        frame["sub_area"] = frame["sub_area"].map(_normalize_sub_area)
    frame["source_file"] = str(path)
    return frame


def clean_emfac_to_parquet(
    *,
    input_path: str,
    output_path: str,
    source_type: str,
    region_label: str | None = None,
) -> Path:
    files = _iter_input_csvs(input_path)
    if source_type == "emissions-inventory":
        if region_label is not None:
            print(
                f"warning: ignoring --region-label {region_label!r} for source-type 'emissions-inventory'",
                file=sys.stderr,
            )
        cleaned_frames = [_clean_emissions_inventory_file(path) for path in files]
    elif source_type == "project-analysis":
        cleaned_frames = [_clean_project_analysis_file(path, region_label=region_label) for path in files]
    elif source_type == "population-inventory":
        cleaned_frames = [_clean_population_inventory_file(path, region_label=region_label) for path in files]
    elif source_type == "trips-inventory":
        cleaned_frames = [_clean_trips_inventory_file(path, region_label=region_label) for path in files]
    else:
        raise ValueError(f"Unsupported source type: {source_type}")
    combined = pd.concat(cleaned_frames, ignore_index=True)

    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".parquet":
        raise ValueError("--output must be a .parquet file path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(destination, index=False)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.tools.emfac.clean_emfac_to_parquet",
        description="Remove EMFAC metadata headers and write a cleaned parquet file.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="EMFAC CSV file or folder containing EMFAC CSV files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output parquet file path. Folder input is concatenated into this single parquet.",
    )
    parser.add_argument(
        "--source-type",
        required=True,
        choices=["emissions-inventory", "project-analysis", "population-inventory", "trips-inventory"],
        help="Type of EMFAC export to clean.",
    )
    parser.add_argument(
        "--region-label",
        default=None,
        help="Optional region label such as SF. Keeps only rows whose sub_area ends with '(SF)', adds region='SF', then strips the suffix from sub_area.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = clean_emfac_to_parquet(
        input_path=args.input,
        output_path=args.output,
        source_type=args.source_type,
        region_label=args.region_label,
    )
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
