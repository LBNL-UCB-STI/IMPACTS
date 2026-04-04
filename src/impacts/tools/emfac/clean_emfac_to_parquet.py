from __future__ import annotations
"""Clean raw EMFAC exports into a single normalized parquet file.

Supported source types:
- emissions-inventory
- project-analysis
- population-inventory
- trips-inventory
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd


SOURCE_TYPES = (
    "emissions-inventory",
    "project-analysis",
    "population-inventory",
    "trips-inventory",
)
SUB_AREA_BASE_COLUMNS = ["calendar_year", "sub_area", "vehicle_class", "fuel", "model_year"]
EMISSIONS_INVENTORY_FUEL_MAP = {
    "Gasoline": "Gas",
    "Diesel": "Dsl",
    "Electricity": "Elec",
    "Natural Gas": "NG",
    "Plug-in Hybrid": "Phe",
}


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


def _required_columns(source_type: str) -> list[str]:
    if source_type == "emissions-inventory":
        return ["region", "calendar_year", "vehicle_category", "model_year", "speed", "fuel"]
    if source_type == "project-analysis":
        return SUB_AREA_BASE_COLUMNS + [
            "season_month",
            "temperature",
            "relative_humidity",
            "process",
            "speed_time",
            "pollutant",
            "emission_rate",
        ]
    if source_type == "population-inventory":
        return SUB_AREA_BASE_COLUMNS + ["population"]
    if source_type == "trips-inventory":
        return SUB_AREA_BASE_COLUMNS + ["trips"]
    raise ValueError(f"Unsupported source type: {source_type}")


def _string_columns(source_type: str) -> set[str]:
    if source_type == "emissions-inventory":
        return {"region", "vehicle_category", "fuel"}
    if source_type == "project-analysis":
        return {"season_month", "sub_area", "vehicle_class", "fuel", "process", "pollutant"}
    if source_type in {"population-inventory", "trips-inventory"}:
        return {"sub_area", "vehicle_class", "fuel"}
    raise ValueError(f"Unsupported source type: {source_type}")


def _has_sub_area(source_type: str) -> bool:
    return source_type != "emissions-inventory"


def _detect_skiprows(path: Path, source_type: str) -> int:
    required_columns = set(_required_columns(source_type))
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            normalized_row = {_normalize_column_name(value) for value in row if value.strip()}
            if required_columns.issubset(normalized_row):
                return index
    raise ValueError(f"Could not find {source_type} header row in: {path}")


def _normalize_column_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


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


def _filter_calendar_year(frame: pd.DataFrame, year: int | None) -> pd.DataFrame:
    if year is None:
        return frame
    if "calendar_year" not in frame.columns:
        raise ValueError("--year is only supported for source types with a calendar_year column.")
    return frame.loc[frame["calendar_year"] == year].copy()


def _normalize_emissions_inventory_fuel(frame: pd.DataFrame) -> pd.DataFrame:
    unknown = sorted(set(frame["fuel"].dropna().astype(str)) - set(EMISSIONS_INVENTORY_FUEL_MAP))
    if unknown:
        raise ValueError(f"Unsupported emissions-inventory fuel values: {unknown}")
    frame = frame.copy()
    frame["fuel"] = frame["fuel"].map(EMISSIONS_INVENTORY_FUEL_MAP)
    return frame


def _clean_file(path: Path, *, source_type: str, region_label: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path, skiprows=_detect_skiprows(path, source_type))
    frame = frame.rename(columns={column: _normalize_column_name(column) for column in frame.columns})

    required_columns = _required_columns(source_type)
    missing_normalized = [column for column in required_columns if column not in frame.columns]
    if missing_normalized:
        raise ValueError(f"Missing expected {source_type} columns in {path}: {', '.join(missing_normalized)}")

    string_columns = _string_columns(source_type)
    for column in frame.columns:
        if column not in string_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if _has_sub_area(source_type):
        frame = _filter_and_normalize_region_label(frame, region_label)
        if region_label is None:
            frame["sub_area"] = frame["sub_area"].map(_normalize_sub_area)
    elif "fuel" in frame.columns:
        frame = _normalize_emissions_inventory_fuel(frame)

    return frame


def clean_emfac_to_parquet(
    *,
    input_path: str,
    output_path: str,
    source_type: str,
    region_label: str | None = None,
    year: int | None = None,
) -> Path:
    files = _iter_input_csvs(input_path)
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"Unsupported source type: {source_type}")
    if region_label is not None and not _has_sub_area(source_type):
        print(
            f"warning: ignoring --region-label {region_label!r} for source-type '{source_type}'",
            file=sys.stderr,
        )
        region_label = None
    cleaned_frames = [
        _clean_file(path, source_type=source_type, region_label=region_label)
        for path in files
    ]
    combined = pd.concat(cleaned_frames, ignore_index=True)
    combined = _filter_calendar_year(combined, year)

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
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Optional calendar year filter applied after cleaning and concatenation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = clean_emfac_to_parquet(
        input_path=args.input,
        output_path=args.output,
        source_type=args.source_type,
        region_label=args.region_label,
        year=args.year,
    )
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
