from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from impacts.pipeline.emfac.activities.step1_prepare_emissions_and_activities_tables import _annualize_daily_values_by_vehicle_category

_CALIFORNIA_COUNTY_FIPS = {
    "alameda": "001",
    "alpine": "003",
    "amador": "005",
    "butte": "007",
    "calaveras": "009",
    "colusa": "011",
    "contra costa": "013",
    "del norte": "015",
    "el dorado": "017",
    "fresno": "019",
    "glenn": "021",
    "humboldt": "023",
    "imperial": "025",
    "inyo": "027",
    "kern": "029",
    "kings": "031",
    "lake": "033",
    "lassen": "035",
    "los angeles": "037",
    "madera": "039",
    "marin": "041",
    "mariposa": "043",
    "mendocino": "045",
    "merced": "047",
    "modoc": "049",
    "mono": "051",
    "monterey": "053",
    "napa": "055",
    "nevada": "057",
    "orange": "059",
    "placer": "061",
    "plumas": "063",
    "riverside": "065",
    "sacramento": "067",
    "san benito": "069",
    "san bernardino": "071",
    "san diego": "073",
    "san francisco": "075",
    "san joaquin": "077",
    "san luis obispo": "079",
    "san mateo": "081",
    "santa barbara": "083",
    "santa clara": "085",
    "santa cruz": "087",
    "shasta": "089",
    "sierra": "091",
    "siskiyou": "093",
    "solano": "095",
    "sonoma": "097",
    "stanislaus": "099",
    "sutter": "101",
    "tehama": "103",
    "trinity": "105",
    "tulare": "107",
    "tuolumne": "109",
    "ventura": "111",
    "yolo": "113",
    "yuba": "115",
}


def _read_table(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    lower = target.name.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(target)
    if lower.endswith(".csv.gz"):
        return pd.read_csv(target, compression="gzip")
    if lower.endswith(".csv"):
        return pd.read_csv(target)
    raise ValueError(f"Unsupported input format: {target}")


def _write_table(df: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lower = target.name.lower()
    if lower.endswith(".parquet"):
        df.to_parquet(target, index=False)
        return
    if lower.endswith(".csv.gz"):
        df.to_csv(target, index=False, compression="gzip")
        return
    if lower.endswith(".csv"):
        df.to_csv(target, index=False)
        return
    raise ValueError(f"Unsupported output format: {target}")


def _normalize_countyfp(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(3)


def _normalize_input_frame(
    frame: pd.DataFrame,
    *,
    path: str,
    county_col: str,
    vehicle_category_col: str,
    year_col: str,
    vmt_col: str,
    trips_col: str,
    default_year: Optional[int],
) -> pd.DataFrame:
    normalized = frame.copy()

    if county_col not in normalized.columns:
        raise ValueError(f"Input {path} is missing required county column '{county_col}'.")
    normalized[county_col] = _normalize_countyfp(normalized[county_col])

    if vehicle_category_col not in normalized.columns:
        raise ValueError(f"Input {path} is missing required vehicle category column '{vehicle_category_col}'.")

    if year_col not in normalized.columns:
        if default_year is not None:
            normalized[year_col] = int(default_year)
        else:
            raise ValueError(f"Input {path} is missing required year column '{year_col}'.")

    if vmt_col not in normalized.columns:
        raise ValueError(f"Input {path} is missing required VMT column '{vmt_col}'.")

    if trips_col not in normalized.columns:
        raise ValueError(f"Input {path} is missing required trips column '{trips_col}'.")

    if normalized[county_col].eq("").any():
        raise ValueError(f"Input {path} contains invalid county values in '{county_col}'.")

    return normalized[[county_col, vehicle_category_col, year_col, vmt_col, trips_col]].copy()


def _flatten_cli_values(values: Optional[list[object]]) -> list[object]:
    flattened: list[object] = []
    for value in values or []:
        if isinstance(value, (list, tuple)):
            flattened.extend(value)
        else:
            flattened.append(value)
    return flattened


def aggregate_emfac_activity(
    *,
    input_paths: list[str],
    output_path: str,
    county_col: str = "countyfp",
    vehicle_category_col: str = "vehicleCategory",
    year_col: str = "year",
    vmt_col: str = "totVMT",
    trips_col: str = "totTrips",
    default_year: Optional[int] = None,
    county_fips_filters: Optional[list[str]] = None,
    year_filters: Optional[list[int]] = None,
) -> pd.DataFrame:
    if not input_paths:
        raise ValueError("At least one input path is required.")

    frames: list[pd.DataFrame] = []
    for path in input_paths:
        frame = _read_table(path)
        frames.append(
            _normalize_input_frame(
                frame,
                path=path,
                county_col=county_col,
                vehicle_category_col=vehicle_category_col,
                year_col=year_col,
                vmt_col=vmt_col,
                trips_col=trips_col,
                default_year=default_year,
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    combined[county_col] = _normalize_countyfp(combined[county_col])
    combined[year_col] = pd.to_numeric(combined[year_col], errors="raise").astype(int)
    combined[vmt_col] = _annualize_daily_values_by_vehicle_category(
        combined,
        source_column=vmt_col,
        vehicle_category_column=vehicle_category_col,
    )
    combined[trips_col] = _annualize_daily_values_by_vehicle_category(
        combined,
        source_column=trips_col,
        vehicle_category_column=vehicle_category_col,
    )

    if county_fips_filters:
        normalized_filters = {
            _normalize_countyfp(pd.Series([value])).iloc[0] for value in _flatten_cli_values(county_fips_filters)
        }
        combined = combined.loc[combined[county_col].isin(normalized_filters)].copy()

    if year_filters:
        normalized_years = {int(value) for value in _flatten_cli_values(year_filters)}
        combined = combined.loc[combined[year_col].isin(normalized_years)].copy()

    grouped = (
        combined.groupby([county_col, year_col], dropna=False)[[vmt_col, trips_col]]
        .sum()
        .reset_index()
        .sort_values([year_col, county_col], kind="stable")
        .reset_index(drop=True)
    )
    grouped = grouped.rename(
        columns={
            vmt_col: "tot_vmt_vehicle_miles_per_year",
            trips_col: "tot_trips_per_year",
        }
    )
    _write_table(grouped, output_path)
    return grouped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.tools.emfac.aggregate_emfac_activity",
        description="Aggregate EMFAC county-year activity and output annual totals with explicit units.",
    )
    parser.add_argument("--input", dest="inputs", action="append", required=True, help="Input CSV/CSV.GZ/Parquet file. Repeat for multiple files.")
    parser.add_argument("--output", required=True, help="Output CSV/CSV.GZ/Parquet file.")
    parser.add_argument("--county-col", default="countyfp")
    parser.add_argument("--vehicle-category-col", default="vehicleCategory")
    parser.add_argument("--year-col", default="year")
    parser.add_argument("--vmt-col", default="totVMT")
    parser.add_argument("--trips-col", default="totTrips")
    parser.add_argument("--year", dest="default_year", type=int)
    parser.add_argument(
        "--county-fips",
        dest="county_fips_filters",
        nargs="+",
        default=None,
        help="Filter to one or more county FIPS codes.",
    )
    parser.add_argument(
        "--filter-year",
        dest="year_filters",
        nargs="+",
        type=int,
        default=None,
        help="Filter to one or more years.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    aggregate_emfac_activity(
        input_paths=args.inputs,
        output_path=args.output,
        county_col=args.county_col,
        vehicle_category_col=args.vehicle_category_col,
        year_col=args.year_col,
        vmt_col=args.vmt_col,
        trips_col=args.trips_col,
        default_year=args.default_year,
        county_fips_filters=args.county_fips_filters,
        year_filters=args.year_filters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
