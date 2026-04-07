from __future__ import annotations

import csv
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from impacts.emfac.common import assert_row_count
from impacts.emfac.common import frame_summary
from impacts.emfac.common import write_trace

SUB_AREA_BASE_COLUMNS = ["calendar_year", "sub_area", "vehicle_class", "fuel", "model_year"]
EMISSIONS_INVENTORY_FUEL_MAP = {
    "Gasoline": "Gas",
    "Diesel": "Dsl",
    "Electricity": "Elec",
    "Natural Gas": "NG",
    "Plug-in Hybrid": "Phe",
}
EMISSIONS_REQUIRED_COLUMNS = [
    "region",
    "calendar_year",
    "vehicleCategory",
    "modelYear",
    "speed",
    "fuel",
]
POPULATION_REQUIRED_COLUMNS = [
    "calendar_year",
    "county",
    "vehicleCategory",
    "fuel",
    "modelYear",
    "population",
]
TRIPS_REQUIRED_COLUMNS = [
    "calendar_year",
    "county",
    "vehicleCategory",
    "fuel",
    "modelYear",
    "trips",
]
ACTIVITY_JOIN_KEYS = ["calendar_year", "county", "vehicleCategory", "fuel", "modelYear"]
REQUIRED_COLUMNS_BY_SOURCE = {
    "emissions-inventory": ["region", "calendar_year", "vehicle_category", "model_year", "speed", "fuel"],
    "project-analysis": SUB_AREA_BASE_COLUMNS
    + ["process", "speed_time", "pollutant", "emission_rate"],
    "population-inventory": SUB_AREA_BASE_COLUMNS + ["population"],
    "trips-inventory": SUB_AREA_BASE_COLUMNS + ["trips"],
}
STRING_COLUMNS_BY_SOURCE = {
    "emissions-inventory": {"region", "vehicle_category", "fuel"},
    "project-analysis": {"sub_area", "vehicle_class", "fuel", "process", "pollutant"},
    "population-inventory": {"sub_area", "vehicle_class", "fuel"},
    "trips-inventory": {"sub_area", "vehicle_class", "fuel"},
}
OUTPUT_COLUMNS_BY_SOURCE = {
    "emissions-inventory": [
        "region",
        "calendar_year",
        "vehicleCategory",
        "modelYear",
        "speed",
        "fuel",
        "total_vmt",
        "cvmt",
        "evmt",
        "nox_runex",
        "pm2_5_runex",
        "pm10_runex",
        "co2_runex",
        "ch4_runex",
        "n2o_runex",
        "rog_runex",
        "tog_runex",
        "co_runex",
        "sox_runex",
        "nh3_runex",
        "pm10_pmbw",
        "pm2_5_pmbw",
        "fuel_consumption",
        "energy_consumption",
    ],
    "project-analysis": [
        "county",
        "vehicleCategory",
        "fuel",
        "modelYear",
        "process",
        "speedMps_timeMin",
        "pollutant",
        "rateGram",
    ],
    "population-inventory": [
        "calendar_year",
        "county",
        "vehicleCategory",
        "fuel",
        "modelYear",
        "population",
    ],
    "trips-inventory": [
        "calendar_year",
        "county",
        "vehicleCategory",
        "fuel",
        "modelYear",
        "trips",
    ],
}


def run_step1(workflow: dict[str, object]) -> dict[str, object]:
    print("  Step 1. Clean EMFAC Input Tables")
    print("    1.1 Clean project-analysis")
    project_analysis_path = clean_emfac_to_parquet(
        input_path=workflow["inputs"]["project_analysis_raw"],
        output_path=workflow["paths"]["project_analysis_clean"],
        source_type="project-analysis",
        region_label=workflow["run"]["region_label"],
        year=workflow["run"]["calendar_year"],
        drop_columns=["calendar_year"],
    )
    write_trace(
        workflow,
        "step1_1_project_analysis_clean",
        {
            "output_path": str(project_analysis_path),
            "summary": frame_summary(pd.read_parquet(project_analysis_path), name="project_analysis_clean"),
        },
    )
    print("    1.2 Build emissions inventory with population and trips")
    emissions_inventory_path = process_emissions_inventory_and_activities(
        emissions_inventory_input=workflow["inputs"]["emissions_inventory_raw"],
        population_input=workflow["inputs"]["population_raw"],
        trips_input=workflow["inputs"]["trips_raw"],
        output_path=workflow["paths"]["emissions_inventory_and_activities"],
        region_label=workflow["run"]["region_label"],
        year=workflow["run"]["calendar_year"],
    )
    write_trace(
        workflow,
        "step1_2_emissions_inventory_and_activities",
        {
            "output_path": str(emissions_inventory_path),
            "summary": frame_summary(pd.read_parquet(emissions_inventory_path), name="emissions_inventory_and_activities"),
        },
    )
    print("    1.3 Clean statewide emissions inventory")
    statewide_inventory_path = clean_emfac_to_parquet(
        input_path=workflow["inputs"]["statewide_emissions_inventory_raw"],
        output_path=workflow["paths"]["statewide_emissions_inventory"],
        source_type="emissions-inventory",
        year=workflow["run"]["calendar_year"],
        drop_columns=["region", "calendar_year"],
    )
    write_trace(
        workflow,
        "step1_3_statewide_emissions_inventory",
        {
            "output_path": str(statewide_inventory_path),
            "summary": frame_summary(pd.read_parquet(statewide_inventory_path), name="statewide_emissions_inventory"),
        },
    )
    return workflow


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


def _normalize_column_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def _detect_skiprows(path: Path, source_type: str) -> int:
    required_columns = set(REQUIRED_COLUMNS_BY_SOURCE[source_type])
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            normalized_row = {_normalize_column_name(value) for value in row if value.strip()}
            if required_columns.issubset(normalized_row):
                return index
    raise ValueError(f"Could not find {source_type} header row in: {path}")


def _normalize_sub_area(value: object) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(value)).strip()


def _filter_and_normalize_region_label(frame: pd.DataFrame, region_label: str | None) -> pd.DataFrame:
    if region_label is None:
        return frame
    pattern = rf"\({re.escape(region_label)}\)\s*$"
    matched_mask = frame["sub_area"].astype(str).str.contains(pattern, regex=True, na=False)
    if not matched_mask.any():
        raise ValueError(f"No sub_area rows found for region label: {region_label}")
    filtered = frame.loc[matched_mask].copy()
    filtered["region"] = region_label
    filtered["sub_area"] = filtered["sub_area"].map(_normalize_sub_area)
    return filtered


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

    required_columns = REQUIRED_COLUMNS_BY_SOURCE[source_type]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected {source_type} columns in {path}: {', '.join(missing)}")

    string_columns = STRING_COLUMNS_BY_SOURCE[source_type]
    for column in frame.columns:
        if column not in string_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if source_type == "emissions-inventory":
        frame = _normalize_emissions_inventory_fuel(frame)
    else:
        frame = _filter_and_normalize_region_label(frame, region_label)
        if region_label is None:
            frame["sub_area"] = frame["sub_area"].map(_normalize_sub_area)

    return frame


def clean_emfac_to_parquet(
    *,
    input_path: str,
    output_path: str,
    source_type: str,
    region_label: str | None = None,
    year: int | None = None,
    drop_columns: list[str] | None = None,
) -> Path:
    if source_type not in REQUIRED_COLUMNS_BY_SOURCE:
        raise ValueError(f"Unsupported source type: {source_type}")

    cleaned_frames = [
        _clean_file(path, source_type=source_type, region_label=region_label)
        for path in _iter_input_csvs(input_path)
    ]
    combined = pd.concat(cleaned_frames, ignore_index=True)
    if year is not None:
        combined = combined.loc[combined["calendar_year"] == year].copy()
    if source_type == "emissions-inventory" and "vehicle_category" in combined.columns:
        combined = combined.rename(columns={"vehicle_category": "vehicleCategory"})
    if source_type != "emissions-inventory" and "vehicle_class" in combined.columns:
        combined = combined.rename(columns={"vehicle_class": "vehicleCategory"})
    if "model_year" in combined.columns:
        combined = combined.rename(columns={"model_year": "modelYear"})
    if source_type == "project-analysis" and "speed_time" in combined.columns:
        combined = combined.rename(columns={"speed_time": "speedMps_timeMin"})
    if source_type == "project-analysis" and "emission_rate" in combined.columns:
        combined = combined.rename(columns={"emission_rate": "rateGram"})
    if source_type != "emissions-inventory" and "sub_area" in combined.columns:
        combined = combined.rename(columns={"sub_area": "county"})
    combined = combined[[column for column in OUTPUT_COLUMNS_BY_SOURCE[source_type] if column in combined.columns]]
    if drop_columns:
        existing_drop_columns = [column for column in drop_columns if column in combined.columns]
        if existing_drop_columns:
            combined = combined.drop(columns=existing_drop_columns)

    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".parquet":
        raise ValueError("Output path must end with .parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(destination, index=False)
    return destination


def _read_parquet(path: Path, required_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    return frame.copy()


def process_emissions_inventory_and_activities(
    *,
    emissions_inventory_input: str,
    population_input: str,
    trips_input: str,
    output_path: str,
    region_label: str,
    year: int,
) -> Path:
    with TemporaryDirectory(prefix="emfac_merge_activity_") as temp_dir:
        temp_root = Path(temp_dir)
        emissions_clean_path = temp_root / "emissions_inventory.parquet"
        population_clean_path = temp_root / "population_inventory.parquet"
        trips_clean_path = temp_root / "trips_inventory.parquet"

        clean_emfac_to_parquet(
            input_path=emissions_inventory_input,
            output_path=str(emissions_clean_path),
            source_type="emissions-inventory",
            year=year,
        )
        clean_emfac_to_parquet(
            input_path=population_input,
            output_path=str(population_clean_path),
            source_type="population-inventory",
            region_label=region_label,
            year=year,
        )
        clean_emfac_to_parquet(
            input_path=trips_input,
            output_path=str(trips_clean_path),
            source_type="trips-inventory",
            region_label=region_label,
            year=year,
        )

        emissions = _read_parquet(emissions_clean_path, EMISSIONS_REQUIRED_COLUMNS)
        emissions["county"] = emissions["region"].astype(str)
        population = _read_parquet(population_clean_path, POPULATION_REQUIRED_COLUMNS)
        trips = _read_parquet(trips_clean_path, TRIPS_REQUIRED_COLUMNS)

        result = emissions.merge(
            population[ACTIVITY_JOIN_KEYS + ["population"]],
            on=ACTIVITY_JOIN_KEYS,
            how="left",
            validate="many_to_one",
        ).merge(
            trips[ACTIVITY_JOIN_KEYS + ["trips"]],
            on=ACTIVITY_JOIN_KEYS,
            how="left",
            validate="many_to_one",
        )
        assert_row_count(len(emissions), len(result), label="Activity merge")
        result = result.drop(columns=["region", "calendar_year"])

        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != ".parquet":
            raise ValueError("Output path must end with .parquet")
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(destination, index=False)
        return destination
