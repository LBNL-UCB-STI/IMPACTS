from __future__ import annotations

import csv
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from functools import lru_cache

import pandas as pd

from impacts.emfac.common import frame_summary
from impacts.emfac.common import write_trace

GRAMS_PER_SHORT_TON = 907_184.74
METRIC_TONS_PER_SHORT_TON = 0.90718474
EMFAC_DAYS_PER_YEAR = 365.0
PTO_PROCESS_NAME = "PTOEX"
OPERATION_DAYS_CSV = Path(__file__).with_name("vehicle_operation_days_per_year.csv")
FUEL_LABEL_MAP = {
    "Diesel": "Dsl",
    "Electricity": "Elec",
    "Gasoline": "Gas",
    "Natural Gas": "NG",
    "Plug-in Hybrid": "Phe",
}
EMFAC202X_VEHICLE_CATEGORY_ALIASES = {
    "LHD1 Public": "LHD1",
    "LHD1 Other": "LHD1",
    "LHD2 Public": "LHD2",
    "LHD2 Other": "LHD2",
    "All Other Buses": "OBUS",
}
BEAM_TO_CARB_ROAD_CATEGORY = {
    "motorway": "Freeway",
    "motorway_link": "Freeway",
    "trunk": "Freeway",
    "trunk_link": "Major",
    "primary": "Major",
    "primary_link": "Major",
    "secondary": "Collector",
    "secondary_link": "Collector",
    "tertiary": "Collector",
    "tertiary_link": "Collector",
    "unclassified": "Collector",
    "residential": "Local Urban",
}
PROJECT_ANALYSIS_KEY_COLUMNS = ["county", "vehicleCategory", "fuel", "modelYear", "process", "speedMph_timeMin"]
HEADER_DETECTION_COLUMNS = {
    "emissions-inventory": {"region", "calendar_year", "vehicle_category", "model_year", "speed", "fuel"},
}
ANNUAL_ACTIVITY_COLUMN_MAP = {
    "total_vmt": "total_vmt_vehicle_miles_per_year",
    "cvmt": "cvmt_vehicle_miles_per_year",
    "evmt": "evmt_vehicle_miles_per_year",
    "trips": "trips_per_year",
    "population": "population_vehicles",
    "pto_total_vmt": "pto_total_vmt_vehicle_miles_per_year",
}


def pollutant_to_column_name(pollutant: object) -> str:
    label = str(pollutant).strip().lower().replace(".", "_").replace("-", "_")
    label = re.sub(r"[^a-z0-9_]+", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return f"{label}_gram"


def expand_pto_vehicle_category(
    frame: pd.DataFrame,
    pto_config: dict[str, object] | None,
    *,
    vehicle_category_column: str = "vehicleCategory",
    process_column: str | None = None,
    source_vehicle_category: str = "PTO",
) -> pd.DataFrame:
    if not pto_config or not pto_config.get("enabled"):
        return frame
    targets = pto_config.get("targets") or []
    if not targets:
        return frame
    source_mask = frame[vehicle_category_column].astype(str) == source_vehicle_category
    if not source_mask.any():
        return frame
    source_rows = frame.loc[source_mask].copy()
    expanded_frames: list[pd.DataFrame] = []
    for target in targets:
        target_rows = source_rows.copy()
        if target_rows.empty:
            continue
        target_rows[vehicle_category_column] = str(target)
        if process_column is not None:
            target_rows[process_column] = PTO_PROCESS_NAME
        expanded_frames.append(target_rows)
    if not expanded_frames:
        return frame.loc[~source_mask].copy()
    result = pd.concat([frame.loc[~source_mask].copy(), *expanded_frames], ignore_index=True)
    return result.drop_duplicates().reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _require_unique_keys(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicate_count = int(frame.duplicated(keys).sum())
    if duplicate_count:
        raise ValueError(f"{label} contains {duplicate_count} duplicate rows for keys {keys}.")


def _extract_silt_road_categories(frame: pd.DataFrame) -> list[str]:
    base_columns = {"Air Basin", "County", "Air District"}
    road_categories = [column for column in frame.columns if column not in base_columns]
    if not road_categories:
        raise ValueError("Silt loading file does not contain any road-category columns.")
    return road_categories


def _read_county_air_basin_table(
    path: str | Path,
    *,
    required_columns: set[str],
    air_basin_region: list[str] | None,
    label: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, required_columns, label)
    frame["County"] = frame["County"].astype(str).str.strip().str.title()
    frame["Air Basin"] = frame["Air Basin"].astype(str).str.strip()
    if not air_basin_region:
        return frame
    filtered = frame[frame["Air Basin"].isin(air_basin_region)].copy()
    if filtered.empty:
        raise ValueError(f"No {label.lower()} rows found for air basins: {air_basin_region}")
    return filtered


def _read_silt_loading_table(
    path: str | Path,
    *,
    air_basin_region: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    frame = _read_county_air_basin_table(
        path,
        required_columns={"County", "Air Basin"},
        air_basin_region=air_basin_region,
        label="Silt loading file",
    )
    road_categories = _extract_silt_road_categories(frame)
    return frame, road_categories


def _calculate_road_dust_emissions_series(
    silt_loading: pd.Series,
    rainy_days: pd.Series,
    *,
    annual_days: int = 365,
) -> pd.DataFrame:
    k = 0.0022
    pm25_fraction = 0.0686
    pm10_fraction = 0.4572
    pounds_to_grams = 453.592
    pm10_lb_per_vmt = (
        k
        * (silt_loading.astype(float) ** 0.91)
        * (1 - rainy_days.astype(float) / annual_days / 4)
    )
    total_pm_lb_per_vmt = pm10_lb_per_vmt / pm10_fraction
    return pd.DataFrame(
        {
            "pm2_5_rate": total_pm_lb_per_vmt * pm25_fraction * pounds_to_grams,
            "pm10_rate": pm10_lb_per_vmt * pounds_to_grams,
            "pm_rate": total_pm_lb_per_vmt * pounds_to_grams,
        }
    )


def _beam_road_mapping(*, supported_categories: list[str]) -> pd.DataFrame:
    supported_category_set = set(supported_categories)
    rows = []
    for road_category, carb_road_category in BEAM_TO_CARB_ROAD_CATEGORY.items():
        if carb_road_category not in supported_category_set:
            raise ValueError(f"Unsupported road type '{carb_road_category}' in beam road mapping.")
        rows.append({"roadCategory": road_category, "carb_road_category": carb_road_category})
    return pd.DataFrame(rows)

def build_road_dust_rows(
    project_analysis: pd.DataFrame,
    *,
    rainy_days_file: str,
    silt_loading_file: str,
    air_basins: list[str] | None,
) -> pd.DataFrame:
    silt_filtered, road_categories = _read_silt_loading_table(
        silt_loading_file,
        air_basin_region=air_basins,
    )
    county_averages = silt_filtered.groupby("County")[road_categories].mean().reset_index()
    rainy_filtered = _read_county_air_basin_table(
        rainy_days_file,
        required_columns={"County", "Air Basin", "Annual Rainfall Days"},
        air_basin_region=air_basins,
        label="Rainy days file",
    )
    rainfall_averages = rainy_filtered.groupby("County")["Annual Rainfall Days"].mean().reset_index()
    county_inputs = pd.merge(county_averages, rainfall_averages, on="County", how="inner")
    road_mappings = _beam_road_mapping(supported_categories=road_categories)
    county_long = county_inputs.melt(
        id_vars=["County", "Annual Rainfall Days"],
        value_vars=road_categories,
        var_name="carb_road_category",
        value_name="silt_loading",
    )
    result = county_long.merge(road_mappings, on="carb_road_category", how="inner")
    emission_rates = _calculate_road_dust_emissions_series(
        result["silt_loading"],
        result["Annual Rainfall Days"],
    )
    result = pd.concat([result, emission_rates], axis=1)

    result = result.rename(columns={"County": "county", "Annual Rainfall Days": "rainy_days"})
    cohort_keys = ["county", "vehicleCategory", "fuel", "modelYear"]
    cohorts = project_analysis[cohort_keys].drop_duplicates().reset_index(drop=True)
    merged = cohorts.merge(
        result[
            [
                "county",
                "roadCategory",
                "rainy_days",
                "silt_loading",
                "pm_rate",
                "pm10_rate",
                "pm2_5_rate",
            ]
        ],
        on=["county"],
        how="inner",
    )
    merged["process"] = "PRDUST"
    long = merged.melt(
        id_vars=[
            "county",
            "vehicleCategory",
            "fuel",
            "modelYear",
            "process",
            "roadCategory",
            "rainy_days",
            "silt_loading",
        ],
        value_vars=["pm_rate", "pm10_rate", "pm2_5_rate"],
        var_name="rate_column",
        value_name="rateGram",
    )
    long["pollutant"] = long["rate_column"].map({"pm_rate": "PM", "pm10_rate": "PM10", "pm2_5_rate": "PM2_5"})
    long["speedMph_timeMin"] = pd.NA
    return long.drop(columns=["rate_column", "rainy_days", "silt_loading"]).reset_index(drop=True)


def build_black_carbon_rows(
    path: str,
    *,
    region_label: str | None,
    source_pollutant: str,
    pto_config: dict[str, object] | None,
) -> pd.DataFrame:
    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported black-carbon format for {target}. Expected .csv")

    frame = pd.read_csv(target)
    missing = [
        column
        for column in ["sub_area", "vehicle_class", "fuel", "model_year", "process", "speed_time", "pollutant", "emission_rate"]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Black-carbon CSV is missing required columns: {', '.join(missing)}")

    if region_label:
        suffix = f"({region_label})"
        frame = frame.loc[frame["sub_area"].astype(str).str.endswith(suffix)].copy()
        if frame.empty:
            raise ValueError(f"No black-carbon rows matched region label {region_label!r} in {target}")

    frame["county"] = frame["sub_area"].astype(str).str.replace(r"\s*\([^)]*\)$", "", regex=True).str.strip()
    frame["vehicleCategory"] = frame["vehicle_class"].astype(str).str.strip()
    frame = frame.loc[frame["pollutant"].astype(str).str.strip() == source_pollutant].copy()
    for column in ["model_year", "speed_time", "emission_rate"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["modelYear"] = frame["model_year"].astype(int)
    frame["speedMph_timeMin"] = frame["speed_time"]
    frame["rateGram"] = frame["emission_rate"]
    frame["pollutant"] = "BC"
    for column in ["county", "vehicleCategory", "fuel", "process", "pollutant"]:
        frame[column] = frame[column].astype(str).str.strip()
    frame = frame[["county", "vehicleCategory", "fuel", "modelYear", "process", "speedMph_timeMin", "pollutant", "rateGram"]]
    frame = expand_pto_vehicle_category(frame, pto_config, process_column="process")
    return frame.drop_duplicates().reset_index(drop=True)


def build_nh3_inventory_rows(
    emissions_inventory: pd.DataFrame,
    *,
    pto_config: dict[str, object] | None,
) -> pd.DataFrame:
    frame = emissions_inventory.copy()
    frame["county"] = frame["county"].astype(str)
    frame["vehicleCategory"] = frame["vehicleCategory"].astype(str)
    frame["fuel"] = frame["fuel"].astype(str)
    frame["modelYear"] = pd.to_numeric(frame["modelYear"], errors="raise").astype(int)
    frame["speed"] = pd.to_numeric(frame["speed"], errors="raise")
    frame["total_vmt_vehicle_miles_per_year"] = pd.to_numeric(
        frame["total_vmt_vehicle_miles_per_year"], errors="coerce"
    )
    frame["nh3_runex_short_tons_per_year"] = pd.to_numeric(
        frame["nh3_runex_short_tons_per_year"], errors="coerce"
    )

    rates: list[pd.DataFrame] = []

    runex = frame.loc[
        frame["total_vmt_vehicle_miles_per_year"].notna()
        & (frame["total_vmt_vehicle_miles_per_year"] > 0)
        & frame["nh3_runex_short_tons_per_year"].notna()
    ].copy()
    if not runex.empty:
        runex["process"] = "RUNEX"
        runex["pollutant"] = "NH3"
        runex["speedMph_timeMin"] = runex["speed"]
        runex["rateGram"] = (
            runex["nh3_runex_short_tons_per_year"] / runex["total_vmt_vehicle_miles_per_year"]
        ) * GRAMS_PER_SHORT_TON
        rates.append(runex[["county", "vehicleCategory", "fuel", "modelYear", "process", "speedMph_timeMin", "pollutant", "rateGram"]])

    if pto_config and pto_config.get("enabled") and {
        "pto_total_vmt_vehicle_miles_per_year",
        "nh3_pto_short_tons_per_year",
    }.issubset(frame.columns):
        frame["pto_total_vmt_vehicle_miles_per_year"] = pd.to_numeric(
            frame["pto_total_vmt_vehicle_miles_per_year"], errors="coerce"
        )
        frame["nh3_pto_short_tons_per_year"] = pd.to_numeric(
            frame["nh3_pto_short_tons_per_year"], errors="coerce"
        )
        pto = frame.loc[
            frame["pto_total_vmt_vehicle_miles_per_year"].notna()
            & (frame["pto_total_vmt_vehicle_miles_per_year"] > 0)
            & frame["nh3_pto_short_tons_per_year"].notna()
        ].copy()
        if not pto.empty:
            pto["process"] = PTO_PROCESS_NAME
            pto["pollutant"] = "NH3"
            pto["speedMph_timeMin"] = pto["speed"]
            pto["rateGram"] = (
                pto["nh3_pto_short_tons_per_year"] / pto["pto_total_vmt_vehicle_miles_per_year"]
            ) * GRAMS_PER_SHORT_TON
            rates.append(pto[["county", "vehicleCategory", "fuel", "modelYear", "process", "speedMph_timeMin", "pollutant", "rateGram"]])

    if not rates:
        return pd.DataFrame(columns=["county", "vehicleCategory", "fuel", "modelYear", "process", "speedMph_timeMin", "pollutant", "rateGram"])
    return pd.concat(rates, ignore_index=True).drop_duplicates().reset_index(drop=True)


def run_step1(workflow: dict[str, object]) -> dict[str, object]:
    print("  Step 1. Prepare Emissions And Activities Tables")
    print("    1.1 Clean raw project-analysis and write cleaned long-form source")
    project_analysis_path = clean_emfac_to_parquet(
        input_path=workflow["inputs"]["project_analysis_raw"],
        output_path=workflow["paths"]["project_analysis_source"],
        source_type="project-analysis",
        region_label=workflow["run"]["region_label"],
        year=workflow["run"]["calendar_year"],
    )
    project_analysis = _normalize_project_analysis_activity(pd.read_parquet(project_analysis_path))
    project_analysis = expand_pto_vehicle_category(
        project_analysis,
        workflow["run"].get("pto_as_process"),
        process_column="process",
    )
    project_analysis = _pivot_project_analysis(project_analysis)
    project_analysis.to_parquet(project_analysis_path, index=False)
    print("    1.2 Build study-area and statewide emissions inventories")
    emissions_inventory_path = process_emissions_inventory(
        vmt_input=workflow["inputs"]["vmt_raw"],
        population_input=workflow["inputs"]["population_raw"],
        trips_input=workflow["inputs"]["trips_raw"],
        emission_input=workflow["inputs"]["emission_raw"],
        ghg_input=workflow["inputs"].get("ghg_raw"),
        output_path=workflow["paths"]["emissions_inventory"],
        region_label=workflow["run"]["region_label"],
        year=workflow["run"]["calendar_year"],
        pto_config=workflow["run"].get("pto_as_process"),
    )
    emissions_inventory_frame = pd.read_parquet(emissions_inventory_path)
    write_trace(
        workflow,
        "step1_2_emissions_inventory",
        {
            "output_path": str(emissions_inventory_path),
            "summary": frame_summary(emissions_inventory_frame, name="emissions_inventory"),
        },
    )
    print("    1.3 Build PRDUST rates in project-analysis structure")
    project_analysis_prdust = build_road_dust_rows(
        project_analysis,
        rainy_days_file=workflow["inputs"]["rainy_days_file"],
        silt_loading_file=workflow["inputs"]["silt_loading_file"],
        air_basins=workflow["inputs"].get("air_basins"),
    )
    project_analysis_prdust = _pivot_project_analysis_rates(
        project_analysis_prdust,
        key_columns=[
            "county",
            "vehicleCategory",
            "fuel",
            "modelYear",
            "process",
            "speedMph_timeMin",
            "roadCategory",
        ],
    )
    prdust_support_path = Path(workflow["paths"]["project_analysis_prdust"]).expanduser().resolve()
    project_analysis_prdust.to_parquet(prdust_support_path, index=False)
    print("    1.4 Build BC rates in project-analysis structure")
    project_analysis_bc = build_black_carbon_rows(
        workflow["inputs"]["black_carbon_raw"],
        region_label=workflow["run"]["region_label"],
        source_pollutant=str(workflow["inputs"]["black_carbon_pollutant"]),
        pto_config=workflow["run"].get("pto_as_process"),
    )
    project_analysis_bc = _pivot_project_analysis(project_analysis_bc)
    bc_path = Path(workflow["paths"]["project_analysis_bc"]).expanduser().resolve()
    project_analysis_bc.to_parquet(bc_path, index=False)
    print("    1.5 Build NH3 rates in project-analysis structure")
    project_analysis_nh3_rates = build_nh3_inventory_rows(
        emissions_inventory_frame,
        pto_config=workflow["run"].get("pto_as_process"),
    )
    project_analysis_nh3_rates = _pivot_project_analysis(project_analysis_nh3_rates)
    nh3_rates_path = Path(workflow["paths"]["project_analysis_nh3_rates"]).expanduser().resolve()
    project_analysis_nh3_rates.to_parquet(nh3_rates_path, index=False)
    write_trace(
        workflow,
        "step1_1_project_analysis",
        {
            "output_path": str(project_analysis_path),
            "summary": frame_summary(project_analysis, name="project_analysis_source"),
            "prdust_support_path": str(prdust_support_path),
            "prdust_support_summary": frame_summary(project_analysis_prdust, name="project_analysis_prdust"),
            "bc_path": str(bc_path),
            "bc_summary": frame_summary(project_analysis_bc, name="project_analysis_bc"),
            "nh3_rates_path": str(nh3_rates_path),
            "nh3_rates_summary": frame_summary(project_analysis_nh3_rates, name="project_analysis_nh3_rates"),
        },
    )
    statewide_inventory_path = clean_emfac_to_parquet(
        input_path=workflow["inputs"]["statewide_inventory_raw"],
        output_path=workflow["paths"]["statewide_inventory"],
        source_type="emissions-inventory",
        year=workflow["run"]["calendar_year"],
    )
    write_trace(
        workflow,
        "step1_3_statewide_inventory",
        {
            "output_path": str(statewide_inventory_path),
            "summary": frame_summary(pd.read_parquet(statewide_inventory_path), name="statewide_inventory"),
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


def _detect_header_row(path: Path, source_type: str) -> int:
    required_columns = HEADER_DETECTION_COLUMNS[source_type]
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for index, row in enumerate(reader):
            normalized_row = {_normalize_column_name(value) for value in row if str(value).strip()}
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
    filtered["sub_area"] = filtered["sub_area"].map(_normalize_sub_area)
    return filtered


def _normalize_emissions_inventory_fuel(frame: pd.DataFrame) -> pd.DataFrame:
    unknown = sorted(set(frame["fuel"].dropna().astype(str)) - set(FUEL_LABEL_MAP))
    if unknown:
        raise ValueError(f"Unsupported emissions-inventory fuel values: {unknown}")
    frame = frame.copy()
    frame["fuel"] = frame["fuel"].map(FUEL_LABEL_MAP)
    return frame


def _normalize_source_frame(frame: pd.DataFrame, source_type: str) -> pd.DataFrame:
    if source_type == "emissions-inventory" and "vehicle_category" in frame.columns:
        frame = frame.rename(columns={"vehicle_category": "vehicleCategory"})
    if source_type != "emissions-inventory" and "vehicle_class" in frame.columns:
        frame = frame.rename(columns={"vehicle_class": "vehicleCategory"})
    if "model_year" in frame.columns:
        frame = frame.rename(columns={"model_year": "modelYear"})
    if "sub_area" in frame.columns:
        frame = frame.rename(columns={"sub_area": "county"})
    if "speed_time" in frame.columns:
        frame = frame.rename(columns={"speed_time": "speedMph_timeMin"})
    if "emission_rate" in frame.columns:
        frame = frame.rename(columns={"emission_rate": "rateGram"})
    if "energy_cons" in frame.columns:
        frame = frame.rename(columns={"energy_cons": "energy_consumption"})
    frame = frame.rename(
        columns={
            "pm2_5_runex": "pm25_runex",
            "pm2_5_pmbw": "pm25_pmbw",
        }
    )
    if "vehicleCategory" in frame.columns:
        categories = frame["vehicleCategory"].astype(str).str.strip()
        frame["vehicleCategory"] = categories.map(
            lambda value: EMFAC202X_VEHICLE_CATEGORY_ALIASES.get(value, value)
        )
    return frame


def _normalize_project_analysis_activity(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "speedMph_timeMin" not in result.columns or "process" not in result.columns:
        return result
    result["speedMph_timeMin"] = pd.to_numeric(result["speedMph_timeMin"], errors="coerce")
    return result


def _project_analysis_pollutant_column(pollutant: object) -> str:
    label = pollutant_to_column_name(pollutant)
    if label == "pm2_5_gram":
        return "pm25_gram"
    return label


def _standardize_statewide_inventory_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source_column, target_column in {
        "total_vmt": "total_vmt_vehicle_miles_per_year",
        "cvmt": "cvmt_vehicle_miles_per_year",
        "evmt": "evmt_vehicle_miles_per_year",
        "fuel_consumption": "fuel_consumption_1000_gallons_per_year",
        "energy_consumption": "energy_consumption_kwh_per_year",
    }.items():
        if source_column in result.columns:
            result = result.rename(columns={source_column: target_column})

    emission_suffix_map: dict[str, str] = {}
    for column in result.columns:
        match = re.fullmatch(r"([a-z0-9]+)_([a-z0-9]+)", column)
        if match is None:
            continue
        pollutant, process = match.groups()
        if pollutant in {"total", "fuel", "energy", "vehicle", "model", "speed"}:
            continue
        if column in {"vehicleCategory", "modelYear"}:
            continue
        if column.endswith("_per_year"):
            continue
        unit_suffix = "metric_tons_per_year" if pollutant == "co2e" else "short_tons_per_year"
        emission_suffix_map[column] = f"{pollutant}_{process}_{unit_suffix}"
    if emission_suffix_map:
        result = result.rename(columns=emission_suffix_map)
    return result


def _pivot_project_analysis_rates(frame: pd.DataFrame, *, key_columns: list[str]) -> pd.DataFrame:
    pivot = (
        frame.assign(pollutant_column=frame["pollutant"].map(_project_analysis_pollutant_column))
        .groupby(key_columns + ["pollutant_column"], dropna=False)["rateGram"]
        .mean()
        .unstack("pollutant_column")
        .reset_index()
    )
    pivot.columns.name = None
    return pivot


def _pivot_project_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    return _pivot_project_analysis_rates(frame, key_columns=PROJECT_ANALYSIS_KEY_COLUMNS)


def _select_output_columns(frame: pd.DataFrame, source_type: str) -> pd.DataFrame:
    columns_by_source = {
        "project-analysis": [
            "county",
            "vehicleCategory",
            "fuel",
            "modelYear",
            "process",
            "speedMph_timeMin",
            "pollutant",
            "rateGram",
        ],
        "population-inventory": ["county", "vehicleCategory", "fuel", "modelYear", "population"],
        "trips-inventory": ["county", "vehicleCategory", "fuel", "modelYear", "trips"],
        "vmt-inventory": ["county", "vehicleCategory", "fuel", "modelYear", "speed", "total_vmt", "cvmt", "evmt"],
        "emission-inventory": [
            "county",
            "vehicleCategory",
            "fuel",
            "modelYear",
            "speed",
            "process",
            "pollutant",
            "emission",
            "emission_annualized",
        ],
        "ghg-inventory": [
            "county",
            "vehicleCategory",
            "fuel",
            "modelYear",
            "speed",
            "process",
            "pollutant",
            "emission",
            "emission_annualized",
        ],
        "emissions-inventory": [
            "vehicleCategory",
            "fuel",
            "modelYear",
            "speed",
            "total_vmt",
            "cvmt",
            "evmt",
            "nox_runex",
            "pm25_runex",
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
            "pm25_pmbw",
            "fuel_consumption",
        ],
    }
    requested = columns_by_source[source_type]
    selected = [column for column in requested if column in frame.columns]
    missing_required = [
        column
        for column in requested
        if column not in frame.columns
    ]
    if missing_required:
        raise ValueError(f"{source_type} is missing expected columns after normalization: {missing_required}")
    result = frame[selected].copy()
    if source_type == "emissions-inventory":
        result = _standardize_statewide_inventory_columns(result)
    return result


def _clean_file(path: Path, *, source_type: str, region_label: str | None) -> pd.DataFrame:
    if source_type in HEADER_DETECTION_COLUMNS:
        frame = pd.read_csv(path, skiprows=_detect_header_row(path, source_type))
    else:
        frame = pd.read_csv(path)
    frame = frame.rename(columns={column: _normalize_column_name(column) for column in frame.columns})

    if source_type == "emissions-inventory":
        frame = _normalize_emissions_inventory_fuel(frame)
    else:
        frame = _filter_and_normalize_region_label(frame, region_label)
        if region_label is None:
            frame["sub_area"] = frame["sub_area"].map(_normalize_sub_area)

    frame = _normalize_source_frame(frame, source_type)
    return frame


def clean_emfac_to_parquet(
    *,
    input_path: str,
    output_path: str,
    source_type: str,
    region_label: str | None = None,
    year: int | None = None,
) -> Path:
    supported_source_types = {
        "project-analysis",
        "population-inventory",
        "trips-inventory",
        "vmt-inventory",
        "emission-inventory",
        "ghg-inventory",
        "emissions-inventory",
    }
    if source_type not in supported_source_types:
        raise ValueError(f"Unsupported source type: {source_type}")

    cleaned_frames = [
        _clean_file(path, source_type=source_type, region_label=region_label)
        for path in _iter_input_csvs(input_path)
    ]
    combined = pd.concat(cleaned_frames, ignore_index=True)
    if year is not None:
        combined = combined.loc[combined["calendar_year"] == year].copy()
    combined = _select_output_columns(combined, source_type)

    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.lower() != ".parquet":
        raise ValueError("Output path must end with .parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(destination, index=False)
    return destination


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).copy()


def _normalize_metric_label(value: object) -> str:
    label = str(value).strip().lower()
    label = re.sub(r"[^a-z0-9_]+", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")
    if label == "pm2_5":
        return "pm25"
    return label


def _annualize_daily_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") * EMFAC_DAYS_PER_YEAR


@lru_cache(maxsize=1)
def _load_operation_days_lookup() -> dict[str, float]:
    frame = pd.read_csv(OPERATION_DAYS_CSV)
    _require_columns(frame, {"vehicleCategory", "operation_days_per_year"}, "Vehicle operation days CSV")
    result: dict[str, float] = {}
    for row in frame.itertuples(index=False):
        result[str(row.vehicleCategory).strip()] = float(row.operation_days_per_year)
    return result


def _operation_days_for_vehicle_categories(frame: pd.DataFrame, *, vehicle_category_column: str = "vehicleCategory") -> pd.Series:
    if vehicle_category_column not in frame.columns:
        raise ValueError(f"Expected vehicle category column '{vehicle_category_column}' for operation-day annualization.")
    categories = frame[vehicle_category_column].astype(str).str.strip()
    operation_days = categories.map(_load_operation_days_lookup())
    if operation_days.isna().any():
        missing = sorted(categories.loc[operation_days.isna()].drop_duplicates().tolist())
        raise ValueError(
            "Vehicle operation days CSV is missing vehicle categories: "
            f"{missing[:20]}"
        )
    return pd.to_numeric(operation_days, errors="raise")


def _annualize_daily_values_by_vehicle_category(
    frame: pd.DataFrame,
    *,
    source_column: str,
    vehicle_category_column: str = "vehicleCategory",
    output_unit: str = "same",
) -> pd.Series:
    values = pd.to_numeric(frame[source_column], errors="coerce")
    annualized = values * _operation_days_for_vehicle_categories(
        frame,
        vehicle_category_column=vehicle_category_column,
    )
    if output_unit == "metric_tons_per_year":
        annualized = annualized * METRIC_TONS_PER_SHORT_TON
    return annualized


def _annualized_emission_metric_name(pollutant: object, process: object) -> str:
    pollutant_label = _normalize_metric_label(pollutant)
    process_label = _normalize_metric_label(process)
    if pollutant_label == "co2e":
        unit_suffix = "metric_tons_per_year"
    else:
        unit_suffix = "short_tons_per_year"
    return f"{pollutant_label}_{process_label}_{unit_suffix}"


def _pivot_inventory_measurements(
    frame: pd.DataFrame,
    *,
    value_columns: list[str],
    metric_builder,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["county", "vehicleCategory", "fuel", "modelYear", "speed"])
    keys = ["county", "vehicleCategory", "fuel", "modelYear", "speed"]
    base = frame[keys].drop_duplicates().reset_index(drop=True)
    for value_column in value_columns:
        working = frame[keys + ["process", "pollutant", value_column]].copy()
        working["metric"] = working.apply(metric_builder, axis=1)
        pivot = (
            working.pivot_table(
                index=keys,
                columns="metric",
                values=value_column,
                aggfunc="sum",
                dropna=False,
            )
            .reset_index()
        )
        pivot.columns.name = None
        base = base.merge(pivot, on=keys, how="left", validate="one_to_one")
    _require_unique_keys(base, keys, "Pivoted inventory measurements")
    return base


def _pivot_emission_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    if "emission" not in frame.columns:
        raise ValueError("Emission inventory must include emission.")
    working = frame.copy()
    working["emission_short_tons_per_year"] = _annualize_daily_values_by_vehicle_category(
        working,
        source_column="emission",
    )
    return _pivot_inventory_measurements(
        working,
        value_columns=["emission_short_tons_per_year"],
        metric_builder=lambda row: _annualized_emission_metric_name(row["pollutant"], row["process"]),
    )


def _pivot_ghg_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    def _metric_name(row: pd.Series) -> str:
        pollutant = _normalize_metric_label(row["pollutant"])
        process = _normalize_metric_label(row["process"])
        if pollutant == "fuel":
            return "fuel_consumption_1000_gallons_per_year"
        if pollutant in {"energy", "electricity"}:
            return "energy_consumption_kwh_per_year"
        return _annualized_emission_metric_name(pollutant, process)

    if "emission" not in frame.columns:
        raise ValueError("GHG inventory must include emission.")
    working = frame.copy()
    working["annualized_output_value"] = _annualize_daily_values_by_vehicle_category(
        working,
        source_column="emission",
    )
    co2e_mask = working["pollutant"].astype(str).str.strip().str.lower().eq("co2e")
    if co2e_mask.any():
        working.loc[co2e_mask, "annualized_output_value"] = _annualize_daily_values_by_vehicle_category(
            working.loc[co2e_mask].copy(),
            source_column="emission",
            output_unit="metric_tons_per_year",
        ).to_numpy()
    return _pivot_inventory_measurements(
        working,
        value_columns=["annualized_output_value"],
        metric_builder=_metric_name,
    )


def process_emissions_inventory(
    *,
    vmt_input: str,
    population_input: str,
    trips_input: str,
    emission_input: str,
    ghg_input: str | None,
    output_path: str,
    region_label: str,
    year: int,
    pto_config: dict[str, object] | None,
) -> Path:
    with TemporaryDirectory(prefix="emfac_merge_activity_") as temp_dir:
        temp_root = Path(temp_dir)
        vmt_clean_path = temp_root / "vmt_inventory.parquet"
        population_clean_path = temp_root / "population_inventory.parquet"
        trips_clean_path = temp_root / "trips_inventory.parquet"
        emission_clean_path = temp_root / "emission_inventory.parquet"
        ghg_clean_path = temp_root / "ghg_inventory.parquet"

        clean_emfac_to_parquet(
            input_path=vmt_input,
            output_path=str(vmt_clean_path),
            source_type="vmt-inventory",
            region_label=region_label,
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
        clean_emfac_to_parquet(
            input_path=emission_input,
            output_path=str(emission_clean_path),
            source_type="emission-inventory",
            region_label=region_label,
            year=year,
        )
        if ghg_input:
            clean_emfac_to_parquet(
                input_path=ghg_input,
                output_path=str(ghg_clean_path),
                source_type="ghg-inventory",
                region_label=region_label,
                year=year,
            )
        emissions = _read_parquet(vmt_clean_path)[
            ["county", "vehicleCategory", "fuel", "modelYear", "speed", "total_vmt", "cvmt", "evmt"]
        ]
        emissions["county"] = emissions["county"].astype(str)
        for source_column, output_column in {
            key: value
            for key, value in ANNUAL_ACTIVITY_COLUMN_MAP.items()
            if key in {"total_vmt", "cvmt", "evmt"}
        }.items():
            emissions[output_column] = _annualize_daily_values_by_vehicle_category(
                emissions,
                source_column=source_column,
            )
        emissions = emissions.drop(columns=["total_vmt", "cvmt", "evmt"])
        _require_unique_keys(
            emissions,
            ["county", "vehicleCategory", "fuel", "modelYear", "speed"],
            "VMT inventory",
        )
        population = _read_parquet(population_clean_path)
        _require_unique_keys(
            population,
            ["county", "vehicleCategory", "fuel", "modelYear"],
            "Population inventory",
        )
        trips = _read_parquet(trips_clean_path)
        _require_unique_keys(
            trips,
            ["county", "vehicleCategory", "fuel", "modelYear"],
            "Trips inventory",
        )
        result = emissions.merge(
            population[["county", "vehicleCategory", "fuel", "modelYear", "population"]].rename(
                columns={"population": "population_vehicles"}
            ),
            on=["county", "vehicleCategory", "fuel", "modelYear"],
            how="left",
            validate="many_to_one",
        ).merge(
            trips[["county", "vehicleCategory", "fuel", "modelYear", "trips"]].assign(
                trips_per_year=lambda df: _annualize_daily_values_by_vehicle_category(
                    df,
                    source_column="trips",
                )
            )[["county", "vehicleCategory", "fuel", "modelYear", "trips_per_year"]],
            on=["county", "vehicleCategory", "fuel", "modelYear"],
            how="left",
            validate="many_to_one",
        )
        emission = _read_parquet(emission_clean_path)
        result = result.merge(
            _pivot_emission_inventory(emission),
            on=["county", "vehicleCategory", "fuel", "modelYear", "speed"],
            how="left",
            validate="one_to_one",
        )
        if ghg_clean_path.exists():
            ghg = _read_parquet(ghg_clean_path)
            result = result.merge(
                _pivot_ghg_inventory(ghg),
                on=["county", "vehicleCategory", "fuel", "modelYear", "speed"],
                how="left",
                validate="one_to_one",
            )
        if pto_config and pto_config.get("enabled"):
            pto_activity = result.loc[result["vehicleCategory"].astype(str) == "PTO"].copy()
            if not pto_activity.empty:
                pto_vmt = expand_pto_vehicle_category(
                    pto_activity[["county", "vehicleCategory", "fuel", "modelYear", "speed", "total_vmt_vehicle_miles_per_year"]].rename(
                        columns={"total_vmt_vehicle_miles_per_year": "pto_total_vmt_vehicle_miles_per_year"}
                    ),
                    pto_config,
                    process_column=None,
                )
                pto_vmt = pto_vmt.drop_duplicates(["county", "vehicleCategory", "fuel", "modelYear", "speed"])
                result = result.merge(
                    pto_vmt,
                    on=["county", "vehicleCategory", "fuel", "modelYear", "speed"],
                    how="left",
                    validate="many_to_one",
                )
                if "nh3_runex_short_tons_per_year" in pto_activity.columns:
                    pto_nh3 = expand_pto_vehicle_category(
                        pto_activity[["county", "vehicleCategory", "fuel", "modelYear", "speed", "nh3_runex_short_tons_per_year"]].rename(
                            columns={"nh3_runex_short_tons_per_year": "nh3_pto_short_tons_per_year"}
                        ),
                        pto_config,
                        process_column=None,
                    )
                    pto_nh3 = pto_nh3.drop_duplicates(["county", "vehicleCategory", "fuel", "modelYear", "speed"])
                    result = result.merge(
                        pto_nh3,
                        on=["county", "vehicleCategory", "fuel", "modelYear", "speed"],
                        how="left",
                        validate="many_to_one",
                    )
                result = result.loc[result["vehicleCategory"].astype(str) != "PTO"].copy()
        _require_unique_keys(
            result,
            ["county", "vehicleCategory", "fuel", "modelYear", "speed"],
            "Study-area emissions inventory",
        )
        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.lower() != ".parquet":
            raise ValueError("Output path must end with .parquet")
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(destination, index=False)
        return destination
