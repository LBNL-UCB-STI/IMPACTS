from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.emfac.common import assert_row_count
from impacts.emfac.common import frame_summary
from impacts.emfac.common import write_trace
from impacts.emfac.mappings import load_beam_road_type_mapping

RATE_COLUMN = "rateGram"
SPEED_COLUMN = "speedMps_timeMin"
PROJECT_ANALYSIS_COLUMNS = [
    "county",
    "vehicleCategory",
    "fuel",
    "modelYear",
    "process",
    SPEED_COLUMN,
    "pollutant",
    RATE_COLUMN,
]
ROAD_DUST_EXTRA_COLUMNS = [
    "road_category",
    "carb_road_category",
    "f_class",
    "vehicle_weight_tons",
    "rainy_days",
    "silt_loading",
]
ROAD_DUST_FUEL_MAP = {
    "Diesel": "Dsl",
    "Electricity": "Elec",
    "Gasoline": "Gas",
    "Natural Gas": "NG",
    "Plug-in Hybrid": "Phe",
}
FUEL_CASE_MAP = {
    "diesel": "Diesel",
    "electricity": "Electricity",
    "gasoline": "Gasoline",
    "natural gas": "Natural Gas",
    "naturalgas": "Natural Gas",
    "plug-in hybrid": "Plug-in Hybrid",
    "pluginhybrid": "Plug-in Hybrid",
}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _extract_silt_road_categories(frame: pd.DataFrame) -> list[str]:
    base_columns = {"Air Basin", "County", "Air District"}
    road_categories = [column for column in frame.columns if column not in base_columns]
    if not road_categories:
        raise ValueError("Silt loading file does not contain any road-category columns.")
    return road_categories


def _normalize_fuel_label(value: str) -> str:
    fuel = value.strip()
    return FUEL_CASE_MAP.get(fuel.lower(), fuel)


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
    vehicle_weight_tons: pd.Series,
    annual_days: int = 365,
) -> pd.DataFrame:
    k = 0.0022
    pm25_fraction = 0.0686
    pm10_fraction = 0.4572
    pounds_to_grams = 453.592
    pm10_lb_per_vmt = (
        k
        * (silt_loading.astype(float) ** 0.91)
        * (vehicle_weight_tons.astype(float) ** 1.02)
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


def _resolve_carb_road_category(
    road_type: str,
    *,
    local_road_category: str,
    supported_categories: set[str],
) -> str:
    road_type = road_type.strip()
    if road_type == "Local":
        road_type = local_road_category
    if road_type in supported_categories:
        return road_type
    raise ValueError(f"Unsupported road type '{road_type}' in road type lookup.")


def _load_road_mapping(
    road_type_lookup_file: str | Path | None,
    *,
    local_road_category: str,
    supported_categories: list[str],
) -> list[dict[str, object]]:
    if road_type_lookup_file is None:
        return load_beam_road_type_mapping()

    lookup = pd.read_csv(road_type_lookup_file)
    _require_columns(lookup, {"F_class", "road_type"}, "Road type lookup")
    rows: list[dict[str, object]] = []
    supported_category_set = set(supported_categories)
    for _, row in lookup.iterrows():
        carb_road_category = _resolve_carb_road_category(
            str(row["road_type"]),
            local_road_category=local_road_category,
            supported_categories=supported_category_set,
        )
        rows.append(
            {
                "road_category": f"f_class_{int(row['F_class'])}",
                "carb_road_category": carb_road_category,
                "f_class": int(row["F_class"]),
            }
        )
    return rows


def _road_mapping_frame(
    road_type_lookup_file: str | Path | None,
    *,
    local_road_category: str,
    supported_categories: list[str],
) -> pd.DataFrame:
    if road_type_lookup_file is None:
        return pd.DataFrame(load_beam_road_type_mapping())[["road_category", "carb_road_category", "f_class"]]
    return pd.DataFrame(
        _load_road_mapping(
            road_type_lookup_file,
            local_road_category=local_road_category,
            supported_categories=supported_categories,
        )
    )


def _vehicle_weight_frame(vehicle_weight_file: str | Path | None) -> pd.DataFrame:
    if vehicle_weight_file is None:
        return pd.DataFrame([{"vehicleCategory": None, "fuel": None, "vehicle_weight_tons": 2.4}])

    weights = pd.read_csv(vehicle_weight_file)
    _require_columns(weights, {"vehicle type", "fuel type", "Final weight (US tons)"}, "Vehicle weight file")
    weights = weights.rename(
        columns={
            "vehicle type": "vehicleCategory",
            "fuel type": "fuel",
            "Final weight (US tons)": "vehicle_weight_tons",
        }
    )
    weights["vehicleCategory"] = weights["vehicleCategory"].astype(str).str.strip()
    weights["fuel"] = weights["fuel"].astype(str).map(_normalize_fuel_label)
    weights["vehicle_weight_tons"] = pd.to_numeric(weights["vehicle_weight_tons"], errors="coerce")
    return weights.loc[weights["vehicle_weight_tons"].notna(), ["vehicleCategory", "fuel", "vehicle_weight_tons"]]


def generate_road_dust_rates(
    rainy_days_file: str | Path,
    silt_loading_file: str | Path,
    air_basin_region: list[str] | None = None,
    road_type_lookup_file: str | Path | None = None,
    vehicle_weight_file: str | Path | None = None,
    local_road_category: str = "Local Urban",
) -> pd.DataFrame:
    ordered_columns = [
        "vehicleCategory",
        "fuel",
        "process",
        "pollutant",
        "emission_rate",
        "county",
        "road_category",
        "carb_road_category",
        "f_class",
        "vehicle_weight_tons",
        "rainy_days",
        "silt_loading",
    ]
    silt_filtered, road_categories = _read_silt_loading_table(
        silt_loading_file,
        air_basin_region=air_basin_region,
    )
    county_averages = silt_filtered.groupby("County")[road_categories].mean().reset_index()

    rainy_filtered = _read_county_air_basin_table(
        rainy_days_file,
        required_columns={"County", "Air Basin", "Annual Rainfall Days"},
        air_basin_region=air_basin_region,
        label="Rainy days file",
    )
    rainfall_averages = rainy_filtered.groupby("County")["Annual Rainfall Days"].mean().reset_index()
    county_inputs = pd.merge(county_averages, rainfall_averages, on="County", how="inner")
    road_mappings = _road_mapping_frame(
        road_type_lookup_file,
        local_road_category=local_road_category,
        supported_categories=road_categories,
    )
    vehicle_weights = _vehicle_weight_frame(vehicle_weight_file)

    county_long = county_inputs.melt(
        id_vars=["County", "Annual Rainfall Days"],
        value_vars=road_categories,
        var_name="carb_road_category",
        value_name="silt_loading",
    )
    result = county_long.merge(road_mappings, on="carb_road_category", how="inner")
    result = result.merge(vehicle_weights, how="cross")
    emission_rates = _calculate_road_dust_emissions_series(
        result["silt_loading"],
        result["Annual Rainfall Days"],
        vehicle_weight_tons=result["vehicle_weight_tons"],
    )
    result = pd.concat([result, emission_rates], axis=1)

    if result.empty:
        return pd.DataFrame(columns=ordered_columns)
    base = result.rename(
        columns={"County": "county", "Annual Rainfall Days": "rainy_days", "vehicle_class": "vehicleCategory"}
    )[
        [
            "vehicleCategory",
            "fuel",
            "county",
            "road_category",
            "carb_road_category",
            "f_class",
            "vehicle_weight_tons",
            "rainy_days",
            "silt_loading",
            "pm_rate",
            "pm10_rate",
            "pm2_5_rate",
        ]
    ].copy()
    base["process"] = "PRDUST"
    long = base.melt(
        id_vars=[
            "vehicleCategory",
            "fuel",
            "county",
            "road_category",
            "carb_road_category",
            "f_class",
            "vehicle_weight_tons",
            "rainy_days",
            "silt_loading",
            "process",
        ],
        value_vars=["pm_rate", "pm10_rate", "pm2_5_rate"],
        var_name="rate_column",
        value_name="emission_rate",
    )
    long["pollutant"] = long["rate_column"].map(
        {"pm_rate": "PM", "pm10_rate": "PM10", "pm2_5_rate": "PM2_5"}
    )
    return long.drop(columns=["rate_column"])[ordered_columns].reset_index(drop=True)


def _read_project_analysis(path: str) -> pd.DataFrame:
    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported project-analysis format for {target}. Expected .parquet")
    frame = pd.read_parquet(target)
    missing = [column for column in PROJECT_ANALYSIS_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Project-analysis parquet is missing required columns: {', '.join(missing)}")
    return frame.copy()


def _build_road_dust_rows(
    project_analysis: pd.DataFrame,
    *,
    rainy_days_file: str,
    silt_loading_file: str,
    road_type_lookup_file: str | None,
    vehicle_weight_file: str | None,
    air_basins: list[str] | None,
    local_road_category: str,
) -> pd.DataFrame:
    road_dust = generate_road_dust_rates(
        rainy_days_file=rainy_days_file,
        silt_loading_file=silt_loading_file,
        air_basin_region=air_basins,
        road_type_lookup_file=road_type_lookup_file,
        vehicle_weight_file=vehicle_weight_file,
        local_road_category=local_road_category,
    )
    road_dust["fuel"] = road_dust["fuel"].map(ROAD_DUST_FUEL_MAP).fillna(road_dust["fuel"])

    cohort_keys = ["county", "vehicleCategory", "fuel", "modelYear"]
    cohorts = project_analysis[cohort_keys].drop_duplicates().reset_index(drop=True)
    road_dust = road_dust.rename(columns={"emission_rate": "road_dust_emission_rate"})
    merged = cohorts.merge(road_dust, on=["county", "vehicleCategory", "fuel"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=PROJECT_ANALYSIS_COLUMNS + ROAD_DUST_EXTRA_COLUMNS)

    merged[SPEED_COLUMN] = pd.NA
    merged["process"] = "PRDUST"
    merged[RATE_COLUMN] = merged["road_dust_emission_rate"]
    merged = merged.drop(columns=["road_dust_emission_rate"])
    ordered_columns = PROJECT_ANALYSIS_COLUMNS + ROAD_DUST_EXTRA_COLUMNS
    return merged[ordered_columns].reset_index(drop=True)


def _load_road_dust_inputs(
    *,
    project_analysis_path: str,
) -> pd.DataFrame:
    return _read_project_analysis(project_analysis_path)


def _append_road_dust_rows(
    project_analysis: pd.DataFrame,
    road_dust_rows: pd.DataFrame,
    *,
    drop_existing_prdust: bool,
) -> pd.DataFrame:
    result = project_analysis.copy()
    for column in ROAD_DUST_EXTRA_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    if drop_existing_prdust:
        result = result.loc[result["process"] != "PRDUST"].copy()
    return pd.concat([result, road_dust_rows], ignore_index=True)


def _write_project_analysis(frame: pd.DataFrame, output_path: str) -> str:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported output format for {target}. Expected .parquet")
    frame.to_parquet(target, index=False)
    return str(target)


def merge_road_dust_into_project_analysis(
    *,
    rainy_days_file: str,
    silt_loading_file: str,
    project_analysis_path: str,
    output_path: str,
    road_type_lookup_file: str | None = None,
    vehicle_weight_file: str | None = None,
    air_basins: list[str] | None = None,
    local_road_category: str = "Local Urban",
    drop_existing_prdust: bool = True,
) -> str:
    project_analysis = _load_road_dust_inputs(project_analysis_path=project_analysis_path)
    road_dust_rows = _build_road_dust_rows(
        project_analysis,
        rainy_days_file=rainy_days_file,
        silt_loading_file=silt_loading_file,
        road_type_lookup_file=road_type_lookup_file,
        vehicle_weight_file=vehicle_weight_file,
        air_basins=air_basins,
        local_road_category=local_road_category,
    )
    result = _append_road_dust_rows(project_analysis, road_dust_rows, drop_existing_prdust=drop_existing_prdust)
    return _write_project_analysis(result, output_path)


def run_step4(workflow: dict[str, object]) -> dict[str, object]:
    print("  Step 4. Append PRDUST")
    print("    4.1 Load project-analysis and road-dust inputs")
    project_analysis = _load_road_dust_inputs(
        project_analysis_path=workflow["paths"]["project_analysis_with_nh3_bc"],
    )
    print("    4.2 Build and append PRDUST rows")
    road_dust_rows = _build_road_dust_rows(
        project_analysis,
        rainy_days_file=workflow["inputs"]["rainy_days_file"],
        silt_loading_file=workflow["inputs"]["silt_loading_file"],
        road_type_lookup_file=workflow["inputs"]["road_type_lookup_file"],
        vehicle_weight_file=workflow["inputs"]["vehicle_weight_file"],
        air_basins=workflow["inputs"].get("air_basins"),
        local_road_category=workflow["inputs"].get("local_road_category", "Local Urban"),
    )
    result = _append_road_dust_rows(project_analysis, road_dust_rows, drop_existing_prdust=True)
    non_prdust_input_rows = int((project_analysis["process"] != "PRDUST").sum())
    non_prdust_output_rows = int((result["process"] != "PRDUST").sum())
    assert_row_count(non_prdust_input_rows, non_prdust_output_rows, label="PRDUST append non-PRDUST preservation")
    assert_row_count(len(road_dust_rows), int((result["process"] == "PRDUST").sum()), label="PRDUST row append")
    print("    4.3 Write project-analysis with PRDUST")
    _write_project_analysis(result, workflow["paths"]["project_analysis_with_nh3_bc_prdust"])
    write_trace(
        workflow,
        "step4_append_road_dust",
        {
            "input": frame_summary(project_analysis, name="project_analysis_with_nh3_bc"),
            "road_dust_rows": frame_summary(road_dust_rows, name="road_dust_rows"),
            "result": frame_summary(result, name="project_analysis_with_nh3_bc_prdust"),
        },
    )
    return workflow
