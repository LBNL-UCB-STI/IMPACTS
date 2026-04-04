from __future__ import annotations

"""Generate paved road dust emission rates from rainy-days and silt-loading inputs."""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


DEFAULT_BEAM_TO_CARB_ROAD_MAP = {
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


def calculate_road_dust_emissions(
    silt_loading: float,
    rainy_days: float,
    *,
    vehicle_weight_tons: float = 2.4,
    annual_days: int = 365,
) -> tuple[float, float, float]:
    """Calculate PM2.5, PM10, and total PM paved road dust rates in grams per vehicle mile."""
    k = 0.0022

    pm25_fraction = 0.0686
    pm10_fraction = 0.4572

    pm10_lb_per_vmt = (
        k
        * (silt_loading ** 0.91)
        * (vehicle_weight_tons ** 1.02)
        * (1 - rainy_days / annual_days / 4)
    )
    total_pm_lb_per_vmt = pm10_lb_per_vmt / pm10_fraction
    pm25_lb_per_vmt = total_pm_lb_per_vmt * pm25_fraction

    pounds_to_grams = 453.592
    return (
        pm25_lb_per_vmt * pounds_to_grams,
        pm10_lb_per_vmt * pounds_to_grams,
        total_pm_lb_per_vmt * pounds_to_grams,
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
        return [
            {
                "road_category": road_category,
                "carb_road_category": carb_road_category,
                "f_class": None,
            }
            for road_category, carb_road_category in DEFAULT_BEAM_TO_CARB_ROAD_MAP.items()
        ]

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


def _load_vehicle_weights(vehicle_weight_file: str | Path | None) -> list[dict[str, object]]:
    if vehicle_weight_file is None:
        return [
            {
                "vehicle_class": None,
                "fuel": None,
                "vehicle_weight_tons": 2.4,
            }
        ]

    weights = pd.read_csv(vehicle_weight_file)
    _require_columns(weights, {"vehicle type", "fuel type", "Final weight (US tons)"}, "Vehicle weight file")
    weights = weights.rename(
        columns={
            "vehicle type": "vehicle_class",
            "fuel type": "fuel",
            "Final weight (US tons)": "vehicle_weight_tons",
        }
    )
    weights["vehicle_class"] = weights["vehicle_class"].astype(str).str.strip()
    weights["fuel"] = weights["fuel"].astype(str).map(_normalize_fuel_label)
    weights["vehicle_weight_tons"] = pd.to_numeric(weights["vehicle_weight_tons"], errors="coerce")
    weights = weights.loc[weights["vehicle_weight_tons"].notna()].copy()
    return weights[["vehicle_class", "fuel", "vehicle_weight_tons"]].to_dict(orient="records")


def generate_road_dust_rates(
    rainy_days_file: str | Path,
    silt_loading_file: str | Path,
    air_basin_region: list[str] | None = None,
    road_type_lookup_file: str | Path | None = None,
    vehicle_weight_file: str | Path | None = None,
    local_road_category: str = "Local Urban",
) -> pd.DataFrame:
    """Build county and road-category PRDUST rates from CARB-style inputs."""
    ordered_columns = [
        "vehicle_class",
        "fuel",
        "process",
        "pollutant",
        "emission_rate",
        "sub_area",
        "road_category",
        "carb_road_category",
        "f_class",
        "vehicle_weight_tons",
        "rainy_days",
        "silt_loading",
    ]
    silt_frame = pd.read_csv(silt_loading_file)
    _require_columns(silt_frame, {"County", "Air Basin"}, "Silt loading file")
    road_categories = _extract_silt_road_categories(silt_frame)
    silt_frame["County"] = silt_frame["County"].astype(str).str.strip().str.title()
    silt_frame["Air Basin"] = silt_frame["Air Basin"].astype(str).str.strip()
    if air_basin_region:
        silt_filtered_df = silt_frame[silt_frame["Air Basin"].isin(air_basin_region)].copy()
        if silt_filtered_df.empty:
            raise ValueError(f"No silt loading file rows found for air basins: {air_basin_region}")
    else:
        silt_filtered_df = silt_frame
    county_averages = silt_filtered_df.groupby("County")[road_categories].mean().reset_index()

    rainy_filtered_df = _read_county_air_basin_table(
        rainy_days_file,
        required_columns={"County", "Air Basin", "Annual Rainfall Days"},
        air_basin_region=air_basin_region,
        label="Rainy days file",
    )
    rainfall_averages = rainy_filtered_df.groupby("County")["Annual Rainfall Days"].mean().reset_index()

    merged = pd.merge(county_averages, rainfall_averages, on="County", how="inner")
    road_mappings = _load_road_mapping(
        road_type_lookup_file,
        local_road_category=local_road_category,
        supported_categories=road_categories,
    )
    vehicle_weights = _load_vehicle_weights(vehicle_weight_file)
    rows: list[dict[str, object]] = []

    total_iterations = len(merged) * len(road_mappings) * len(vehicle_weights)
    with tqdm(total=total_iterations, desc="Generating road dust rates") as progress:
        for _, row in merged.iterrows():
            county = row["County"]
            rainy_days = row["Annual Rainfall Days"]
            carb_road_to_silt = {road_type: row[road_type] for road_type in road_categories}
            for road_mapping in road_mappings:
                carb_road_category = str(road_mapping["carb_road_category"])
                silt_loading = carb_road_to_silt[carb_road_category]
                for vehicle_weight in vehicle_weights:
                    pm25, pm10, total_pm = calculate_road_dust_emissions(
                        silt_loading,
                        rainy_days,
                        vehicle_weight_tons=float(vehicle_weight["vehicle_weight_tons"]),
                    )
                    common = {
                        "sub_area": county,
                        "road_category": road_mapping["road_category"],
                        "carb_road_category": carb_road_category,
                        "f_class": road_mapping["f_class"],
                        "vehicle_class": vehicle_weight["vehicle_class"],
                        "fuel": vehicle_weight["fuel"],
                        "vehicle_weight_tons": vehicle_weight["vehicle_weight_tons"],
                        "rainy_days": rainy_days,
                        "silt_loading": silt_loading,
                        "process": "PRDUST",
                    }
                    rows.extend(
                        [
                            {**common, "pollutant": "PM", "emission_rate": total_pm},
                            {**common, "pollutant": "PM10", "emission_rate": pm10},
                            {**common, "pollutant": "PM2_5", "emission_rate": pm25},
                        ]
                    )
                    progress.update(1)

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=ordered_columns)
    return result[ordered_columns]


def _write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lower = target.name.lower()
    if lower.endswith(".parquet"):
        frame.to_parquet(target, index=False)
        return target
    if lower.endswith(".csv.gz"):
        frame.to_csv(target, index=False, compression="gzip")
        return target
    if lower.endswith(".csv"):
        frame.to_csv(target, index=False)
        return target
    raise ValueError(f"Unsupported output format: {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.tools.emfac.generate_road_dust_rates",
        description="Generate PRDUST paved road dust rates from rainy-days and silt-loading inputs.",
    )
    parser.add_argument("--rainy-days-file", required=True)
    parser.add_argument("--silt-loading-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--road-type-lookup-file",
        default=None,
        help="Optional road-type lookup CSV with columns F_class and road_type.",
    )
    parser.add_argument(
        "--vehicle-weight-file",
        default=None,
        help="Optional vehicle-weight CSV with columns vehicle type, fuel type, and Final weight (US tons).",
    )
    parser.add_argument(
        "--air-basin",
        dest="air_basins",
        action="append",
        default=None,
        help="Optional air basin filter. Pass multiple times to keep multiple air basins.",
    )
    parser.add_argument(
        "--local-road-category",
        choices=["Local Urban", "Local Rural"],
        default="Local Urban",
        help="How to resolve 'Local' rows from a road-type lookup file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    road_dust_rates = generate_road_dust_rates(
        rainy_days_file=args.rainy_days_file,
        silt_loading_file=args.silt_loading_file,
        air_basin_region=args.air_basins,
        road_type_lookup_file=args.road_type_lookup_file,
        vehicle_weight_file=args.vehicle_weight_file,
        local_road_category=args.local_road_category,
    )
    output = _write_table(road_dust_rates, args.output)
    print(f"output={output}")
    print(f"rows={len(road_dust_rates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
