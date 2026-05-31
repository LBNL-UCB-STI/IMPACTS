"""Fleet Step 1: build source vehicle types before EMFAC assignment.

Substeps:
1.1 Materialize and write passenger non-car vehicle types from source vehicle types.
1.2 Build passenger car targets from ATLAS households and vehicles.
1.3 Materialize and write passenger car vehicle types from passenger car targets.
1.4 Build freight vehicle-type targets from FRISM carriers and tours.
1.5 Materialize and write freight vehicle types from freight vehicle-type targets.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd

from impacts.config.settings import ATLAS_HOUSEHOLDS_SCHEMA
from impacts.config.settings import ATLAS_PERSONS_SCHEMA
from impacts.config.settings import read_table
from impacts.config.settings import resolve_workflow_path
from impacts.pipeline.emfac.common import read_atlas_vehicles_input
from impacts.pipeline.emfac.common import read_frism_carriers_input


def _read_csv(
    path_like: str,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    schema: dict[str, str] | None = None,
) -> pd.DataFrame:
    return read_table(path_like, columns=columns, schema=schema)


def _require_column(frame: pd.DataFrame, column_name: str, frame_name: str) -> None:
    if column_name not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column '{column_name}'")


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_lower(value: object) -> str:
    return _normalize_text(value).lower()


def _normalize_atlas_fuel_aliases(config: dict[str, Any]) -> dict[str, str]:
    aliases = config.get("atlas", {}).get("fuel_map", {}) or {}
    return {
        str(source_fuel).strip().lower(): str(target_fuel).strip().lower()
        for source_fuel, target_fuel in aliases.items()
        if str(source_fuel).strip() and str(target_fuel).strip()
    }


def _apply_atlas_fuel_aliases(vehicles: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    prepared = vehicles.copy()
    _require_column(prepared, "adopt_fuel", "ATLAS vehicles file")
    aliases = _normalize_atlas_fuel_aliases(config)
    prepared["adopt_fuel"] = prepared["adopt_fuel"].map(_normalize_lower)
    if not aliases:
        prepared["beamFuel"] = prepared["adopt_fuel"]
        return prepared
    prepared["beamFuel"] = prepared["adopt_fuel"].map(lambda token: aliases.get(token, token))
    return prepared


def _normalize_bodytype(value: object) -> str:
    return str(value).strip().lower()


def _build_atlas_vehicle_type_ids(
    bodytypes: pd.Series,
    adopt_fuels: pd.Series,
    modelyear_groups: pd.Series,
) -> pd.Series:
    body = bodytypes.astype(str).str.strip()
    body = body.str[:1].str.upper() + body.str[1:]
    fuel = adopt_fuels.astype(str).str.strip()
    fuel = fuel.str[:1].str.upper() + fuel.str[1:]
    groups = modelyear_groups.astype(str).str.strip()
    return body + fuel + groups


def _step1_tmp_dir(output_dir: str) -> Path:
    return Path(resolve_workflow_path(output_dir)) / "_tmp"


def _normalize_energy_file_path(value: Any) -> str:
    if pd.isna(value):
        return ""
    token = str(value).strip()
    if not token:
        return ""
    return token.replace("\\", "/").lstrip("/")


def _normalize_energy_file_columns(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column_name in ["primaryVehicleEnergyFile", "secondaryVehicleEnergyFile"]:
        if column_name in prepared.columns:
            prepared[column_name] = prepared[column_name].map(_normalize_energy_file_path)
    return prepared


def _attach_adopt_fuel_column(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    _require_column(prepared, "primaryFuelType", "Vehicle types file")
    _require_column(prepared, "secondaryFuelType", "Vehicle types file")
    prepared["adopt_fuel"] = [
        _compose_adopt_fuel(primary_fuel, secondary_fuel)
        for primary_fuel, secondary_fuel in zip(
            prepared["primaryFuelType"],
            prepared["secondaryFuelType"],
        )
    ]
    return prepared


def _write_new_vehicle_types_file(frame: pd.DataFrame, output_dir: str) -> str:
    target = _step1_tmp_dir(output_dir) / "vehicleTypes--passenger-car.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return str(target)


def _build_year_scenario_token(*, year: object, scenario: object) -> str:
    year_token = re.sub(r"[^A-Za-z0-9._-]+", "-", _normalize_text(year)).strip("-")
    scenario_token = re.sub(r"[^A-Za-z0-9._-]+", "-", _normalize_text(scenario)).strip("-")
    if year_token and scenario_token:
        return f"{year_token}-{scenario_token}"
    return year_token or scenario_token


def _passenger_vehicle_types_output_file(output_dir: str, *, year: object, scenario: object) -> str:
    return str(
        Path(resolve_workflow_path(output_dir))
        / f"vehicleTypes--atlas--{_build_year_scenario_token(year=year, scenario=scenario)}--EM.csv"
    )


def _freight_vehicle_types_output_file(output_dir: str, *, year: object, scenario: object) -> str:
    return str(
        Path(resolve_workflow_path(output_dir))
        / f"vehicleTypes--frism--{_build_year_scenario_token(year=year, scenario=scenario)}--EM.csv"
    )


def _write_vehicle_type_crosswalk_file(frame: pd.DataFrame, output_dir: str) -> str:
    target = _step1_tmp_dir(output_dir) / "vehicleTypeCrosswalk--atlas.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return str(target)


def _write_new_freight_vehicle_types_file(frame: pd.DataFrame, output_dir: str) -> str:
    target = _step1_tmp_dir(output_dir) / "vehicleTypes--freight.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return str(target)


def _write_passenger_section_vehicle_types_file(section_name: str, frame: pd.DataFrame, output_dir: str) -> str:
    target = _step1_tmp_dir(output_dir) / f"vehicleTypes--passenger-{section_name}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return str(target)


def _remove_stale_step1_output_files(output_dir: str) -> None:
    root = Path(resolve_workflow_path(output_dir))
    tmp_root = _step1_tmp_dir(output_dir)
    for name in [
        "vehicleTypes--beam--step1-built.csv",
        "vehicle_type_atlas_crosswalk.csv",
        "freightVehicleTypePopulation--step1.csv",
        "vehicleTypes--passenger-car--step1-built.csv",
        "vehicleTypeCrosswalk--passenger-atlas--step1.csv",
        "vehicleTypePopulation--freight--step1.csv",
        "vehicleTypePopulation--freight.csv",
        "vehicleTypes--freight--step1-built.csv",
        "vehicleTypes--passenger-noncar--step1-built.csv",
        "vehicleTypes--passenger-noncar.csv",
        "vehicleTypes--passenger-bus.csv",
        "vehicleTypes--passenger-bike.csv",
        "vehicleTypes--passenger-other.csv",
        "vehicleTypes--passenger-car.csv",
        "vehicleTypes--passenger.csv",
        "vehicleTypeCrosswalk--atlas.csv",
        "vehicleTypes--freight.csv",
    ]:
        for base_dir in (root, tmp_root):
            target = base_dir / name
            if target.exists():
                target.unlink()


def _align_vehicle_types_to_source_schema(
    frame: pd.DataFrame,
    source_columns: list[str],
    *,
    frame_name: str,
) -> pd.DataFrame:
    missing = [column for column in source_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{frame_name} is missing source vehicle-types columns:\n" + "\n".join(missing)
    )
    return frame.loc[:, source_columns].copy()


def _load_atlas_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    atlas_vehicles = read_atlas_vehicles_input(config["atlas"]["vehicles_file"])
    atlas_households = _read_csv(
        config["atlas"]["households_file"],
        columns=["household_id", "income_segment", "income_in_thousands"],
        schema=ATLAS_HOUSEHOLDS_SCHEMA,
    )
    atlas_persons = _read_csv(
        config["atlas"]["persons_file"],
        columns=["household_id"],
        schema=ATLAS_PERSONS_SCHEMA,
    )
    return atlas_vehicles, atlas_households, atlas_persons


def _build_atlas_vehicle_type_targets(
    vehicles: pd.DataFrame,
    *,
    config: dict[str, Any],
) -> pd.DataFrame:
    _require_column(vehicles, "bodytype", "ATLAS vehicles file")
    _require_column(vehicles, "modelyear", "ATLAS vehicles file")
    _require_column(vehicles, "adopt_fuel", "ATLAS vehicles file")

    prepared = vehicles[["bodytype", "modelyear", "adopt_fuel"]].copy()
    prepared["adopt_fuel"] = prepared["adopt_fuel"].map(_normalize_lower)
    prepared["modelyear"] = pd.to_numeric(prepared["modelyear"], errors="coerce")
    if prepared["modelyear"].isna().any():
        raise ValueError("ATLAS vehicles file contains non-numeric modelyear values")
    prepared["atlasVehicleTypeId"] = _build_atlas_vehicle_type_ids(
        prepared["bodytype"],
        prepared["adopt_fuel"],
        prepared["modelyear"],
    )
    prepared["beamFuel"] = _apply_atlas_fuel_aliases(prepared[["adopt_fuel"]], config)["beamFuel"]
    grouped = (
        prepared.groupby(
            ["atlasVehicleTypeId", "bodytype", "modelyear", "adopt_fuel", "beamFuel"],
            dropna=False,
        )
        .agg(
            vehicleCount=("modelyear", "size"),
        )
        .reset_index()
    )
    total = grouped["vehicleCount"].sum()
    prepared = grouped.copy()
    prepared["modelyear"] = pd.to_numeric(prepared["modelyear"], errors="coerce").round().astype("Int64")
    prepared["fleetShare"] = prepared["vehicleCount"] / total if total > 0 else 0.0
    return prepared[
        [
            "atlasVehicleTypeId",
            "bodytype",
            "modelyear",
            "adopt_fuel",
            "beamFuel",
            "vehicleCount",
            "fleetShare",
        ]
    ]


def _build_freight_vehicle_type_population(
    carriers: pd.DataFrame,
    tours: pd.DataFrame,
) -> pd.DataFrame:
    _require_column(carriers, "tourId", "FRISM carriers file")
    _require_column(carriers, "vehicleTypeId", "FRISM carriers file")
    _require_column(tours, "tourId", "FRISM tours file")

    prepared_carriers = carriers[["tourId", "vehicleTypeId"]].copy()
    prepared_carriers["tourId"] = prepared_carriers["tourId"].astype(str)
    prepared_carriers["vehicleTypeId"] = prepared_carriers["vehicleTypeId"].astype(str)

    prepared_tours = tours[["tourId"]].copy()
    prepared_tours["tourId"] = prepared_tours["tourId"].astype(str)

    matched = prepared_carriers.merge(
        prepared_tours.drop_duplicates(),
        on="tourId",
        how="inner",
    )
    if matched.empty:
        raise ValueError("No FRISM carrier rows could be matched to FRISM tours on tourId")

    grouped = (
        matched.groupby("vehicleTypeId", dropna=False)
        .size()
        .reset_index(name="vehicleCount")
        .sort_values(["vehicleCount", "vehicleTypeId"], ascending=[False, True])
        .reset_index(drop=True)
    )
    total = grouped["vehicleCount"].sum()
    grouped["fleetShare"] = grouped["vehicleCount"] / total if total > 0 else 0.0
    return grouped[["vehicleTypeId", "vehicleCount", "fleetShare"]]


def _create_catch_all_income_probability_string(probability: float) -> str:
    return f"income | 0-999999:{float(probability):.6f}"


def _normalize_freight_vehicle_type_id(value: object) -> str:
    normalized = re.sub(r"^ft[-_]+", "", str(value).strip(), flags=re.IGNORECASE)
    tokens = [token for token in re.split(r"[-_]+", normalized) if token]
    if not tokens:
        return ""
    return "".join(token[:1].upper() + token[1:].lower() for token in tokens)


def _build_freight_vehicle_types_from_population(
    freight_vehicle_types: pd.DataFrame,
    freight_vehicle_type_population: pd.DataFrame,
) -> pd.DataFrame:
    _require_column(freight_vehicle_types, "vehicleTypeId", "FRISM freight vehicle types file")
    _require_column(freight_vehicle_type_population, "vehicleTypeId", "Freight vehicle type population")
    _require_column(freight_vehicle_type_population, "fleetShare", "Freight vehicle type population")

    prepared_types = freight_vehicle_types.copy()
    prepared_types["vehicleTypeId"] = prepared_types["vehicleTypeId"].astype(str)

    population = freight_vehicle_type_population[["vehicleTypeId", "fleetShare"]].drop_duplicates()
    population["vehicleTypeId"] = population["vehicleTypeId"].astype(str)
    population["fleetShare"] = pd.to_numeric(population["fleetShare"], errors="coerce").fillna(0.0)

    population_for_join = population.rename(columns={"vehicleTypeId": "sourceVehicleTypeId"})
    prepared_types["sourceVehicleTypeId"] = prepared_types["vehicleTypeId"].astype(str)
    prepared = prepared_types.merge(population_for_join, on="sourceVehicleTypeId", how="inner")
    if prepared.empty:
        raise ValueError("No FRISM freight vehicle types intersect with the freight vehicle-type population")
    source_vehicle_type_columns = freight_vehicle_types.columns.tolist()

    def _sample_string_value(series: pd.Series) -> object:
        non_null = series.dropna()
        if non_null.empty:
            return pd.NA
        return non_null.sample(n=1, random_state=0).iloc[0]

    def _build_default_row(source_rows: pd.DataFrame) -> pd.Series:
        if source_rows.empty:
            return pd.Series({column_name: pd.NA for column_name in source_vehicle_type_columns}, dtype="object")
        default_values: dict[str, object] = {}
        for column_name in source_vehicle_type_columns:
            series = source_rows[column_name]
            non_null = series.dropna()
            if non_null.empty:
                default_values[column_name] = pd.NA
                continue
            numeric = pd.to_numeric(non_null, errors="coerce")
            if numeric.notna().all():
                default_values[column_name] = float(numeric.mean())
                continue
            default_values[column_name] = _sample_string_value(non_null)
        return pd.Series(default_values, dtype="object")

    materialized_rows: list[pd.Series] = []
    for row in prepared.itertuples(index=False):
        matching_source_rows = freight_vehicle_types.copy()
        if "vehicleCategory" in matching_source_rows.columns:
            matching_source_rows = matching_source_rows[
                matching_source_rows["vehicleCategory"].astype(str).str.strip().eq(str(getattr(row, "vehicleCategory", "")).strip())
            ].copy()
        if "primaryFuelType" in matching_source_rows.columns:
            matching_source_rows = matching_source_rows[
                matching_source_rows["primaryFuelType"].astype(str).str.strip().str.lower().eq(
                    _normalize_lower(getattr(row, "primaryFuelType", ""))
                )
            ].copy()
        if "secondaryFuelType" in matching_source_rows.columns:
            matching_source_rows = matching_source_rows[
                matching_source_rows["secondaryFuelType"].fillna("").astype(str).str.strip().str.lower().eq(
                    _normalize_lower(getattr(row, "secondaryFuelType", ""))
                )
            ].copy()
        fallback_source_rows = freight_vehicle_types.copy()
        if "vehicleCategory" in fallback_source_rows.columns:
            fallback_source_rows = fallback_source_rows[
                fallback_source_rows["vehicleCategory"].astype(str).str.strip().eq(str(getattr(row, "vehicleCategory", "")).strip())
            ].copy()
        selected = _build_default_row(
            matching_source_rows if not matching_source_rows.empty else fallback_source_rows
        )
        normalized_vehicle_type_id = _normalize_freight_vehicle_type_id(getattr(row, "sourceVehicleTypeId"))
        selected["sourceVehicleTypeId"] = str(getattr(row, "sourceVehicleTypeId"))
        if "vehicleTypeId" in selected.index:
            selected["vehicleTypeId"] = normalized_vehicle_type_id
        if "vehicleCategory" in selected.index:
            selected["vehicleCategory"] = getattr(row, "vehicleCategory")
        if "primaryFuelType" in selected.index:
            selected["primaryFuelType"] = getattr(row, "primaryFuelType")
        if "secondaryFuelType" in selected.index:
            selected["secondaryFuelType"] = getattr(row, "secondaryFuelType")
        if "sampleProbabilityWithinCategory" in selected.index:
            selected["sampleProbabilityWithinCategory"] = f"{float(row.fleetShare):.6f}"
        if "sampleProbabilityString" in selected.index:
            selected["sampleProbabilityString"] = _create_catch_all_income_probability_string(float(row.fleetShare))
        selected["adopt_fuel"] = _compose_adopt_fuel(
            getattr(row, "primaryFuelType", ""),
            getattr(row, "secondaryFuelType", ""),
        )
        if "primaryVehicleEnergyFile" in selected.index:
            selected["primaryVehicleEnergyFile"] = ""
        if "secondaryVehicleEnergyFile" in selected.index:
            selected["secondaryVehicleEnergyFile"] = ""
        selected_columns = list(
            dict.fromkeys(source_vehicle_type_columns + ["sourceVehicleTypeId", "adopt_fuel"])
        )
        materialized_rows.append(selected[selected_columns])

    built = pd.DataFrame(materialized_rows).reset_index(drop=True)
    duplicate_vehicle_type_ids = built["vehicleTypeId"][built["vehicleTypeId"].duplicated()].drop_duplicates()
    if not duplicate_vehicle_type_ids.empty:
        raise ValueError(
            "Freight vehicleTypeId normalization produced duplicates:\n"
            + "\n".join(duplicate_vehicle_type_ids.astype(str).tolist())
        )
    return built


def _load_atlas_passenger_category_mapping(config: dict[str, Any]) -> pd.DataFrame:
    passenger_mapping = config.get("passenger_mapping", {}) or {}
    vehicle_categories = passenger_mapping.get("body_types", {})
    if not isinstance(vehicle_categories, dict) or not vehicle_categories:
        raise ValueError(
            "Passenger mapping is missing body_types required for passenger Step 1 mapping."
        )

    rows: list[dict[str, str]] = []
    seen_bodytypes: set[str] = set()
    for bodytype in vehicle_categories.keys():
        normalized_bodytype = _normalize_bodytype(bodytype)
        if not normalized_bodytype or normalized_bodytype in seen_bodytypes:
            continue
        seen_bodytypes.add(normalized_bodytype)
        rows.append(
            {
                "body_type": normalized_bodytype,
                "passenger_beam_category": _normalize_bodytype("Car"),
            }
        )
    prepared = pd.DataFrame(rows)
    if prepared.empty:
        raise ValueError(
            "Passenger mapping body_types produced no passenger Step 1 bodytype mappings."
        )
    return prepared.drop_duplicates(subset=["body_type"], keep="first").reset_index(drop=True)


def _compose_adopt_fuel(primary_fuel: object, secondary_fuel: object) -> str:
    primary = _normalize_lower(primary_fuel)
    secondary = _normalize_lower(secondary_fuel)
    if secondary:
        return f"{primary}+{secondary}"
    return primary


def _derive_passenger_beam_fuels_from_beam_fuel(
    beam_fuel: object,
    *,
    valid_beam_fuels: set[str],
) -> tuple[str, str]:
    token = _normalize_lower(beam_fuel)
    parts = [part.strip() for part in token.split("+") if part.strip()]
    if not parts:
        return "", ""
    primary = parts[0]
    secondary = parts[1] if len(parts) > 1 else ""
    invalid = [fuel for fuel in [primary, secondary] if fuel and valid_beam_fuels and fuel not in valid_beam_fuels]
    if invalid:
        raise ValueError(
            "Passenger beamFuel resolves to unsupported BEAM fuel types: "
            + ", ".join(sorted(invalid))
        )
    return primary, secondary


def _split_passenger_vehicle_types(vehicle_types: pd.DataFrame) -> dict[str, pd.DataFrame]:
    _require_column(vehicle_types, "vehicleTypeId", "Passenger vehicle types file")
    _require_column(vehicle_types, "vehicleCategory", "Passenger vehicle types file")

    category = vehicle_types["vehicleCategory"].astype(str)
    vehicle_type_id = vehicle_types["vehicleTypeId"].astype(str)
    bus_mask = category.eq("MediumDutyPassenger") & vehicle_type_id.str.contains("BUS-", case=False, na=False)
    return {
        "car": vehicle_types[category.eq("Car")].copy(),
        "bus": vehicle_types[bus_mask].copy(),
        "bike": vehicle_types[category.eq("Bike")].copy(),
        "other": vehicle_types[~(category.eq("Car") | bus_mask | category.eq("Bike"))].copy(),
    }


def _format_income_bin(min_value: object, max_value: object) -> str:
    if pd.isna(min_value) or pd.isna(max_value):
        return "all"
    minimum = float(pd.to_numeric(pd.Series([min_value]), errors="coerce").iloc[0])
    maximum = float(pd.to_numeric(pd.Series([max_value]), errors="coerce").iloc[0])
    if minimum >= 100.0 or maximum >= 100.0:
        return "100-9999"
    lower = int(round(minimum))
    upper = int(round(maximum))
    return f"{lower}-{upper}"


def _validate_income_bins(income_bins: list[object] | None) -> list[float] | None:
    if income_bins in (None, []):
        return None
    numeric_bins = [float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]) for value in income_bins]
    if any(pd.isna(value) for value in numeric_bins):
        raise ValueError("atlas.income_bins must contain only numeric interval edges")
    if len(numeric_bins) < 2:
        raise ValueError("atlas.income_bins must contain at least two interval edges")
    if any(left >= right for left, right in zip(numeric_bins[:-1], numeric_bins[1:])):
        raise ValueError("atlas.income_bins must be strictly increasing")
    return numeric_bins


def _format_configured_income_bin_labels(income_bins: list[float]) -> list[str]:
    return [f"{int(lower)}-{int(upper)}" for lower, upper in zip(income_bins[:-1], income_bins[1:])]


def _lowest_income_bin_label(income_bins: list[object] | None) -> str:
    normalized_income_bins = _validate_income_bins(income_bins)
    if normalized_income_bins is not None:
        return _format_configured_income_bin_labels(normalized_income_bins)[0]
    return "0-30"


def _build_atlas_income_bin_targets(
    vehicles: pd.DataFrame,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    income_bins: list[object] | None = None,
    model_year_groups: dict[str, list[dict[str, object]]] | None = None,
) -> pd.DataFrame:
    _require_column(vehicles, "household_id", "ATLAS vehicles file")
    _require_column(persons, "household_id", "ATLAS persons file")

    prepared_households = households.reset_index()
    _require_column(prepared_households, "household_id", "ATLAS households file")
    _require_column(prepared_households, "income_segment", "ATLAS households file")
    _require_column(prepared_households, "income_in_thousands", "ATLAS households file")

    prepared_households["household_id"] = pd.to_numeric(prepared_households["household_id"], errors="coerce").astype("Int64")
    prepared_households["income_segment"] = pd.to_numeric(prepared_households["income_segment"], errors="coerce")
    prepared_households["income_in_thousands"] = pd.to_numeric(
        prepared_households["income_in_thousands"],
        errors="coerce",
    )
    valid_person_household_ids = pd.to_numeric(persons["household_id"], errors="coerce").astype("Int64")
    prepared_households = prepared_households[
        prepared_households["household_id"].isin(valid_person_household_ids.dropna().unique())
    ].copy()

    prepared_vehicles = vehicles[["household_id", "bodytype", "modelyear", "adopt_fuel"]].copy()
    prepared_vehicles["household_id"] = pd.to_numeric(prepared_vehicles["household_id"], errors="coerce").astype("Int64")
    if model_year_groups is None:
        raise ValueError("Step 1 income-bin targets require configured model_year_groups")
    prepared_vehicles["atlasVehicleTypeId"] = _build_atlas_vehicle_type_ids(
        prepared_vehicles["bodytype"],
        prepared_vehicles["adopt_fuel"],
        prepared_vehicles["modelyear"],
    )

    merged = prepared_vehicles.merge(
        prepared_households[["household_id", "income_segment", "income_in_thousands"]],
        on="household_id",
        how="left",
    )
    merged = merged.dropna(subset=["income_segment"])
    if merged.empty:
        raise ValueError("No ATLAS vehicles could be matched to ATLAS households with income_segment")

    normalized_income_bins = _validate_income_bins(income_bins)
    if normalized_income_bins is not None:
        merged = merged.dropna(subset=["income_in_thousands"])
        if merged.empty:
            raise ValueError("No ATLAS vehicles could be matched to ATLAS households with income_in_thousands")
        merged["incomeBin"] = pd.cut(
            merged["income_in_thousands"],
            bins=normalized_income_bins,
            labels=_format_configured_income_bin_labels(normalized_income_bins),
            include_lowest=True,
            right=True,
        )
        merged = merged.dropna(subset=["incomeBin"])
        income_counts = (
            merged.groupby(["atlasVehicleTypeId", "incomeBin"], dropna=False)
            .agg(
                incomeVehicleCount=("household_id", "size"),
                representativeIncomeK=("income_in_thousands", "mean"),
            )
            .reset_index()
        )
        representative = (
            income_counts.sort_values(["atlasVehicleTypeId", "incomeVehicleCount", "incomeBin"], ascending=[True, False, True])
            .drop_duplicates(subset=["atlasVehicleTypeId"], keep="first")
            .copy()
        )
        income_totals = (
            representative.groupby("incomeBin", dropna=False)["incomeVehicleCount"]
            .sum()
            .reset_index(name="incomeBinTotalVehicleCount")
        )
        representative = representative.merge(income_totals, on="incomeBin", how="left")
    else:
        income_counts = (
            merged.groupby(["atlasVehicleTypeId", "income_segment"], dropna=False)
            .agg(
                incomeVehicleCount=("household_id", "size"),
                incomeMinK=("income_in_thousands", "min"),
                incomeMaxK=("income_in_thousands", "max"),
                representativeIncomeK=("income_in_thousands", "mean"),
            )
            .reset_index()
        )
        representative = (
            income_counts.sort_values(["atlasVehicleTypeId", "incomeVehicleCount", "income_segment"], ascending=[True, False, True])
            .drop_duplicates(subset=["atlasVehicleTypeId"], keep="first")
            .copy()
        )
        income_totals = (
            representative.groupby("income_segment", dropna=False)["incomeVehicleCount"]
            .sum()
            .reset_index(name="incomeBinTotalVehicleCount")
        )
        representative = representative.merge(income_totals, on="income_segment", how="left")
        representative["incomeBin"] = representative.apply(
            lambda row: _format_income_bin(row["incomeMinK"], row["incomeMaxK"]),
            axis=1,
        )
    representative["incomeBin"] = representative["incomeBin"].astype(str)
    representative["incomeProbability"] = representative["incomeVehicleCount"] / representative["incomeBinTotalVehicleCount"]
    return representative[
        ["atlasVehicleTypeId", "incomeBin", "incomeVehicleCount", "incomeProbability", "representativeIncomeK"]
    ]


def _round_fleet_share(frame: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    _require_column(frame, "fleetShare", "Vehicle type atlas crosswalk")
    prepared = frame.copy()
    prepared["fleetShare"] = pd.to_numeric(prepared["fleetShare"], errors="coerce").fillna(0.0)
    prepared["fleetShare"] = prepared["fleetShare"].round(digits)
    if not prepared.empty:
        remainder = round(1.0 - prepared.iloc[:-1]["fleetShare"].sum(), digits)
        prepared.iloc[-1, prepared.columns.get_loc("fleetShare")] = remainder
    return prepared


def _create_probability_string(income_bin: str, income_probability: float, ridehail_probability: float) -> str:
    return f"ridehail | all:{ridehail_probability:.6f}; income | {income_bin}:{income_probability:.6f}"


def _build_passenger_vehicle_types_from_atlas_targets(
    *,
    config: dict[str, Any],
    source_car_vehicle_types: pd.DataFrame,
    atlas_vehicle_type_targets: pd.DataFrame,
) -> pd.DataFrame:
    for column_name in [
        "atlasVehicleTypeId",
        "bodytype",
        "passenger_bodytype_norm",
        "modelyear",
        "adopt_fuel",
        "beamFuel",
        "fleetShare",
        "incomeBin",
        "incomeProbability",
    ]:
        if column_name == "passenger_bodytype_norm":
            continue
        _require_column(atlas_vehicle_type_targets, column_name, "ATLAS vehicle type targets")

    source_vehicle_type_columns = source_car_vehicle_types.columns.tolist()
    passenger_fuel_types = {
        str(fuel).strip().lower()
        for fuel in config.get("passenger_mapping", {}).get("fuel_types", {}).keys()
        if str(fuel).strip()
    }

    def _sample_string_value(series: pd.Series) -> object:
        non_null = series.dropna()
        if non_null.empty:
            return pd.NA
        sampled = non_null.sample(n=1, random_state=0)
        return sampled.iloc[0]

    def _build_default_row(source_rows: pd.DataFrame) -> pd.Series:
        if source_rows.empty:
            return pd.Series({column_name: pd.NA for column_name in source_vehicle_type_columns}, dtype="object")
        default_values: dict[str, object] = {}
        for column_name in source_vehicle_type_columns:
            series = source_rows[column_name]
            non_null = series.dropna()
            if non_null.empty:
                default_values[column_name] = pd.NA
                continue
            numeric = pd.to_numeric(non_null, errors="coerce")
            if numeric.notna().all():
                default_values[column_name] = float(numeric.mean())
                continue
            default_values[column_name] = _sample_string_value(non_null)
        return pd.Series(default_values, dtype="object")

    built_rows: list[pd.Series] = []
    for row in atlas_vehicle_type_targets.itertuples(index=False):
        primary_fuel, secondary_fuel = _derive_passenger_beam_fuels_from_beam_fuel(
            getattr(row, "beamFuel", ""),
            valid_beam_fuels=passenger_fuel_types,
        )
        matching_source_rows = source_car_vehicle_types.copy()
        if "vehicleCategory" in matching_source_rows.columns:
            matching_source_rows = matching_source_rows[
                matching_source_rows["vehicleCategory"].astype(str).str.strip().eq("Car")
            ].copy()
        if "primaryFuelType" in matching_source_rows.columns:
            matching_source_rows = matching_source_rows[
                matching_source_rows["primaryFuelType"].astype(str).str.strip().str.lower().eq(primary_fuel)
            ].copy()
        if "secondaryFuelType" in matching_source_rows.columns:
            matching_source_rows = matching_source_rows[
                matching_source_rows["secondaryFuelType"].fillna("").astype(str).str.strip().str.lower().eq(secondary_fuel)
            ].copy()
        fallback_source_rows = source_car_vehicle_types.copy()
        if "vehicleCategory" in fallback_source_rows.columns:
            fallback_source_rows = fallback_source_rows[
                fallback_source_rows["vehicleCategory"].astype(str).str.strip().eq("Car")
            ].copy()
        selected = _build_default_row(
            matching_source_rows if not matching_source_rows.empty else fallback_source_rows
        )
        selected["atlasVehicleTypeId"] = str(row.atlasVehicleTypeId)
        if "vehicleTypeId" in selected.index:
            selected["vehicleTypeId"] = str(row.atlasVehicleTypeId)
        if "vehicleCategory" in selected.index:
            selected["vehicleCategory"] = "Car"
        if "sampleProbabilityWithinCategory" in selected.index:
            selected["sampleProbabilityWithinCategory"] = f"{float(row.fleetShare):.6f}"
        if "sampleProbabilityString" in selected.index:
            selected["sampleProbabilityString"] = _create_probability_string(
                str(row.incomeBin),
                float(row.incomeProbability),
                float(row.fleetShare),
            )
        selected["adopt_fuel"] = str(getattr(row, "beamFuel", row.adopt_fuel))
        selected["bodytype"] = str(row.bodytype)
        selected["passenger_bodytype_norm"] = str(
            getattr(row, "passenger_bodytype_norm", "") or getattr(row, "bodytype", "")
        )
        selected["modelyear"] = row.modelyear
        selected["beamFuel"] = str(getattr(row, "beamFuel", ""))
        if "primaryFuelType" in selected.index:
            selected["primaryFuelType"] = primary_fuel
        if "secondaryFuelType" in selected.index:
            selected["secondaryFuelType"] = secondary_fuel
        if "primaryVehicleEnergyFile" in selected.index:
            selected["primaryVehicleEnergyFile"] = ""
        if "secondaryVehicleEnergyFile" in selected.index:
            selected["secondaryVehicleEnergyFile"] = ""
        selected_columns = list(
            dict.fromkeys(
                source_vehicle_type_columns
                + ["atlasVehicleTypeId", "adopt_fuel", "beamFuel", "bodytype", "passenger_bodytype_norm", "modelyear"]
            )
        )
        built_rows.append(selected[selected_columns])

    built = pd.DataFrame(built_rows).reset_index(drop=True)
    if built["vehicleTypeId"].duplicated().any():
        duplicate_values = built.loc[built["vehicleTypeId"].duplicated(), "vehicleTypeId"].drop_duplicates().tolist()
        raise ValueError(
            "Duplicate ATLAS passenger vehicleTypeId values were generated in Step 1:\n"
            + "\n".join(duplicate_values)
        )
    return built


def _build_passenger_car_vehicle_types(
    *,
    config: dict[str, Any],
    source_car_vehicle_types: pd.DataFrame,
) -> dict[str, Any]:
    print("=== Step 1.2: build passenger car targets from ATLAS households and vehicles ===")
    atlas_passenger_category_mapping = _load_atlas_passenger_category_mapping(config)
    atlas_vehicles, atlas_households, atlas_persons = _load_atlas_inputs(config)
    model_year_groups = config["activities"]["model_year_groups"]
    atlas_vehicle_type_targets = _build_atlas_vehicle_type_targets(
        atlas_vehicles,
        config=config,
    )
    atlas_income_bin_targets = _build_atlas_income_bin_targets(
        atlas_vehicles,
        atlas_households,
        atlas_persons,
        config["atlas"].get("income_bins"),
        model_year_groups=model_year_groups,
    )
    atlas_vehicle_type_targets = atlas_vehicle_type_targets.merge(
        atlas_income_bin_targets,
        on="atlasVehicleTypeId",
        how="left",
    )
    atlas_vehicle_type_targets["incomeBin"] = atlas_vehicle_type_targets["incomeBin"].fillna(
        _lowest_income_bin_label(config["atlas"].get("income_bins"))
    )
    atlas_vehicle_type_targets["incomeProbability"] = pd.to_numeric(
        atlas_vehicle_type_targets["incomeProbability"],
        errors="coerce",
    ).fillna(0.0)
    atlas_vehicle_type_targets["bodytype_norm"] = atlas_vehicle_type_targets["bodytype"].apply(_normalize_bodytype)
    atlas_vehicle_type_targets = atlas_vehicle_type_targets.merge(
        atlas_passenger_category_mapping.rename(
            columns={
                "body_type": "bodytype_norm",
                "passenger_beam_category": "passenger_bodytype_norm",
            }
        ),
        on="bodytype_norm",
        how="left",
    )
    missing_bodytype_mapping = atlas_vehicle_type_targets[
        atlas_vehicle_type_targets["passenger_bodytype_norm"].isna()
    ]["bodytype"].drop_duplicates()
    if not missing_bodytype_mapping.empty:
        raise ValueError(
            "ATLAS body types are missing passenger BEAM category mappings:\n"
            + "\n".join(missing_bodytype_mapping.astype(str).tolist())
        )

    print("=== Step 1.3: materialize and write passenger car vehicle types from passenger car targets ===")
    built_vehicle_types = _build_passenger_vehicle_types_from_atlas_targets(
        config=config,
        source_car_vehicle_types=source_car_vehicle_types,
        atlas_vehicle_type_targets=atlas_vehicle_type_targets,
    )
    built_vehicle_types_file = _write_new_vehicle_types_file(built_vehicle_types, config["output"])
    vehicle_type_atlas_crosswalk = atlas_vehicle_type_targets[
        [
            "atlasVehicleTypeId",
            "bodytype",
            "modelyear",
            "adopt_fuel",
            "beamFuel",
            "fleetShare",
            "incomeBin",
            "incomeProbability",
        ]
    ].rename(
        columns={
            "atlasVehicleTypeId": "vehicleTypeId",
            "adopt_fuel": "sourceAdoptFuel",
            "beamFuel": "adopt_fuel",
        }
    ).copy()
    vehicle_type_atlas_crosswalk_file = _write_vehicle_type_crosswalk_file(
        vehicle_type_atlas_crosswalk,
        config["output"],
    )

    return {
        "source_atlas_vehicles": atlas_vehicles,
        "source_atlas_households": atlas_households,
        "source_atlas_persons": atlas_persons,
        "built_passenger_car_vehicle_types": built_vehicle_types,
        "built_passenger_car_vehicle_types_file": built_vehicle_types_file,
        "atlas_vehicle_type_targets": atlas_vehicle_type_targets,
        "atlas_income_bin_targets": atlas_income_bin_targets,
        "vehicle_type_atlas_crosswalk": vehicle_type_atlas_crosswalk,
        "vehicle_type_atlas_crosswalk_file": vehicle_type_atlas_crosswalk_file,
    }


def _build_passenger_noncar_vehicle_types(
    *,
    config: dict[str, Any],
    passenger_vehicle_type_sections: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    source_vehicle_columns = (
        passenger_vehicle_type_sections["car"].columns.tolist()
        if not passenger_vehicle_type_sections["car"].empty
        else passenger_vehicle_type_sections["bus"].columns.tolist()
    )
    built_passenger_bus_vehicle_types = _align_vehicle_types_to_source_schema(
        passenger_vehicle_type_sections["bus"],
        source_vehicle_columns,
        frame_name="Built passenger bus vehicle types",
    )
    built_passenger_bus_vehicle_types = _normalize_energy_file_columns(built_passenger_bus_vehicle_types)
    built_passenger_bus_vehicle_types = _attach_adopt_fuel_column(built_passenger_bus_vehicle_types)
    built_passenger_bike_vehicle_types = _align_vehicle_types_to_source_schema(
        passenger_vehicle_type_sections["bike"],
        source_vehicle_columns,
        frame_name="Built passenger bike vehicle types",
    )
    built_passenger_bike_vehicle_types = _normalize_energy_file_columns(built_passenger_bike_vehicle_types)
    built_passenger_bike_vehicle_types = _attach_adopt_fuel_column(built_passenger_bike_vehicle_types)
    built_passenger_other_vehicle_types = _align_vehicle_types_to_source_schema(
        passenger_vehicle_type_sections["other"],
        source_vehicle_columns,
        frame_name="Built passenger other vehicle types",
    )
    built_passenger_other_vehicle_types = _normalize_energy_file_columns(built_passenger_other_vehicle_types)
    built_passenger_other_vehicle_types = _attach_adopt_fuel_column(built_passenger_other_vehicle_types)
    built_passenger_bus_vehicle_types_file = _write_passenger_section_vehicle_types_file(
        "bus",
        built_passenger_bus_vehicle_types,
        config["output"],
    )
    built_passenger_bike_vehicle_types_file = _write_passenger_section_vehicle_types_file(
        "bike",
        built_passenger_bike_vehicle_types,
        config["output"],
    )
    built_passenger_other_vehicle_types_file = _write_passenger_section_vehicle_types_file(
        "other",
        built_passenger_other_vehicle_types,
        config["output"],
    )

    return {
        "passenger_vehicle_type_sections": passenger_vehicle_type_sections,
        "built_passenger_bus_vehicle_types": built_passenger_bus_vehicle_types,
        "built_passenger_bus_vehicle_types_file": built_passenger_bus_vehicle_types_file,
        "built_passenger_bike_vehicle_types": built_passenger_bike_vehicle_types,
        "built_passenger_bike_vehicle_types_file": built_passenger_bike_vehicle_types_file,
        "built_passenger_other_vehicle_types": built_passenger_other_vehicle_types,
        "built_passenger_other_vehicle_types_file": built_passenger_other_vehicle_types_file,
    }


def _run_step1_passenger_substeps(
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    vehicle_types = _read_csv(
        config["passenger_vehicle_types_file"],
        columns=[
            "vehicleTypeId",
            "curbWeightInKg",
            "seatingCapacity",
            "standingRoomCapacity",
            "lengthInMeter",
            "primaryFuelType",
            "primaryFuelConsumptionInJoulePerMeter",
            "primaryFuelCapacityInJoule",
            "primaryVehicleEnergyFile",
            "secondaryFuelType",
            "secondaryFuelConsumptionInJoulePerMeter",
            "secondaryVehicleEnergyFile",
            "secondaryFuelCapacityInJoule",
            "automationLevel",
            "maxVelocity",
            "passengerCarUnit",
            "rechargeLevel2RateLimitInWatts",
            "rechargeLevel3RateLimitInWatts",
            "vehicleCategory",
            "sampleProbabilityWithinCategory",
            "sampleProbabilityString",
        ],
    )
    _require_column(vehicle_types, "vehicleTypeId", "passenger vehicle types source file")
    print(
        "=== Step 1.1: materialize and write passenger non-car vehicle types from source vehicle types ==="
    )
    passenger_vehicle_type_sections = _split_passenger_vehicle_types(vehicle_types)
    passenger_noncar_outputs = _build_passenger_noncar_vehicle_types(
        config=config,
        passenger_vehicle_type_sections=passenger_vehicle_type_sections,
    )
    passenger_car_outputs = _build_passenger_car_vehicle_types(
        config=config,
        source_car_vehicle_types=passenger_vehicle_type_sections["car"],
    )

    return {
        "passenger_vehicle_type_sections": passenger_vehicle_type_sections,
        **passenger_car_outputs,
        **passenger_noncar_outputs,
    }


def _run_step1_freight_substeps(
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    print("=== Step 1.4: build freight vehicle-type targets from FRISM carriers and tours ===")
    frism_carriers = read_frism_carriers_input(config["frism"]["carriers_file"])
    frism_tours = _read_csv(
        config["frism"]["tours_file"],
        columns=["tourId"],
    )
    freight_vehicle_type_population = _build_freight_vehicle_type_population(
        frism_carriers,
        frism_tours,
    )

    print("=== Step 1.5: materialize and write freight vehicle types from freight vehicle-type targets ===")
    freight_vehicle_types = _read_csv(
        config["freight_vehicle_types_file"],
    )
    freight_vehicle_type_population = _round_fleet_share(freight_vehicle_type_population)
    built_freight_vehicle_types = _build_freight_vehicle_types_from_population(
        freight_vehicle_types,
        freight_vehicle_type_population,
    )
    built_freight_vehicle_types_file = _write_new_freight_vehicle_types_file(
        built_freight_vehicle_types,
        config["output"],
    )

    return {
        "source_frism_carriers": frism_carriers,
        "source_frism_tours": frism_tours,
        "freight_vehicle_type_population": freight_vehicle_type_population,
        "source_freight_vehicle_types": freight_vehicle_types,
        "built_freight_vehicle_types": built_freight_vehicle_types,
        "built_freight_vehicle_types_file": built_freight_vehicle_types_file,
    }


def run_step1(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 1: build BEAM vehicle types for the target ATLAS year."""
    config = workflow["config"]
    _remove_stale_step1_output_files(config["output"])
    workflow.update(_run_step1_passenger_substeps(config=config))
    workflow.update(_run_step1_freight_substeps(config=config))
    return workflow
