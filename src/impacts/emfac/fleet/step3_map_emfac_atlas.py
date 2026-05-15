"""Fleet Step 3: map passenger car vehicle types to EMFAC.

Substeps:
3.1 Build passenger EMFAC candidate surface.
3.2 Map passenger car vehicle types to EMFAC.
3.3 Assign fuel-consumption fields to mapped passenger vehicle types.
3.4 Sample mapped passenger vehicleTypeId values onto ATLAS vehicles.
3.5 Write mapped passenger vehicle types and mapped ATLAS vehicles.
"""

from __future__ import annotations

from decimal import Decimal
from decimal import InvalidOperation
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from impacts.emfac.config import ATLAS_HOUSEHOLDS_SCHEMA
from impacts.emfac.config import build_fuel_consumption_emfac_assignment_catalog
from impacts.emfac.config import EMFAC_ACTIVITY_SCHEMA
from impacts.emfac.config import EMFAC_KEY_SCHEMA
from impacts.emfac.config import read_table
from impacts.emfac.config import resolve_workflow_path
from impacts.emfac.common import attach_emissions_rates_filepaths_from_config
from impacts.emfac.common import attach_idle_time_fraction_from_config
from impacts.emfac.common import assign_model_year_groups
from impacts.emfac.common import build_hashed_vehicle_type_ids
from impacts.emfac.common import model_year_group_id_component
from impacts.emfac.common import normalize_probabilities_to_fixed_precision
from impacts.emfac.common import read_atlas_vehicles_input
from impacts.emfac.fleet.step1_build_vehicle_types import _build_atlas_vehicle_type_ids
from impacts.emfac.fleet.step1_build_vehicle_types import _apply_atlas_fuel_aliases
from impacts.emfac.fleet.step1_build_vehicle_types import _format_configured_income_bin_labels
from impacts.emfac.fleet.step1_build_vehicle_types import _normalize_energy_file_columns
from impacts.emfac.fleet.step1_build_vehicle_types import _passenger_vehicle_types_output_file
from impacts.emfac.fleet.step1_build_vehicle_types import _validate_income_bins


_EMFAC_KEY_COLUMNS = ["vehicleCategory", "fuel", "modelYear"]
_PASSENGER_SOURCE_VEHICLE_TYPES_SCHEMA = {
    "vehicleTypeId": "string",
    "vehicleCategory": "string",
    "bodytype": "string",
    "adopt_fuel": "string",
    "modelyear": "Int64",
    "sampleProbabilityWithinCategory": "string",
    "sampleProbabilityString": "string",
}
_PASSENGER_OPTIONAL_VEHICLE_TYPES_SCHEMA = {
    "curbWeightInKg": "Float64",
    "seatingCapacity": "Int64",
    "standingRoomCapacity": "Int64",
    "lengthInMeter": "Float64",
    "primaryFuelType": "string",
    "primaryFuelConsumptionInJoulePerMeter": "Float64",
    "primaryFuelCapacityInJoule": "Float64",
    "primaryVehicleEnergyFile": "string",
    "secondaryFuelType": "string",
    "secondaryFuelConsumptionInJoulePerMeter": "Float64",
    "secondaryVehicleEnergyFile": "string",
    "secondaryFuelCapacityInJoule": "Float64",
    "automationLevel": "Float64",
    "maxVelocity": "Float64",
    "passengerCarUnit": "string",
    "rechargeLevel2RateLimitInWatts": "Float64",
    "rechargeLevel3RateLimitInWatts": "Float64",
    "emfacModelYearGroup": "string",
    "atlasVehicleTypeId": "string",
    "emfacId": "string",
    "emfacVehicleCategory": "string",
    "emfacFuel": "string",
    "emfacResolvedModelYear": "string",
    "fleetVmtPrior": "Float64",
    "fleetPopulationPrior": "Float64",
    "fuelConsumptionId": "string",
    "msrp_usd": "Float64",
    "template_modelyear": "Int64",
    "mappingVehicleTypeId": "string",
    "emissionsRatesFile": "string",
    "idleTimeFraction": "Float64",
}
def _require_column(frame: pd.DataFrame, column_name: str, frame_name: str) -> None:
    if column_name not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column '{column_name}'")


def _require_non_null_column(frame: pd.DataFrame, column_name: str, frame_name: str) -> None:
    _require_column(frame, column_name, frame_name)
    if frame[column_name].isna().any():
        raise ValueError(f"{frame_name} contains null values in required column '{column_name}'")


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_lower(value: object) -> str:
    return _normalize_text(value).lower()


def _normalize_bodytype(value: object) -> str:
    return str(value).strip().lower()


def _normalize_beam_identifier_text(value: object) -> str:
    text = _normalize_text(value)
    if text == "":
        return ""
    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        return text
    if not decimal_value.is_finite():
        return text
    if decimal_value == decimal_value.to_integral_value():
        return format(decimal_value.quantize(Decimal("1")), "f")
    return text


def _sanitize_emfac_component(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", _normalize_text(value)).strip("_")
    return token.replace("_", "")


def _sanitize_output_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", _normalize_text(value)).strip("-")


def _sanitize_vehicle_type_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", _normalize_text(value))


def _build_year_scenario_token(*, year: object, scenario: object) -> str:
    year_token = _sanitize_output_component(year)
    scenario_token = _sanitize_output_component(scenario)
    if year_token and scenario_token:
        return f"{year_token}-{scenario_token}"
    return year_token or scenario_token


def _build_emfac_id(*, vehicle_category: object, fuel: object, model_year: object) -> str:
    return (
        f"{_sanitize_emfac_component(model_year_group_id_component(model_year))}"
        f"{_sanitize_emfac_component(vehicle_category)}"
        f"{_sanitize_emfac_component(fuel)}"
    )


def _build_valid_emfac_candidates(config: dict[str, Any]) -> pd.DataFrame:
    emfac_config = config["activities"]
    rates = read_table(
        emfac_config["passenger_rates_file"],
        schema=EMFAC_KEY_SCHEMA,
    )[_EMFAC_KEY_COLUMNS].drop_duplicates()
    activity = read_table(
        emfac_config["passenger_activity_file"],
        schema=EMFAC_ACTIVITY_SCHEMA,
    )
    fleet = read_table(
        emfac_config["passenger_fleet_file"],
        schema=EMFAC_KEY_SCHEMA,
    )[_EMFAC_KEY_COLUMNS].drop_duplicates()

    candidates = (
        activity.groupby(_EMFAC_KEY_COLUMNS, dropna=False, as_index=False)[
            ["population_vehicles", "total_vmt_vehicle_miles_per_year"]
        ]
        .max()
        .merge(rates, on=_EMFAC_KEY_COLUMNS, how="inner")
        .merge(fleet, on=_EMFAC_KEY_COLUMNS, how="inner")
        .drop_duplicates()
    )
    if candidates.empty:
        raise ValueError("No valid EMFAC candidates remain after intersecting passenger rates, activity, and fleet inputs")
    total_vmt = pd.to_numeric(
        candidates["total_vmt_vehicle_miles_per_year"], errors="coerce"
    ).fillna(0.0).sum()
    if total_vmt <= 0:
        raise ValueError(
            "Passenger EMFAC candidates have zero total_vmt_vehicle_miles_per_year; cannot derive fleetVmtPrior"
        )
    total_population = pd.to_numeric(candidates["population_vehicles"], errors="coerce").fillna(0.0).sum()
    if total_population <= 0:
        raise ValueError("Passenger EMFAC candidates have zero population_vehicles; cannot derive fleetPopulationPrior")
    candidates["fleetVmtPrior"] = (
        pd.to_numeric(candidates["total_vmt_vehicle_miles_per_year"], errors="coerce").fillna(0.0) / total_vmt
    )
    candidates["fleetPopulationPrior"] = (
        pd.to_numeric(candidates["population_vehicles"], errors="coerce").fillna(0.0) / total_population
    )
    candidates["emfacId"] = candidates.apply(
        lambda row: _build_emfac_id(
            vehicle_category=row["vehicleCategory"],
            fuel=row["fuel"],
            model_year=row["modelYear"],
        ),
        axis=1,
    )
    return candidates


def _build_passenger_mapping_context(config: dict[str, Any]) -> dict[str, Any]:
    emfac_candidates = _build_valid_emfac_candidates(config)
    body_type_mapping = _load_passenger_body_type_mapping(config)
    fuel_mapping = _load_emfac_fuel_mapping(config)
    return {
        "emfac_candidates": emfac_candidates,
        "body_type_mapping": body_type_mapping,
        "fuel_mapping": fuel_mapping,
        "passenger_mapping": config.get("passenger_mapping", {}) or {},
    }


def _load_passenger_body_type_mapping(config: dict[str, Any]) -> pd.DataFrame:
    passenger_mapping = config.get("passenger_mapping", {}) or {}
    vehicle_categories = passenger_mapping.get("body_types", {})
    if not isinstance(vehicle_categories, dict) or not vehicle_categories:
        raise ValueError(
            "Passenger mapping is missing body_types required for passenger vehicle-category support."
        )

    rows: list[dict[str, object]] = []
    for body_type, vehicle_category_list in vehicle_categories.items():
        normalized_body_type = _normalize_bodytype(body_type)
        if normalized_body_type == "":
            continue
        if not isinstance(vehicle_category_list, list):
            raise ValueError(
                "Passenger mapping body_types entries must be lists of EMFAC vehicle-category strings."
            )
        for vehicle_category in vehicle_category_list:
            category = _normalize_text(vehicle_category)
            if category == "":
                continue
            rows.append(
                {
                    "body_type": normalized_body_type,
                    "vehicleCategory": category,
                }
            )

    prepared = pd.DataFrame(rows)
    if prepared.empty:
        raise ValueError(
            "Passenger mapping body_types produced no passenger vehicle-category support rows."
        )
    return prepared.drop_duplicates().reset_index(drop=True)


def _load_emfac_fuel_mapping(config: dict[str, Any]) -> pd.DataFrame:
    passenger_mapping = config.get("passenger_mapping", {}) or {}
    vehicle_categories = passenger_mapping.get("body_types", {})
    fuel_types = passenger_mapping.get("fuel_types", {})
    if not isinstance(vehicle_categories, dict) or not vehicle_categories:
        raise ValueError(
            "Passenger mapping is missing body_types required for passenger fuel mapping."
        )
    if not isinstance(fuel_types, dict) or not fuel_types:
        raise ValueError(
            "Passenger mapping is missing fuel_types required for passenger fuel mapping."
        )

    passenger_categories = sorted(
        {
            _normalize_text(vehicle_category)
            for vehicle_category_list in vehicle_categories.values()
            if isinstance(vehicle_category_list, list)
            for vehicle_category in vehicle_category_list
            if _normalize_text(vehicle_category)
        }
    )
    if not passenger_categories:
        raise ValueError(
            "Passenger mapping body_types produced no passenger EMFAC categories for passenger fuel mapping."
        )

    rows: list[dict[str, str]] = []
    for vehicle_category in passenger_categories:
        for adopt_fuel, emfac_fuels in fuel_types.items():
            normalized_adopt_fuel = _normalize_lower(adopt_fuel)
            if normalized_adopt_fuel == "":
                continue
            if not isinstance(emfac_fuels, list):
                raise ValueError(
                    "Passenger mapping fuel_types entries must be lists of EMFAC fuel strings."
                )
            for emfac_fuel in emfac_fuels:
                fuel = _normalize_text(emfac_fuel)
                if fuel == "":
                    continue
                rows.append(
                    {
                        "emfac_vehicle_category": vehicle_category,
                        "emfac_fuel": fuel,
                        "adopt_fuel": normalized_adopt_fuel,
                    }
                )
    prepared = pd.DataFrame(rows)
    if prepared.empty:
        raise ValueError(
            "Passenger mapping body_types and fuel_types produced no passenger fuel mapping rows."
        )
    return prepared.drop_duplicates().reset_index(drop=True)


def _extract_emfac_bodytype_candidates(
    *,
    bodytype: object,
    body_type_mapping: pd.DataFrame,
) -> list[str]:
    bodytype_key = _normalize_lower(bodytype)
    direct = body_type_mapping[body_type_mapping["body_type"] == bodytype_key]
    return direct["vehicleCategory"].astype(str).drop_duplicates().tolist()


def _extract_emfac_fuel_candidates(
    *,
    adopt_fuel: object,
    fuel_mapping: pd.DataFrame,
) -> list[str]:
    adopt_fuel = _normalize_lower(adopt_fuel)
    base_matches = fuel_mapping[fuel_mapping["adopt_fuel"] == adopt_fuel].copy()
    return base_matches["emfac_fuel"].astype(str).drop_duplicates().tolist()


def _coerce_random_generator(seed: int | np.random.Generator) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(int(seed))


def _attach_vehicle_household_income(
    *,
    vehicles: pd.DataFrame,
    households: pd.DataFrame,
) -> pd.DataFrame:
    _require_column(vehicles, "household_id", "ATLAS vehicles file")
    _require_column(households, "income_in_thousands", "ATLAS households file")
    _require_column(households, "household_id", "ATLAS households file")

    households_prepared = households.copy()
    vehicle_income = vehicles.merge(
        households_prepared[["household_id", "income_in_thousands"]].drop_duplicates(),
        on="household_id",
        how="left",
    )
    vehicle_income["income_in_thousands"] = pd.to_numeric(
        vehicle_income["income_in_thousands"],
        errors="coerce",
    )
    if vehicle_income["income_in_thousands"].isna().any():
        raise ValueError("Step 3.4 requires non-null household income_in_thousands for all sampled ATLAS vehicles")
    return vehicle_income


def _build_passenger_vehicle_sampling_table(
    passenger_car_vehicle_types: pd.DataFrame,
    *,
    require_msrp: bool,
) -> pd.DataFrame:
    _require_column(passenger_car_vehicle_types, "vehicleTypeId", "Passenger car vehicle types file")
    _require_column(passenger_car_vehicle_types, "atlasVehicleTypeId", "Passenger car vehicle types file")
    _require_column(passenger_car_vehicle_types, "fleetVmtPrior", "Passenger car vehicle types file")
    _require_column(passenger_car_vehicle_types, "fleetPopulationPrior", "Passenger car vehicle types file")
    if require_msrp:
        _require_column(passenger_car_vehicle_types, "msrp_usd", "Passenger car vehicle types file")

    selected_columns = ["vehicleTypeId", "atlasVehicleTypeId", "fleetVmtPrior", "fleetPopulationPrior"]
    if "msrp_usd" in passenger_car_vehicle_types.columns:
        selected_columns.append("msrp_usd")
    prepared = passenger_car_vehicle_types[selected_columns].copy()
    prepared["atlasVehicleTypeToken"] = prepared["atlasVehicleTypeId"].astype(str)
    prepared["fleetVmtPrior"] = pd.to_numeric(prepared["fleetVmtPrior"], errors="coerce").fillna(0.0)
    prepared["fleetPopulationPrior"] = pd.to_numeric(prepared["fleetPopulationPrior"], errors="coerce").fillna(0.0)
    if "msrp_usd" in prepared.columns:
        prepared["msrp_usd"] = pd.to_numeric(prepared["msrp_usd"], errors="coerce")
    else:
        prepared["msrp_usd"] = pd.Series(pd.NA, index=prepared.index, dtype="Float64")
    if require_msrp and prepared["msrp_usd"].isna().any():
        raise ValueError("Passenger car vehicle types file is missing msrp_usd required for passenger_bayesian_dag")
    return prepared[
        ["vehicleTypeId", "atlasVehicleTypeToken", "fleetVmtPrior", "fleetPopulationPrior", "msrp_usd"]
    ].copy()


def _load_passenger_bayesian_dag(config: dict[str, Any]) -> dict[str, float]:
    dag = config.get("passenger_bayesian_dag", {}) or {}
    required = [
        "likelihood_floor",
        "fleet_vmt_prior_weight",
        "fleet_population_prior_weight",
        "income_weight",
        "income_enabled",
    ]
    missing = [key for key in required if key not in dag]
    if missing:
        raise ValueError(
            "Passenger Bayesian DAG config is missing required keys for Step 3.4: "
            + ", ".join(sorted(missing))
        )
    result = {
        "likelihood_floor": float(dag["likelihood_floor"]),
        "fleet_vmt_prior_weight": float(dag["fleet_vmt_prior_weight"]),
        "fleet_population_prior_weight": float(dag["fleet_population_prior_weight"]),
        "income_weight": float(dag["income_weight"]),
        "income_enabled": bool(dag["income_enabled"]),
    }
    if result["income_enabled"]:
        for key in ("income_center_ratio", "income_sigma_ratio"):
            if key not in dag:
                raise ValueError(
                    "Passenger Bayesian DAG config is missing required income evidence keys for Step 3.4: "
                    + ", ".join(sorted(key for key in ("income_center_ratio", "income_sigma_ratio") if key not in dag))
                )
        result["income_center_ratio"] = float(dag["income_center_ratio"])
        result["income_sigma_ratio"] = float(dag["income_sigma_ratio"])
    return result


def _load_fuel_consumption_msrp_lookup(config: dict[str, Any]) -> dict[str, float]:
    model_file = str(config.get("vehicle_type_assignment", {}).get("model_file", "")).strip()
    breakdown_path = str(config.get("beam", {}).get("fuel_consumption_catalog", "")).strip()
    if not model_file or not breakdown_path:
        raise ValueError(
            "Passenger fuel-consumption assignment requires vehicle_type_assignment.model_file and beam.fuel_consumption_catalog"
        )
    assignment_catalog = build_fuel_consumption_emfac_assignment_catalog(model_file, breakdown_path)
    assignment_catalog["msrp_usd"] = pd.to_numeric(assignment_catalog["msrp_usd"], errors="coerce")
    grouped = (
        assignment_catalog.dropna(subset=["msrp_usd"])
        .groupby("fastsim_id", dropna=False)["msrp_usd"]
        .mean()
        .reset_index()
    )
    if grouped.empty:
        raise ValueError("Fuel-consumption catalog produced no msrp_usd values for passenger Step 3.3")
    return {
        str(row.fastsim_id): float(row.msrp_usd)
        for row in grouped.itertuples(index=False)
    }


def _load_passenger_fuel_consumption_mapping(config: dict[str, Any]) -> pd.DataFrame:
    assignments = config.get("fuel_consumption_mapping", [])
    if not isinstance(assignments, list) or not assignments:
        raise ValueError(
            "Passenger fuel-consumption assignment requires fleet_assignment.mappings.fuel_consumption rows."
        )

    rows: list[dict[str, str]] = []
    for item in assignments:
        if not isinstance(item, dict):
            continue
        fastsim_id = _normalize_text(item.get("fastsim_id"))
        vehicle_categories = item.get("vehicle_categories", [])
        fuel_types = item.get("fuel_types", [])
        if not fastsim_id:
            continue
        if not isinstance(vehicle_categories, list) or not isinstance(fuel_types, list):
            raise ValueError(
                "Fuel-consumption assignment rows must define list-valued vehicle_categories and fuel_types."
            )
        for vehicle_category in vehicle_categories:
            normalized_vehicle_category = _normalize_text(vehicle_category)
            if normalized_vehicle_category == "":
                continue
            for fuel in fuel_types:
                normalized_fuel = _normalize_text(fuel)
                if normalized_fuel == "":
                    continue
                rows.append(
                    {
                        "fuelConsumptionId": fastsim_id,
                        "emfacVehicleCategory": normalized_vehicle_category,
                        "emfacFuel": normalized_fuel,
                    }
                )
    mapping = pd.DataFrame(rows)
    if mapping.empty:
        raise ValueError(
            "Passenger fuel-consumption assignment produced no fuel-consumption-id/category/fuel rows."
        )
    return mapping.drop_duplicates().reset_index(drop=True)


def _assign_passenger_fuel_consumption_fields(
    *,
    passenger_car_vehicle_types: pd.DataFrame,
    config: dict[str, Any],
    seed: int | np.random.Generator,
) -> pd.DataFrame:
    frame_name = "EMFAC-mapped passenger car vehicle types"
    for column_name in ["emfacId", "emfacVehicleCategory", "emfacFuel", "atlasVehicleTypeId"]:
        _require_non_null_column(passenger_car_vehicle_types, column_name, frame_name)

    mapping = _load_passenger_fuel_consumption_mapping(config)
    msrp_lookup = _load_fuel_consumption_msrp_lookup(config)
    breakdown_path = str(config.get("beam", {}).get("fuel_consumption_catalog", "")).strip()
    model_file = str(config.get("vehicle_type_assignment", {}).get("model_file", "")).strip()
    if not breakdown_path or not model_file:
        raise ValueError(
            "Passenger fuel-consumption assignment requires beam.fuel_consumption_catalog "
            "and vehicle_type_assignment.model_file"
        )
    assignment_catalog = build_fuel_consumption_emfac_assignment_catalog(model_file, breakdown_path)
    random = _coerce_random_generator(seed)
    prepared = passenger_car_vehicle_types.copy()
    prepared = _normalize_energy_file_columns(prepared)

    def _has_passenger_baseline_values(row: Any) -> bool:
        primary_fuel_type = str(getattr(row, "primaryFuelType", "") or "").strip()
        primary_consumption = pd.to_numeric(
            pd.Series([getattr(row, "primaryFuelConsumptionInJoulePerMeter", np.nan)]),
            errors="coerce",
        ).iloc[0]
        primary_capacity = pd.to_numeric(
            pd.Series([getattr(row, "primaryFuelCapacityInJoule", np.nan)]),
            errors="coerce",
        ).iloc[0]
        msrp_usd = pd.to_numeric(pd.Series([getattr(row, "msrp_usd", np.nan)]), errors="coerce").iloc[0]
        return (
            primary_fuel_type != ""
            and pd.notna(primary_consumption)
            and float(primary_consumption) > 0.0
            and pd.notna(primary_capacity)
            and float(primary_capacity) > 0.0
            and pd.notna(msrp_usd)
            and float(msrp_usd) > 0.0
        )

    def _build_unassigned_row(row: Any, *, reason: str) -> pd.Series:
        updated = pd.Series(row._asdict()).copy()
        updated["fuelConsumptionId"] = ""
        for column_name in ["primaryVehicleEnergyFile", "secondaryVehicleEnergyFile"]:
            updated[column_name] = ""
        print(
            "WARNING: Passenger Step 3.3 leaving fuel-consumption template fields empty for "
            f"vehicleTypeId={getattr(row, 'vehicleTypeId', '')}, "
            f"atlasVehicleTypeId={getattr(row, 'atlasVehicleTypeId', '')}, "
            f"emfacVehicleCategory={getattr(row, 'emfacVehicleCategory', '')}, "
            f"emfacFuel={getattr(row, 'emfacFuel', '')}: {reason}"
        )
        return updated

    assigned_rows: list[pd.Series] = []
    for row in prepared.itertuples(index=False):
        matches = mapping[
            mapping["emfacVehicleCategory"].astype(str).eq(str(row.emfacVehicleCategory))
            & mapping["emfacFuel"].astype(str).eq(str(row.emfacFuel))
        ]["fuelConsumptionId"].astype(str).drop_duplicates()
        if matches.empty:
            if _has_passenger_baseline_values(row):
                assigned_rows.append(
                    _build_unassigned_row(
                        row,
                        reason="no fuel-consumption mapping matched this EMFAC class/fuel, but the passenger "
                        "vehicle type already carries baseline fuel consumption, capacity, and msrp values",
                    )
                )
                continue
            raise ValueError(
                "No fuel-consumption mapping matched EMFAC-assigned passenger vehicle type "
                f"vehicleTypeId={getattr(row, 'vehicleTypeId', '')}, "
                f"emfacVehicleCategory={getattr(row, 'emfacVehicleCategory', '')}, "
                f"emfacFuel={getattr(row, 'emfacFuel', '')}"
            )
        fuel_consumption_id = str(random.choice(matches.to_numpy(), size=1)[0])

        assignment_matches = assignment_catalog[
            assignment_catalog["fastsim_id"].astype(str).eq(fuel_consumption_id)
        ][["fastsim_id", "fastsim_relative_path"]].drop_duplicates().copy()
        if assignment_matches.empty:
            raise ValueError(
                "No fuel-consumption catalog row matched passenger fuel-consumption id "
                f"fuelConsumptionId={fuel_consumption_id}, vehicleTypeId={getattr(row, 'vehicleTypeId', '')}"
            )
        selected_path = str(
            assignment_matches.sort_values(["fastsim_relative_path"], kind="mergesort").iloc[0]["fastsim_relative_path"]
        ).strip()
        updated = pd.Series(row._asdict()).copy()
        updated["fuelConsumptionId"] = fuel_consumption_id
        updated["primaryVehicleEnergyFile"] = selected_path
        updated["secondaryVehicleEnergyFile"] = ""
        assigned_rows.append(updated)

    prepared = pd.DataFrame(assigned_rows).reset_index(drop=True)
    mapped_msrp = prepared["fuelConsumptionId"].map(msrp_lookup)
    if "msrp_usd" in prepared.columns:
        existing_msrp = pd.to_numeric(prepared["msrp_usd"], errors="coerce")
    else:
        existing_msrp = pd.Series(pd.NA, index=prepared.index, dtype="Float64")
    prepared["msrp_usd"] = existing_msrp.fillna(mapped_msrp)
    unresolved_msrp = prepared["fuelConsumptionId"].astype(str).str.strip().ne("") & prepared["msrp_usd"].isna()
    if unresolved_msrp.any():
        missing_fuel_consumption_ids = prepared.loc[
            unresolved_msrp, "fuelConsumptionId"
        ].drop_duplicates().tolist()
        raise ValueError(
            "No msrp_usd value was found in the fuel-consumption catalog for passenger fuelConsumptionId values:\n"
            + "\n".join(str(value) for value in missing_fuel_consumption_ids)
        )
    atlas_vehicle_type_id = prepared["atlasVehicleTypeId"].astype(str)
    fuel_consumption_prefix = prepared["fuelConsumptionId"].map(_sanitize_vehicle_type_component).astype(str)
    fuel_consumption_prefix = fuel_consumption_prefix.where(fuel_consumption_prefix != "", "unmapped")
    prepared["vehicleTypeId"] = (
        fuel_consumption_prefix + "--" + prepared["emfacId"].astype(str) + "--" + atlas_vehicle_type_id
    )
    prepared = build_hashed_vehicle_type_ids(
        prepared,
        frame_name="Passenger Step 3 vehicle types",
        prefix="paxcar",
    )
    duplicate_vehicle_type_ids = prepared["vehicleTypeId"][prepared["vehicleTypeId"].duplicated()].drop_duplicates()
    if not duplicate_vehicle_type_ids.empty:
        raise ValueError(
            "Passenger car Step 3 generated duplicate vehicleTypeId values after fuel-consumption assignment:\n"
            + "\n".join(duplicate_vehicle_type_ids.astype(str).tolist())
        )
    return prepared


def _sample_passenger_vehicle_type_ids_for_vehicles(
    *,
    vehicles: pd.DataFrame,
    passenger_car_vehicle_types: pd.DataFrame,
    households: pd.DataFrame,
    config: dict[str, Any],
    seed: int | np.random.Generator,
) -> pd.DataFrame:
    for column_name in ["bodytype", "adopt_fuel", "modelyear"]:
        _require_column(vehicles, column_name, "ATLAS vehicles file")

    prepared = _attach_vehicle_household_income(
        vehicles=vehicles.copy(),
        households=households,
    )
    prepared["atlasVehicleTypeToken"] = _build_atlas_vehicle_type_ids(
        prepared["bodytype"].map(_normalize_text),
        prepared["adopt_fuel"].map(_normalize_text),
        pd.to_numeric(prepared["modelyear"], errors="coerce").round().astype("Int64").astype(str),
    )

    dag = _load_passenger_bayesian_dag(config)
    sampling_table = _build_passenger_vehicle_sampling_table(
        passenger_car_vehicle_types,
        require_msrp=bool(dag["income_enabled"]),
    )

    sampling_groups = {
        group_key: group[["vehicleTypeId", "fleetVmtPrior", "fleetPopulationPrior", "msrp_usd"]].reset_index(drop=True)
        for group_key, group in sampling_table.groupby("atlasVehicleTypeToken", dropna=False, sort=False)
    }
    sampled_vehicle_type_ids = pd.Series(index=prepared.index, dtype="object")
    random = _coerce_random_generator(seed)

    group_columns = ["atlasVehicleTypeToken"]
    for group_key, group in prepared.groupby(group_columns, dropna=False, sort=False):
        candidates = sampling_groups.get(group_key[0])
        if candidates is None or candidates.empty:
            raise ValueError(
                "No passenger car vehicle-type candidates available for atlas vehicle type "
                f"atlasVehicleTypeId={group_key[0]}"
            )
        candidates = candidates.copy()
        vmt_prior = pd.to_numeric(candidates["fleetVmtPrior"], errors="coerce").fillna(0.0).clip(lower=0.0)
        vmt_prior = vmt_prior / float(vmt_prior.sum()) if float(vmt_prior.sum()) > 0.0 else pd.Series(
            np.full(len(candidates), 1.0 / len(candidates)),
            index=candidates.index,
            dtype="float64",
        )
        population_prior = pd.to_numeric(candidates["fleetPopulationPrior"], errors="coerce").fillna(0.0).clip(lower=0.0)
        population_prior = population_prior / float(population_prior.sum()) if float(population_prior.sum()) > 0.0 else pd.Series(
            np.full(len(candidates), 1.0 / len(candidates)),
            index=candidates.index,
            dtype="float64",
        )
        log_vmt_prior = np.log(np.maximum(vmt_prior.to_numpy(dtype="float64"), 1e-12))
        log_population_prior = np.log(np.maximum(population_prior.to_numpy(dtype="float64"), 1e-12))
        base_posterior_log = (
            float(dag["fleet_vmt_prior_weight"]) * log_vmt_prior
            + float(dag["fleet_population_prior_weight"]) * log_population_prior
        )
        if bool(dag["income_enabled"]):
            income_usd = (
                pd.to_numeric(group["income_in_thousands"], errors="coerce").fillna(0.0).clip(lower=1e-3).to_numpy(dtype="float64")
                * 1000.0
            )
            affordability_ratio = (
                pd.to_numeric(candidates["msrp_usd"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")[None, :]
                / income_usd[:, None]
            )
            sigma_ratio = max(float(dag["income_sigma_ratio"]), 1e-9)
            standardized = (affordability_ratio - float(dag["income_center_ratio"])) / sigma_ratio
            income_likelihood = np.exp(-0.5 * np.square(standardized))
            income_likelihood = np.maximum(income_likelihood, float(dag["likelihood_floor"]))
            log_income = np.log(np.maximum(income_likelihood, 1e-12))
            posterior_log = base_posterior_log[None, :] + float(dag["income_weight"]) * log_income
            posterior = np.exp(posterior_log - np.max(posterior_log, axis=1, keepdims=True))
            posterior_sum = posterior.sum(axis=1, keepdims=True)
            probabilities = np.divide(
                posterior,
                posterior_sum,
                out=np.full_like(posterior, 1.0 / len(candidates)),
                where=posterior_sum > 0.0,
            )
            cumulative_probabilities = np.cumsum(probabilities, axis=1)
            draws = random.random(len(group))
            sampled_indexes = (draws[:, None] > cumulative_probabilities).sum(axis=1)
            sampled_vehicle_type_ids.loc[group.index] = candidates["vehicleTypeId"].to_numpy()[sampled_indexes]
            continue
        sampled_vehicle_type_ids.loc[group.index] = random.choice(
            candidates["vehicleTypeId"].to_numpy(),
            size=len(group),
            p=np.exp(base_posterior_log - float(np.max(base_posterior_log)))
            / float(np.exp(base_posterior_log - float(np.max(base_posterior_log))).sum()),
        )

    prepared["vehicleTypeId"] = sampled_vehicle_type_ids.astype(str)
    return prepared.drop(columns=["atlasVehicleTypeToken"])


def _assign_income_bin_labels(
    income_in_thousands: pd.Series,
    *,
    config: dict[str, Any],
) -> pd.Series:
    normalized_income_bins = _validate_income_bins(config.get("atlas", {}).get("income_bins"))
    numeric_income = pd.to_numeric(income_in_thousands, errors="coerce")
    if normalized_income_bins is None:
        return pd.Series("all", index=income_in_thousands.index, dtype="string")
    labels = _format_configured_income_bin_labels(normalized_income_bins)
    return pd.cut(
        numeric_income,
        bins=normalized_income_bins,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype("string").fillna(labels[0])


def _format_probability_entries(entries: list[tuple[str, float]]) -> str:
    return "; ".join(f"income | {income_bin}:{probability:.6f}" for income_bin, probability in entries)


def _finalize_passenger_vehicle_type_probabilities(
    *,
    passenger_car_vehicle_types: pd.DataFrame,
    sampled_vehicles: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    prepared = passenger_car_vehicle_types.copy()
    if prepared.empty:
        return prepared

    category_lookup = prepared[["vehicleTypeId", "vehicleCategory"]].drop_duplicates().copy()
    assigned = sampled_vehicles[["vehicleTypeId"]].copy()
    assigned["vehicleTypeId"] = assigned["vehicleTypeId"].astype(str)
    assigned = assigned.merge(category_lookup, on="vehicleTypeId", how="left")
    counts = (
        assigned.groupby(["vehicleCategory", "vehicleTypeId"], dropna=False)
        .size()
        .reset_index(name="vehicleCount")
    )
    counts = normalize_probabilities_to_fixed_precision(
        counts,
        group_columns=["vehicleCategory"],
        weight_column="vehicleCount",
        output_column="sampleProbabilityWithinCategory",
    )
    probability_lookup = {
        (str(row.vehicleCategory), str(row.vehicleTypeId)): float(row.sampleProbabilityWithinCategory)
        for row in counts.itertuples(index=False)
    }
    prepared["sampleProbabilityWithinCategory"] = prepared.apply(
        lambda row: f"{probability_lookup.get((str(row['vehicleCategory']), str(row['vehicleTypeId'])), 0.0):.6f}",
        axis=1,
    )

    income_assignments = sampled_vehicles[["vehicleTypeId", "income_in_thousands"]].copy()
    income_assignments["vehicleTypeId"] = income_assignments["vehicleTypeId"].astype(str)
    income_assignments = income_assignments.merge(category_lookup, on="vehicleTypeId", how="left")
    income_assignments["incomeBin"] = _assign_income_bin_labels(
        income_assignments["income_in_thousands"],
        config=config,
    )
    income_counts = (
        income_assignments.groupby(["vehicleCategory", "incomeBin", "vehicleTypeId"], dropna=False)
        .size()
        .reset_index(name="vehicleCount")
    )
    income_counts = normalize_probabilities_to_fixed_precision(
        income_counts,
        group_columns=["vehicleCategory", "incomeBin"],
        weight_column="vehicleCount",
        output_column="incomeProbability",
    )
    income_lookup: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for row in income_counts.sort_values(["vehicleCategory", "incomeBin", "vehicleTypeId"], kind="mergesort").itertuples(index=False):
        income_lookup.setdefault((str(row.vehicleCategory), str(row.vehicleTypeId)), []).append(
            (str(row.incomeBin), float(row.incomeProbability))
        )
    prepared["sampleProbabilityString"] = prepared.apply(
        lambda row: _format_probability_entries(
            income_lookup.get((str(row["vehicleCategory"]), str(row["vehicleTypeId"])), [])
        ),
        axis=1,
    )
    return prepared


def _prepare_mapped_passenger_vehicles_output(vehicles: pd.DataFrame) -> pd.DataFrame:
    frame_name = "Mapped passenger vehicles"
    prepared = vehicles.copy()
    for column_name in ("household_id", "vehicle_id", "vehicleTypeId"):
        _require_non_null_column(prepared, column_name, frame_name)

    household_id = prepared["household_id"].astype(str)
    vehicle_id = prepared["vehicle_id"].astype(str)
    vehicle_type_id = prepared["vehicleTypeId"].astype(str)
    household_id_alias = prepared["household_id"].map(_normalize_beam_identifier_text)
    vehicle_id_alias = prepared["vehicle_id"].map(_normalize_beam_identifier_text)
    if household_id.str.strip().eq("").any():
        raise ValueError(f"{frame_name} contains blank values in required column 'household_id'")
    if vehicle_id.str.strip().eq("").any():
        raise ValueError(f"{frame_name} contains blank values in required column 'vehicle_id'")
    if vehicle_type_id.str.strip().eq("").any():
        raise ValueError(f"{frame_name} contains blank values in required column 'vehicleTypeId'")
    if household_id_alias.str.strip().eq("").any():
        raise ValueError(f"{frame_name} produced blank values in required alias column 'householdId'")
    if vehicle_id_alias.str.strip().eq("").any():
        raise ValueError(f"{frame_name} produced blank values in required alias component 'vehicle_id'")

    result = prepared.copy()
    result["household_id"] = household_id
    result["vehicle_id"] = vehicle_id
    result["householdId"] = household_id_alias
    result["vehicleId"] = household_id_alias + "-" + vehicle_id_alias
    result["vehicleTypeId"] = vehicle_type_id
    result["initialSoc"] = pd.Series(pd.NA, index=prepared.index, dtype="Float64")
    if result["vehicleId"].duplicated().any():
        raise ValueError(f"{frame_name} produced duplicate BEAM vehicleId values")
    return result


def _build_passenger_emfac_candidates(
    *,
    vehicle_type_id: str,
    bodytype: object,
    model_year_group: object,
    adopt_fuel: object,
    emfac_candidates: pd.DataFrame,
    body_type_mapping: pd.DataFrame,
    fuel_mapping: pd.DataFrame,
    passenger_mapping: dict[str, Any],
) -> pd.DataFrame:
    vehicle_category_candidates = _extract_emfac_bodytype_candidates(
        bodytype=bodytype,
        body_type_mapping=body_type_mapping,
    )
    if not vehicle_category_candidates:
        raise ValueError(
            f"No EMFAC category candidates available for passenger car vehicleTypeId={vehicle_type_id}, bodytype={bodytype}"
        )

    fuel_candidates = _extract_emfac_fuel_candidates(
        adopt_fuel=adopt_fuel,
        fuel_mapping=fuel_mapping,
    )
    if not fuel_candidates:
        raise ValueError(
            "No EMFAC fuel candidates available for passenger car "
            f"vehicleTypeId={vehicle_type_id}, adopt_fuel={adopt_fuel}"
        )

    matched = emfac_candidates[
        emfac_candidates["vehicleCategory"].isin(vehicle_category_candidates)
        & emfac_candidates["fuel"].isin(fuel_candidates)
    ].copy()
    if matched.empty:
        raise ValueError(
            "No passenger EMFAC candidates available after applying category/fuel mapping for "
            f"vehicleTypeId={vehicle_type_id}, bodytype={bodytype}, adopt_fuel={adopt_fuel}"
        )

    requested_model_year_group = str(model_year_group).strip()
    exact_matched = matched[matched["modelYear"].astype(str).eq(requested_model_year_group)].copy()
    if exact_matched.empty:
        fallback_candidates = []
        for item in passenger_mapping.get("fuel_fallbacks", []):
            if not isinstance(item, dict):
                continue
            if str(item.get("source_fuel", "")).strip().lower() != str(adopt_fuel).strip().lower():
                continue
            if str(item.get("if_model_year", "")).strip() != requested_model_year_group:
                continue
            fallback_candidates.extend(
                [
                    str(value).strip()
                    for value in item.get("fallback_emfac_fuels", [])
                    if str(value).strip()
                ]
            )
        if fallback_candidates:
            fallback_matched = emfac_candidates[
                emfac_candidates["vehicleCategory"].isin(vehicle_category_candidates)
                & emfac_candidates["fuel"].isin(sorted(set(fallback_candidates)))
                & emfac_candidates["modelYear"].astype(str).eq(requested_model_year_group)
            ].copy()
            if not fallback_matched.empty:
                matched = fallback_matched
            else:
                available = [
                    str(value).strip()
                    for value in matched["modelYear"].dropna().astype(str).drop_duplicates().tolist()
                    if str(value).strip()
                ]
                raise ValueError(
                    "Passenger fuel fallback matched but no EMFAC candidates remained for "
                    f"vehicleTypeId={vehicle_type_id}, adopt_fuel={adopt_fuel}, "
                    f"modelYearGroup={requested_model_year_group}, fallback_emfac_fuels={sorted(set(fallback_candidates))}, "
                    f"available={available}"
                )
        else:
            available = [
                str(value).strip()
                for value in matched["modelYear"].dropna().astype(str).drop_duplicates().tolist()
                if str(value).strip()
            ]
            raise ValueError(
                "No passenger EMFAC candidates matched the configured modelYear group for "
                f"vehicleTypeId={vehicle_type_id}, modelYearGroup={requested_model_year_group}, available={available}"
            )
    else:
        matched = exact_matched
    matched["fleetVmtPrior"] = pd.to_numeric(matched["fleetVmtPrior"], errors="coerce").fillna(0.0)
    matched["fleetPopulationPrior"] = pd.to_numeric(matched["fleetPopulationPrior"], errors="coerce").fillna(0.0)
    vmt_total = float(matched["fleetVmtPrior"].sum())
    population_total = float(matched["fleetPopulationPrior"].sum())
    if vmt_total <= 0.0:
        raise ValueError(
            "Passenger EMFAC candidates have zero fleetVmtPrior after mapping for "
            f"vehicleTypeId={vehicle_type_id}"
        )
    if population_total <= 0.0:
        raise ValueError(
            "Passenger EMFAC candidates have zero fleetPopulationPrior after mapping for "
            f"vehicleTypeId={vehicle_type_id}"
        )
    matched["fleetVmtPrior"] = matched["fleetVmtPrior"] / vmt_total
    matched["fleetPopulationPrior"] = matched["fleetPopulationPrior"] / population_total
    matched = matched.sort_values(
        by=[
            "total_vmt_vehicle_miles_per_year",
            "population_vehicles",
            "vehicleCategory",
            "fuel",
            "modelYear",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return matched


def _build_passenger_car_emfac_mapping(
    *,
    passenger_car_vehicle_types: pd.DataFrame,
    mapping_context: dict[str, Any],
    model_year_groups: dict[str, list[dict[str, object] | str]],
) -> pd.DataFrame:
    for column_name in [
        "vehicleTypeId",
        "vehicleCategory",
        "adopt_fuel",
        "sampleProbabilityWithinCategory",
        "sampleProbabilityString",
    ]:
        _require_column(passenger_car_vehicle_types, column_name, "Passenger car vehicle types file")
    for column_name in ["bodytype", "modelyear"]:
        _require_column(passenger_car_vehicle_types, column_name, "Passenger car vehicle types file")

    emfac_candidates = mapping_context["emfac_candidates"]
    body_type_mapping = mapping_context["body_type_mapping"]
    fuel_mapping = mapping_context["fuel_mapping"]
    passenger_mapping = mapping_context["passenger_mapping"]
    prepared = passenger_car_vehicle_types.copy()
    prepared["emfacGroupingVehicleCategory"] = "LDA"
    prepared = assign_model_year_groups(
        prepared,
        model_year_groups,
        year_column="modelyear",
        category_column="emfacGroupingVehicleCategory",
        output_column="emfacModelYearGroup",
    )
    prepared = prepared.drop(columns=["emfacGroupingVehicleCategory"])

    expanded_rows: list[dict[str, Any]] = []
    for row in prepared.itertuples(index=False):
        row_payload = row._asdict()
        atlas_vehicle_type_id = str(getattr(row, "atlasVehicleTypeId", getattr(row, "vehicleTypeId")))
        candidates = _build_passenger_emfac_candidates(
            vehicle_type_id=str(row.vehicleTypeId),
            bodytype=row.bodytype,
            model_year_group=row.emfacModelYearGroup,
            adopt_fuel=row.adopt_fuel,
            emfac_candidates=emfac_candidates,
            body_type_mapping=body_type_mapping,
            fuel_mapping=fuel_mapping,
            passenger_mapping=passenger_mapping,
        )
        for candidate in candidates.itertuples(index=False):
            updated = dict(row_payload)
            updated["atlasVehicleTypeId"] = atlas_vehicle_type_id
            updated["emfacId"] = str(candidate.emfacId)
            updated["emfacVehicleCategory"] = str(candidate.vehicleCategory)
            updated["emfacFuel"] = str(candidate.fuel)
            updated["emfacResolvedModelYear"] = str(candidate.modelYear)
            updated["fleetVmtPrior"] = f"{float(candidate.fleetVmtPrior):.6f}"
            updated["fleetPopulationPrior"] = f"{float(candidate.fleetPopulationPrior):.6f}"
            updated["vehicleTypeId"] = f"{candidate.emfacId}--{atlas_vehicle_type_id}"
            updated["sampleProbabilityWithinCategory"] = "0.000000"
            updated["sampleProbabilityString"] = ""
            expanded_rows.append(updated)

    prepared = pd.DataFrame(expanded_rows)
    duplicate_vehicle_type_ids = prepared["vehicleTypeId"][prepared["vehicleTypeId"].duplicated()].drop_duplicates()
    if not duplicate_vehicle_type_ids.empty:
        raise ValueError(
            "Passenger car Step 3 generated duplicate vehicleTypeId values:\n"
            + "\n".join(duplicate_vehicle_type_ids.astype(str).tolist())
        )
    return prepared


def _write_vehicle_types(frame: pd.DataFrame, path_like: str) -> str:
    output_path = Path(resolve_workflow_path(path_like))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return str(output_path)


def _read_step3_passenger_vehicle_types(path_like: str) -> pd.DataFrame:
    resolved = Path(resolve_workflow_path(path_like))
    if resolved.suffix.lower() == ".parquet":
        available_columns = list(pd.read_parquet(resolved).columns)
    else:
        available_columns = pd.read_csv(resolved, nrows=0).columns.tolist()
    missing_required = [
        column_name for column_name in _PASSENGER_SOURCE_VEHICLE_TYPES_SCHEMA if column_name not in available_columns
    ]
    if missing_required:
        raise ValueError(
            "Passenger car vehicle types file is missing required columns: "
            + ", ".join(sorted(missing_required))
        )
    schema = {
        column_name: dtype_name
        for column_name, dtype_name in _PASSENGER_SOURCE_VEHICLE_TYPES_SCHEMA.items()
        if column_name in available_columns
    }
    for column_name in available_columns:
        if column_name in schema:
            continue
        schema[column_name] = _PASSENGER_OPTIONAL_VEHICLE_TYPES_SCHEMA.get(column_name, "string")
    return read_table(str(path_like), schema=schema)


def _combine_passenger_vehicle_types_for_output(
    *,
    passenger_car_with_emfac: pd.DataFrame,
    passenger_bus_with_emfac: pd.DataFrame,
    passenger_bike_with_emfac: pd.DataFrame,
    passenger_other_with_emfac: pd.DataFrame,
) -> pd.DataFrame:
    sections = [
        passenger_car_with_emfac.copy(),
        passenger_bus_with_emfac.copy(),
        passenger_bike_with_emfac.copy(),
        passenger_other_with_emfac.copy(),
    ]
    ordered_columns: list[str] = []
    for section in sections:
        for column_name in section.columns.tolist():
            if column_name not in ordered_columns:
                ordered_columns.append(column_name)
    if "emfacId" not in ordered_columns:
        ordered_columns.append("emfacId")
    if "emissionsRatesFile" not in ordered_columns:
        ordered_columns.append("emissionsRatesFile")
    if "idleTimeFraction" not in ordered_columns:
        ordered_columns.append("idleTimeFraction")

    passenger_other_prepared = passenger_other_with_emfac.copy()
    for column_name in [
        "emfacId",
        "emfacVehicleCategory",
        "emfacFuel",
        "emfacResolvedModelYear",
        "emissionsRatesFile",
        "idleTimeFraction",
    ]:
        passenger_other_prepared[column_name] = ""
    sections[-1] = passenger_other_prepared

    aligned_sections = [
        section.reindex(columns=ordered_columns, fill_value="")
        for section in sections
    ]
    return pd.concat(aligned_sections, ignore_index=True)


def _write_parquet(frame: pd.DataFrame, path_like: str) -> str:
    output_path = Path(resolve_workflow_path(path_like))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return str(output_path)


def _map_passenger_car_vehicle_types(workflow: dict[str, Any], *, mapping_context: dict[str, Any]) -> pd.DataFrame:
    passenger_car_file = workflow.get("built_passenger_car_vehicle_types_file")
    if not passenger_car_file:
        raise ValueError("Step 3 requires passenger car vehicle types from Step 1")
    passenger_car_vehicle_types = _read_step3_passenger_vehicle_types(str(passenger_car_file))
    return _build_passenger_car_emfac_mapping(
        passenger_car_vehicle_types=passenger_car_vehicle_types,
        mapping_context=mapping_context,
        model_year_groups=workflow["config"]["activities"]["model_year_groups"],
    )


def _sample_mapped_passenger_vehicles(
    *,
    workflow: dict[str, Any],
    passenger_car_with_emfac: pd.DataFrame,
    seed: int | np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = workflow["config"]
    atlas_config = config["atlas"]
    atlas_vehicles = workflow.get("source_atlas_vehicles")
    if atlas_vehicles is None:
        atlas_vehicles = read_atlas_vehicles_input(atlas_config["vehicles_file"])
        workflow["source_atlas_vehicles"] = atlas_vehicles
    atlas_vehicles = _apply_atlas_fuel_aliases(atlas_vehicles, config)
    atlas_households = workflow.get("source_atlas_households")
    if atlas_households is None:
        atlas_households = read_table(
            atlas_config["households_file"],
            schema=ATLAS_HOUSEHOLDS_SCHEMA,
        )
        workflow["source_atlas_households"] = atlas_households
    vehicles_with_em = _sample_passenger_vehicle_type_ids_for_vehicles(
        vehicles=atlas_vehicles,
        passenger_car_vehicle_types=passenger_car_with_emfac,
        households=atlas_households,
        config=config,
        seed=seed,
    )
    passenger_car_with_emfac = _finalize_passenger_vehicle_type_probabilities(
        passenger_car_vehicle_types=passenger_car_with_emfac,
        sampled_vehicles=vehicles_with_em,
        config=config,
    )
    vehicles_with_em = _prepare_mapped_passenger_vehicles_output(vehicles_with_em)
    return vehicles_with_em, passenger_car_with_emfac


def _write_mapped_passenger_vehicle_types(
    *,
    workflow: dict[str, Any],
    passenger_car_with_emfac: pd.DataFrame,
) -> str:
    passenger_car_with_emfac = attach_emissions_rates_filepaths_from_config(
        passenger_car_with_emfac,
        config=workflow["config"],
        scenario=workflow["scenario"],
        output_root=str(workflow["config"]["output"]),
        step_label="Fleet Step 3",
    )
    passenger_car_with_emfac = attach_idle_time_fraction_from_config(
        passenger_car_with_emfac,
        config=workflow["config"],
        step_label="Fleet Step 3",
    )
    passenger_bus_with_emfac = workflow.get("built_passenger_bus_vehicle_types")
    passenger_bike_with_emfac = workflow.get("built_passenger_bike_vehicle_types")
    passenger_other_with_emfac = workflow.get("built_passenger_other_vehicle_types")
    if passenger_bus_with_emfac is None or passenger_bike_with_emfac is None or passenger_other_with_emfac is None:
        raise ValueError("Step 3 requires passenger bus, bike, and other vehicle types from Step 2")
    passenger_vehicle_types = _combine_passenger_vehicle_types_for_output(
        passenger_car_with_emfac=passenger_car_with_emfac,
        passenger_bus_with_emfac=passenger_bus_with_emfac,
        passenger_bike_with_emfac=passenger_bike_with_emfac,
        passenger_other_with_emfac=passenger_other_with_emfac,
    )
    workflow["mapped_passenger_vehicle_types"] = passenger_vehicle_types
    passenger_vehicle_types_file = _write_vehicle_types(
        passenger_vehicle_types,
        _passenger_vehicle_types_output_file(
            str(workflow["config"]["output"]),
            year=workflow["config"]["atlas"]["year"],
            scenario=workflow["scenario"],
        ),
    )
    return passenger_vehicle_types_file


def _write_mapped_passenger_vehicles(
    *,
    workflow: dict[str, Any],
    vehicles_with_em: pd.DataFrame,
) -> str:
    config = workflow["config"]
    atlas_config = config["atlas"]
    vehicles_output_name = f"vehicles--{_build_year_scenario_token(year=atlas_config['year'], scenario=workflow['scenario'])}--EM.parquet"
    return _write_parquet(
        vehicles_with_em,
        str(Path(config["output"]) / vehicles_output_name),
    )


def run_step3(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 3: map passenger car vehicle types to EMFAC."""
    print("=== Step 3.1: build passenger EMFAC candidate surface ===")
    mapping_context = _build_passenger_mapping_context(workflow["config"])
    random = np.random.default_rng(int(workflow["config"]["seed"]))

    print("=== Step 3.2: map passenger car vehicle types to EMFAC ===")
    passenger_car_with_emfac = _map_passenger_car_vehicle_types(
        workflow,
        mapping_context=mapping_context,
    )

    print("=== Step 3.3: assign fuel-consumption fields to mapped passenger vehicle types ===")
    passenger_car_with_emfac = _assign_passenger_fuel_consumption_fields(
        passenger_car_vehicle_types=passenger_car_with_emfac,
        config=workflow["config"],
        seed=random,
    )
    workflow["mapped_passenger_car_vehicle_types"] = passenger_car_with_emfac

    print("=== Step 3.4: sample mapped passenger vehicleTypeId values for ATLAS vehicles ===")
    vehicles_with_em, passenger_car_with_emfac = _sample_mapped_passenger_vehicles(
        workflow=workflow,
        passenger_car_with_emfac=passenger_car_with_emfac,
        seed=random,
    )
    workflow["mapped_passenger_car_vehicle_types"] = passenger_car_with_emfac

    print("=== Step 3.5: write mapped passenger vehicle types and ATLAS vehicles ===")
    mapped_passenger_vehicle_types_file = _write_mapped_passenger_vehicle_types(
        workflow=workflow,
        passenger_car_with_emfac=passenger_car_with_emfac,
    )
    vehicles_output_file = _write_mapped_passenger_vehicles(
        workflow=workflow,
        vehicles_with_em=vehicles_with_em,
    )
    workflow["mapped_passenger_vehicle_types_file"] = mapped_passenger_vehicle_types_file
    workflow["mapped_passenger_vehicles"] = vehicles_with_em
    workflow["mapped_passenger_vehicles_file"] = vehicles_output_file
    return workflow
