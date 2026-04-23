"""Fleet Step 3: expand passenger car vehicle types into EMFAC-specific rows."""

from __future__ import annotations

from decimal import Decimal
from decimal import InvalidOperation
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from impacts.emfac.config import read_table
from impacts.emfac.config import resolve_workflow_path
from impacts.emfac.fleet.step1_build_vehicle_types import _build_atlas_vehicle_type_ids
from impacts.emfac.fleet.step1_build_vehicle_types import _compose_adopt_fuel
from impacts.emfac.fleet.step1_build_vehicle_types import _normalize_energy_file_columns
from impacts.emfac.model_year_groups import model_year_group_label
from impacts.emfac.model_year_groups import assign_model_year_groups


_EMFAC_KEY_COLUMNS = ["vehicleCategory", "fuel", "modelYear"]
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
    token = str(value).strip().lower()
    mapping = {
        "car": "car",
        "suv": "suv",
        "pickup": "pickup",
        "truck": "pickup",
        "van": "minvan",
        "minvan": "minvan",
    }
    return mapping.get(token, token)


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


def _build_year_scenario_token(*, year: object, scenario: object) -> str:
    year_token = _sanitize_output_component(year)
    scenario_token = _sanitize_output_component(scenario)
    if year_token and scenario_token:
        return f"{year_token}-{scenario_token}"
    return year_token or scenario_token


def _build_emfac_id(*, vehicle_category: object, fuel: object, model_year: object) -> str:
    return (
        f"{_sanitize_emfac_component(model_year)}"
        f"{_sanitize_emfac_component(vehicle_category)}"
        f"{_sanitize_emfac_component(fuel)}"
    )


def _build_valid_emfac_candidates(config: dict[str, Any]) -> pd.DataFrame:
    emfac_config = config["activities"]
    rates = read_table(
        emfac_config["passenger_rates_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS,
    )[_EMFAC_KEY_COLUMNS].drop_duplicates()
    activity = read_table(
        emfac_config["passenger_activity_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS + ["population_vehicles", "total_vmt_vehicle_miles_per_year"],
    )
    fleet = read_table(
        emfac_config["passenger_fleet_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS,
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
        raise ValueError("Passenger EMFAC candidates have zero total_vmt_vehicle_miles_per_year; cannot derive fleetShare")
    candidates["fleetShare"] = (
        pd.to_numeric(candidates["total_vmt_vehicle_miles_per_year"], errors="coerce").fillna(0.0) / total_vmt
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


def _load_passenger_vehicle_category_weights(config: dict[str, Any]) -> pd.DataFrame:
    passenger_model = config.get("passenger_bayesian_dag", {}) or {}
    vehicle_categories = passenger_model.get("vehicle_categories", {})
    if not isinstance(vehicle_categories, dict) or not vehicle_categories:
        raise ValueError(
            "Passenger Bayesian DAG model is missing evidence.vehicle_categories required for passenger vehicle-category support."
        )

    rows: list[dict[str, object]] = []
    for body_type, vehicle_category_list in vehicle_categories.items():
        normalized_body_type = _normalize_lower(body_type)
        if normalized_body_type == "":
            continue
        if not isinstance(vehicle_category_list, list):
            raise ValueError(
                "Passenger Bayesian DAG evidence.vehicle_categories entries must be lists of EMFAC vehicle-category strings."
            )
        for vehicle_category in vehicle_category_list:
            category = _normalize_text(vehicle_category)
            if category == "":
                continue
            rows.append(
                {
                    "body_type": normalized_body_type,
                    "vehicleCategory": category,
                    "bodytypeWeight": 1.0,
                }
            )

    prepared = pd.DataFrame(rows)
    if prepared.empty:
        raise ValueError(
            "Passenger Bayesian DAG evidence.vehicle_categories produced no passenger vehicle-category support rows."
        )
    return prepared.drop_duplicates().reset_index(drop=True)


def _load_emfac_fuel_mapping(config: dict[str, Any]) -> pd.DataFrame:
    passenger_model = config.get("passenger_bayesian_dag", {}) or {}
    vehicle_categories = passenger_model.get("vehicle_categories", {})
    fuel_types = passenger_model.get("fuel_types", {})
    category_fuel_support = passenger_model.get("category_fuel_support", {})
    if not isinstance(vehicle_categories, dict) or not vehicle_categories:
        raise ValueError(
            "Passenger Bayesian DAG model is missing evidence.vehicle_categories required for passenger fuel mapping."
        )
    if not isinstance(fuel_types, dict) or not fuel_types:
        raise ValueError(
            "Passenger Bayesian DAG model is missing evidence.fuel_types required for passenger fuel mapping."
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
            "Passenger Bayesian DAG evidence.vehicle_categories produced no passenger EMFAC categories for passenger fuel mapping."
        )

    rows: list[dict[str, str]] = []
    for vehicle_category in passenger_categories:
        supported_fuels = {
            _normalize_text(emfac_fuel)
            for emfac_fuel in category_fuel_support.get(vehicle_category, [])
            if _normalize_text(emfac_fuel)
        }
        if not supported_fuels:
            continue
        for adopt_fuel, emfac_fuels in fuel_types.items():
            normalized_adopt_fuel = _normalize_lower(adopt_fuel)
            if normalized_adopt_fuel == "":
                continue
            if not isinstance(emfac_fuels, list):
                raise ValueError(
                    "Passenger Bayesian DAG evidence.fuel_types entries must be lists of EMFAC fuel strings."
                )
            for emfac_fuel in emfac_fuels:
                fuel = _normalize_text(emfac_fuel)
                if fuel == "" or fuel not in supported_fuels:
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
            "Passenger Bayesian DAG evidence.vehicle_categories and evidence.fuel_types produced no passenger fuel mapping rows."
        )
    return prepared.drop_duplicates().reset_index(drop=True)


def _normalize_to_fastsim_adopt_fuel(*, fuel_domain: str, adopt_fuel: object) -> str:
    token = _normalize_lower(adopt_fuel)
    if fuel_domain == "ldv":
        mapping = {
            "gasoline": "conv",
            "diesel": "conv",
            "biodiesel": "conv",
            "cng": "conv",
            "naturalgas": "conv",
            "hybrid": "conv",
            "electricity": "ev",
            "hydrogen": "fuelcell",
            "electricity+gasoline": "phev",
            "electricity+diesel": "phev",
        }
        return mapping.get(token, token)
    mapping = {
        "diesel": "diesel",
        "gasoline": "diesel",
        "biodiesel": "diesel",
        "cng": "diesel",
        "naturalgas": "diesel",
        "electricity": "electricity",
        "hydrogen": "electricity",
        "electricity+gasoline": "electricity",
        "electricity+diesel": "electricity",
    }
    return mapping.get(token, token)


def _emfac_fuel_to_template_adopt_fuel(emfac_fuel: object) -> str:
    token = _normalize_text(emfac_fuel)
    mapping = {
        "Gas": "gasoline",
        "Dsl": "diesel",
        "Elec": "electricity",
        "Phe": "electricity+gasoline",
        "NG": "naturalgas",
    }
    return mapping.get(token, token.lower())


def _extract_emfac_bodytype_candidates(
    *,
    bodytype: object,
    vehicle_category_weights: pd.DataFrame,
) -> dict[str, float]:
    bodytype_key = _normalize_lower(bodytype)
    weights: dict[str, float] = {}

    direct = vehicle_category_weights[vehicle_category_weights["body_type"] == bodytype_key]
    for row in direct.itertuples(index=False):
        weights[str(row.vehicleCategory)] = max(weights.get(str(row.vehicleCategory), 0.0), float(row.bodytypeWeight))
    return weights


def _extract_emfac_fuel_candidates(
    *,
    adopt_fuel: object,
    fuel_mapping: pd.DataFrame,
) -> dict[str, float]:
    adopt_fuel = _normalize_to_fastsim_adopt_fuel(fuel_domain="ldv", adopt_fuel=adopt_fuel)

    base_matches = fuel_mapping[
        fuel_mapping["adopt_fuel"] == adopt_fuel
    ].copy()

    weights: dict[str, float] = {}
    for row in base_matches.itertuples(index=False):
        weights[str(row.emfac_fuel)] = max(weights.get(str(row.emfac_fuel), 0.0), 1.0)
    return weights


def _build_emfac_target_share_lookup(
    *,
    candidates: pd.DataFrame,
    population_bias: float,
    vmt_bias: float,
) -> dict[str, float]:
    working = candidates.copy()
    working["population_weight"] = pd.to_numeric(working.get("population_vehicles", 0.0), errors="coerce").fillna(0.0)
    working["vmt_weight"] = pd.to_numeric(
        working.get("total_vmt_vehicle_miles_per_year", 0.0), errors="coerce"
    ).fillna(0.0)
    aggregated = (
        working.groupby("emfacId", dropna=False)[["population_weight", "vmt_weight"]]
        .sum(min_count=1)
        .reset_index()
    )
    aggregated["population_share"] = 0.0
    population_total = float(aggregated["population_weight"].sum())
    if population_total > 0.0:
        aggregated["population_share"] = aggregated["population_weight"] / population_total
    aggregated["vmt_share"] = 0.0
    vmt_total = float(aggregated["vmt_weight"].sum())
    if vmt_total > 0.0:
        aggregated["vmt_share"] = aggregated["vmt_weight"] / vmt_total
    aggregated["target_score"] = (
        aggregated["population_share"].pow(float(population_bias))
        * aggregated["vmt_share"].pow(float(vmt_bias))
    )
    positive = aggregated["target_score"].gt(0)
    if not positive.any():
        aggregated["target_score"] = 1.0
    target_total = float(aggregated["target_score"].sum())
    if target_total <= 0.0:
        return {str(row.emfacId): 0.0 for row in aggregated.itertuples(index=False)}
    aggregated["target_share"] = aggregated["target_score"] / target_total
    return {str(row.emfacId): float(row.target_share) for row in aggregated.itertuples(index=False)}


def _normalize_branch_weights(branch_weights: dict[str, float]) -> dict[str, float]:
    normalized = {key: max(0.0, float(value)) for key, value in branch_weights.items()}
    total = float(sum(normalized.values()))
    if total <= 0.0:
        uniform = 1.0 / float(len(normalized)) if normalized else 1.0
        return {key: uniform for key in normalized}
    return {key: value / total for key, value in normalized.items()}


def _build_passenger_dag_log_score(
    *,
    matched: pd.DataFrame,
    branch_weights: dict[str, float],
) -> pd.Series:
    weights = _normalize_branch_weights(branch_weights)
    bodytype_log = np.log(pd.to_numeric(matched["vehicleCategoryWeight"], errors="coerce").fillna(0.0).clip(lower=1e-9))
    fuel_log = np.log(pd.to_numeric(matched["fuelWeight"], errors="coerce").fillna(0.0).clip(lower=1e-9))
    return (
        (float(weights["bodytype"]) * bodytype_log)
        + (float(weights["fuel"]) * fuel_log)
    )


def _apply_global_emfac_target_balancing(
    *,
    candidate_rows: pd.DataFrame,
    target_share_lookup: dict[str, float],
    iteration_count: int = 8,
    ratio_floor: float = 0.25,
    ratio_ceiling: float = 4.0,
) -> pd.DataFrame:
    if candidate_rows.empty:
        return candidate_rows
    prepared = candidate_rows.copy()
    prepared["base_probability"] = pd.to_numeric(prepared["base_probability"], errors="coerce").fillna(0.0)
    prepared["local_score"] = pd.to_numeric(prepared["local_score"], errors="coerce").fillna(0.0)
    prepared = prepared.loc[prepared["base_probability"].gt(0.0) & prepared["local_score"].gt(0.0)].copy()
    if prepared.empty:
        return prepared

    multipliers = {str(emfac_id): 1.0 for emfac_id in prepared["emfacId"].astype(str).unique().tolist()}
    for emfac_id, target_share in target_share_lookup.items():
        if emfac_id in multipliers:
            multipliers[emfac_id] = max(float(target_share), 1e-9)

    for _ in range(max(1, int(iteration_count))):
        prepared["global_multiplier"] = prepared["emfacId"].astype(str).map(multipliers).fillna(1.0)
        prepared["adjusted_score"] = prepared["local_score"] * prepared["global_multiplier"]
        prepared["adjusted_score_sum"] = prepared.groupby("source_row_id", dropna=False)["adjusted_score"].transform("sum")
        prepared["probabilityShare"] = np.where(
            prepared["adjusted_score_sum"].gt(0.0),
            prepared["adjusted_score"] / prepared["adjusted_score_sum"],
            0.0,
        )
        projected = (
            prepared.assign(projected_weight=prepared["base_probability"] * prepared["probabilityShare"])
            .groupby("emfacId", dropna=False)["projected_weight"]
            .sum()
            .reset_index()
        )
        projected_total = float(projected["projected_weight"].sum())
        if projected_total <= 0.0:
            break
        projected["projected_share"] = projected["projected_weight"] / projected_total
        for row in projected.itertuples(index=False):
            emfac_id = str(row.emfacId)
            target_share = float(target_share_lookup.get(emfac_id, 0.0))
            current_share = float(row.projected_share)
            if target_share <= 0.0 or current_share <= 0.0:
                continue
            ratio = target_share / current_share
            ratio = max(float(ratio_floor), min(float(ratio_ceiling), float(ratio)))
            multipliers[emfac_id] = multipliers.get(emfac_id, 1.0) * ratio

    prepared["global_multiplier"] = prepared["emfacId"].astype(str).map(multipliers).fillna(1.0)
    prepared["adjusted_score"] = prepared["local_score"] * prepared["global_multiplier"]
    prepared["adjusted_score_sum"] = prepared.groupby("source_row_id", dropna=False)["adjusted_score"].transform("sum")
    prepared["probabilityShare"] = np.where(
        prepared["adjusted_score_sum"].gt(0.0),
        prepared["adjusted_score"] / prepared["adjusted_score_sum"],
        0.0,
    )
    return prepared


def _parse_model_year_label(label: object) -> tuple[float, float] | None:
    text = str(label).strip()
    if not text:
        return None
    if text.startswith("pre") and text[3:].isdigit():
        return (float("-inf"), float(int(text[3:]) - 1))
    if text.startswith("post") and text[4:].isdigit():
        return (float(int(text[4:]) + 1), float("inf"))
    if "to" in text:
        left, _, right = text.partition("to")
        if left.isdigit() and right.isdigit():
            return (float(int(left)), float(int(right)))
    return None


def _interval_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    a_min, a_max = a
    b_min, b_max = b
    if a_max < b_min:
        return b_min - a_max
    if b_max < a_min:
        return a_min - b_max
    return 0.0


def _resolve_passenger_model_year_groups(
    *,
    requested_model_year_group: object,
    matched_candidates: pd.DataFrame,
) -> list[str]:
    requested = str(requested_model_year_group).strip()
    available = [
        str(value).strip()
        for value in matched_candidates["modelYear"].dropna().astype(str).drop_duplicates().tolist()
        if str(value).strip()
    ]
    if requested in available:
        return [requested]

    requested_interval = _parse_model_year_label(requested)
    available_intervals = {
        label: _parse_model_year_label(label)
        for label in available
    }
    if requested_interval is None:
        raise ValueError(
            "No passenger EMFAC candidates matched the configured modelYear group and the requested "
            f"group could not be parsed for fallback: modelYearGroup={requested}, available={available}"
        )

    overlapping = [
        label
        for label, interval in available_intervals.items()
        if interval is not None and _interval_distance(requested_interval, interval) == 0.0
    ]
    if overlapping:
        return overlapping

    ranked = [
        (label, _interval_distance(requested_interval, interval))
        for label, interval in available_intervals.items()
        if interval is not None
    ]
    if not ranked:
        raise ValueError(
            "No passenger EMFAC candidates matched the configured modelYear group and no compatible "
            f"fallback group could be resolved for modelYearGroup={requested}, available={available}"
        )
    best_distance = min(distance for _, distance in ranked)
    return [label for label, distance in ranked if distance == best_distance]


def _parse_probability_string(probability_string: object) -> tuple[str, float | None, str, float | None]:
    text = _normalize_text(probability_string)
    if not text:
        return ("all", None, "all", None)

    income_bin = "all"
    income_probability: float | None = None
    ridehail_bin = "all"
    ridehail_probability: float | None = None

    for part in [token.strip() for token in text.split(";") if token.strip()]:
        side, _, remainder = part.partition("|")
        bucket, _, probability_token = remainder.strip().partition(":")
        probability_value = pd.to_numeric(pd.Series([probability_token.strip()]), errors="coerce").iloc[0]
        if side.strip().lower() == "income":
            income_bin = bucket.strip() or "all"
            income_probability = None if pd.isna(probability_value) else float(probability_value)
        elif side.strip().lower() == "ridehail":
            ridehail_bin = bucket.strip() or "all"
            ridehail_probability = None if pd.isna(probability_value) else float(probability_value)
    return income_bin, income_probability, ridehail_bin, ridehail_probability


def _format_probability_string(
    *,
    income_bin: str,
    income_probability: float | None,
    ridehail_bin: str,
    ridehail_probability: float | None,
) -> str:
    parts: list[str] = []
    if ridehail_probability is not None:
        parts.append(f"ridehail | {ridehail_bin}:{ridehail_probability:.6f}")
    if income_probability is not None:
        parts.append(f"income | {income_bin}:{income_probability:.6f}")
    return "; ".join(parts)


def _normalize_probability_vector(series: pd.Series, *, decimals: int = 6) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    rounded = numeric.round(decimals)
    if rounded.empty:
        return rounded
    remainder = round(1.0 - float(rounded.sum()), decimals)
    if remainder != 0:
        target_index = rounded[rounded.gt(0)].index[-1] if rounded.gt(0).any() else rounded.index[-1]
        rounded.loc[target_index] = round(float(rounded.loc[target_index]) + remainder, decimals)
    return rounded


def _validate_income_bins(income_bins: list[object] | None) -> list[float]:
    if not income_bins:
        raise ValueError("Step 3.2 requires atlas.income_bins to be configured")
    numeric_bins = [float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]) for value in income_bins]
    if any(pd.isna(value) for value in numeric_bins):
        raise ValueError("atlas.income_bins must contain only numeric interval edges")
    if any(left >= right for left, right in zip(numeric_bins[:-1], numeric_bins[1:])):
        raise ValueError("atlas.income_bins must be strictly increasing")
    return numeric_bins


def _lowest_income_bin_label(income_bins: list[object] | None) -> str:
    numeric_bins = _validate_income_bins(income_bins)
    return f"{int(numeric_bins[0])}-{int(numeric_bins[1])}"


def _normalize_written_passenger_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["sampleProbabilityWithinCategory"] = _normalize_probability_vector(
        prepared["sampleProbabilityWithinCategory"]
    )

    parsed = prepared["sampleProbabilityString"].apply(_parse_probability_string)
    prepared["incomeBinKey"] = parsed.apply(lambda value: value[0])
    prepared["incomeProbabilityValue"] = parsed.apply(lambda value: value[1])
    prepared["ridehailBinKey"] = parsed.apply(lambda value: value[2])
    prepared["ridehailProbabilityValue"] = parsed.apply(lambda value: value[3])

    for income_bin in prepared["incomeBinKey"].dropna().unique():
        mask = prepared["incomeBinKey"] == income_bin
        prepared.loc[mask, "incomeProbabilityValue"] = _normalize_probability_vector(
            prepared.loc[mask, "incomeProbabilityValue"]
        ).to_numpy()
    for ridehail_bin in prepared["ridehailBinKey"].dropna().unique():
        mask = prepared["ridehailBinKey"] == ridehail_bin
        prepared.loc[mask, "ridehailProbabilityValue"] = _normalize_probability_vector(
            prepared.loc[mask, "ridehailProbabilityValue"]
        ).to_numpy()

    prepared["sampleProbabilityWithinCategory"] = prepared["sampleProbabilityWithinCategory"].map(
        lambda value: f"{float(value):.6f}"
    )
    prepared["sampleProbabilityString"] = prepared.apply(
        lambda row: _format_probability_string(
            income_bin=str(row["incomeBinKey"]),
            income_probability=None if pd.isna(row["incomeProbabilityValue"]) else float(row["incomeProbabilityValue"]),
            ridehail_bin=str(row["ridehailBinKey"]),
            ridehail_probability=None if pd.isna(row["ridehailProbabilityValue"]) else float(row["ridehailProbabilityValue"]),
        ),
        axis=1,
    )
    return prepared.drop(
        columns=["incomeBinKey", "incomeProbabilityValue", "ridehailBinKey", "ridehailProbabilityValue"]
    )


def _build_vehicle_income_bins(
    *,
    vehicles: pd.DataFrame,
    households: pd.DataFrame,
    income_bins: list[object] | None,
) -> pd.DataFrame:
    _require_column(vehicles, "household_id", "ATLAS vehicles file")
    _require_column(households, "income_in_thousands", "ATLAS households file")

    households_prepared = households.copy()
    if "household_id" not in households_prepared.columns:
        households_prepared = households_prepared.reset_index()
        candidate_columns = [
            column_name
            for column_name in households_prepared.columns
            if column_name not in {"income_in_thousands"} and pd.api.types.is_numeric_dtype(households_prepared[column_name])
        ]
        if not candidate_columns:
            raise ValueError("ATLAS households file is missing a usable household identifier column or index")
        households_prepared = households_prepared.rename(columns={candidate_columns[0]: "household_id"})

    vehicle_income = vehicles.merge(
        households_prepared[["household_id", "income_in_thousands"]].drop_duplicates(),
        on="household_id",
        how="left",
    )
    numeric_bins = _validate_income_bins(income_bins)
    labels = [f"{int(left)}-{int(right)}" for left, right in zip(numeric_bins[:-1], numeric_bins[1:])]
    vehicle_income["incomeBin"] = pd.cut(
        pd.to_numeric(vehicle_income["income_in_thousands"], errors="coerce"),
        bins=numeric_bins,
        labels=labels,
        right=False,
        include_lowest=True,
    ).astype("object")
    vehicle_income["incomeBin"] = vehicle_income["incomeBin"].fillna(_lowest_income_bin_label(income_bins))
    return vehicle_income


def _build_passenger_vehicle_sampling_table(passenger_car_vehicle_types: pd.DataFrame) -> pd.DataFrame:
    _require_column(passenger_car_vehicle_types, "vehicleTypeId", "Passenger car vehicle types file")
    _require_column(passenger_car_vehicle_types, "oldVehicleTypeId", "Passenger car vehicle types file")
    _require_column(passenger_car_vehicle_types, "sampleProbabilityString", "Passenger car vehicle types file")

    prepared = passenger_car_vehicle_types[["vehicleTypeId", "oldVehicleTypeId", "sampleProbabilityString"]].copy()
    parsed = prepared["sampleProbabilityString"].apply(_parse_probability_string)
    old_ids = prepared["oldVehicleTypeId"].astype(str)
    split_ids = old_ids.str.split("--", n=1).str[1]
    prepared["atlasVehicleTypeToken"] = split_ids.where(split_ids.notna(), old_ids)
    prepared["incomeBin"] = parsed.apply(lambda value: value[0])
    prepared["incomeProbability"] = parsed.apply(lambda value: value[1])
    prepared["incomeProbability"] = pd.to_numeric(prepared["incomeProbability"], errors="coerce").fillna(0.0)
    return prepared[["vehicleTypeId", "atlasVehicleTypeToken", "incomeBin", "incomeProbability"]].copy()


def _sample_passenger_vehicle_type_ids_for_vehicles(
    *,
    vehicles: pd.DataFrame,
    passenger_car_vehicle_types: pd.DataFrame,
    households: pd.DataFrame,
    income_bins: list[object] | None,
    model_year_groups: dict[str, list[dict[str, object]]],
    seed: int,
) -> pd.DataFrame:
    for column_name in ["bodytype", "adopt_fuel", "modelyear"]:
        _require_column(vehicles, column_name, "ATLAS vehicles file")

    prepared = _build_vehicle_income_bins(
        vehicles=vehicles.copy(),
        households=households,
        income_bins=income_bins,
    )
    prepared["vehicleCategory"] = "LDA"
    prepared = assign_model_year_groups(
        prepared,
        model_year_groups,
        year_column="modelyear",
        category_column="vehicleCategory",
        output_column="emfacModelYearGroup",
    )
    prepared["atlasVehicleTypeToken"] = _build_atlas_vehicle_type_ids(
        prepared["bodytype"].map(_normalize_text),
        prepared["adopt_fuel"].map(_normalize_text),
        prepared["emfacModelYearGroup"].astype(str),
    )

    sampling_table = _build_passenger_vehicle_sampling_table(passenger_car_vehicle_types)
    lowest_income_bin = _lowest_income_bin_label(income_bins)
    lowest_bin_alias_rows: list[pd.DataFrame] = []
    for atlas_vehicle_type_token, group in sampling_table.groupby("atlasVehicleTypeToken", sort=False):
        if (group["incomeBin"] == lowest_income_bin).any():
            continue
        alias_group = group.copy()
        alias_group["incomeBin"] = lowest_income_bin
        lowest_bin_alias_rows.append(alias_group)
    if lowest_bin_alias_rows:
        sampling_table = pd.concat([sampling_table] + lowest_bin_alias_rows, ignore_index=True)

    sampling_groups = {
        group_key: group[["vehicleTypeId", "incomeProbability"]].reset_index(drop=True)
        for group_key, group in sampling_table.groupby(["atlasVehicleTypeToken", "incomeBin"], dropna=False, sort=False)
    }
    sampled_vehicle_type_ids = pd.Series(index=prepared.index, dtype="object")
    random = np.random.default_rng(int(seed))

    group_columns = ["atlasVehicleTypeToken", "incomeBin"]
    for group_key, group in prepared.groupby(group_columns, dropna=False, sort=False):
        candidates = sampling_groups.get(group_key)
        if candidates is None or candidates.empty:
            candidates = sampling_groups.get((group_key[0], lowest_income_bin))
        if candidates is None or candidates.empty:
            raise ValueError(
                "No passenger car vehicle-type candidates available for atlas/income group "
                f"atlasVehicleTypeId={group_key[0]}, incomeBin={group_key[1]}, "
                f"lowestIncomeBin={lowest_income_bin}"
            )
        candidates = candidates.copy()
        probabilities = pd.to_numeric(candidates["incomeProbability"], errors="coerce").fillna(0.0).to_numpy()
        probability_sum = probabilities.sum()
        if probability_sum <= 0:
            probabilities = None
        else:
            probabilities = probabilities / probability_sum
        sampled_vehicle_type_ids.loc[group.index] = random.choice(
            candidates["vehicleTypeId"].to_numpy(),
            size=len(group),
            p=probabilities,
        )

    prepared["vehicleTypeId"] = sampled_vehicle_type_ids.astype(str)
    return prepared.drop(columns=["income_in_thousands", "incomeBin", "atlasVehicleTypeToken", "emfacModelYearGroup", "vehicleCategory"])


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

    result = pd.DataFrame(
        {
            "household_id": household_id,
            "vehicle_id": vehicle_id,
            "householdId": household_id_alias,
            "vehicleId": household_id_alias + "-" + vehicle_id_alias,
            "vehicleTypeId": vehicle_type_id,
            "initialSoc": pd.Series(pd.NA, index=prepared.index, dtype="Float64"),
        }
    )
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
    vehicle_category_weights: pd.DataFrame,
    fuel_mapping: pd.DataFrame,
    bodytype_weight: float = 1.0,
    fuel_weight: float = 1.0,
) -> pd.DataFrame:
    vehicle_category_candidates = _extract_emfac_bodytype_candidates(
        bodytype=bodytype,
        vehicle_category_weights=vehicle_category_weights,
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
        emfac_candidates["vehicleCategory"].isin(vehicle_category_candidates.keys())
        & emfac_candidates["fuel"].isin(fuel_candidates.keys())
    ].copy()
    if matched.empty:
        raise ValueError(
            "No passenger EMFAC candidates available after applying category/fuel mapping for "
            f"vehicleTypeId={vehicle_type_id}, bodytype={bodytype}, adopt_fuel={adopt_fuel}"
        )

    matched["vehicleCategoryWeight"] = matched["vehicleCategory"].map(vehicle_category_candidates).fillna(0.0)
    matched["fuelWeight"] = matched["fuel"].map(fuel_candidates).fillna(0.0)
    resolved_model_year_groups = _resolve_passenger_model_year_groups(
        requested_model_year_group=model_year_group,
        matched_candidates=matched,
    )
    matched = matched[matched["modelYear"].astype(str).isin(resolved_model_year_groups)].copy()
    if matched.empty:
        raise ValueError(
            "No passenger EMFAC candidates matched the configured modelYear group for "
            f"vehicleTypeId={vehicle_type_id}, modelYearGroup={model_year_group}"
        )
    passenger_log_score = _build_passenger_dag_log_score(
        matched=matched,
        branch_weights={
            "bodytype": bodytype_weight,
            "fuel": fuel_weight,
        },
    )
    matched["score"] = np.exp(passenger_log_score - passenger_log_score.max())
    matched = matched[matched["score"].gt(0)].copy()
    if matched.empty:
        raise ValueError(
            "No passenger EMFAC candidates retained a positive score for "
            f"vehicleTypeId={vehicle_type_id}"
        )
    matched = matched.sort_values(
        by=[
            "score",
            "total_vmt_vehicle_miles_per_year",
            "population_vehicles",
            "vehicleCategory",
            "fuel",
            "modelYear",
        ],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return matched


def _build_passenger_car_emfac_mapping(
    *,
    passenger_car_vehicle_types: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    for column_name in ["vehicleTypeId", "adopt_fuel", "sampleProbabilityWithinCategory", "sampleProbabilityString"]:
        _require_column(passenger_car_vehicle_types, column_name, "Passenger car vehicle types file")
    for column_name in ["bodytype", "emfacModelYearGroup"]:
        _require_column(passenger_car_vehicle_types, column_name, "Passenger car vehicle types file")

    emfac_candidates = _build_valid_emfac_candidates(config)
    vehicle_category_weights = _load_passenger_vehicle_category_weights(config)
    fuel_mapping = _load_emfac_fuel_mapping(config)
    passenger_model = config.get("passenger_bayesian_dag", {}) or {}
    bodytype_weight = float(passenger_model.get("bodytype", 1.0))
    fuel_weight = float(passenger_model.get("fuel", 1.0))
    emfac_population_weight = float(passenger_model.get("emfac_population", 1.0))
    emfac_vmt_weight = float(passenger_model.get("emfac_vmt", 0.0))
    target_share_lookup = _build_emfac_target_share_lookup(
        candidates=emfac_candidates,
        population_bias=emfac_population_weight,
        vmt_bias=emfac_vmt_weight,
    )
    prepared = passenger_car_vehicle_types.copy()

    expanded_rows: list[dict[str, Any]] = []
    pending_candidates: list[dict[str, Any]] = []
    for source_row_id, row in enumerate(prepared.itertuples(index=False)):
        row_payload = row._asdict()
        base_probability = float(pd.to_numeric(pd.Series([row.sampleProbabilityWithinCategory]), errors="coerce").fillna(0.0).iloc[0])
        income_bin, income_probability, ridehail_bin, ridehail_probability = _parse_probability_string(row.sampleProbabilityString)
        try:
            candidates = _build_passenger_emfac_candidates(
                vehicle_type_id=str(row.vehicleTypeId),
                bodytype=row.bodytype,
                model_year_group=row.emfacModelYearGroup,
                adopt_fuel=row.adopt_fuel,
                emfac_candidates=emfac_candidates,
                vehicle_category_weights=vehicle_category_weights,
                fuel_mapping=fuel_mapping,
                bodytype_weight=bodytype_weight,
                fuel_weight=fuel_weight,
            )
        except ValueError as error:
            if "No EMFAC fuel candidates available for passenger car" in str(error):
                updated = dict(row_payload)
                updated["oldVehicleTypeId"] = str(row.vehicleTypeId)
                updated["emfacId"] = ""
                updated["emfacVehicleCategory"] = ""
                updated["emfacFuel"] = ""
                updated["emfacResolvedModelYear"] = ""
                expanded_rows.append(updated)
                continue
            raise
        for candidate in candidates.itertuples(index=False):
            pending_candidates.append(
                {
                    "source_row_id": int(source_row_id),
                    "base_probability": float(base_probability),
                    "income_bin": income_bin,
                    "income_probability": income_probability,
                    "ridehail_bin": ridehail_bin,
                    "ridehail_probability": ridehail_probability,
                    "local_score": float(candidate.score),
                    "row_payload": row_payload,
                    "oldVehicleTypeId": str(row.vehicleTypeId),
                    "emfacId": str(candidate.emfacId),
                    "emfacVehicleCategory": str(candidate.vehicleCategory),
                    "emfacFuel": str(candidate.fuel),
                    "emfacResolvedModelYear": str(candidate.modelYear),
                }
            )

    if pending_candidates:
        calibrated = _apply_global_emfac_target_balancing(
            candidate_rows=pd.DataFrame(pending_candidates),
            target_share_lookup=target_share_lookup,
        )
        for candidate in calibrated.itertuples(index=False):
            share = float(candidate.probabilityShare)
            income_split = None if candidate.income_probability is None else float(candidate.income_probability) * share
            ridehail_split = (
                None if candidate.ridehail_probability is None else float(candidate.ridehail_probability) * share
            )
            updated = dict(candidate.row_payload)
            updated["oldVehicleTypeId"] = str(candidate.oldVehicleTypeId)
            updated["emfacId"] = str(candidate.emfacId)
            updated["emfacVehicleCategory"] = str(candidate.emfacVehicleCategory)
            updated["emfacFuel"] = str(candidate.emfacFuel)
            updated["emfacResolvedModelYear"] = str(candidate.emfacResolvedModelYear)
            updated["vehicleTypeId"] = f"{candidate.emfacId}--{candidate.oldVehicleTypeId}"
            updated["sampleProbabilityWithinCategory"] = f"{float(candidate.base_probability) * share:.6f}"
            updated["sampleProbabilityString"] = _format_probability_string(
                income_bin=str(candidate.income_bin),
                income_probability=income_split,
                ridehail_bin=str(candidate.ridehail_bin),
                ridehail_probability=ridehail_split,
            )
            expanded_rows.append(updated)

    prepared = pd.DataFrame(expanded_rows)
    duplicate_vehicle_type_ids = prepared["vehicleTypeId"][prepared["vehicleTypeId"].duplicated()].drop_duplicates()
    if not duplicate_vehicle_type_ids.empty:
        raise ValueError(
            "Passenger car Step 3 generated duplicate vehicleTypeId values:\n"
            + "\n".join(duplicate_vehicle_type_ids.astype(str).tolist())
        )
    return _normalize_written_passenger_probabilities(prepared)


def _attach_passenger_fastsim_templates(
    *,
    passenger_car_vehicle_types: pd.DataFrame,
    source_vehicle_types: pd.DataFrame,
    vehicle_type_mapping: pd.DataFrame,
) -> pd.DataFrame:
    templates = source_vehicle_types.copy()
    templates["fastsimVehicleTypeId"] = templates["vehicleTypeId"].astype(str)
    template_mapping = vehicle_type_mapping[
        ["vehicleTypeId", "body_type", "modelyear", "primaryFuelType", "secondaryFuelType"]
    ].drop_duplicates().copy()
    template_mapping["fastsimVehicleTypeId"] = template_mapping["vehicleTypeId"].astype(str)
    template_mapping["bodytype_norm"] = template_mapping["body_type"].apply(_normalize_bodytype)
    template_mapping["template_adopt_fuel"] = [
        _compose_adopt_fuel(primary_fuel, secondary_fuel)
        for primary_fuel, secondary_fuel in zip(
            template_mapping["primaryFuelType"],
            template_mapping["secondaryFuelType"],
        )
    ]
    template_mapping["template_modelyear"] = pd.to_numeric(template_mapping["modelyear"], errors="coerce")
    templates = templates.merge(
        template_mapping[["fastsimVehicleTypeId", "bodytype_norm", "template_adopt_fuel", "template_modelyear"]],
        on="fastsimVehicleTypeId",
        how="inner",
    )

    rewritten_rows: list[pd.Series] = []
    template_columns = [column for column in source_vehicle_types.columns if column != "vehicleTypeId"]
    for row in passenger_car_vehicle_types.itertuples(index=False):
        if not _normalize_text(getattr(row, "emfacFuel", "")):
            rewritten_rows.append(pd.Series(row._asdict()).copy())
            continue
        bodytype_norm = _normalize_bodytype(
            getattr(row, "passenger_bodytype_norm", "") or getattr(row, "bodytype", "")
        )
        template_adopt_fuel = _emfac_fuel_to_template_adopt_fuel(getattr(row, "emfacFuel", ""))
        candidates = templates.loc[
            templates["bodytype_norm"].eq(bodytype_norm)
            & templates["template_adopt_fuel"].astype(str).str.lower().eq(template_adopt_fuel)
        ].copy()
        if candidates.empty:
            raise ValueError(
                "No FASTSim passenger template matched EMFAC-assigned class/fuel for "
                f"vehicleTypeId={getattr(row, 'vehicleTypeId', '')}, bodytype={getattr(row, 'bodytype', '')}, "
                f"emfacFuel={getattr(row, 'emfacFuel', '')}"
            )
        requested_interval = _parse_model_year_label(getattr(row, "emfacResolvedModelYear", ""))
        if requested_interval is None:
            raise ValueError(
                "Passenger FASTSim attachment could not parse EMFAC model year label "
                f"{getattr(row, 'emfacResolvedModelYear', '')} for vehicleTypeId={getattr(row, 'vehicleTypeId', '')}"
            )
        candidates["yearDistance"] = candidates["template_modelyear"].map(
            lambda value: float("inf")
            if pd.isna(value)
            else _interval_distance((float(value), float(value)), requested_interval)
        )
        selected = candidates.sort_values(
            ["yearDistance", "template_modelyear", "fastsimVehicleTypeId"],
            ascending=[True, True, True],
            kind="mergesort",
        ).iloc[0]

        updated = pd.Series(row._asdict()).copy()
        for column in template_columns:
            updated[column] = selected[column]
        updated["vehicleTypeId"] = getattr(row, "vehicleTypeId")
        updated["oldVehicleTypeId"] = getattr(row, "oldVehicleTypeId")
        updated["sampleProbabilityWithinCategory"] = getattr(row, "sampleProbabilityWithinCategory")
        updated["sampleProbabilityString"] = getattr(row, "sampleProbabilityString")
        updated["adopt_fuel"] = getattr(row, "adopt_fuel")
        updated["emfacId"] = getattr(row, "emfacId")
        updated["emfacVehicleCategory"] = getattr(row, "emfacVehicleCategory")
        updated["emfacFuel"] = getattr(row, "emfacFuel")
        updated["emfacResolvedModelYear"] = getattr(row, "emfacResolvedModelYear")
        updated["bodytype"] = getattr(row, "bodytype")
        updated["passenger_bodytype_norm"] = getattr(
            row,
            "passenger_bodytype_norm",
            getattr(row, "bodytype", ""),
        )
        updated["emfacModelYearGroup"] = getattr(row, "emfacModelYearGroup")
        updated["modelyear"] = getattr(row, "modelyear")
        rewritten_rows.append(updated)

    return _normalize_energy_file_columns(pd.DataFrame(rewritten_rows).reset_index(drop=True))


def _write_vehicle_types(frame: pd.DataFrame, path_like: str) -> str:
    output_path = Path(resolve_workflow_path(path_like))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return str(output_path)


def _write_parquet(frame: pd.DataFrame, path_like: str) -> str:
    output_path = Path(resolve_workflow_path(path_like))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return str(output_path)


def _run_step3_substep_map_vehicle_types(workflow: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    passenger_car_file = workflow.get("built_vehicle_types_file")
    if not passenger_car_file:
        raise ValueError("Step 3 requires passenger car vehicle types from Step 1")
    passenger_car_vehicle_types = read_table(str(passenger_car_file), dtype=None)
    passenger_car_with_emfac = _build_passenger_car_emfac_mapping(
        passenger_car_vehicle_types=passenger_car_vehicle_types,
        config=workflow["config"],
    )
    passenger_car_with_emfac = _attach_passenger_fastsim_templates(
        passenger_car_vehicle_types=passenger_car_with_emfac,
        source_vehicle_types=workflow["source_fastsim_passenger_vehicle_types"],
        vehicle_type_mapping=workflow["source_fastsim_passenger_vehicle_type_mapping"],
    )
    return passenger_car_with_emfac, _write_vehicle_types(passenger_car_with_emfac, str(passenger_car_file))


def _run_step3_substep_sample_atlas_vehicles(
    *,
    workflow: dict[str, Any],
    passenger_car_with_emfac: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    config = workflow["config"]
    atlas_config = config["atlas"]
    atlas_vehicles = read_table(atlas_config["vehicles_file"], dtype=None, columns=None)
    atlas_households = read_table(
        atlas_config["households_file"],
        dtype=None,
        columns=None,
    )
    vehicles_with_em = _sample_passenger_vehicle_type_ids_for_vehicles(
        vehicles=atlas_vehicles,
        passenger_car_vehicle_types=passenger_car_with_emfac,
        households=atlas_households,
        income_bins=atlas_config.get("income_bins"),
        model_year_groups=config["activities"]["model_year_groups"],
        seed=int(config["seed"]),
    )
    vehicles_with_em = _prepare_mapped_passenger_vehicles_output(vehicles_with_em)
    vehicles_output_name = f"vehicles--{_build_year_scenario_token(year=atlas_config['year'], scenario=workflow['scenario'])}--EM.parquet"
    vehicles_output_file = _write_parquet(
        vehicles_with_em,
        str(Path(config["output"]) / vehicles_output_name),
    )
    return vehicles_with_em, vehicles_output_file


def run_step3(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 3: assign EMFAC ids to passenger car vehicle types."""
    print("=== Step 3.1: assign emfacId to passenger car vehicle types ===")
    passenger_car_with_emfac, output_file = _run_step3_substep_map_vehicle_types(workflow)
    workflow["built_vehicle_types"] = passenger_car_with_emfac
    workflow["built_vehicle_types_file"] = output_file

    print("=== Step 3.2: sample passenger vehicleTypeId for ATLAS vehicles ===")
    vehicles_with_em, vehicles_output_file = _run_step3_substep_sample_atlas_vehicles(
        workflow=workflow,
        passenger_car_with_emfac=passenger_car_with_emfac,
    )
    workflow["mapped_passenger_vehicles"] = vehicles_with_em
    workflow["mapped_passenger_vehicles_file"] = vehicles_output_file
    return workflow
