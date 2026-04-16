"""Fleet Step 3: expand passenger car vehicle types into EMFAC-specific rows."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from impacts.emfac.config import read_table
from impacts.emfac.config import resolve_workflow_path


_EMFAC_KEY_COLUMNS = ["vehicleCategory", "fuel", "modelYear"]


def _require_column(frame: pd.DataFrame, column_name: str, frame_name: str) -> None:
    if column_name not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column '{column_name}'")


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_lower(value: object) -> str:
    return _normalize_text(value).lower()


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
        emfac_config["rates_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS,
    )[_EMFAC_KEY_COLUMNS].drop_duplicates()
    activity = read_table(
        emfac_config["activity_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS + ["population_vehicles", "total_vmt_vehicle_miles_per_year"],
    )
    fleet = read_table(
        emfac_config["fleet_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS,
    )[_EMFAC_KEY_COLUMNS].drop_duplicates()

    candidates = (
        activity.groupby(_EMFAC_KEY_COLUMNS, dropna=False, as_index=False)[
            ["population_vehicles", "total_vmt_vehicle_miles_per_year"]
        ]
        .sum()
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
    category_map = read_table(config["mapping"]["emfac_beam_category_map"], dtype=None)
    atlas_map = read_table(config["mapping"]["emfac_atlas_map"], dtype=None)
    _require_column(category_map, "emfac", "EMFAC BEAM category mapping file")
    _require_column(category_map, "beamVehicleCategory", "EMFAC BEAM category mapping file")
    _require_column(atlas_map, "body_type", "ATLAS EMFAC crosswalk file")

    atlas_emfac_columns = {column for column in atlas_map.columns if column != "body_type"}
    passenger_categories = (
        category_map[
            (category_map["beamVehicleCategory"].apply(_normalize_text) == "Car")
            & (category_map["emfac"].astype(str).isin(atlas_emfac_columns))
        ]["emfac"]
        .dropna()
        .astype(str)
        .tolist()
    )
    if not passenger_categories:
        raise ValueError("EMFAC BEAM category mapping file has no passenger Car mappings")

    missing = [column for column in passenger_categories if column not in atlas_map.columns]
    if missing:
        raise ValueError(
            "ATLAS EMFAC crosswalk file is missing passenger EMFAC category columns:\n"
            + "\n".join(missing)
        )

    prepared = atlas_map[["body_type"] + passenger_categories].copy()
    prepared["body_type"] = prepared["body_type"].apply(_normalize_lower)
    long = prepared.melt(
        id_vars=["body_type"],
        value_vars=passenger_categories,
        var_name="vehicleCategory",
        value_name="bodytypeWeight",
    )
    long["bodytypeWeight"] = pd.to_numeric(long["bodytypeWeight"], errors="coerce").fillna(0.0)
    return long[long["bodytypeWeight"].gt(0)].reset_index(drop=True)


def _load_bodytype_alternatives(config: dict[str, Any]) -> pd.DataFrame:
    frame = read_table(config["mapping"]["atlas_bodytype_alternatives_map"], dtype=None)
    for column_name in ["source", "target", "priority"]:
        _require_column(frame, column_name, "Passenger bodytype alternatives file")
    prepared = frame.copy()
    prepared["source"] = prepared["source"].apply(_normalize_lower)
    prepared["target"] = prepared["target"].apply(_normalize_lower)
    prepared["priority"] = pd.to_numeric(prepared["priority"], errors="coerce").fillna(999)
    return prepared


def _load_emfac_fuel_mapping(config: dict[str, Any]) -> pd.DataFrame:
    frame = read_table(config["mapping"]["emfac_beam_fuel_map"], dtype=None)
    for column_name in ["group", "emfac", "beam_primary", "beam_secondary"]:
        _require_column(frame, column_name, "EMFAC BEAM fuel mapping file")
    prepared = frame.copy()
    prepared["group"] = prepared["group"].apply(_normalize_lower)
    prepared["emfac"] = prepared["emfac"].apply(_normalize_text)
    prepared["beam_primary"] = prepared["beam_primary"].apply(_normalize_lower)
    prepared["beam_secondary"] = prepared["beam_secondary"].apply(_normalize_lower)
    return prepared


def _load_emfac_fuel_alternatives(config: dict[str, Any]) -> pd.DataFrame:
    frame = read_table(config["mapping"]["emfac_fuel_alternatives_map"], dtype=None)
    for column_name in ["source", "target", "priority"]:
        _require_column(frame, column_name, "EMFAC BEAM fuel alternatives file")
    prepared = frame.copy()
    prepared["source"] = prepared["source"].apply(_normalize_text)
    prepared["target"] = prepared["target"].apply(_normalize_text)
    prepared["priority"] = pd.to_numeric(prepared["priority"], errors="coerce").fillna(999)
    return prepared


def _extract_emfac_bodytype_candidates(
    *,
    bodytype: object,
    vehicle_category_weights: pd.DataFrame,
    bodytype_alternatives: pd.DataFrame,
) -> dict[str, float]:
    bodytype_key = _normalize_lower(bodytype)
    weights: dict[str, float] = {}

    direct = vehicle_category_weights[vehicle_category_weights["body_type"] == bodytype_key]
    for row in direct.itertuples(index=False):
        weights[str(row.vehicleCategory)] = max(weights.get(str(row.vehicleCategory), 0.0), float(row.bodytypeWeight))
    if weights:
        return weights

    alternatives = bodytype_alternatives[bodytype_alternatives["source"] == bodytype_key].sort_values("priority")
    for row in alternatives.itertuples(index=False):
        target_matches = vehicle_category_weights[vehicle_category_weights["body_type"] == row.target]
        for target_row in target_matches.itertuples(index=False):
            penalty = 1.0 / (1.0 + float(row.priority))
            candidate_weight = float(target_row.bodytypeWeight) * penalty
            vehicle_category = str(target_row.vehicleCategory)
            weights[vehicle_category] = max(weights.get(vehicle_category, 0.0), candidate_weight)
    return weights


def _extract_emfac_fuel_candidates(
    *,
    primary_fuel: object,
    secondary_fuel: object,
    fuel_mapping: pd.DataFrame,
    fuel_alternatives: pd.DataFrame,
) -> dict[str, float]:
    primary_key = _normalize_lower(primary_fuel)
    secondary_key = _normalize_lower(secondary_fuel)

    base_matches = fuel_mapping[
        (fuel_mapping["beam_primary"] == primary_key)
        & (
            (fuel_mapping["beam_secondary"] == "")
            | (fuel_mapping["beam_secondary"] == "any")
            | (fuel_mapping["beam_secondary"] == secondary_key)
        )
    ].copy()

    weights: dict[str, float] = {}
    passenger_matches = base_matches[base_matches["group"] == "passenger"]
    generic_matches = base_matches[base_matches["group"] != "passenger"]
    for row in passenger_matches.itertuples(index=False):
        weights[str(row.emfac)] = max(weights.get(str(row.emfac), 0.0), 1.0)
    if not weights:
        for row in generic_matches.itertuples(index=False):
            weights[str(row.emfac)] = max(weights.get(str(row.emfac), 0.0), 0.75)

    alternatives = fuel_alternatives[fuel_alternatives["source"].isin(list(weights.keys()))].sort_values("priority")
    for row in alternatives.itertuples(index=False):
        source_weight = weights.get(str(row.source))
        if source_weight is None:
            continue
        candidate_weight = float(source_weight) * (1.0 / (1.0 + float(row.priority)))
        weights[str(row.target)] = max(weights.get(str(row.target), 0.0), candidate_weight)
    return weights


def _extract_secondary_fuel_fallback_candidates(
    *,
    secondary_fuel: object,
    fuel_mapping: pd.DataFrame,
    fuel_alternatives: pd.DataFrame,
) -> dict[str, float]:
    secondary_key = _normalize_lower(secondary_fuel)
    if not secondary_key:
        return {}

    base_matches = fuel_mapping[
        (fuel_mapping["beam_primary"] == secondary_key)
        & ((fuel_mapping["beam_secondary"] == "") | (fuel_mapping["beam_secondary"] == "any"))
    ].copy()

    weights: dict[str, float] = {}
    passenger_matches = base_matches[base_matches["group"] == "passenger"]
    generic_matches = base_matches[base_matches["group"] != "passenger"]
    for row in passenger_matches.itertuples(index=False):
        weights[str(row.emfac)] = max(weights.get(str(row.emfac), 0.0), 0.5)
    if not weights:
        for row in generic_matches.itertuples(index=False):
            weights[str(row.emfac)] = max(weights.get(str(row.emfac), 0.0), 0.375)

    alternatives = fuel_alternatives[fuel_alternatives["source"].isin(list(weights.keys()))].sort_values("priority")
    for row in alternatives.itertuples(index=False):
        source_weight = weights.get(str(row.source))
        if source_weight is None:
            continue
        candidate_weight = float(source_weight) * (1.0 / (1.0 + float(row.priority)))
        weights[str(row.target)] = max(weights.get(str(row.target), 0.0), candidate_weight)
    return weights


def _model_year_group_contains(actual_year: object, model_year_group: object) -> bool:
    year_value = pd.to_numeric(actual_year, errors="coerce")
    if pd.isna(year_value):
        return True
    year_int = int(year_value)
    token = _normalize_lower(model_year_group)
    range_match = re.fullmatch(r"(\d{4})to(\d{4})", token)
    if range_match is not None:
        start_year = int(range_match.group(1))
        end_year = int(range_match.group(2))
        return start_year <= year_int <= end_year
    pre_match = re.fullmatch(r"pre(\d{4})", token)
    if pre_match is not None:
        cutoff_year = int(pre_match.group(1))
        return year_int < cutoff_year
    post_match = re.fullmatch(r"post(\d{4})", token)
    if post_match is not None:
        cutoff_year = int(post_match.group(1))
        return year_int > cutoff_year
    year_match = re.fullmatch(r"(\d{4})", token)
    if year_match is not None:
        return year_int == int(year_match.group(1))
    return False


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
    prepared["atlasVehicleTypeToken"] = prepared["oldVehicleTypeId"].astype(str).str.split("--", n=1).str[1]
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
    seed: int,
) -> pd.DataFrame:
    for column_name in ["bodytype", "adopt_fuel", "modelyear"]:
        _require_column(vehicles, column_name, "ATLAS vehicles file")

    prepared = _build_vehicle_income_bins(
        vehicles=vehicles.copy(),
        households=households,
        income_bins=income_bins,
    )
    modelyear_numeric = pd.to_numeric(prepared["modelyear"], errors="coerce")
    modelyear_token = modelyear_numeric.fillna(prepared["modelyear"]).astype(str).str.replace(r"\.0$", "", regex=True)
    prepared["atlasVehicleTypeToken"] = (
        prepared["bodytype"].map(_normalize_text).str.capitalize()
        + prepared["adopt_fuel"].map(_normalize_text).str.capitalize()
        + modelyear_token
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
    return prepared.drop(columns=["income_in_thousands", "incomeBin", "atlasVehicleTypeToken"])


def _build_passenger_emfac_candidates(
    *,
    vehicle_type_id: str,
    bodytype: object,
    modelyear: object,
    primary_fuel: object,
    secondary_fuel: object,
    emfac_candidates: pd.DataFrame,
    vehicle_category_weights: pd.DataFrame,
    bodytype_alternatives: pd.DataFrame,
    fuel_mapping: pd.DataFrame,
    fuel_alternatives: pd.DataFrame,
) -> pd.DataFrame:
    vehicle_category_candidates = _extract_emfac_bodytype_candidates(
        bodytype=bodytype,
        vehicle_category_weights=vehicle_category_weights,
        bodytype_alternatives=bodytype_alternatives,
    )
    if not vehicle_category_candidates:
        raise ValueError(
            f"No EMFAC category candidates available for passenger car vehicleTypeId={vehicle_type_id}, bodytype={bodytype}"
        )

    fuel_candidates = _extract_emfac_fuel_candidates(
        primary_fuel=primary_fuel,
        secondary_fuel=secondary_fuel,
        fuel_mapping=fuel_mapping,
        fuel_alternatives=fuel_alternatives,
    )
    if not fuel_candidates:
        raise ValueError(
            "No EMFAC fuel candidates available for passenger car "
            f"vehicleTypeId={vehicle_type_id}, primaryFuelType={primary_fuel}, secondaryFuelType={secondary_fuel}"
        )

    matched = emfac_candidates[
        emfac_candidates["vehicleCategory"].isin(vehicle_category_candidates.keys())
        & emfac_candidates["fuel"].isin(fuel_candidates.keys())
    ].copy()
    if matched.empty:
        raise ValueError(
            "No passenger EMFAC candidates available after applying category/fuel mapping for "
            f"vehicleTypeId={vehicle_type_id}, bodytype={bodytype}, primaryFuelType={primary_fuel}, secondaryFuelType={secondary_fuel}"
        )

    matched["vehicleCategoryWeight"] = matched["vehicleCategory"].map(vehicle_category_candidates).fillna(0.0)
    matched["fuelWeight"] = matched["fuel"].map(fuel_candidates).fillna(0.0)
    matched = matched[matched["modelYear"].map(lambda value: _model_year_group_contains(modelyear, value))].copy()
    if matched.empty:
        secondary_fuel_candidates = _extract_secondary_fuel_fallback_candidates(
            secondary_fuel=secondary_fuel,
            fuel_mapping=fuel_mapping,
            fuel_alternatives=fuel_alternatives,
        )
        if secondary_fuel_candidates:
            matched = emfac_candidates[
                emfac_candidates["vehicleCategory"].isin(vehicle_category_candidates.keys())
                & emfac_candidates["fuel"].isin(secondary_fuel_candidates.keys())
                & emfac_candidates["modelYear"].map(lambda value: _model_year_group_contains(modelyear, value))
            ].copy()
            if not matched.empty:
                matched["vehicleCategoryWeight"] = matched["vehicleCategory"].map(vehicle_category_candidates).fillna(0.0)
                matched["fuelWeight"] = matched["fuel"].map(secondary_fuel_candidates).fillna(0.0)
        if matched.empty:
            raise ValueError(
                "No passenger EMFAC candidates matched the modelYear interval for "
                f"vehicleTypeId={vehicle_type_id}, modelyear={modelyear}"
            )
    matched["score"] = (
        matched["vehicleCategoryWeight"]
        * matched["fuelWeight"]
        * pd.to_numeric(matched["fleetShare"], errors="coerce").fillna(0.0)
    )
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
    vehicle_type_atlas_crosswalk: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    for column_name in ["vehicleTypeId", "primaryFuelType", "secondaryFuelType", "sampleProbabilityWithinCategory", "sampleProbabilityString"]:
        _require_column(passenger_car_vehicle_types, column_name, "Passenger car vehicle types file")
    for column_name in ["vehicleTypeId", "bodytype", "modelyear"]:
        _require_column(vehicle_type_atlas_crosswalk, column_name, "Passenger FASTSim-ATLAS crosswalk file")

    emfac_candidates = _build_valid_emfac_candidates(config)
    vehicle_category_weights = _load_passenger_vehicle_category_weights(config)
    bodytype_alternatives = _load_bodytype_alternatives(config)
    fuel_mapping = _load_emfac_fuel_mapping(config)
    fuel_alternatives = _load_emfac_fuel_alternatives(config)

    crosswalk_keys = vehicle_type_atlas_crosswalk[["vehicleTypeId", "bodytype", "modelyear"]].drop_duplicates().copy()
    prepared = passenger_car_vehicle_types.merge(crosswalk_keys, on="vehicleTypeId", how="left")
    missing_crosswalk = prepared[prepared["bodytype"].isna()]["vehicleTypeId"].drop_duplicates()
    if not missing_crosswalk.empty:
        raise ValueError(
            "Passenger car vehicle types are missing crosswalk metadata:\n"
            + "\n".join(missing_crosswalk.astype(str).tolist())
        )

    expanded_rows: list[dict[str, Any]] = []
    for row in prepared.itertuples(index=False):
        candidates = _build_passenger_emfac_candidates(
            vehicle_type_id=str(row.vehicleTypeId),
            bodytype=row.bodytype,
            modelyear=row.modelyear,
            primary_fuel=row.primaryFuelType,
            secondary_fuel=row.secondaryFuelType,
            emfac_candidates=emfac_candidates,
            vehicle_category_weights=vehicle_category_weights,
            bodytype_alternatives=bodytype_alternatives,
            fuel_mapping=fuel_mapping,
            fuel_alternatives=fuel_alternatives,
        )
        score_sum = candidates["score"].sum()
        if score_sum <= 0:
            raise ValueError(f"Passenger car EMFAC candidate scores sum to zero for vehicleTypeId={row.vehicleTypeId}")
        candidates["probabilityShare"] = candidates["score"] / score_sum

        base_probability = float(pd.to_numeric(pd.Series([row.sampleProbabilityWithinCategory]), errors="coerce").fillna(0.0).iloc[0])
        income_bin, income_probability, ridehail_bin, ridehail_probability = _parse_probability_string(row.sampleProbabilityString)

        row_payload = row._asdict()
        for candidate in candidates.itertuples(index=False):
            share = float(candidate.probabilityShare)
            income_split = None if income_probability is None else income_probability * share
            ridehail_split = None if ridehail_probability is None else ridehail_probability * share
            updated = dict(row_payload)
            updated["oldVehicleTypeId"] = str(row.vehicleTypeId)
            updated["emfacId"] = str(candidate.emfacId)
            updated["emfacVehicleCategory"] = str(candidate.vehicleCategory)
            updated["vehicleTypeId"] = f"{candidate.emfacId}--{row.vehicleTypeId}"
            updated["sampleProbabilityWithinCategory"] = f"{base_probability * share:.6f}"
            updated["sampleProbabilityString"] = _format_probability_string(
                income_bin=income_bin,
                income_probability=income_split,
                ridehail_bin=ridehail_bin,
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
    prepared = prepared.drop(columns=["bodytype", "modelyear"]).reset_index(drop=True)
    return _normalize_written_passenger_probabilities(prepared)


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


def run_step3(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 3: assign EMFAC ids to passenger car vehicle types."""
    print("=== Step 3.1: assign emfacId to passenger car vehicle types ===")

    passenger_car_file = workflow.get("built_vehicle_types_file")
    crosswalk_file = workflow.get("vehicle_type_atlas_crosswalk_file")
    if not passenger_car_file or not crosswalk_file:
        raise ValueError("Step 3 requires passenger car vehicle types and the FASTSim-ATLAS crosswalk from Step 1")

    passenger_car_vehicle_types = read_table(str(passenger_car_file), dtype=None)
    vehicle_type_atlas_crosswalk = read_table(str(crosswalk_file), dtype=None)

    passenger_car_with_emfac = _build_passenger_car_emfac_mapping(
        passenger_car_vehicle_types=passenger_car_vehicle_types,
        vehicle_type_atlas_crosswalk=vehicle_type_atlas_crosswalk,
        config=workflow["config"],
    )
    output_file = _write_vehicle_types(passenger_car_with_emfac, str(passenger_car_file))
    workflow["built_vehicle_types"] = passenger_car_with_emfac
    workflow["built_vehicle_types_file"] = output_file

    print("=== Step 3.2: sample passenger vehicleTypeId for ATLAS vehicles ===")
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
        seed=int(config["seed"]),
    )
    vehicles_output_name = f"vehicles--{_build_year_scenario_token(year=atlas_config['year'], scenario=workflow['scenario'])}--EM.parquet"
    vehicles_output_file = _write_parquet(
        vehicles_with_em,
        str(Path(config["output"]) / vehicles_output_name),
    )
    workflow["mapped_passenger_vehicles"] = vehicles_with_em
    workflow["mapped_passenger_vehicles_file"] = vehicles_output_file
    return workflow
