"""Fleet Step 4: map EMFAC freight distributions onto FRISM vehicle types and tours."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from impacts.fleet.config import read_table
from impacts.fleet.config import resolve_workflow_path


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


def _normalize_naics_code(value: object) -> str:
    token = _normalize_text(value)
    if token.endswith(".0") and token[:-2].isdigit():
        token = token[:-2]
    return token


def _extract_naics_sector_code(value: object) -> str:
    token = _normalize_naics_code(value)
    if len(token) >= 2 and token[:2].isdigit():
        return token[:2]
    return ""


def _sanitize_emfac_component(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", _normalize_text(value)).strip("_")
    return token.replace("_", "")


def _build_emfac_id(*, vehicle_category: object, fuel: object, model_year: object) -> str:
    return (
        f"{_sanitize_emfac_component(model_year)}"
        f"{_sanitize_emfac_component(vehicle_category)}"
        f"{_sanitize_emfac_component(fuel)}"
    )


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


def _normalize_freight_vehicle_type_id(value: object) -> str:
    token = _normalize_text(value)
    token = re.sub(r"^ft[-_]+", "", token, flags=re.IGNORECASE)
    parts = [part for part in re.split(r"[-_]+", token) if part]
    return "".join(part[:1].upper() + part[1:].lower() for part in parts)


def _map_freight_beam_class(vehicle_type_id: object, vehicle_category: object, vehicle_class: object) -> str:
    normalized_id = _normalize_freight_vehicle_type_id(vehicle_type_id)
    prefix_map = {
        "Ld1": "Class12aVocational",
        "Ld3": "Class2b3Vocational",
        "Mdv": "Class456Vocational",
        "Hdt": "Class78Tractor",
        "Hdv": "Class78Vocational",
    }
    for prefix, beam_class in prefix_map.items():
        if normalized_id.startswith(prefix):
            return beam_class

    vehicle_category_token = _normalize_text(vehicle_category)
    if vehicle_category_token in {
        "Class12aVocational",
        "Class2b3Vocational",
        "Class456Vocational",
        "Class78Tractor",
        "Class78Vocational",
    }:
        return vehicle_category_token

    vehicle_class_key = _normalize_lower(vehicle_class)
    class_text_map = {
        "class 1&2a vocational": "Class12aVocational",
        "class 2&b3 vocational": "Class2b3Vocational",
        "class 4-6 vocational": "Class456Vocational",
        "class 7&8 tractor": "Class78Tractor",
        "class 7&8 vocational": "Class78Vocational",
    }
    mapped = class_text_map.get(vehicle_class_key)
    if mapped:
        return mapped
    raise ValueError(
        "Could not determine freight BEAM class for vehicleTypeId="
        f"{vehicle_type_id}, vehicleCategory={vehicle_category}, vehicleClass={vehicle_class}"
    )


def _build_valid_freight_emfac_candidates(config: dict[str, Any]) -> pd.DataFrame:
    class_map = read_table(config["mapping"]["emfac_beam_class_map"], dtype=None)
    freight_emfac_categories = (
        class_map[class_map["group"].map(_normalize_lower) == "freight"]["emfac"].dropna().astype(str).unique().tolist()
    )
    if not freight_emfac_categories:
        raise ValueError("EMFAC BEAM class mapping file has no freight mappings")

    emfac_config = config["emfac"]
    rates = read_table(emfac_config["rates_file"], dtype=None, columns=_EMFAC_KEY_COLUMNS)[_EMFAC_KEY_COLUMNS].drop_duplicates()
    activity = read_table(
        emfac_config["activity_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS + ["population", "total_vmt"],
    )
    fleet = read_table(emfac_config["fleet_file"], dtype=None, columns=_EMFAC_KEY_COLUMNS)[_EMFAC_KEY_COLUMNS].drop_duplicates()

    candidates = (
        activity.groupby(_EMFAC_KEY_COLUMNS, dropna=False, as_index=False)[["population", "total_vmt"]]
        .sum()
        .merge(rates, on=_EMFAC_KEY_COLUMNS, how="inner")
        .merge(fleet, on=_EMFAC_KEY_COLUMNS, how="inner")
        .drop_duplicates()
    )
    candidates = candidates[candidates["vehicleCategory"].isin(freight_emfac_categories)].copy()
    if candidates.empty:
        raise ValueError("No valid freight EMFAC candidates remain after intersecting rates, activity, and fleet inputs")

    total_vmt = pd.to_numeric(candidates["total_vmt"], errors="coerce").fillna(0.0).sum()
    if total_vmt <= 0:
        raise ValueError("Freight EMFAC candidates have zero total_vmt; cannot derive fleetShare")
    candidates["fleetShare"] = pd.to_numeric(candidates["total_vmt"], errors="coerce").fillna(0.0) / total_vmt
    candidates["emfacId"] = candidates.apply(
        lambda row: _build_emfac_id(
            vehicle_category=row["vehicleCategory"],
            fuel=row["fuel"],
            model_year=row["modelYear"],
        ),
        axis=1,
    )
    return candidates


def _load_freight_class_mapping(config: dict[str, Any]) -> pd.DataFrame:
    frame = read_table(config["mapping"]["emfac_beam_class_map"], dtype=None)
    for column_name in ["group", "emfac", "beam"]:
        _require_column(frame, column_name, "EMFAC BEAM class mapping file")
    prepared = frame[frame["group"].map(_normalize_lower) == "freight"].copy()
    prepared["beam"] = prepared["beam"].map(_normalize_text)
    prepared["emfac"] = prepared["emfac"].map(_normalize_text)
    return prepared[["beam", "emfac"]].drop_duplicates().reset_index(drop=True)


def _load_freight_class_alternatives(config: dict[str, Any]) -> pd.DataFrame:
    frame = read_table(config["mapping"]["beam_freight_class_alternatives_map"], dtype=None)
    for column_name in ["group", "source", "target", "priority"]:
        _require_column(frame, column_name, "Freight class alternatives file")
    prepared = frame.copy()
    prepared["group"] = prepared["group"].map(_normalize_lower)
    prepared["source"] = prepared["source"].map(_normalize_text)
    prepared["target"] = prepared["target"].map(_normalize_text)
    prepared["priority"] = pd.to_numeric(prepared["priority"], errors="coerce").fillna(999)
    return prepared


def _load_freight_fuel_mapping(config: dict[str, Any]) -> pd.DataFrame:
    frame = read_table(config["mapping"]["emfac_beam_fuel_map"], dtype=None)
    for column_name in ["group", "emfac", "beam_primary", "beam_secondary"]:
        _require_column(frame, column_name, "EMFAC BEAM fuel mapping file")
    prepared = frame.copy()
    prepared["group"] = prepared["group"].map(_normalize_lower)
    prepared["emfac"] = prepared["emfac"].map(_normalize_text)
    prepared["beam_primary"] = prepared["beam_primary"].map(_normalize_lower)
    prepared["beam_secondary"] = prepared["beam_secondary"].map(_normalize_lower)
    return prepared


def _load_freight_fuel_alternatives(config: dict[str, Any]) -> pd.DataFrame:
    frame = read_table(config["mapping"]["emfac_beam_fuel_alternatives_map"], dtype=None)
    for column_name in ["group", "source", "target", "priority"]:
        _require_column(frame, column_name, "EMFAC BEAM fuel alternatives file")
    prepared = frame.copy()
    prepared["group"] = prepared["group"].map(_normalize_lower)
    prepared["source"] = prepared["source"].map(_normalize_text)
    prepared["target"] = prepared["target"].map(_normalize_text)
    prepared["priority"] = pd.to_numeric(prepared["priority"], errors="coerce").fillna(999)
    return prepared


def _load_naics_sector_mapping(config: dict[str, Any]) -> pd.DataFrame:
    frame = read_table(config["mapping"]["naics_emfac_sector_map"], dtype=None)
    for column_name in ["naics_source", "naics_code_2", "vehicleCategory"]:
        _require_column(frame, column_name, "NAICS EMFAC sector mapping file")
    prepared = frame.copy()
    prepared["naics_source"] = prepared["naics_source"].map(_normalize_lower)
    prepared["naics_code_2"] = prepared["naics_code_2"].map(_normalize_text)
    prepared["vehicleCategory"] = prepared["vehicleCategory"].map(_normalize_text)
    return prepared[["naics_source", "naics_code_2", "vehicleCategory"]]


def _load_naics_priority_mapping(config: dict[str, Any]) -> pd.DataFrame:
    frame = read_table(config["mapping"]["naics_emfac_priority_map"], dtype=None)
    for column_name in ["naics_source", "naics_code", "vehicleCategory"]:
        _require_column(frame, column_name, "NAICS EMFAC priority mapping file")
    prepared = frame.copy()
    prepared["naics_source"] = prepared["naics_source"].map(_normalize_lower)
    prepared["naics_code"] = prepared["naics_code"].map(_normalize_text)
    prepared["vehicleCategory"] = prepared["vehicleCategory"].map(_normalize_text)
    if "match_level" in prepared.columns:
        prepared["match_level"] = pd.to_numeric(prepared["match_level"], errors="coerce").fillna(0).astype(int)
    else:
        prepared["match_level"] = prepared["naics_code"].map(len)
    return prepared[["naics_source", "naics_code", "match_level", "vehicleCategory"]]


def _load_payloadtype_mapping(config: dict[str, Any]) -> pd.DataFrame:
    frame = read_table(config["mapping"]["payloadtype_emfac_map"], dtype=None)
    for column_name in ["payloadType", "vehicleCategory"]:
        _require_column(frame, column_name, "PayloadType EMFAC mapping file")
    prepared = frame.copy()
    prepared["payloadType"] = prepared["payloadType"].map(_normalize_lower)
    prepared["vehicleCategory"] = prepared["vehicleCategory"].map(_normalize_text)
    return prepared[["payloadType", "vehicleCategory"]]


def _build_freight_payload_profiles(
    *,
    carriers: pd.DataFrame,
    payloads: pd.DataFrame,
) -> pd.DataFrame:
    for column_name in ["tourId", "vehicleTypeId"]:
        _require_column(carriers, column_name, "FRISM carriers file")
    for column_name in ["tourId", "sellerNAICS", "buyerNAICS", "payloadType", "weightInKg"]:
        _require_column(payloads, column_name, "FRISM payloads file")

    joined = payloads.merge(carriers[["tourId", "vehicleTypeId"]], on="tourId", how="inner")
    joined["oldVehicleTypeId"] = joined["vehicleTypeId"].map(_normalize_freight_vehicle_type_id)
    return joined[["oldVehicleTypeId", "sellerNAICS", "buyerNAICS", "payloadType", "weightInKg"]].copy()


def _build_freight_naics_weight_lookup(
    *,
    payload_profiles: pd.DataFrame,
    sector_mapping: pd.DataFrame,
    priority_mapping: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    if payload_profiles.empty:
        return {}

    sector_lookup: dict[tuple[str, str], dict[str, float]] = {}
    for (naics_source, naics_code_2), group in sector_mapping.groupby(["naics_source", "naics_code_2"], sort=False):
        categories = sorted(group["vehicleCategory"].dropna().astype(str).unique().tolist())
        total = float(len(categories))
        if total > 0:
            sector_lookup[(str(naics_source), str(naics_code_2))] = {
                vehicle_category: 1.0 / total for vehicle_category in categories
            }

    priority_lookup: dict[tuple[str, str], dict[str, float]] = {}
    for (naics_source, naics_code), group in priority_mapping.groupby(["naics_source", "naics_code"], sort=False):
        categories = sorted(group["vehicleCategory"].dropna().astype(str).unique().tolist())
        total = float(len(categories))
        if total > 0:
            priority_lookup[(str(naics_source), str(naics_code))] = {
                vehicle_category: 1.0 / total for vehicle_category in categories
            }

    lookup: dict[str, dict[str, float]] = {}
    priority_mix = 0.7
    for old_vehicle_type_id, group in payload_profiles.groupby("oldVehicleTypeId", sort=False):
        category_totals: dict[str, float] = {}
        row_count = len(group.index)
        for row in group.itertuples(index=False):
            for source_name in ("seller", "buyer"):
                raw_code = _normalize_naics_code(getattr(row, f"{source_name}NAICS"))
                if not raw_code:
                    continue
                sector_code = _extract_naics_sector_code(raw_code)
                sector_probs = sector_lookup.get((source_name, sector_code), {}) if sector_code else {}
                priority_candidates = [
                    (code, probs)
                    for (mapped_source, code), probs in priority_lookup.items()
                    if mapped_source == source_name and raw_code.startswith(code)
                ]
                priority_probs: dict[str, float] = {}
                if priority_candidates:
                    best_code, priority_probs = max(priority_candidates, key=lambda item: len(item[0]))

                categories = set(sector_probs.keys()) | set(priority_probs.keys())
                if not categories:
                    continue
                for vehicle_category in categories:
                    sector_weight = float(sector_probs.get(vehicle_category, 0.0))
                    priority_weight = float(priority_probs.get(vehicle_category, 0.0))
                    if priority_probs:
                        combined_weight = ((1.0 - priority_mix) * sector_weight) + (priority_mix * priority_weight)
                    else:
                        combined_weight = sector_weight
                    if combined_weight > 0:
                        category_totals[vehicle_category] = category_totals.get(vehicle_category, 0.0) + combined_weight
        if category_totals:
            total = float(sum(category_totals.values()))
            lookup[str(old_vehicle_type_id)] = {
                vehicle_category: weight / total
                for vehicle_category, weight in category_totals.items()
            }
    return lookup


def _build_freight_payloadtype_weight_lookup(
    *,
    payload_profiles: pd.DataFrame,
    payloadtype_mapping: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    if payload_profiles.empty or payloadtype_mapping.empty:
        return {}

    payloadtype_lookup: dict[str, dict[str, float]] = {}
    for payload_type, group in payloadtype_mapping.groupby("payloadType", sort=False):
        categories = sorted(group["vehicleCategory"].dropna().astype(str).unique().tolist())
        total = float(len(categories))
        if total > 0:
            payloadtype_lookup[str(payload_type)] = {
                vehicle_category: 1.0 / total for vehicle_category in categories
            }

    lookup: dict[str, dict[str, float]] = {}
    for old_vehicle_type_id, group in payload_profiles.groupby("oldVehicleTypeId", sort=False):
        category_totals: dict[str, float] = {}
        matched_rows = 0
        for row in group.itertuples(index=False):
            payload_type = _normalize_lower(row.payloadType)
            payload_probs = payloadtype_lookup.get(payload_type)
            if not payload_probs:
                continue
            matched_rows += 1
            for vehicle_category, weight in payload_probs.items():
                category_totals[vehicle_category] = category_totals.get(vehicle_category, 0.0) + float(weight)
        if matched_rows > 0 and category_totals:
            total = float(sum(category_totals.values()))
            lookup[str(old_vehicle_type_id)] = {
                vehicle_category: weight / total
                for vehicle_category, weight in category_totals.items()
            }
    return lookup


def _build_payload_mass_thresholds(payload_profiles: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if payload_profiles.empty:
        return {}
    thresholds: dict[str, tuple[float, float]] = {}
    grouped = payload_profiles.groupby("oldVehicleTypeId")["weightInKg"]
    for old_vehicle_type_id, series in grouped:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            continue
        light_threshold = float(numeric.quantile(0.5))
        heavy_threshold = float(numeric.quantile(0.9))
        thresholds[str(old_vehicle_type_id)] = (light_threshold, heavy_threshold)
    return thresholds


def _payload_mass_category_weight(*, vehicle_category: str, median_mass_kg: float, heavy_mass_kg: float) -> float:
    category = _normalize_text(vehicle_category)
    if heavy_mass_kg >= 3000:
        if category.startswith("T7") or category in {"T6 Instate Tractor Class 7", "T6 Instate Delivery Class 7", "T6 Instate Other Class 7"}:
            return 1.0
        if category.startswith("T6"):
            return 0.55
        return 0.08
    if median_mass_kg >= 800:
        if category.startswith("T7"):
            return 0.55
        if category.startswith("T6"):
            return 1.0
        return 0.30
    if median_mass_kg >= 200:
        if category.startswith("T6"):
            return 1.0
        if category in {"LHD1", "LHD2", "MDV"}:
            return 0.75
        if category.startswith("T7"):
            return 0.30
        return 0.45
    if category in {"LDA", "LDT1", "LDT2", "LHD1", "LHD2", "MDV"}:
        return 1.0
    if category.startswith("T6"):
        return 0.45
    return 0.08


def _extract_freight_class_candidates(
    *,
    beam_class: str,
    class_mapping: pd.DataFrame,
    class_alternatives: pd.DataFrame,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    direct = class_mapping[class_mapping["beam"] == beam_class]
    for row in direct.itertuples(index=False):
        weights[str(row.emfac)] = max(weights.get(str(row.emfac), 0.0), 1.0)
    if weights:
        return weights

    alternatives = class_alternatives[
        (class_alternatives["group"] == "freight")
        & (class_alternatives["source"] == beam_class)
    ].sort_values("priority")
    for row in alternatives.itertuples(index=False):
        target_matches = class_mapping[class_mapping["beam"] == row.target]
        for target_row in target_matches.itertuples(index=False):
            penalty = 1.0 / (1.0 + float(row.priority))
            emfac_category = str(target_row.emfac)
            weights[emfac_category] = max(weights.get(emfac_category, 0.0), penalty)
    return weights


def _extract_freight_fuel_candidates(
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
    freight_matches = base_matches[base_matches["group"] == "freight"]
    generic_matches = base_matches[base_matches["group"] != "freight"]
    for row in freight_matches.itertuples(index=False):
        weights[str(row.emfac)] = max(weights.get(str(row.emfac), 0.0), 1.0)
    if not weights:
        for row in generic_matches.itertuples(index=False):
            weights[str(row.emfac)] = max(weights.get(str(row.emfac), 0.0), 0.75)

    alternatives = fuel_alternatives[
        (fuel_alternatives["group"].isin(["any", "freight"]))
        & (fuel_alternatives["source"].isin(list(weights.keys())))
    ].sort_values("priority")
    for row in alternatives.itertuples(index=False):
        source_weight = weights.get(str(row.source))
        if source_weight is None:
            continue
        candidate_weight = float(source_weight) * (1.0 / (1.0 + float(row.priority)))
        weights[str(row.target)] = max(weights.get(str(row.target), 0.0), candidate_weight)
    return weights


def _build_freight_emfac_candidates(
    *,
    vehicle_type_id: str,
    beam_class: str,
    primary_fuel: object,
    secondary_fuel: object,
    emfac_candidates: pd.DataFrame,
    class_mapping: pd.DataFrame,
    class_alternatives: pd.DataFrame,
    fuel_mapping: pd.DataFrame,
    fuel_alternatives: pd.DataFrame,
    naics_weights: dict[str, float] | None = None,
    payloadtype_weights: dict[str, float] | None = None,
    median_mass_kg: float = 0.0,
    heavy_mass_kg: float = 0.0,
) -> pd.DataFrame:
    class_candidates = _extract_freight_class_candidates(
        beam_class=beam_class,
        class_mapping=class_mapping,
        class_alternatives=class_alternatives,
    )
    if not class_candidates:
        raise ValueError(
            f"No freight EMFAC class candidates available for vehicleTypeId={vehicle_type_id}, beamClass={beam_class}"
        )

    fuel_candidates = _extract_freight_fuel_candidates(
        primary_fuel=primary_fuel,
        secondary_fuel=secondary_fuel,
        fuel_mapping=fuel_mapping,
        fuel_alternatives=fuel_alternatives,
    )
    if not fuel_candidates:
        raise ValueError(
            "No freight EMFAC fuel candidates available for "
            f"vehicleTypeId={vehicle_type_id}, primaryFuelType={primary_fuel}, secondaryFuelType={secondary_fuel}"
        )

    matched = emfac_candidates[
        emfac_candidates["vehicleCategory"].isin(class_candidates.keys())
        & emfac_candidates["fuel"].isin(fuel_candidates.keys())
    ].copy()
    if matched.empty:
        raise ValueError(
            "No freight EMFAC candidates available after applying class/fuel mapping for "
            f"vehicleTypeId={vehicle_type_id}, beamClass={beam_class}, "
            f"primaryFuelType={primary_fuel}, secondaryFuelType={secondary_fuel}"
        )

    matched["classWeight"] = matched["vehicleCategory"].map(class_candidates).fillna(0.0)
    matched["fuelWeight"] = matched["fuel"].map(fuel_candidates).fillna(0.0)
    naics_default_weight = 1.0 if not naics_weights else 0.01
    matched["naicsWeight"] = matched["vehicleCategory"].map((naics_weights or {})).fillna(naics_default_weight)
    payloadtype_default_weight = 1.0 if not payloadtype_weights else 0.01
    matched["payloadTypeWeight"] = matched["vehicleCategory"].map((payloadtype_weights or {})).fillna(payloadtype_default_weight)
    matched["payloadMassWeight"] = matched["vehicleCategory"].map(
        lambda vehicle_category: _payload_mass_category_weight(
            vehicle_category=str(vehicle_category),
            median_mass_kg=median_mass_kg,
            heavy_mass_kg=heavy_mass_kg,
        )
    )
    matched["score"] = (
        matched["classWeight"]
        * matched["fuelWeight"]
        * matched["naicsWeight"]
        * matched["payloadTypeWeight"]
        * matched["payloadMassWeight"]
        * pd.to_numeric(matched["fleetShare"], errors="coerce").fillna(0.0)
    )
    matched = matched[matched["score"].gt(0)].copy()
    if matched.empty:
        raise ValueError(f"No freight EMFAC candidates retained a positive score for vehicleTypeId={vehicle_type_id}")
    return matched.sort_values(
        by=["score", "total_vmt", "population", "vehicleCategory", "fuel", "modelYear"],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _normalize_written_freight_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["sampleProbabilityWithinCategory"] = _normalize_probability_vector(prepared["sampleProbabilityWithinCategory"])
    prepared["sampleProbabilityWithinCategory"] = prepared["sampleProbabilityWithinCategory"].map(
        lambda value: f"{float(value):.6f}"
    )
    prepared["sampleProbabilityString"] = prepared["sampleProbabilityWithinCategory"].map(
        lambda value: f"all | all:{float(value):.6f}"
    )
    return prepared


def _build_freight_vehicle_types_with_emfac(
    *,
    freight_vehicle_types: pd.DataFrame,
    config: dict[str, Any],
    source_frism_carriers: pd.DataFrame,
    source_frism_payloads: pd.DataFrame,
) -> pd.DataFrame:
    for column_name in ["vehicleTypeId", "primaryFuelType", "secondaryFuelType", "sampleProbabilityWithinCategory"]:
        _require_column(freight_vehicle_types, column_name, "Freight vehicle types file")

    emfac_candidates = _build_valid_freight_emfac_candidates(config)
    class_mapping = _load_freight_class_mapping(config)
    class_alternatives = _load_freight_class_alternatives(config)
    fuel_mapping = _load_freight_fuel_mapping(config)
    fuel_alternatives = _load_freight_fuel_alternatives(config)
    naics_sector_mapping = _load_naics_sector_mapping(config)
    naics_priority_mapping = _load_naics_priority_mapping(config)
    payloadtype_mapping = _load_payloadtype_mapping(config)
    payload_profiles = _build_freight_payload_profiles(
        carriers=source_frism_carriers,
        payloads=source_frism_payloads,
    )
    naics_weight_lookup = _build_freight_naics_weight_lookup(
        payload_profiles=payload_profiles,
        sector_mapping=naics_sector_mapping,
        priority_mapping=naics_priority_mapping,
    )
    payloadtype_weight_lookup = _build_freight_payloadtype_weight_lookup(
        payload_profiles=payload_profiles,
        payloadtype_mapping=payloadtype_mapping,
    )
    payload_mass_thresholds = _build_payload_mass_thresholds(payload_profiles)

    prepared = freight_vehicle_types.copy()
    prepared["beamClass"] = prepared.apply(
        lambda row: _map_freight_beam_class(row["vehicleTypeId"], row.get("vehicleCategory"), row.get("vehicleClass")),
        axis=1,
    )

    expanded_rows: list[dict[str, Any]] = []
    for row in prepared.itertuples(index=False):
        median_mass_kg, heavy_mass_kg = payload_mass_thresholds.get(str(row.vehicleTypeId), (0.0, 0.0))
        candidates = _build_freight_emfac_candidates(
            vehicle_type_id=str(row.vehicleTypeId),
            beam_class=str(row.beamClass),
            primary_fuel=row.primaryFuelType,
            secondary_fuel=row.secondaryFuelType,
            emfac_candidates=emfac_candidates,
            class_mapping=class_mapping,
            class_alternatives=class_alternatives,
            fuel_mapping=fuel_mapping,
            fuel_alternatives=fuel_alternatives,
            naics_weights=naics_weight_lookup.get(str(row.vehicleTypeId), {}),
            payloadtype_weights=payloadtype_weight_lookup.get(str(row.vehicleTypeId), {}),
            median_mass_kg=median_mass_kg,
            heavy_mass_kg=heavy_mass_kg,
        )
        score_sum = candidates["score"].sum()
        if score_sum <= 0:
            raise ValueError(f"Freight EMFAC candidate scores sum to zero for vehicleTypeId={row.vehicleTypeId}")
        candidates["probabilityShare"] = candidates["score"] / score_sum

        base_probability = float(pd.to_numeric(pd.Series([row.sampleProbabilityWithinCategory]), errors="coerce").fillna(0.0).iloc[0])
        row_payload = row._asdict()
        for candidate in candidates.itertuples(index=False):
            share = float(candidate.probabilityShare)
            updated = dict(row_payload)
            updated["oldVehicleTypeId"] = str(row.vehicleTypeId)
            updated["emfacId"] = str(candidate.emfacId)
            updated["vehicleTypeId"] = f"{candidate.emfacId}--{row.vehicleTypeId}"
            updated["sampleProbabilityWithinCategory"] = base_probability * share
            updated["sampleProbabilityString"] = ""
            expanded_rows.append(updated)

    mapped = pd.DataFrame(expanded_rows)
    duplicate_vehicle_type_ids = mapped["vehicleTypeId"][mapped["vehicleTypeId"].duplicated()].drop_duplicates()
    if not duplicate_vehicle_type_ids.empty:
        raise ValueError(
            "Freight Step 4.1 generated duplicate vehicleTypeId values:\n"
            + "\n".join(duplicate_vehicle_type_ids.astype(str).tolist())
        )
    mapped = mapped.drop(columns=["beamClass"]).reset_index(drop=True)
    return _normalize_written_freight_probabilities(mapped)


def _build_freight_sampling_table(mapped_freight_vehicle_types: pd.DataFrame) -> dict[str, pd.DataFrame]:
    _require_column(mapped_freight_vehicle_types, "vehicleTypeId", "Mapped freight vehicle types file")
    _require_column(mapped_freight_vehicle_types, "oldVehicleTypeId", "Mapped freight vehicle types file")
    _require_column(mapped_freight_vehicle_types, "sampleProbabilityWithinCategory", "Mapped freight vehicle types file")

    prepared = mapped_freight_vehicle_types[["vehicleTypeId", "oldVehicleTypeId", "sampleProbabilityWithinCategory"]].copy()
    prepared["sampleProbabilityWithinCategory"] = pd.to_numeric(
        prepared["sampleProbabilityWithinCategory"], errors="coerce"
    ).fillna(0.0)
    sampling_groups: dict[str, pd.DataFrame] = {}
    for old_vehicle_type_id, group in prepared.groupby("oldVehicleTypeId", sort=False):
        group_prepared = group[["vehicleTypeId", "sampleProbabilityWithinCategory"]].reset_index(drop=True)
        probability_sum = group_prepared["sampleProbabilityWithinCategory"].sum()
        if probability_sum <= 0:
            group_prepared["samplingProbability"] = 1.0 / len(group_prepared)
        else:
            group_prepared["samplingProbability"] = group_prepared["sampleProbabilityWithinCategory"] / probability_sum
        sampling_groups[str(old_vehicle_type_id)] = group_prepared[["vehicleTypeId", "samplingProbability"]]
    return sampling_groups


def _map_freight_carriers_and_tours(
    *,
    carriers: pd.DataFrame,
    tours: pd.DataFrame,
    mapped_freight_vehicle_types: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for column_name in ["tourId", "vehicleTypeId"]:
        _require_column(carriers, column_name, "FRISM carriers file")
    _require_column(tours, "tourId", "FRISM tours file")

    sampling_groups = _build_freight_sampling_table(mapped_freight_vehicle_types)
    prepared_carriers = carriers.copy()
    prepared_carriers["oldVehicleTypeId"] = prepared_carriers["vehicleTypeId"].map(_normalize_freight_vehicle_type_id)

    sampled_vehicle_type_ids = pd.Series(index=prepared_carriers.index, dtype="object")
    random = np.random.default_rng(int(seed))
    for old_vehicle_type_id, group in prepared_carriers.groupby("oldVehicleTypeId", sort=False):
        candidates = sampling_groups.get(str(old_vehicle_type_id))
        if candidates is None or candidates.empty:
            raise ValueError(f"No mapped freight vehicle types available for oldVehicleTypeId={old_vehicle_type_id}")
        probabilities = candidates["samplingProbability"].to_numpy()
        sampled_vehicle_type_ids.loc[group.index] = random.choice(
            candidates["vehicleTypeId"].to_numpy(),
            size=len(group),
            p=probabilities,
        )
    prepared_carriers["vehicleTypeId"] = sampled_vehicle_type_ids.astype(str)

    tour_vehicle_types = prepared_carriers[["tourId", "vehicleTypeId"]].drop_duplicates()
    duplicate_tours = tour_vehicle_types["tourId"][tour_vehicle_types["tourId"].duplicated()].drop_duplicates()
    if not duplicate_tours.empty:
        raise ValueError(
            "Mapped freight carriers produced multiple vehicleTypeId values for the same tourId:\n"
            + "\n".join(duplicate_tours.astype(str).tolist())
        )

    prepared_tours = tours.copy()
    prepared_tours = prepared_tours.merge(tour_vehicle_types, on="tourId", how="left")
    missing_tour_vehicle_type = prepared_tours[prepared_tours["vehicleTypeId"].isna()]["tourId"].drop_duplicates()
    if not missing_tour_vehicle_type.empty:
        raise ValueError(
            "Mapped freight tours are missing sampled vehicleTypeId values for tourId values:\n"
            + "\n".join(missing_tour_vehicle_type.astype(str).tolist())
        )
    return prepared_carriers, prepared_tours


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


def run_step4(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 4: map EMFAC freight distributions onto FRISM vehicle types and tours."""
    print("=== Step 4.1: map emfacId to freight vehicle types ===")

    freight_vehicle_types_file = workflow.get("built_freight_vehicle_types_file")
    if not freight_vehicle_types_file:
        raise ValueError("Step 4 requires freight vehicle types from Step 1")
    freight_vehicle_types = read_table(str(freight_vehicle_types_file), dtype=None)
    carriers = workflow.get("source_frism_carriers")
    if carriers is None:
        carriers = read_table(workflow["config"]["frism"]["carriers_file"], dtype=None)
    payloads = workflow.get("source_frism_payloads")
    if payloads is None:
        payloads = read_table(workflow["config"]["frism"]["payloads_file"], dtype=None)
    mapped_freight_vehicle_types = _build_freight_vehicle_types_with_emfac(
        freight_vehicle_types=freight_vehicle_types,
        config=workflow["config"],
        source_frism_carriers=carriers,
        source_frism_payloads=payloads,
    )
    mapped_freight_vehicle_types_file = _write_vehicle_types(mapped_freight_vehicle_types, str(freight_vehicle_types_file))
    workflow["built_freight_vehicle_types"] = mapped_freight_vehicle_types
    workflow["built_freight_vehicle_types_file"] = mapped_freight_vehicle_types_file

    print("=== Step 4.2: distribute freight vehicleTypeId to FRISM carriers and tours ===")
    tours = workflow.get("source_frism_tours")
    if tours is None:
        tours = read_table(workflow["config"]["frism"]["tours_file"], dtype=None)

    mapped_carriers, mapped_tours = _map_freight_carriers_and_tours(
        carriers=carriers,
        tours=tours,
        mapped_freight_vehicle_types=mapped_freight_vehicle_types,
        seed=int(workflow["config"]["seed"]),
    )
    output_dir = Path(workflow["config"]["output"])
    mapped_carriers_file = _write_parquet(mapped_carriers, str(output_dir / "carriers--freight-emfac.parquet"))
    mapped_tours_file = _write_parquet(mapped_tours, str(output_dir / "tours--freight-emfac.parquet"))
    workflow["mapped_freight_carriers"] = mapped_carriers
    workflow["mapped_freight_carriers_file"] = mapped_carriers_file
    workflow["mapped_freight_tours"] = mapped_tours
    workflow["mapped_freight_tours_file"] = mapped_tours_file
    return workflow
