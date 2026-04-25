"""Fleet Step 4: map EMFAC freight distributions onto FRISM vehicle types and tours."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import yaml

from impacts.emfac.config import build_fuel_consumption_emfac_assignment_catalog
from impacts.emfac.config import EMFAC_ACTIVITY_SCHEMA
from impacts.emfac.config import EMFAC_KEY_SCHEMA
from impacts.emfac.config import read_table
from impacts.emfac.config import resolve_workflow_path
from impacts.emfac.common import attach_emissions_rates_filepaths_from_config
from impacts.emfac.common import attach_idle_time_fraction_from_config
from impacts.emfac.common import model_year_interval_distance
from impacts.emfac.common import model_year_group_id_component
from impacts.emfac.common import parse_model_year_group_interval
from impacts.emfac.fleet.step1_build_vehicle_types import _extract_model_year_from_vehicle_type_id
from impacts.emfac.fleet.step1_build_vehicle_types import _normalize_energy_file_columns
from impacts.emfac.fleet.step1_build_vehicle_types import _normalize_energy_file_path


_EMFAC_KEY_COLUMNS = ["vehicleCategory", "fuel", "modelYear"]
_FRISM_CARRIERS_SCHEMA = {
    "tourId": "string",
    "vehicleTypeId": "string",
}
_FRISM_PAYLOADS_SCHEMA = {
    "tourId": "string",
    "sellerNAICS": "string",
    "buyerNAICS": "string",
    "payloadType": "string",
    "weightInKg": "Float64",
    "sequenceRank": "Float64",
    "activityType": "string",
    "locationZone": "string",
}
_FRISM_TOURS_SCHEMA = {
    "tourId": "string",
}
_GVWR_METADATA_SCHEMA = {
    "emfac_vehicle_category": "string",
    "gvwr_lbs": "string",
}
_FREIGHT_VEHICLE_TYPES_SCHEMA = {
    "vehicleTypeId": "string",
    "vehicleCategory": "string",
    "adopt_fuel": "string",
    "sampleProbabilityWithinCategory": "string",
    "sampleProbabilityString": "string",
    "primaryFuelType": "string",
    "primaryFuelConsumptionInJoulePerMeter": "Float64",
    "primaryFuelCapacityInJoule": "Float64",
    "primaryVehicleEnergyFile": "string",
    "secondaryVehicleEnergyFile": "string",
}
_FREIGHT_FUEL_CONSUMPTION_TEMPLATE_SCHEMA = {
    "vehicleTypeId": "string",
    "primaryVehicleEnergyFile": "string",
    "secondaryVehicleEnergyFile": "string",
}
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


def _normalize_zone_id(value: object) -> str:
    token = _normalize_text(value)
    if not token:
        return ""
    if token.endswith(".0") and token[:-2].isdigit():
        token = token[:-2]
    return token


def _extract_naics_sector_code(value: object) -> str:
    token = _normalize_naics_code(value)
    if len(token) >= 2 and token[:2].isdigit():
        return token[:2]
    return ""


def _sanitize_emfac_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", _normalize_text(value))


def _sanitize_vehicle_type_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", _normalize_text(value))


def _build_emfac_id(*, vehicle_category: object, fuel: object, model_year: object) -> str:
    return (
        f"{_sanitize_emfac_component(model_year_group_id_component(model_year))}"
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
        rounded.loc[target_index] = max(0.0, round(float(rounded.loc[target_index]) + remainder, decimals))
    return rounded


def _normalize_freight_vehicle_type_id(value: object) -> str:
    token = _normalize_text(value)
    token = re.sub(r"^ft[-_]+", "", token, flags=re.IGNORECASE)
    parts = [part for part in re.split(r"[-_]+", token) if part]
    return "".join(part[:1].upper() + part[1:].lower() for part in parts)


def _require_freight_beam_vehicle_category(vehicle_category: object, vehicle_type_id: object) -> str:
    vehicle_category_token = _normalize_text(vehicle_category)
    if vehicle_category_token == "":
        raise ValueError(
            "Freight vehicle types file is missing vehicleCategory for "
            f"vehicleTypeId={vehicle_type_id}"
        )
    return vehicle_category_token


def _build_valid_freight_emfac_candidates(config: dict[str, Any]) -> pd.DataFrame:
    freight_mapping = config.get("freight_mapping", {}) or {}
    freight_vehicle_categories = freight_mapping.get("vehicle_categories", {})
    freight_emfac_categories = sorted(
        {
            str(emfac_vehicle_category).strip()
            for emfac_categories in freight_vehicle_categories.values()
            if isinstance(emfac_categories, list)
            for emfac_vehicle_category in emfac_categories
            if str(emfac_vehicle_category).strip()
        }
    )
    if not freight_emfac_categories:
        raise ValueError("Freight Bayesian DAG model has no freight vehicle_categories mappings")

    emfac_config = config["activities"]
    rates = read_table(emfac_config["freight_rates_file"], schema=EMFAC_KEY_SCHEMA)[_EMFAC_KEY_COLUMNS].drop_duplicates()
    activity = read_table(
        emfac_config["freight_activity_file"],
        schema=EMFAC_ACTIVITY_SCHEMA,
    )
    fleet = read_table(emfac_config["freight_fleet_file"], schema=EMFAC_KEY_SCHEMA)[_EMFAC_KEY_COLUMNS].drop_duplicates()

    candidates = (
        activity.groupby(_EMFAC_KEY_COLUMNS, dropna=False, as_index=False)[
            ["population_vehicles", "total_vmt_vehicle_miles_per_year"]
        ]
        .max()
        .merge(rates, on=_EMFAC_KEY_COLUMNS, how="inner")
        .merge(fleet, on=_EMFAC_KEY_COLUMNS, how="inner")
        .drop_duplicates()
    )
    candidates = candidates[candidates["vehicleCategory"].isin(freight_emfac_categories)].copy()
    if candidates.empty:
        raise ValueError("No valid freight EMFAC candidates remain after intersecting rates, activity, and fleet inputs")

    total_vmt = pd.to_numeric(
        candidates["total_vmt_vehicle_miles_per_year"], errors="coerce"
    ).fillna(0.0).sum()
    if total_vmt <= 0:
        raise ValueError("Freight EMFAC candidates have zero total_vmt_vehicle_miles_per_year; cannot derive fleetShare")
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


def _load_freight_category_mapping(config: dict[str, Any]) -> pd.DataFrame:
    freight_mapping = config.get("freight_mapping", {}) or {}
    vehicle_categories = freight_mapping.get("vehicle_categories", {})
    if not isinstance(vehicle_categories, dict) or not vehicle_categories:
        raise ValueError(
            "Freight mapping is missing vehicle_categories required for freight category mapping."
        )
    rows: list[dict[str, str]] = []
    for freight_beam_category, emfac_categories in vehicle_categories.items():
        beam_category = _normalize_text(freight_beam_category)
        if beam_category == "":
            continue
        if not isinstance(emfac_categories, list):
            raise ValueError(
                "Freight mapping vehicle_categories entries must be lists of EMFAC vehicle-category strings."
            )
        for emfac_category in emfac_categories:
            vehicle_category = _normalize_text(emfac_category)
            if vehicle_category == "":
                continue
            rows.append({"freight_beam_category": beam_category, "emfac": vehicle_category})
    prepared = pd.DataFrame(rows)
    if prepared.empty:
        raise ValueError(
            "Freight mapping vehicle_categories produced no freight category mapping rows."
        )
    return prepared.drop_duplicates().reset_index(drop=True)


def _load_freight_fuel_mapping(config: dict[str, Any]) -> pd.DataFrame:
    freight_mapping = config.get("freight_mapping", {}) or {}
    vehicle_categories = freight_mapping.get("vehicle_categories", {})
    fuel_types = freight_mapping.get("fuel_types", {})
    if not isinstance(vehicle_categories, dict) or not vehicle_categories:
        raise ValueError(
            "Freight mapping is missing vehicle_categories required for freight fuel mapping."
        )
    if not isinstance(fuel_types, dict) or not fuel_types:
        raise ValueError(
            "Freight mapping is missing fuel_types required for freight fuel mapping."
        )
    rows: list[dict[str, str]] = []
    for freight_beam_category, emfac_categories in vehicle_categories.items():
        beam_category = _normalize_text(freight_beam_category)
        if beam_category == "":
            continue
        if not isinstance(emfac_categories, list):
            raise ValueError(
                "Freight mapping vehicle_categories entries must be lists of EMFAC vehicle-category strings."
            )
        for emfac_category in emfac_categories:
            vehicle_category = _normalize_text(emfac_category)
            if vehicle_category == "":
                continue
            for adopt_fuel, emfac_fuels in fuel_types.items():
                normalized_adopt_fuel = _normalize_lower(adopt_fuel)
                if normalized_adopt_fuel == "":
                    continue
                if not isinstance(emfac_fuels, list):
                    raise ValueError(
                        "Freight mapping fuel_types entries must be lists of EMFAC fuel strings."
                    )
                for emfac_fuel in emfac_fuels:
                    fuel = _normalize_text(emfac_fuel)
                    if fuel == "":
                        continue
                    rows.append(
                        {
                            "emfac_vehicle_category": vehicle_category,
                            "emfac_fuel": fuel,
                            "beam_category": beam_category,
                            "adopt_fuel": normalized_adopt_fuel,
                        }
                    )
    prepared = pd.DataFrame(rows)
    if prepared.empty:
        raise ValueError(
            "Freight mapping vehicle_categories and fuel_types produced no freight fuel mapping rows."
        )
    return prepared.drop_duplicates().reset_index(drop=True)


def _load_naics_sector_mapping(config: dict[str, Any]) -> pd.DataFrame:
    frame = _load_vehicle_type_assignment_table(config, "naics_sector")
    for column_name in ["naics_code_2", "vehicleCategory"]:
        _require_column(frame, column_name, "NAICS EMFAC sector mapping file")
    prepared = frame.copy()
    if "naics_source" in prepared.columns:
        prepared["naics_source"] = prepared["naics_source"].map(_normalize_lower)
    else:
        prepared["naics_source"] = "all"
    prepared["naics_code_2"] = prepared["naics_code_2"].map(_normalize_text)
    prepared["vehicleCategory"] = prepared["vehicleCategory"].map(_normalize_text)
    return prepared[["naics_source", "naics_code_2", "vehicleCategory"]]


def _zone_id_candidates(value: object) -> list[str]:
    token = _normalize_zone_id(value)
    if not token:
        return []
    candidates = [token]
    stripped = token.lstrip("0")
    if stripped and stripped != token:
        candidates.append(stripped)
    return candidates


def _load_model_spec(config: dict[str, Any]) -> dict[str, Any]:
    model_file = _normalize_text(config.get("vehicle_type_assignment", {}).get("model_file"))
    if not model_file:
        return {}
    path = Path(model_file)
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _load_vehicle_type_assignment_table(config: dict[str, Any], evidence_source: str) -> pd.DataFrame:
    model_spec = _load_model_spec(config)
    fleet_assignment = model_spec.get("fleet_assignment", {})
    mappings = fleet_assignment.get("mappings", {}) if isinstance(fleet_assignment, dict) else {}
    freight_mapping = mappings.get("freight", {}) if isinstance(mappings, dict) else {}
    if evidence_source == "naics_sector":
        rows = freight_mapping.get("naics_sector", [])
        if not isinstance(rows, list):
            raise ValueError(
                "Vehicle type assignment model file mappings.freight.naics_sector must be a list."
            )
        expanded_rows: list[dict[str, Any]] = []
        for row in rows:
            categories = row.get("vehicle_category", [])
            if not isinstance(categories, list):
                categories = [categories]
            row_variants = [{k: v for k, v in row.items() if k != "vehicle_category"}]
            for code_key in ("naics_code_2", "naics_code"):
                next_variants: list[dict[str, Any]] = []
                for variant in row_variants:
                    code_value = variant.get(code_key)
                    if isinstance(code_value, list):
                        for single_code in code_value:
                            expanded_variant = dict(variant)
                            expanded_variant[code_key] = single_code
                            next_variants.append(expanded_variant)
                    else:
                        next_variants.append(variant)
                row_variants = next_variants
            for variant in row_variants:
                for category in categories:
                    expanded = dict(variant)
                    expanded["vehicleCategory"] = category
                    expanded_rows.append(expanded)
        return pd.DataFrame(expanded_rows).fillna("")
    if evidence_source == "port_zone":
        port_evidence = freight_mapping.get("port_location", [])
        if not isinstance(port_evidence, list):
            raise ValueError(
                "Vehicle type assignment model file mappings.freight.port_location must be a list."
            )
        expanded_rows: list[dict[str, Any]] = []
        for item in port_evidence:
            categories = item.get("vehicle_category", [])
            zones = item.get("zone_codes", [])
            if not isinstance(categories, list):
                categories = [categories]
            if not isinstance(zones, list):
                zones = [zones]
            for category in categories:
                for zone in zones:
                    expanded_rows.append(
                        {
                            "zone": zone,
                            "emfac_vehicle_category": category,
                            "port_name": "",
                        }
                    )
        return pd.DataFrame(expanded_rows).fillna("")
    raise ValueError(
        f"Could not resolve evidence source '{evidence_source}' from vehicle_type_assignment.model_file."
    )


def _load_port_zone_mapping(config: dict[str, Any]) -> pd.DataFrame:
    frame = _load_vehicle_type_assignment_table(config, "port_zone")
    if frame.empty:
        return pd.DataFrame(columns=["zone", "emfac_vehicle_category", "port_name"])
    for column_name in ["port_name", "zone", "emfac_vehicle_category"]:
        _require_column(frame, column_name, "Port zone EMFAC mapping file")
    prepared = frame.copy()
    prepared["port_name"] = prepared["port_name"].map(_normalize_text)
    prepared["zone"] = prepared["zone"].map(_normalize_zone_id)
    prepared["emfac_vehicle_category"] = prepared["emfac_vehicle_category"].map(_normalize_text)
    prepared = prepared[prepared["zone"] != ""].copy()

    rows: list[dict[str, str]] = []
    for row in prepared.itertuples(index=False):
        for candidate in _zone_id_candidates(row.zone):
            rows.append(
                {
                    "zone": candidate,
                    "emfac_vehicle_category": str(row.emfac_vehicle_category),
                    "port_name": str(row.port_name),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["zone", "emfac_vehicle_category", "port_name"])
    return pd.DataFrame(rows)[["zone", "emfac_vehicle_category", "port_name"]].drop_duplicates().reset_index(drop=True)


def _load_configured_port_classes(config: dict[str, Any]) -> set[str]:
    frame = _load_vehicle_type_assignment_table(config, "port_zone")
    if frame.empty:
        return set()
    _require_column(frame, "emfac_vehicle_category", "Port class likelihood table")
    return {
        _normalize_text(value)
        for value in frame["emfac_vehicle_category"].tolist()
        if _normalize_text(value)
    }


def _build_freight_payload_profiles(
    *,
    carriers: pd.DataFrame,
    payloads: pd.DataFrame,
) -> pd.DataFrame:
    for column_name in ["tourId", "vehicleTypeId"]:
        _require_column(carriers, column_name, "FRISM carriers file")
    for column_name in [
        "tourId",
        "sellerNAICS",
        "buyerNAICS",
        "payloadType",
        "weightInKg",
        "sequenceRank",
        "activityType",
    ]:
        _require_column(payloads, column_name, "FRISM payloads file")

    joined = payloads.merge(carriers[["tourId", "vehicleTypeId"]], on="tourId", how="inner")
    joined["frismVehicleTypeId"] = joined["vehicleTypeId"].map(_normalize_freight_vehicle_type_id)
    if "locationZone" in joined.columns:
        joined["locationZone"] = joined["locationZone"].map(_normalize_zone_id)
    return joined[
        [
            "tourId",
            "frismVehicleTypeId",
            "sequenceRank",
            "activityType",
            "sellerNAICS",
            "buyerNAICS",
            "payloadType",
            "weightInKg",
            "locationZone",
        ]
    ].copy()


def _normalize_likelihoods(category_totals: dict[str, float]) -> dict[str, float]:
    """Normalize raw category totals to a proper conditional probability distribution.

    Absent categories receive likelihood_floor at scoring time so that the
    log-space Naive Bayes score does not collapse to -inf for candidates that
    are merely unlikely rather than impossible.
    """
    total = float(sum(category_totals.values()))
    if total <= 0:
        return {}
    return {category: weight / total for category, weight in category_totals.items()}


def _normalize_branch_weights(branch_weights: dict[str, float]) -> dict[str, float]:
    normalized = {key: max(0.0, float(value)) for key, value in branch_weights.items()}
    total = float(sum(normalized.values()))
    if total <= 0.0:
        uniform = 1.0 / float(len(normalized)) if normalized else 1.0
        return {key: uniform for key in normalized}
    return {key: value / total for key, value in normalized.items()}


def _build_freight_bayesian_log_score(
    *,
    matched: pd.DataFrame,
    branch_weights: dict[str, float],
) -> pd.Series:
    """Build the freight Bayesian score in log space.

    The assignment DAG is:

      vehicle category prior -> category
      category -> naics sector evidence
      category -> mass evidence
      category -> port evidence

    The final score is a weighted geometric mean over the prior, NAICS-sector,
    payload-mass, and port-location branches. Branch coefficients are
    normalized to sum to 1 so every coefficient lives on the same scale.
    """
    weights = _normalize_branch_weights(branch_weights)
    fleet_share = pd.to_numeric(matched["fleetShare"], errors="coerce").fillna(0.0)
    prior_log = np.log(fleet_share.clip(lower=1e-9))
    naics_sector_log = np.log(pd.to_numeric(matched["naicsSectorLikelihood"], errors="coerce").fillna(1.0).clip(lower=1e-9))
    mass_log = np.log(pd.to_numeric(matched["payloadMassLikelihood"], errors="coerce").fillna(1.0).clip(lower=1e-9))
    port_log = np.log(pd.to_numeric(matched["portLikelihood"], errors="coerce").fillna(1.0).clip(lower=1e-9))
    return (
        (float(weights["fleet_vmt_prior"]) * prior_log)
        + (float(weights["naics_sector"]) * naics_sector_log)
        + (float(weights["payload_mass"]) * mass_log)
        + (float(weights["port_location"]) * port_log)
    )

def _build_freight_naics_weight_lookup(
    *,
    payload_profiles: pd.DataFrame,
    sector_mapping: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Return P(NAICS signal | vehicle_type) for each old vehicle type id.

    The returned dictionary maps frismVehicleTypeId to a normalized conditional
    probability distribution over EMFAC vehicle categories.  Absent categories
    receive the likelihood_floor value at scoring time rather than here so that the
    distribution remains proper (sums to 1) over the observed categories.
    """
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

    lookup: dict[str, dict[str, float]] = {}
    for old_vehicle_type_id, group in payload_profiles.groupby("frismVehicleTypeId", sort=False):
        category_totals: dict[str, float] = {}
        for row in group.itertuples(index=False):
            for source_name in ("seller", "buyer"):
                raw_code = _normalize_naics_code(getattr(row, f"{source_name}NAICS"))
                if not raw_code:
                    continue
                sector_code = _extract_naics_sector_code(raw_code)
                sector_probs = {}
                if sector_code:
                    sector_probs = sector_lookup.get((source_name, sector_code), {})
                    if not sector_probs:
                        sector_probs = sector_lookup.get(("all", sector_code), {})
                categories = set(sector_probs.keys())
                if not categories:
                    continue
                for vehicle_category in categories:
                    sector_weight = float(sector_probs.get(vehicle_category, 0.0))
                    combined_weight = sector_weight
                    if combined_weight > 0:
                        category_totals[vehicle_category] = category_totals.get(vehicle_category, 0.0) + combined_weight
        if category_totals:
            lookup[str(old_vehicle_type_id)] = _normalize_likelihoods(category_totals)
    return lookup


def _build_freight_naics_sector_weight_lookup(
    *,
    payload_profiles: pd.DataFrame,
    sector_mapping: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Return P(NAICS-sector evidence | vehicle_type) from coarse NAICS sectors only."""
    return _build_freight_naics_weight_lookup(
        payload_profiles=payload_profiles,
        sector_mapping=sector_mapping,
    )


def _signed_payload_change_kg(*, activity_type: object, weight_kg: object) -> float:
    weight = float(pd.to_numeric(pd.Series([weight_kg]), errors="coerce").fillna(0.0).iloc[0])
    activity = _normalize_lower(activity_type)
    if activity == "loading":
        return weight
    if activity == "unloading":
        return -weight
    return 0.0


def _build_payload_mass_thresholds(payload_profiles: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if payload_profiles.empty:
        return {}
    thresholds: dict[str, tuple[float, float]] = {}
    required_columns = ["tourId", "frismVehicleTypeId", "sequenceRank", "activityType", "weightInKg"]
    missing = set(required_columns).difference(payload_profiles.columns)
    if missing:
        raise ValueError(f"Payload profiles are missing required columns for payload mass calculation: {sorted(missing)}")

    working = payload_profiles[required_columns].copy()
    working["sequenceRank"] = pd.to_numeric(working["sequenceRank"], errors="coerce")
    working["weightInKg"] = pd.to_numeric(working["weightInKg"], errors="coerce").fillna(0.0)
    working["signedChangeKg"] = working.apply(
        lambda row: _signed_payload_change_kg(activity_type=row["activityType"], weight_kg=row["weightInKg"]),
        axis=1,
    )

    tour_peaks: list[dict[str, float | str]] = []
    for (old_vehicle_type_id, tour_id), group in working.groupby(["frismVehicleTypeId", "tourId"], sort=False):
        ordered = group.sort_values(by=["sequenceRank"], kind="mergesort").copy()
        ordered["rawOnboardKg"] = ordered["signedChangeKg"].cumsum()
        min_onboard = float(ordered["rawOnboardKg"].min()) if not ordered.empty else 0.0
        initial_onboard = max(0.0, -min_onboard)
        ordered["onboardKg"] = ordered["rawOnboardKg"] + initial_onboard
        peak_onboard = float(ordered["onboardKg"].max()) if not ordered.empty else 0.0
        final_onboard = float(ordered["onboardKg"].iloc[-1]) if not ordered.empty else 0.0
        tour_peaks.append(
            {
                "frismVehicleTypeId": str(old_vehicle_type_id),
                "tourId": str(tour_id),
                "peakOnboardKg": peak_onboard,
                "finalOnboardKg": final_onboard,
            }
        )

    if not tour_peaks:
        return {}

    peaks = pd.DataFrame(tour_peaks)
    for old_vehicle_type_id, group in peaks.groupby("frismVehicleTypeId", sort=False):
        numeric = pd.to_numeric(group["peakOnboardKg"], errors="coerce").dropna()
        if numeric.empty:
            continue
        light_threshold = float(numeric.quantile(0.5))
        heavy_threshold = float(numeric.quantile(0.9))
        thresholds[str(old_vehicle_type_id)] = (light_threshold, heavy_threshold)
    return thresholds


def _build_tour_port_weight_lookup(
    *,
    payload_profiles: pd.DataFrame,
    port_zone_mapping: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    if payload_profiles.empty or port_zone_mapping.empty:
        return {}

    zone_to_port_class = {
        str(row.zone): str(row.emfac_vehicle_category)
        for row in port_zone_mapping.itertuples(index=False)
        if str(row.zone) and str(row.emfac_vehicle_category)
    }
    lookup: dict[str, dict[str, float]] = {}
    for tour_id, group in payload_profiles.groupby("tourId", sort=False):
        matched = (
            group["locationZone"]
            .map(lambda value: zone_to_port_class.get(_normalize_zone_id(value), ""))
            .replace("", pd.NA)
            .dropna()
        )
        if matched.empty:
            continue
        lookup[str(tour_id)] = {str(key): 1.0 for key in matched.unique().tolist()}
    return lookup


def _parse_gvwr_lbs_upper_bound(value: object) -> float | None:
    token = _normalize_text(value).replace(",", "")
    if token == "":
        return None
    matches = re.findall(r"\d+(?:\.\d+)?", token)
    if not matches:
        return None
    return max(float(item) for item in matches)


def _load_emfac_gvwr_kg_lookup(config: dict[str, Any]) -> dict[str, float]:
    metadata_path = config.get("vehicle_category_attributes_file")
    if metadata_path in (None, ""):
        raise ValueError("Freight Step 4.4 requires vehicle_category_attributes_file in the EMFAC config.")
    metadata = read_table(str(metadata_path), schema=_GVWR_METADATA_SCHEMA)
    prepared = metadata[["emfac_vehicle_category", "gvwr_lbs"]].copy()
    prepared["emfac_vehicle_category"] = prepared["emfac_vehicle_category"].fillna("").str.strip()
    prepared["gvwr_lbs_upper"] = prepared["gvwr_lbs"].map(_parse_gvwr_lbs_upper_bound)
    prepared = prepared.loc[
        prepared["emfac_vehicle_category"].ne("") & prepared["gvwr_lbs_upper"].notna()
    ].copy()
    pounds_to_kg = 0.45359237
    return (
        prepared.drop_duplicates(subset=["emfac_vehicle_category"], keep="first")
        .assign(gvwr_kg=lambda df: df["gvwr_lbs_upper"] * pounds_to_kg)
        .set_index("emfac_vehicle_category")["gvwr_kg"]
        .to_dict()
    )


def _payload_mass_gvwr_likelihood(
    *,
    vehicle_category: str,
    observed_peak_payload_kg: float,
    gvwr_kg_lookup: dict[str, float],
    likelihood_floor: float,
    overload_penalty_power: float,
) -> float:
    if observed_peak_payload_kg <= 0.0:
        return 1.0
    gvwr_kg = gvwr_kg_lookup.get(_normalize_text(vehicle_category))
    if gvwr_kg is None or gvwr_kg <= 0.0:
        return likelihood_floor
    if observed_peak_payload_kg <= gvwr_kg:
        return max(likelihood_floor, observed_peak_payload_kg / gvwr_kg)
    penalty_power = max(float(overload_penalty_power), 1e-9)
    return max(likelihood_floor, (gvwr_kg / observed_peak_payload_kg) ** penalty_power)


def _port_category_weight(
    *,
    vehicle_category: str,
    port_weights: dict[str, float],
    configured_port_classes: set[str],
    likelihood_floor: float,
) -> float:
    category = _normalize_text(vehicle_category)
    if not configured_port_classes:
        return 1.0

    observed_port_classes = {
        port_class for port_class, value in port_weights.items() if float(value) > 0.0 and port_class in configured_port_classes
    }
    if not observed_port_classes:
        return likelihood_floor if category in configured_port_classes else 1.0
    if category in observed_port_classes:
        return 1.0
    return likelihood_floor


def _filter_required_port_classes(
    *,
    matched: pd.DataFrame,
    port_weights: dict[str, float],
    configured_port_classes: set[str],
) -> pd.DataFrame:
    if matched.empty or not configured_port_classes:
        return matched

    allowed_port_classes = {
        port_class for port_class, value in port_weights.items() if float(value) > 0.0 and port_class in configured_port_classes
    }
    if allowed_port_classes == configured_port_classes:
        return matched

    port_class_mask = matched["vehicleCategory"].isin(configured_port_classes)
    if not port_class_mask.any():
        return matched
    if not allowed_port_classes:
        return matched.loc[~port_class_mask].copy()
    return matched.loc[~port_class_mask | matched["vehicleCategory"].isin(allowed_port_classes)].copy()


def _extract_freight_category_candidates(
    *,
    freight_category: str,
    category_mapping: pd.DataFrame,
) -> set[str]:
    direct = category_mapping[category_mapping["freight_beam_category"] == freight_category]
    categories = {str(row.emfac) for row in direct.itertuples(index=False)}
    return categories


def _extract_freight_fuel_candidates(
    *,
    adopt_fuel: object,
    fuel_mapping: pd.DataFrame,
) -> set[str]:
    adopt_fuel = _normalize_lower(adopt_fuel)
    base_matches = fuel_mapping[fuel_mapping["adopt_fuel"] == adopt_fuel]

    return {str(row.emfac_fuel) for row in base_matches.itertuples(index=False)}


def _build_freight_emfac_candidates(
    *,
    vehicle_type_id: str,
    beam_vehicle_category: str,
    adopt_fuel: object,
    emfac_candidates: pd.DataFrame,
    category_mapping: pd.DataFrame,
    fuel_mapping: pd.DataFrame,
) -> pd.DataFrame:
    category_candidates = _extract_freight_category_candidates(
        freight_category=beam_vehicle_category,
        category_mapping=category_mapping,
    )
    if not category_candidates:
        raise ValueError(
            "No freight EMFAC category candidates available for "
            f"vehicleTypeId={vehicle_type_id}, freight_beam_category={beam_vehicle_category}"
        )

    fuel_candidates = _extract_freight_fuel_candidates(
        adopt_fuel=adopt_fuel,
        fuel_mapping=fuel_mapping,
    )
    if not fuel_candidates:
        raise ValueError(
            "No freight EMFAC fuel candidates available for "
            f"vehicleTypeId={vehicle_type_id}, adopt_fuel={adopt_fuel}"
        )

    # Hard filters — eliminate physically impossible candidates before scoring.
    matched = emfac_candidates[
        emfac_candidates["vehicleCategory"].isin(category_candidates)
        & emfac_candidates["fuel"].isin(fuel_candidates)
    ].copy()
    if matched.empty:
        raise ValueError(
            "No freight EMFAC candidates available after applying category/fuel hard filters for "
            f"vehicleTypeId={vehicle_type_id}, freight_beam_category={beam_vehicle_category}, "
            f"adopt_fuel={adopt_fuel}"
        )
    fleet_share = pd.to_numeric(matched["fleetShare"], errors="coerce").fillna(0.0)
    matched = matched[fleet_share.gt(0)].copy()
    if matched.empty:
        raise ValueError(f"No freight EMFAC candidates have positive fleetShare for vehicleTypeId={vehicle_type_id}")
    return matched.sort_values(
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


def _normalize_written_freight_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["sampleProbabilityWithinCategory"] = _normalize_probability_vector(prepared["sampleProbabilityWithinCategory"])
    prepared["sampleProbabilityWithinCategory"] = prepared["sampleProbabilityWithinCategory"].map(
        lambda value: f"{float(value):.6f}"
    )
    prepared["sampleProbabilityString"] = prepared["sampleProbabilityWithinCategory"].map(
        lambda value: f"income | 0-999999:{float(value):.6f}"
    )
    return prepared


def _build_freight_vehicle_types_with_emfac(
    *,
    freight_vehicle_types: pd.DataFrame,
    mapping_context: dict[str, Any],
) -> pd.DataFrame:
    for column_name in ["vehicleTypeId", "adopt_fuel", "sampleProbabilityWithinCategory"]:
        _require_column(freight_vehicle_types, column_name, "Freight vehicle types file")

    emfac_candidates = mapping_context["emfac_candidates"]
    category_mapping = mapping_context["category_mapping"]
    fuel_mapping = mapping_context["fuel_mapping"]

    prepared = freight_vehicle_types.copy()
    prepared["freight_beam_category"] = prepared.apply(
        lambda row: _require_freight_beam_vehicle_category(
            row.get("vehicleCategory"),
            row["vehicleTypeId"],
        ),
        axis=1,
    )

    expanded_rows: list[dict[str, Any]] = []
    for row in prepared.itertuples(index=False):
        candidates = _build_freight_emfac_candidates(
            vehicle_type_id=str(row.vehicleTypeId),
            beam_vehicle_category=str(row.freight_beam_category),
            adopt_fuel=row.adopt_fuel,
            emfac_candidates=emfac_candidates,
            category_mapping=category_mapping,
            fuel_mapping=fuel_mapping,
        )
        base_probability = float(pd.to_numeric(pd.Series([row.sampleProbabilityWithinCategory]), errors="coerce").fillna(0.0).iloc[0])
        fleet_share = pd.to_numeric(candidates["fleetShare"], errors="coerce").fillna(0.0)
        fleet_share_total = float(fleet_share.sum())
        if fleet_share_total <= 0.0:
            raise ValueError(f"No freight EMFAC candidates have positive fleetShare for vehicleTypeId={row.vehicleTypeId}")
        split_shares = fleet_share / fleet_share_total
        row_payload = row._asdict()
        for candidate, split_share in zip(candidates.itertuples(index=False), split_shares.tolist(), strict=False):
            updated = dict(row_payload)
            updated["frismVehicleTypeId"] = str(row.vehicleTypeId)
            updated["vehicleCategory"] = str(row.freight_beam_category)
            updated["emfacId"] = str(candidate.emfacId)
            updated["emfacVehicleCategory"] = str(candidate.vehicleCategory)
            updated["emfacFuel"] = str(candidate.fuel)
            updated["emfacResolvedModelYear"] = str(candidate.modelYear)
            updated["vehicleTypeId"] = f"{candidate.emfacId}--{row.vehicleTypeId}"
            updated["sampleProbabilityWithinCategory"] = float(base_probability) * float(split_share)
            updated["sampleProbabilityString"] = ""
            expanded_rows.append(updated)

    mapped = pd.DataFrame(expanded_rows)
    duplicate_vehicle_type_ids = mapped["vehicleTypeId"][mapped["vehicleTypeId"].duplicated()].drop_duplicates()
    if not duplicate_vehicle_type_ids.empty:
        raise ValueError(
            "Freight Step 4.1 generated duplicate vehicleTypeId values:\n"
            + "\n".join(duplicate_vehicle_type_ids.astype(str).tolist())
        )
    mapped = mapped.drop(columns=["freight_beam_category"], errors="ignore").reset_index(drop=True)
    return _normalize_written_freight_probabilities(mapped)


def _build_freight_mapping_context(
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    emfac_candidates = _build_valid_freight_emfac_candidates(config)
    category_mapping = _load_freight_category_mapping(config)
    fuel_mapping = _load_freight_fuel_mapping(config)
    return {
        "emfac_candidates": emfac_candidates,
        "category_mapping": category_mapping,
        "fuel_mapping": fuel_mapping,
    }


def _build_freight_sampling_table(mapped_freight_vehicle_types: pd.DataFrame) -> dict[str, pd.DataFrame]:
    _require_column(mapped_freight_vehicle_types, "vehicleTypeId", "Mapped freight vehicle types file")
    _require_column(mapped_freight_vehicle_types, "frismVehicleTypeId", "Mapped freight vehicle types file")
    _require_column(mapped_freight_vehicle_types, "sampleProbabilityWithinCategory", "Mapped freight vehicle types file")
    _require_column(mapped_freight_vehicle_types, "emfacVehicleCategory", "Mapped freight vehicle types file")

    prepared = mapped_freight_vehicle_types[
        ["vehicleTypeId", "frismVehicleTypeId", "emfacVehicleCategory", "sampleProbabilityWithinCategory"]
    ].copy()
    prepared["sampleProbabilityWithinCategory"] = pd.to_numeric(
        prepared["sampleProbabilityWithinCategory"], errors="coerce"
    ).fillna(0.0)
    sampling_groups: dict[str, pd.DataFrame] = {}
    for old_vehicle_type_id, group in prepared.groupby("frismVehicleTypeId", sort=False):
        sampling_groups[str(old_vehicle_type_id)] = group.rename(
            columns={
                "emfacVehicleCategory": "vehicleCategory",
                "sampleProbabilityWithinCategory": "baseProbability",
            }
        )[["vehicleTypeId", "vehicleCategory", "baseProbability"]].reset_index(drop=True)
    return sampling_groups


def _build_freight_bayesian_sampling_context(
    *,
    config: dict[str, Any],
    carriers: pd.DataFrame,
    payloads: pd.DataFrame,
) -> dict[str, Any]:
    freight_dag = config.get("freight_bayesian_dag", {}) or {}
    required_keys = ("likelihood_floor", "fleet_vmt_prior", "naics_sector", "payload_mass", "port_location")
    missing = [key for key in required_keys if key not in freight_dag]
    if missing:
        raise ValueError(
            "Freight Step 4.4 requires freight_bayesian_dag config keys: "
            + ", ".join(sorted(missing))
        )
    payload_profiles = _build_freight_payload_profiles(carriers=carriers, payloads=payloads)
    port_zone_mapping = _load_port_zone_mapping(config)
    return {
        "likelihood_floor": float(freight_dag["likelihood_floor"]),
        "fleet_vmt_prior": float(freight_dag["fleet_vmt_prior"]),
        "naics_sector": float(freight_dag["naics_sector"]),
        "payload_mass": float(freight_dag["payload_mass"]),
        "port_location": float(freight_dag["port_location"]),
        "payload_mass_enabled": bool(freight_dag.get("payload_mass_enabled", False)),
        "payload_mass_source": str(freight_dag.get("payload_mass_source", "")).strip(),
        "payload_mass_unit": str(freight_dag.get("payload_mass_unit", "")).strip(),
        "payload_mass_overload_penalty_power": float(freight_dag.get("payload_mass_overload_penalty_power", 2.0)),
        "gvwr_kg_lookup": (
            _load_emfac_gvwr_kg_lookup(config)
            if bool(freight_dag.get("payload_mass_enabled", False))
            else {}
        ),
        "naics_sector_weight_lookup": _build_freight_naics_sector_weight_lookup(
            payload_profiles=payload_profiles,
            sector_mapping=_load_naics_sector_mapping(config),
        ),
        "payload_mass_thresholds": _build_payload_mass_thresholds(payload_profiles),
        "tour_port_weight_lookup": _build_tour_port_weight_lookup(
            payload_profiles=payload_profiles,
            port_zone_mapping=port_zone_mapping,
        ),
        "configured_port_classes": _load_configured_port_classes(config),
    }


def _score_freight_sampling_candidates(
    *,
    candidates: pd.DataFrame,
    naics_sector_weights: dict[str, float],
    median_mass_kg: float,
    heavy_mass_kg: float,
    port_weights: dict[str, float],
    configured_port_classes: set[str],
    freight_dag: dict[str, float],
) -> pd.DataFrame:
    prepared = candidates.copy()
    prepared["baseProbability"] = pd.to_numeric(prepared["baseProbability"], errors="coerce").fillna(0.0)
    prepared = prepared[prepared["baseProbability"].gt(0.0)].copy()
    if prepared.empty:
        raise ValueError("Freight Step 4.4 cannot score candidates with non-positive base probabilities")

    likelihood_floor = float(freight_dag["likelihood_floor"])
    prepared["naicsSectorLikelihood"] = (
        prepared["vehicleCategory"].map(naics_sector_weights).fillna(likelihood_floor)
        if naics_sector_weights
        else 1.0
    )
    if bool(freight_dag.get("payload_mass_enabled", False)):
        if str(freight_dag.get("payload_mass_source", "")).strip() != "gvwr_lbs":
            raise ValueError(
                "Freight Step 4.4 currently supports only payload_mass.source=gvwr_lbs"
            )
        if str(freight_dag.get("payload_mass_unit", "")).strip().lower() != "lbs":
            raise ValueError(
                "Freight Step 4.4 currently supports only payload_mass.unit=lbs"
            )
        observed_peak_payload_kg = max(float(median_mass_kg), float(heavy_mass_kg))
        prepared["payloadMassLikelihood"] = prepared["vehicleCategory"].map(
            lambda vehicle_category: _payload_mass_gvwr_likelihood(
                vehicle_category=str(vehicle_category),
                observed_peak_payload_kg=observed_peak_payload_kg,
                gvwr_kg_lookup=freight_dag["gvwr_kg_lookup"],
                likelihood_floor=likelihood_floor,
                overload_penalty_power=float(freight_dag["payload_mass_overload_penalty_power"]),
            )
        )
    else:
        prepared["payloadMassLikelihood"] = 1.0
    prepared["portLikelihood"] = prepared["vehicleCategory"].map(
        lambda vehicle_category: _port_category_weight(
            vehicle_category=str(vehicle_category),
            port_weights=port_weights,
            configured_port_classes=configured_port_classes,
            likelihood_floor=likelihood_floor,
        )
    )
    log_score = _build_freight_bayesian_log_score(
        matched=prepared.rename(columns={"baseProbability": "fleetShare"}),
        branch_weights={
            "fleet_vmt_prior": float(freight_dag["fleet_vmt_prior"]),
            "naics_sector": float(freight_dag["naics_sector"]),
            "payload_mass": float(freight_dag["payload_mass"]),
            "port_location": float(freight_dag["port_location"]),
        },
    )
    prepared["posteriorScore"] = np.exp(log_score - log_score.max())
    posterior_total = float(prepared["posteriorScore"].sum())
    if posterior_total <= 0.0:
        raise ValueError("Freight Step 4.4 produced zero posterior score across all candidates")
    prepared["samplingProbability"] = prepared["posteriorScore"] / posterior_total
    return prepared


def _attach_freight_fuel_consumption_templates(
    *,
    mapped_freight_vehicle_types: pd.DataFrame,
    source_vehicle_types: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    model_file = str(config.get("vehicle_type_assignment", {}).get("model_file", "")).strip()
    breakdown_path = str(config.get("beam", {}).get("fuel_consumption_catalog", "")).strip()
    if not model_file or not breakdown_path:
        raise ValueError(
            "Freight fuel-consumption template attachment requires "
            "vehicle_type_assignment.model_file and beam.fuel_consumption_catalog"
        )
    assignment_catalog = build_fuel_consumption_emfac_assignment_catalog(model_file, breakdown_path)

    templates = _normalize_energy_file_columns(source_vehicle_types.copy())
    templates["fuelConsumptionId"] = templates["vehicleTypeId"].astype(str)
    templates["template_modelyear"] = _extract_model_year_from_vehicle_type_id(templates["vehicleTypeId"])

    rewritten_rows: list[pd.Series] = []
    template_columns = [column for column in source_vehicle_types.columns if column != "vehicleTypeId"]

    def _has_baseline_fuel_consumption_values(row: Any) -> bool:
        primary_fuel_type = _normalize_text(getattr(row, "primaryFuelType", ""))
        primary_consumption = pd.to_numeric(
            pd.Series([getattr(row, "primaryFuelConsumptionInJoulePerMeter", np.nan)]),
            errors="coerce",
        ).iloc[0]
        primary_capacity = pd.to_numeric(
            pd.Series([getattr(row, "primaryFuelCapacityInJoule", np.nan)]),
            errors="coerce",
        ).iloc[0]
        return (
            primary_fuel_type != ""
            and pd.notna(primary_consumption)
            and float(primary_consumption) > 0.0
            and pd.notna(primary_capacity)
            and float(primary_capacity) > 0.0
        )

    def _build_unassigned_template_row(row: Any, *, reason: str) -> pd.Series:
        updated = pd.Series(row._asdict()).copy()
        updated["fuelConsumptionId"] = ""
        for column_name in ["primaryVehicleEnergyFile", "secondaryVehicleEnergyFile"]:
            if column_name in updated.index:
                updated[column_name] = ""
        updated["vehicleTypeId"] = (
            "unmapped--"
            + str(getattr(row, "emfacId", ""))
            + "--"
            + str(getattr(row, "frismVehicleTypeId", ""))
        )
        print(
            "WARNING: Freight Step 4.3 leaving fuel-consumption template fields empty for "
            f"vehicleTypeId={getattr(row, 'vehicleTypeId', '')}, "
            f"frismVehicleTypeId={getattr(row, 'frismVehicleTypeId', '')}, "
            f"emfacVehicleCategory={getattr(row, 'emfacVehicleCategory', '')}, "
            f"emfacFuel={getattr(row, 'emfacFuel', '')}: {reason}"
        )
        return updated

    for row in mapped_freight_vehicle_types.itertuples(index=False):
        emfac_vehicle_category = str(getattr(row, "emfacVehicleCategory", "")).strip()
        emfac_fuel = str(getattr(row, "emfacFuel", "")).strip()
        if not emfac_vehicle_category or not emfac_fuel:
            rewritten_rows.append(pd.Series(row._asdict()).copy())
            continue
        candidates = assignment_catalog[
            assignment_catalog["emfac_vehicle_category"].astype(str).eq(emfac_vehicle_category)
            & assignment_catalog["emfac_fuel"].astype(str).eq(emfac_fuel)
        ][["fastsim_relative_path"]].drop_duplicates().copy()
        if candidates.empty:
            if _has_baseline_fuel_consumption_values(row):
                rewritten_rows.append(
                    _build_unassigned_template_row(
                        row,
                        reason="no fuel-consumption mapping matched this EMFAC class/fuel, but the source freight "
                        "vehicle type already carries baseline primary fuel consumption and capacity values",
                    )
                )
                continue
            raise ValueError(
                "No fuel-consumption freight assignment matched EMFAC-assigned class/fuel for "
                f"vehicleTypeId={getattr(row, 'vehicleTypeId', '')}, emfacVehicleCategory={emfac_vehicle_category}, "
                f"emfacFuel={emfac_fuel}"
            )
        candidates["fastsim_relative_path"] = candidates["fastsim_relative_path"].map(_normalize_energy_file_path)
        candidates = templates.merge(
            candidates,
            left_on="primaryVehicleEnergyFile",
            right_on="fastsim_relative_path",
            how="inner",
        )
        if emfac_fuel == "Phe":
            candidates = candidates[
                candidates["secondaryVehicleEnergyFile"].astype(str).str.strip().ne("")
            ].copy()
            if candidates.empty:
                raise ValueError(
                    "Freight PHEV fuel-consumption assignment requires a secondaryVehicleEnergyFile in the "
                    f"source freight vehicle types for vehicleTypeId={getattr(row, 'vehicleTypeId', '')}"
                )
        if candidates.empty:
            if _has_baseline_fuel_consumption_values(row):
                rewritten_rows.append(
                    _build_unassigned_template_row(
                        row,
                        reason="the configured fuel-consumption mapping matched, but no source freight template row "
                        "matched the assigned class/fuel; keeping baseline fuel consumption and capacity values only",
                    )
                )
                continue
            raise ValueError(
                "No freight fuel-consumption template row matched the configured assignment for "
                f"vehicleTypeId={getattr(row, 'vehicleTypeId', '')}, emfacVehicleCategory={emfac_vehicle_category}, "
                f"emfacFuel={emfac_fuel}"
            )
        try:
            requested_interval = parse_model_year_group_interval(getattr(row, "emfacResolvedModelYear", ""))
        except ValueError as error:
            raise ValueError(
                "Freight fuel-consumption template attachment could not parse EMFAC model year label "
                f"{getattr(row, 'emfacResolvedModelYear', '')} for vehicleTypeId={getattr(row, 'vehicleTypeId', '')}"
            ) from error
        candidates["yearDistance"] = candidates["template_modelyear"].map(
            lambda value: float("inf")
            if pd.isna(value)
            else model_year_interval_distance((float(value), float(value)), requested_interval)
        )
        selected = candidates.sort_values(
            ["yearDistance", "template_modelyear", "fuelConsumptionId"],
            ascending=[True, True, True],
            kind="mergesort",
        ).iloc[0]

        updated = pd.Series(row._asdict()).copy()
        for column in template_columns:
            updated[column] = selected[column]
        updated["fuelConsumptionId"] = str(selected["fuelConsumptionId"])
        updated["vehicleTypeId"] = (
            _sanitize_vehicle_type_component(selected["fuelConsumptionId"])
            + "--"
            + str(getattr(row, "emfacId"))
            + "--"
            + str(getattr(row, "frismVehicleTypeId"))
        )
        updated["frismVehicleTypeId"] = getattr(row, "frismVehicleTypeId")
        updated["sampleProbabilityWithinCategory"] = getattr(row, "sampleProbabilityWithinCategory")
        updated["sampleProbabilityString"] = getattr(row, "sampleProbabilityString")
        updated["emfacId"] = getattr(row, "emfacId")
        updated["emfacVehicleCategory"] = getattr(row, "emfacVehicleCategory")
        updated["emfacFuel"] = getattr(row, "emfacFuel")
        updated["emfacResolvedModelYear"] = getattr(row, "emfacResolvedModelYear")
        updated["vehicleCategory"] = getattr(row, "vehicleCategory", updated.get("vehicleCategory", ""))
        rewritten_rows.append(updated)

    return pd.DataFrame(rewritten_rows).reset_index(drop=True)


def _map_freight_carriers_and_tours(
    *,
    carriers: pd.DataFrame,
    tours: pd.DataFrame,
    payloads: pd.DataFrame,
    mapped_freight_vehicle_types: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for column_name in ["tourId", "vehicleTypeId"]:
        _require_column(carriers, column_name, "FRISM carriers file")
    _require_column(tours, "tourId", "FRISM tours file")

    sampling_groups = _build_freight_sampling_table(mapped_freight_vehicle_types)
    freight_dag = _build_freight_bayesian_sampling_context(
        config=config,
        carriers=carriers,
        payloads=payloads,
    )
    prepared_carriers = carriers.copy()
    prepared_carriers["frismVehicleTypeId"] = prepared_carriers["vehicleTypeId"].map(_normalize_freight_vehicle_type_id)

    sampled_vehicle_type_ids = pd.Series(index=prepared_carriers.index, dtype="object")
    random = np.random.default_rng(int(seed))
    for row in prepared_carriers.itertuples():
        candidates = sampling_groups.get(str(row.frismVehicleTypeId))
        if candidates is None or candidates.empty:
            raise ValueError(
                f"No mapped freight vehicle types available for frismVehicleTypeId={row.frismVehicleTypeId}"
            )
        filtered = _filter_required_port_classes(
            matched=candidates,
            port_weights=freight_dag["tour_port_weight_lookup"].get(str(row.tourId), {}),
            configured_port_classes=freight_dag["configured_port_classes"],
        )
        if filtered.empty:
            filtered = candidates
        median_mass_kg, heavy_mass_kg = freight_dag["payload_mass_thresholds"].get(
            str(row.frismVehicleTypeId),
            (0.0, 0.0),
        )
        scored = _score_freight_sampling_candidates(
            candidates=filtered,
            naics_sector_weights=freight_dag["naics_sector_weight_lookup"].get(str(row.frismVehicleTypeId), {}),
            median_mass_kg=median_mass_kg,
            heavy_mass_kg=heavy_mass_kg,
            port_weights=freight_dag["tour_port_weight_lookup"].get(str(row.tourId), {}),
            configured_port_classes=freight_dag["configured_port_classes"],
            freight_dag=freight_dag,
        )
        sampled_vehicle_type_ids.loc[row.Index] = random.choice(
            scored["vehicleTypeId"].to_numpy(),
            size=1,
            p=scored["samplingProbability"].to_numpy(),
        )[0]
    prepared_carriers["vehicleTypeId"] = sampled_vehicle_type_ids.astype(str)
    prepared_carriers = prepared_carriers.drop(columns=["frismVehicleTypeId"], errors="ignore")

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


def _write_mapped_freight_vehicle_types(frame: pd.DataFrame, path_like: str) -> str:
    output_path = Path(resolve_workflow_path(path_like))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return str(output_path)


def _write_mapped_freight_carriers(frame: pd.DataFrame, path_like: str) -> str:
    output_path = Path(resolve_workflow_path(path_like))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return str(output_path)


def _build_freight_emfac_mapping_context(
    workflow: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    carriers = read_table(workflow["config"]["frism"]["carriers_file"], schema=_FRISM_CARRIERS_SCHEMA)
    payloads = workflow.get("source_frism_payloads")
    if payloads is None:
        payloads = read_table(workflow["config"]["frism"]["payloads_file"], schema=_FRISM_PAYLOADS_SCHEMA)
    mapping_context = _build_freight_mapping_context(
        config=workflow["config"],
    )
    return mapping_context, carriers, payloads


def _map_freight_vehicle_types_to_emfac(
    workflow: dict[str, Any],
    *,
    mapping_context: dict[str, Any],
) -> pd.DataFrame:
    freight_vehicle_types_file = workflow.get("built_freight_vehicle_types_file")
    if not freight_vehicle_types_file:
        raise ValueError("Step 4 requires freight vehicle types from Step 1")
    freight_vehicle_types = read_table(str(freight_vehicle_types_file), schema=_FREIGHT_VEHICLE_TYPES_SCHEMA)
    return _build_freight_vehicle_types_with_emfac(
        freight_vehicle_types=freight_vehicle_types,
        mapping_context=mapping_context,
    )


def _assign_freight_fuel_consumption_templates(
    workflow: dict[str, Any],
    *,
    mapped_freight_vehicle_types: pd.DataFrame,
) -> pd.DataFrame:
    return _attach_freight_fuel_consumption_templates(
        mapped_freight_vehicle_types=mapped_freight_vehicle_types,
        source_vehicle_types=workflow["source_freight_vehicle_types"],
        config=workflow["config"],
    )


def _sample_mapped_freight_vehicle_type_ids_for_carriers(
    *,
    workflow: dict[str, Any],
    carriers: pd.DataFrame,
    payloads: pd.DataFrame,
    mapped_freight_vehicle_types: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tours = workflow.get("source_frism_tours")
    if tours is None:
        tours = read_table(workflow["config"]["frism"]["tours_file"], schema=_FRISM_TOURS_SCHEMA)
    return _map_freight_carriers_and_tours(
        carriers=carriers,
        tours=tours,
        payloads=payloads,
        mapped_freight_vehicle_types=mapped_freight_vehicle_types,
        config=workflow["config"],
        seed=int(workflow["config"]["seed"]),
    )


def _write_mapped_freight_outputs(
    *,
    workflow: dict[str, Any],
    mapped_freight_vehicle_types: pd.DataFrame,
    mapped_carriers: pd.DataFrame,
) -> tuple[str, str]:
    freight_vehicle_types_file = workflow.get("built_freight_vehicle_types_file")
    if not freight_vehicle_types_file:
        raise ValueError("Step 4 requires freight vehicle types from Step 1")
    mapped_freight_vehicle_types = attach_emissions_rates_filepaths_from_config(
        mapped_freight_vehicle_types,
        config=workflow["config"],
        scenario=workflow["scenario"],
        output_root=str(workflow["config"]["output"]),
        step_label="Fleet Step 4",
    )
    mapped_freight_vehicle_types = attach_idle_time_fraction_from_config(
        mapped_freight_vehicle_types,
        config=workflow["config"],
        step_label="Fleet Step 4",
    )
    mapped_freight_vehicle_types_file = _write_mapped_freight_vehicle_types(
        mapped_freight_vehicle_types,
        str(freight_vehicle_types_file),
    )
    output_dir = Path(workflow["config"]["output"])
    carriers_source_path = Path(resolve_workflow_path(workflow["config"]["frism"]["carriers_file"]))
    mapped_carriers_file = _write_mapped_freight_carriers(
        mapped_carriers,
        str(output_dir / f"{carriers_source_path.stem}--EM.parquet"),
    )
    return mapped_freight_vehicle_types_file, mapped_carriers_file


def run_step4(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 4: map EMFAC freight distributions onto FRISM vehicle types and tours."""
    print("=== Step 4.1: build freight EMFAC candidate surface ===")
    mapping_context, carriers, payloads = _build_freight_emfac_mapping_context(workflow)

    print("=== Step 4.2: map freight vehicle types to EMFAC ===")
    mapped_freight_vehicle_types = _map_freight_vehicle_types_to_emfac(
        workflow,
        mapping_context=mapping_context,
    )

    print("=== Step 4.3: attach fuel-consumption templates to freight vehicle types ===")
    mapped_freight_vehicle_types = _assign_freight_fuel_consumption_templates(
        workflow,
        mapped_freight_vehicle_types=mapped_freight_vehicle_types,
    )

    print("=== Step 4.4: sample mapped freight vehicleTypeId onto FRISM carriers and tours ===")
    mapped_carriers, _mapped_tours = _sample_mapped_freight_vehicle_type_ids_for_carriers(
        workflow=workflow,
        carriers=carriers,
        payloads=payloads,
        mapped_freight_vehicle_types=mapped_freight_vehicle_types,
    )

    print("=== Step 4.5: write mapped freight vehicle types and carriers ===")
    mapped_freight_vehicle_types_file, mapped_carriers_file = _write_mapped_freight_outputs(
        workflow=workflow,
        mapped_freight_vehicle_types=mapped_freight_vehicle_types,
        mapped_carriers=mapped_carriers,
    )

    workflow["built_freight_vehicle_types"] = mapped_freight_vehicle_types
    workflow["built_freight_vehicle_types_file"] = mapped_freight_vehicle_types_file
    workflow["mapped_freight_carriers"] = mapped_carriers
    workflow["mapped_freight_carriers_file"] = mapped_carriers_file
    return workflow
