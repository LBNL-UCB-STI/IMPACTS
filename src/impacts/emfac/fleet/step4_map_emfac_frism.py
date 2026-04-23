"""Fleet Step 4: map EMFAC freight distributions onto FRISM vehicle types and tours."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import yaml

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
        rounded.loc[target_index] = max(0.0, round(float(rounded.loc[target_index]) + remainder, decimals))
    return rounded


def _normalize_freight_vehicle_type_id(value: object) -> str:
    token = _normalize_text(value)
    token = re.sub(r"^ft[-_]+", "", token, flags=re.IGNORECASE)
    parts = [part for part in re.split(r"[-_]+", token) if part]
    return "".join(part[:1].upper() + part[1:].lower() for part in parts)


def _map_freight_beam_vehicle_category(vehicle_type_id: object, vehicle_category: object, vehicle_class: object) -> str:
    normalized_id = _normalize_freight_vehicle_type_id(vehicle_type_id)
    prefix_map = {
        "Ld1": "Class12aVocational",
        "Ld3": "Class2b3Vocational",
        "Mdv": "Class456Vocational",
        "Hdt": "Class78Tractor",
        "Hdv": "Class78Vocational",
    }
    for prefix, beam_vehicle_category in prefix_map.items():
        if normalized_id.startswith(prefix):
            return beam_vehicle_category

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
        "Could not determine freight BEAM vehicle category for vehicleTypeId="
        f"{vehicle_type_id}, vehicleCategory={vehicle_category}, vehicleClass={vehicle_class}"
    )


def _build_valid_freight_emfac_candidates(config: dict[str, Any]) -> pd.DataFrame:
    freight_model = config.get("freight_bayesian_dag", {}) or {}
    freight_vehicle_categories = freight_model.get("vehicle_categories", {})
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
    rates = read_table(emfac_config["freight_rates_file"], dtype=None, columns=_EMFAC_KEY_COLUMNS)[_EMFAC_KEY_COLUMNS].drop_duplicates()
    activity = read_table(
        emfac_config["freight_activity_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS + ["population_vehicles", "total_vmt_vehicle_miles_per_year"],
    )
    fleet = read_table(emfac_config["freight_fleet_file"], dtype=None, columns=_EMFAC_KEY_COLUMNS)[_EMFAC_KEY_COLUMNS].drop_duplicates()

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
    freight_model = config.get("freight_bayesian_dag", {}) or {}
    vehicle_categories = freight_model.get("vehicle_categories", {})
    if not isinstance(vehicle_categories, dict) or not vehicle_categories:
        raise ValueError(
            "Freight Bayesian DAG model is missing evidence.vehicle_categories required for freight category mapping."
        )
    rows: list[dict[str, str]] = []
    for freight_beam_category, emfac_categories in vehicle_categories.items():
        beam_category = _normalize_text(freight_beam_category)
        if beam_category == "":
            continue
        if not isinstance(emfac_categories, list):
            raise ValueError(
                "Freight Bayesian DAG evidence.vehicle_categories entries must be lists of EMFAC vehicle-category strings."
            )
        for emfac_category in emfac_categories:
            vehicle_category = _normalize_text(emfac_category)
            if vehicle_category == "":
                continue
            rows.append({"freight_beam_category": beam_category, "emfac": vehicle_category})
    prepared = pd.DataFrame(rows)
    if prepared.empty:
        raise ValueError(
            "Freight Bayesian DAG evidence.vehicle_categories produced no freight category mapping rows."
        )
    return prepared.drop_duplicates().reset_index(drop=True)


def _load_freight_fuel_mapping(config: dict[str, Any]) -> pd.DataFrame:
    freight_model = config.get("freight_bayesian_dag", {}) or {}
    vehicle_categories = freight_model.get("vehicle_categories", {})
    fuel_types = freight_model.get("fuel_types", {})
    category_fuel_support = freight_model.get("category_fuel_support", {})
    if not isinstance(vehicle_categories, dict) or not vehicle_categories:
        raise ValueError(
            "Freight Bayesian DAG model is missing evidence.vehicle_categories required for freight fuel mapping."
        )
    if not isinstance(fuel_types, dict) or not fuel_types:
        raise ValueError(
            "Freight Bayesian DAG model is missing evidence.fuel_types required for freight fuel mapping."
        )
    rows: list[dict[str, str]] = []
    for freight_beam_category, emfac_categories in vehicle_categories.items():
        beam_category = _normalize_text(freight_beam_category)
        if beam_category == "":
            continue
        if not isinstance(emfac_categories, list):
            raise ValueError(
                "Freight Bayesian DAG evidence.vehicle_categories entries must be lists of EMFAC vehicle-category strings."
            )
        for emfac_category in emfac_categories:
            vehicle_category = _normalize_text(emfac_category)
            if vehicle_category == "":
                continue
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
                        "Freight Bayesian DAG evidence.fuel_types entries must be lists of EMFAC fuel strings."
                    )
                for emfac_fuel in emfac_fuels:
                    fuel = _normalize_text(emfac_fuel)
                    if fuel == "" or fuel not in supported_fuels:
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
            "Freight Bayesian DAG evidence.vehicle_categories and evidence.fuel_types produced no freight fuel mapping rows."
        )
    return prepared.drop_duplicates().reset_index(drop=True)


def _normalize_to_fastsim_adopt_fuel(*, fuel_domain: str, adopt_fuel: object) -> str:
    token = _normalize_lower(adopt_fuel)
    if fuel_domain == "ldv":
        mapping = {
            "gasoline": "gasoline",
            "diesel": "diesel",
            "biodiesel": "diesel",
            "cng": "naturalgas",
            "naturalgas": "naturalgas",
            "electricity": "electricity",
            "hydrogen": "hydrogen",
            "electricity+gasoline": "electricity",
            "electricity+diesel": "electricity",
        }
        return mapping.get(token, token)
    mapping = {
        "diesel": "diesel",
        "gasoline": "gasoline",
        "biodiesel": "diesel",
        "cng": "naturalgas",
        "naturalgas": "naturalgas",
        "electricity": "electricity",
        "hydrogen": "hydrogen",
        "electricity+gasoline": "electricity",
        "electricity+diesel": "electricity",
    }
    return mapping.get(token, token)


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
    models = model_spec.get("models", {})
    model_section = models.get("freight_bayesian_dag", {}) if isinstance(models, dict) else {}
    evidence = model_section.get("evidence", {})
    if evidence_source == "naics_sector":
        rows = evidence.get("naics_sector", [])
        if not isinstance(rows, list):
            raise ValueError(
                "Vehicle type assignment model file evidence.naics_sector must be a list."
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
        port_evidence = evidence.get("port_location", [])
        if not isinstance(port_evidence, list):
            raise ValueError(
                "Vehicle type assignment model file evidence.port_location must be a list."
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
    joined["oldVehicleTypeId"] = joined["vehicleTypeId"].map(_normalize_freight_vehicle_type_id)
    if "locationZone" in joined.columns:
        joined["locationZone"] = joined["locationZone"].map(_normalize_zone_id)
    return joined[
        [
            "tourId",
            "oldVehicleTypeId",
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


def _log_geometric_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(0.0, index=frame.index, dtype="float64")
    log_terms = [np.log(pd.to_numeric(frame[column], errors="coerce").fillna(1.0).clip(lower=1e-9)) for column in columns]
    return sum(log_terms) / float(len(log_terms))


def _normalize_branch_weights(branch_weights: dict[str, float]) -> dict[str, float]:
    normalized = {key: max(0.0, float(value)) for key, value in branch_weights.items()}
    total = float(sum(normalized.values()))
    if total <= 0.0:
        uniform = 1.0 / float(len(normalized)) if normalized else 1.0
        return {key: uniform for key in normalized}
    return {key: value / total for key, value in normalized.items()}


def _build_dag_log_score(
    *,
    matched: pd.DataFrame,
    branch_weights: dict[str, float],
) -> pd.Series:
    """Build a grouped Bayesian-network score in log space.

    The assignment DAG is:

      vehicle category prior -> category
      category -> naics sector evidence
      category -> mass evidence
      category -> port evidence

    Mass and port remain separate nodes. The NAICS-sector branch is currently
    the only business-context evidence source, so it is scored directly.

    The final score is a weighted geometric mean over the prior, NAICS-sector,
    mass, and port branches. Branch coefficients are normalized to sum to 1 so
    every coefficient lives on the same scale and no branch gets implicit extra
    influence from coefficient magnitude alone.
    """
    weights = _normalize_branch_weights(branch_weights)
    fleet_share = pd.to_numeric(matched["fleetShare"], errors="coerce").fillna(0.0)
    prior_log = np.log(fleet_share.clip(lower=1e-9))
    naics_sector_log = np.log(pd.to_numeric(matched["naicsSectorLikelihood"], errors="coerce").fillna(1.0).clip(lower=1e-9))
    mass_log = np.log(pd.to_numeric(matched["massLikelihood"], errors="coerce").fillna(1.0).clip(lower=1e-9))
    port_log = np.log(pd.to_numeric(matched["portLikelihood"], errors="coerce").fillna(1.0).clip(lower=1e-9))
    return (
        (float(weights["prior"]) * prior_log)
        + (float(weights["naics_sector"]) * naics_sector_log)
        + (float(weights["mass"]) * mass_log)
        + (float(weights["port"]) * port_log)
    )


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


def _build_freight_naics_weight_lookup(
    *,
    payload_profiles: pd.DataFrame,
    sector_mapping: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Return P(NAICS signal | vehicle_type) for each old vehicle type id.

    The returned dictionary maps oldVehicleTypeId to a normalized conditional
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
    for old_vehicle_type_id, group in payload_profiles.groupby("oldVehicleTypeId", sort=False):
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
    required_columns = ["tourId", "oldVehicleTypeId", "sequenceRank", "activityType", "weightInKg"]
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
    for (old_vehicle_type_id, tour_id), group in working.groupby(["oldVehicleTypeId", "tourId"], sort=False):
        ordered = group.sort_values(by=["sequenceRank"], kind="mergesort").copy()
        ordered["rawOnboardKg"] = ordered["signedChangeKg"].cumsum()
        min_onboard = float(ordered["rawOnboardKg"].min()) if not ordered.empty else 0.0
        initial_onboard = max(0.0, -min_onboard)
        ordered["onboardKg"] = ordered["rawOnboardKg"] + initial_onboard
        peak_onboard = float(ordered["onboardKg"].max()) if not ordered.empty else 0.0
        final_onboard = float(ordered["onboardKg"].iloc[-1]) if not ordered.empty else 0.0
        tour_peaks.append(
            {
                "oldVehicleTypeId": str(old_vehicle_type_id),
                "tourId": str(tour_id),
                "peakOnboardKg": peak_onboard,
                "finalOnboardKg": final_onboard,
            }
        )

    if not tour_peaks:
        return {}

    peaks = pd.DataFrame(tour_peaks)
    for old_vehicle_type_id, group in peaks.groupby("oldVehicleTypeId", sort=False):
        numeric = pd.to_numeric(group["peakOnboardKg"], errors="coerce").dropna()
        if numeric.empty:
            continue
        light_threshold = float(numeric.quantile(0.5))
        heavy_threshold = float(numeric.quantile(0.9))
        thresholds[str(old_vehicle_type_id)] = (light_threshold, heavy_threshold)
    return thresholds


def _build_freight_port_weight_lookup(
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
    for old_vehicle_type_id, group in payload_profiles.groupby("oldVehicleTypeId", sort=False):
        matched = (
            group["locationZone"]
            .map(lambda value: zone_to_port_class.get(_normalize_zone_id(value), ""))
            .replace("", pd.NA)
            .dropna()
        )
        if matched.empty:
            continue
        lookup[str(old_vehicle_type_id)] = {str(key): 1.0 for key in matched.unique().tolist()}
    return lookup


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
    adopt_fuel = _normalize_to_fastsim_adopt_fuel(fuel_domain="mhdv", adopt_fuel=adopt_fuel)
    base_matches = fuel_mapping[
        fuel_mapping["adopt_fuel"] == adopt_fuel
    ]

    return {str(row.emfac_fuel) for row in base_matches.itertuples(index=False)}


def _build_freight_emfac_candidates(
    *,
    vehicle_type_id: str,
    beam_vehicle_category: str,
    fuel_domain: str,
    adopt_fuel: object,
    emfac_candidates: pd.DataFrame,
    category_mapping: pd.DataFrame,
    fuel_mapping: pd.DataFrame,
    naics_sector_weights: dict[str, float],
    port_weights: dict[str, float],
    median_mass_kg: float = 0.0,
    heavy_mass_kg: float = 0.0,
    likelihood_floor: float,
    prior_vmt_share: float,
    naics_sector: float,
    port_location: float,
    configured_port_classes: set[str] | None = None,
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
    matched = _filter_required_port_classes(
        matched=matched,
        port_weights=port_weights,
        configured_port_classes=configured_port_classes or set(),
    )
    if matched.empty:
        raise ValueError(
            "No freight EMFAC candidates remain after enforcing port-class matching for "
            f"vehicleTypeId={vehicle_type_id}"
        )

    # Signal likelihoods — P(signal | type_i) for each soft evidence signal.
    # When a signal is observed (lookup is non-empty), absent categories receive
    # likelihood_floor so the score is penalised but not zeroed.
    # When a signal is absent (no observation), the factor is marginalized out
    # by using a uniform likelihood of 1.0 across all candidates.
    matched["naicsSectorLikelihood"] = matched["vehicleCategory"].map(naics_sector_weights).fillna(likelihood_floor) if naics_sector_weights else 1.0
    matched["massLikelihood"] = matched["vehicleCategory"].map(
        lambda vc: _payload_mass_category_weight(vehicle_category=str(vc), median_mass_kg=median_mass_kg, heavy_mass_kg=heavy_mass_kg)
    )
    matched["portLikelihood"] = matched["vehicleCategory"].map(
        lambda vc: _port_category_weight(
            vehicle_category=str(vc),
            port_weights=port_weights,
            configured_port_classes=configured_port_classes or set(),
            likelihood_floor=likelihood_floor,
        )
    )

    fleet_share = pd.to_numeric(matched["fleetShare"], errors="coerce").fillna(0.0)
    matched = matched[fleet_share.gt(0)].copy()
    if matched.empty:
        raise ValueError(f"No freight EMFAC candidates have positive fleetShare for vehicleTypeId={vehicle_type_id}")

    log_score = _build_dag_log_score(
        matched=matched,
        branch_weights={
            "prior": prior_vmt_share,
            "naics_sector": naics_sector,
            "mass": 1.0,
            "port": port_location,
        },
    )
    matched["score"] = np.exp(log_score - log_score.max())
    return matched.sort_values(
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
    config: dict[str, Any],
    source_frism_carriers: pd.DataFrame,
    source_frism_payloads: pd.DataFrame,
) -> pd.DataFrame:
    for column_name in ["vehicleTypeId", "adopt_fuel", "sampleProbabilityWithinCategory"]:
        _require_column(freight_vehicle_types, column_name, "Freight vehicle types file")

    emfac_candidates = _build_valid_freight_emfac_candidates(config)
    target_share_lookup = _build_emfac_target_share_lookup(
        candidates=emfac_candidates,
        population_bias=float(config.get("vehicle_type_assignment", {}).get("emfac_population_bias", 1.0)),
        vmt_bias=float(config.get("vehicle_type_assignment", {}).get("emfac_vmt_bias", 0.0)),
    )
    category_mapping = _load_freight_category_mapping(config)
    fuel_mapping = _load_freight_fuel_mapping(config)
    naics_sector_mapping = _load_naics_sector_mapping(config)
    port_zone_mapping = _load_port_zone_mapping(config)
    payload_profiles = _build_freight_payload_profiles(
        carriers=source_frism_carriers,
        payloads=source_frism_payloads,
    )
    naics_sector_weight_lookup = _build_freight_naics_sector_weight_lookup(
        payload_profiles=payload_profiles,
        sector_mapping=naics_sector_mapping,
    )
    port_weight_lookup = _build_freight_port_weight_lookup(
        payload_profiles=payload_profiles,
        port_zone_mapping=port_zone_mapping,
    )
    payload_mass_thresholds = _build_payload_mass_thresholds(payload_profiles)
    assignment_config = config.get("vehicle_type_assignment", {})
    if "likelihood_floor" not in assignment_config:
        raise ValueError("vehicle_type_assignment.likelihood_floor must be configured from the model YAML")
    likelihood_floor = float(assignment_config["likelihood_floor"])
    for key in ("prior_vmt_share", "naics_sector", "port_location"):
        if key not in assignment_config:
            raise ValueError(f"vehicle_type_assignment.{key} must be configured from the model YAML")
    prior_vmt_share = float(assignment_config["prior_vmt_share"])
    naics_sector = float(assignment_config["naics_sector"])
    port_location = float(assignment_config["port_location"])
    configured_port_classes = _load_configured_port_classes(config)

    prepared = freight_vehicle_types.copy()
    prepared["freight_beam_category"] = prepared.apply(
        lambda row: _map_freight_beam_vehicle_category(row["vehicleTypeId"], row.get("vehicleCategory"), row.get("vehicleClass")),
        axis=1,
    )

    expanded_rows: list[dict[str, Any]] = []
    pending_candidates: list[dict[str, Any]] = []
    for source_row_id, row in enumerate(prepared.itertuples(index=False)):
        median_mass_kg, heavy_mass_kg = payload_mass_thresholds.get(str(row.vehicleTypeId), (0.0, 0.0))
        candidates = _build_freight_emfac_candidates(
            vehicle_type_id=str(row.vehicleTypeId),
            beam_vehicle_category=str(row.freight_beam_category),
            fuel_domain="ldv" if str(row.freight_beam_category) in {"Class12aVocational", "Class2b3Vocational"} else "mhdv",
            adopt_fuel=row.adopt_fuel,
            emfac_candidates=emfac_candidates,
            category_mapping=category_mapping,
            fuel_mapping=fuel_mapping,
            naics_sector_weights=naics_sector_weight_lookup.get(str(row.vehicleTypeId), {}),
            port_weights=port_weight_lookup.get(str(row.vehicleTypeId), {}),
            median_mass_kg=median_mass_kg,
            heavy_mass_kg=heavy_mass_kg,
            likelihood_floor=likelihood_floor,
            prior_vmt_share=prior_vmt_share,
            naics_sector=naics_sector,
            port_location=port_location,
            configured_port_classes=configured_port_classes,
        )
        base_probability = float(pd.to_numeric(pd.Series([row.sampleProbabilityWithinCategory]), errors="coerce").fillna(0.0).iloc[0])
        row_payload = row._asdict()
        for candidate in candidates.itertuples(index=False):
            pending_candidates.append(
                {
                    "source_row_id": int(source_row_id),
                    "base_probability": float(base_probability),
                    "local_score": float(candidate.score),
                    "row_payload": row_payload,
                    "oldVehicleTypeId": str(row.vehicleTypeId),
                    "vehicleCategory": str(row.freight_beam_category),
                    "emfacId": str(candidate.emfacId),
                    "emfacVehicleCategory": str(candidate.vehicleCategory),
                }
            )

    if pending_candidates:
        calibrated = _apply_global_emfac_target_balancing(
            candidate_rows=pd.DataFrame(pending_candidates),
            target_share_lookup=target_share_lookup,
        )
        for candidate in calibrated.itertuples(index=False):
            share = float(candidate.probabilityShare)
            updated = dict(candidate.row_payload)
            updated["oldVehicleTypeId"] = str(candidate.oldVehicleTypeId)
            updated["vehicleCategory"] = str(candidate.vehicleCategory)
            updated["emfacId"] = str(candidate.emfacId)
            updated["emfacVehicleCategory"] = str(candidate.emfacVehicleCategory)
            updated["vehicleTypeId"] = f"{candidate.emfacId}--{candidate.oldVehicleTypeId}"
            updated["sampleProbabilityWithinCategory"] = float(candidate.base_probability) * share
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


def _build_freight_sampling_table(mapped_freight_vehicle_types: pd.DataFrame) -> dict[str, pd.DataFrame]:
    _require_column(mapped_freight_vehicle_types, "vehicleTypeId", "Mapped freight vehicle types file")
    _require_column(mapped_freight_vehicle_types, "oldVehicleTypeId", "Mapped freight vehicle types file")
    _require_column(mapped_freight_vehicle_types, "sampleProbabilityWithinCategory", "Mapped freight vehicle types file")
    _require_column(mapped_freight_vehicle_types, "emfacVehicleCategory", "Mapped freight vehicle types file")

    prepared = mapped_freight_vehicle_types[
        ["vehicleTypeId", "oldVehicleTypeId", "emfacVehicleCategory", "sampleProbabilityWithinCategory"]
    ].copy()
    prepared["sampleProbabilityWithinCategory"] = pd.to_numeric(
        prepared["sampleProbabilityWithinCategory"], errors="coerce"
    ).fillna(0.0)
    sampling_groups: dict[str, pd.DataFrame] = {}
    for old_vehicle_type_id, group in prepared.groupby("oldVehicleTypeId", sort=False):
        group_prepared = group.rename(columns={"emfacVehicleCategory": "vehicleCategory"})[
            ["vehicleTypeId", "vehicleCategory", "sampleProbabilityWithinCategory"]
        ].reset_index(drop=True)
        probability_sum = group_prepared["sampleProbabilityWithinCategory"].sum()
        if probability_sum <= 0:
            group_prepared["samplingProbability"] = 1.0 / len(group_prepared)
        else:
            group_prepared["samplingProbability"] = group_prepared["sampleProbabilityWithinCategory"] / probability_sum
        sampling_groups[str(old_vehicle_type_id)] = group_prepared[["vehicleTypeId", "vehicleCategory", "samplingProbability"]]
    return sampling_groups


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
    payload_profiles = _build_freight_payload_profiles(carriers=carriers, payloads=payloads)
    port_zone_mapping = _load_port_zone_mapping(config)
    configured_port_classes = _load_configured_port_classes(config)
    tour_port_weight_lookup = _build_tour_port_weight_lookup(
        payload_profiles=payload_profiles,
        port_zone_mapping=port_zone_mapping,
    )
    prepared_carriers = carriers.copy()
    prepared_carriers["oldVehicleTypeId"] = prepared_carriers["vehicleTypeId"].map(_normalize_freight_vehicle_type_id)

    sampled_vehicle_type_ids = pd.Series(index=prepared_carriers.index, dtype="object")
    random = np.random.default_rng(int(seed))
    for row in prepared_carriers.itertuples():
        candidates = sampling_groups.get(str(row.oldVehicleTypeId))
        if candidates is None or candidates.empty:
            raise ValueError(f"No mapped freight vehicle types available for oldVehicleTypeId={row.oldVehicleTypeId}")
        filtered = _filter_required_port_classes(
            matched=candidates,
            port_weights=tour_port_weight_lookup.get(str(row.tourId), {}),
            configured_port_classes=configured_port_classes,
        )
        if filtered.empty:
            filtered = candidates
        probabilities = pd.to_numeric(filtered["samplingProbability"], errors="coerce").fillna(0.0)
        probability_sum = float(probabilities.sum())
        if probability_sum <= 0.0:
            probabilities = np.full(len(filtered), 1.0 / len(filtered))
        else:
            probabilities = (probabilities / probability_sum).to_numpy()
        sampled_vehicle_type_ids.loc[row.Index] = random.choice(
            filtered["vehicleTypeId"].to_numpy(),
            size=1,
            p=probabilities,
        )[0]
    prepared_carriers["vehicleTypeId"] = sampled_vehicle_type_ids.astype(str)
    prepared_carriers = prepared_carriers.drop(columns=["oldVehicleTypeId"], errors="ignore")

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


def _run_step4_substep_map_freight_vehicle_types(
    workflow: dict[str, Any],
) -> tuple[pd.DataFrame, str, pd.DataFrame, pd.DataFrame]:
    freight_vehicle_types_file = workflow.get("built_freight_vehicle_types_file")
    if not freight_vehicle_types_file:
        raise ValueError("Step 4 requires freight vehicle types from Step 1")
    freight_vehicle_types = read_table(str(freight_vehicle_types_file), dtype=None)
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
    return mapped_freight_vehicle_types, mapped_freight_vehicle_types_file, carriers, payloads


def _run_step4_substep_map_carriers(
    *,
    workflow: dict[str, Any],
    carriers: pd.DataFrame,
    payloads: pd.DataFrame,
    mapped_freight_vehicle_types: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    tours = workflow.get("source_frism_tours")
    if tours is None:
        tours = read_table(workflow["config"]["frism"]["tours_file"], dtype=None)
    mapped_carriers, mapped_tours = _map_freight_carriers_and_tours(
        carriers=carriers,
        tours=tours,
        payloads=payloads,
        mapped_freight_vehicle_types=mapped_freight_vehicle_types,
        config=workflow["config"],
        seed=int(workflow["config"]["seed"]),
    )
    output_dir = Path(workflow["config"]["output"])
    carriers_source_path = Path(resolve_workflow_path(workflow["config"]["frism"]["carriers_file"]))
    mapped_carriers_file = _write_parquet(
        mapped_carriers,
        str(output_dir / f"{carriers_source_path.stem}--EM.parquet"),
    )
    return mapped_carriers, mapped_carriers_file


def run_step4(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 4: map EMFAC freight distributions onto FRISM vehicle types and tours."""
    print("=== Step 4.1: map emfacId to freight vehicle types ===")
    mapped_freight_vehicle_types, mapped_freight_vehicle_types_file, carriers, payloads = (
        _run_step4_substep_map_freight_vehicle_types(workflow)
    )
    workflow["built_freight_vehicle_types"] = mapped_freight_vehicle_types
    workflow["built_freight_vehicle_types_file"] = mapped_freight_vehicle_types_file

    print("=== Step 4.2: distribute freight vehicleTypeId to FRISM carriers and tours ===")
    mapped_carriers, mapped_carriers_file = _run_step4_substep_map_carriers(
        workflow=workflow,
        carriers=carriers,
        payloads=payloads,
        mapped_freight_vehicle_types=mapped_freight_vehicle_types,
    )
    workflow["mapped_freight_carriers"] = mapped_carriers
    workflow["mapped_freight_carriers_file"] = mapped_carriers_file
    return workflow
