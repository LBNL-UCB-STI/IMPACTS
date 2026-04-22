from __future__ import annotations

from pathlib import Path
import re
from typing import Optional

import numpy as np
import pandas as pd

from ...common import read_table
from ...config.defaults import grams_per_short_ton
from ...config.defaults import meters_per_mile as _METERS_PER_MILE
from ...config.defaults import pollutants as default_prepared_pollutants
from ...config.defaults import representative_days_per_year as default_representative_days_per_year


_SKIMS_DIMENSION_COLS = {
    "linkId",
    "vehicleTypeId",
    "process",
    "totTrips",
    "totVMT",
    "roadCategory",
}


_TRANSIT_VEHICLETYPE_PATTERN = re.compile(
    r"(^|[-_])(BUS|RAIL|FERRY|SUBWAY|TRAM|TRAIN|COACH)($|[-_])",
    re.IGNORECASE,
)


def _is_transit_vehicle_type(vehicle_type_id: object) -> bool:
    token = str("" if pd.isna(vehicle_type_id) else vehicle_type_id).strip()
    return bool(token) and bool(_TRANSIT_VEHICLETYPE_PATTERN.search(token))


def _build_skims_scale_factors(
    prepared: pd.DataFrame,
    *,
    population_sample: float,
    transit_sample: float,
) -> pd.Series:
    if not 0 < population_sample <= 1:
        raise ValueError(f"population_sample must be in the interval (0, 1], got {population_sample}")
    if not 0 < transit_sample <= 1:
        raise ValueError(f"transit_sample must be in the interval (0, 1], got {transit_sample}")

    scale_factors = pd.Series(
        np.full(len(prepared), 1.0 / population_sample, dtype=float),
        index=prepared.index,
    )
    if "vehicleTypeId" not in prepared.columns:
        return scale_factors

    transit_mask = prepared["vehicleTypeId"].map(_is_transit_vehicle_type)
    if transit_mask.any():
        scale_factors.loc[transit_mask] = 1.0 / transit_sample
    return scale_factors


def annualize_prepared_skims_for_grid_allocation(
    prepared_skims_path: str,
    output_path: str,
    *,
    network_path: str,
    beam_length_col: str,
    group_cols: Optional[list[str]] = None,
    required_pollutants: Optional[list[str]] = None,
    annualization_days_or_file: float | str = default_representative_days_per_year,
    passenger_vehicle_types_path: Optional[str] = None,
    freight_vehicle_types_path: Optional[str] = None,
    population_sample: float = 1.0,
    transit_sample: float = 1.0,
) -> pd.DataFrame:
    prepared = read_table(prepared_skims_path)
    link_lengths = read_table(network_path)
    prepared_group_cols = group_cols or ["linkId", "vehicleTypeId", "process"]
    required = required_pollutants or default_prepared_pollutants
    missing_group_cols = [col for col in prepared_group_cols if col not in prepared.columns]
    if missing_group_cols:
        raise ValueError(f"Annualized skims missing required grouping columns: {missing_group_cols}")

    if "linkId" not in link_lengths.columns or beam_length_col not in link_lengths.columns:
        raise ValueError(
            f"Link lengths table must include 'linkId' and '{beam_length_col}'."
        )
    network_cols = ["linkId", beam_length_col]
    if "attributeOrigType" in link_lengths.columns:
        network_cols.append("attributeOrigType")
    link_lengths = link_lengths[network_cols].copy()
    link_lengths[beam_length_col] = pd.to_numeric(link_lengths[beam_length_col], errors="coerce").fillna(0.0)
    prepared = prepared.merge(link_lengths, how="left", on="linkId")
    prepared[beam_length_col] = pd.to_numeric(prepared[beam_length_col], errors="coerce").fillna(0.0)
    if "attributeOrigType" in prepared.columns:
        prepared = prepared.rename(columns={"attributeOrigType": "roadCategory"})
    scale_factors = _build_skims_scale_factors(
        prepared,
        population_sample=population_sample,
        transit_sample=transit_sample,
    )
    annualization_factors = resolve_skims_annualization_factors(
        prepared,
        annualization_days_or_file=annualization_days_or_file,
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
    )

    retained_dim_cols = [col for col in prepared.columns if col in _SKIMS_DIMENSION_COLS and col not in prepared_group_cols]
    out = prepared[prepared_group_cols + retained_dim_cols].copy()
    prepared_observations = pd.to_numeric(prepared.get("observations", 0.0), errors="coerce").fillna(0.0)
    out["totTrips"] = prepared_observations * scale_factors * annualization_factors
    out["totVMT"] = out["totTrips"] * prepared[beam_length_col] / _METERS_PER_MILE
    out = out.drop(columns=[col for col in [beam_length_col] if col in out.columns], errors="ignore")
    for pollutant in required:
        values = (
            pd.to_numeric(prepared[pollutant], errors="coerce").fillna(0.0)
            if pollutant in prepared.columns
            else pd.Series(np.zeros(len(prepared), dtype=float), index=prepared.index)
        )
        out[f"tons_per_year_{pollutant}"] = values * scale_factors * annualization_factors / grams_per_short_ton

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        out.to_parquet(output, index=False)
    elif output.name.lower().endswith(".csv.gz"):
        out.to_csv(output, index=False, compression="gzip")
    else:
        raise ValueError("Annualized skims output must be .parquet or .csv.gz")
    return out


def _sanitize_emfac_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str("" if pd.isna(value) else value).strip())


def _load_vehicle_operation_days_lookup(csv_path: str) -> tuple[dict[str, float], list[tuple[str, str]]]:
    frame = read_table(csv_path)
    required = {"vehicleCategory", "operation_days_per_year"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Vehicle operation days CSV is missing required columns: {missing}")
    lookup: dict[str, float] = {}
    sanitized_categories: list[tuple[str, str]] = []
    for row in frame[["vehicleCategory", "operation_days_per_year"]].itertuples(index=False):
        category = str(row.vehicleCategory).strip()
        if not category:
            continue
        days = float(row.operation_days_per_year)
        if days <= 0:
            raise ValueError(f"Operation days must be positive for vehicleCategory={category!r}")
        lookup[category] = days
        sanitized_categories.append((category, _sanitize_emfac_token(category)))
    sanitized_categories.sort(key=lambda item: len(item[1]), reverse=True)
    return lookup, sanitized_categories


def _infer_emfac_vehicle_category_from_emfac_id(
    emfac_id: object,
    *,
    sanitized_categories: list[tuple[str, str]],
) -> str:
    token = str("" if pd.isna(emfac_id) else emfac_id).strip()
    if not token:
        raise ValueError("Encountered blank emfacId while resolving annualization days.")
    sanitized_emfac_id = _sanitize_emfac_token(token)
    matches = [category for category, sanitized in sanitized_categories if sanitized and sanitized in sanitized_emfac_id]
    if not matches:
        raise ValueError(
            "Could not infer EMFAC vehicle category from emfacId="
            f"{token!r} using the configured annualization CSV."
        )
    return matches[0]


def _load_vehicle_type_category_lookup(
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
    *,
    category_lookup: dict[str, float],
    sanitized_categories: list[tuple[str, str]],
) -> dict[str, str]:
    passenger = read_table(passenger_vehicle_types_path).copy()
    freight = read_table(freight_vehicle_types_path).copy()
    vehicle_types = pd.concat([passenger, freight], ignore_index=True, sort=False)
    if "vehicleTypeId" not in vehicle_types.columns:
        raise ValueError("Vehicle types input must include vehicleTypeId for annualization lookup.")

    prepared = vehicle_types.copy()
    prepared["vehicleTypeId"] = prepared["vehicleTypeId"].astype(str).str.strip()
    prepared = prepared.loc[prepared["vehicleTypeId"].ne("")].copy()

    for candidate in ("emfacVehicleCategory", "vehicleCategory"):
        if candidate not in prepared.columns:
            continue
        prepared[candidate] = prepared[candidate].where(prepared[candidate].notna(), "")
        prepared[candidate] = prepared[candidate].astype(str).str.strip()
        category_rows = prepared.loc[
            prepared[candidate].ne("") & ~prepared[candidate].str.lower().eq("nan")
        ].copy()
        if category_rows.empty:
            continue
        missing_categories = sorted(set(category_rows[candidate]) - set(category_lookup))
        if missing_categories:
            raise ValueError(
                "Vehicle types input contains EMFAC categories not present in the configured annualization CSV: "
                f"{missing_categories[:10]}"
            )
        return (
            category_rows[["vehicleTypeId", candidate]]
            .drop_duplicates(subset=["vehicleTypeId"], keep="first")
            .set_index("vehicleTypeId")[candidate]
            .to_dict()
        )

    if "emfacId" not in prepared.columns:
        raise ValueError(
            "Vehicle types input must include emfacVehicleCategory or emfacId when annualization_days_or_file is a CSV path."
        )

    prepared["resolved_emfac_category"] = prepared["emfacId"].map(
        lambda value: _infer_emfac_vehicle_category_from_emfac_id(
            value,
            sanitized_categories=sanitized_categories,
        )
    )
    return (
        prepared[["vehicleTypeId", "resolved_emfac_category"]]
        .drop_duplicates(subset=["vehicleTypeId"], keep="first")
        .set_index("vehicleTypeId")["resolved_emfac_category"]
        .to_dict()
    )


def resolve_skims_annualization_factors(
    prepared: pd.DataFrame,
    *,
    annualization_days_or_file: float | str,
    passenger_vehicle_types_path: Optional[str] = None,
    freight_vehicle_types_path: Optional[str] = None,
) -> pd.Series:
    if isinstance(annualization_days_or_file, str):
        csv_path = str(annualization_days_or_file).strip()
        if not csv_path:
            raise ValueError("Annualization days file path must be non-empty.")
        if "vehicleTypeId" not in prepared.columns:
            raise ValueError("Prepared skims must include vehicleTypeId when annualization_days_or_file is a CSV path.")
        if not passenger_vehicle_types_path or not freight_vehicle_types_path:
            raise ValueError("A vehicle types input is required when annualization_days_or_file is a CSV path.")
        category_lookup, sanitized_categories = _load_vehicle_operation_days_lookup(csv_path)
        vehicle_type_category_lookup = _load_vehicle_type_category_lookup(
            passenger_vehicle_types_path,
            freight_vehicle_types_path,
            category_lookup=category_lookup,
            sanitized_categories=sanitized_categories,
        )
        categories = prepared["vehicleTypeId"].astype(str).map(vehicle_type_category_lookup)
        missing_vehicle_types = (
            prepared.loc[categories.isna(), "vehicleTypeId"]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        if missing_vehicle_types:
            raise ValueError(
                "Could not resolve EMFAC vehicle category for some skim vehicleTypeId values using "
                "the configured passenger/freight vehicle types files: "
                f"sample={missing_vehicle_types[:10]}"
            )
        return categories.map(category_lookup).astype(float)
    days = float(annualization_days_or_file)
    if days <= 0:
        raise ValueError(f"Annualization days must be positive, got {annualization_days_or_file}")
    return pd.Series(np.full(len(prepared), days, dtype=float), index=prepared.index)
