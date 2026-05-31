from __future__ import annotations

import logging
from pathlib import Path
import re
import time
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from ...common import _configure_duckdb_progress_bar
from ...common import _should_show_duckdb_progress_bar
from ...common import configure_duckdb_connection
from ...common import parquet_row_count
from ...common import read_table
from ...config.defaults import annualization_days_by_vehicle_group as default_annualization_days_by_vehicle_group
from ...config.defaults import grams_per_short_ton
from ...config.defaults import meters_per_mile as _METERS_PER_MILE
from ...config.defaults import pollutants as default_prepared_pollutants

logger = logging.getLogger(__name__)


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
    vehicle_category_metadata_file: Optional[str] = None,
    annualization_days: Optional[dict[str, float]] = None,
    passenger_vehicle_types_path: Optional[str] = None,
    freight_vehicle_types_path: Optional[str] = None,
    population_sample: float = 1.0,
    transit_sample: float = 1.0,
) -> pd.DataFrame:
    started = time.perf_counter()
    link_lengths = read_table(network_path)
    prepared_group_cols = group_cols or ["linkId", "vehicleTypeId", "process"]
    required = required_pollutants or default_prepared_pollutants

    if "linkId" not in link_lengths.columns or beam_length_col not in link_lengths.columns:
        raise ValueError(
            f"Link lengths table must include 'linkId' and '{beam_length_col}'."
        )
    network_cols = ["linkId", beam_length_col]
    if "attributeOrigType" in link_lengths.columns:
        network_cols.append("attributeOrigType")
    link_lengths = link_lengths[network_cols].copy()
    link_lengths[beam_length_col] = pd.to_numeric(link_lengths[beam_length_col], errors="coerce").fillna(0.0)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    if Path(prepared_skims_path).suffix.lower() != ".parquet":
        raise ValueError("Annualized skims input must be .parquet")
    if output.suffix.lower() != ".parquet":
        raise ValueError("Annualized skims output must be .parquet")
    annualization_days_lookup = _resolve_vehicle_type_annualization_days_lookup(
        vehicle_category_metadata_file=vehicle_category_metadata_file,
        annualization_days=annualization_days,
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
    )
    tailpipe_height_lookup = _resolve_vehicle_type_tailpipe_height_lookup(
        vehicle_category_metadata_file=vehicle_category_metadata_file,
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
    )
    return _annualize_prepared_skims_with_duckdb(
        prepared_skims_path=prepared_skims_path,
        output_path=output,
        prepared_group_cols=prepared_group_cols,
        required_pollutants=required,
        link_lengths=link_lengths,
        beam_length_col=beam_length_col,
        annualization_days_lookup=annualization_days_lookup,
        tailpipe_height_lookup=tailpipe_height_lookup,
        population_sample=population_sample,
        transit_sample=transit_sample,
        started=started,
    )


def _duckdb_scan_expression(path: str | Path) -> str:
    target = Path(path)
    if target.suffix.lower() != ".parquet":
        raise ValueError("Annualized skims input must be .parquet")
    path_sql = str(target).replace("'", "''")
    return f"read_parquet('{path_sql}')"


def _annualize_prepared_skims_with_duckdb(
    *,
    prepared_skims_path: str,
    output_path: Path,
    prepared_group_cols: list[str],
    required_pollutants: list[str],
    link_lengths: pd.DataFrame,
    beam_length_col: str,
    annualization_days_lookup: dict[str, float],
    tailpipe_height_lookup: dict[str, float],
    population_sample: float,
    transit_sample: float,
    started: float,
) -> pd.DataFrame:
    scan = _duckdb_scan_expression(prepared_skims_path)
    con = duckdb.connect(database=":memory:")
    show_progress = _should_show_duckdb_progress_bar()
    lookup_df = pd.DataFrame(
        {
            "vehicleTypeId": list(annualization_days_lookup.keys()),
            "annualization_days": list(annualization_days_lookup.values()),
        }
    )
    has_height_lookup = bool(tailpipe_height_lookup)
    height_lookup_df = pd.DataFrame(
        {
            "vehicleTypeId": list(tailpipe_height_lookup.keys()),
            "tailpipe_height_meters": list(tailpipe_height_lookup.values()),
        }
    ) if has_height_lookup else None
    output_columns = list(prepared_group_cols)
    if "attributeOrigType" in link_lengths.columns:
        output_columns.append("roadCategory")
    if has_height_lookup:
        output_columns.append("source_release_height")
    output_columns.extend(["totTrips", "totVMT"])
    output_columns.extend([f"tons_per_year_{pollutant}" for pollutant in required_pollutants])
    try:
        configure_duckdb_connection(con, working_dir=output_path, show_progress=False, profile="balanced")
        con.register("link_lengths", link_lengths)
        con.register("annualization_lookup", lookup_df)
        if has_height_lookup:
            con.register("height_lookup", height_lookup_df)
        missing_vehicle_types = [
            row[0]
            for row in con.execute(
                f"""
                SELECT DISTINCT trim(CAST(source.vehicleTypeId AS VARCHAR)) AS vehicleTypeId
                FROM {scan} AS source
                LEFT JOIN annualization_lookup AS lookup
                  ON trim(CAST(source.vehicleTypeId AS VARCHAR)) = lookup.vehicleTypeId
                WHERE lookup.annualization_days IS NULL
                LIMIT 10
                """
            ).fetchall()
            if row[0]
        ]
        if missing_vehicle_types:
            raise ValueError(
                "Could not resolve annualization days for some skim vehicleTypeId values using "
                "the configured passenger/freight vehicle types files: "
                f"sample={missing_vehicle_types[:10]}"
            )

        vehicle_type_expr = "trim(CAST(source.vehicleTypeId AS VARCHAR))"
        scale_expr = (
            f"CASE WHEN regexp_matches(upper({vehicle_type_expr}), '(^|[-_])(BUS|RAIL|FERRY|SUBWAY|TRAM|TRAIN|COACH)($|[-_])') "
            f"THEN {1.0 / transit_sample} ELSE {1.0 / population_sample} END"
        )
        obs_expr = "COALESCE(TRY_CAST(source.observations AS DOUBLE), 0.0)"
        annualization_expr = "lookup.annualization_days"
        tot_trips_expr = f"({obs_expr} * {scale_expr} * {annualization_expr})"
        select_parts = []
        for col in prepared_group_cols:
            if col == "vehicleTypeId":
                select_parts.append(f'{vehicle_type_expr} AS "{col}"')
            else:
                select_parts.append(f'source."{col}" AS "{col}"')
        if "attributeOrigType" in link_lengths.columns:
            select_parts.append('link_lengths.attributeOrigType AS "roadCategory"')
        if has_height_lookup:
            select_parts.append(
                f'COALESCE(height_lookup.tailpipe_height_meters, {_DEFAULT_TAILPIPE_HEIGHT_METERS}) AS "source_release_height"'
            )
        select_parts.append(f"{tot_trips_expr} AS totTrips")
        select_parts.append(
            f"({tot_trips_expr} * COALESCE(TRY_CAST(link_lengths.\"{beam_length_col}\" AS DOUBLE), 0.0) / {_METERS_PER_MILE}) AS totVMT"
        )
        for pollutant in required_pollutants:
            select_parts.append(
                f"""(
                    COALESCE(TRY_CAST(source."{pollutant}" AS DOUBLE), 0.0)
                    * {scale_expr}
                    * {annualization_expr}
                    / {grams_per_short_ton}
                ) AS "tons_per_year_{pollutant}" """
            )
        height_join = (
            f"\n            LEFT JOIN height_lookup ON {vehicle_type_expr} = height_lookup.vehicleTypeId"
            if has_height_lookup else ""
        )
        query = f"""
            SELECT
                {", ".join(select_parts)}
            FROM {scan} AS source
            LEFT JOIN link_lengths
              ON source.linkId = link_lengths.linkId
            LEFT JOIN annualization_lookup AS lookup
              ON {vehicle_type_expr} = lookup.vehicleTypeId{height_join}
        """
        output_sql = str(output_path).replace("'", "''")
        if show_progress:
            _configure_duckdb_progress_bar(con, enabled=True)
        con.execute(f"COPY ({query}) TO '{output_sql}' (FORMAT PARQUET)")
        if show_progress:
            _configure_duckdb_progress_bar(con, enabled=False)
    finally:
        con.close()
    output_rows = parquet_row_count(output_path)
    logger.info(
        "Annualized prepared skims via DuckDB in %.2fs: input=%s output_rows=%d output=%s",
        time.perf_counter() - started,
        prepared_skims_path,
        output_rows,
        output_path,
    )
    return pd.DataFrame(columns=output_columns)
def _resolve_vehicle_type_annualization_days_lookup(
    *,
    vehicle_category_metadata_file: Optional[str],
    annualization_days: Optional[dict[str, float]] = None,
    passenger_vehicle_types_path: Optional[str] = None,
    freight_vehicle_types_path: Optional[str] = None,
) -> dict[str, float]:
    defaults = _default_annualization_days(annualization_days)
    if not vehicle_category_metadata_file:
        raise ValueError("vehicle_category_metadata_file is required to resolve annualization factors.")
    if not passenger_vehicle_types_path or not freight_vehicle_types_path:
        raise ValueError("Passenger and freight vehicle types inputs are required to resolve annualization factors.")
    csv_path = str(vehicle_category_metadata_file).strip()
    if not csv_path:
        raise ValueError("vehicle_category_metadata_file must be non-empty.")
    category_lookup, sanitized_categories = _load_vehicle_operation_days_lookup(csv_path)
    vehicle_type_category_lookup = _load_vehicle_type_category_lookup(
        passenger_vehicle_types_path,
        freight_vehicle_types_path,
        category_lookup=category_lookup,
        sanitized_categories=sanitized_categories,
    )
    resolved: dict[str, float] = {}
    for vehicle_type_id, category in vehicle_type_category_lookup.items():
        if category in category_lookup:
            resolved[vehicle_type_id] = float(category_lookup[category])
        else:
            resolved[vehicle_type_id] = float(defaults[_vehicle_group_for_emfac_category(category)])
    return resolved


_DEFAULT_TAILPIPE_HEIGHT_METERS = 1.0


def _load_vehicle_tailpipe_height_lookup(csv_path: str) -> dict[str, float]:
    frame = read_table(csv_path)
    category_column = "emfac_vehicle_category"
    if category_column not in frame.columns or "tailpipe_height_meters" not in frame.columns:
        return {}
    lookup: dict[str, float] = {}
    for row in frame[[category_column, "tailpipe_height_meters"]].itertuples(index=False):
        category = str(row[0]).strip()
        if not category or pd.isna(row[1]) or str(row[1]).strip() == "":
            continue
        lookup[category] = float(row[1])
    return lookup


def _resolve_vehicle_type_tailpipe_height_lookup(
    *,
    vehicle_category_metadata_file: Optional[str],
    passenger_vehicle_types_path: Optional[str],
    freight_vehicle_types_path: Optional[str],
) -> dict[str, float]:
    if not vehicle_category_metadata_file or not passenger_vehicle_types_path or not freight_vehicle_types_path:
        return {}
    csv_path = str(vehicle_category_metadata_file).strip()
    if not csv_path:
        return {}
    height_by_category = _load_vehicle_tailpipe_height_lookup(csv_path)
    if not height_by_category:
        return {}
    _, sanitized_categories = _load_vehicle_operation_days_lookup(csv_path)
    vehicle_type_category_lookup = _load_vehicle_type_category_lookup(
        passenger_vehicle_types_path,
        freight_vehicle_types_path,
        category_lookup=height_by_category,
        sanitized_categories=sanitized_categories,
    )
    return {
        vehicle_type_id: float(height_by_category.get(category, _DEFAULT_TAILPIPE_HEIGHT_METERS))
        for vehicle_type_id, category in vehicle_type_category_lookup.items()
    }


def _resolve_skims_annualization_factors_from_lookup(
    prepared: pd.DataFrame,
    *,
    vehicle_type_annualization_days_lookup: dict[str, float],
) -> pd.Series:
    if "vehicleTypeId" not in prepared.columns:
        raise ValueError("Prepared skims must include vehicleTypeId to resolve annualization factors.")
    resolved = prepared["vehicleTypeId"].astype(str).map(vehicle_type_annualization_days_lookup)
    missing_vehicle_types = (
        prepared.loc[resolved.isna(), "vehicleTypeId"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if missing_vehicle_types:
        raise ValueError(
            "Could not resolve annualization days for some skim vehicleTypeId values using "
            "the configured passenger/freight vehicle types files: "
            f"sample={missing_vehicle_types[:10]}"
        )
    return resolved.astype(float)


def _sanitize_emfac_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str("" if pd.isna(value) else value).strip())


def _default_annualization_days(annualization_days: Optional[dict[str, float]]) -> dict[str, float]:
    merged = dict(default_annualization_days_by_vehicle_group)
    if annualization_days:
        merged.update({str(key): float(value) for key, value in annualization_days.items()})
    for key in ("light_duty", "medium_heavy_duty"):
        value = float(merged[key])
        if value <= 0:
            raise ValueError(f"Annualization days must be positive for vehicle group {key}, got {value}")
        merged[key] = value
    return merged


def _load_vehicle_operation_days_lookup(csv_path: str) -> tuple[dict[str, float], list[tuple[str, str]]]:
    frame = read_table(csv_path)
    category_column = "emfac_vehicle_category"
    if category_column not in frame.columns:
        raise ValueError(
            "Vehicle category metadata CSV is missing required column 'emfac_vehicle_category'"
        )
    lookup: dict[str, float] = {}
    sanitized_categories: list[tuple[str, str]] = []
    for category in frame[category_column].astype(str).str.strip():
        if not category:
            continue
        sanitized_categories.append((category, _sanitize_emfac_token(category)))
    if "operation_days_per_year" in frame.columns:
        for row in frame[[category_column, "operation_days_per_year"]].itertuples(index=False):
            category = str(row[0]).strip()
            if not category:
                continue
            if pd.isna(row[1]) or str(row[1]).strip() == "":
                continue
            days = float(row[1])
            if days <= 0:
                raise ValueError(f"Operation days must be positive for vehicle category={category!r}")
            lookup[category] = days
    sanitized_categories.sort(key=lambda item: len(item[1]), reverse=True)
    return lookup, sanitized_categories


def _vehicle_group_for_emfac_category(category: object) -> str:
    category_token = str("" if pd.isna(category) else category).strip()
    if category_token in {"LDA", "LDT1", "LDT2"}:
        return "light_duty"
    return "medium_heavy_duty"


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

    if "emfacVehicleCategory" in prepared.columns:
        prepared["emfacVehicleCategory"] = prepared["emfacVehicleCategory"].where(
            prepared["emfacVehicleCategory"].notna(), ""
        )
        prepared["emfacVehicleCategory"] = prepared["emfacVehicleCategory"].astype(str).str.strip()
        category_rows = prepared.loc[
            prepared["emfacVehicleCategory"].ne("") & ~prepared["emfacVehicleCategory"].str.lower().eq("nan")
        ]
        if not category_rows.empty:
            missing_categories = sorted(set(category_rows["emfacVehicleCategory"]) - set(category_lookup))
            if missing_categories:
                raise ValueError(
                    "Vehicle types input contains EMFAC categories not present in the configured annualization CSV: "
                    f"{missing_categories[:10]}"
                )
            return (
                category_rows[["vehicleTypeId", "emfacVehicleCategory"]]
                .drop_duplicates(subset=["vehicleTypeId"], keep="first")
                .set_index("vehicleTypeId")["emfacVehicleCategory"]
                .to_dict()
            )

    raise ValueError(
        "Vehicle types input must include non-empty emfacVehicleCategory when resolving "
        "annualization days from vehicle_category_metadata_file."
    )


def resolve_skims_annualization_factors(
    prepared: pd.DataFrame,
    *,
    vehicle_category_metadata_file: Optional[str],
    annualization_days: Optional[dict[str, float]] = None,
    passenger_vehicle_types_path: Optional[str] = None,
    freight_vehicle_types_path: Optional[str] = None,
) -> pd.Series:
    vehicle_type_annualization_days_lookup = _resolve_vehicle_type_annualization_days_lookup(
        vehicle_category_metadata_file=vehicle_category_metadata_file,
        annualization_days=annualization_days,
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
    )
    return _resolve_skims_annualization_factors_from_lookup(
        prepared,
        vehicle_type_annualization_days_lookup=vehicle_type_annualization_days_lookup,
    )
