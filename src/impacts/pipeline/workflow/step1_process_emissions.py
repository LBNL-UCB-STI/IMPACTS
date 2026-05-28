"""Step 1 — Emissions processing path."""
from __future__ import annotations

import logging
from pathlib import Path
import re
import time
from typing import Any
from typing import Dict
from typing import Optional

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from ...common import configure_duckdb_connection
from ...common import log_step_banner
from ...common import log_substep_banner
from ...common import normalize_county_fips
from ...common import read_table
from ...common import read_vector
from ...common import resolve_required_manifest_input
from ...manifest.schema import PipelineConfig
from .prepare_emissions_from_skims import _build_zone_allocated_table
from .prepare_emissions_from_skims import _build_zone_grouped_table
from .prepare_emissions_from_skims import _load_vehicle_type_activity_lookup
from .prepare_emissions_from_skims import _reuse_existing_outputs
from .prepare_emissions_from_skims import load_or_prepare_skims_df
from . import _step_label

logger = logging.getLogger(__name__)

_VMT_PROCESSES = {"RUNEX", "PMBW", "PMTW", "PRDUST", "RUNLOSS", "PTOEX"}
_TRIP_PROCESSES = {"HOTSOAK", "DIURN", "STREX"}
_CORRECTION_ASSIGNMENT_GROUPS = {"passenger", "freight"}

_OSM_TO_AERMOD_TEMPORAL = {
    "motorway": "FREEWAY",
    "motorway_link": "FREEWAY",
    "trunk": "FREEWAY",
    "trunk_link": "CITYSTREET",
    "primary": "CITYSTREET",
    "primary_link": "CITYSTREET",
    "secondary": "CITYSTREET",
    "secondary_link": "CITYSTREET",
    "tertiary": "CITYSTREET",
    "tertiary_link": "CITYSTREET",
    "residential": "CITYSTREET",
    "residential_link": "CITYSTREET",
}


def _step1_scratch_dir(raw_dir: Path) -> Path:
    return raw_dir / "tmp" / "step1_process_emissions"


def _log_step1_elapsed(step_id: str, label: str, started: float, **details: object) -> None:
    detail_parts = [f"{key}={value}" for key, value in details.items()]
    suffix = "" if not detail_parts else ": " + " ".join(detail_parts)
    logger.info(
        "%s %s in %.2fs%s",
        _step_label(step_id),
        label,
        time.perf_counter() - started,
        suffix,
    )


def _require_columns(df: pd.DataFrame, *, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _log_aermod_urban_class_trace(label: str, df: pd.DataFrame) -> None:
    tracked_cols = [
        "aermod_cell_id",
        "source_urban_class",
        "source_urban_class_x",
        "source_urban_class_y",
        "cell_source_urban_class",
    ]
    present_cols = [col for col in tracked_cols if col in df.columns]
    logger.info("%s trace %s columns=%s", _step_label("1.4"), label, list(df.columns))
    logger.info(
        "%s trace %s tracked_present=%s row_count=%d columns_unique=%s duplicate_columns=%s",
        _step_label("1.4"),
        label,
        {col: (col in df.columns) for col in tracked_cols},
        len(df),
        bool(df.columns.is_unique),
        df.columns[df.columns.duplicated()].tolist(),
    )
    if present_cols and not df.empty:
        logger.info(
            "%s trace %s sample=%s",
            _step_label("1.4"),
            label,
            df[present_cols].head(5).to_dict(orient="records"),
        )


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    label: str,
) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    den = pd.to_numeric(denominator, errors="coerce").fillna(0.0)
    ratio = pd.Series(np.ones(len(num), dtype=np.float64), index=num.index)
    valid = den.gt(0.0)
    ratio.loc[valid] = num.loc[valid] / den.loc[valid]
    invalid = (~valid) & num.ne(0.0)
    if invalid.any():
        logger.warning(
            "%s zero source totals encountered for %s in %d county rows; using neutral factor 1.0",
            _step_label("1.3"),
            label,
            int(invalid.sum()),
        )
    return ratio


def _build_county_name_lookup(county_boundaries_path: str) -> dict[str, str]:
    county_gdf = read_vector(county_boundaries_path)
    if "COUNTYFP" not in county_gdf.columns or "NAME" not in county_gdf.columns:
        raise ValueError("County boundaries must include COUNTYFP and NAME for inventory-based correction.")
    lookup = county_gdf[["COUNTYFP", "NAME"]].drop_duplicates().copy()
    lookup["countyfp"] = normalize_county_fips(lookup["COUNTYFP"])
    lookup["NAME"] = lookup["NAME"].astype(str).str.strip()
    lookup = lookup.loc[lookup["countyfp"].notna() & lookup["NAME"].ne("")].copy()
    return dict(zip(lookup["NAME"], lookup["countyfp"]))


def _should_apply_inventory_activity_corrections(pipeline: PipelineConfig) -> bool:
    return bool(
        pipeline.enable_passenger_inventory_activity_correction
        or pipeline.enable_freight_inventory_activity_correction
    )


def _build_beam_activity_details(
    *,
    skims_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_skims = {"linkId", "vehicleTypeId", "process", "totVMT", "totTrips"}
    missing_skims = sorted(required_skims - set(skims_df.columns))
    if missing_skims:
        raise ValueError(f"Prepared skims are missing required activity columns: {missing_skims}")
    required_grouped = {"linkId", "county_COUNTYFP", "county_proportion"}
    missing_grouped = sorted(required_grouped - set(grouped_df.columns))
    if missing_grouped:
        raise ValueError(f"Grouped intersection is missing required county allocation columns: {missing_grouped}")

    vehicle_lookup = _load_vehicle_type_activity_lookup(
        passenger_vehicle_types_path,
        freight_vehicle_types_path,
    )
    county_link = grouped_df[["linkId", "county_COUNTYFP", "county_proportion"]].copy()
    county_link["countyfp"] = normalize_county_fips(county_link["county_COUNTYFP"])
    county_link["county_proportion"] = pd.to_numeric(
        county_link["county_proportion"], errors="coerce"
    ).fillna(0.0)
    county_link = (
        county_link.groupby(["linkId", "countyfp"], dropna=False)["county_proportion"]
        .sum()
        .reset_index()
    )

    activity = skims_df[["linkId", "vehicleTypeId", "process", "totVMT", "totTrips"]].copy()
    activity["vehicleTypeId"] = activity["vehicleTypeId"].astype(str).str.strip()
    activity["process"] = activity["process"].astype(str).str.upper().str.strip()
    activity["totVMT"] = pd.to_numeric(activity["totVMT"], errors="coerce").fillna(0.0)
    activity["totTrips"] = pd.to_numeric(activity["totTrips"], errors="coerce").fillna(0.0)
    activity = (
        activity.groupby(["linkId", "vehicleTypeId", "process"], dropna=False)[["totVMT", "totTrips"]]
        .max()
        .reset_index()
    )
    detailed = activity.merge(county_link, how="inner", on="linkId")
    detailed["totVMT"] = detailed["totVMT"] * detailed["county_proportion"]
    detailed["totTrips"] = detailed["totTrips"] * detailed["county_proportion"]
    detailed = detailed.merge(vehicle_lookup, how="left", on="vehicleTypeId")
    detailed = detailed.loc[detailed["assignment_group"].isin(_CORRECTION_ASSIGNMENT_GROUPS)].copy()
    missing_vehicle_types = detailed.loc[
        detailed["assignment_group"].isna() | detailed["modelYear"].isna(),
        "vehicleTypeId",
    ].drop_duplicates().tolist()
    if missing_vehicle_types:
        raise ValueError(
            "Could not resolve assignment_group/modelYear for skim vehicleTypeId values: "
            f"{missing_vehicle_types[:10]}"
        )
    detailed = (
        detailed.groupby(["countyfp", "assignment_group", "modelYear", "process"], dropna=False)[["totVMT", "totTrips"]]
        .sum()
        .reset_index()
    )
    return detailed.copy(), detailed


def _derive_inventory_activity_targets_for_assignment(
    *,
    inventory_path: str,
    county_name_lookup: dict[str, str],
    assignment_group: str,
) -> pd.DataFrame:
    inventory = read_table(inventory_path)
    required = {
        "county",
        "modelYear",
        "process",
        "total_vmt_vehicle_miles_per_year",
        "trips_per_year",
    }
    missing = sorted(required - set(inventory.columns))
    if missing:
        raise ValueError(f"Inventory file missing required activity columns: {missing}")

    prepared = inventory.copy()
    prepared["county"] = prepared["county"].astype(str).str.strip()
    prepared["countyfp"] = prepared["county"].map(county_name_lookup)
    missing_counties = prepared.loc[prepared["countyfp"].isna(), "county"].drop_duplicates().tolist()
    if missing_counties:
        raise ValueError(
            "Could not map some inventory counties to FIPS codes using county boundaries: "
            f"{missing_counties[:10]}"
        )
    prepared["modelYear"] = prepared["modelYear"].astype(str).str.strip()
    prepared["process"] = prepared["process"].astype(str).str.upper().str.strip()
    prepared["total_vmt_vehicle_miles_per_year"] = pd.to_numeric(
        prepared["total_vmt_vehicle_miles_per_year"], errors="coerce"
    ).fillna(0.0)
    prepared["trips_per_year"] = pd.to_numeric(prepared["trips_per_year"], errors="coerce").fillna(0.0)
    targets = (
        prepared.groupby(["countyfp", "modelYear", "process"], dropna=False)[
            ["total_vmt_vehicle_miles_per_year", "trips_per_year"]
        ]
        .sum()
        .reset_index()
        .rename(
            columns={
                "total_vmt_vehicle_miles_per_year": "totVMT",
                "trips_per_year": "totTrips",
            }
        )
    )
    targets["assignment_group"] = assignment_group
    return targets[["countyfp", "assignment_group", "modelYear", "process", "totVMT", "totTrips"]]


def _derive_county_correction_factors(
    beam_activity_totals: pd.DataFrame,
    inventory_targets: pd.DataFrame,
) -> pd.DataFrame:
    merged = inventory_targets.merge(
        beam_activity_totals.rename(
            columns={
                "totVMT": "totVMT_source",
                "totTrips": "totTrips_source",
            }
        ),
        how="outer",
        on=["countyfp", "assignment_group", "modelYear", "process"],
    )
    for col in ["totVMT", "totTrips", "totVMT_source", "totTrips_source"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    merged["factor_totVMT"] = _safe_ratio(
        merged["totVMT"],
        merged["totVMT_source"],
        label="totVMT by county/assignment_group/modelYear/process",
    )
    merged["factor_totTrips"] = _safe_ratio(
        merged["totTrips"],
        merged["totTrips_source"],
        label="totTrips by county/assignment_group/modelYear/process",
    )
    return merged


def apply_county_corrections(
    allocated_df: pd.DataFrame,
    county_correction_factors: pd.DataFrame,
    *,
    county_col: str,
    scratch_dir: Path,
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
) -> pd.DataFrame:
    vehicle_lookup = _load_vehicle_type_activity_lookup(
        passenger_vehicle_types_path,
        freight_vehicle_types_path,
    )[["vehicleTypeId", "assignment_group", "modelYear"]].copy()
    process_upper = allocated_df.get("process", pd.Series("", index=allocated_df.index)).astype(str).str.upper().str.strip()
    unique_processes = sorted(process_upper[process_upper.ne("")].dropna().unique().tolist())
    vmt_used = sorted([proc for proc in unique_processes if proc in _VMT_PROCESSES])
    trip_used = sorted([proc for proc in unique_processes if proc in _TRIP_PROCESSES])
    neutral_used = sorted([proc for proc in unique_processes if proc not in _VMT_PROCESSES and proc not in _TRIP_PROCESSES])
    if vmt_used:
        logger.info("%s using factor_totVMT for processes: %s", _step_label("1.3"), ", ".join(vmt_used))
    if trip_used:
        logger.info("%s using factor_totTrips for processes: %s", _step_label("1.3"), ", ".join(trip_used))
    if neutral_used:
        logger.warning(
            "%s no county correction factor mapping for processes; using neutral factor 1.0: %s",
            _step_label("1.3"),
            ", ".join(neutral_used),
        )
    emission_allocated_cols = [
        c for c in allocated_df.columns
        if c.startswith("tons_per_year_") and c.endswith("_allocated")
    ]
    original_cols = list(allocated_df.columns)
    passthrough_cols = [col for col in original_cols if col not in emission_allocated_cols]
    normalized_county_expr = (
        f"CASE WHEN regexp_extract(CAST(a.\"{county_col}\" AS VARCHAR), '(\\d+)', 1) = '' "
        f"THEN NULL ELSE lpad(regexp_extract(CAST(a.\"{county_col}\" AS VARCHAR), '(\\d+)', 1), 3, '0') END"
    )

    con = duckdb.connect(database=":memory:")
    try:
        configure_duckdb_connection(con, working_dir=scratch_dir, show_progress=True, profile="memory_heavy")
        con.register("allocated_df", allocated_df)
        con.register("vehicle_lookup", vehicle_lookup)
        con.register("county_correction_factors", county_correction_factors)

        missing_vehicle_types = [
            row[0]
            for row in con.execute(
                """
                SELECT DISTINCT trim(CAST(a."vehicleTypeId" AS VARCHAR)) AS vehicleTypeId
                FROM allocated_df AS a
                LEFT JOIN vehicle_lookup AS v
                    ON trim(CAST(a."vehicleTypeId" AS VARCHAR)) = trim(CAST(v."vehicleTypeId" AS VARCHAR))
                WHERE COALESCE(trim(CAST(a."vehicleTypeId" AS VARCHAR)), '') <> ''
                  AND v.assignment_group IN ('passenger', 'freight')
                  AND COALESCE(trim(CAST(v.modelYear AS VARCHAR)), '') = ''
                """
            ).fetchall()
            if row[0]
        ]
        unresolved_vehicle_types = [
            row[0]
            for row in con.execute(
                """
                SELECT DISTINCT trim(CAST(a."vehicleTypeId" AS VARCHAR)) AS vehicleTypeId
                FROM allocated_df AS a
                LEFT JOIN vehicle_lookup AS v
                    ON trim(CAST(a."vehicleTypeId" AS VARCHAR)) = trim(CAST(v."vehicleTypeId" AS VARCHAR))
                WHERE COALESCE(trim(CAST(a."vehicleTypeId" AS VARCHAR)), '') <> ''
                  AND v.assignment_group IS NULL
                """
            ).fetchall()
            if row[0]
        ]
        if missing_vehicle_types:
            raise ValueError(
                "Could not resolve modelYear for correction-eligible allocated rows with vehicleTypeId values: "
                f"{missing_vehicle_types[:10]}"
            )
        if unresolved_vehicle_types:
            raise ValueError(
                "Could not resolve vehicle type assignment for allocated rows with vehicleTypeId values: "
                f"{unresolved_vehicle_types[:10]}"
            )

        select_parts = [f'a."{col}" AS "{col}"' for col in passthrough_cols]
        for col in emission_allocated_cols:
            select_parts.append(
                f"""
                COALESCE(TRY_CAST(a."{col}" AS DOUBLE), 0.0) *
                CASE
                    WHEN upper(trim(CAST(a."process" AS VARCHAR))) IN ({", ".join(f"'{proc}'" for proc in sorted(_VMT_PROCESSES))})
                        THEN COALESCE(TRY_CAST(f.factor_totVMT AS DOUBLE), 1.0)
                    WHEN upper(trim(CAST(a."process" AS VARCHAR))) IN ({", ".join(f"'{proc}'" for proc in sorted(_TRIP_PROCESSES))})
                        THEN COALESCE(TRY_CAST(f.factor_totTrips AS DOUBLE), 1.0)
                    ELSE 1.0
                END AS "{col}"
                """.strip()
            )
        query = f"""
            SELECT
                {", ".join(select_parts)}
            FROM allocated_df AS a
            LEFT JOIN vehicle_lookup AS v
                ON trim(CAST(a."vehicleTypeId" AS VARCHAR)) = trim(CAST(v."vehicleTypeId" AS VARCHAR))
            LEFT JOIN county_correction_factors AS f
                ON {normalized_county_expr} = CAST(f.countyfp AS VARCHAR)
               AND v.assignment_group = f.assignment_group
               AND COALESCE(trim(CAST(v.modelYear AS VARCHAR)), '') = COALESCE(trim(CAST(f.modelYear AS VARCHAR)), '')
               AND upper(trim(CAST(a."process" AS VARCHAR))) = upper(trim(CAST(f.process AS VARCHAR)))
        """
        result = con.execute(query).fetchdf()
    finally:
        con.close()

    return result


def _build_corrected_source_totals(
    county_corrected_df: Optional[pd.DataFrame],
    *,
    scratch_dir: Path,
) -> Optional[pd.DataFrame]:
    if county_corrected_df is None or county_corrected_df.empty:
        return None
    group_cols = ["linkId", "vehicleTypeId", "process"]
    passthrough_cols = []
    if "roadCategory" in county_corrected_df.columns:
        passthrough_cols.append("roadCategory")
    if "source_release_height" in county_corrected_df.columns:
        passthrough_cols.append("source_release_height")
    required = set(group_cols)
    missing = sorted(required - set(county_corrected_df.columns))
    if missing:
        raise ValueError(f"County-corrected emissions are missing source grouping columns: {missing}")
    value_cols = [col for col in county_corrected_df.columns if col.endswith("_county_allocated")]
    con = duckdb.connect(database=":memory:")
    try:
        configure_duckdb_connection(con, working_dir=scratch_dir, show_progress=True, profile="memory_heavy")
        con.register("county_corrected_df", county_corrected_df)
        select_parts = [f'"{col}"' for col in group_cols]
        if "roadCategory" in passthrough_cols:
            select_parts.append('ANY_VALUE("roadCategory") AS "roadCategory"')
        if "source_release_height" in passthrough_cols:
            select_parts.append(
                'MAX(COALESCE(TRY_CAST("source_release_height" AS DOUBLE), 1.0)) AS "source_release_height"'
            )
        select_parts.extend(
            f'SUM(COALESCE(TRY_CAST("{col}" AS DOUBLE), 0.0)) AS "{col}"'
            for col in value_cols
        )
        aggregated = con.execute(
            f"""
            SELECT
                {", ".join(select_parts)}
            FROM county_corrected_df
            GROUP BY 1, 2, 3
            """
        ).fetchdf()
    finally:
        con.close()
    rename_map = {
        "totVMT_county_allocated": "totVMT",
        "totTrips_county_allocated": "totTrips",
    }
    for col in value_cols:
        if col.startswith("tons_per_year_"):
            rename_map[col] = col.removesuffix("_county_allocated")
    aggregated = aggregated.rename(columns=rename_map)
    return aggregated


def _save_grid_emissions(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    grid_path: str,
    output_epsg: int,
    output_stem: Path,
    *,
    step_id: str,
) -> gpd.GeoDataFrame:
    started = time.perf_counter()
    grid_gdf = read_vector(grid_path, columns=[right_col, "geometry"])
    if right_col not in grid_gdf.columns:
        raise ValueError(
            f"Step 1 grid export expected grid column '{right_col}' in {grid_path}. "
            f"Available columns: {list(grid_gdf.columns)}"
        )
    if grid_gdf.crs is not None:
        grid_gdf = grid_gdf.to_crs(epsg=output_epsg)
    expected_grid_ids = set(pd.to_numeric(df[left_col], errors="coerce").dropna().astype(int).unique().tolist())
    tabular = df.copy()
    join_started = time.perf_counter()
    con = duckdb.connect(database=":memory:")
    try:
        configure_duckdb_connection(con, working_dir=output_stem.parent, show_progress=False, profile="export")
        con.register("emissions_df", tabular)
        con.register("grid_ids", pd.DataFrame({right_col: grid_gdf[right_col]}))
        joined = con.execute(
            f"""
            SELECT emissions_df.*
            FROM emissions_df
            LEFT JOIN grid_ids
              ON emissions_df."{left_col}" = grid_ids."{right_col}"
            WHERE grid_ids."{right_col}" IS NOT NULL
            """
        ).fetchdf()
    finally:
        con.close()
    _log_step1_elapsed(step_id, "grid export tabular join complete", join_started, rows=len(joined), grid=output_stem.name)
    merge_started = time.perf_counter()
    joined = joined.merge(
        grid_gdf[[right_col, "geometry"]],
        how="left",
        left_on=left_col,
        right_on=right_col,
    )
    if right_col != left_col:
        joined = joined.drop(columns=[right_col])
    geo = gpd.GeoDataFrame(joined, geometry="geometry", crs=grid_gdf.crs)
    _log_step1_elapsed(step_id, "grid export geometry join complete", merge_started, rows=len(geo), grid=output_stem.name)
    actual_grid_ids = set(pd.to_numeric(geo[left_col], errors="coerce").dropna().astype(int).unique().tolist())
    if actual_grid_ids != expected_grid_ids:
        missing = sorted(expected_grid_ids - actual_grid_ids)[:10]
        extra = sorted(actual_grid_ids - expected_grid_ids)[:10]
        raise ValueError(
            f"Step 1 grid export mismatch for {left_col}: expected {len(expected_grid_ids)} grid ids, "
            f"got {len(actual_grid_ids)}. sample_missing={missing} sample_extra={extra}"
        )
    missing_geometry = int(geo.geometry.isna().sum())
    if missing_geometry:
        raise ValueError(
            f"Step 1 grid export missing geometry for {missing_geometry} rows in {output_stem}"
        )
    parquet_path = Path(str(output_stem) + ".parquet")
    gpkg_path = Path(str(output_stem) + ".gpkg")
    logger.info(
        "Step 1 grid export: writing %d rows to %s",
        len(geo),
        parquet_path,
    )
    parquet_started = time.perf_counter()
    geo.to_parquet(parquet_path, index=False)
    _log_step1_elapsed(step_id, "grid export parquet write complete", parquet_started, rows=len(geo), path=parquet_path)
    logger.info(
        "Step 1 grid export: writing %d rows to %s",
        len(geo),
        gpkg_path,
    )
    gpkg_started = time.perf_counter()
    geo.to_file(gpkg_path, driver="GPKG")
    _log_step1_elapsed(step_id, "grid export gpkg write complete", gpkg_started, rows=len(geo), path=gpkg_path)
    study_area_grid = (
        geo[[left_col, "geometry"]]
        .drop_duplicates(subset=[left_col])
        .reset_index(drop=True)
    )
    _log_step1_elapsed(step_id, "grid export complete", started, rows=len(geo), unique_grid_ids=len(study_area_grid), grid=output_stem.name)
    return gpd.GeoDataFrame(study_area_grid, geometry="geometry", crs=geo.crs)


def _aggregate_aermod_emissions_for_export(
    aermod_allocated_df: pd.DataFrame,
    *,
    scratch_dir: Path,
) -> pd.DataFrame:
    started = time.perf_counter()
    _require_columns(
        aermod_allocated_df,
        columns=[
            "aermod_cell_id",
            "source_temporal_class",
            "source_release_height",
            "source_urban_class",
        ],
        label="AERMOD allocated emissions table",
    )
    emission_cols = [
        c for c in aermod_allocated_df.columns
        if c.startswith("tons_per_year_") and c.endswith("_aermod_allocated")
    ]
    if not emission_cols:
        raise ValueError("AERMOD allocated emissions table is missing tons_per_year_*_aermod_allocated columns.")

    select_parts = [
        '"aermod_cell_id" AS "aermod_cell_id"',
        'ANY_VALUE("source_temporal_class") AS "source_temporal_class"',
        'MAX(COALESCE(TRY_CAST("source_release_height" AS DOUBLE), 1.0)) AS "source_release_height"',
        'MAX(COALESCE(TRY_CAST("source_urban_class" AS BIGINT), 0)) AS "source_urban_class"',
        *[
            f'SUM(COALESCE(TRY_CAST("{col}" AS DOUBLE), 0.0)) AS "{col}"'
            for col in emission_cols
        ],
    ]

    con = duckdb.connect(database=":memory:")
    try:
        configure_duckdb_connection(
            con,
            working_dir=scratch_dir,
            show_progress=True,
            profile="memory_heavy",
        )
        con.register("aermod_allocated_df", aermod_allocated_df)
        aggregated = con.execute(
            f"""
            SELECT
                {", ".join(select_parts)}
            FROM aermod_allocated_df
            GROUP BY 1
            """
        ).fetchdf()
    finally:
        con.close()

    logger.info(
        "%s aggregated AERMOD export rows from %d detailed rows to %d source cells",
        _step_label("1.6", "aermod"),
        len(aermod_allocated_df),
        len(aggregated),
    )
    _log_step1_elapsed(
        "1.6",
        "AERMOD export aggregation complete",
        started,
        detailed_rows=len(aermod_allocated_df),
        source_cells=len(aggregated),
    )
    return aggregated


def _build_county_corrected_table(
    *,
    county_allocated_df: pd.DataFrame,
    county_grouped_df: pd.DataFrame,
    skims_df: pd.DataFrame,
    pipeline: PipelineConfig,
    manifest_inputs: Dict[str, Any],
    scratch_dir: Path,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if county_allocated_df is None or county_allocated_df.empty:
        return None, None, None
    if not _should_apply_inventory_activity_corrections(pipeline):
        logger.info("%s inventory activity corrections disabled; skipping county correction stage", _step_label("1.3"))
        return county_allocated_df.copy(), None, None

    progress = tqdm(
        total=4,
        desc="Step 1.3 activity corrections",
        leave=True,
        dynamic_ncols=True,
        disable=not logger.isEnabledFor(logging.INFO),
    )
    passenger_vehicle_types_path = resolve_required_manifest_input(manifest_inputs, key="passenger_vehicle_types_input")
    freight_vehicle_types_path = resolve_required_manifest_input(manifest_inputs, key="freight_vehicle_types_input")
    county_boundaries_path = resolve_required_manifest_input(manifest_inputs, key="county_boundaries")
    try:
        _, beam_activity_totals = _build_beam_activity_details(
            skims_df=skims_df,
            grouped_df=county_grouped_df,
            passenger_vehicle_types_path=passenger_vehicle_types_path,
            freight_vehicle_types_path=freight_vehicle_types_path,
        )
        progress.update(1)
        county_name_lookup = _build_county_name_lookup(county_boundaries_path)
        inventory_targets_frames: list[pd.DataFrame] = []
        if pipeline.enable_passenger_inventory_activity_correction:
            passenger_inventory_path = pipeline.passenger_inventory_file
            logger.info(
                "%s deriving passenger county activity correction factors from inventory %s",
                _step_label("1.3"),
                passenger_inventory_path,
            )
            inventory_targets_frames.append(
                _derive_inventory_activity_targets_for_assignment(
                    inventory_path=passenger_inventory_path,
                    county_name_lookup=county_name_lookup,
                    assignment_group="passenger",
                )
            )
        if pipeline.enable_freight_inventory_activity_correction:
            freight_inventory_path = pipeline.freight_inventory_file
            logger.info(
                "%s deriving freight county activity correction factors from inventory %s",
                _step_label("1.3"),
                freight_inventory_path,
            )
            inventory_targets_frames.append(
                _derive_inventory_activity_targets_for_assignment(
                    inventory_path=freight_inventory_path,
                    county_name_lookup=county_name_lookup,
                    assignment_group="freight",
                )
            )
        inventory_targets = pd.concat(inventory_targets_frames, ignore_index=True)
        progress.update(1)
        county_correction_factors = _derive_county_correction_factors(
            beam_activity_totals,
            inventory_targets,
        )
        progress.update(1)
        logger.info(
            "%s correcting allocated emissions by county/modelYear/process/assignment factors",
            _step_label("1.3"),
        )
        corrected = apply_county_corrections(
            county_allocated_df,
            county_correction_factors,
            county_col="county_COUNTYFP",
            scratch_dir=scratch_dir,
            passenger_vehicle_types_path=passenger_vehicle_types_path,
            freight_vehicle_types_path=freight_vehicle_types_path,
        )
        progress.update(1)
    finally:
        progress.close()
    return corrected, beam_activity_totals, county_correction_factors


def run(
    pipeline: PipelineConfig,
    raw_dir: Path,
    input_root: Path,
    intersection_paths: Dict[str, Optional[str]],
    manifest_inputs: Optional[Dict[str, Any]] = None,
    scratch_root: Optional[Path] = None,
) -> Dict[str, Optional[str]]:
    log_step_banner("Step 1", "Process Emissions", logger=logger)
    step_started = time.perf_counter()
    reused = _reuse_existing_outputs(raw_dir)
    if reused is not None:
        return reused
    scratch_dir = _step1_scratch_dir(scratch_root if scratch_root is not None else raw_dir)

    log_substep_banner("1.0", "prepare skims inputs", logger=logger)
    skims_started = time.perf_counter()
    skims_df = load_or_prepare_skims_df(
        input_root=input_root,
        intersection_path=(intersection_paths.get("county") or ""),
        beam_length_col=pipeline.beam_length_col,
        prepared_skims_group_cols=list(pipeline.prepared_skims_group_cols),
        pollutants=list(pipeline.pollutants),
        pollutants_map=dict(pipeline.pollutants_map),
        vehicle_category_metadata_file=str(pipeline.vehicle_category_metadata_file),
        annualization_days=dict(pipeline.annualization_days),
        population_sample=float(pipeline.population_sample),
        transit_sample=float(pipeline.transit_sample),
        include_passenger=bool(pipeline.include_passenger),
        include_freight=bool(pipeline.include_freight),
        manifest_inputs=manifest_inputs,
        require_aermod_support=bool(pipeline.aermod_grid_path),
    )
    _log_step1_elapsed("1.0", "prepare skims inputs complete", skims_started, rows=len(skims_df))

    log_substep_banner("1.1", "group county, inmap, and aermod intersections separately", logger=logger)
    grouping_started = time.perf_counter()
    county_grouped_df = _build_zone_grouped_table(
        intersection_path=intersection_paths.get("county"),
        intersection_df=None,
        zone_label="county",
        scratch_dir=scratch_dir,
    )
    inmap_grouped_df = _build_zone_grouped_table(
        intersection_path=intersection_paths.get("inmap"),
        intersection_df=None,
        zone_label="inmap",
        scratch_dir=scratch_dir,
    )
    aermod_grouped_df = _build_zone_grouped_table(
        intersection_path=intersection_paths.get("aermod"),
        intersection_df=None,
        zone_label="aermod",
        scratch_dir=scratch_dir,
    )
    _log_step1_elapsed(
        "1.1",
        "intersection grouping complete",
        grouping_started,
        county_rows=0 if county_grouped_df is None else len(county_grouped_df),
        inmap_rows=0 if inmap_grouped_df is None else len(inmap_grouped_df),
        aermod_rows=0 if aermod_grouped_df is None else len(aermod_grouped_df),
    )

    log_substep_banner("1.2[county]", "allocate emissions to county surface", logger=logger)
    county_alloc_started = time.perf_counter()
    county_allocated_df = _build_zone_allocated_table(
        grouped_df=county_grouped_df,
        skims_df=skims_df,
        zone_label="county",
        scratch_dir=scratch_dir,
        step_id="1.2",
    )
    _log_step1_elapsed(
        "1.2",
        "county allocation complete",
        county_alloc_started,
        rows=0 if county_allocated_df is None else len(county_allocated_df),
    )
    log_substep_banner("1.3", "apply activity corrections", logger=logger)
    correction_started = time.perf_counter()
    county_corrected_df, beam_activity_totals, county_correction_factors = _build_county_corrected_table(
        county_allocated_df=county_allocated_df,
        county_grouped_df=county_grouped_df,
        skims_df=skims_df,
        pipeline=pipeline,
        manifest_inputs=manifest_inputs or {},
        scratch_dir=scratch_dir,
    )
    _log_step1_elapsed(
        "1.3",
        "activity corrections complete",
        correction_started,
        county_rows=0 if county_corrected_df is None else len(county_corrected_df),
        activity_rows=0 if beam_activity_totals is None else len(beam_activity_totals),
        factor_rows=0 if county_correction_factors is None else len(county_correction_factors),
    )
    corrected_totals_started = time.perf_counter()
    corrected_source_df = _build_corrected_source_totals(
        county_corrected_df,
        scratch_dir=scratch_dir,
    )
    _log_step1_elapsed(
        "1.3",
        "corrected source totals complete",
        corrected_totals_started,
        rows=0 if corrected_source_df is None else len(corrected_source_df),
    )

    log_substep_banner("1.4[inmap/aermod]", "allocate corrected totals independently to inmap and aermod surfaces", logger=logger)
    surface_alloc_started = time.perf_counter()
    inmap_allocated_df = _build_zone_allocated_table(
        grouped_df=inmap_grouped_df,
        skims_df=corrected_source_df if corrected_source_df is not None else skims_df,
        zone_label="inmap",
        scratch_dir=scratch_dir,
        step_id="1.4",
    )
    aermod_allocated_df = _build_zone_allocated_table(
        grouped_df=aermod_grouped_df,
        skims_df=corrected_source_df if corrected_source_df is not None else skims_df,
        zone_label="aermod",
        scratch_dir=scratch_dir,
        step_id="1.4",
    )
    _log_step1_elapsed(
        "1.4",
        "surface allocations complete",
        surface_alloc_started,
        inmap_rows=0 if inmap_allocated_df is None else len(inmap_allocated_df),
        aermod_rows=0 if aermod_allocated_df is None else len(aermod_allocated_df),
    )
    if aermod_allocated_df is not None and not aermod_allocated_df.empty:
        _log_aermod_urban_class_trace("post zone allocation output", aermod_allocated_df)

    if aermod_allocated_df is not None and not aermod_allocated_df.empty:
        aermod_attrs_started = time.perf_counter()
        aermod_allocated_df = aermod_allocated_df.copy()
        _require_columns(
            aermod_allocated_df,
            columns=["roadCategory", "source_release_height"],
            label="AERMOD allocated emissions table",
        )
        aermod_allocated_df["source_temporal_class"] = aermod_allocated_df.groupby("aermod_cell_id")["roadCategory"].transform(
            lambda cats: "FREEWAY" if cats.map(_OSM_TO_AERMOD_TEMPORAL).fillna("CITYSTREET").eq("FREEWAY").any() else "CITYSTREET"
        )
        aermod_allocated_df["source_release_height"] = aermod_allocated_df.groupby("aermod_cell_id")["source_release_height"].transform("max")
        _log_aermod_urban_class_trace("before source_urban_class init", aermod_allocated_df)
        aermod_allocated_df["source_urban_class"] = 0
        _log_aermod_urban_class_trace("after source_urban_class init", aermod_allocated_df)
        cell_population_path = (manifest_inputs or {}).get("aermod_cell_population")
        if cell_population_path:
            cell_population_path = resolve_required_manifest_input(manifest_inputs, key="aermod_cell_population")
            cell_pop = read_table(cell_population_path)[["aermod_cell_id", "source_urban_class"]].rename(
                columns={"source_urban_class": "cell_source_urban_class"}
            )
            cell_pop["aermod_cell_id"] = pd.to_numeric(cell_pop["aermod_cell_id"], errors="coerce")
            aermod_allocated_df["aermod_cell_id"] = pd.to_numeric(aermod_allocated_df["aermod_cell_id"], errors="coerce")
            _log_aermod_urban_class_trace("before population merge", aermod_allocated_df)
            logger.info(
                "%s trace aermod cell population rows=%d duplicate_aermod_cell_id=%d null_aermod_cell_id=%d null_cell_source_urban_class=%d sample=%s",
                _step_label("1.4"),
                len(cell_pop),
                int(cell_pop["aermod_cell_id"].duplicated().sum()),
                int(cell_pop["aermod_cell_id"].isna().sum()),
                int(cell_pop["cell_source_urban_class"].isna().sum()),
                cell_pop[["aermod_cell_id", "cell_source_urban_class"]].head(5).to_dict(orient="records"),
            )
            aermod_allocated_df = aermod_allocated_df.merge(cell_pop, on="aermod_cell_id", how="left")
            _log_aermod_urban_class_trace("after population merge", aermod_allocated_df)
            aermod_allocated_df["source_urban_class"] = (
                pd.to_numeric(aermod_allocated_df["cell_source_urban_class"], errors="coerce")
                .fillna(aermod_allocated_df["source_urban_class"])
                .fillna(0)
                .astype(int)
            )
            aermod_allocated_df = aermod_allocated_df.drop(columns=["cell_source_urban_class"])
        _log_step1_elapsed(
            "1.4",
            "AERMOD source attributes prepared",
            aermod_attrs_started,
            rows=len(aermod_allocated_df),
        )

    beam_activity_totals_path = None
    beam_activity_correction_factors_path = None
    beam_emissions_by_county_process_path = None
    log_substep_banner("1.5[county]", "write county activity correction artifacts", logger=logger)
    county_write_started = time.perf_counter()
    if beam_activity_totals is not None and not beam_activity_totals.empty:
        beam_activity_totals_path = str(raw_dir / "beam_activity_totals.parquet")
        beam_activity_totals.to_parquet(beam_activity_totals_path, index=False)
        logger.info("%s BEAM activity totals → %s", _step_label("1.5", "county"), beam_activity_totals_path)
    if county_correction_factors is not None and not county_correction_factors.empty:
        beam_activity_correction_factors_path = str(raw_dir / "beam_activity_correction_factors.parquet")
        county_correction_factors.to_parquet(beam_activity_correction_factors_path, index=False)
        logger.info(
            "%s BEAM activity correction factors → %s",
            _step_label("1.5", "county"),
            beam_activity_correction_factors_path,
        )
    if county_corrected_df is not None and not county_corrected_df.empty:
        beam_emissions_by_county_process_path = str(raw_dir / "beam_emissions_by_county_process.parquet")
        county_corrected_df.to_parquet(beam_emissions_by_county_process_path, index=False)
        logger.info(
            "%s county-intersected BEAM emissions → %s",
            _step_label("1.5", "county"),
            beam_emissions_by_county_process_path,
        )
    _log_step1_elapsed(
        "1.5",
        "county artifacts write complete",
        county_write_started,
        emissions_rows=0 if county_corrected_df is None else len(county_corrected_df),
    )

    beam_emissions_for_aermod_path = None

    if pipeline.aermod_grid_path and aermod_allocated_df is not None and not aermod_allocated_df.empty:
        if not pipeline.aermod_grid_id:
            raise ValueError("pipeline.aermod_grid_id must be configured before writing AERMOD emissions.")
        log_substep_banner("1.6[aermod]", "write AERMOD emissions table", logger=logger)
        aermod_write_started = time.perf_counter()
        aermod_export_df = _aggregate_aermod_emissions_for_export(
            aermod_allocated_df,
            scratch_dir=scratch_dir,
        )
        beam_emissions_for_aermod_stem = raw_dir / "beam_emissions_for_aermod"
        _save_grid_emissions(
            aermod_export_df,
            left_col="aermod_cell_id",
            right_col=str(pipeline.aermod_grid_id),
            grid_path=pipeline.aermod_grid_path,
            output_epsg=int(pipeline.output_epsg),
            output_stem=beam_emissions_for_aermod_stem,
            step_id="1.6",
        )
        beam_emissions_for_aermod_path = str(beam_emissions_for_aermod_stem) + ".parquet"
        logger.info(
            "%s BEAM emissions for AERMOD → %s",
            _step_label("1.6", "aermod"),
            beam_emissions_for_aermod_path,
        )
        _log_step1_elapsed(
            "1.6",
            "AERMOD emissions export complete",
            aermod_write_started,
            source_rows=len(aermod_export_df),
            path=beam_emissions_for_aermod_path,
        )

    beam_emissions_for_inmap_path = None
    beam_inmap_study_area_grid_path = None
    if inmap_allocated_df is not None and not inmap_allocated_df.empty:
        if "grid_id" not in pipeline.mapping_columns or not str(pipeline.mapping_columns["grid_id"]).strip():
            raise ValueError("pipeline.mapping_columns.grid_id must be configured before writing InMAP emissions.")
        log_substep_banner("1.7[inmap]", "write InMAP emissions table", logger=logger)
        inmap_write_started = time.perf_counter()
        beam_emissions_for_inmap_stem = raw_dir / "beam_emissions_for_inmap"
        inmap_study_area_grid = _save_grid_emissions(
            inmap_allocated_df,
            left_col="inmap_cell_id",
            right_col=str(pipeline.mapping_columns["grid_id"]),
            grid_path=pipeline.inmap_grid_path,
            output_epsg=int(pipeline.output_epsg),
            output_stem=beam_emissions_for_inmap_stem,
            step_id="1.7",
        )
        beam_inmap_study_area_grid_path = str(raw_dir / "beam_inmap_study_area_grid.gpkg")
        inmap_study_area_grid.to_file(beam_inmap_study_area_grid_path, driver="GPKG")
        logger.info("%s InMAP study area grid → %s", _step_label("1.7", "inmap"), beam_inmap_study_area_grid_path)
        beam_emissions_for_inmap_path = str(beam_emissions_for_inmap_stem) + ".parquet"
        logger.info(
            "%s BEAM emissions for InMAP → %s",
            _step_label("1.7", "inmap"),
            beam_emissions_for_inmap_path,
        )
        _log_step1_elapsed(
            "1.7",
            "InMAP emissions export complete",
            inmap_write_started,
            source_rows=len(inmap_allocated_df),
            path=beam_emissions_for_inmap_path,
        )

    _log_step1_elapsed("1", "process emissions complete", step_started)

    return {
        "beam_activity_totals": beam_activity_totals_path,
        "beam_activity_correction_factors": beam_activity_correction_factors_path,
        "beam_emissions_by_county_process": beam_emissions_by_county_process_path,
        "beam_emissions_for_aermod": beam_emissions_for_aermod_path,
        "beam_emissions_for_inmap": beam_emissions_for_inmap_path,
        "beam_inmap_study_area_grid": beam_inmap_study_area_grid_path,
    }
