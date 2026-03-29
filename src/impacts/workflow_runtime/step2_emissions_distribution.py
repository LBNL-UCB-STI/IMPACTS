"""Step 2 — Combined emissions distribution path."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import geopandas as gpd
import pandas as pd
import duckdb
import numpy as np

from ..config.defaults import chunk_size as default_chunk_size

from ..manifest.schema import PipelineConfig

logger = logging.getLogger(__name__)


def _load_grid_geometries(grid_path: str) -> gpd.GeoDataFrame:
    path = Path(grid_path)
    if path.suffix.lower() == ".parquet":
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def _read_table(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    lower = target.name.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(target)
    if lower.endswith(".csv.gz"):
        return pd.read_csv(target, compression="gzip")
    if lower.endswith(".csv"):
        return pd.read_csv(target)
    raise ValueError(f"Unsupported table format: {target}")


def _first_existing(df: pd.DataFrame, candidates) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _resolve_column_config(config: Optional[Dict[str, str]], defaults: Dict[str, str]) -> Dict[str, str]:
    resolved = defaults.copy()
    if config:
        resolved.update({k: v for k, v in config.items() if v})
    return resolved


def _normalize_county_fips(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d+)")[0]
    normalized = extracted.where(extracted.isna(), extracted.str.zfill(3))
    return normalized.astype("string")

_COUNTY_FIPS_CANDIDATES = ["county_zone_COUNTYFP", "county_COUNTYFP", "zone_COUNTYFP", "COUNTYFP", "countyfp"]
_COUNTY_CORRECTION_COLUMNS = {
    "county_fips": "countyfp",
    "tot_vmt": "totVMT",
    "tot_trips": "totTrips",
}
_VMT_PROCESSES = {"RUNEX", "PMBW", "PMTW", "PRDUST", "RUNLOSS"}
_TRIP_PROCESSES = {"HOTSOAK", "DIURN", "STREX"}
_METERS_PER_MILE = 1609.344


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
            _step_label("3"),
            label,
            int(invalid.sum()),
        )
    return ratio


def _build_beam_activity_totals(
    allocated_df: pd.DataFrame,
    *,
    county_col: str,
) -> pd.DataFrame:
    county_activity = allocated_df[[county_col, "totVMT_county_allocated", "totTrips_county_allocated"]].copy()
    county_activity["_fips_norm"] = _normalize_county_fips(county_activity[county_col])
    county_activity["totVMT_county_allocated"] = pd.to_numeric(county_activity["totVMT_county_allocated"], errors="coerce").fillna(0.0)
    county_activity["totTrips_county_allocated"] = pd.to_numeric(county_activity["totTrips_county_allocated"], errors="coerce").fillna(0.0)
    grouped = county_activity.groupby("_fips_norm", dropna=False).agg(
        totVMT=("totVMT_county_allocated", "sum"),
        totTrips=("totTrips_county_allocated", "sum"),
    ).reset_index()
    grouped = grouped.rename(columns={"_fips_norm": "countyfp"})
    zero_null_mask = grouped["countyfp"].isna() & grouped["totVMT"].eq(0.0) & grouped["totTrips"].eq(0.0)
    if zero_null_mask.any():
        grouped = grouped.loc[~zero_null_mask].reset_index(drop=True)
    return grouped


def _derive_county_correction_factors(
    beam_activity_totals: pd.DataFrame,
    corrections_path: str,
    *,
    correction_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    columns = _resolve_column_config(correction_columns, _COUNTY_CORRECTION_COLUMNS)
    corrections = _read_table(corrections_path)
    fips_source_col = _first_existing(corrections, [columns["county_fips"]])
    if fips_source_col is None:
        raise ValueError(
            f"County corrections file must include county FIPS column: {columns['county_fips']}."
        )

    required_targets = {
        "totVMT": columns["tot_vmt"],
        "totTrips": columns["tot_trips"],
    }
    missing = [target_col for target_col in required_targets.values() if target_col not in corrections.columns]
    if missing:
        raise ValueError(f"County corrections file missing required activity total columns: {missing}")

    targets = corrections.copy()
    targets["countyfp"] = _normalize_county_fips(targets[fips_source_col])
    for output_col, source_col in required_targets.items():
        targets[output_col] = pd.to_numeric(targets[source_col], errors="coerce").fillna(0.0)
    targets = targets[["countyfp", "totVMT", "totTrips"]].drop_duplicates("countyfp")

    merged = targets.merge(
        beam_activity_totals.rename(
            columns={
                "totVMT": "totVMT_source",
                "totTrips": "totTrips_source",
            }
        ),
        how="left",
        on="countyfp",
    )
    for col in ["totVMT_source", "totTrips_source"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["factor_totVMT"] = _safe_ratio(merged["totVMT"], merged["totVMT_source"], label="totVMT")
    merged["factor_totTrips"] = _safe_ratio(merged["totTrips"], merged["totTrips_source"], label="totTrips")
    return merged


def apply_county_corrections(
    allocated_df: pd.DataFrame,
    county_correction_factors: pd.DataFrame,
    *,
    county_col: str,
) -> pd.DataFrame:
    result = allocated_df.copy()
    result["_fips_norm"] = _normalize_county_fips(result[county_col])
    result = result.merge(
        county_correction_factors[["countyfp", "factor_totVMT", "factor_totTrips"]].rename(columns={"countyfp": "_fips_norm"}),
        how="left",
        on="_fips_norm",
    )
    result["factor_totVMT"] = pd.to_numeric(result["factor_totVMT"], errors="coerce").fillna(1.0)
    result["factor_totTrips"] = pd.to_numeric(result["factor_totTrips"], errors="coerce").fillna(1.0)

    process_upper = result.get("process", pd.Series("", index=result.index)).astype(str).str.upper()
    unique_processes = sorted(process_upper.dropna().unique().tolist())
    vmt_used = sorted([proc for proc in unique_processes if proc in _VMT_PROCESSES])
    trip_used = sorted([proc for proc in unique_processes if proc in _TRIP_PROCESSES])
    neutral_used = sorted([proc for proc in unique_processes if proc not in _VMT_PROCESSES and proc not in _TRIP_PROCESSES])
    if vmt_used:
        logger.info("%s using factor_totVMT for processes: %s", _step_label("4"), ", ".join(vmt_used))
    if trip_used:
        logger.info("%s using factor_totTrips for processes: %s", _step_label("4"), ", ".join(trip_used))
    if neutral_used:
        logger.warning(
            "%s no county correction factor mapping for processes; using neutral factor 1.0: %s",
            _step_label("4"),
            ", ".join(neutral_used),
        )

    factor_arr = np.ones(len(result), dtype=np.float32)
    factor_arr = np.where(process_upper.isin(_VMT_PROCESSES), result["factor_totVMT"].to_numpy(dtype=np.float32), factor_arr)
    factor_arr = np.where(process_upper.isin(_TRIP_PROCESSES), result["factor_totTrips"].to_numpy(dtype=np.float32), factor_arr)

    emission_allocated_cols = [
        c for c in result.columns
        if c.startswith("tons_per_year_") and c.endswith("_allocated")
    ]
    for col in emission_allocated_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) * factor_arr

    return result.drop(columns=["_fips_norm", "factor_totVMT", "factor_totTrips"])


def _save_grid_emissions(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    grid_path: str,
    output_epsg: int,
    output_stem: Path,
) -> None:
    grid_gdf = _load_grid_geometries(grid_path)
    if grid_gdf.crs is not None:
        grid_gdf = grid_gdf.to_crs(epsg=output_epsg)
    expected_grid_ids = set(pd.to_numeric(df[left_col], errors="coerce").dropna().astype(int).unique().tolist())
    joined = df.merge(
        grid_gdf[[right_col, "geometry"]],
        how="left",
        left_on=left_col,
        right_on=right_col,
    )
    if right_col != left_col:
        joined = joined.drop(columns=[right_col])
    geo = gpd.GeoDataFrame(joined, geometry="geometry", crs=grid_gdf.crs)
    actual_grid_ids = set(pd.to_numeric(geo[left_col], errors="coerce").dropna().astype(int).unique().tolist())
    if actual_grid_ids != expected_grid_ids:
        missing = sorted(expected_grid_ids - actual_grid_ids)[:10]
        extra = sorted(actual_grid_ids - expected_grid_ids)[:10]
        raise ValueError(
            f"Step 2 grid export mismatch for {left_col}: expected {len(expected_grid_ids)} grid ids, "
            f"got {len(actual_grid_ids)}. sample_missing={missing} sample_extra={extra}"
        )
    missing_geometry = int(geo.geometry.isna().sum())
    if missing_geometry:
        raise ValueError(
            f"Step 2 grid export missing geometry for {missing_geometry} rows in {output_stem}"
        )
    geo.to_parquet(Path(str(output_stem) + ".parquet"), index=False)
    geo.to_file(Path(str(output_stem) + ".gpkg"), driver="GPKG")


def _load_intersection_subset(path: str, columns: list[str]) -> pd.DataFrame:
    target = Path(path)
    frame = _read_table(target)
    return frame[columns].copy()


def _load_intersection_subset_or_df(
    *,
    path: str,
    columns: list[str],
    intersection_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if intersection_df is not None:
        return intersection_df[columns].copy()
    return _load_intersection_subset(path, columns)

def _step_label(step: str, zone_label: Optional[str] = None) -> str:
    suffix = f"[{zone_label}]" if zone_label else ""
    return f"Step 2.{step}{suffix}"


def _existing_output(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _reuse_existing_step4_outputs(raw_dir: Path) -> Optional[Dict[str, Optional[str]]]:
    beam_emissions_for_inmap = _existing_output(raw_dir / "beam_emissions_for_inmap.parquet")
    if not beam_emissions_for_inmap:
        return None

    outputs = {
        "beam_activity_totals": _existing_output(raw_dir / "beam_activity_totals.parquet"),
        "beam_activity_correction_factors": _existing_output(raw_dir / "beam_activity_correction_factors.parquet"),
        "beam_emissions_for_aermod": _existing_output(raw_dir / "beam_emissions_for_aermod.parquet"),
        "beam_emissions_for_inmap": beam_emissions_for_inmap,
    }
    logger.info(
        "%s reusing existing emissions outputs; skipping Step 2 recomputation (inmap=%s, aermod=%s)",
        _step_label("0"),
        outputs["beam_emissions_for_inmap"],
        outputs["beam_emissions_for_aermod"],
    )
    return outputs


def _build_combined_grouped_table(
    *,
    intersection_path: str,
    intersection_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    required_cols = {
        "linkId",
        "countyfp",
        "county_zone_edge_proportion",
        "county_edge_link_length_m",
        "county_zone_link_length_m",
        "aermod_srv_cell_id",
        "inmap_srm_cell_id",
        "aermod_zone_edge_proportion",
        "aermod_edge_link_length_m",
        "aermod_zone_link_length_m",
        "inmap_zone_edge_proportion",
        "inmap_edge_link_length_m",
        "inmap_zone_link_length_m",
    }
    intersection = _load_intersection_subset_or_df(
        path=intersection_path,
        columns=list(required_cols),
        intersection_df=intersection_df,
    )
    missing = [col for col in required_cols if col not in intersection.columns]
    if missing:
        raise ValueError(
            f"{_step_label('1')} requires canonical Step 1 columns. Missing: {missing}"
        )

    metric_cols = [
        "county_zone_edge_proportion",
        "county_edge_link_length_m",
        "county_zone_link_length_m",
        "aermod_zone_edge_proportion",
        "aermod_edge_link_length_m",
        "aermod_zone_link_length_m",
        "inmap_zone_edge_proportion",
        "inmap_edge_link_length_m",
        "inmap_zone_link_length_m",
    ]

    con = duckdb.connect(database=":memory:")
    try:
        con.register("intersection_df", intersection)
        metric_select = ",\n                ".join([f"SUM(COALESCE(i.{col}, 0.0)) AS {col}" for col in metric_cols])
        grouped = con.execute(
            f"""
            SELECT
                i.linkId,
                i.aermod_srv_cell_id,
                i.inmap_srm_cell_id
                , i.countyfp
                {"," if metric_select else ""} {metric_select}
            FROM intersection_df AS i
            WHERE i.aermod_srv_cell_id IS NOT NULL OR i.inmap_srm_cell_id IS NOT NULL
            GROUP BY
                i.linkId,
                i.aermod_srv_cell_id,
                i.inmap_srm_cell_id
                , i.countyfp
            """
        ).df()
    finally:
        con.close()

    if grouped.empty:
        return None

    logger.info("%s BEAM mapping across grids rows=%d", _step_label("1"), len(grouped))
    return grouped


def _build_combined_allocated_table(
    *,
    grouped_df: pd.DataFrame,
    skims_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    if grouped_df is None or grouped_df.empty:
        return None

    emission_cols = [
        c for c in skims_df.columns
        if c.startswith("tons_per_year_") and pd.api.types.is_numeric_dtype(skims_df[c])
    ]
    activity_cols = [
        c for c in ("totVMT", "totTrips")
        if c in skims_df.columns and pd.api.types.is_numeric_dtype(skims_df[c])
    ]
    county_prop = "county_zone_edge_proportion"
    aermod_prop = "aermod_zone_edge_proportion"
    inmap_prop = "inmap_zone_edge_proportion"

    metric_cols = [
        col for col in grouped_df.columns
        if (col.startswith("aermod_") or col.startswith("inmap_")) and any(tag in col for tag in ("_proportion", "_length_m", "_surface_m2"))
    ]

    con = duckdb.connect(database=":memory:")
    try:
        con.register("grouped_df", grouped_df)
        con.register("skims_df", skims_df[["linkId", "vehicleTypeId", "process"] + activity_cols + emission_cols].copy())
        metric_select = ",\n                ".join([f"g.{col}" for col in metric_cols])
        county_activity = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{county_prop} AS DOUBLE), 0.0) AS {col}_county_allocated"
            for col in activity_cols
        ]) if county_prop else ""
        county_alloc = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{county_prop} AS DOUBLE), 0.0) AS {col}_county_allocated"
            for col in emission_cols
        ]) if county_prop else ""
        aermod_activity = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{aermod_prop} AS DOUBLE), 0.0) AS {col}_aermod_allocated"
            for col in activity_cols
        ]) if aermod_prop else ""
        inmap_activity = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{inmap_prop} AS DOUBLE), 0.0) AS {col}_inmap_allocated"
            for col in activity_cols
        ]) if inmap_prop else ""
        aermod_alloc = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{aermod_prop} AS DOUBLE), 0.0) AS {col}_aermod_allocated"
            for col in emission_cols
        ]) if aermod_prop else ""
        inmap_alloc = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{inmap_prop} AS DOUBLE), 0.0) AS {col}_inmap_allocated"
            for col in emission_cols
        ]) if inmap_prop else ""
        extra_select = ",\n                ".join([
            part
            for part in [
                metric_select,
                county_activity,
                aermod_activity,
                inmap_activity,
                county_alloc,
                aermod_alloc,
                inmap_alloc,
            ]
            if part
        ])
        allocated = con.execute(
            f"""
            SELECT
                g.linkId,
                s.vehicleTypeId,
                s.process,
                g.aermod_srv_cell_id,
                g.inmap_srm_cell_id,
                g.countyfp
                {"," if extra_select else ""} {extra_select}
            FROM grouped_df AS g
            LEFT JOIN skims_df AS s
                ON g.linkId = s.linkId
            """
        ).df()
    finally:
        con.close()

    if allocated.empty:
        return None

    logger.info("%s BEAM emissions allocated across grids rows=%d", _step_label("2"), len(allocated))
    return allocated


def _split_zone_allocated(
    *,
    combined_df: pd.DataFrame,
    zone_label: str,
    cell_col: str,
) -> Optional[pd.DataFrame]:
    if combined_df is None or combined_df.empty or cell_col not in combined_df.columns:
        return None
    zone_metric_cols = [
        col for col in combined_df.columns
        if col.startswith(f"{zone_label}_") and col != cell_col
    ]
    zone_allocated_cols = [
        col for col in combined_df.columns
        if col.endswith(f"_{zone_label}_allocated")
    ]
    keep_cols = [col for col in ["linkId", "vehicleTypeId", "process", cell_col] if col in combined_df.columns]
    if "countyfp" in combined_df.columns and "countyfp" not in keep_cols:
        keep_cols.append("countyfp")
    keep_cols.extend(zone_metric_cols)
    keep_cols.extend(zone_allocated_cols)
    keep_cols = [col for i, col in enumerate(keep_cols) if col not in keep_cols[:i]]
    zone_df = combined_df[keep_cols].copy()
    zone_df = zone_df[zone_df[cell_col].notna()].copy()
    if zone_df.empty:
        return None

    sum_cols = [col for col in zone_df.columns if col not in {"linkId", "vehicleTypeId", "process", cell_col, "countyfp"}]
    group_cols = [col for col in ["linkId", "vehicleTypeId", "process", cell_col, "countyfp"] if col in zone_df.columns]
    zone_df = zone_df.groupby(group_cols, dropna=False)[sum_cols].sum().reset_index()
    return zone_df


def _split_zone_outputs(
    *,
    zone_df: pd.DataFrame,
    zone_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    activity_suffixes = {
        f"totVMT_{zone_label}_allocated",
        f"totTrips_{zone_label}_allocated",
    }
    emissions_cols = [col for col in zone_df.columns if col not in activity_suffixes]
    activity_cols = [
        col for col in zone_df.columns
        if not col.startswith("tons_per_year_") or col in activity_suffixes
    ]

    emissions_df = zone_df[emissions_cols].copy()
    activity_df = zone_df[activity_cols].copy()
    return emissions_df, activity_df


def _build_combined_corrected_table(
    *,
    allocated_df: pd.DataFrame,
    pipeline: PipelineConfig,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if allocated_df is None or allocated_df.empty:
        return None, None, None

    county_col = _first_existing(allocated_df, _COUNTY_FIPS_CANDIDATES)
    if county_col is None:
        raise ValueError(
            "Allocated DataFrame must include a county FIPS column. "
            f"Expected one of: {_COUNTY_FIPS_CANDIDATES}."
        )

    if pipeline.activity_totals_file:
        logger.info(
            "%s deriving county activity correction factors from %s",
            _step_label("3"),
            pipeline.activity_totals_file,
        )
        beam_activity_totals = _build_beam_activity_totals(
            allocated_df,
            county_col=county_col,
        )
        county_correction_factors = _derive_county_correction_factors(
            beam_activity_totals,
            corrections_path=pipeline.activity_totals_file,
            correction_columns=pipeline.activity_totals_columns or None,
        )
        logger.info("%s correcting allocated emissions by county/process factors", _step_label("4"))
        corrected = apply_county_corrections(
            allocated_df,
            county_correction_factors,
            county_col=county_col,
        )
    else:
        corrected = allocated_df
        beam_activity_totals = None
        county_correction_factors = None
        logger.info("%s no corrections configured; using allocated totals as-is", _step_label("4"))
    return corrected, beam_activity_totals, county_correction_factors


def run(
    pipeline: PipelineConfig,
    raw_dir: Path,
    skims_df: pd.DataFrame,
    intersection_path: str,
    intersection_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Optional[str]]:
    reused = _reuse_existing_step4_outputs(raw_dir)
    if reused is not None:
        return reused

    combined_grouped_df = _build_combined_grouped_table(
        intersection_path=intersection_path,
        intersection_df=intersection_df,
    )
    combined_allocated_df = _build_combined_allocated_table(
        grouped_df=combined_grouped_df,
        skims_df=skims_df,
    )
    combined_corrected_df, beam_activity_totals, county_correction_factors = _build_combined_corrected_table(
        allocated_df=combined_allocated_df,
        pipeline=pipeline,
    )

    beam_activity_totals_path = None
    beam_activity_correction_factors_path = None
    if beam_activity_totals is not None and not beam_activity_totals.empty:
        beam_activity_totals_path = str(raw_dir / "beam_activity_totals.parquet")
        beam_activity_totals.to_parquet(beam_activity_totals_path, index=False)
        logger.info("%s BEAM activity totals → %s", _step_label("3"), beam_activity_totals_path)
    if county_correction_factors is not None and not county_correction_factors.empty:
        beam_activity_correction_factors_path = str(raw_dir / "beam_activity_correction_factors.parquet")
        county_correction_factors.to_parquet(beam_activity_correction_factors_path, index=False)
        logger.info("%s BEAM activity correction factors → %s", _step_label("3"), beam_activity_correction_factors_path)

    beam_emissions_for_aermod_path = None

    if pipeline.aermod_grid_path and combined_grouped_df is not None:
        aermod_corrected_df = _split_zone_allocated(
            combined_df=combined_corrected_df,
            zone_label="aermod",
            cell_col="aermod_srv_cell_id",
        )
        if aermod_corrected_df is not None and not aermod_corrected_df.empty:
            aermod_emissions_df, _ = _split_zone_outputs(
                zone_df=aermod_corrected_df,
                zone_label="aermod",
            )
            beam_emissions_for_aermod_stem = raw_dir / "beam_emissions_for_aermod"
            _save_grid_emissions(
                aermod_emissions_df,
                left_col="aermod_srv_cell_id",
                right_col="srv_cell_id",
                grid_path=pipeline.aermod_grid_path,
                output_epsg=int(pipeline.output_epsg),
                output_stem=beam_emissions_for_aermod_stem,
            )
            beam_emissions_for_aermod_path = str(beam_emissions_for_aermod_stem) + ".parquet"
            logger.info(
                "%s BEAM emissions for AERMOD → %s",
                _step_label("5", "aermod"),
                beam_emissions_for_aermod_path,
            )

    inmap_corrected_df = _split_zone_allocated(
        combined_df=combined_corrected_df,
        zone_label="inmap",
        cell_col="inmap_srm_cell_id",
    )
    beam_emissions_for_inmap_path = None
    if inmap_corrected_df is not None and not inmap_corrected_df.empty:
        inmap_emissions_df, _ = _split_zone_outputs(
            zone_df=inmap_corrected_df,
            zone_label="inmap",
        )
        beam_emissions_for_inmap_stem = raw_dir / "beam_emissions_for_inmap"
        _save_grid_emissions(
            inmap_emissions_df,
            left_col="inmap_srm_cell_id",
            right_col="srm_cell_id",
            grid_path=pipeline.inmap_grid_path,
            output_epsg=int(pipeline.output_epsg),
            output_stem=beam_emissions_for_inmap_stem,
        )
        beam_emissions_for_inmap_path = str(beam_emissions_for_inmap_stem) + ".parquet"
        logger.info(
            "%s BEAM emissions for InMAP → %s",
            _step_label("6", "inmap"),
            beam_emissions_for_inmap_path,
        )

    return {
        "beam_activity_totals": beam_activity_totals_path,
        "beam_activity_correction_factors": beam_activity_correction_factors_path,
        "beam_emissions_for_aermod": beam_emissions_for_aermod_path,
        "beam_emissions_for_inmap": beam_emissions_for_inmap_path,
    }
