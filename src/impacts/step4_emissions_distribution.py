"""Step 4 — Combined emissions distribution path."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict
from typing import Optional

import geopandas as gpd
import pandas as pd
import duckdb
from impacts.emissions.emissions_grid_mapping import apply_county_corrections

from .manifest_models import PipelineConfig

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
            f"Step 4 grid export mismatch for {left_col}: expected {len(expected_grid_ids)} grid ids, "
            f"got {len(actual_grid_ids)}. sample_missing={missing} sample_extra={extra}"
        )
    missing_geometry = int(geo.geometry.isna().sum())
    if missing_geometry:
        raise ValueError(
            f"Step 4 grid export missing geometry for {missing_geometry} rows in {output_stem}"
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
    return f"Step 4.{step}{suffix}"


def _existing_output(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _reuse_existing_step4_outputs(raw_dir: Path) -> Optional[Dict[str, Optional[str]]]:
    beam_emissions_for_inmap = _existing_output(raw_dir / "beam_emissions_for_inmap.parquet")
    if not beam_emissions_for_inmap:
        return None

    outputs = {
        "beam_emissions_for_aermod": _existing_output(raw_dir / "beam_emissions_for_aermod.parquet"),
        "beam_emissions_for_inmap": beam_emissions_for_inmap,
    }
    logger.info(
        "%s reusing existing emissions outputs; skipping Step 4 recomputation (inmap=%s, aermod=%s)",
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
            f"{_step_label('1')} requires canonical Step 3 columns. Missing: {missing}"
        )

    metric_cols = [
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
    aermod_prop = "aermod_zone_edge_proportion"
    inmap_prop = "inmap_zone_edge_proportion"

    metric_cols = [
        col for col in grouped_df.columns
        if (col.startswith("aermod_") or col.startswith("inmap_")) and any(tag in col for tag in ("_proportion", "_length_m", "_surface_m2"))
    ]

    con = duckdb.connect(database=":memory:")
    try:
        con.register("grouped_df", grouped_df)
        con.register("skims_df", skims_df[["linkId", "vehicleTypeId", "process"] + emission_cols].copy())
        metric_select = ",\n                ".join([f"g.{col}" for col in metric_cols])
        aermod_alloc = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{aermod_prop} AS DOUBLE), 0.0) AS {col}_aermod_allocated"
            for col in emission_cols
        ]) if aermod_prop else ""
        inmap_alloc = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{inmap_prop} AS DOUBLE), 0.0) AS {col}_inmap_allocated"
            for col in emission_cols
        ]) if inmap_prop else ""
        extra_select = ",\n                ".join([part for part in [metric_select, aermod_alloc, inmap_alloc] if part])
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
    zone_emission_cols = [col for col in combined_df.columns if col.startswith("tons_per_year_") and col.endswith(f"_{zone_label}_allocated")]
    keep_cols = [col for col in ["linkId", "vehicleTypeId", "process", cell_col] if col in combined_df.columns]
    if "countyfp" in combined_df.columns and "countyfp" not in keep_cols:
        keep_cols.append("countyfp")
    keep_cols.extend(zone_metric_cols)
    keep_cols.extend(zone_emission_cols)
    keep_cols = [col for i, col in enumerate(keep_cols) if col not in keep_cols[:i]]
    zone_df = combined_df[keep_cols].copy()
    zone_df = zone_df[zone_df[cell_col].notna()].copy()
    if zone_df.empty:
        return None

    sum_cols = [col for col in zone_df.columns if col not in {"linkId", "vehicleTypeId", "process", cell_col, "countyfp"}]
    group_cols = [col for col in ["linkId", "vehicleTypeId", "process", cell_col, "countyfp"] if col in zone_df.columns]
    zone_df = zone_df.groupby(group_cols, dropna=False)[sum_cols].sum().reset_index()
    return zone_df


def _build_combined_corrected_table(
    *,
    allocated_df: pd.DataFrame,
    pipeline: PipelineConfig,
) -> Optional[pd.DataFrame]:
    if allocated_df is None or allocated_df.empty:
        return None

    if pipeline.activity_corrections_path:
        logger.info(
            "%s applying county corrections from %s",
            _step_label("3"),
            pipeline.activity_corrections_path,
        )
        corrected = apply_county_corrections(
            allocated_df,
            corrections_path=pipeline.activity_corrections_path,
            correction_columns=pipeline.activity_corrections_columns or None,
        )
    else:
        corrected = allocated_df
        logger.info("%s no corrections configured; using allocated totals as-is", _step_label("3"))
    return corrected


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
    combined_corrected_df = _build_combined_corrected_table(
        allocated_df=combined_allocated_df,
        pipeline=pipeline,
    )

    beam_emissions_for_aermod_path = None
    if pipeline.aermod_grid_path and combined_grouped_df is not None:
        aermod_corrected_df = _split_zone_allocated(
            combined_df=combined_corrected_df,
            zone_label="aermod",
            cell_col="aermod_srv_cell_id",
        )
        if aermod_corrected_df is not None and not aermod_corrected_df.empty:
            beam_emissions_for_aermod_stem = raw_dir / "beam_emissions_for_aermod"
            _save_grid_emissions(
                aermod_corrected_df,
                left_col="aermod_srv_cell_id",
                right_col="srv_cell_id",
                grid_path=pipeline.aermod_grid_path,
                output_epsg=int(pipeline.output_epsg),
                output_stem=beam_emissions_for_aermod_stem,
            )
            beam_emissions_for_aermod_path = str(beam_emissions_for_aermod_stem) + ".parquet"
            logger.info(
                "%s BEAM emissions for AERMOD → %s",
                _step_label("4", "aermod"),
                beam_emissions_for_aermod_path,
            )

    inmap_corrected_df = _split_zone_allocated(
        combined_df=combined_corrected_df,
        zone_label="inmap",
        cell_col="inmap_srm_cell_id",
    )
    beam_emissions_for_inmap_path = None
    if inmap_corrected_df is not None and not inmap_corrected_df.empty:
        beam_emissions_for_inmap_stem = raw_dir / "beam_emissions_for_inmap"
        _save_grid_emissions(
            inmap_corrected_df,
            left_col="inmap_srm_cell_id",
            right_col="srm_cell_id",
            grid_path=pipeline.inmap_grid_path,
            output_epsg=int(pipeline.output_epsg),
            output_stem=beam_emissions_for_inmap_stem,
        )
        beam_emissions_for_inmap_path = str(beam_emissions_for_inmap_stem) + ".parquet"
        logger.info(
            "%s BEAM emissions for InMAP → %s",
            _step_label("5", "inmap"),
            beam_emissions_for_inmap_path,
        )

    return {
        "beam_emissions_for_aermod": beam_emissions_for_aermod_path,
        "beam_emissions_for_inmap": beam_emissions_for_inmap_path,
    }
