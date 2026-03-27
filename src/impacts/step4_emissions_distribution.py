"""Step 4 — Combined emissions distribution path."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict
from typing import Optional

import geopandas as gpd
import pandas as pd
import duckdb
import pyarrow.parquet as pq

from .contract_utils import parquet_available
from impacts.emissions.emissions_grid_mapping import apply_county_corrections

from .manifest_models import PipelineConfig

logger = logging.getLogger(__name__)
_INTERSECTION_METRIC_SUFFIXES = (
    "_proportion",
    "_length_m",
    "_surface_m2",
)


def _table_path(parent: Path, stem: str) -> Path:
    suffix = ".parquet" if parquet_available() else ".csv.gz"
    path = parent / f"{stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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


def _write_table(df: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lower = target.name.lower()
    if lower.endswith(".parquet"):
        df.to_parquet(target, index=False, engine="pyarrow", compression="snappy")
        return
    if lower.endswith(".csv.gz"):
        df.to_csv(target, index=False, compression="gzip")
        return
    if lower.endswith(".csv"):
        df.to_csv(target, index=False)
        return
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
    joined = df.merge(
        grid_gdf[[right_col, "geometry"]],
        how="left",
        left_on=left_col,
        right_on=right_col,
    )
    if right_col != left_col:
        joined = joined.drop(columns=[right_col])
    geo = gpd.GeoDataFrame(joined, geometry="geometry", crs=grid_gdf.crs)
    geo.to_parquet(Path(str(output_stem) + ".parquet"), index=False)
    geo.to_file(Path(str(output_stem) + ".gpkg"), driver="GPKG")


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _intersection_zone_metric_cols(columns: list[str], zone_label: str) -> list[str]:
    prefix = f"{zone_label}_"
    return [
        col for col in columns
        if col.startswith(prefix) and any(tag in col for tag in _INTERSECTION_METRIC_SUFFIXES)
    ]


def _normalize_zone_columns(df: pd.DataFrame, zone_label: str, cell_col: str) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    prefixed_cell = f"edge_{cell_col}"
    if prefixed_cell in df.columns:
        # For road-intersection outputs, the edge-prefixed cell id is the
        # authoritative "road hit" label. The plain cell id can come from
        # carried zone-side metadata and should not be used to resurrect
        # non-intersecting cells.
        df[cell_col] = df[prefixed_cell]
        df = df.drop(columns=[prefixed_cell])
    if "edge_linkId" in df.columns:
        if "linkId" in df.columns:
            df["linkId"] = df["edge_linkId"].combine_first(df["linkId"])
        else:
            rename_map["edge_linkId"] = "linkId"
    zone_prefix = f"edge_{zone_label}_"
    for col in df.columns:
        if col.startswith(zone_prefix):
            normalized = col.removeprefix("edge_")
            if normalized in df.columns:
                df[normalized] = df[col].combine_first(df[normalized])
            else:
                rename_map[col] = normalized
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _load_intersection_subset(path: str, columns: list[str]) -> pd.DataFrame:
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        return pd.read_parquet(target, columns=columns)
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


def _intersection_columns(path: str, intersection_df: Optional[pd.DataFrame]) -> list[str]:
    if intersection_df is not None:
        return list(intersection_df.columns)
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        return list(pq.read_schema(target).names)
    return list(_read_table(target).columns)


def _step_label(step: str, zone_label: Optional[str] = None) -> str:
    suffix = f"[{zone_label}]" if zone_label else ""
    return f"Step 4.{step}{suffix}"


def _write_intermediates() -> bool:
    return (os.getenv("IMPACTS_STEP4_WRITE_INTERMEDIATES", "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _build_combined_grouped_table(
    *,
    intersection_path: str,
    raw_dir: Path,
    pipeline: PipelineConfig,
    intersection_df: Optional[pd.DataFrame],
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    intersection_cols = _intersection_columns(intersection_path, intersection_df)
    needed = {
        "linkId",
        "edge_linkId",
        "countyfp",
        "county_COUNTYFP",
        "zone_COUNTYFP",
        "COUNTYFP",
        "aermod_srv_cell_id",
        "edge_aermod_srv_cell_id",
        "inmap_srm_cell_id",
        "edge_inmap_srm_cell_id",
    }
    needed.update(_intersection_zone_metric_cols(intersection_cols, "aermod"))
    needed.update(_intersection_zone_metric_cols(intersection_cols, "inmap"))
    needed.update(
        col for col in intersection_cols
        if col.startswith("edge_aermod_") or col.startswith("edge_inmap_")
    )
    source_cols = [col for col in intersection_cols if col in needed]
    for plain_col, edge_col in (
        ("aermod_srv_cell_id", "edge_aermod_srv_cell_id"),
        ("inmap_srm_cell_id", "edge_inmap_srm_cell_id"),
    ):
        if edge_col in source_cols and plain_col in source_cols:
            source_cols = [col for col in source_cols if col != plain_col]
    intersection = _load_intersection_subset_or_df(
        path=intersection_path,
        columns=source_cols,
        intersection_df=intersection_df,
    )
    intersection = _normalize_zone_columns(intersection, "aermod", "aermod_srv_cell_id")
    intersection = _normalize_zone_columns(intersection, "inmap", "inmap_srm_cell_id")

    county_col = _first_existing_col(intersection, ["countyfp", "county_COUNTYFP", "zone_COUNTYFP", "COUNTYFP"])
    if county_col and county_col != "countyfp":
        intersection = intersection.rename(columns={county_col: "countyfp"})
        county_col = "countyfp"

    link_col = (pipeline.mapping_columns or {}).get("link_id", "edge_linkId")
    if link_col in intersection.columns and link_col != "linkId":
        intersection = intersection.rename(columns={link_col: "linkId"})
    if "linkId" not in intersection.columns:
        raise ValueError(f"{_step_label('1')} requires linkId in the combined mapping table.")

    metric_cols = _intersection_zone_metric_cols(list(intersection.columns), "aermod") + _intersection_zone_metric_cols(list(intersection.columns), "inmap")
    metric_cols = [col for i, col in enumerate(metric_cols) if col not in metric_cols[:i]]

    con = duckdb.connect(database=":memory:")
    try:
        con.register("intersection_df", intersection)
        metric_select = ",\n                ".join([f"SUM(COALESCE(i.{col}, 0.0)) AS {col}" for col in metric_cols])
        county_select = ", i.countyfp" if county_col == "countyfp" else ", NULL AS countyfp"
        county_group = ", i.countyfp" if county_col == "countyfp" else ""
        grouped = con.execute(
            f"""
            SELECT
                i.linkId,
                i.aermod_srv_cell_id,
                i.inmap_srm_cell_id
                {county_select}
                {"," if metric_select else ""} {metric_select}
            FROM intersection_df AS i
            WHERE i.aermod_srv_cell_id IS NOT NULL OR i.inmap_srm_cell_id IS NOT NULL
            GROUP BY
                i.linkId,
                i.aermod_srv_cell_id,
                i.inmap_srm_cell_id
                {county_group}
            """
        ).df()
    finally:
        con.close()

    if grouped.empty:
        return None, None

    grouped_path = raw_dir / "combined_mapping_grouped.parquet"
    if _write_intermediates():
        grouped.to_parquet(grouped_path, index=False)
        logger.info("%s combined grouped mapping rows=%d → %s", _step_label("1"), len(grouped), grouped_path)
        return grouped, str(grouped_path)
    logger.info("%s combined grouped mapping rows=%d", _step_label("1"), len(grouped))
    return grouped, None


def _build_combined_allocated_table(
    *,
    grouped_df: pd.DataFrame,
    skims_df: pd.DataFrame,
    raw_dir: Path,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    if grouped_df is None or grouped_df.empty:
        return None, None

    emission_cols = [
        c for c in skims_df.columns
        if c.startswith("tons_per_year_") and pd.api.types.is_numeric_dtype(skims_df[c])
    ]
    aermod_prop = _first_existing_col(grouped_df, ["aermod_zone_piece_proportion", "aermod_zone_edge_proportion"])
    inmap_prop = _first_existing_col(grouped_df, ["inmap_zone_piece_proportion", "inmap_zone_edge_proportion"])

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
        return None, None

    allocated_path = raw_dir / "combined_emissions_allocated.parquet"
    if _write_intermediates():
        allocated.to_parquet(allocated_path, index=False)
        logger.info("%s combined allocated emissions rows=%d → %s", _step_label("2"), len(allocated), allocated_path)
        return allocated, str(allocated_path)
    logger.info("%s combined allocated emissions rows=%d", _step_label("2"), len(allocated))
    return allocated, None


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
    raw_dir: Path,
    pipeline: PipelineConfig,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    if allocated_df is None or allocated_df.empty:
        return None, None

    corrected_path = _table_path(raw_dir, "combined_emissions_corrected")
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
    if _write_intermediates():
        _write_table(corrected, corrected_path)
        return corrected, str(corrected_path)
    return corrected, None


def run(
    pipeline: PipelineConfig,
    raw_dir: Path,
    skims_df: pd.DataFrame,
    intersection_path: str,
    intersection_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Optional[str]]:
    combined_grouped_df, combined_grouped_path = _build_combined_grouped_table(
        intersection_path=intersection_path,
        raw_dir=raw_dir,
        pipeline=pipeline,
        intersection_df=intersection_df,
    )
    combined_allocated_df, combined_allocated_path = _build_combined_allocated_table(
        grouped_df=combined_grouped_df,
        skims_df=skims_df,
        raw_dir=raw_dir,
    )
    combined_corrected_df, combined_corrected_path = _build_combined_corrected_table(
        allocated_df=combined_allocated_df,
        raw_dir=raw_dir,
        pipeline=pipeline,
    )

    aermod_grouped_path = None
    aermod_allocated_path = None
    aermod_corrected_path = None
    aermod_grid_emissions_path = None
    if pipeline.aermod_grid_path and combined_grouped_df is not None:
        aermod_grouped_df = _split_zone_allocated(
            combined_df=combined_grouped_df,
            zone_label="aermod",
            cell_col="aermod_srv_cell_id",
        )
        if aermod_grouped_df is not None and not aermod_grouped_df.empty:
            if _write_intermediates():
                grouped_stem = raw_dir / "aermod_mapping_grouped"
                _save_grid_emissions(
                    aermod_grouped_df,
                    left_col="aermod_srv_cell_id",
                    right_col="srv_cell_id",
                    grid_path=pipeline.aermod_grid_path,
                    output_epsg=int(pipeline.output_epsg),
                    output_stem=grouped_stem,
                )
                aermod_grouped_path = str(grouped_stem) + ".parquet"

        aermod_allocated_df = _split_zone_allocated(
            combined_df=combined_allocated_df,
            zone_label="aermod",
            cell_col="aermod_srv_cell_id",
        )
        if aermod_allocated_df is not None and not aermod_allocated_df.empty:
            if _write_intermediates():
                allocated_stem = raw_dir / "aermod_emissions_allocated"
                _save_grid_emissions(
                    aermod_allocated_df,
                    left_col="aermod_srv_cell_id",
                    right_col="srv_cell_id",
                    grid_path=pipeline.aermod_grid_path,
                    output_epsg=int(pipeline.output_epsg),
                    output_stem=allocated_stem,
                )
                aermod_allocated_path = str(allocated_stem) + ".parquet"
        aermod_corrected_df = _split_zone_allocated(
            combined_df=combined_corrected_df,
            zone_label="aermod",
            cell_col="aermod_srv_cell_id",
        )
        if aermod_corrected_df is not None and not aermod_corrected_df.empty:
            corrected_path = _table_path(raw_dir, "aermod_emissions_corrected")
            if _write_intermediates():
                _write_table(aermod_corrected_df, corrected_path)
                aermod_corrected_path = str(corrected_path)
            corrected_stem = raw_dir / "aermod_grid_emissions"
            _save_grid_emissions(
                aermod_corrected_df,
                left_col="aermod_srv_cell_id",
                right_col="srv_cell_id",
                grid_path=pipeline.aermod_grid_path,
                output_epsg=int(pipeline.output_epsg),
                output_stem=corrected_stem,
            )
            aermod_grid_emissions_path = str(corrected_stem) + ".parquet"
            logger.info("%s grid emissions → %s", _step_label("4", "aermod"), aermod_grid_emissions_path)

    inmap_grouped_df = _split_zone_allocated(
        combined_df=combined_grouped_df,
        zone_label="inmap",
        cell_col="inmap_srm_cell_id",
    )
    inmap_grouped_path = None
    if inmap_grouped_df is not None and not inmap_grouped_df.empty:
        if _write_intermediates():
            grouped_stem = raw_dir / "inmap_mapping_grouped"
            _save_grid_emissions(
                inmap_grouped_df,
                left_col="inmap_srm_cell_id",
                right_col="srm_cell_id",
                grid_path=pipeline.inmap_grid_path,
                output_epsg=int(pipeline.output_epsg),
                output_stem=grouped_stem,
            )
            inmap_grouped_path = str(grouped_stem) + ".parquet"

    inmap_allocated_df = _split_zone_allocated(
        combined_df=combined_allocated_df,
        zone_label="inmap",
        cell_col="inmap_srm_cell_id",
    )
    inmap_allocated_path = None
    if inmap_allocated_df is not None and not inmap_allocated_df.empty:
        if _write_intermediates():
            allocated_stem = raw_dir / "inmap_emissions_allocated"
            _save_grid_emissions(
                inmap_allocated_df,
                left_col="inmap_srm_cell_id",
                right_col="srm_cell_id",
                grid_path=pipeline.inmap_grid_path,
                output_epsg=int(pipeline.output_epsg),
                output_stem=allocated_stem,
            )
            inmap_allocated_path = str(allocated_stem) + ".parquet"
    inmap_corrected_df = _split_zone_allocated(
        combined_df=combined_corrected_df,
        zone_label="inmap",
        cell_col="inmap_srm_cell_id",
    )
    inmap_corrected_path = None
    inmap_grid_emissions_path = None
    if inmap_corrected_df is not None and not inmap_corrected_df.empty:
        corrected_path = _table_path(raw_dir, "inmap_emissions_corrected")
        if _write_intermediates():
            _write_table(inmap_corrected_df, corrected_path)
            inmap_corrected_path = str(corrected_path)
        corrected_stem = raw_dir / "inmap_grid_emissions"
        _save_grid_emissions(
            inmap_corrected_df,
            left_col="inmap_srm_cell_id",
            right_col="srm_cell_id",
            grid_path=pipeline.inmap_grid_path,
            output_epsg=int(pipeline.output_epsg),
            output_stem=corrected_stem,
        )
        inmap_grid_emissions_path = str(corrected_stem) + ".parquet"
        logger.info("%s grid emissions → %s", _step_label("5", "inmap"), inmap_grid_emissions_path)

    return {
        "combined_mapping_grouped": combined_grouped_path,
        "combined_emissions_allocated": combined_allocated_path,
        "combined_emissions_corrected": combined_corrected_path,
        "aermod_mapping_grouped": aermod_grouped_path,
        "aermod_emissions_allocated": aermod_allocated_path,
        "aermod_emissions_corrected": aermod_corrected_path,
        "aermod_grid_emissions": aermod_grid_emissions_path,
        "inmap_mapping_grouped": inmap_grouped_path,
        "inmap_emissions_allocated": inmap_allocated_path,
        "inmap_emissions_corrected": inmap_corrected_path,
        "inmap_grid_emissions": inmap_grid_emissions_path,
        "emissions_allocated": inmap_allocated_path,
        "emissions_corrected": inmap_grid_emissions_path,
    }
