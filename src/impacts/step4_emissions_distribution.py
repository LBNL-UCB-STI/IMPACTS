"""Step 4 — Emissions distribution.

Allocate annualized skims emissions to the labeled grid intersection,
apply optional county-level correction factors, then collapse to
per-grid-cell totals for AERMOD and inMAP.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict
from typing import Optional

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

from .contract_utils import parquet_available
from .manifest_models import PipelineConfig

logger = logging.getLogger(__name__)
_INTERSECTION_METRIC_SUFFIXES = (
    "_proportion",
    "_length_m",
    "_surface_m2",
)
_ZONE_BRANCH_INDEX = {
    "aermod": "4.1",
    "inmap": "4.2",
}


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


def _zone_emission_cols(df: pd.DataFrame, zone_label: str) -> list[str]:
    suffix = f"_{zone_label}_allocated"
    return [c for c in df.columns if c.startswith("tons_per_year_") and c.endswith(suffix)]


def _step_label(step: str, zone_label: str) -> str:
    branch = _ZONE_BRANCH_INDEX.get(zone_label, "4.0")
    return f"Step {branch}.{step}[{zone_label}]"


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
        if cell_col in df.columns:
            df[cell_col] = df[prefixed_cell].combine_first(df[cell_col])
        else:
            rename_map[prefixed_cell] = cell_col
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


def _build_grouped_zone_table(
    *,
    intersection_path: str,
    raw_dir: Path,
    zone_label: str,
    cell_col: str,
    grid_path: str,
    grid_id_col: str,
    pipeline: PipelineConfig,
    intersection_df: Optional[pd.DataFrame],
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    import duckdb

    intersection_cols = _intersection_columns(intersection_path, intersection_df)
    normalized_cols = set(intersection_cols)
    normalized_cols.update(
        col.removeprefix("edge_")
        for col in intersection_cols
        if col == f"edge_{cell_col}" or col == "edge_linkId" or col.startswith(f"edge_{zone_label}_")
    )
    if cell_col not in normalized_cols:
        return None, None
    county_col = _first_existing_col(pd.DataFrame(columns=list(normalized_cols)), ["countyfp", "county_COUNTYFP", "zone_COUNTYFP", "COUNTYFP"])
    metric_cols = _intersection_zone_metric_cols(list(normalized_cols), zone_label)
    select_cols = ["linkId" if "linkId" in normalized_cols else (pipeline.mapping_columns or {}).get("link_id", "edge_linkId"), cell_col]
    if county_col:
        select_cols.append(county_col)
    select_cols.extend(metric_cols)
    select_cols = [col for i, col in enumerate(select_cols) if col and col not in select_cols[:i]]
    source_select_cols: list[str] = []
    for col in select_cols:
        if col in intersection_cols:
            source_select_cols.append(col)
        if col == "linkId" and "edge_linkId" in intersection_cols:
            source_select_cols.append("edge_linkId")
        prefixed = f"edge_{col}"
        if prefixed in intersection_cols:
            source_select_cols.append(prefixed)
    source_select_cols = [col for i, col in enumerate(source_select_cols) if col not in source_select_cols[:i]]
    intersection = _load_intersection_subset_or_df(
        path=intersection_path,
        columns=source_select_cols,
        intersection_df=intersection_df,
    )
    intersection = _normalize_zone_columns(intersection, zone_label, cell_col)

    link_col = (pipeline.mapping_columns or {}).get("link_id", "edge_linkId")
    if link_col in intersection.columns and link_col != "linkId":
        intersection = intersection.rename(columns={link_col: "linkId"})
    if "linkId" not in intersection.columns:
        raise ValueError(f"{_step_label('1', zone_label)} requires linkId in the grouped zone tables.")
    if county_col and county_col in intersection.columns and county_col != "countyfp":
        intersection = intersection.rename(columns={county_col: "countyfp"})
        county_col = "countyfp"

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
                i.{cell_col} AS {cell_col}
                {county_select}
                {"," if metric_select else ""} {metric_select}
            FROM intersection_df AS i
            WHERE i.{cell_col} IS NOT NULL
            GROUP BY
                i.linkId,
                i.{cell_col}
                {county_group}
            """
        ).df()
    finally:
        con.close()

    if grouped.empty:
        return None, None

    grouped_stem = raw_dir / f"{zone_label}_mapping_grouped"
    _save_grid_emissions(
        grouped,
        left_col=cell_col,
        right_col=grid_id_col,
        grid_path=grid_path,
        output_epsg=int(pipeline.output_epsg),
        output_stem=grouped_stem,
    )
    grouped_path = str(grouped_stem) + ".parquet"
    logger.info("%s grouped mapping rows=%d → %s", _step_label("1", zone_label), len(grouped), grouped_path)
    return grouped, grouped_path


def _build_allocated_zone_table(
    *,
    skims_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    raw_dir: Path,
    zone_label: str,
    cell_col: str,
    grid_path: str,
    grid_id_col: str,
    pipeline: PipelineConfig,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    import duckdb

    if grouped_df is None or grouped_df.empty:
        return None, None

    proportion_col = _first_existing_col(grouped_df, [f"{zone_label}_zone_piece_proportion", f"{zone_label}_zone_edge_proportion"])
    if not proportion_col:
        raise ValueError(f"{_step_label('2', zone_label)} requires a cumulated {zone_label} proportion column.")

    emission_cols = [
        c for c in skims_df.columns
        if c.startswith("tons_per_year_") and pd.api.types.is_numeric_dtype(skims_df[c])
    ]
    con = duckdb.connect(database=":memory:")
    try:
        con.register("grouped_df", grouped_df)
        con.register("skims_df", skims_df[["linkId", "vehicleTypeId", "process"] + emission_cols].copy())
        metric_cols = [col for col in grouped_df.columns if col.startswith(f"{zone_label}_") and any(tag in col for tag in _INTERSECTION_METRIC_SUFFIXES)]
        metric_select = ",\n                ".join([f"g.{col}" for col in metric_cols])
        alloc_select = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{proportion_col} AS DOUBLE), 0.0) AS {col}_{zone_label}_allocated"
            for col in emission_cols
        ])
        county_select = ", g.countyfp" if "countyfp" in grouped_df.columns else ", NULL AS countyfp"
        allocated = con.execute(
            f"""
            SELECT
                g.linkId,
                s.vehicleTypeId,
                s.process,
                g.{cell_col} AS {cell_col}
                {county_select}
                {"," if metric_select else ""} {metric_select}
                {"," if alloc_select else ""} {alloc_select}
            FROM grouped_df AS g
            LEFT JOIN skims_df AS s
                ON g.linkId = s.linkId
            """
        ).df()
    finally:
        con.close()

    if allocated.empty:
        return None, None

    allocated_stem = raw_dir / f"{zone_label}_emissions_allocated"
    _save_grid_emissions(
        allocated,
        left_col=cell_col,
        right_col=grid_id_col,
        grid_path=grid_path,
        output_epsg=int(pipeline.output_epsg),
        output_stem=allocated_stem,
    )
    allocated_path = str(allocated_stem) + ".parquet"
    logger.info("%s allocated emissions rows=%d → %s", _step_label("2", zone_label), len(allocated), allocated_path)
    return allocated, allocated_path


def _build_corrected_zone_table(
    *,
    allocated_df: pd.DataFrame,
    raw_dir: Path,
    zone_label: str,
    cell_col: str,
    grid_path: str,
    grid_id_col: str,
    pipeline: PipelineConfig,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    from impacts.emissions.emissions_grid_mapping import apply_county_corrections

    if allocated_df is None or allocated_df.empty:
        return None, None

    corrected_path = _table_path(raw_dir, f"{zone_label}_emissions_corrected")
    if pipeline.activity_corrections_path:
        logger.info(
            "%s applying county corrections from %s",
            _step_label("3", zone_label),
            pipeline.activity_corrections_path,
        )
        corrected = apply_county_corrections(
            allocated_df,
            corrections_path=pipeline.activity_corrections_path,
            correction_columns=pipeline.activity_corrections_columns or None,
        )
        _write_table(corrected, corrected_path)
    else:
        corrected = allocated_df
        _write_table(corrected, corrected_path)
        logger.info("%s no corrections configured; using allocated totals as-is", _step_label("3", zone_label))

    corrected_stem = raw_dir / f"{zone_label}_grid_emissions"
    _save_grid_emissions(
        corrected,
        left_col=cell_col,
        right_col=grid_id_col,
        grid_path=grid_path,
        output_epsg=int(pipeline.output_epsg),
        output_stem=corrected_stem,
    )
    logger.info("%s grid emissions → %s", _step_label("4", zone_label), str(corrected_stem) + ".parquet")
    return corrected, str(corrected_path)


def run(
    pipeline: PipelineConfig,
    raw_dir: Path,
    skims_df: pd.DataFrame,
    intersection_path: str,
    intersection_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Optional[str]]:
    """Allocate, correct, and store AERMOD and inMAP grid-emissions tables.

    Returns dict of output paths keyed by output name.
    """
    # Step 4.1-4.4: build separate grouped, allocated, corrected, and geospatial tables for each grid
    aermod_grouped_df: Optional[pd.DataFrame] = None
    aermod_allocated_path: Optional[str] = None
    aermod_corrected_path: Optional[str] = None
    aermod_grouped_path: Optional[str] = None
    aermod_grid_emissions_path: Optional[str] = None
    if pipeline.aermod_grid_path:
        aermod_grouped_df, aermod_grouped_path = _build_grouped_zone_table(
            intersection_path=intersection_path,
            raw_dir=raw_dir,
            zone_label="aermod",
            cell_col="aermod_srv_cell_id",
            grid_path=pipeline.aermod_grid_path,
            grid_id_col="srv_cell_id",
            pipeline=pipeline,
            intersection_df=intersection_df,
        )
        aermod_allocated_df, aermod_allocated_path = _build_allocated_zone_table(
            skims_df=skims_df,
            grouped_df=aermod_grouped_df,
            raw_dir=raw_dir,
            zone_label="aermod",
            cell_col="aermod_srv_cell_id",
            grid_path=pipeline.aermod_grid_path,
            grid_id_col="srv_cell_id",
            pipeline=pipeline,
        )
        aermod_df, aermod_corrected_path = _build_corrected_zone_table(
            allocated_df=aermod_allocated_df,
            raw_dir=raw_dir,
            zone_label="aermod",
            cell_col="aermod_srv_cell_id",
            grid_path=pipeline.aermod_grid_path,
            grid_id_col="srv_cell_id",
            pipeline=pipeline,
        )
        if aermod_df is not None:
            aermod_grid_emissions_path = str(raw_dir / "aermod_grid_emissions.parquet")

    inmap_grouped_df, inmap_grouped_path = _build_grouped_zone_table(
        intersection_path=intersection_path,
        raw_dir=raw_dir,
        zone_label="inmap",
        cell_col="inmap_srm_cell_id",
        grid_path=pipeline.inmap_grid_path,
        grid_id_col="srm_cell_id",
        pipeline=pipeline,
        intersection_df=intersection_df,
    )
    inmap_allocated_df, inmap_allocated_path = _build_allocated_zone_table(
        skims_df=skims_df,
        grouped_df=inmap_grouped_df,
        raw_dir=raw_dir,
        zone_label="inmap",
        cell_col="inmap_srm_cell_id",
        grid_path=pipeline.inmap_grid_path,
        grid_id_col="srm_cell_id",
        pipeline=pipeline,
    )
    inmap_df, inmap_corrected_path = _build_corrected_zone_table(
        allocated_df=inmap_allocated_df,
        raw_dir=raw_dir,
        zone_label="inmap",
        cell_col="inmap_srm_cell_id",
        grid_path=pipeline.inmap_grid_path,
        grid_id_col="srm_cell_id",
        pipeline=pipeline,
    )
    inmap_grid_emissions_path = str(raw_dir / "inmap_grid_emissions.parquet") if inmap_df is not None else None

    return {
        "emissions_allocated": inmap_allocated_path,
        "emissions_corrected": inmap_grid_emissions_path,
        "aermod_emissions_allocated": aermod_allocated_path,
        "aermod_mapping_grouped": aermod_grouped_path,
        "aermod_emissions_corrected": aermod_corrected_path,
        "inmap_emissions_allocated": inmap_allocated_path,
        "inmap_mapping_grouped": inmap_grouped_path,
        "inmap_emissions_corrected": inmap_corrected_path,
        "aermod_grid_emissions": aermod_grid_emissions_path,
        "inmap_grid_emissions": inmap_grid_emissions_path,
    }
