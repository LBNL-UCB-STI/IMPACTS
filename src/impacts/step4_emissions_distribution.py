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

from .contract_utils import parquet_available
from .manifest_models import PipelineConfig

logger = logging.getLogger(__name__)


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


def run(
    pipeline: PipelineConfig,
    raw_dir: Path,
    skims_df: pd.DataFrame,
    intersection_path: str,
) -> Dict[str, Optional[str]]:
    """Allocate, correct, and collapse emissions to grid cells.

    Returns dict of output paths keyed by output name.
    """
    from impacts.emissions.emissions_grid_mapping import allocate_emissions_to_labeled_intersection
    from impacts.emissions.emissions_grid_mapping import apply_county_corrections

    # Step 4.1: allocate skims to labeled grid intersection
    emissions_allocated_path = _table_path(raw_dir, "emissions_allocated")
    logger.info("Step 4.1: allocating skims to labeled grid intersection")
    n_rows = allocate_emissions_to_labeled_intersection(
        skims_path=skims_df,
        intersection_path=intersection_path,
        output_path=str(emissions_allocated_path),
        mapping_columns=pipeline.mapping_columns,
    )
    logger.info("Step 4.1 complete: %d rows → %s", n_rows, emissions_allocated_path)

    # Step 4.2: apply county-based correction factors
    emissions_corrected_path = _table_path(raw_dir, "emissions_corrected")
    if pipeline.activity_corrections_path:
        logger.info("Step 4.2: applying county corrections from %s", pipeline.activity_corrections_path)
        allocated_df = pd.read_parquet(emissions_allocated_path)
        corrected_df = apply_county_corrections(
            allocated_df,
            corrections_path=pipeline.activity_corrections_path,
            correction_columns=pipeline.activity_corrections_columns or None,
        )
        corrected_df.to_parquet(emissions_corrected_path, index=False, engine="pyarrow", compression="snappy")
        logger.info("Step 4.2 complete: %d rows → %s", len(corrected_df), emissions_corrected_path)
    else:
        emissions_corrected_path = emissions_allocated_path
        logger.info("Step 4.2: no corrections configured, using Step 4.1 output as-is")

    # Step 4.3: collapse to grid cells by vehicleType
    logger.info("Step 4.3: collapsing emissions to grid cells by vehicleType")
    corrected_df = pd.read_parquet(emissions_corrected_path)
    emission_cols = [c for c in corrected_df.columns if c.startswith("tons_per_year_")]

    aermod_grid_emissions_path: Optional[str] = None
    if pipeline.aermod_grid_path and "aermod_srv_cell_id" in corrected_df.columns:
        aermod_collapsed = (
            corrected_df[["aermod_srv_cell_id", "vehicleTypeId"] + emission_cols]
            .groupby(["aermod_srv_cell_id", "vehicleTypeId"], dropna=False)[emission_cols]
            .sum()
            .reset_index()
        )
        aermod_stem = raw_dir / "aermod_grid_emissions"
        _save_grid_emissions(
            aermod_collapsed,
            left_col="aermod_srv_cell_id",
            right_col="srv_cell_id",
            grid_path=pipeline.aermod_grid_path,
            output_epsg=int(pipeline.output_epsg),
            output_stem=aermod_stem,
        )
        aermod_grid_emissions_path = str(aermod_stem) + ".parquet"
        logger.info("Step 4.3: aermod grid emissions → %s", aermod_grid_emissions_path)

    inmap_stem = raw_dir / "inmap_grid_emissions"
    inmap_collapsed = (
        corrected_df[["inmap_srm_cell_id", "vehicleTypeId"] + emission_cols]
        .groupby(["inmap_srm_cell_id", "vehicleTypeId"], dropna=False)[emission_cols]
        .sum()
        .reset_index()
    )
    _save_grid_emissions(
        inmap_collapsed,
        left_col="inmap_srm_cell_id",
        right_col="srm_cell_id",
        grid_path=pipeline.inmap_grid_path,
        output_epsg=int(pipeline.output_epsg),
        output_stem=inmap_stem,
    )
    inmap_grid_emissions_path = str(inmap_stem) + ".parquet"
    logger.info("Step 4.3 complete: inmap grid emissions → %s", inmap_grid_emissions_path)

    return {
        "emissions_allocated": str(emissions_allocated_path),
        "emissions_corrected": str(emissions_corrected_path),
        "aermod_grid_emissions": aermod_grid_emissions_path,
        "inmap_grid_emissions": inmap_grid_emissions_path,
    }