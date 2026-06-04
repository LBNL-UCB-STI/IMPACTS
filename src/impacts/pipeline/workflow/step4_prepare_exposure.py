"""Step 4 — Prepare exposure workflow."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from ...common import log_step_banner
from ...common import log_substep_banner
from ...common import read_table
from ...common import read_vector
from ...common import resolve_required_manifest_input
from ...config.defaults import primary_pm25_integration_strategies as default_primary_pm25_integration_strategies
from ...config.defaults import primary_pm25_strategy_impute_inmap_primary_with_aermod
from ...config.defaults import primary_pm25_strategy_inmap_only
from ...config.defaults import primary_pm25_strategy_scale_aermod_to_inmap_primary
from ...manifest.schema import PipelineConfig
from . import _step_label

logger = logging.getLogger(__name__)
_INMAP_SOURCE_ID_COLUMN = "inmap_cell_id"
_AERMOD_SOURCE_ID_COLUMN = "aermod_cell_id"
_PRIMARY_PM25_MASS_EPSILON = 1e-12


def _trace_frame(step: str, label: str, df: pd.DataFrame) -> None:
    logger.info("%s trace %s shape=%s", _step_label(f"4.{step}"), label, df.shape)
    preview = list(df.columns[:20])
    suffix = "" if len(df.columns) <= 20 else " ..."
    logger.info("%s trace %s columns(%d): %s%s", _step_label(f"4.{step}"), label, len(df.columns), preview, suffix)


def _prepare_inmap_exposure_inputs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if _INMAP_SOURCE_ID_COLUMN not in gdf.columns:
        raise ValueError(f"InMAP concentrations must include '{_INMAP_SOURCE_ID_COLUMN}'.")
    if "TotalPM25" not in gdf.columns:
        raise ValueError("InMAP concentrations must include 'TotalPM25'.")
    if "PrimaryPM25" not in gdf.columns:
        raise ValueError("InMAP concentrations must include 'PrimaryPM25'.")
    if "BC" not in gdf.columns:
        raise ValueError("InMAP concentrations must include 'BC'.")
    if "NO2" not in gdf.columns:
        raise ValueError("InMAP concentrations must include 'NO2'.")

    prepared = gdf.copy()
    prepared["inmap_SecondaryPM25"] = (
        pd.to_numeric(prepared["TotalPM25"], errors="coerce").fillna(0.0)
        - pd.to_numeric(prepared["PrimaryPM25"], errors="coerce").fillna(0.0)
    )
    prepared = prepared.rename(columns={"PrimaryPM25": "inmap_PrimaryPM25", "BC": "inmap_BC", "NO2": "inmap_NO2"})
    keep = [_INMAP_SOURCE_ID_COLUMN, "inmap_PrimaryPM25", "inmap_SecondaryPM25", "inmap_BC", "inmap_NO2"]
    return prepared[keep + (["geometry"] if "geometry" in prepared.columns else [])]


def _prepare_aermod_exposure_inputs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if _AERMOD_SOURCE_ID_COLUMN not in gdf.columns:
        raise ValueError(f"AERMOD concentrations must include '{_AERMOD_SOURCE_ID_COLUMN}'.")
    prepared = gdf.rename(columns={"PrimaryPM25": "aermod_PrimaryPM25", "BC": "aermod_BC", "NO2": "aermod_NO2"})
    if "aermod_PrimaryPM25" not in prepared.columns:
        prepared["aermod_PrimaryPM25"] = np.nan
    if "aermod_BC" not in prepared.columns:
        prepared["aermod_BC"] = np.nan
    if "aermod_NO2" not in prepared.columns:
        prepared["aermod_NO2"] = np.nan
    prepared["aermod_SecondaryPM25"] = 0.0  # AERMOD models primary dispersion only
    support_map = {
        "has_aermod_primarypm25": "has_aermod_primarypm25",
        "has_aermod_bc": "has_aermod_bc",
        "has_aermod_no2": "has_aermod_no2",
    }
    keep = [_AERMOD_SOURCE_ID_COLUMN, "aermod_PrimaryPM25", "aermod_SecondaryPM25", "aermod_BC", "aermod_NO2"]
    for source_col, renamed_col in support_map.items():
        if source_col in prepared.columns:
            prepared = prepared.rename(columns={source_col: renamed_col})
            keep.append(renamed_col)
    return prepared[keep + (["geometry"] if "geometry" in prepared.columns else [])]


def _safe_geometry_area(gdf: gpd.GeoDataFrame) -> pd.Series:
    area = pd.Series(gdf.geometry.area, index=gdf.index, dtype="float64")
    return area.where(np.isfinite(area) & area.gt(0.0), 1.0)


def _scale_aermod_primary_to_inmap_primary(result: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    scaled = result.copy()
    area = _safe_geometry_area(scaled)
    raw_aermod_primary = np.where(
        scaled["has_aermod_primarypm25"],
        scaled["aermod_PrimaryPM25"].clip(lower=0.0),
        0.0,
    )
    budgets = pd.DataFrame(
        {
            _INMAP_SOURCE_ID_COLUMN: scaled[_INMAP_SOURCE_ID_COLUMN],
            "target_budget": scaled["inmap_PrimaryPM25"].clip(lower=0.0) * area,
            "aermod_budget": raw_aermod_primary * area,
        }
    )
    budget_by_inmap = budgets.groupby(_INMAP_SOURCE_ID_COLUMN, dropna=False)[
        ["target_budget", "aermod_budget"]
    ].sum()
    factors = budget_by_inmap["target_budget"] / budget_by_inmap["aermod_budget"].where(
        budget_by_inmap["aermod_budget"].gt(_PRIMARY_PM25_MASS_EPSILON)
    )

    factor_by_cell = scaled[_INMAP_SOURCE_ID_COLUMN].map(factors)
    target_budget_by_cell = scaled[_INMAP_SOURCE_ID_COLUMN].map(budget_by_inmap["target_budget"])
    aermod_budget_by_cell = scaled[_INMAP_SOURCE_ID_COLUMN].map(budget_by_inmap["aermod_budget"])
    fallback_mask = (
        aermod_budget_by_cell.le(_PRIMARY_PM25_MASS_EPSILON)
        & target_budget_by_cell.gt(_PRIMARY_PM25_MASS_EPSILON)
    )

    scaled_primary = pd.Series(raw_aermod_primary, index=scaled.index, dtype="float64") * factor_by_cell.fillna(0.0)
    scaled_primary.loc[fallback_mask] = scaled.loc[fallback_mask, "inmap_PrimaryPM25"].clip(lower=0.0)
    scaled["aermod_PrimaryPM25_scale_factor"] = factor_by_cell
    scaled["aermod_PrimaryPM25_scaled"] = scaled_primary.fillna(0.0)

    finite_factors = factors.replace([np.inf, -np.inf], np.nan).dropna()
    extreme = finite_factors.loc[(finite_factors < 0.1) | (finite_factors > 10.0)]
    if not extreme.empty:
        logger.warning(
            "Step 4 exposure integration found %d InMAP cells with extreme AERMOD PrimaryPM25 "
            "mass-conservation scale factors; min=%.4g max=%.4g",
            len(extreme),
            float(extreme.min()),
            float(extreme.max()),
        )
    if fallback_mask.any():
        logger.warning(
            "Step 4 exposure integration used uniform InMAP PrimaryPM25 in %d AERMOD receptor cells "
            "because their InMAP cells had no positive AERMOD PrimaryPM25 signal to scale.",
            int(fallback_mask.sum()),
        )
    return scaled


def _build_full_exposure_grid(
    *,
    pipeline: PipelineConfig,
    prepared_inmap: gpd.GeoDataFrame,
    prepared_aermod: Optional[gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    if not pipeline.aermod_full_grid_path:
        raise ValueError("pipeline.aermod_full_grid_path must be configured before building the full exposure grid.")
    full_grid = read_vector(pipeline.aermod_full_grid_path)
    required_grid_cols = [_AERMOD_SOURCE_ID_COLUMN, _INMAP_SOURCE_ID_COLUMN]
    missing_grid_cols = [col for col in required_grid_cols if col not in full_grid.columns]
    if missing_grid_cols:
        raise ValueError(
            f"Full exposure grid is missing required columns {missing_grid_cols} in {pipeline.aermod_full_grid_path}."
        )
    result = full_grid[required_grid_cols + ["geometry"]]

    inmap_cols = [c for c in prepared_inmap.columns if c != "geometry"]
    inmap_lookup = prepared_inmap[inmap_cols].drop_duplicates(subset=[_INMAP_SOURCE_ID_COLUMN])
    result = result.merge(inmap_lookup, how="left", on=_INMAP_SOURCE_ID_COLUMN)

    if prepared_aermod is not None:
        aermod_cols = [c for c in prepared_aermod.columns if c != "geometry"]
        aermod_lookup = prepared_aermod[aermod_cols].drop_duplicates(subset=[_AERMOD_SOURCE_ID_COLUMN])
        result = result.merge(aermod_lookup, how="left", on=_AERMOD_SOURCE_ID_COLUMN)
    else:
        result["aermod_PrimaryPM25"] = np.nan
        result["aermod_SecondaryPM25"] = 0.0
        result["aermod_BC"] = np.nan
        result["aermod_NO2"] = np.nan
        result["has_aermod_primarypm25"] = False
        result["has_aermod_bc"] = False
        result["has_aermod_no2"] = False

    for col, default in (
        ("aermod_PrimaryPM25", np.nan),
        ("aermod_SecondaryPM25", 0.0),
        ("aermod_BC", np.nan),
        ("aermod_NO2", np.nan),
        ("has_aermod_primarypm25", False),
        ("has_aermod_bc", False),
        ("has_aermod_no2", False),
    ):
        if col not in result.columns:
            result[col] = default

    for col in ("inmap_PrimaryPM25", "inmap_SecondaryPM25", "aermod_PrimaryPM25", "aermod_SecondaryPM25", "aermod_BC", "aermod_NO2"):
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    for col in ("inmap_BC", "inmap_NO2"):
        if col not in result.columns:
            result[col] = 0.0
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    for col in ("has_aermod_primarypm25", "has_aermod_bc", "has_aermod_no2"):
        if col not in result.columns:
            result[col] = False
        result[col] = result[col].fillna(False).astype(bool)

    primary_strategy = pipeline.primary_pm25_integration_strategy
    if primary_strategy == primary_pm25_strategy_inmap_only:
        result["aermod_PrimaryPM25_scale_factor"] = np.nan
        result["aermod_PrimaryPM25_scaled"] = 0.0
        result["PrimaryPM25"] = result["inmap_PrimaryPM25"]
    elif primary_strategy == primary_pm25_strategy_impute_inmap_primary_with_aermod:
        result["aermod_PrimaryPM25_scale_factor"] = np.nan
        result["aermod_PrimaryPM25_scaled"] = np.where(
            result["has_aermod_primarypm25"],
            result["aermod_PrimaryPM25"],
            0.0,
        )
        result["PrimaryPM25"] = np.where(
            result["has_aermod_primarypm25"],
            result["aermod_PrimaryPM25"],
            result["inmap_PrimaryPM25"],
        )
    elif primary_strategy == primary_pm25_strategy_scale_aermod_to_inmap_primary:
        result = _scale_aermod_primary_to_inmap_primary(gpd.GeoDataFrame(result, geometry="geometry", crs=full_grid.crs))
        result["PrimaryPM25"] = result["aermod_PrimaryPM25_scaled"]
    else:
        raise ValueError(
            "pipeline.primary_pm25_integration_strategy must be one of "
            f"{list(default_primary_pm25_integration_strategies)}, "
            f"got {primary_strategy!r}"
        )
    result["SecondaryPM25"] = result["inmap_SecondaryPM25"]
    result["TotalPM25"] = result["SecondaryPM25"] + result["PrimaryPM25"]
    result["BC"] = np.where(
        result["has_aermod_bc"],
        result["inmap_BC"] + result["aermod_BC"],
        result["inmap_BC"],
    )
    result["NO2"] = np.where(
        result["has_aermod_no2"],
        result["inmap_NO2"] + result["aermod_NO2"],
        result["inmap_NO2"],
    )

    ordered = [
        _AERMOD_SOURCE_ID_COLUMN, _INMAP_SOURCE_ID_COLUMN,
        "TotalPM25", "PrimaryPM25", "SecondaryPM25", "BC", "NO2",
        "inmap_PrimaryPM25", "inmap_SecondaryPM25",
        "aermod_PrimaryPM25", "aermod_PrimaryPM25_scaled", "aermod_PrimaryPM25_scale_factor",
        "aermod_SecondaryPM25",
        "has_aermod_primarypm25", "has_aermod_bc", "has_aermod_no2",
    ]
    for col in ("inmap_BC", "inmap_NO2", "aermod_BC", "aermod_NO2"):
        if col in result.columns:
            ordered.append(col)
    ordered.append("geometry")
    return gpd.GeoDataFrame(result[ordered], geometry="geometry", crs=full_grid.crs)


def _write_exposure_grid(
    *,
    exposure_grid: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exposure_grid.to_parquet(output_path, index=False)
    exposure_grid.to_file(output_path.with_suffix(".gpkg"), driver="GPKG")




def run(
    *,
    pipeline: PipelineConfig,
    raw_dir: Path,
    inmap_concentrations_path: Optional[str] = None,
    aermod_concentrations_path: Optional[str] = None,
    manifest_inputs: Optional[dict[str, Any]] = None,
) -> tuple[Optional[gpd.GeoDataFrame], Path, Optional[Path], Optional[Path]]:
    """Step 4: prepare the merged exposure artifact from dispersion outputs."""

    log_step_banner("Step 4", "Prepare Exposure", logger=logger)

    log_substep_banner("4.1", "prepare concentration inputs", logger=logger)
    prepared_inmap: Optional[gpd.GeoDataFrame] = None
    prepared_aermod: Optional[gpd.GeoDataFrame] = None
    if inmap_concentrations_path:
        inmap_gdf = read_vector(inmap_concentrations_path)
        prepared_inmap = _prepare_inmap_exposure_inputs(inmap_gdf)
        _trace_frame("1", "prepared_inmap_concentrations", pd.DataFrame(prepared_inmap.drop(columns="geometry", errors="ignore")))
    if aermod_concentrations_path:
        aermod_gdf = read_vector(aermod_concentrations_path)
        prepared_aermod = _prepare_aermod_exposure_inputs(aermod_gdf)
        _trace_frame("1", "prepared_aermod_concentrations", pd.DataFrame(prepared_aermod.drop(columns="geometry", errors="ignore")))

    log_substep_banner("4.2", "build full exposure grid", logger=logger)
    if prepared_inmap is None:
        raise ValueError("Step 4.2 requires prepared InMAP concentrations to build the full exposure grid.")
    full_exposure_grid = _build_full_exposure_grid(
        pipeline=pipeline,
        prepared_inmap=prepared_inmap,
        prepared_aermod=prepared_aermod,
    )
    _trace_frame("2", "full_exposure_grid", pd.DataFrame(full_exposure_grid.drop(columns="geometry", errors="ignore")))
    output_path = raw_dir / "beam_concentration_distribution.parquet"
    _write_exposure_grid(exposure_grid=full_exposure_grid, output_path=output_path)
    logger.info("%s concentration distribution → %s", _step_label("4.2"), output_path)

    population_distribution_output_path: Optional[Path] = None
    population_counts_output_path: Optional[Path] = None

    manifest = manifest_inputs or {}
    if "staged_population" not in manifest or "aermod_cell_population" not in manifest:
        raise ValueError(
            "Step 4 requires staged_population and aermod_cell_population from preprocessing. "
            "Ensure preprocessing step 4 ran successfully before running the workflow."
        )

    log_substep_banner("4.3", "write population distribution", logger=logger)
    staged_path = resolve_required_manifest_input(manifest, key="staged_population")
    population_table = read_table(staged_path)
    if _AERMOD_SOURCE_ID_COLUMN not in population_table.columns:
        raise ValueError("staged_population is missing aermod_cell_id — rerun preprocessing step 4.")
    _trace_frame("3", "population_distribution", population_table)
    population_distribution_output_path = raw_dir / "beam_population_distribution.parquet"
    population_distribution_output_path.parent.mkdir(parents=True, exist_ok=True)
    population_table.to_parquet(population_distribution_output_path, index=False)
    logger.info("%s population distribution → %s", _step_label("4.3"), population_distribution_output_path)

    log_substep_banner("4.4", "build population counts with geometry", logger=logger)
    cell_pop_path = resolve_required_manifest_input(manifest, key="aermod_cell_population")
    cell_counts = read_table(cell_pop_path)[[_AERMOD_SOURCE_ID_COLUMN, "person_count"]]
    cell_counts[_AERMOD_SOURCE_ID_COLUMN] = pd.to_numeric(cell_counts[_AERMOD_SOURCE_ID_COLUMN], errors="coerce").astype(int)
    grid_lookup = full_exposure_grid[[_AERMOD_SOURCE_ID_COLUMN, "geometry"]].drop_duplicates(subset=[_AERMOD_SOURCE_ID_COLUMN])
    population_counts = gpd.GeoDataFrame(
        grid_lookup.merge(cell_counts, how="inner", on=_AERMOD_SOURCE_ID_COLUMN),
        geometry="geometry",
        crs=full_exposure_grid.crs,
    )
    _trace_frame("4", "population_counts", pd.DataFrame(population_counts.drop(columns="geometry", errors="ignore")))
    population_counts_output_path = raw_dir / "beam_population_counts.parquet"
    population_counts_output_path.parent.mkdir(parents=True, exist_ok=True)
    population_counts.to_parquet(population_counts_output_path, index=False)
    population_counts.to_file(population_counts_output_path.with_suffix(".gpkg"), driver="GPKG")
    logger.info("%s population counts → %s", _step_label("4.4"), population_counts_output_path)

    return full_exposure_grid, output_path, population_distribution_output_path, population_counts_output_path
