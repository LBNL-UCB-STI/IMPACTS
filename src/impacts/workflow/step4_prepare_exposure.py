"""Step 4 — Prepare exposure workflow."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from ..common import log_step_banner
from ..common import log_substep_banner
from ..common import read_table
from ..common import read_vector
from ..common import resolve_required_manifest_input
from ..manifest.schema import PipelineConfig
from . import _step_label

logger = logging.getLogger(__name__)
_INMAP_SOURCE_ID_COLUMN = "inmap_cell_id"
_AERMOD_SOURCE_ID_COLUMN = "aermod_cell_id"
_PERSON_REQUIRED_COLUMNS = ["person_id", "household_id", "home_x", "home_y"]
_HOUSEHOLD_REQUIRED_COLUMNS = ["household_id"]


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

    prepared = gdf.copy()
    prepared["inmap_SecondaryPM25"] = (
        pd.to_numeric(prepared["TotalPM25"], errors="coerce").fillna(0.0)
        - pd.to_numeric(prepared["PrimaryPM25"], errors="coerce").fillna(0.0)
    )
    prepared = prepared.rename(columns={"PrimaryPM25": "inmap_PrimaryPM25", "BC": "inmap_BC", "NO2": "inmap_NO2"})
    keep = [_INMAP_SOURCE_ID_COLUMN, "inmap_PrimaryPM25", "inmap_SecondaryPM25"]
    for col in ("inmap_BC", "inmap_NO2"):
        if col in prepared.columns:
            keep.append(col)
    return prepared[keep + (["geometry"] if "geometry" in prepared.columns else [])].copy()


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
    return prepared[keep + (["geometry"] if "geometry" in prepared.columns else [])].copy()


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
    result = full_grid[required_grid_cols + ["geometry"]].copy()

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
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    for col in ("has_aermod_primarypm25", "has_aermod_bc", "has_aermod_no2"):
        if col not in result.columns:
            result[col] = False
        result[col] = result[col].fillna(False).astype(bool)

    # Use explicit AERMOD support masks rather than numeric > 0 heuristics.
    result["PrimaryPM25"] = np.where(
        result["has_aermod_primarypm25"],
        result["aermod_PrimaryPM25"],
        result["inmap_PrimaryPM25"],
    )
    result["SecondaryPM25"] = result["inmap_SecondaryPM25"]
    result["TotalPM25"] = result["SecondaryPM25"] + result["PrimaryPM25"]
    result["BC"] = np.where(
        result["has_aermod_bc"],
        result["aermod_BC"],
        result.get("inmap_BC", 0.0),
    )
    result["NO2"] = np.where(
        result["has_aermod_no2"],
        result["aermod_NO2"],
        result.get("inmap_NO2", 0.0),
    )

    ordered = [
        _AERMOD_SOURCE_ID_COLUMN, _INMAP_SOURCE_ID_COLUMN,
        "TotalPM25", "PrimaryPM25", "SecondaryPM25", "BC", "NO2",
        "inmap_PrimaryPM25", "inmap_SecondaryPM25",
        "aermod_PrimaryPM25", "aermod_SecondaryPM25",
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


def _load_population_table(entry: dict[str, Any], label: str) -> pd.DataFrame:
    return read_table(resolve_required_manifest_input({label: entry}, key=label))


def _normalize_identity_field(df: pd.DataFrame, *, field: str, label: str) -> pd.DataFrame:
    result = df.copy()
    if field not in result.columns and result.index.name == field:
        result = result.reset_index()
    if field not in result.columns:
        raise ValueError(f"{label} table is missing required identity field: {field}")
    return result


def _prepare_population_table(
    *,
    persons_df: pd.DataFrame,
    households_df: pd.DataFrame,
    target_epsg: int,
) -> gpd.GeoDataFrame:
    persons = _normalize_identity_field(persons_df, field="person_id", label="Persons")
    persons = _normalize_identity_field(persons, field="household_id", label="Persons")
    households = _normalize_identity_field(households_df, field="household_id", label="Households")

    missing_person_cols = [col for col in _PERSON_REQUIRED_COLUMNS if col not in persons.columns]
    if missing_person_cols:
        raise ValueError(f"Persons table is missing required columns: {missing_person_cols}")
    missing_household_cols = [col for col in _HOUSEHOLD_REQUIRED_COLUMNS if col not in households.columns]
    if missing_household_cols:
        raise ValueError(f"Households table is missing required columns: {missing_household_cols}")
    if "Unnamed: 0" in persons.columns:
        persons = persons.drop(columns=["Unnamed: 0"])
    if "Unnamed: 0" in households.columns:
        households = households.drop(columns=["Unnamed: 0"])

    household_renames = {}
    for column in households.columns:
        if column == "household_id":
            continue
        if column in persons.columns:
            household_renames[column] = f"household_{column}"
    if household_renames:
        households = households.rename(columns=household_renames)

    merged = persons.merge(households, how="left", on="household_id")
    merged["home_x"] = pd.to_numeric(merged["home_x"], errors="coerce")
    merged["home_y"] = pd.to_numeric(merged["home_y"], errors="coerce")
    merged = merged.loc[merged["home_x"].notna() & merged["home_y"].notna()].copy()
    population = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["home_x"], merged["home_y"]),
        crs="EPSG:4326",
    )
    return population.to_crs(epsg=target_epsg)


def _assign_population_to_exposure_grid(
    *,
    population_gdf: gpd.GeoDataFrame,
    exposure_grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    joined = gpd.sjoin(
        population_gdf,
        exposure_grid[[_AERMOD_SOURCE_ID_COLUMN, "geometry"]],
        how="inner",
        predicate="within",
    )
    joined = joined.drop(columns=["index_right", "geometry"], errors="ignore")
    if _AERMOD_SOURCE_ID_COLUMN not in joined.columns:
        raise ValueError("Population overlay did not produce aermod_cell_id.")
    return pd.DataFrame(joined)


def _aggregate_population_by_aermod_cell(population_table: pd.DataFrame) -> pd.DataFrame:
    if _AERMOD_SOURCE_ID_COLUMN not in population_table.columns:
        raise ValueError("Population exposure table must include aermod_cell_id before aggregation.")
    counts = (
        population_table.groupby(_AERMOD_SOURCE_ID_COLUMN, dropna=False)
        .size()
        .rename("person_count")
        .reset_index()
    )
    counts[_AERMOD_SOURCE_ID_COLUMN] = pd.to_numeric(counts[_AERMOD_SOURCE_ID_COLUMN], errors="raise").astype(int)
    counts["person_count"] = pd.to_numeric(counts["person_count"], errors="raise").astype(int)
    return counts


def _build_population_exposure_distribution(
    *,
    population_table: pd.DataFrame,
    exposure_grid: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    counts = _aggregate_population_by_aermod_cell(population_table)
    grid_lookup = exposure_grid[[_AERMOD_SOURCE_ID_COLUMN, "geometry"]].drop_duplicates(
        subset=[_AERMOD_SOURCE_ID_COLUMN]
    ).copy()
    distribution = grid_lookup.merge(counts, how="inner", on=_AERMOD_SOURCE_ID_COLUMN)
    ordered = [_AERMOD_SOURCE_ID_COLUMN, "person_count", "geometry"]
    return gpd.GeoDataFrame(distribution[ordered], geometry="geometry", crs=exposure_grid.crs)


def run(
    *,
    pipeline: PipelineConfig,
    raw_dir: Path,
    inmap_concentrations_path: Optional[str] = None,
    aermod_concentrations_path: Optional[str] = None,
    population_inputs: Optional[dict[str, Any]] = None,
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
    if population_inputs and population_inputs.get("persons") and population_inputs.get("households"):
        log_substep_banner("4.3", "create population table", logger=logger)
        persons_df = _load_population_table(population_inputs["persons"], "persons")
        households_df = _load_population_table(population_inputs["households"], "households")
        population_gdf = _prepare_population_table(
            persons_df=persons_df,
            households_df=households_df,
            target_epsg=int(pipeline.output_epsg),
        )
        _trace_frame("3", "population_with_households", pd.DataFrame(population_gdf.drop(columns="geometry", errors="ignore")))
        population_table = _assign_population_to_exposure_grid(
            population_gdf=population_gdf,
            exposure_grid=full_exposure_grid,
        )
        _trace_frame("3", "population_distribution", population_table)
        population_distribution_output_path = raw_dir / "beam_population_distribution.parquet"
        population_distribution_output_path.parent.mkdir(parents=True, exist_ok=True)
        population_table.to_parquet(population_distribution_output_path, index=False)
        logger.info("%s population distribution → %s", _step_label("4.3"), population_distribution_output_path)

        log_substep_banner("4.4", "build population counts", logger=logger)
        population_counts = _build_population_exposure_distribution(
            population_table=population_table,
            exposure_grid=full_exposure_grid,
        )
        _trace_frame(
            "4",
            "population_counts",
            pd.DataFrame(population_counts.drop(columns="geometry", errors="ignore")),
        )
        population_counts_output_path = raw_dir / "beam_population_counts.parquet"
        population_counts_output_path.parent.mkdir(parents=True, exist_ok=True)
        population_counts.to_parquet(population_counts_output_path, index=False)
        population_counts.to_file(population_counts_output_path.with_suffix(".gpkg"), driver="GPKG")
        logger.info("%s population counts → %s", _step_label("4.4"), population_counts_output_path)
    else:
        logger.info("%s population table skipped: persons/households not both available", _step_label("4.3"))

    return full_exposure_grid, output_path, population_distribution_output_path, population_counts_output_path
