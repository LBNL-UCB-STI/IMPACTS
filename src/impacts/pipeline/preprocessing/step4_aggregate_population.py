"""Preprocess Step 4 — Aggregate population counts to AERMOD grid cells.

Spatially joins UrbanSim persons/households to the AERMOD grid and writes a
per-cell person count and urban class (0 / 1000 / 10000) used by the pipeline
to select AERMOD dispersion patterns.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from ...common import log_step_banner
from ...common import log_substep_banner
from ...common import read_table
from ...common import read_vector
from ...common import resolve_required_manifest_input
from ...manifest.schema import PipelineConfig

logger = logging.getLogger(__name__)

_AERMOD_CELL_ID = "aermod_cell_id"
_PERSON_REQUIRED = ["person_id", "household_id", "home_x", "home_y"]
_HOUSEHOLD_REQUIRED = ["household_id"]
_OUTPUT_FILENAME = "aermod_cell_population.parquet"
_STAGED_POPULATION_FILENAME = "staged_population.parquet"


def _classify_urban(person_count: pd.Series) -> pd.Series:
    values = pd.to_numeric(person_count, errors="coerce").fillna(0.0)
    return pd.Series(
        np.where(values < 1000, 0, np.where(values < 10000, 1000, 10000)),
        index=person_count.index,
        dtype="int64",
    )


def _load_and_merge_population(
    *,
    persons_entry: Dict[str, Any],
    households_entry: Dict[str, Any],
    target_epsg: int,
) -> gpd.GeoDataFrame:
    persons = read_table(resolve_required_manifest_input({"persons": persons_entry}, key="persons")).copy()
    households = read_table(resolve_required_manifest_input({"households": households_entry}, key="households")).copy()

    for field in ("person_id", "household_id"):
        if field not in persons.columns and persons.index.name == field:
            persons = persons.reset_index()
    for field in ("household_id",):
        if field not in households.columns and households.index.name == field:
            households = households.reset_index()

    missing_p = [c for c in _PERSON_REQUIRED if c not in persons.columns]
    if missing_p:
        raise ValueError(f"Preprocess step 4: persons table missing columns: {missing_p}")
    missing_h = [c for c in _HOUSEHOLD_REQUIRED if c not in households.columns]
    if missing_h:
        raise ValueError(f"Preprocess step 4: households table missing columns: {missing_h}")

    for col in households.columns:
        if col != "household_id" and col in persons.columns:
            households = households.rename(columns={col: f"household_{col}"})

    merged = persons.merge(households, how="left", on="household_id")
    merged["home_x"] = pd.to_numeric(merged["home_x"], errors="coerce")
    merged["home_y"] = pd.to_numeric(merged["home_y"], errors="coerce")
    merged = merged.loc[merged["home_x"].notna() & merged["home_y"].notna()].copy()

    gdf = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["home_x"], merged["home_y"]),
        crs="EPSG:4326",
    )
    return gdf.to_crs(epsg=target_epsg)


def _aggregate_to_aermod_grid(
    *,
    population_gdf: gpd.GeoDataFrame,
    aermod_grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    joined = gpd.sjoin(
        population_gdf[[_AERMOD_CELL_ID if _AERMOD_CELL_ID in population_gdf.columns else "geometry", "geometry"]],
        aermod_grid[[_AERMOD_CELL_ID, "geometry"]],
        how="inner",
        predicate="within",
    )
    joined = joined.drop(columns=["index_right", "geometry"], errors="ignore")

    counts = (
        joined.groupby(_AERMOD_CELL_ID, dropna=False)
        .size()
        .rename("person_count")
        .reset_index()
    )
    counts[_AERMOD_CELL_ID] = pd.to_numeric(counts[_AERMOD_CELL_ID], errors="coerce").astype(int)
    counts["person_count"] = counts["person_count"].astype(int)
    counts["source_urban_class"] = _classify_urban(counts["person_count"]).astype(int)
    return counts[[_AERMOD_CELL_ID, "person_count", "source_urban_class"]]


def run(
    pipeline: PipelineConfig,
    output_root: Path,
    *,
    population_inputs: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[str], Optional[str]]:
    log_step_banner("Preprocess Step 4", "Aggregate Population to AERMOD Grid", logger=logger)

    if not population_inputs or not population_inputs.get("persons") or not population_inputs.get("households"):
        logger.info("Preprocess step 4: no population inputs available — skipping.")
        return None, None

    if not pipeline.aermod_grid_path:
        logger.info("Preprocess step 4: no AERMOD grid configured — skipping.")
        return None, None

    log_substep_banner("4.1", "load and merge persons + households", logger=logger)
    population_gdf = _load_and_merge_population(
        persons_entry=population_inputs["persons"],
        households_entry=population_inputs["households"],
        target_epsg=int(pipeline.output_epsg),
    )
    logger.info("Preprocess step 4.1: loaded %d persons with valid coordinates", len(population_gdf))

    log_substep_banner("4.2", "join population to AERMOD grid and aggregate", logger=logger)
    aermod_grid = read_vector(pipeline.aermod_grid_path)
    if aermod_grid.crs is not None:
        aermod_grid = aermod_grid.to_crs(epsg=int(pipeline.output_epsg))

    aermod_grid_id = str(pipeline.aermod_grid_id)
    if aermod_grid_id not in aermod_grid.columns:
        raise ValueError(
            f"Preprocess step 4: AERMOD grid missing expected id column '{aermod_grid_id}'. "
            f"Available: {list(aermod_grid.columns)}"
        )
    aermod_grid = aermod_grid.rename(columns={aermod_grid_id: _AERMOD_CELL_ID})

    counts = _aggregate_to_aermod_grid(population_gdf=population_gdf, aermod_grid=aermod_grid)
    logger.info(
        "Preprocess step 4.2: %d persons assigned across %d AERMOD cells (urban=%d, suburban=%d, rural=%d)",
        counts["person_count"].sum(),
        len(counts),
        (counts["source_urban_class"] == 10000).sum(),
        (counts["source_urban_class"] == 1000).sum(),
        (counts["source_urban_class"] == 0).sum(),
    )

    log_substep_banner("4.3", "write staged population and per-cell counts", logger=logger)
    output_root.mkdir(parents=True, exist_ok=True)
    staged_population_path = output_root / _STAGED_POPULATION_FILENAME
    population_gdf.to_parquet(staged_population_path, index=False)
    logger.info("Preprocess step 4.3: staged population → %s", staged_population_path)

    cell_population_path = output_root / _OUTPUT_FILENAME
    counts.to_parquet(cell_population_path, index=False)
    logger.info("Preprocess step 4.3: aermod cell population → %s", cell_population_path)

    return str(cell_population_path), str(staged_population_path)