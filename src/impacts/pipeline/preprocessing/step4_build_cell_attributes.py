"""Preprocess Step 4 — Build AERMOD cell attributes and population staging.

Writes one attribute row for every AERMOD study-area cell. When population
inputs are available, it also spatially joins UrbanSim persons/households to the
AERMOD grid for exposure outputs. Cells without residents remain in the
attribute table with person_count=0 and source_urban_class=0 so source
attributes never shrink the dispersion domain to populated cells.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

from ...common import configure_duckdb_connection
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
_OUTPUT_FILENAME = "aermod_cell_attributes.parquet"
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
        raise ValueError(f"Preprocess Step 4: persons table missing columns: {missing_p}")
    missing_h = [c for c in _HOUSEHOLD_REQUIRED if c not in households.columns]
    if missing_h:
        raise ValueError(f"Preprocess Step 4: households table missing columns: {missing_h}")

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


def _join_population_to_aermod_grid(
    *,
    population_gdf: gpd.GeoDataFrame,
    aermod_grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    joined = gpd.sjoin(
        population_gdf,
        aermod_grid[[_AERMOD_CELL_ID, "geometry"]],
        how="inner",
        predicate="within",
    )
    return pd.DataFrame(joined.drop(columns=["index_right", "geometry"], errors="ignore"))


def _complete_cell_attributes(
    aermod_grid: pd.DataFrame,
    counts: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if _AERMOD_CELL_ID not in aermod_grid.columns:
        raise ValueError(f"AERMOD grid attributes require '{_AERMOD_CELL_ID}'.")
    cells = pd.DataFrame(
        {
            _AERMOD_CELL_ID: (
                pd.to_numeric(aermod_grid[_AERMOD_CELL_ID], errors="coerce")
                .dropna()
                .astype(int)
            )
        }
    ).drop_duplicates(subset=[_AERMOD_CELL_ID], keep="first")
    cells = cells.sort_values(_AERMOD_CELL_ID).reset_index(drop=True)
    if counts is None or counts.empty:
        cells["person_count"] = 0
    else:
        counts = counts.copy()
        counts[_AERMOD_CELL_ID] = pd.to_numeric(counts[_AERMOD_CELL_ID], errors="coerce").astype(int)
        counts["person_count"] = pd.to_numeric(counts["person_count"], errors="coerce").fillna(0).astype(int)
        cells = cells.merge(
            counts[[_AERMOD_CELL_ID, "person_count"]],
            how="left",
            on=_AERMOD_CELL_ID,
        )
        cells["person_count"] = cells["person_count"].fillna(0).astype(int)
    cells["source_urban_class"] = _classify_urban(cells["person_count"]).astype(int)
    return cells[[_AERMOD_CELL_ID, "person_count", "source_urban_class"]]


def _counts_from_joined(
    joined: pd.DataFrame,
    *,
    aermod_grid: pd.DataFrame,
    working_dir: Optional[Path] = None,
) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        configure_duckdb_connection(
            con,
            working_dir=working_dir or Path.cwd(),
            show_progress=False,
            profile="balanced",
        )
        con.register("_joined", joined)
        counts = con.execute(
            f'SELECT "{_AERMOD_CELL_ID}", COUNT(*) AS person_count'
            f' FROM _joined GROUP BY "{_AERMOD_CELL_ID}"'
        ).df()
    finally:
        con.close()
    return _complete_cell_attributes(aermod_grid, counts)


def _load_aermod_grid(pipeline: PipelineConfig) -> gpd.GeoDataFrame:
    aermod_grid = read_vector(str(pipeline.aermod_grid_path))
    if aermod_grid.crs is not None:
        aermod_grid = aermod_grid.to_crs(epsg=int(pipeline.output_epsg))

    aermod_grid_id = str(pipeline.aermod_grid_id)
    if aermod_grid_id not in aermod_grid.columns:
        raise ValueError(
            f"Preprocess Step 4: AERMOD grid missing expected id column '{aermod_grid_id}'. "
            f"Available: {list(aermod_grid.columns)}"
        )
    return aermod_grid.rename(columns={aermod_grid_id: _AERMOD_CELL_ID})


def run(
    pipeline: PipelineConfig,
    output_root: Path,
    *,
    population_inputs: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[str], Optional[str]]:
    log_step_banner("Preprocess Step 4", "Build AERMOD Cell Attributes", logger=logger)

    if not pipeline.aermod_grid_path:
        logger.info("Preprocess Step 4: no AERMOD grid configured; skipping.")
        return None, None

    output_root.mkdir(parents=True, exist_ok=True)
    cell_attributes_path = output_root / _OUTPUT_FILENAME

    log_substep_banner("4.1", "load AERMOD study-area grid", logger=logger)
    aermod_grid = _load_aermod_grid(pipeline)
    logger.info("Preprocess Step 4.1: loaded %d AERMOD study-area cells", len(aermod_grid))

    joined: Optional[pd.DataFrame] = None
    log_substep_banner("4.2", "assign population to AERMOD cells", logger=logger)
    if not population_inputs or not population_inputs.get("persons") or not population_inputs.get("households"):
        logger.info(
            "Preprocess Step 4.2: no population inputs available; staged population output will be skipped."
        )
    else:
        population_gdf = _load_and_merge_population(
            persons_entry=population_inputs["persons"],
            households_entry=population_inputs["households"],
            target_epsg=int(pipeline.output_epsg),
        )
        logger.info("Preprocess Step 4.2: loaded %d persons with valid coordinates", len(population_gdf))

        joined = _join_population_to_aermod_grid(population_gdf=population_gdf, aermod_grid=aermod_grid)
        logger.info(
            "Preprocess Step 4.2: assigned %d persons across %d populated AERMOD cells",
            len(joined),
            joined[_AERMOD_CELL_ID].nunique(),
        )

    log_substep_banner("4.3", "build AERMOD cell attributes", logger=logger)
    counts = (
        _complete_cell_attributes(aermod_grid)
        if joined is None
        else _counts_from_joined(joined, aermod_grid=aermod_grid, working_dir=output_root)
    )
    logger.info(
        "Preprocess Step 4.3: built attributes for %d AERMOD domain cells "
        "(urban=%d, suburban=%d, rural=%d)",
        len(counts),
        (counts["source_urban_class"] == 10000).sum(),
        (counts["source_urban_class"] == 1000).sum(),
        (counts["source_urban_class"] == 0).sum(),
    )

    staged_population_path: Optional[Path] = None
    log_substep_banner("4.4", "write staged population", logger=logger)
    if joined is not None:
        staged_population_path = output_root / _STAGED_POPULATION_FILENAME
        joined.to_parquet(staged_population_path, index=False)
        logger.info(
            "Preprocess Step 4.4: staged population (per-person, with aermod_cell_id) → %s",
            staged_population_path,
        )
    else:
        logger.info("Preprocess Step 4.4: no staged population output to write.")

    log_substep_banner("4.5", "write AERMOD cell attributes", logger=logger)
    counts.to_parquet(cell_attributes_path, index=False)
    logger.info("Preprocess Step 4.5: AERMOD cell attributes → %s", cell_attributes_path)

    return str(cell_attributes_path), str(staged_population_path) if staged_population_path else None
