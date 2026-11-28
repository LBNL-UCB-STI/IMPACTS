"""Preprocess Step 4 — Assign raw UrbanSim residents to the exposure grid."""
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
from ...common import resolve_required_manifest_input
from ...manifest.schema import PipelineConfig
from ._common import load_aermod_grid

logger = logging.getLogger(__name__)

_AERMOD_CELL_ID = "aermod_cell_id"
_STAGED_POPULATION_FILENAME = "staged_population.parquet"
_POPULATION_COUNTS_FILENAME = "population_cell_counts.parquet"
_HOUSEHOLD_METADATA_FILENAME = "population_households.parquet"


def _load_urbansim_population(
    *,
    population_h5_entry: Dict[str, Any],
    year: int,
    target_epsg: int,
    require_acs_metadata: bool,
) -> tuple[gpd.GeoDataFrame, int, int, Optional[pd.DataFrame]]:
    population_h5 = Path(
        resolve_required_manifest_input({"population_h5": population_h5_entry}, key="population_h5")
    )
    table_paths = {name: f"/{year}/{name}" for name in ("persons", "households", "blocks")}
    with pd.HDFStore(population_h5, mode="r") as store:
        missing_tables = [key for key in table_paths.values() if key not in store]
        if missing_tables:
            raise ValueError(
                "Preprocess Step 4: UrbanSim HDF5 is missing exact-year tables "
                f"for {year}: {missing_tables}"
            )
        persons, households, blocks = (
            store[table_paths[name]] for name in ("persons", "households", "blocks")
        )

    for frame, key in ((persons, "person_id"), (households, "household_id"), (blocks, "block_id")):
        if key in frame.columns:
            frame.set_index(key, inplace=True)
        if frame.index.name != key or frame.index.hasnans or not frame.index.is_unique:
            raise ValueError(
                f"Preprocess Step 4: UrbanSim {key} identifiers must be unique and non-null"
            )

    missing_person_columns = [column for column in ("household_id",) if column not in persons.columns]
    required_household_columns = ("block_id", "persons") if require_acs_metadata else ("block_id",)
    required_block_columns = ("x", "y", "county_id") if require_acs_metadata else ("x", "y")
    missing_household_columns = [column for column in required_household_columns if column not in households.columns]
    missing_block_columns = [column for column in required_block_columns if column not in blocks.columns]
    if missing_person_columns or missing_household_columns or missing_block_columns:
        raise ValueError(
            "Preprocess Step 4: UrbanSim HDF5 lacks columns needed to locate home coordinates "
            f"(persons={missing_person_columns}, households={missing_household_columns}, "
            f"blocks={missing_block_columns})"
        )

    household_columns = ["block_id", "persons"] if require_acs_metadata else ["block_id"]
    block_columns = ["x", "y", "county_id"] if require_acs_metadata else ["x", "y"]
    home_coordinates = households[household_columns].join(
        blocks[block_columns], on="block_id", validate="many_to_one"
    )
    households["home_x"] = pd.to_numeric(home_coordinates["x"], errors="coerce")
    households["home_y"] = pd.to_numeric(home_coordinates["y"], errors="coerce")
    people = persons
    people["home_x"] = people["household_id"].map(households["home_x"])
    people["home_y"] = people["household_id"].map(households["home_y"])
    merged = people.reset_index()
    merged["home_x"] = pd.to_numeric(merged["home_x"], errors="coerce")
    merged["home_y"] = pd.to_numeric(merged["home_y"], errors="coerce")
    valid_lonlat = merged["home_x"].between(-180, 180) & merged["home_y"].between(-90, 90)
    merged.loc[~valid_lonlat, ["home_x", "home_y"]] = np.nan

    supplied = len(merged)
    valid_coordinates = (
        np.isfinite(merged["home_x"].to_numpy(dtype=float))
        & np.isfinite(merged["home_y"].to_numpy(dtype=float))
    )
    missing_coordinates = int((~valid_coordinates).sum())
    merged = merged.loc[valid_coordinates].copy()
    population = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["home_x"], merged["home_y"]),
        crs="EPSG:4326",
    )
    household_metadata = None
    if require_acs_metadata:
        household_metadata = pd.DataFrame(
            {
                "household_id": households.index,
                "county_geoid": home_coordinates["county_id"].astype("string").to_numpy(),
                "household_size": pd.to_numeric(home_coordinates["persons"], errors="coerce").to_numpy(),
            }
        )
    return population.to_crs(epsg=target_epsg), supplied, missing_coordinates, household_metadata


def _join_population_to_grid(
    *, population: gpd.GeoDataFrame, grid: gpd.GeoDataFrame
) -> pd.DataFrame:
    joined = gpd.sjoin(
        population,
        grid[[_AERMOD_CELL_ID, "geometry"]],
        how="inner",
        predicate="within",
    )
    # Overlapping grid features must not count a resident more than once.
    joined = joined.sort_values(_AERMOD_CELL_ID).drop_duplicates("person_id", keep="first")
    return pd.DataFrame(joined.drop(columns=["index_right", "geometry"], errors="ignore"))


def _count_population_by_cell(joined: pd.DataFrame, *, working_dir: Path) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        configure_duckdb_connection(
            con, working_dir=working_dir, show_progress=False, profile="balanced"
        )
        con.register("_joined", joined)
        return con.execute(
            f'SELECT "{_AERMOD_CELL_ID}", COUNT(*) AS person_count '
            f'FROM _joined GROUP BY "{_AERMOD_CELL_ID}"'
        ).df()
    finally:
        con.close()


def run(
    pipeline: PipelineConfig,
    output_root: Path,
    *,
    population_inputs: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[str], Optional[str], Optional[Dict[str, int]], Optional[str]]:
    log_step_banner("Preprocess Step 4", "Assign Raw Population to Exposure Grid", logger=logger)

    if not pipeline.aermod_grid_path:
        if pipeline.exposure_enabled:
            raise ValueError("Exposure requires the configured 100 m grid for resident assignment.")
        logger.info("Preprocess Step 4: no exposure grid configured; skipping.")
        return None, None, None, None
    if pipeline.exposure_enabled and (not population_inputs or not population_inputs.get("h5")):
        raise ValueError("Exposure is enabled, but the UrbanSim HDF5 was not supplied to IMPACTS.")
    if not population_inputs or not population_inputs.get("h5"):
        logger.info("Preprocess Step 4: no UrbanSim HDF5 supplied; skipping population assignment.")
        return None, None, None, None

    output_root.mkdir(parents=True, exist_ok=True)
    log_substep_banner("4.1", "load exposure grid", logger=logger)
    grid = load_aermod_grid(pipeline)
    logger.info("Preprocess Step 4.1: loaded %d exposure-grid cells", len(grid))

    log_substep_banner("4.2", "assign UrbanSim residents by home coordinate", logger=logger)
    population, supplied, missing_coordinates, household_metadata = _load_urbansim_population(
        population_h5_entry=population_inputs["h5"],
        year=int(population_inputs.get("year") or pipeline.start_year),
        target_epsg=int(pipeline.output_epsg),
        require_acs_metadata=bool(population_inputs.get("acs_tables")),
    )
    joined = _join_population_to_grid(population=population, grid=grid)
    assignment_report = {
        "residents_supplied": int(supplied),
        "residents_assigned_to_cell": int(len(joined)),
        "residents_missing_home_coordinates": int(missing_coordinates),
        "residents_outside_grid": int(len(population) - len(joined)),
    }
    logger.info("Preprocess Step 4 population assignment summary: %s", assignment_report)

    staged_population_path = output_root / _STAGED_POPULATION_FILENAME
    joined.to_parquet(staged_population_path, index=False)
    logger.info("Preprocess Step 4: per-person grid assignment → %s", staged_population_path)

    counts_path = output_root / _POPULATION_COUNTS_FILENAME
    _count_population_by_cell(joined, working_dir=output_root).to_parquet(counts_path, index=False)
    logger.info("Preprocess Step 4: population counts by cell → %s", counts_path)

    household_metadata_path = None
    if household_metadata is not None:
        household_metadata_path = output_root / _HOUSEHOLD_METADATA_FILENAME
        household_metadata.to_parquet(household_metadata_path, index=False)
        logger.info("Preprocess Step 4: household metadata → %s", household_metadata_path)
    return (
        str(staged_population_path),
        str(counts_path),
        assignment_report,
        str(household_metadata_path) if household_metadata_path else None,
    )
