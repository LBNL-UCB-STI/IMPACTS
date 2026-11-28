"""Preprocess Step 8 — Allocate ACS group quarters residents and assign them to the grid."""
from __future__ import annotations

import logging
from pathlib import Path
import re
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd

from ...acs_data import read_acs_estimates
from ...common import log_step_banner, resolve_required_manifest_input
from ...manifest.schema import PipelineConfig
from ._common import load_aermod_grid
from .step4_assign_raw_population_to_grid import _join_population_to_grid

logger = logging.getLogger(__name__)
CELL = "aermod_cell_id"
POPULATION = "group_quarters_population"


def _read_p42(path: str, counties: tuple[str, ...]) -> pd.Series:
    records = {}
    prefixes = tuple(county.encode("ascii") for county in counties)
    with ZipFile(path) as archive:
        geography, = (name for name in archive.namelist() if re.fullmatch(r"[a-z]{2}geo2010\.sgq", name))
        with archive.open(geography) as header, archive.open(f"{geography[:2]}000012010.sgq") as values:
            for geo, data in zip(header, values, strict=True):
                if geo[8:11] != b"750" or geo[27:32] not in prefixes:
                    continue
                fields = data.split(b",")
                if geo[18:25] != fields[4]:
                    raise ValueError("P42 geography and population records have different LOGRECNO values")
                # Census GQ Summary File, geographic header positions for state, county, tract and block.
                block = (geo[27:32] + geo[54:60] + geo[61:65]).decode("ascii")
                population = int(fields[5])
                if not re.fullmatch(r"\d{15}", block) or population < 0 or block in records:
                    raise ValueError(f"Invalid or duplicate P42 block record: {block}")
                records[block] = population
    if not records:
        raise ValueError("P42 archive contains no blocks in the configured counties")
    return pd.Series(records, dtype="float64")


def _read_block_weights(
    path: str, *, id_column: str, weight_column: str, summary_rows: tuple[str, ...] = ()
) -> pd.Series:
    frame = pd.read_csv(path, usecols=[id_column, weight_column], dtype={id_column: str})
    frame = frame.loc[~frame[id_column].isin(summary_rows)]
    weights = pd.to_numeric(frame[weight_column], errors="raise")
    if (
        not frame[id_column].str.fullmatch(r"\d{15}", na=False).all()
        or frame[id_column].duplicated().any()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
    ):
        raise ValueError(f"Invalid block IDs or weights in {path}")
    return pd.Series(weights.to_numpy(dtype=float), index=frame[id_column])


def _tract_targets(acs_tables: dict, counties: tuple[str, ...]) -> pd.DataFrame:
    county = read_acs_estimates(
        acs_tables["1yr"]["B26001"], table="B26001", geography="county", geoid_prefixes=counties
    ).set_index("geoid")["estimate"]
    tract = read_acs_estimates(
        acs_tables["5yr"]["B26001"], table="B26001", geography="tract", geoid_prefixes=counties
    ).rename(columns={"geoid": "tract_geoid", "estimate": "acs_5yr_estimate", "moe": "acs_5yr_moe"})
    county_totals = tract.groupby(tract["tract_geoid"].str[:5])["acs_5yr_estimate"].sum()
    if set(counties) - set(county.index) or (county < 0).any() or (tract["acs_5yr_estimate"] < 0).any():
        raise ValueError("ACS group quarters controls have missing counties or negative estimates")
    if (county.gt(0) & county_totals.reindex(county.index).fillna(0).eq(0)).any():
        raise ValueError("A positive ACS county group quarters target has no positive tract estimates")
    tract["county_scale"] = tract["tract_geoid"].str[:5].map(county / county_totals.replace(0, np.nan)).fillna(0)
    tract["tract_target"] = tract["acs_5yr_estimate"] * tract["county_scale"]
    return tract.loc[tract["tract_target"].gt(0)]


def _allocate_blocks(
    targets: pd.DataFrame, p42: pd.Series, luca: pd.Series, occupancy: pd.Series
) -> pd.DataFrame:
    candidates = {
        name: {tract: rows for tract, rows in source[source.gt(0)].groupby(source.index.str[:11][source.gt(0)])}
        for name, source in (("p42", p42), ("luca", luca))
    }
    allocations = []
    for tract in targets.itertuples(index=False):
        weights = candidates["p42"].get(tract.tract_geoid)
        method = "p42_residents"
        if weights is None:
            weights = candidates["luca"].get(tract.tract_geoid)
            if weights is None:
                raise ValueError(f"No P42 residents or LUCA group quarters addresses for tract {tract.tract_geoid}")
            method = "luca_single_block" if len(weights) == 1 else "luca_address_count"
            if len(weights) > 1:
                observed = occupancy.reindex(weights.index)
                if observed.notna().all() and observed.sum() > 0:
                    weights = observed
                    method = "luca_verified_occupancy"
        shares = weights / weights.sum()
        allocations.append(pd.DataFrame({
            "block_geoid": shares.index,
            "tract_geoid": tract.tract_geoid,
            "acs_5yr_estimate": tract.acs_5yr_estimate,
            "acs_5yr_moe": tract.acs_5yr_moe,
            "tract_target": tract.tract_target,
            "method": method,
            "weight": shares.to_numpy(),
            POPULATION: shares.to_numpy() * tract.tract_target,
        }))
    columns = ["block_geoid", "tract_geoid", "acs_5yr_estimate", "acs_5yr_moe",
               "tract_target", "method", "weight", POPULATION]
    if not allocations:
        return pd.DataFrame(columns=columns).astype({POPULATION: float})
    blocks = pd.concat(allocations, ignore_index=True)
    allocated = blocks.groupby("tract_geoid")[POPULATION].sum()
    expected = targets.set_index("tract_geoid")["tract_target"]
    if not np.allclose(allocated.reindex(expected.index), expected):
        raise ValueError("Group quarters block allocations do not preserve ACS tract targets")
    return blocks


def run(
    pipeline: PipelineConfig, output_root: Path, *, population_inputs: dict
) -> tuple[str, str, dict]:
    log_step_banner("Preprocess Step 8", "Assign Group Quarters to Exposure Grid", logger=logger)
    bundle = population_inputs["census_bundle"]
    counties = tuple(f"{pipeline.county_state_fips}{code}" for code in pipeline.county_fips_codes)
    targets = _tract_targets(population_inputs["acs_tables"], counties)
    p42 = _read_p42(bundle["p42"], counties)
    luca = _read_block_weights(
        bundle["luca"],
        id_column="GEOID",
        weight_column="CURRENT_COUNT_OF_GQs",
        summary_rows=("Unable to Geocode", "TOTAL"),
    )
    occupancy = pd.Series(dtype=float)
    if bundle["occupancy"]:
        occupancy = _read_block_weights(
            bundle["occupancy"], id_column="block_geoid", weight_column="residents"
        )
    blocks = _allocate_blocks(targets, p42, luca, occupancy)

    h5_path = resolve_required_manifest_input(
        {"population_h5": population_inputs["h5"]}, key="population_h5"
    )
    coordinates = pd.read_hdf(h5_path, key=f"/{population_inputs['year']}/blocks")
    if "block_id" in coordinates.columns:
        coordinates = coordinates.set_index("block_id")
    coordinates.index = coordinates.index.astype(str)
    blocks = blocks.join(coordinates[["x", "y"]], on="block_geoid", validate="one_to_one")
    blocks[["x", "y"]] = blocks[["x", "y"]].apply(pd.to_numeric, errors="coerce")
    has_coordinates = blocks["x"].between(-180, 180) & blocks["y"].between(-90, 90)
    located = blocks.loc[has_coordinates, ["block_geoid", "x", "y"]]
    points = gpd.GeoDataFrame(located, geometry=gpd.points_from_xy(located.x, located.y), crs=4326)
    grid = load_aermod_grid(pipeline)
    joined = _join_population_to_grid(
        population=points.to_crs(grid.crs), grid=grid, id_column="block_geoid"
    )
    blocks[CELL] = blocks["block_geoid"].map(joined.set_index("block_geoid")[CELL]).astype("Int64")
    assigned = blocks[CELL].notna()
    cell_counts = blocks.loc[assigned].groupby(CELL, as_index=False)[POPULATION].sum()
    blocks_path = output_root / "group_quarters_block_allocations.parquet"
    cells_path = output_root / "group_quarters_cell_counts.parquet"
    blocks.to_parquet(blocks_path, index=False)
    cell_counts.to_parquet(cells_path, index=False)
    report = {
        "acs_tract_target": float(targets["tract_target"].sum()),
        "assigned_to_grid": float(blocks.loc[assigned, POPULATION].sum()),
        "blocks_missing_coordinates": int((~has_coordinates).sum()),
        "missing_block_coordinates": float(blocks.loc[~has_coordinates, POPULATION].sum()),
        "blocks_outside_grid": int((has_coordinates & ~assigned).sum()),
        "outside_grid": float(blocks.loc[has_coordinates & ~assigned, POPULATION].sum()),
        "allocated_by_method": blocks.groupby("method")[POPULATION].sum().to_dict(),
    }
    logger.info("Preprocess Step 8 group quarters assignment: %s", report)
    return str(blocks_path), str(cells_path), report
