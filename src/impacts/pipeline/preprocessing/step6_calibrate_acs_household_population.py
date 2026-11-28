"""Preprocess Step 6 — Calibrate household weights to ACS county controls."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ...acs_data import read_acs_estimates
from ...common import log_step_banner
from ...common import log_substep_banner

logger = logging.getLogger(__name__)


def _calibrate_county(households: pd.DataFrame, *, target_households: float, target_population: float) -> np.ndarray:
    sizes = households["household_size"].to_numpy(dtype=float)
    if not np.isfinite(sizes).all() or (sizes < 1).any():
        raise ValueError("ACS calibration requires a positive household size for every household")
    if target_households <= 0 or target_population <= 0:
        raise ValueError("ACS household and household-population controls must be positive")

    target_mean = target_population / target_households
    minimum, maximum = float(sizes.min()), float(sizes.max())
    if target_mean < minimum or target_mean > maximum:
        raise ValueError(
            "ACS target mean household size is outside the supplied UrbanSim range: "
            f"target={target_mean:.6f}, range=[{minimum:.0f}, {maximum:.0f}]"
        )

    centered = sizes - minimum

    def weighted_mean(log_size_slope: float) -> float:
        log_weights = log_size_slope * centered
        scaled_weights = np.exp(log_weights - log_weights.max())
        return float(np.dot(sizes, scaled_weights) / scaled_weights.sum())

    if minimum == maximum:
        slope = 0.0
    else:
        slope = brentq(lambda value: weighted_mean(value) - target_mean, -50.0, 50.0)
    log_weights = slope * centered
    raw_weights = np.exp(log_weights - log_weights.max())
    return raw_weights * (target_households / raw_weights.sum())


def run(
    *,
    household_metadata_path: str | Path,
    acs_tables: dict[str, dict[str, str]],
    state_fips: str,
    county_fips_codes: list[str],
    year: int,
    output_root: Path,
) -> tuple[str, str, dict[str, dict[str, float]]]:
    log_step_banner("Preprocess Step 6", "Calibrate ACS Household Population", logger=logger)
    log_substep_banner("6.1", "read baseline-year ACS county controls", logger=logger)
    county_geoids = [f"{state_fips}{county}" for county in county_fips_codes]
    prefixes = tuple(county_geoids)
    households = pd.read_parquet(household_metadata_path)
    required = {"household_id", "county_geoid", "household_size"}
    if missing := required - set(households.columns):
        raise ValueError(f"Household metadata is missing calibration columns: {sorted(missing)}")
    if households["household_id"].duplicated().any():
        raise ValueError("Household metadata must contain one row per household_id")

    household_controls = read_acs_estimates(
        acs_tables["1yr"]["B11001"],
        table="B11001",
        geography="county",
        geoid_prefixes=prefixes,
    ).set_index("geoid")
    population_controls = read_acs_estimates(
        acs_tables["1yr"]["B11002"],
        table="B11002",
        geography="county",
        geoid_prefixes=prefixes,
    ).set_index("geoid")
    missing_geoids = set(county_geoids) - set(household_controls.index) | (
        set(county_geoids) - set(population_controls.index)
    )
    if missing_geoids:
        raise ValueError(f"ACS county controls are missing required GEOIDs: {sorted(missing_geoids)}")

    households["county_geoid"] = households["county_geoid"].astype("string")
    if households["county_geoid"].isna().any():
        raise ValueError("UrbanSim household metadata contains a household with no county GEOID")
    unexpected_geoids = set(households["county_geoid"].unique()) - set(county_geoids)
    if unexpected_geoids:
        raise ValueError(
            "UrbanSim household metadata contains counties outside the configured ACS region: "
            f"{sorted(unexpected_geoids)}"
        )
    weights = pd.Series(index=households.index, dtype="float64")
    diagnostics: dict[str, dict[str, float]] = {}
    log_substep_banner("6.2", "fit one weight per whole household", logger=logger)
    for geoid in county_geoids:
        mask = households["county_geoid"].eq(geoid)
        county_households = households.loc[mask]
        if county_households.empty:
            raise ValueError(f"UrbanSim population has no households in ACS county {geoid}")
        target_households = float(household_controls.at[geoid, "estimate"])
        target_population = float(population_controls.at[geoid, "estimate"])
        weights.loc[mask] = _calibrate_county(
            county_households,
            target_households=target_households,
            target_population=target_population,
        )
        diagnostics[geoid] = {
            "acs_households": target_households,
            "acs_households_moe": float(household_controls.at[geoid, "moe"]),
            "acs_household_population": target_population,
            "acs_household_population_moe": float(population_controls.at[geoid, "moe"]),
            "urbansim_households": float(len(county_households)),
            "urbansim_residents": float(county_households["household_size"].sum()),
            "calibrated_households": float(weights.loc[mask].sum()),
            "calibrated_residents": float(
                (weights.loc[mask].to_numpy() * county_households["household_size"].to_numpy()).sum()
            ),
        }

    result = households[["household_id", "county_geoid", "household_size"]].copy()
    result["acs_household_weight"] = weights.to_numpy()
    output_root.mkdir(parents=True, exist_ok=True)
    weights_path = output_root / "acs_household_weights.parquet"
    result.to_parquet(weights_path, index=False)
    summary_path = output_root / "acs_household_calibration.json"
    summary = {
        "baseline_year": int(year),
        "acs_1yr_release_year": int(year),
        "acs_5yr_period": f"{int(year) - 4}-{int(year)}",
        "method": "county-level entropy calibration by household size to B11001 households and B11002 household population",
        "household_rows_are_weighted_as_units": True,
        "group_quarters_included": False,
        "counties": diagnostics,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Preprocess Step 6 calibrated household weights → %s", weights_path)
    return str(weights_path), str(summary_path), diagnostics
