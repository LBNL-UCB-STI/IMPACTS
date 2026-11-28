"""Preprocess Step 7 — Assign ACS-weighted households to the exposure grid."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ...common import log_step_banner
from ...common import log_substep_banner
from ...common import resolve_required_manifest_input

logger = logging.getLogger(__name__)


def run(
    *,
    staged_population_entry: dict[str, Any],
    household_weights_entry: dict[str, Any],
    output_root: Path,
) -> tuple[str, dict[str, float]]:
    log_step_banner("Preprocess Step 7", "Assign ACS-Weighted Households to Exposure Grid", logger=logger)
    log_substep_banner("7.1", "read raw resident assignments and household weights", logger=logger)
    staged_path = resolve_required_manifest_input(
        {"staged_population": staged_population_entry}, key="staged_population"
    )
    weights_path = resolve_required_manifest_input(
        {"acs_household_weights": household_weights_entry}, key="acs_household_weights"
    )
    population = pd.read_parquet(staged_path, columns=["person_id", "household_id", "aermod_cell_id"])
    weights = pd.read_parquet(weights_path, columns=["household_id", "acs_household_weight"])
    if population["person_id"].duplicated().any():
        raise ValueError("Raw resident assignments must contain one row per person_id")
    if weights["household_id"].duplicated().any():
        raise ValueError("ACS household weights must contain one row per household_id")

    log_substep_banner("7.2", "apply each household weight to its residents", logger=logger)
    assigned = population.merge(weights, on="household_id", how="left", validate="many_to_one")
    missing_weights = int(assigned["acs_household_weight"].isna().sum())
    if missing_weights:
        missing_households = assigned.loc[
            assigned["acs_household_weight"].isna(), "household_id"
        ].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"ACS household weights are missing for {missing_weights} assigned residents; "
            f"example household_ids={missing_households}"
        )
    counts = (
        assigned.groupby("aermod_cell_id", as_index=False)
        .agg(acs_household_population=("acs_household_weight", "sum"))
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "acs_household_population_cell_counts.parquet"
    counts.to_parquet(output_path, index=False)
    summary = {
        "residents_assigned_to_grid": float(len(assigned)),
        "acs_household_population_assigned_to_grid": float(assigned["acs_household_weight"].sum()),
        "aermod_cells_with_acs_household_population": float(len(counts)),
    }
    logger.info("Preprocess Step 7 assignment summary: %s", summary)
    logger.info("Preprocess Step 7 ACS-adjusted cell counts → %s", output_path)
    return str(output_path), summary
