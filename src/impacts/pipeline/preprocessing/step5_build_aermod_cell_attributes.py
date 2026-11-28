"""Preprocess Step 5 — Build AERMOD cell attributes."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ...common import log_step_banner
from ...common import log_substep_banner
from ...manifest.schema import PipelineConfig
from ._common import load_aermod_grid

logger = logging.getLogger(__name__)

_AERMOD_CELL_ID = "aermod_cell_id"
_OUTPUT_FILENAME = "aermod_cell_attributes.parquet"


def _classify_urban(person_count: pd.Series) -> pd.Series:
    values = pd.to_numeric(person_count, errors="coerce").fillna(0.0)
    return pd.Series(
        np.where(values < 1000, 0, np.where(values < 10000, 1000, 10000)),
        index=person_count.index,
        dtype="int64",
    )


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


def run(
    pipeline: PipelineConfig,
    output_root: Path,
    *,
    population_counts_path: Optional[str | Path] = None,
) -> Optional[str]:
    log_step_banner("Preprocess Step 5", "Build AERMOD Cell Attributes", logger=logger)

    if not pipeline.aermod_grid_path:
        logger.info("Preprocess Step 5: no AERMOD grid configured; skipping.")
        return None

    output_root.mkdir(parents=True, exist_ok=True)
    cell_attributes_path = output_root / _OUTPUT_FILENAME

    log_substep_banner("5.1", "load AERMOD study-area grid", logger=logger)
    aermod_grid = load_aermod_grid(pipeline)
    logger.info("Preprocess Step 5.1: loaded %d AERMOD study-area cells", len(aermod_grid))

    counts = pd.read_parquet(population_counts_path) if population_counts_path else None
    log_substep_banner("5.2", "build AERMOD cell attributes", logger=logger)
    cell_attributes = _complete_cell_attributes(aermod_grid, counts)
    logger.info(
        "Preprocess Step 5.2: built attributes for %d AERMOD domain cells "
        "(urban=%d, suburban=%d, rural=%d)",
        len(cell_attributes),
        (cell_attributes["source_urban_class"] == 10000).sum(),
        (cell_attributes["source_urban_class"] == 1000).sum(),
        (cell_attributes["source_urban_class"] == 0).sum(),
    )

    cell_attributes.to_parquet(cell_attributes_path, index=False)
    logger.info("Preprocess Step 5: AERMOD cell attributes → %s", cell_attributes_path)
    return str(cell_attributes_path)
