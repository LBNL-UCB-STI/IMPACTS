"""Workflow skims emissions preparation.

Build or load the skims-emissions table, filter to configured pollutants,
and annualize from grams/day to tons/year.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from typing import Tuple

import pandas as pd

from ..manifest.file_ops import parquet_available
from ..manifest.schema import PipelineConfig

logger = logging.getLogger(__name__)


def _table_path(parent: Path, stem: str) -> Path:
    suffix = ".parquet" if parquet_available() else ".csv.gz"
    path = parent / f"{stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_rates(rates_dir: Optional[str]):
    if not rates_dir:
        return None
    from impacts.utils.utils_events_to_skims_emissions import read_rates_directory
    return read_rates_directory(rates_dir)


def run(pipeline: PipelineConfig, raw_dir: Path) -> Tuple[pd.DataFrame, Path]:
    """Build or load skims emissions, filter to pollutants, annualize.

    Returns (skims_df, skims_path).
    """
    from impacts.workflow.step4_emissions_distribution import annualize_skims
    from impacts.workflow.step4_emissions_distribution import read_skims_emissions
    from impacts.utils.utils_events_to_skims_emissions import build_skims_emissions_from_events
    from impacts.utils.utils_events_to_skims_emissions import write_skims_emissions

    skims_path = _table_path(raw_dir, "skims_emissions")
    prepared_skims_path = Path(pipeline.prepared_skims_input_path) if pipeline.prepared_skims_input_path else None
    staged_skims_path = Path(pipeline.skims_input_path) if pipeline.skims_input_path else None

    # Step 1.1: prefer preprocess-produced annualized skims when available.
    using_prepared_skims = False
    if prepared_skims_path and prepared_skims_path.exists():
        skims_df = None
        skims_path = prepared_skims_path
        using_prepared_skims = True
        logger.info("Step 1.1: using prepared staged skims from %s", skims_path)
    elif staged_skims_path and staged_skims_path.exists():
        skims_df = None
        skims_path = staged_skims_path
        logger.info("Step 1.1: using staged raw skims from %s", skims_path)
    elif not skims_path.exists():
        if not pipeline.events_path:
            raise ValueError("Skims file not found and no events_path configured to build from.")
        logger.info("Step 1.1: building skims from events %s", pipeline.events_path)
        rates_df = None
        if pipeline.use_rates:
            rates_df = _load_rates(pipeline.rates_dir)
            logger.info("Loaded rates from %s", pipeline.rates_dir)
        skims_df = build_skims_emissions_from_events(
            events_path=pipeline.events_path,
            network_path=pipeline.link_length_path,
            rates_df=rates_df,
            iterations=int(pipeline.iterations),
        )
        write_skims_emissions(skims_df, str(skims_path))
        logger.info("Step 1.1 complete: wrote %s", skims_path)
    else:
        skims_df = None
        logger.info("Step 1.1: skims already available at %s", skims_path)

    # Step 1.2: load, filter to pollutants, annualize
    source_pollutants = list(pipeline.pollutants_map.values()) if pipeline.pollutants_map else list(pipeline.pollutants)
    pollutants = source_pollutants if source_pollutants else None
    if skims_df is None:
        logger.info("Step 1.2: loading skims from %s", skims_path)
        load_pollutants = None if using_prepared_skims else pollutants
        skims_df = read_skims_emissions(str(skims_path), pollutants=load_pollutants)
        logger.info("Step 1.2: loaded %d rows", len(skims_df))
    elif pollutants:
        dim_cols = [c for c in skims_df.columns if c in {"linkId", "vehicleTypeId", "process"}]
        pollutant_cols = [c for c in pollutants if c in skims_df.columns]
        skims_df = skims_df[dim_cols + pollutant_cols]
        logger.info("Step 1.2: filtered in-memory skims to %d pollutant columns", len(pollutant_cols))

    if pollutants:
        logger.info("Step 1.2: annualizing skims (%g days)", pipeline.annualization_days)
        skims_df = annualize_skims(skims_df, pollutants, float(pipeline.annualization_days))
        logger.info("Step 1.2 complete: annualized %d pollutants", len(pollutants))

    return skims_df, skims_path
