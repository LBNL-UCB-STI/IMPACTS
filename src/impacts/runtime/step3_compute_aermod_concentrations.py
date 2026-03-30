"""Reserved AERMOD concentrations stage.

This module is intentionally a placeholder until the AERMOD workflow is implemented.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..common import log_step_banner
from ..common import log_substep_banner
from ..manifest.schema import PipelineConfig

logger = logging.getLogger(__name__)


def run(
    *,
    pipeline: PipelineConfig,
    raw_dir: Path,
    emissions_input_path: str,
    output_path: str,
) -> pd.DataFrame:
    log_step_banner("Step 3", "Compute AERMOD Concentrations", logger=logger)
    log_substep_banner("3.0", "AERMOD implementation placeholder", logger=logger)
    del pipeline
    del raw_dir
    del emissions_input_path
    del output_path
    raise NotImplementedError("AERMOD concentrations are reserved but not implemented yet.")
