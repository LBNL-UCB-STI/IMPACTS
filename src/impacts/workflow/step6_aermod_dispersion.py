"""Reserved AERMOD dispersion stage.

This module is intentionally a placeholder until the AERMOD workflow is implemented.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .manifest_models import PipelineConfig

logger = logging.getLogger(__name__)


def run(
    *,
    pipeline: PipelineConfig,
    raw_dir: Path,
    emissions_input_path: str,
    output_path: str,
) -> pd.DataFrame:
    del pipeline
    del raw_dir
    del emissions_input_path
    del output_path
    raise NotImplementedError("AERMOD dispersion is reserved but not implemented yet.")
