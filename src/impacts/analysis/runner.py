from __future__ import annotations

from pathlib import Path
from typing import Dict

from impacts.runner import run_analysis_from_pipeline_manifest
from impacts.runner import run_analysis_from_settings


def run_from_settings(
    *,
    settings_path: str | Path,
) -> Dict[str, str]:
    return run_analysis_from_settings(settings_path=settings_path)


def run_from_pipeline_manifest(
    *,
    run_manifest_path: str | Path,
) -> Dict[str, str]:
    return run_analysis_from_pipeline_manifest(run_manifest_path=run_manifest_path)
