from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config.builders import build_runtime_config_from_pilates
from .contract_utils import write_structured_file


def derive_runtime_config_from_pilates(
    *,
    pilates_settings_path: str | Path,
    impacts_model_config_path: str | Path,
    output_path: str | Path | None = None,
) -> Dict:
    runtime_config = build_runtime_config_from_pilates(
        pilates_settings=pilates_settings_path,
        impacts_overlay=impacts_model_config_path,
    )
    payload = runtime_config.to_dict()
    if output_path is not None:
        write_structured_file(output_path, payload)
    return payload
