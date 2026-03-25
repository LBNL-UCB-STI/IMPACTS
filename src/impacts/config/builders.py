from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict

from impacts.adapters.pilates import build_runtime_payload_from_pilates
from impacts.contract_utils import load_structured_file
from impacts.contract_utils import write_structured_file
from impacts.runtime_config import ImpactsRuntimeConfig


def build_runtime_config_from_runtime_yaml(path: str | Path) -> ImpactsRuntimeConfig:
    payload = load_structured_file(path)
    return ImpactsRuntimeConfig.from_dict(payload)


def build_runtime_config_from_pilates(
    pilates_settings: Dict[str, Any] | str | Path,
    impacts_overlay: Dict[str, Any] | str | Path,
) -> ImpactsRuntimeConfig:
    pilates_payload = (
        load_structured_file(pilates_settings)
        if isinstance(pilates_settings, (str, Path))
        else dict(pilates_settings)
    )
    overlay_payload = (
        load_structured_file(impacts_overlay)
        if isinstance(impacts_overlay, (str, Path))
        else dict(impacts_overlay)
    )
    runtime_payload = build_runtime_payload_from_pilates(pilates_payload, overlay_payload)
    return ImpactsRuntimeConfig.from_dict(runtime_payload)


def write_runtime_config(runtime_config: ImpactsRuntimeConfig, output_path: str | Path) -> Dict[str, Any]:
    payload = runtime_config.to_dict()
    write_structured_file(output_path, payload)
    return payload
