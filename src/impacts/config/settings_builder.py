from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict

from impacts.adapters.pilates import build_settings_payload_from_pilates
from impacts.manifest.file_ops import load_structured_file
from impacts.manifest.file_ops import write_structured_file
from impacts.config.settings import ImpactsSettings


def load_settings_from_yaml(path: str | Path) -> ImpactsSettings:
    payload = load_structured_file(path)
    if isinstance(payload, dict):
        payload = dict(payload)
        payload["__source_root__"] = str(Path(path).resolve().parent)
    settings_payload = build_settings_payload_from_pilates(
        payload,
        {
            "impacts": dict(payload.get("impacts", {}) or {}),
            "__source_root__": payload.get("__source_root__"),
        },
    )
    return ImpactsSettings.from_dict(settings_payload)


def build_settings_from_pilates(
    pilates_settings: Dict[str, Any] | str | Path,
    impacts_overlay: Dict[str, Any] | str | Path,
) -> ImpactsSettings:
    pilates_payload = (
        load_structured_file(pilates_settings)
        if isinstance(pilates_settings, (str, Path))
        else dict(pilates_settings)
    )
    if isinstance(pilates_settings, (str, Path)):
        pilates_payload = dict(pilates_payload)
        pilates_payload["__source_root__"] = str(Path(pilates_settings).resolve().parent)
    overlay_payload = (
        load_structured_file(impacts_overlay)
        if isinstance(impacts_overlay, (str, Path))
        else dict(impacts_overlay)
    )
    if isinstance(impacts_overlay, (str, Path)):
        overlay_payload = dict(overlay_payload)
        overlay_payload["__source_root__"] = str(Path(impacts_overlay).resolve().parent)
    settings_payload = build_settings_payload_from_pilates(pilates_payload, overlay_payload)
    return ImpactsSettings.from_dict(settings_payload)


def write_settings(settings: ImpactsSettings, output_path: str | Path) -> Dict[str, Any]:
    payload = settings.to_dict()
    write_structured_file(output_path, payload)
    return payload
