from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict

from .common import log_step_banner
from .config.runtime_builder import build_runtime_config_from_runtime_yaml
from .manifest.file_ops import load_structured_file
from .manifest.file_ops import resolve_path
from .manifest.file_ops import write_structured_file
from .manifest.schema import PostprocessManifest
from .manifest.schema import RunManifest

logger = logging.getLogger(__name__)


def postprocess_from_run_manifest(
    run_manifest_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    run_manifest = RunManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    log_step_banner("Postprocess", "Impacts Complete", logger=logger)
    logger.info("End of impacts concluded.")

    completion_path = output_root / "impacts_complete.txt"
    completion_path.write_text("End of impacts concluded.\n", encoding="utf-8")

    postprocess_manifest = {
        "contract_version": run_manifest.get("contract_version", "1"),
        "model": "impacts",
        "run_manifest_path": str(Path(run_manifest_path).resolve()),
        "output_dir": str(output_root),
        "canonical_artifact": {
            "name": "impacts_complete",
            "path": str(completion_path),
        },
        "validation": {
            "completed": True,
        },
        "notes": [
            "Postprocessor intentionally left empty.",
            "End of impacts concluded.",
        ],
    }
    output_manifest = Path(manifest_path) if manifest_path else output_root / "postprocess_manifest.yaml"
    postprocess_manifest["postprocess_manifest_path"] = str(output_manifest)
    typed_manifest = PostprocessManifest.from_dict(postprocess_manifest)
    write_structured_file(output_manifest, typed_manifest.to_dict())
    logger.info("Postprocess manifest written: %s", output_manifest)
    return typed_manifest.to_dict()


def postprocess_from_runtime_config(
    runtime_config_path: str | Path,
    workspace: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    from impacts.runner import run_from_runtime_config

    workspace_root = Path(workspace).resolve()
    runtime_config = build_runtime_config_from_runtime_yaml(runtime_config_path)
    output_root = Path(
        resolve_path(runtime_config.impacts.local_output_folder, runtime_config_path) or runtime_config.impacts.local_output_folder
    ).resolve()
    run_manifest = run_from_runtime_config(
        runtime_config_path=runtime_config_path,
        workspace=workspace_root,
        run_dispersion=True,
    )
    return postprocess_from_run_manifest(
        run_manifest_path=run_manifest["run_manifest_path"],
        output_dir=output_root,
        manifest_path=manifest_path,
    )
