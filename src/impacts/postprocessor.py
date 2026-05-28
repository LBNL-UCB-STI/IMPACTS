from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict

from .common import log_step_banner
from .config.settings_builder import load_settings_from_yaml
from .manifest.file_ops import load_structured_file
from .manifest.file_ops import write_structured_file
from .manifest.schema import PostprocessManifest
from .manifest.schema import RunManifest

logger = logging.getLogger(__name__)


def postprocess_from_run_manifest(
    run_manifest_path: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    from .runner import run_analysis_from_run_manifest

    run_manifest = RunManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    output_root = Path(str(run_manifest.get("output_dir"))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    log_step_banner("Postprocess", "Impacts Complete", logger=logger)
    logger.info("End of impacts concluded.")

    completion_path = output_root / "impacts_complete.txt"
    completion_path.write_text("End of impacts concluded.\n", encoding="utf-8")
    analysis_outputs = run_analysis_from_run_manifest(
        run_manifest_path=run_manifest_path,
    )

    postprocess_manifest = {
        "contract_version": run_manifest.get("contract_version", "1"),
        "model": "impacts",
        "run_manifest_path": str(Path(run_manifest_path).resolve()),
        "output_dir": str(output_root),
        "canonical_artifact": {
            "name": "impacts_complete",
            "path": str(completion_path),
        },
        "analysis_outputs": analysis_outputs,
        "validation": {
            "completed": True,
        },
        "notes": [
            "Postprocess runs maintained analysis outputs after workflow completion.",
            "End of impacts concluded.",
        ],
    }
    output_manifest = Path(manifest_path) if manifest_path else output_root / "postprocess_manifest.yaml"
    postprocess_manifest["postprocess_manifest_path"] = str(output_manifest)
    typed_manifest = PostprocessManifest.from_dict(postprocess_manifest)
    write_structured_file(output_manifest, typed_manifest.to_dict())
    logger.info("Postprocess manifest written: %s", output_manifest)
    return typed_manifest.to_dict()


def postprocess_from_settings(
    settings_path: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    from impacts.preprocessor import preprocess_workflow
    from impacts.runner import run_aermod_from_run_manifest
    from impacts.runner import run_emissions_from_run_manifest
    from impacts.runner import run_exposure_from_run_manifest
    from impacts.runner import run_inmap_from_run_manifest

    settings = load_settings_from_yaml(settings_path)
    preprocess_manifest = preprocess_workflow(
        settings_path=settings_path,
    )
    run_manifest_path = preprocess_manifest["run_manifest_path"]
    if settings.impacts.pipeline.postsim.emissions:
        run_manifest = run_emissions_from_run_manifest(
            run_manifest_path=run_manifest_path,
        )
        run_manifest_path = run_manifest["run_manifest_path"]
    if settings.impacts.pipeline.postsim.inmap:
        run_manifest = run_inmap_from_run_manifest(
            run_manifest_path=run_manifest_path,
        )
        run_manifest_path = run_manifest["run_manifest_path"]
    if settings.impacts.pipeline.postsim.aermod:
        run_manifest = run_aermod_from_run_manifest(
            run_manifest_path=run_manifest_path,
        )
        run_manifest_path = run_manifest["run_manifest_path"]
    if settings.impacts.pipeline.postsim.exposure:
        run_manifest = run_exposure_from_run_manifest(
            run_manifest_path=run_manifest_path,
        )
        run_manifest_path = run_manifest["run_manifest_path"]
    return postprocess_from_run_manifest(
        run_manifest_path=run_manifest_path,
        manifest_path=manifest_path,
    )
