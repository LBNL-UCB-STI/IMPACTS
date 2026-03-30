from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from .manifest.file_ops import load_structured_file
from .manifest.file_ops import write_structured_file
from .manifest.schema import InputsManifest
from .manifest.schema import PipelineConfig
from .manifest.schema import RunManifest

logger = logging.getLogger(__name__)

def _log_step_banner(label: str, name: str) -> None:
    banner = f"========== ENTERING {label}: {name.upper()} =========="
    sys.stdout.write("\n")
    sys.stdout.flush()
    logger.info("%s", banner)


def run_from_input_manifest(
    input_manifest_path: str | Path,
    output_dir: str | Path,
    run_manifest_path: str | Path | None = None,
    run_dispersion: bool = False,
) -> Dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=False,
    )
    manifest = InputsManifest.from_dict(load_structured_file(input_manifest_path)).to_dict()
    pipeline = PipelineConfig.from_dict(manifest.get("pipeline", {}) or {})
    population_inputs = manifest.get("population_inputs", {}) or {}
    input_root = Path(manifest.get("input_dir", "")).resolve()

    output_root = Path(output_dir).resolve()
    raw_dir = output_root / "outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded input manifest: %s", Path(input_manifest_path).resolve())
    logger.info("Output directory: %s", output_root)

    from .preprocessing.step3_integrate_grids import run as run_grid_intersection
    from .runtime.step1_process_emissions import run as run_emissions_processing
    from .runtime.prepare_emissions_from_skims import resolve_prepared_skims_path

    intersection_df = None
    _log_step_banner("PREPROCESS STEP 3", "network mapping and grid intersection")
    grid_intersection_path, intersection_df = run_grid_intersection(pipeline, raw_dir, input_root)

    _log_step_banner("STEP 1", "emissions processing")
    logger.info("Using Step 1 implementation: emissions_processing")
    emissions_outputs = run_emissions_processing(
        pipeline,
        raw_dir,
        input_root,
        grid_intersection_path,
        intersection_df=intersection_df,
    )
    prepared_skims_path = resolve_prepared_skims_path(input_root)

    concentration_path: Optional[Path] = None
    if run_dispersion:
        from .runtime.step2_compute_inmap_concentrations import run as run_inmap_dispersion
        _log_step_banner("STEP 2", "inmap concentrations")
        logger.info("Using Step 2 implementation: inmap_concentrations_and_export")
        _, _, concentration_path = run_inmap_dispersion(
            pipeline=pipeline,
            raw_dir=raw_dir,
            emissions_input_path=emissions_outputs["beam_emissions_for_inmap"],
        )
        logger.info("InMAP concentrations complete: wrote %s", concentration_path)
        _log_step_banner("STEP 3", "aermod concentrations")
        logger.info("Step 3 placeholder: aermod concentrations not run yet")
    else:
        logger.info("Dispersion skipped")

    run_manifest = {
        "contract_version": manifest.get("contract_version", "1"),
        "model": "impacts",
        "input_manifest_path": str(Path(input_manifest_path).resolve()),
        "output_dir": str(output_root),
        "raw_output_dir": str(raw_dir),
        "command": " ".join(sys.argv),
        "image": "not_recorded",
        "raw_outputs": {
            "skims_emissions": prepared_skims_path,
            "grid_intersection": str(grid_intersection_path),
            **emissions_outputs,
            "beam_inmap_concentrations": str(concentration_path) if concentration_path else None,
            "beam_inmap_concentrations_gpkg": (
                str(concentration_path.with_suffix(".gpkg")) if concentration_path else None
            ),
        },
        "pipeline": pipeline.to_dict(),
        "population_inputs": population_inputs,
        "deterministic_contract": {
            "uses_only_manifest_paths": True,
            "uses_baked_work_data": False,
        },
        "execution": {
            "dispersion_completed": run_dispersion,
            "stopped_after": "step2_compute_inmap_concentrations" if run_dispersion else "step1_process_emissions",
        },
    }
    output_manifest = Path(run_manifest_path) if run_manifest_path else output_root / "run_manifest.yaml"
    run_manifest["run_manifest_path"] = str(output_manifest)
    typed_manifest = RunManifest.from_dict(run_manifest)
    write_structured_file(output_manifest, typed_manifest.to_dict())
    logger.info("Run manifest written: %s", output_manifest)
    return typed_manifest.to_dict()


def run_from_runtime_config(
    runtime_config_path: str | Path,
    workspace: str | Path,
    run_manifest_path: str | Path | None = None,
    run_dispersion: bool = False,
) -> Dict[str, Any]:
    from impacts.preprocessor import preprocess_workflow

    workspace_root = Path(workspace).resolve()
    preprocess_manifest = preprocess_workflow(
        runtime_config_path=runtime_config_path,
        staging_dir=workspace_root,
    )
    return run_from_input_manifest(
        input_manifest_path=preprocess_manifest["inputs_manifest_path"],
        output_dir=workspace_root,
        run_manifest_path=run_manifest_path,
        run_dispersion=run_dispersion,
    )
