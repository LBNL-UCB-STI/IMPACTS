from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from .contract_utils import load_structured_file
from .contract_utils import write_structured_file
from .manifest_models import InputsManifest
from .manifest_models import PipelineConfig
from .manifest_models import RunManifest

logger = logging.getLogger(__name__)


def _log_step_banner(step_num: int, name: str) -> None:
    banner = f"========== ENTERING STEP {step_num}: {name.upper()} =========="
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

    output_root = Path(output_dir).resolve()
    raw_dir = output_root / "outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded input manifest: %s", Path(input_manifest_path).resolve())
    logger.info("Output directory: %s", output_root)

    from .step1_skims_preparation import run as run_step1
    from .step2_network_osm_mapping import run as run_step2
    from .step3_grid_intersection import run as run_step3
    from .step4_emissions_distribution import run as run_step4

    _log_step_banner(1, "skims preparation")
    skims_df, skims_path = run_step1(pipeline, raw_dir)

    mapping_input_path = pipeline.mapping_input_path
    mapping_input_df = None
    if not mapping_input_path:
        _log_step_banner(2, "network osm mapping")
        buffered_network = run_step2(pipeline, raw_dir)
        _log_step_banner(3, "grid intersection")
        mapping_input_path, mapping_input_df = run_step3(pipeline, raw_dir, buffered_network)
    else:
        logger.info("Using staged mapping input: %s", mapping_input_path)

    _log_step_banner(4, "emissions distribution")
    logger.info("Using Step 4 implementation: combined")
    step4_outputs = run_step4(
        pipeline,
        raw_dir,
        skims_df,
        mapping_input_path,
        intersection_df=mapping_input_df,
    )

    concentration_path: Optional[Path] = None
    if run_dispersion:
        from .step5_inmap_dispersion import run as run_step5
        _log_step_banner(5, "inmap concentrations")
        logger.info("Using Step 5 implementation: inmap_concentrations_and_export")
        _, _, concentration_path = run_step5(
            pipeline=pipeline,
            raw_dir=raw_dir,
            emissions_input_path=step4_outputs["beam_emissions_for_inmap"],
        )
        logger.info("InMAP concentrations complete: wrote %s", concentration_path)
        _log_step_banner(6, "aermod dispersion")
        logger.info("Step 6 placeholder: aermod_dispersion not run yet")
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
            "skims_emissions": str(skims_path),
            "grid_intersection": str(mapping_input_path),
            **step4_outputs,
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
            "stopped_after": "dispersion" if run_dispersion else "emissions_distribution",
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
