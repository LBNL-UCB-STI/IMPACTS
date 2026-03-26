from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from .contract_utils import load_structured_file
from .contract_utils import parquet_available
from .contract_utils import write_structured_file
from .defaults import DEFAULT_CONCENTRATION_FACTOR
from .manifest_models import InputsManifest
from .manifest_models import PipelineConfig
from .manifest_models import RunManifest

logger = logging.getLogger(__name__)


def _table_path(parent: Path, stem: str) -> Path:
    suffix = ".parquet" if parquet_available() else ".csv.gz"
    path = parent / f"{stem}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded input manifest: %s", Path(input_manifest_path).resolve())
    logger.info("Output directory: %s", output_root)

    from .step1_skims_preparation import run as run_step1
    from .step2_network_osm_mapping import run as run_step2
    from .step3_grid_intersection import run as run_step3
    from .step4_emissions_distribution import run as run_step4

    skims_df, skims_path = run_step1(pipeline, raw_dir)

    mapping_input_path = pipeline.mapping_input_path
    mapping_input_df = None
    if not mapping_input_path:
        buffered_network = run_step2(pipeline, raw_dir)
        mapping_input_path, mapping_input_df = run_step3(pipeline, raw_dir, buffered_network)
    else:
        logger.info("Using staged mapping input: %s", mapping_input_path)

    step4_outputs = run_step4(
        pipeline,
        raw_dir,
        skims_df,
        mapping_input_path,
        intersection_df=mapping_input_df,
    )

    concentration_path: Optional[Path] = None
    if run_dispersion:
        from impacts.dispersion.isrm_dispersion import run_dispersion_from_file
        concentration_path = _table_path(raw_dir, "grid_concentration")
        logger.info("Dispersion: computing concentrations from allocated grid emissions")
        run_dispersion_from_file(
            emissions_input_path=step4_outputs["emissions_corrected"],
            output_path=str(concentration_path),
            isrm_url=pipeline.isrm_url,
            factor=float(pipeline.concentration_factor or DEFAULT_CONCENTRATION_FACTOR),
            include_bc=bool(pipeline.include_bc),
            include_health=bool(pipeline.include_health),
        )
        logger.info("Dispersion complete: wrote %s", concentration_path)
    else:
        logger.info("Dispersion skipped")

    run_manifest = {
        "contract_version": manifest.get("contract_version", "1"),
        "model": "impacts",
        "input_manifest_path": str(Path(input_manifest_path).resolve()),
        "output_dir": str(output_root),
        "raw_output_dir": str(raw_dir),
        "command": " ".join(sys.argv),
        "image": os.getenv("IMPACTS_IMAGE", "unknown"),
        "raw_outputs": {
            "skims_emissions": str(skims_path),
            "grid_intersection": str(mapping_input_path),
            **step4_outputs,
            "grid_concentration": str(concentration_path) if concentration_path else None,
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
        output_dir=workspace_root / "output",
        run_manifest_path=run_manifest_path,
        run_dispersion=run_dispersion,
    )
