from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from .config.settings_builder import load_settings_from_yaml
from .manifest.file_ops import load_structured_file
from .manifest.file_ops import resolve_path
from .manifest.file_ops import write_structured_file
from .manifest.schema import InputsManifest
from .manifest.schema import PipelineConfig
from .manifest.schema import RunManifest

logger = logging.getLogger(__name__)

def _normalized_stage_label(label: str) -> str:
    text = str(label).strip()
    upper = text.upper()
    if upper.startswith("PREPROCESS STEP"):
        return upper
    if upper.startswith("STEP"):
        return f"WORKFLOW {upper}"
    return upper


def _log_step_banner(label: str, name: str) -> None:
    banner = f"========== ENTERING {_normalized_stage_label(label)}: {name.upper()} =========="
    sys.stdout.write("\n")
    sys.stdout.flush()
    logger.info("%s", banner)


def _resolve_runtime_output_root(
    *,
    input_manifest: Dict[str, Any],
) -> Path:
    settings_source = input_manifest.get("settings_source")
    if not settings_source:
        raise ValueError("Input manifest is missing settings_source; cannot resolve impacts.local_output_folder.")
    settings = load_settings_from_yaml(settings_source)
    resolved = resolve_path(settings.impacts.local_output_folder, settings_source)
    if not resolved:
        raise ValueError("Could not resolve impacts.local_output_folder from settings.")
    return Path(resolved).resolve()


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
    manifest_inputs = manifest.get("inputs", {}) or {}
    input_root = Path(manifest.get("input_dir", "")).resolve()

    output_root = _resolve_runtime_output_root(
        input_manifest=manifest,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded input manifest: %s", Path(input_manifest_path).resolve())
    logger.info("Output directory: %s", output_root)

    from .pipeline.preprocessing.step3_integrate_grids import run as run_grid_intersection
    from .pipeline.workflow.step1_process_emissions import run as run_emissions_processing
    from .common import prepared_table_target

    _log_step_banner("PREPROCESS STEP 3", "network mapping and grid intersection")
    grid_intersection_path, intersection_df = run_grid_intersection(
        pipeline,
        output_root,
        input_root,
        manifest_inputs=manifest_inputs,
    )

    _log_step_banner("STEP 1", "emissions processing")
    logger.info("Using Step 1 implementation: emissions_processing")
    emissions_outputs = run_emissions_processing(
        pipeline,
        output_root,
        input_root,
        grid_intersection_path,
        intersection_df=intersection_df,
        manifest_inputs=manifest_inputs,
    )
    prepared_skims_candidate = prepared_table_target(input_root, "prepared_skims_for_grid_allocation")
    prepared_skims_path = str(prepared_skims_candidate) if prepared_skims_candidate.exists() else None

    concentration_path: Optional[Path] = None
    aermod_concentration_path: Optional[Path] = None
    exposure_grid_path: Optional[Path] = None
    population_distribution_path: Optional[Path] = None
    population_counts_path: Optional[Path] = None
    if run_dispersion:
        from .workflow.step2_compute_inmap_concentrations import run as run_inmap_dispersion
        from .workflow.step3_compute_aermod_concentrations import run as run_aermod_dispersion
        from .workflow.step4_prepare_exposure import run as run_prepare_exposure
        if pipeline.inmap_enabled and emissions_outputs.get("beam_emissions_for_inmap"):
            _log_step_banner("STEP 2", "inmap concentrations")
            logger.info("Using Step 2 implementation: inmap_concentrations_and_export")
            _, _, concentration_path = run_inmap_dispersion(
                pipeline=pipeline,
                raw_dir=output_root,
                emissions_input_path=emissions_outputs["beam_emissions_for_inmap"],
                inmap_study_area_grid_path=emissions_outputs.get("beam_inmap_study_area_grid"),
            )
            logger.info("InMAP concentrations complete: wrote %s", concentration_path)
        else:
            logger.info(
                "InMAP concentrations skipped: inmap_enabled=%s beam_emissions_for_inmap=%s",
                pipeline.inmap_enabled,
                emissions_outputs.get("beam_emissions_for_inmap"),
            )
        if pipeline.aermod_enabled and pipeline.asrv_patterns_file and emissions_outputs.get("beam_emissions_for_aermod"):
            _log_step_banner("STEP 3", "aermod concentrations")
            logger.info("Using Step 3 implementation: aermod_concentrations_and_export")
            _, _, aermod_concentration_path = run_aermod_dispersion(
                pipeline=pipeline,
                raw_dir=output_root,
                emissions_input_path=emissions_outputs["beam_emissions_for_aermod"],
            )
            logger.info("AERMOD concentrations complete: wrote %s", aermod_concentration_path)
        else:
            logger.info(
                "AERMOD concentrations skipped: aermod_enabled=%s asrv_patterns_file=%s beam_emissions_for_aermod=%s",
                pipeline.aermod_enabled,
                pipeline.asrv_patterns_file,
                emissions_outputs.get("beam_emissions_for_aermod"),
            )
        if concentration_path is not None:
            _log_step_banner("STEP 4", "prepare exposure")
            logger.info("Using Step 4 implementation: prepare_exposure")
            _, exposure_grid_path, population_distribution_path, population_counts_path = run_prepare_exposure(
                pipeline=pipeline,
                raw_dir=output_root,
                inmap_concentrations_path=str(concentration_path),
                aermod_concentrations_path=str(aermod_concentration_path) if aermod_concentration_path else None,
                population_inputs=population_inputs,
            )
            logger.info("Concentration distribution complete: wrote %s", exposure_grid_path)
            if population_distribution_path is not None:
                logger.info("Population distribution complete: wrote %s", population_distribution_path)
            if population_counts_path is not None:
                logger.info("Population counts complete: wrote %s", population_counts_path)
        else:
            logger.info("Exposure preparation skipped: beam_inmap_concentrations was not produced")
    else:
        logger.info("Dispersion skipped")

    run_manifest = {
        "contract_version": manifest.get("contract_version", "1"),
        "model": "impacts",
        "input_manifest_path": str(Path(input_manifest_path).resolve()),
        "output_dir": str(output_root),
        "outputs_dir": str(output_root),
        "command": " ".join(sys.argv),
        "image": "not_recorded",
        "outputs": {
            "skims_emissions": prepared_skims_path,
            "grid_intersection": str(grid_intersection_path),
            "aermod_full_grid": pipeline.aermod_full_grid_path,
            **emissions_outputs,
            "beam_inmap_concentrations": str(concentration_path) if concentration_path else None,
            "beam_inmap_concentrations_gpkg": (
                str(concentration_path.with_suffix(".gpkg")) if concentration_path else None
            ),
            "beam_aermod_concentrations": str(aermod_concentration_path) if aermod_concentration_path else None,
            "beam_aermod_concentrations_gpkg": (
                str(aermod_concentration_path.with_suffix(".gpkg")) if aermod_concentration_path else None
            ),
            "beam_concentration_distribution": str(exposure_grid_path) if exposure_grid_path else None,
            "beam_concentration_distribution_gpkg": (
                str(exposure_grid_path.with_suffix(".gpkg")) if exposure_grid_path else None
            ),
            "beam_population_distribution": str(population_distribution_path) if population_distribution_path else None,
            "beam_population_counts": (
                str(population_counts_path) if population_counts_path else None
            ),
            "beam_population_counts_gpkg": (
                str(population_counts_path.with_suffix(".gpkg"))
                if population_counts_path
                else None
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
            "stopped_after": (
                "step4_prepare_exposure"
                if exposure_grid_path is not None or population_distribution_path is not None
                else (
                    "step3_compute_aermod_concentrations"
                    if aermod_concentration_path is not None
                    else (
                        "step2_compute_inmap_concentrations"
                        if concentration_path is not None
                        else "step1_process_emissions"
                    )
                )
            ),
        },
    }
    output_manifest = Path(run_manifest_path) if run_manifest_path else output_root / "run_manifest.yaml"
    run_manifest["run_manifest_path"] = str(output_manifest)
    typed_manifest = RunManifest.from_dict(run_manifest)
    write_structured_file(output_manifest, typed_manifest.to_dict())
    logger.info("Run manifest written: %s", output_manifest)
    return typed_manifest.to_dict()


def run_from_settings(
    settings_path: str | Path,
    run_manifest_path: str | Path | None = None,
    run_dispersion: bool = False,
) -> Dict[str, Any]:
    from impacts.preprocessor import preprocess_workflow

    preprocess_manifest = preprocess_workflow(
        settings_path=settings_path,
    )
    return run_from_input_manifest(
        input_manifest_path=preprocess_manifest["inputs_manifest_path"],
        output_dir=preprocess_manifest["input_dir"],
        run_manifest_path=run_manifest_path,
        run_dispersion=run_dispersion,
    )
