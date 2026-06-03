from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

from .config.settings_builder import load_settings_from_yaml
from .manifest.file_ops import load_structured_file
from .manifest.file_ops import resolve_path
from .manifest.file_ops import write_structured_file
from .manifest.schema import PreprocessManifest
from .manifest.schema import PipelineConfig
from .manifest.schema import PipelineManifest
from .common import resolve_required_manifest_input

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


def _record_stage_timing(stage_timings: dict[str, float], key: str, started_at: float) -> None:
    stage_timings[key] = round(time.perf_counter() - started_at, 2)


def _log_stage_timing_summary(stage_timings: dict[str, float]) -> None:
    if not stage_timings:
        return
    summary = ", ".join(f"{stage}={seconds:.2f}s" for stage, seconds in stage_timings.items())
    logger.info("Stage timing summary: %s", summary)


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


def _resolve_existing_run_manifest_path(
    *,
    output_root: Path,
    run_manifest_path: str | Path | None,
) -> Path | None:
    if run_manifest_path:
        candidate = Path(run_manifest_path).resolve()
        return candidate if candidate.exists() else None
    candidate = output_root / "pipeline_manifest.yaml"
    return candidate if candidate.exists() else None


def _load_existing_run_manifest(
    *,
    output_root: Path,
    run_manifest_path: str | Path | None,
) -> Dict[str, Any]:
    candidate = _resolve_existing_run_manifest_path(output_root=output_root, run_manifest_path=run_manifest_path)
    if candidate is None:
        return {}
    return PipelineManifest.from_dict(load_structured_file(candidate)).to_dict()


def _load_run_manifest_context(
    run_manifest_path: str | Path,
) -> tuple[dict[str, Any], Path, str]:
    manifest = PipelineManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    preprocess_manifest_path = manifest.get("preprocess_manifest_path")
    if not preprocess_manifest_path:
        raise ValueError("Pipeline manifest is missing preprocess_manifest_path.")
    output_dir = manifest.get("output_dir")
    if not output_dir:
        raise ValueError("Pipeline manifest is missing output_dir.")
    return manifest, Path(str(output_dir)).resolve(), str(preprocess_manifest_path)


def _resolve_staged_intersection_paths(
    *,
    manifest_inputs: Dict[str, Any],
    pipeline: PipelineConfig,
) -> Dict[str, Optional[str]]:
    paths = {
        "county": resolve_required_manifest_input(manifest_inputs, key="county_intersection"),
        "inmap": (
            resolve_required_manifest_input(manifest_inputs, key="inmap_intersection")
            if pipeline.inmap_enabled
            else None
        ),
        "aermod": (
            resolve_required_manifest_input(manifest_inputs, key="aermod_intersection")
            if pipeline.aermod_enabled
            else None
        ),
    }
    for zone_label, path in paths.items():
        if not path:
            continue
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Run requires staged {zone_label} intersection output from preprocess, but it was not found: {path}"
            )
    return paths


def _run_stages_from_preprocess_manifest(
    preprocess_manifest_path: str | Path,
    run_manifest_path: str | Path | None = None,
    run_dispersion: bool = False,
    run_emissions: bool | None = None,
    run_inmap: bool | None = None,
    run_aermod: bool | None = None,
    run_exposure: bool | None = None,
) -> Dict[str, Any]:
    manifest = PreprocessManifest.from_dict(load_structured_file(preprocess_manifest_path)).to_dict()
    pipeline = PipelineConfig.from_dict(manifest.get("pipeline", {}) or {})
    population_inputs = manifest.get("population_inputs", {}) or {}
    manifest_inputs = manifest.get("inputs", {}) or {}
    input_root = Path(manifest.get("input_dir", "")).resolve()

    output_root = _resolve_runtime_output_root(
        input_manifest=manifest,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "emissions").mkdir(parents=True, exist_ok=True)
    (output_root / "concentrations").mkdir(parents=True, exist_ok=True)
    (output_root / "exposure").mkdir(parents=True, exist_ok=True)
    logger.info("Loaded preprocess manifest: %s", Path(preprocess_manifest_path).resolve())
    logger.info("Output directory: %s", output_root)

    from .common import prepared_table_target
    stage_timings: dict[str, float] = {}
    grid_intersection_paths = _resolve_staged_intersection_paths(
        manifest_inputs=manifest_inputs,
        pipeline=pipeline,
    )
    existing_run_manifest = _load_existing_run_manifest(
        output_root=output_root,
        run_manifest_path=run_manifest_path,
    )
    existing_outputs = dict(existing_run_manifest.get("outputs", {}) or {})

    execute_emissions = pipeline.emissions_enabled if run_emissions is None else bool(run_emissions)
    execute_inmap = pipeline.inmap_enabled if run_inmap is None else bool(run_inmap)
    execute_aermod = pipeline.aermod_enabled if run_aermod is None else bool(run_aermod)
    execute_exposure = pipeline.exposure_enabled if run_exposure is None else bool(run_exposure)
    if run_dispersion:
        execute_inmap = execute_inmap and pipeline.inmap_enabled
        execute_aermod = execute_aermod and pipeline.aermod_enabled
        execute_exposure = execute_exposure and pipeline.exposure_enabled
    elif run_inmap is None and run_aermod is None and run_exposure is None:
        execute_inmap = False
        execute_aermod = False
        execute_exposure = False

    emissions_outputs = {
        key: existing_outputs.get(key)
        for key in (
            "beam_emissions_by_county_process",
            "beam_emissions_for_inmap",
            "beam_inmap_study_area_grid",
            "beam_emissions_for_aermod",
        )
    }
    if execute_emissions:
        from .pipeline.workflow.step1_process_emissions import run as run_emissions_processing

        _log_step_banner("STEP 1", "emissions processing")
        logger.info("Using Step 1 implementation: emissions_processing")
        stage_started = time.perf_counter()
        emissions_outputs = run_emissions_processing(
            pipeline,
            output_root / "emissions",
            output_root / "emissions",
            grid_intersection_paths,
            manifest_inputs=manifest_inputs,
        )
        _record_stage_timing(stage_timings, "step1_process_emissions", stage_started)
    else:
        logger.info("Emissions processing skipped")
    prepared_skims_candidate = prepared_table_target(output_root / "emissions", "prepared_skims_for_grid_allocation")
    prepared_skims_path = str(prepared_skims_candidate) if prepared_skims_candidate.exists() else None

    concentration_path = (
        Path(existing_outputs["beam_inmap_concentrations"]).resolve()
        if existing_outputs.get("beam_inmap_concentrations")
        else None
    )
    aermod_concentration_path = (
        Path(existing_outputs["beam_aermod_concentrations"]).resolve()
        if existing_outputs.get("beam_aermod_concentrations")
        else None
    )
    exposure_grid_path = (
        Path(existing_outputs["beam_concentration_distribution"]).resolve()
        if existing_outputs.get("beam_concentration_distribution")
        else None
    )
    population_distribution_path = (
        Path(existing_outputs["beam_population_distribution"]).resolve()
        if existing_outputs.get("beam_population_distribution")
        else None
    )
    population_counts_path = (
        Path(existing_outputs["beam_population_counts"]).resolve()
        if existing_outputs.get("beam_population_counts")
        else None
    )
    if execute_inmap or execute_aermod or execute_exposure:
        from .pipeline.workflow.step2_compute_inmap_concentrations import run as run_inmap_dispersion
        from .pipeline.workflow.step3_compute_aermod_concentrations import run as run_aermod_dispersion
        from .pipeline.workflow.step4_prepare_exposure import run as run_prepare_exposure
        if execute_inmap and emissions_outputs.get("beam_emissions_for_inmap"):
            _log_step_banner("STEP 2", "inmap concentrations")
            logger.info("Using Step 2 implementation: inmap_concentrations_and_export")
            stage_started = time.perf_counter()
            _, _, concentration_path = run_inmap_dispersion(
                pipeline=pipeline,
                raw_dir=output_root / "concentrations",
                emissions_input_path=emissions_outputs["beam_emissions_for_inmap"],
                inmap_study_area_grid_path=emissions_outputs.get("beam_inmap_study_area_grid"),
            )
            _record_stage_timing(stage_timings, "step2_compute_inmap_concentrations", stage_started)
            logger.info("InMAP concentrations complete: wrote %s", concentration_path)
        else:
            logger.info(
                "InMAP concentrations skipped: inmap_enabled=%s beam_emissions_for_inmap=%s",
                execute_inmap,
                emissions_outputs.get("beam_emissions_for_inmap"),
            )
        if execute_aermod and pipeline.asrv_patterns_file and emissions_outputs.get("beam_emissions_for_aermod"):
            _log_step_banner("STEP 3", "aermod concentrations")
            logger.info("Using Step 3 implementation: aermod_concentrations_and_export")
            stage_started = time.perf_counter()
            _, _, aermod_concentration_path = run_aermod_dispersion(
                pipeline=pipeline,
                raw_dir=output_root / "concentrations",
                cache_dir=output_root / "_tmp",
                emissions_input_path=emissions_outputs["beam_emissions_for_aermod"],
            )
            _record_stage_timing(stage_timings, "step3_compute_aermod_concentrations", stage_started)
            logger.info("AERMOD concentrations complete: wrote %s", aermod_concentration_path)
        else:
            logger.info(
                "AERMOD concentrations skipped: aermod_enabled=%s asrv_patterns_file=%s beam_emissions_for_aermod=%s",
                execute_aermod,
                pipeline.asrv_patterns_file,
                emissions_outputs.get("beam_emissions_for_aermod"),
            )
        if execute_exposure and concentration_path is not None:
            _log_step_banner("STEP 4", "prepare exposure")
            logger.info("Using Step 4 implementation: prepare_exposure")
            stage_started = time.perf_counter()
            _, exposure_grid_path, population_distribution_path, population_counts_path = run_prepare_exposure(
                pipeline=pipeline,
                raw_dir=output_root / "exposure",
                inmap_concentrations_path=str(concentration_path),
                aermod_concentrations_path=str(aermod_concentration_path) if aermod_concentration_path else None,
                manifest_inputs=manifest_inputs,
            )
            _record_stage_timing(stage_timings, "step4_prepare_exposure", stage_started)
            logger.info("Concentration distribution complete: wrote %s", exposure_grid_path)
            if population_distribution_path is not None:
                logger.info("Population distribution complete: wrote %s", population_distribution_path)
            if population_counts_path is not None:
                logger.info("Population counts complete: wrote %s", population_counts_path)
        else:
            logger.info(
                "Exposure preparation skipped: exposure_enabled=%s beam_inmap_concentrations=%s",
                execute_exposure,
                concentration_path,
            )
    else:
        logger.info("Concentration and exposure stages skipped")

    _log_stage_timing_summary(stage_timings)
    run_manifest = {
        "contract_version": manifest.get("contract_version", "1"),
        "model": "impacts",
        "preprocess_manifest_path": str(Path(preprocess_manifest_path).resolve()),
        "output_dir": str(output_root),
        "command": " ".join(sys.argv),
        "image": "not_recorded",
        "outputs": {
            **existing_outputs,
            "skims_emissions": prepared_skims_path,
            "county_intersection": grid_intersection_paths.get("county"),
            "inmap_intersection": grid_intersection_paths.get("inmap"),
            "aermod_intersection": grid_intersection_paths.get("aermod"),
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
            "dispersion_completed": bool(execute_inmap or execute_aermod),
            "stage_timings_seconds": stage_timings,
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
    shutil.rmtree(output_root / "_tmp", ignore_errors=True)
    output_manifest = Path(run_manifest_path) if run_manifest_path else output_root / "pipeline_manifest.yaml"
    run_manifest["pipeline_manifest_path"] = str(output_manifest)
    typed_manifest = PipelineManifest.from_dict(run_manifest)
    write_structured_file(output_manifest, typed_manifest.to_dict())
    logger.info("Pipeline manifest written: %s", output_manifest)
    logger.info(
        "Workflow stage complete: stopped_after=%s pipeline_manifest=%s",
        run_manifest["execution"]["stopped_after"],
        output_manifest,
    )
    return typed_manifest.to_dict()


def run_emissions_from_pipeline_manifest(
    *,
    run_manifest_path: str | Path,
) -> Dict[str, Any]:
    _, _, preprocess_manifest_path = _load_run_manifest_context(run_manifest_path)
    return _run_stages_from_preprocess_manifest(
        preprocess_manifest_path=preprocess_manifest_path,
        run_manifest_path=run_manifest_path,
        run_emissions=True,
        run_inmap=False,
        run_aermod=False,
        run_exposure=False,
    )


def run_inmap_from_pipeline_manifest(
    *,
    run_manifest_path: str | Path,
) -> Dict[str, Any]:
    _, _, preprocess_manifest_path = _load_run_manifest_context(run_manifest_path)
    return _run_stages_from_preprocess_manifest(
        preprocess_manifest_path=preprocess_manifest_path,
        run_manifest_path=run_manifest_path,
        run_emissions=False,
        run_inmap=True,
        run_aermod=False,
        run_exposure=False,
    )


def run_aermod_from_pipeline_manifest(
    *,
    run_manifest_path: str | Path,
) -> Dict[str, Any]:
    _, _, preprocess_manifest_path = _load_run_manifest_context(run_manifest_path)
    return _run_stages_from_preprocess_manifest(
        preprocess_manifest_path=preprocess_manifest_path,
        run_manifest_path=run_manifest_path,
        run_emissions=False,
        run_inmap=False,
        run_aermod=True,
        run_exposure=False,
    )


def run_exposure_from_pipeline_manifest(
    *,
    run_manifest_path: str | Path,
) -> Dict[str, Any]:
    _, _, preprocess_manifest_path = _load_run_manifest_context(run_manifest_path)
    return _run_stages_from_preprocess_manifest(
        preprocess_manifest_path=preprocess_manifest_path,
        run_manifest_path=run_manifest_path,
        run_emissions=False,
        run_inmap=False,
        run_aermod=False,
        run_exposure=True,
    )
